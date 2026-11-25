import path from "node:path";
import { describe, expect, it } from "vitest";
import {
	AppDatabase,
	BudgetStore,
	bootstrapHarness,
	createFastModelBudget,
	DEFAULT_PERMISSIONS,
	FastModelBudgetExceededError,
	MemoryConsolidator,
	SqliteMemoryService,
	type FastModelClient,
	type TenantContext,
} from "../src/index.js";
import { clientSafeDetail, clientSafeReason } from "../src/runtime/client-safe.js";

const TENANT: TenantContext = {
	tenantId: "tenant-gaps",
	userId: "user-gaps",
	permissions: [...DEFAULT_PERMISSIONS],
};
const OTHER: TenantContext = { tenantId: "tenant-other", userId: "user-other", permissions: [...DEFAULT_PERMISSIONS] };

const BOOT = {
	piDbPath: ":memory:",
	appDbPath: ":memory:",
	useMockModels: true,
	skillsDir: path.resolve("skills"),
} as const;

describe("client-safe failure text (SSE run_status / submissions)", () => {
	it("only exposes stable reason codes and budget-veto details", () => {
		expect(clientSafeReason("model_error")).toBe("model_error");
		expect(clientSafeReason("ENOENT: open '/etc/secret' failed")).toBe("unanswered");
		expect(clientSafeDetail("BUDGET_EXCEEDED conversation_tokens: out of budget")).toContain("BUDGET_EXCEEDED");
		expect(clientSafeDetail("SQLITE_ERROR: no such table pi_tasks at C:\\data\\db.sqlite")).toBeUndefined();
		expect(clientSafeDetail(undefined)).toBeUndefined();
		expect(clientSafeDetail("BUDGET_EXCEEDED " + "x".repeat(2000))!.length).toBeLessThanOrEqual(500);
	});
});

describe("consolidator distrusts model-claimed evidence counts", () => {
	it("clamps claimed evidence to what the batch contains so one episode cannot pass the repetition gate", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const memory = new SqliteMemoryService(appDb);
			await memory.recordEpisode({
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				conversationId: "conv-1",
				runId: "run-1",
				userMessage: "I asked about the weather.",
				assistantSummary: "Told them it was sunny.",
				toolTrace: [],
				frictionSignals: [],
				frictionScore: 0,
			});
			const plan = {
				operations: [
					{
						op: "ADD",
						key: "user.weather_interest",
						kind: "fact",
						content: "Cares about the weather",
						confidence: 0.9,
						evidenceCount: 9,
						distinctConversations: 5,
					},
				],
			};
			const fastModel: FastModelClient = {
				async completeJson<T>() {
					return { rawText: JSON.stringify(plan), parsed: plan as T };
				},
			};
			const report = await new MemoryConsolidator({ memory, fastModel }).consolidate(TENANT.tenantId, TENANT.userId);
			expect(report.addedIds).toHaveLength(0);
			expect(report.rejectedByGate).toHaveLength(1);
			expect(report.rejectedByGate[0]!.reason).toContain("evidence=1");
		} finally {
			appDb.close();
		}
	});
});

describe("skill re-enable", () => {
	it("starts a fresh evaluation window so a stale skill is not disabled again next cycle", async () => {
		const runtime = await bootstrapHarness({ config: BOOT, sleepCycleScheduler: false });
		try {
			const day = 86_400_000;
			const longAgo = Date.now() - 90 * day;
			await runtime.capabilities.telemetry.recordToolInvocation({
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				conversationId: "1",
				callId: "c1",
				toolName: "skill_read",
				skillName: "incident-triage",
				isError: false,
				createdAt: longAgo,
			});

			const first = await runtime.skillDojo.evaluateHealth(TENANT.tenantId, Date.now());
			expect(first.newlyDisabled.map((s) => s.name)).toContain("incident-triage");

			await runtime.skillDojo.reenableSkill(TENANT.tenantId, "incident-triage");
			const second = await runtime.skillDojo.evaluateHealth(TENANT.tenantId, Date.now());
			expect(second.newlyDisabled.map((s) => s.name)).not.toContain("incident-triage");
			expect(second.allDisabled).not.toContain("incident-triage");
		} finally {
			await runtime.close();
		}
	});
});

describe("fast-model daily budget", () => {
	it("blocks once the daily ledger reaches the limit and rolls over by UTC day", () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			let nowMs = Date.UTC(2026, 0, 1, 12);
			const store = new BudgetStore(appDb.db, { now: () => nowMs });
			const budget = createFastModelBudget({ store, maxDailyTokens: 100, maxDailyCostUsd: 0, now: () => nowMs });

			expect(() => budget.guard()).not.toThrow();
			budget.onUsage({ totalTokens: 60 });
			expect(() => budget.guard()).not.toThrow();
			budget.onUsage({ totalTokens: 60 });
			expect(() => budget.guard()).toThrow(FastModelBudgetExceededError);

			nowMs += 24 * 3_600_000;
			expect(() => budget.guard()).not.toThrow();
		} finally {
			appDb.close();
		}
	});

	it("is wired into bootstrapHarness: an exhausted ledger rejects fast-model calls without calling the provider", async () => {
		const runtime = await bootstrapHarness({
			config: { ...BOOT, maxFastModelDailyTokens: 50 },
			sleepCycleScheduler: false,
		});
		try {
			runtime.budgetStore.recordUsage("__fast_model__", 500, 0);
			runtime.testModel.enqueueFastJson({ ok: true });
			await expect(runtime.capabilities.fastModel!.completeJson("sys", "user")).rejects.toThrow(/BUDGET_EXCEEDED/);
		} finally {
			await runtime.close();
		}
	});
});

describe("durable sleep-cycle single-flight", () => {
	it("returns the live task for the conversation, and starts a new one once it has settled", async () => {
		const runtime = await bootstrapHarness({ config: BOOT, sleepCycleScheduler: false });
		try {
			const conv = await runtime.harness.createConversation({ tenant: TENANT });
			runtime.testModel.enqueueFastJson({ operations: [{ op: "NOOP", reason: "none" }] });

			const [a, b] = await Promise.all([
				runtime.harness.triggerSleepCycle(conv.conversationId, { reason: "t1" }),
				runtime.harness.triggerSleepCycle(conv.conversationId, { reason: "t2" }),
			]);
			expect(b.taskId).toBe(a.taskId);

			await runtime.harness.waitTask(a.taskId);
			runtime.testModel.enqueueFastJson({ operations: [{ op: "NOOP", reason: "none" }] });
			const c = await runtime.harness.triggerSleepCycle(conv.conversationId, { reason: "t3" });
			expect(c.taskId).not.toBe(a.taskId);
			await runtime.harness.waitTask(c.taskId);
		} finally {
			await runtime.close();
		}
	});
});

describe("tenant-scoped approvals listing (incremental index)", () => {
	it("picks up conversations created after the index was built, and never mixes tenants", async () => {
		const runtime = await bootstrapHarness({ config: BOOT, sleepCycleScheduler: false });
		try {
			await runtime.harness.createConversation({ tenant: TENANT });
			await runtime.harness.createConversation({ tenant: OTHER });
			expect(await runtime.harness.listPendingApprovals(undefined, { tenant: TENANT })).toEqual([]);

			// Created after the first (index-building) call: the incremental sync must still see it.
			await runtime.harness.createConversation({ tenant: TENANT });
			expect(await runtime.harness.listPendingApprovals(undefined, { tenant: TENANT })).toEqual([]);
			expect(await runtime.harness.listPendingApprovals(undefined, { tenant: OTHER })).toEqual([]);
		} finally {
			await runtime.close();
		}
	});
});

describe("frozen core memory refresh on reset", () => {
	it("keeps the prompt frozen mid-conversation but picks up new core memory after a reset", async () => {
		const runtime = await bootstrapHarness({ config: BOOT, sleepCycleScheduler: false });
		try {
			await runtime.capabilities.memory.pinMemory(TENANT.tenantId, TENANT.userId, {
				key: "user.editor",
				content: "Prefers Neovim with Lua config",
				kind: "preference",
			});
			const conv = await runtime.harness.createConversation({ tenant: TENANT });

			runtime.testModel.enqueueMain({ type: "text", text: "Hi" });
			await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "hello" });

			await runtime.capabilities.memory.pinMemory(TENANT.tenantId, TENANT.userId, {
				key: "user.shell",
				content: "Uses zsh everywhere",
				kind: "preference",
			});

			runtime.testModel.enqueueMain({ type: "text", text: "Still frozen" });
			await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "again" });
			const midSession = runtime.testModel.getRecordedRequests({ modelId: "main" }).at(-1)!;
			expect(midSession.leadingSystemPrompt).not.toContain("zsh everywhere");

			await runtime.harness.resetConversation(conv.conversationId);
			runtime.testModel.clearRecordedRequests();
			runtime.testModel.enqueueMain({ type: "text", text: "Refreshed" });
			await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "after reset" });
			const afterReset = runtime.testModel.getRecordedRequests({ modelId: "main" })[0]!;
			expect(afterReset.leadingSystemPrompt).toContain("zsh everywhere");
			expect(afterReset.leadingSystemPrompt).toContain("Neovim");
		} finally {
			await runtime.close();
		}
	});
});
