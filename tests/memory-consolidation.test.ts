import { describe, expect, it } from "vitest";
import {
	AppDatabase,
	bootstrapHarness,
	DEFAULT_PERMISSIONS,
	MemoryConsolidator,
	SqliteMemoryService,
	type FastModelClient,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-mem",
	userId: "user-mem",
	permissions: [...DEFAULT_PERMISSIONS],
};

function stubFastModel(plan: unknown): FastModelClient {
	return {
		async completeJson<T>() {
			const rawText = JSON.stringify(plan);
			return { rawText, parsed: plan as T };
		},
	};
}

describe("Memory consolidation, promotion gates, and SUPERSEDE tombstones", () => {
	it("enforces confidence & evidence promotion gates and creates SUPERSEDE tombstones for contradictions", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const memory = new SqliteMemoryService(appDb);

			// Seed initial active preference
			// Seeded unpinned: pinned memories are protected from model-originated mutations (see below).
			const oldPref = (await memory.applyMemoryMutation(TENANT.tenantId, TENANT.userId, {
				op: "ADD",
				key: "user.package_manager",
				content: "Prefer npm for JavaScript packages",
				kind: "preference",
				confidence: 0.9,
			}))!;

			// Record an unconsolidated episode expressing a contradictory preference + noisy one-off facts
			await memory.recordEpisode({
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				conversationId: "conv-1",
				runId: "run-1",
				userMessage: "Actually, stop using npm — from now on always use pnpm!",
				assistantSummary: "Switched to pnpm.",
				toolTrace: [],
				frictionSignals: [],
				frictionScore: 3,
			});

			const consolidator = new MemoryConsolidator({
				memory,
				fastModel: stubFastModel({
					operations: [
						// Rejected: below confidence gate (0.7)
						{
							op: "ADD",
							key: "user.guess",
							kind: "fact",
							content: "Might like Go",
							confidence: 0.45,
							evidenceCount: 3,
						},
						// Rejected: single-occurrence non-explicit ADD fails repetition gate
						{
							op: "ADD",
							key: "user.transient_note",
							kind: "fact",
							content: "Mentioned rain once",
							confidence: 0.85,
							explicitUserPreference: false,
							evidenceCount: 1,
							distinctConversations: 1,
						},
						// Accepted: repeated evidence >= 2 passes promotion gate
						{
							op: "ADD",
							key: "user.repo.test_runner",
							kind: "procedure",
							content: "Runs tests with vitest run",
							confidence: 0.88,
							evidenceCount: 2,
							distinctConversations: 2,
						},
						// Accepted: contradictory preference creates SUPERSEDE tombstone
						{
							op: "SUPERSEDE",
							targetId: oldPref.id,
							key: "user.package_manager",
							kind: "preference",
							content: "Always use pnpm instead of npm",
							confidence: 0.96,
							reason: "User explicitly switched package manager from npm to pnpm",
						},
					],
				}),
			});

			const report = await consolidator.consolidate(TENANT.tenantId, TENANT.userId);
			expect(report.processedEpisodeIds).toHaveLength(1);
			expect(report.rejectedByGate).toHaveLength(2);
			expect(report.addedIds).toHaveLength(1);
			expect(report.supersededOldIds).toEqual([oldPref.id]);
			expect(report.supersededNewIds).toHaveLength(1);

			const newPrefId = report.supersededNewIds[0]!;

			// Verify old memory is tombstoned (NOT deleted) and links to new memory
			const tombstoned = await memory.getMemoryById(TENANT.tenantId, TENANT.userId, oldPref.id);
			expect(tombstoned).toBeDefined();
			expect(tombstoned?.status).toBe("superseded");
			expect(tombstoned?.supersededById).toBe(newPrefId);

			// Verify new preference is active
			const activePref = await memory.getMemoryById(TENANT.tenantId, TENANT.userId, newPrefId);
			expect(activePref?.status).toBe("active");
			expect(activePref?.content).toContain("pnpm");

			// Default search returns ONLY active memories (not tombstoned)
			const searchActive = await memory.search(TENANT.tenantId, TENANT.userId, "package_manager");
			expect(searchActive).toHaveLength(1);
			expect(searchActive[0]?.memory.id).toBe(newPrefId);

			// Audit search with includeSuperseded sees both old tombstone and new active row
			const searchAll = await memory.search(TENANT.tenantId, TENANT.userId, "package_manager", {
				includeSuperseded: true,
			});
			expect(searchAll.map((r) => r.memory.status).sort()).toEqual(["active", "superseded"]);

			// Episodes are marked consolidated so a second run is a no-op
			const secondRun = await consolidator.consolidate(TENANT.tenantId, TENANT.userId);
			expect(secondRun.processedEpisodeIds).toHaveLength(0);
		} finally {
			appDb.close();
		}
	});

	it("auto-upgrades conflicting ADD on the same key to SUPERSEDE and runs end-to-end inside Loop 2 SleepCycleTask", async () => {
		const runtime = await bootstrapHarness({
			config: {
				piDbPath: ":memory:",
				appDbPath: ":memory:",
				useMockModels: true,
			},
		});

		try {
			const original = (await runtime.capabilities.memory.applyMemoryMutation(TENANT.tenantId, TENANT.userId, {
				op: "ADD",
				key: "user.theme",
				content: "Light mode",
				kind: "preference",
				confidence: 0.9,
			}))!;

			const conv = await runtime.harness.createConversation({ tenant: TENANT });
			runtime.testModel.enqueueMain({ type: "text", text: "Got it, switching to dark mode." });

			const settled = await runtime.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "No, actually I prefer dark mode now.",
			});
			expect(settled.status).toBe("done");

			// Wait for Loop 1 afterRun CaptureEpisodeTask to finish
			await runtime.harness.waitForIdle(conv.conversationId);

			// Fast model proposes a conflicting ADD on the same key -> consolidator upgrades it to SUPERSEDE!
			runtime.testModel.enqueueFastJson({
				operations: [
					{
						op: "ADD",
						key: "user.theme",
						kind: "preference",
						content: "Dark mode",
						confidence: 0.92,
						explicitUserPreference: true,
					},
				],
			});

			const { taskId } = await runtime.harness.triggerSleepCycle(conv.conversationId, { reason: "test-sleep" });
			const sleepResult = await runtime.harness.waitTask(taskId);
			expect(sleepResult.status).toBe("completed");

			const payload = sleepResult.result as {
				consolidation: { processedEpisodes: number; supersededCount: number };
			};
			expect(payload.consolidation.processedEpisodes).toBeGreaterThanOrEqual(1);
			expect(payload.consolidation.supersededCount).toBe(1);

			const oldRow = await runtime.capabilities.memory.getMemoryById(TENANT.tenantId, TENANT.userId, original.id);
			expect(oldRow?.status).toBe("superseded");
			expect(oldRow?.supersededById).toBeDefined();

			const activeMemories = await runtime.capabilities.memory.listMemories(TENANT.tenantId, TENANT.userId);
			expect(activeMemories).toHaveLength(1);
			expect(activeMemories[0]?.content).toBe("Dark mode");
		} finally {
			await runtime.close();
		}
	});
});
