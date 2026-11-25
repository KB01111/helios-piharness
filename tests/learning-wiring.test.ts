import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { defineExtension } from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { loadSkillCatalog } from "../src/extensions/skills-loader.ext.js";
import { HarnessRegistryManager } from "../src/harness/registry.js";
import { SkillDojo } from "../src/learning/skill-dojo.js";
import {
	listProposedSkills,
	promoteProposedSkill,
	proposeSkillDrafts,
	rejectProposedSkill,
	sanitizeSkillName,
} from "../src/learning/skill-promotion.js";
import {
	AppDatabase,
	bootstrapHarness,
	DEFAULT_PERMISSIONS,
	loadConfig,
	SleepCycleScheduler,
	type PendingEpisodeScope,
	type SleepSchedulerTimer,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = { tenantId: "tenant-wire", userId: "user-wire", permissions: [...DEFAULT_PERMISSIONS] };
const OTHER: TenantContext = { tenantId: "tenant-other", userId: "user-other", permissions: [...DEFAULT_PERMISSIONS] };

class ManualTimer implements SleepSchedulerTimer {
	readonly pending = new Map<number, { callback: () => void; delayMs: number }>();
	private nextId = 1;
	setTimeout(callback: () => void, delayMs: number): unknown {
		const id = this.nextId++;
		this.pending.set(id, { callback, delayMs });
		return id;
	}
	clearTimeout(handle: unknown): void {
		this.pending.delete(handle as number);
	}
	/** Fires the single armed timer (if any). */
	fire(): boolean {
		const first = this.pending.entries().next();
		if (first.done) return false;
		const [id, entry] = first.value;
		this.pending.delete(id);
		entry.callback();
		return true;
	}
}

function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((r) => {
		resolve = r;
	});
	return { promise, resolve };
}

function flush(): Promise<void> {
	return new Promise((resolve) => setImmediate(resolve));
}

const tempDirs: string[] = [];
function tempDir(prefix: string): string {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
	tempDirs.push(dir);
	return dir;
}

afterEach(() => {
	for (const dir of tempDirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
});

describe("HarnessRegistryManager.replaceExtension", () => {
	const ext = (name: string, marker: string) => defineExtension({ name, namespace: { name: `ns-${marker}`, description: marker } });

	it("replaces a core extension in place: one copy, new object, original order preserved", () => {
		const manager = new HarnessRegistryManager();
		const a = ext("ext-a", "a1");
		const b = ext("ext-b", "b1");
		const c = ext("ext-c", "c1");
		for (const e of [a, b, c]) manager.installCoreExtension(e);

		const b2 = ext("ext-b", "b2");
		manager.replaceExtension(b2);

		const installed = manager.getRegistry().snapshot().installed();
		expect(installed.map((e) => e.name)).toEqual(["ext-a", "ext-b", "ext-c"]);
		expect(installed[1]).toBe(b2);
		expect(manager.getRegistry().snapshot().extension("ext-b")).toBe(b2);
		// conversation selections also see the replacement exactly once
		const selected = manager.resolveConversationExtensionSelection([]) ?? [];
		expect(selected.filter((e) => e.name === "ext-b")).toEqual([b2]);
	});

	it("tracks unknown names as core extensions and keeps disabled toolpacks uninstalled", () => {
		const manager = new HarnessRegistryManager();
		manager.replaceExtension(ext("fresh", "f1"));
		expect(manager.getRegistry().snapshot().extension("fresh")).toBeDefined();
		expect(manager.uninstallCoreExtension("fresh")).toBe(true);

		manager.registerToolPack({ name: "pack", description: "p", tools: [] }, { enabled: false });
		manager.replaceExtension(defineExtension({ name: "toolpack:pack", namespace: { name: "pack", description: "new" } }));
		expect(manager.getRegistry().snapshot().extension("toolpack:pack")).toBeUndefined();
		manager.setToolPackEnabled("pack", true);
		expect(manager.getRegistry().snapshot().extension("toolpack:pack")?.namespace?.description).toBe("new");
	});
});

describe("SkillPromotion", () => {
	const draft = (name: string) => ({
		id: `wf_${name}`,
		suggestedSkillName: name,
		draftSkillMarkdown: `---\nname: ${name}\ndescription: Mined workflow.\ntags: [mined]\n---\n\n# ${name}\n`,
	});
	const provenance = { tenantId: "t1", source: "test", proposedAt: Date.UTC(2026, 10, 24), taskId: "7" };

	it("sanitizes names so no path traversal is possible", () => {
		expect(sanitizeSkillName("../../etc/passwd")).toBe("etc-passwd");
		expect(sanitizeSkillName("..")).toBeUndefined();
		expect(sanitizeSkillName("Workflow A/B C")).toBe("workflow-a-b-c");
		expect(sanitizeSkillName("x".repeat(100))?.length).toBe(64);
	});

	it("writes drafts to _proposed only, idempotently, never into the active catalog", () => {
		const dir = tempDir("helios-promote-");
		const first = proposeSkillDrafts(dir, [draft("workflow-a-then-b"), draft("../../escape")], { provenance });
		expect(first.created.map((c) => c.name)).toEqual(["workflow-a-then-b", "escape"]);

		const file = path.join(dir, "_proposed", "workflow-a-then-b", "SKILL.md");
		const text = fs.readFileSync(file, "utf8");
		expect(text.startsWith("---\n")).toBe(true);
		expect(text).toContain("status: proposed");
		expect(text).toContain("proposed-tenant: t1");
		expect(text).toContain("proposed-by-task: 7");
		expect(fs.existsSync(path.join(dir, "..", "escape"))).toBe(false);

		// not visible to the active catalog
		expect(loadSkillCatalog(dir)).toEqual([]);
		expect(listProposedSkills(dir)).toEqual(["escape", "workflow-a-then-b"]);

		// idempotent: no rewrite, no duplicate
		fs.appendFileSync(file, "\nHAND EDIT\n");
		const second = proposeSkillDrafts(dir, [draft("workflow-a-then-b")], { provenance });
		expect(second.created).toEqual([]);
		expect(second.skipped[0]?.reason).toBe("already-proposed");
		expect(fs.readFileSync(file, "utf8")).toContain("HAND EDIT");
	});

	it("respects maxNew, excluded names, and existing active skills", () => {
		const dir = tempDir("helios-promote-");
		fs.mkdirSync(path.join(dir, "taken"), { recursive: true });
		fs.writeFileSync(path.join(dir, "taken", "SKILL.md"), "---\nname: taken\ndescription: x\n---\n");
		const res = proposeSkillDrafts(dir, [draft("taken"), draft("off"), draft("one"), draft("two")], {
			provenance,
			maxNew: 1,
			excludeNames: ["off"],
		});
		expect(res.created.map((c) => c.name)).toEqual(["one"]);
		expect(res.skipped.map((s) => s.reason).sort()).toEqual(["already-active", "cycle-limit", "excluded"]);
	});

	it("promoteProposedSkill moves into the active catalog; refuses overwrite/invalid; reject leaves a tombstone", () => {
		const dir = tempDir("helios-promote-");
		proposeSkillDrafts(dir, [draft("wf-one"), draft("wf-two")], { provenance });
		expect(() => promoteProposedSkill(dir, "../wf-one")).toThrow();
		expect(() => promoteProposedSkill(dir, "missing")).toThrow(/No proposed skill/);

		const promoted = promoteProposedSkill(dir, "wf-one");
		expect(fs.existsSync(promoted.filePath)).toBe(true);
		expect(listProposedSkills(dir)).toEqual(["wf-two"]);
		const catalog = loadSkillCatalog(dir);
		expect(catalog.map((s) => s.name)).toEqual(["wf-one"]);
		expect(fs.readFileSync(promoted.filePath, "utf8")).toContain("status: approved");

		// a second proposal with the same name is refused because it is active now
		expect(proposeSkillDrafts(dir, [draft("wf-one")], { provenance }).created).toEqual([]);

		expect(rejectProposedSkill(dir, "wf-two")).toBe(true);
		expect(listProposedSkills(dir)).toEqual([]);
		// tombstone prevents re-proposal
		expect(proposeSkillDrafts(dir, [draft("wf-two")], { provenance }).created).toEqual([]);
	});
});

describe("SkillDojo threshold guards", () => {
	it("ignores nonsensical thresholds instead of disabling everything", async () => {
		const disabled: string[] = [];
		const telemetry = {
			getSkillMetrics: async () => [
				{ tenantId: "t", name: "fresh", kind: "skill", calls: 1, successes: 1, errors: 0, errorRate: 0, avgDurationMs: 1, lastUsedAt: 1_000, disabled: false },
			],
			listDisabledSkills: async () => [] as string[],
			setSkillDisabled: async (_t: string, name: string) => {
				disabled.push(name);
			},
		};
		const dojo = new SkillDojo(telemetry as never, { staleSkillDays: 0, maxErrorRate: Number.NaN, minErrorSamples: -3 });
		const result = await dojo.evaluateHealth("t", 1_000 + 60_000);
		expect(result.newlyDisabled).toEqual([]);
		expect(disabled).toEqual([]);
	});
});

describe("config", () => {
	it("allows SLEEP_CYCLE_CRON_INTERVAL_MS=0 to disable and exposes the scheduler flag", () => {
		expect(loadConfig({ SLEEP_CYCLE_CRON_INTERVAL_MS: "0" }).sleepCycleIntervalMs).toBe(0);
		expect(loadConfig({}).sleepCycleIntervalMs).toBe(300_000);
		expect(loadConfig({}).sleepCycleSchedulerEnabled).toBe(true);
		expect(loadConfig({ SLEEP_CYCLE_SCHEDULER_ENABLED: "false" }).sleepCycleSchedulerEnabled).toBe(false);
	});
});

describe("SleepCycleScheduler (unit)", () => {
	function makeHarness(options?: { failFirst?: number }) {
		const triggered: string[] = [];
		const waits: Array<ReturnType<typeof deferred<{ taskId: string; status: "completed" | "failed"; error?: string }>>> = [];
		let failuresLeft = options?.failFirst ?? 0;
		let counter = 0;
		return {
			triggered,
			waits,
			harness: {
				triggerSleepCycle: async (conversationId: string) => {
					triggered.push(conversationId);
					if (failuresLeft > 0) {
						failuresLeft -= 1;
						throw new Error(`conversation ${conversationId} unavailable`);
					}
					counter += 1;
					return { taskId: String(counter) };
				},
				waitTask: async (taskId: string) => {
					const d = deferred<{ taskId: string; status: "completed" | "failed"; error?: string }>();
					waits.push(d);
					const settled = await d.promise;
					return { ...settled, taskId };
				},
			},
		};
	}

	function makeMemory(scopes: PendingEpisodeScope[]) {
		return { listTenantsWithPendingEpisodes: async () => scopes } as never;
	}

	const scope = (tenantId: string, conversationIds: string[]): PendingEpisodeScope => ({
		tenantId,
		userId: `${tenantId}-u`,
		pendingCount: 2,
		latestEpisodeAt: 1,
		conversationIds,
	});

	it("never piles up cycles for a running tenant and resumes once it settles", async () => {
		const h = makeHarness();
		const scheduler = new SleepCycleScheduler({
			harness: h.harness,
			memory: makeMemory([scope("a", ["1"]), scope("b", ["2"])]),
			intervalMs: 1000,
			timer: new ManualTimer(),
		});

		const t1 = await scheduler.tick();
		expect(t1.triggered.map((t) => t.tenantId)).toEqual(["a", "b"]);
		const t2 = await scheduler.tick();
		expect(t2.triggered).toEqual([]);
		expect(t2.skippedInFlight).toEqual(["a", "b"]);
		expect(h.triggered).toEqual(["1", "2"]);

		h.waits[0]!.resolve({ taskId: "1", status: "completed" });
		await flush();
		expect(scheduler.isRunning("a")).toBe(false);
		expect(scheduler.isRunning("b")).toBe(true);
		const t3 = await scheduler.tick();
		expect(t3.triggered.map((t) => t.tenantId)).toEqual(["a"]);

		h.waits.forEach((w) => w.resolve({ taskId: "x", status: "completed" }));
		await scheduler.idle();
		expect(scheduler.inFlightCount).toBe(0);
	});

	it("falls through to the next conversation, then backs off exponentially on failure", async () => {
		let nowMs = 1_000_000;
		const errors: string[] = [];
		const h = makeHarness({ failFirst: 3 });
		const scheduler = new SleepCycleScheduler({
			harness: h.harness,
			memory: makeMemory([scope("a", ["1", "2"])]),
			intervalMs: 1000,
			now: () => nowMs,
			random: () => 0.5, // zero jitter
			timer: new ManualTimer(),
			onError: (e) => errors.push((e as Error).message),
		});

		// both candidate conversations fail -> failure recorded, 1s backoff
		const t1 = await scheduler.tick();
		expect(t1.failed).toHaveLength(1);
		expect(h.triggered).toEqual(["1", "2"]);

		const t2 = await scheduler.tick();
		expect(t2.skippedBackoff).toEqual(["a"]);

		nowMs += 1_001;
		// first conversation fails again (3rd failure), second succeeds
		const t3 = await scheduler.tick();
		expect(t3.triggered).toHaveLength(1);
		expect(h.triggered.slice(2)).toEqual(["1", "2"]);

		h.waits[0]!.resolve({ taskId: "x", status: "failed", error: "boom" });
		await scheduler.idle();
		expect(errors).toContain("boom");
		// failed task => backoff (now 2 consecutive failures => 2s)
		expect((await scheduler.tick()).skippedBackoff).toEqual(["a"]);
		nowMs += 1_000;
		expect((await scheduler.tick()).skippedBackoff).toEqual(["a"]);
		nowMs += 1_001;
		expect((await scheduler.tick()).triggered).toHaveLength(1);
		h.waits.at(-1)!.resolve({ taskId: "x", status: "completed" });
		await scheduler.idle();
	});

	it("is a no-op when disabled and disposes its timer", async () => {
		const timer = new ManualTimer();
		const h = makeHarness();
		const off = new SleepCycleScheduler({ harness: h.harness, memory: makeMemory([scope("a", ["1"])]), intervalMs: 0, timer });
		off.start();
		expect(timer.pending.size).toBe(0);
		expect((await off.tick()).triggered).toEqual([]);

		const flagged = new SleepCycleScheduler({
			harness: h.harness,
			memory: makeMemory([scope("a", ["1"])]),
			intervalMs: 1000,
			enabled: false,
			timer,
		});
		flagged.start();
		expect(timer.pending.size).toBe(0);

		const on = new SleepCycleScheduler({ harness: h.harness, memory: makeMemory([]), intervalMs: 1000, timer });
		on.start();
		on.start();
		expect(timer.pending.size).toBe(1);
		on.dispose();
		expect(timer.pending.size).toBe(0);
		expect((await on.tick()).triggered).toEqual([]);
		expect(h.triggered).toEqual([]);
	});

	it("re-arms itself after each sweep and tolerates memory implementations without enumeration", async () => {
		const timer = new ManualTimer();
		const h = makeHarness();
		const scheduler = new SleepCycleScheduler({
			harness: h.harness,
			memory: {} as never,
			intervalMs: 1000,
			timer,
			random: () => 0.5,
		});
		scheduler.start();
		expect(timer.pending.size).toBe(1);
		expect([...timer.pending.values()][0]!.delayMs).toBe(1000);
		timer.fire();
		await flush();
		await flush();
		expect(timer.pending.size).toBe(1);
		scheduler.dispose();
	});

	it("durable marker prevents a second scheduler/process from starting a duplicate and recovers after restart", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const h1 = makeHarness();
			const s1 = new SleepCycleScheduler({
				harness: h1.harness,
				memory: makeMemory([scope("a", ["1"])]),
				intervalMs: 1000,
				appDb,
				timer: new ManualTimer(),
			});
			expect((await s1.tick()).triggered).toHaveLength(1);

			// another scheduler instance sharing the DB cannot start a duplicate
			const h2 = makeHarness();
			const s2 = new SleepCycleScheduler({
				harness: h2.harness,
				memory: makeMemory([scope("a", ["1"])]),
				intervalMs: 1000,
				appDb,
				timer: new ManualTimer(),
			});
			const blocked = await s2.tick();
			expect(blocked.triggered).toEqual([]);
			expect(blocked.skippedInFlight).toEqual(["a"]);
			expect(h2.triggered).toEqual([]);

			// "restart": s1 disposed while the durable task still runs; a new scheduler re-attaches
			s1.dispose();
			await flush();
			const h3 = makeHarness();
			const s3 = new SleepCycleScheduler({
				harness: h3.harness,
				memory: makeMemory([scope("a", ["1"])]),
				intervalMs: 1000,
				appDb,
				timer: new ManualTimer(),
			});
			s3.start();
			expect(s3.isRunning("a")).toBe(true);
			expect((await s3.tick()).skippedInFlight).toEqual(["a"]);
			h3.waits[0]!.resolve({ taskId: "1", status: "completed" });
			await s3.idle();
			expect((await s3.tick()).triggered).toHaveLength(1);
			h3.waits.at(-1)!.resolve({ taskId: "1", status: "completed" });
			await s3.idle();
			s3.dispose();
			s2.dispose();
		} finally {
			appDb.close();
		}
	});

	it("an expired durable marker is reclaimed", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			let nowMs = 0;
			const h = makeHarness();
			const mk = () =>
				new SleepCycleScheduler({
					harness: h.harness,
					memory: makeMemory([scope("a", ["1"])]),
					intervalMs: 1000,
					appDb,
					now: () => nowMs,
					markerTtlMs: 5_000,
					timer: new ManualTimer(),
				});
			const s1 = mk();
			expect((await s1.tick()).triggered).toHaveLength(1);
			s1.dispose();
			await flush();

			nowMs = 10_000; // marker older than TTL: previous process presumed dead
			const s2 = mk();
			expect((await s2.tick()).triggered).toHaveLength(1);
			s2.dispose();
		} finally {
			appDb.close();
		}
	});
});

describe("Sleep cycle wiring (integration)", () => {
	const mined = [
		["domain_lookup_record", "domain_mutate_record", "artifact_read"],
		["domain_lookup_record", "domain_mutate_record", "artifact_read"],
	];

	async function seedTelemetry(
		runtime: Awaited<ReturnType<typeof bootstrapHarness>>,
		tenant: TenantContext,
		conversationId: string,
	) {
		for (const [i, toolNames] of mined.entries()) {
			await runtime.capabilities.telemetry.recordRunTelemetry({
				tenantId: tenant.tenantId,
				userId: tenant.userId,
				conversationId,
				runId: `run-${tenant.tenantId}-${i}`,
				toolNames,
				frictionScore: 0,
				frictionKinds: [],
				createdAt: Date.UTC(2026, 10, 24) + i,
			});
		}
	}

	it("persists mined skills as proposals, records SimLog per tenant, and is idempotent across cycles", async () => {
		const skillsDir = tempDir("helios-wiring-skills-");
		const runtime = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, skillsDir },
			sleepCycleScheduler: false,
		});
		try {
			const conv = await runtime.harness.createConversation({ tenant: TENANT });
			const otherConv = await runtime.harness.createConversation({ tenant: OTHER });
			await seedTelemetry(runtime, TENANT, conv.conversationId);
			await seedTelemetry(runtime, OTHER, otherConv.conversationId);
			await runtime.capabilities.memory.recordEpisode({
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				conversationId: conv.conversationId,
				runId: "r1",
				userMessage: "I prefer pnpm over npm",
				assistantSummary: "ok",
				toolTrace: [],
				frictionSignals: [],
				frictionScore: 0,
			});

			runtime.testModel.enqueueFastJson({ operations: [{ op: "NOOP", reason: "none" }] });
			const first = await runtime.harness.waitTask((await runtime.harness.triggerSleepCycle(conv.conversationId)).taskId);
			expect(first.status).toBe("completed");
			const result = first.result as {
				proposedSkills?: { name: string }[];
				warnings?: string[];
				minedWorkflows: { suggestedSkillName: string }[];
			};
			expect(result.warnings).toBeUndefined();
			const name = result.minedWorkflows[0]!.suggestedSkillName;
			expect(result.proposedSkills?.map((p) => p.name)).toEqual([name]);

			const proposedFile = path.join(skillsDir, "_proposed", name, "SKILL.md");
			expect(fs.existsSync(proposedFile)).toBe(true);
			expect(fs.readFileSync(proposedFile, "utf8")).toContain(`proposed-tenant: ${TENANT.tenantId}`);
			expect(loadSkillCatalog(skillsDir)).toEqual([]); // not active until approved

			const mine = runtime.simLog.list({ tenantId: TENANT.tenantId });
			expect(mine.length).toBeGreaterThanOrEqual(2);
			expect(mine.every((e) => e.tenantId === TENANT.tenantId)).toBe(true);
			expect(mine.map((e) => e.targetModule).sort()).toEqual(["memory-consolidation", "skill-forge"]);
			expect(runtime.simLog.list({ tenantId: "someone-else" })).toEqual([]);

			// second cycle: nothing new is proposed, no duplicate propose entries
			const second = await runtime.harness.waitTask((await runtime.harness.triggerSleepCycle(conv.conversationId)).taskId);
			expect(second.status).toBe("completed");
			expect((second.result as { proposedSkills?: unknown }).proposedSkills).toBeUndefined();
			expect(listProposedSkills(skillsDir)).toEqual([name]);
			expect(runtime.simLog.list({ tenantId: TENANT.tenantId, targetModule: "skill-forge" })).toHaveLength(1);

			// other tenant's cycle is attributed to that tenant only
			const third = await runtime.harness.waitTask((await runtime.harness.triggerSleepCycle(otherConv.conversationId)).taskId);
			expect(third.status).toBe("completed");
			expect(runtime.simLog.list({ tenantId: OTHER.tenantId, targetModule: "skill-forge" })).toHaveLength(0); // already proposed globally
			expect(runtime.simLog.list({ tenantId: TENANT.tenantId, targetModule: "skill-forge" })).toHaveLength(1);

			// explicit approval activates it
			promoteProposedSkill(skillsDir, name);
			expect(loadSkillCatalog(skillsDir).map((s) => s.name)).toEqual([name]);
		} finally {
			await runtime.close();
		}
	});

	it("retries the fast model with backoff, falls back deterministically, and still completes the learn phase", async () => {
		const skillsDir = tempDir("helios-wiring-skills-");
		let calls = 0;
		const delays: number[] = [];
		const runtime = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, skillsDir },
			sleepCycleScheduler: false,
			capabilities: {
				fastModel: {
					completeJson: async () => {
						calls += 1;
						throw new Error("fast model down");
					},
				},
			},
			consolidationRetry: { maxAttempts: 3, baseDelayMs: 10, sleep: async (ms) => void delays.push(ms) },
		});
		try {
			const conv = await runtime.harness.createConversation({ tenant: TENANT });
			await seedTelemetry(runtime, TENANT, conv.conversationId);
			await runtime.capabilities.memory.recordEpisode({
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				conversationId: conv.conversationId,
				runId: "r1",
				userMessage: "I prefer tabs over spaces",
				assistantSummary: "ok",
				toolTrace: [],
				frictionSignals: [],
				frictionScore: 0,
			});

			const settled = await runtime.harness.waitTask((await runtime.harness.triggerSleepCycle(conv.conversationId)).taskId);
			expect(settled.status).toBe("completed");
			const result = settled.result as {
				consolidation: { processedEpisodes: number };
				warnings?: string[];
				proposedSkills?: unknown[];
			};
			expect(calls).toBe(3);
			expect(delays).toEqual([10, 20]);
			expect(result.warnings?.[0]).toMatch(/fast model failed after 3 attempt/);
			expect(result.consolidation.processedEpisodes).toBe(1);
			expect(result.proposedSkills).toHaveLength(1); // learn phase still ran
			expect(
				await runtime.capabilities.memory.listUnconsolidatedEpisodes(TENANT.tenantId, TENANT.userId),
			).toHaveLength(0);
			expect(runtime.simLog.list({ tenantId: TENANT.tenantId, targetModule: "memory-consolidation" })[0]?.notes).toMatch(
				/deterministic fallback/,
			);
		} finally {
			await runtime.close();
		}
	});

	it("the scheduler started by bootstrapHarness consolidates pending episodes and unwinds on close()", async () => {
		const skillsDir = tempDir("helios-wiring-skills-");
		const timer = new ManualTimer();
		const runtime = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, skillsDir, sleepCycleIntervalMs: 1000 },
			sleepCycleScheduler: { timer, random: () => 0.5 },
		});
		try {
			expect(timer.pending.size).toBe(1);
			const conv = await runtime.harness.createConversation({ tenant: TENANT });
			await runtime.capabilities.memory.recordEpisode({
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				conversationId: conv.conversationId,
				runId: "r1",
				userMessage: "I prefer dark mode",
				assistantSummary: "ok",
				toolTrace: [],
				frictionSignals: [],
				frictionScore: 0,
			});
			runtime.testModel.enqueueFastJson({ operations: [{ op: "NOOP", reason: "none" }] });

			const report = await runtime.sleepCycleScheduler.tick();
			expect(report.triggered.map((t) => t.tenantId)).toEqual([TENANT.tenantId]);
			await runtime.sleepCycleScheduler.idle();
			expect(
				await runtime.capabilities.memory.listUnconsolidatedEpisodes(TENANT.tenantId, TENANT.userId),
			).toHaveLength(0);

			// nothing pending -> nothing triggered
			expect((await runtime.sleepCycleScheduler.tick()).triggered).toEqual([]);
		} finally {
			await runtime.close();
		}
		expect(timer.pending.size).toBe(0);
	});

	it("interval 0 / flag disable the scheduler in bootstrapHarness", async () => {
		const timer = new ManualTimer();
		const a = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, sleepCycleIntervalMs: 0 },
			sleepCycleScheduler: { timer },
		});
		const b = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, sleepCycleSchedulerEnabled: false },
			sleepCycleScheduler: { timer },
		});
		try {
			expect(timer.pending.size).toBe(0);
			expect(a.sleepCycleScheduler.enabled).toBe(false);
			expect(b.sleepCycleScheduler.enabled).toBe(false);
		} finally {
			await a.close();
			await b.close();
		}
	});
});
