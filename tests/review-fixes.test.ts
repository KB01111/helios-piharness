import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BudgetGate, buildViolation, createBudgetGatedModels } from "../src/runtime/budget.js";
import { proposeSkillDrafts } from "../src/learning/skill-promotion.js";
import { sanitizeMemoryText } from "../src/memory/sanitize.js";
import {
	AppDatabase,
	loadConfig,
	SleepCycleScheduler,
	SqliteMemoryService,
	type EmbeddingProvider,
	type PendingEpisodeScope,
	type SleepSchedulerTimer,
} from "../src/index.js";

const T = "tenant-rf";
const U = "user-rf";

const noTimer: SleepSchedulerTimer = { setTimeout: () => 0, clearTimeout: () => undefined };

function deferred<V>() {
	let resolve!: (value: V) => void;
	const promise = new Promise<V>((res) => {
		resolve = res;
	});
	return { promise, resolve };
}

async function flush(): Promise<void> {
	for (let i = 0; i < 10; i++) await Promise.resolve();
}

// ─── forgetMemory scoping / tombstone ────────────────────────────────────────

describe("forgetMemory", () => {
	function provider(): EmbeddingProvider {
		return {
			model: "m1",
			async embed(texts) {
				return texts.map((t) => [t.toLowerCase().includes("neovim") ? 1 : 0, 1]);
			},
		};
	}

	it("a cross-tenant forget changes nothing: index and embeddings of the real owner stay intact", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const memory = new SqliteMemoryService(appDb, { embeddingProvider: provider() });
			const mem = await memory.pinMemory(T, U, { key: "user.editor", content: "Daily driver is neovim" });
			await memory.search(T, U, "neovim"); // lazily stores the embedding

			expect(await memory.forgetMemory("attacker-tenant", "mallory", mem.id)).toBe(false);

			const fts = appDb.db.prepare("SELECT COUNT(*) AS c FROM longterm_memories_fts WHERE memory_id = ?").get(mem.id) as { c: number };
			const emb = appDb.db.prepare("SELECT COUNT(*) AS c FROM memory_embeddings WHERE memory_id = ?").get(mem.id) as { c: number };
			expect(fts.c).toBe(1);
			expect(emb.c).toBe(1);
			expect((await memory.search(T, U, "neovim")).map((h) => h.memory.id)).toEqual([mem.id]);
		} finally {
			appDb.close();
		}
	});

	it("blanks the deleted text, keeps tombstone metadata, and purges superseded predecessors of the chain", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const memory = new SqliteMemoryService(appDb);
			const v1 = (await memory.applyMemoryMutation(T, U, { op: "ADD", key: "user.secret", kind: "fact", content: "first secret", confidence: 0.9 }))!;
			const v2 = (await memory.applyMemoryMutation(T, U, {
				op: "SUPERSEDE",
				targetId: v1.id,
				key: "user.secret",
				kind: "fact",
				content: "second secret",
				confidence: 0.9,
			}))!;
			expect(await memory.forgetMemory(T, U, v2.id)).toBe(true);

			const all = await memory.listMemories(T, U, { includeSuperseded: true, includeDeleted: true });
			expect(all).toHaveLength(2);
			for (const row of all) {
				expect(row.content).toBe("");
				expect(row.status).toBe("deleted");
			}
			expect(all.find((m) => m.id === v2.id)?.key).toBe("user.secret"); // tombstone metadata kept
			const fts = appDb.db.prepare("SELECT COUNT(*) AS c FROM longterm_memories_fts WHERE memory_id IN (?, ?)").get(v1.id, v2.id) as { c: number };
			expect(fts.c).toBe(0);
			expect(await memory.forgetMemory(T, U, v2.id)).toBe(false);
		} finally {
			appDb.close();
		}
	});
});

// ─── Embeddings robustness ───────────────────────────────────────────────────

describe("embeddings", () => {
	const hash = (key: string, content: string): string => crypto.createHash("sha1").update(`${key}: ${content}`).digest("hex");

	it("ignores non-finite vectors, never stores them, and keeps vector search working", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const embeddingProvider: EmbeddingProvider = {
				model: "nan-model",
				async embed(texts) {
					return texts.map((t) => (t.includes("broken") ? [Number.NaN, 1] : [1, 0]));
				},
			};
			const memory = new SqliteMemoryService(appDb, { embeddingProvider });
			await memory.pinMemory(T, U, { key: "user.ok", content: "fine entry" });
			const bad = await memory.pinMemory(T, U, { key: "user.bad", content: "broken entry" });
			const hits = await memory.search(T, U, "entry");
			expect(hits.length).toBe(2);
			const rows = appDb.db.prepare("SELECT memory_id FROM memory_embeddings").all() as Array<{ memory_id: string }>;
			expect(rows.map((r) => r.memory_id)).not.toContain(bad.id);
			expect(rows).toHaveLength(1);
		} finally {
			appDb.close();
		}
	});

	it("skips an undecodable stored vector (re-embeds it) instead of disabling vector search", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const calls: string[][] = [];
			const embeddingProvider: EmbeddingProvider = {
				model: "m",
				async embed(texts) {
					calls.push([...texts]);
					return texts.map((t) => (t.toLowerCase().includes("editor") ? [1, 0] : [0, 1]));
				},
			};
			const memory = new SqliteMemoryService(appDb, { embeddingProvider });
			const mem = await memory.pinMemory(T, U, { key: "user.tool", content: "daily editor" });
			// 3 bytes: not a multiple of 4, so Float32Array decoding throws
			appDb.db
				.prepare(
					"INSERT INTO memory_embeddings (memory_id, tenant_id, user_id, model, dim, content_hash, vector, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
				)
				.run(mem.id, T, U, "m", 3, hash("user.tool", "daily editor"), new Uint8Array([1, 2, 3]), 1);

			const hits = await memory.search(T, U, "text editor");
			expect(hits.map((h) => h.memory.id)).toEqual([mem.id]);
			expect(calls.at(-1)).toHaveLength(2); // query + the re-embedded corrupt row
			const stored = appDb.db.prepare("SELECT length(vector) AS n FROM memory_embeddings WHERE memory_id = ?").get(mem.id) as { n: number };
			expect(stored.n).toBe(8);
		} finally {
			appDb.close();
		}
	});

	it("does not resurrect an embedding for a memory forgotten while the provider was embedding", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			let memory!: SqliteMemoryService;
			let targetId = "";
			const embeddingProvider: EmbeddingProvider = {
				model: "race",
				async embed(texts) {
					if (targetId) await memory.forgetMemory(T, U, targetId);
					return texts.map(() => [1, 0]);
				},
			};
			memory = new SqliteMemoryService(appDb, { embeddingProvider });
			const mem = await memory.pinMemory(T, U, { key: "user.tmp", content: "temporary note" });
			targetId = mem.id;
			await memory.search(T, U, "temporary");
			const emb = appDb.db.prepare("SELECT COUNT(*) AS c FROM memory_embeddings WHERE memory_id = ?").get(mem.id) as { c: number };
			expect(emb.c).toBe(0);
		} finally {
			appDb.close();
		}
	});
});

// ─── db.open ─────────────────────────────────────────────────────────────────

describe("AppDatabase.open", () => {
	it("closes the handle when a migration throws, and a re-open of an already-migrated db is a no-op", () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "db-open-"));
		try {
			const dbPath = path.join(dir, "app.db");
			const ok = AppDatabase.open({ dbPath });
			ok.close();
			AppDatabase.open({ dbPath }).close(); // idempotent re-open

			const broken = path.join(dir, "broken.db");
			// A VIEW squatting on a table name makes the v1 migration fail at CREATE INDEX.
			const { DatabaseSync } = require_node_sqlite();
			const raw = new DatabaseSync(broken);
			raw.exec("CREATE VIEW midterm_episodes AS SELECT 1 AS id;");
			raw.close();
			expect(() => AppDatabase.open({ dbPath: broken })).toThrow();
			// If the failed open leaked its handle, Windows cannot delete the file.
			expect(() => fs.rmSync(dir, { recursive: true, force: true })).not.toThrow();
			expect(fs.existsSync(broken)).toBe(false);
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});
});

function require_node_sqlite(): typeof import("node:sqlite") {
	// eslint-disable-next-line @typescript-eslint/no-require-imports
	return process.getBuiltinModule("node:sqlite") as typeof import("node:sqlite");
}

// ─── skill promotion ─────────────────────────────────────────────────────────

describe("proposeSkillDrafts existence checks", () => {
	const draft = { suggestedSkillName: "deploy-flow", draftSkillMarkdown: "---\nname: deploy-flow\ndescription: d\n---\nbody\n" };
	const provenance = { tenantId: "t", source: "test", proposedAt: 1 };

	it("an empty leftover directory does not block the proposal; REJECTED does; no temp files remain", () => {
		const skillsDir = fs.mkdtempSync(path.join(os.tmpdir(), "skills-"));
		try {
			const dir = path.join(skillsDir, "_proposed", "deploy-flow");
			fs.mkdirSync(dir, { recursive: true });

			const first = proposeSkillDrafts(skillsDir, [draft], { provenance });
			expect(first.created.map((c) => c.name)).toEqual(["deploy-flow"]);
			expect(fs.readdirSync(dir)).toEqual(["SKILL.md"]);

			const again = proposeSkillDrafts(skillsDir, [draft], { provenance });
			expect(again.skipped).toEqual([{ name: "deploy-flow", reason: "already-proposed" }]);

			fs.rmSync(path.join(dir, "SKILL.md"));
			fs.writeFileSync(path.join(dir, "REJECTED"), "rejected-at: x\n");
			const rejected = proposeSkillDrafts(skillsDir, [draft], { provenance });
			expect(rejected.created).toEqual([]);
			expect(rejected.skipped).toEqual([{ name: "deploy-flow", reason: "already-proposed" }]);
		} finally {
			fs.rmSync(skillsDir, { recursive: true, force: true });
		}
	});
});

// ─── sanitize ────────────────────────────────────────────────────────────────

describe("sanitizeMemoryText line breaks and invisibles", () => {
	it("turns NEL / VT / FF into spaces so words do not glue together", () => {
		expect(sanitizeMemoryText("foo\u0085bar").text).toBe("foo bar");
		expect(sanitizeMemoryText("a\u000Bb\u000Cc").text).toBe("a b c");
		expect(sanitizeMemoryText("one two three").text).toBe("one two three");
	});

	it("strips CGJ, ALM, Mongolian vowel separator and variation selectors 15/16", () => {
		expect(sanitizeMemoryText("x͏y").text).toBe("xy");
		expect(sanitizeMemoryText("x؜y").text).toBe("xy");
		expect(sanitizeMemoryText("x᠎y").text).toBe("xy");
		expect(sanitizeMemoryText("❤️ ok︎").text).toBe("❤ ok");
	});

	it("stays idempotent", () => {
		const once = sanitizeMemoryText("a\u0085b͏ c").text;
		expect(sanitizeMemoryText(once).text).toBe(once);
	});
});

// ─── budget ──────────────────────────────────────────────────────────────────

describe("budget hardening", () => {
	afterEach(() => vi.restoreAllMocks());

	it("warns loudly (and treats as unlimited) for invalid limit env values", () => {
		const writes: string[] = [];
		vi.spyOn(process.stderr, "write").mockImplementation(((chunk: string | Uint8Array) => {
			writes.push(String(chunk));
			return true;
		}) as typeof process.stderr.write);
		const config = loadConfig({
			USE_MOCK_MODELS: "true",
			MAX_CONVERSATION_TOKENS: "50k",
			MAX_TENANT_DAILY_COST_USD: "-3",
			MAX_TENANT_DAILY_TOKENS: "1000",
		});
		expect(config.maxConversationTokens).toBe(0);
		expect(config.maxTenantDailyCostUsd).toBe(0);
		expect(config.maxTenantDailyTokens).toBe(1000);
		const warnings = writes.filter((w) => w.includes("invalid_budget_limit"));
		expect(warnings).toHaveLength(2);
		expect(warnings.join("")).toContain("MAX_CONVERSATION_TOKENS");
		expect(warnings.join("")).toContain("NOT enforced");
	});

	it("does not warn for unset or blank limits", () => {
		const writes: string[] = [];
		vi.spyOn(process.stderr, "write").mockImplementation(((chunk: string | Uint8Array) => {
			writes.push(String(chunk));
			return true;
		}) as typeof process.stderr.write);
		loadConfig({ USE_MOCK_MODELS: "true", MAX_CONVERSATION_TOKENS: "  " });
		expect(writes.filter((w) => w.includes("invalid_budget_limit"))).toHaveLength(0);
	});

	it("gives every blocked message its own usage object", async () => {
		const gate = new BudgetGate();
		const models = {
			streamSimple: () => {
				throw new Error("must not reach the provider");
			},
			stream: () => {
				throw new Error("must not reach the provider");
			},
		};
		const gated = createBudgetGatedModels(models as never, gate);
		const model = { api: "test", provider: "p", id: "m" };
		const messages = [];
		for (let i = 0; i < 2; i++) {
			gate.block("s1", buildViolation("conversation", "tokens", 1, 2));
			const stream = (gated as unknown as { streamSimple: (...a: unknown[]) => { result(): Promise<{ usage: { input: number; cost: { total: number } } }> } }).streamSimple(
				model,
				{ messages: [] },
				{ sessionId: "s1" },
			);
			messages.push(await stream.result());
		}
		messages[0]!.usage.input = 99;
		messages[0]!.usage.cost.total = 5;
		expect(messages[1]!.usage.input).toBe(0);
		expect(messages[1]!.usage.cost.total).toBe(0);
		expect(messages[0]!.usage).not.toBe(messages[1]!.usage);
	});
});

// ─── sleep scheduler ─────────────────────────────────────────────────────────

describe("SleepCycleScheduler hardening", () => {
	function scope(tenantId: string, userId: string, conversationIds: string[]): PendingEpisodeScope {
		return { tenantId, userId, pendingCount: 2, latestEpisodeAt: 1, conversationIds };
	}
	function memoryOf(scopes: PendingEpisodeScope[]) {
		return { listTenantsWithPendingEpisodes: async () => scopes } as never;
	}

	it("counts a completed cycle whose result carries consolidation warnings as a failure (backoff)", async () => {
		let nowMs = 1_000;
		const waits: Array<ReturnType<typeof deferred<{ taskId: string; status: "completed"; result: unknown }>>> = [];
		const scheduler = new SleepCycleScheduler({
			harness: {
				triggerSleepCycle: async () => ({ taskId: "t1" }),
				waitTask: async () => {
					const d = deferred<{ taskId: string; status: "completed"; result: unknown }>();
					waits.push(d);
					return d.promise;
				},
			},
			memory: memoryOf([scope("a", "u", ["c1"])]),
			intervalMs: 1000,
			now: () => nowMs,
			random: () => 0.5,
			timer: noTimer,
		});
		await scheduler.tick();
		waits[0]!.resolve({ taskId: "t1", status: "completed", result: { warnings: ["consolidation failed: boom"] } });
		await scheduler.idle();
		expect((await scheduler.tick()).skippedBackoff).toEqual(["a"]);

		nowMs += 1_001;
		expect((await scheduler.tick()).triggered).toHaveLength(1);
		waits[1]!.resolve({ taskId: "t1", status: "completed", result: { warnings: ["skill proposal failed: x"] } });
		await scheduler.idle();
		// a non-consolidation warning is not a consolidation failure: no backoff
		expect((await scheduler.tick()).triggered).toHaveLength(1);
		waits[2]!.resolve({ taskId: "t1", status: "completed", result: {} });
		await scheduler.idle();
	});

	it("moves on to the next user of a tenant instead of starving behind a failing scope", async () => {
		const attempts: string[] = [];
		const waits: Array<ReturnType<typeof deferred<{ taskId: string; status: "completed" }>>> = [];
		let counter = 0;
		const scheduler = new SleepCycleScheduler({
			harness: {
				triggerSleepCycle: async (conversationId: string) => {
					attempts.push(conversationId);
					if (conversationId === "bad") throw new Error("host conversation gone");
					counter += 1;
					return { taskId: `t${counter}` };
				},
				waitTask: async (taskId: string) => {
					const d = deferred<{ taskId: string; status: "completed" }>();
					waits.push(d);
					const settled = await d.promise;
					return { ...settled, taskId };
				},
			},
			memory: memoryOf([scope("a", "u1", ["bad"]), scope("a", "u2", ["good"])]),
			intervalMs: 1000,
			now: () => 5_000,
			random: () => 0.5,
			timer: noTimer,
		});

		const first = await scheduler.tick();
		expect(first.failed).toHaveLength(1);
		expect(first.triggered.map((t) => t.userId)).toEqual(["u2"]);
		waits[0]!.resolve({ taskId: "x", status: "completed" });
		await scheduler.idle();

		// u1 is now in backoff; u2 must still be served
		const second = await scheduler.tick();
		expect(second.skippedBackoff).toEqual(["a"]);
		expect(second.triggered.map((t) => t.userId)).toEqual(["u2"]);
		expect(attempts).toEqual(["bad", "good", "good"]);
		waits[1]!.resolve({ taskId: "x", status: "completed" });
		await scheduler.idle();
	});

	it("a marker write failure after the task was triggered still tracks the task and keeps the lock", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const errors: string[] = [];
			const waits: Array<ReturnType<typeof deferred<{ taskId: string; status: "completed" }>>> = [];
			let triggers = 0;
			const scheduler = new SleepCycleScheduler({
				harness: {
					triggerSleepCycle: async () => {
						triggers += 1;
						return { taskId: `t${triggers}` };
					},
					waitTask: async (taskId: string) => {
						const d = deferred<{ taskId: string; status: "completed" }>();
						waits.push(d);
						const settled = await d.promise;
						return { ...settled, taskId };
					},
				},
				memory: memoryOf([scope("a", "u", ["c1", "c2"])]),
				intervalMs: 1000,
				appDb,
				timer: noTimer,
				onError: (e, ctx) => errors.push(`${ctx.phase}:${(e as Error).message}`),
			});
			// Created lazily by the scheduler constructor; make setTask (an UPDATE) fail.
			appDb.db.exec("CREATE TRIGGER fail_set_task BEFORE UPDATE ON sleep_cycle_markers BEGIN SELECT RAISE(ABORT, 'marker boom'); END;");

			const report = await scheduler.tick();
			expect(report.triggered).toEqual([{ tenantId: "a", userId: "u", taskId: "t1" }]);
			expect(report.failed).toEqual([]);
			expect(triggers).toBe(1); // no second cycle on the next conversation
			expect(errors.some((e) => e.startsWith("marker:"))).toBe(true);
			expect(scheduler.isRunning("a")).toBe(true);
			expect((await scheduler.tick()).skippedInFlight.length + (await scheduler.tick()).skippedBackoff.length).toBeGreaterThan(0);
			expect(triggers).toBe(1);

			waits[0]!.resolve({ taskId: "t1", status: "completed" });
			await flush();
			await scheduler.idle();
			expect(scheduler.isRunning("a")).toBe(false);
		} finally {
			appDb.close();
		}
	});
});
