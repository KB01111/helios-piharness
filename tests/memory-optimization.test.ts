import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { describe, expect, it } from "vitest";
import {
	AppDatabase,
	MemoryConsolidator,
	SqliteMemoryService,
	buildRelativeTemporalHints,
	detectHistoricalIntent,
	extractEntities,
	extractTemporalAnchors,
	scoreTemporalAlignment,
	type EmbeddingProvider,
	type FastModelClient,
} from "../src/index.js";

const T = "tenant-opt";
const U = "user-opt";

describe("Helios optimized memory architecture", () => {
	describe("1. Multi-hop entity-graph Personalized PageRank (PPR)", () => {
		it("bridges 2-hop entity chains across separate memories in pure lexical mode", async () => {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			try {
				const memory = new SqliteMemoryService(appDb);
				const hop1 = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "org.project.atlas.lead",
					kind: "fact",
					content: "Project Atlas is led by staff engineer Marcus Chen.",
					confidence: 0.92,
				}))!;
				const hop2 = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "team.marcus_chen.telemetry_db",
					kind: "fact",
					content: "Marcus Chen's team stores high-volume telemetry analytics in ClickHouse.",
					confidence: 0.91,
				}))!;
				await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "user.coffee",
					kind: "preference",
					content: "Drinks Ethiopian Yirgacheffe pour-over coffee.",
					confidence: 0.9,
				});

				// Query mentions "Project Atlas" and "database", NOT "Marcus Chen" or "ClickHouse"
				const hits = await memory.search(T, U, "What database does Project Atlas use for analytics?", { limit: 5 });
				const ids = hits.map((h) => h.memory.id);
				expect(ids).toContain(hop1.id);
				expect(ids).toContain(hop2.id);
				// Verify the graph lane contributed signal to the 2-hop target
				const hop2Hit = hits.find((h) => h.memory.id === hop2.id);
				expect(hop2Hit?.matchSignals?.graph).toBeGreaterThan(0);

				// With useGraph: false, pure-entity-bridge hops with no direct lexical overlap lose graph signal
				const noGraph = await memory.search(T, U, "Who manages Project Atlas telemetry?", {
					limit: 5,
					useGraph: false,
				});
				expect(noGraph.every((h) => h.matchSignals?.graph === undefined)).toBe(true);
			} finally {
				appDb.close();
			}
		});

		it("extracts proper nouns and technical identifiers in linear time on adversarial input", () => {
			const ents = extractEntities("team.marcus_chen.config", "Dr. Priya Patel uses pg_bouncer and maxRetryCount in us-east-1.");
			const names = ents.map((e) => e.canonicalName);
			expect(names).toContain("priya patel");
			expect(names).toContain("pg bouncer");
			expect(names).toContain("maxretrycount");
			expect(names).toContain("us-east-1");

			const start = performance.now();
			extractEntities("k", "Alpha Beta ".repeat(50_000));
			expect(performance.now() - start).toBeLessThan(200);
		});
	});

	describe("2. Bi-temporal anchors, tightness ranking, and historical-audit routing", () => {
		it("extracts dates/ranges and ranks tighter temporal matches higher", () => {
			const qRange = extractTemporalAnchors("What happened between 2015 and 2016?");
			expect(qRange).toHaveLength(1);
			expect(qRange[0]?.precision).toBe("range");

			const qAug = extractTemporalAnchors("What happened in August 1999?");
			const mDay = extractTemporalAnchors("On August 11, 1999, watched a total solar eclipse.");
			const mYear = extractTemporalAnchors("In 1999, learned Linux networking.");
			const mOther = extractTemporalAnchors("In March 2018, moved to Berlin.");

			const dayScore = scoreTemporalAlignment(qAug, mDay);
			const yearScore = scoreTemporalAlignment(qAug, mYear);
			const conflictScore = scoreTemporalAlignment(qAug, mOther);

			expect(dayScore.matched).toBe(true);
			expect(yearScore.matched).toBe(true);
			expect(dayScore.score).toBeGreaterThan(yearScore.score);
			expect(conflictScore.matched).toBe(false);
			expect(conflictScore.conflict).toBe(true);
		});

		it("retrieves temporal-range memories via SQL temporal anchors even with minimal lexical overlap", async () => {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			try {
				const memory = new SqliteMemoryService(appDb);
				const m2015 = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "history.alps",
					kind: "episodic",
					content: "In July 2015, hiked Tour du Mont Blanc.",
					confidence: 0.9,
				}))!;
				const m2016 = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "history.marathon",
					kind: "episodic",
					content: "In April 2016, completed Vienna marathon.",
					confidence: 0.9,
				}))!;
				await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "history.2024",
					kind: "episodic",
					content: "On 2024-03-15, shipped Durable Engine v1.0.",
					confidence: 0.95,
				});

				const hits = await memory.search(T, U, "What outdoor events happened between 2015 and 2016?", { limit: 5 });
				const top2 = hits.slice(0, 2).map((h) => h.memory.id);
				expect(top2).toContain(m2015.id);
				expect(top2).toContain(m2016.id);
			} finally {
				appDb.close();
			}
		});

		it("auto-routes explicit historical queries to superseded versions while keeping default queries active-only", async () => {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			let now = 1_700_000_000_000;
			try {
				const memory = new SqliteMemoryService(appDb, { now: () => now });
				const v1 = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "user.package_manager",
					kind: "preference",
					content: "Uses npm as JavaScript package manager.",
					confidence: 0.85,
				}))!;
				now += 10_000;
				const v2 = (await memory.applyMemoryMutation(T, U, {
					op: "SUPERSEDE",
					targetId: v1.id,
					key: "user.package_manager",
					kind: "preference",
					content: "Switched to pnpm for TypeScript monorepos.",
					confidence: 0.92,
				}))!;
				now += 10_000;
				const v3 = (await memory.applyMemoryMutation(T, U, {
					op: "SUPERSEDE",
					targetId: v2.id,
					key: "user.package_manager",
					kind: "preference",
					content: "Migrated from pnpm to bun for fast installs.",
					confidence: 0.96,
				}))!;

				// Verify bi-temporal valid_until was stamped on superseded rows
				expect((await memory.getMemoryById(T, U, v1.id))?.validUntil).toBeDefined();
				expect((await memory.getMemoryById(T, U, v3.id))?.validUntil).toBeUndefined();

				// Present-tense query stays strictly active-only
				const current = await memory.search(T, U, "What JavaScript package manager do I currently use?");
				expect(current.every((h) => h.memory.status === "active")).toBe(true);
				expect(current[0]?.memory.id).toBe(v3.id);

				// Present-tense procedural "before committing" must NOT falsely trigger historical routing
				expect(detectHistoricalIntent("How do I run tests before committing?").isHistorical).toBe(false);

				// Explicit historical query returns superseded predecessors ranked at the top!
				const hist = await memory.search(T, U, "What JavaScript package manager did I use before bun?");
				expect(hist[0]?.memory.status).toBe("superseded");
				expect([v2.id, v1.id]).toContain(hist[0]?.memory.id);

				// Explicit includeSuperseded: false always overrides auto historical routing
				const forcedActive = await memory.search(T, U, "What JavaScript package manager did I use before bun?", {
					includeSuperseded: false,
				});
				expect(forcedActive.every((h) => h.memory.status === "active")).toBe(true);
			} finally {
				appDb.close();
			}
		});
	});

	describe("3. Abstention precision & relevance-gated metadata", () => {
		it("abstains cleanly on unrelated queries even when high-confidence pinned memories exist", async () => {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			try {
				const memory = new SqliteMemoryService(appDb);
				await memory.pinMemory(T, U, {
					key: "user.editor",
					content: "Daily driver editor is Neovim with Lua.",
					confidence: 1.0,
				});
				await memory.pinMemory(T, U, {
					key: "user.deploy",
					content: "Never push production migrations on Fridays.",
					confidence: 1.0,
				});

				expect(await memory.search(T, U, "What brand of luxury yacht do I own?")).toEqual([]);
				expect(await memory.search(T, U, "What piano concerto am I rehearsing for Carnegie Hall?")).toEqual([]);
			} finally {
				appDb.close();
			}
		});
	});

	describe("4. Scalability beyond 500 memories & vector matrix cache coherency", () => {
		it("retrieves the oldest pure-semantic needle across 1,000+ memories and stays coherent across sibling instances", async () => {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			let now = 1_700_000_000_000;
			const provider: EmbeddingProvider = {
				model: "synth-test-v1",
				async embed(texts) {
					return texts.map((t) => {
						if (t.includes("ObsidianArchive") || t.includes("OrchidQuery")) return [1, 0, 0, 0];
						if (t.includes("UpdatedSignal")) return [0, 1, 0, 0];
						return [0, 0, 0.99, 0.14];
					});
				},
			};

			try {
				const svcA = new SqliteMemoryService(appDb, { now: () => now, embeddingProvider: provider });
				const svcB = new SqliteMemoryService(appDb, { now: () => now, embeddingProvider: provider });

				// Plant oldest pure-semantic needle at index 0, then 650 newer memories (past old LIMIT 500!)
				const needle = (await svcA.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "archive.oldest",
					kind: "fact",
					content: "ObsidianArchive onboarding blueprint.",
					confidence: 0.9,
				}))!;
				for (let i = 0; i < 650; i++) {
					now += 1000;
					await svcA.applyMemoryMutation(T, U, {
						op: "ADD",
						key: `filler.item_${i}`,
						kind: "fact",
						content: `Routine telemetry record number ${i}.`,
						confidence: 0.85,
					});
				}
				await svcA.backfillEmbeddings(T, U, { limit: 1000 });

				// Pure-semantic query with ZERO lexical overlap finds the oldest memory past 500!
				const hits = await svcA.search(T, U, "OrchidQuery", { limit: 5 });
				expect(hits[0]?.memory.id).toBe(needle.id);

				// Sibling instance updates the needle -> svcA's packed vector cache MUST invalidate automatically!
				await svcB.applyMemoryMutation(T, U, {
					op: "UPDATE",
					targetId: needle.id,
					content: "UpdatedSignal replaced old blueprint.",
					confidence: 0.95,
				});
				const afterUpdate = await svcA.search(T, U, "OrchidQuery", { limit: 5 });
				expect(afterUpdate.some((h) => h.memory.id === needle.id)).toBe(false);
			} finally {
				appDb.close();
			}
		});
	});

	describe("5. Recent unconsolidated episodes, GDPR purge, and tenant isolation", () => {
		it("searches unconsolidated episodes when includeEpisodes is true, isolates tenants, and purges on consolidation/forget", async () => {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			try {
				const memory = new SqliteMemoryService(appDb);
				const ep = await memory.recordEpisode({
					tenantId: T,
					userId: U,
					conversationId: "conv-live",
					runId: "run-1",
					userMessage: "Temporary staging rotation ticket is SEC-4819 on Thursday.",
					assistantSummary: "Noted SEC-4819.",
					toolTrace: [],
					frictionSignals: [],
					frictionScore: 0,
				});
				await memory.recordEpisode({
					tenantId: "other-tenant",
					userId: U,
					conversationId: "conv-other",
					runId: "run-1",
					userMessage: "Secret ticket SEC-4819 for another tenant.",
					assistantSummary: "Noted.",
					toolTrace: [],
					frictionSignals: [],
					frictionScore: 0,
				});

				// Default search omits unconsolidated episodes
				expect(await memory.search(T, U, "SEC-4819")).toHaveLength(0);

				// includeEpisodes: true finds ONLY this tenant's episode
				const epHits = await memory.search(T, U, "staging rotation ticket SEC-4819", { includeEpisodes: true });
				expect(epHits).toHaveLength(1);
				expect(epHits[0]?.memory.id).toBe(ep.id);
				expect(epHits[0]?.source).toBe("episode");

				// Marking episode consolidated removes it from unconsolidated episode search
				await memory.markEpisodesConsolidated([ep.id]);
				expect(await memory.search(T, U, "SEC-4819", { includeEpisodes: true })).toHaveLength(0);

				// GDPR forgetMemory purges entities, links, relations, and temporal anchors atomically
				const secretMem = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "secret.project",
					kind: "fact",
					content: "On 2024-03-15, ZephyrSecret collaborated with OrionSecret.",
					confidence: 0.9,
				}))!;
				expect(
					(appDb.db.prepare("SELECT COUNT(*) AS c FROM memory_entity_links WHERE memory_id = ?").get(secretMem.id) as { c: number }).c,
				).toBeGreaterThan(0);
				expect(
					(appDb.db.prepare("SELECT COUNT(*) AS c FROM memory_temporal_anchors WHERE memory_id = ?").get(secretMem.id) as { c: number }).c,
				).toBeGreaterThan(0);

				expect(await memory.forgetMemory(T, U, secretMem.id)).toBe(true);
				expect(
					(appDb.db.prepare("SELECT COUNT(*) AS c FROM memory_entity_links WHERE memory_id = ?").get(secretMem.id) as { c: number }).c,
				).toBe(0);
				expect(
					(appDb.db.prepare("SELECT COUNT(*) AS c FROM memory_relations WHERE memory_id = ?").get(secretMem.id) as { c: number }).c,
				).toBe(0);
				expect(
					(appDb.db.prepare("SELECT COUNT(*) AS c FROM memory_temporal_anchors WHERE memory_id = ?").get(secretMem.id) as { c: number }).c,
				).toBe(0);
				expect(
					(appDb.db.prepare("SELECT COUNT(*) AS c FROM memory_entities WHERE tenant_id = ? AND canonical_name LIKE '%secret%'").get(T) as { c: number }).c,
				).toBe(0);
			} finally {
				appDb.close();
			}
		});
	});

	describe("6. Streaming feedback weights, access tracking, and upgraded consolidator", () => {
		it("updates feedback weights via streaming formula and ranks higher-utility memories first in core memory", async () => {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			try {
				const memory = new SqliteMemoryService(appDb);
				const a = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "user.note.a",
					kind: "fact",
					content: "Note A about architecture.",
					confidence: 0.85,
				}))!;
				const b = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "user.note.b",
					kind: "fact",
					content: "Note B about architecture.",
					confidence: 0.85,
				}))!;

				await memory.applyFeedback(T, U, [b.id], 1.0, 0.5); // 0.5 -> 0.75
				await memory.applyFeedback(T, U, [a.id], 0.0, 0.5); // 0.5 -> 0.25
				await memory.recordAccess(T, U, [b.id]);

				const bAfter = await memory.getMemoryById(T, U, b.id);
				expect(bAfter?.feedbackWeight).toBeCloseTo(0.75, 5);
				expect(bAfter?.accessCount).toBe(1);

				const core = await memory.getCoreMemory(T, U, 10);
				expect(core[0]?.id).toBe(b.id);
			} finally {
				appDb.close();
			}
		});

		it("consolidator derives topic-specific keys in fallback, auto-supersedes cross-key contradictions, and injects temporal hints", async () => {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			try {
				const memory = new SqliteMemoryService(appDb);

				// 1. Model-free fallback keeps distinct preferences under separate topic keys instead of overwriting general
				await memory.recordEpisode({
					tenantId: T,
					userId: U,
					conversationId: "c1",
					runId: "r1",
					userMessage: "I prefer Neovim for coding.",
					assistantSummary: "ok",
					toolTrace: [],
					frictionSignals: [],
					frictionScore: 0,
				});
				await memory.recordEpisode({
					tenantId: T,
					userId: U,
					conversationId: "c1",
					runId: "r2",
					userMessage: "Always use dark mode for editors.",
					assistantSummary: "ok",
					toolTrace: [],
					frictionSignals: [],
					frictionScore: 0,
				});
				const fallback = new MemoryConsolidator({ memory });
				const r1 = await fallback.consolidate(T, U);
				expect(r1.addedIds).toHaveLength(2);
				const keys = (await memory.listMemories(T, U)).map((m) => m.key).sort();
				expect(keys).toEqual(["user.editor", "user.theme"]);

				// 2. Cross-key contradiction guard: fast model emits ADD under drifted key replacing npm -> auto-SUPERSEDE!
				const oldPm = (await memory.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "user.package_manager",
					kind: "preference",
					content: "Prefer npm for JavaScript packages.",
					confidence: 0.9,
				}))!;
				await memory.recordEpisode({
					tenantId: T,
					userId: U,
					conversationId: "c2",
					runId: "r3",
					userMessage: "Yesterday I switched from npm to pnpm; from now on use pnpm.",
					assistantSummary: "Switched.",
					toolTrace: [],
					frictionSignals: [],
					frictionScore: 0,
					createdAt: Date.UTC(2026, 10, 25, 12, 0, 0),
				});

				let capturedPrompt = "";
				const stub: FastModelClient = {
					async completeJson<R>(_sys: string, usr: string) {
						capturedPrompt = usr;
						const plan = {
							operations: [
								{
									op: "ADD",
									key: "tools.js_pkg_manager",
									kind: "preference",
									content: "Switched from npm to pnpm for all JS repos.",
									confidence: 0.95,
								},
							],
						};
						return { rawText: JSON.stringify(plan), parsed: plan as R };
					},
				};
				const consolidator = new MemoryConsolidator({ memory, fastModel: stub });
				const r2 = await consolidator.consolidate(T, U);
				const parsedPrompt = JSON.parse(capturedPrompt) as { episodes: Array<{ temporalHints?: string[] }> };
				expect(parsedPrompt.episodes[0]?.temporalHints).toContain('"yesterday" = 2026-11-24');
				expect(r2.supersededOldIds).toEqual([oldPm.id]);
				expect((await memory.getMemoryById(T, U, oldPm.id))?.status).toBe("superseded");

				// Direct temporal hints helper check
				expect(buildRelativeTemporalHints("Yesterday and last month", Date.UTC(2026, 10, 25))).toEqual([
					'"yesterday" = 2026-11-24',
					'"last month" = 2026-10',
				]);
			} finally {
				appDb.close();
			}
		});

		it("lazily backfills graph and temporal indexes when opening an existing v1/v2 database file", async () => {
			const dir = fs.mkdtempSync(path.join(os.tmpdir(), "helios-v3-mig-"));
			const dbPath = path.join(dir, "app.db");
			try {
				const db1 = AppDatabase.open({ dbPath });
				const mem1 = new SqliteMemoryService(db1);
				await mem1.applyMemoryMutation(T, U, {
					op: "ADD",
					key: "org.project.orion",
					kind: "fact",
					content: "On 2024-03-15, Project Orion was handed to Elena Rostova.",
					confidence: 0.92,
				});
				db1.close();

				// Wipe v3 auxiliary tables as if upgraded from v2
				const raw = new DatabaseSync(dbPath);
				raw.exec("DELETE FROM memory_entity_links; DELETE FROM memory_entities; DELETE FROM memory_temporal_anchors;");
				raw.close();

				const db2 = AppDatabase.open({ dbPath });
				try {
					const mem2 = new SqliteMemoryService(db2);
					const hits = await mem2.search(T, U, "What happened on 2024-03-15?");
					expect(hits[0]?.memory.key).toBe("org.project.orion");
					expect(
						(db2.db.prepare("SELECT COUNT(*) AS c FROM memory_entity_links WHERE tenant_id = ?").get(T) as { c: number }).c,
					).toBeGreaterThan(0);
				} finally {
					db2.close();
				}
			} finally {
				fs.rmSync(dir, { recursive: true, force: true });
			}
		});
	});
});
