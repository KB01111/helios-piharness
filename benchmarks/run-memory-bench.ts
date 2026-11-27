/**
 * Comprehensive AI Agent Memory Benchmark Runner.
 *
 * Benchmarks Helios SqliteMemoryService (lexical + hybrid) head-to-head against:
 *  - Graphiti-Ref (bi-temporal knowledge graph + RRF)
 *  - HippoRAG-Ref (entity-memory bipartite graph + Personalized PageRank)
 *  - Optional real external engines when `--external <dir>` points to a folder with
 *    `@cognee/cognee-ts` (native Rust cognee-rs) and `mem0ai/oss` installed!
 *
 * Metrics measured identically across all engines:
 *  - Overall Recall@1, Recall@3, Recall@5, MRR, nDCG@5
 *  - Per-category accuracy (single_hop, semantic_gap, multi_hop [target + both-hops],
 *    knowledge_update [fresh@1 + zero-stale], temporal_window, historical_audit,
 *    abstention precision, session_recent)
 *  - Latency & scalability at N = 100, 500, 2000 memories (ingest ms/op, search p50/p95 ms)
 *  - Security (prompt-injection sanitization & multi-tenant isolation)
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";
import { AppDatabase } from "../src/memory/db.js";
import { SqliteMemoryService } from "../src/memory/sqlite-memory.js";
import type { EmbeddingProvider } from "../src/contracts/app-capabilities.js";
import { loadEmbedServerConfig } from "../src/embeddings/config.js";
import { buildEmbeddingService } from "../src/embeddings/factory.js";
import { HttpEmbeddingProvider, HttpReranker, RouterEmbeddingProvider } from "../src/embeddings/provider.js";
import type { Reranker } from "../src/contracts/app-capabilities.js";
import {
	BENCHMARK_EPISODES as CORE_EPISODES,
	BENCHMARK_MEMORIES as CORE_MEMORIES,
	BENCHMARK_NOW_MS as CORE_NOW_MS,
	BENCHMARK_QUERIES as CORE_QUERIES,
	type BenchmarkQuery,
	type QueryCategory,
} from "./memory-dataset.js";
import { HELDOUT_EPISODES, HELDOUT_MEMORIES, HELDOUT_NOW_MS, HELDOUT_QUERIES } from "./memory-dataset-heldout.js";

/** `--dataset core` (development set, default) or `--dataset heldout` (report-only generalisation set). */
const DATASET_NAME = (() => {
	const i = process.argv.indexOf("--dataset");
	const v = i >= 0 ? process.argv[i + 1] : "core";
	if (v !== "core" && v !== "heldout") throw new Error(`--dataset must be core or heldout (got ${v})`);
	return v;
})();
const BENCHMARK_MEMORIES = DATASET_NAME === "heldout" ? HELDOUT_MEMORIES : CORE_MEMORIES;
const BENCHMARK_EPISODES = DATASET_NAME === "heldout" ? HELDOUT_EPISODES : CORE_EPISODES;
const BENCHMARK_QUERIES = DATASET_NAME === "heldout" ? HELDOUT_QUERIES : CORE_QUERIES;
const BENCHMARK_NOW_MS = DATASET_NAME === "heldout" ? HELDOUT_NOW_MS : CORE_NOW_MS;
import {
	CachedHttpEmbeddingProvider,
	GraphitiRefEngine,
	HippoRagRefEngine,
	type BenchMemoryEngine,
	type RankedHit,
} from "./reference-engines.js";

const DAY_MS = 86_400_000;
const TENANT = "bench-tenant";
const USER = "bench-user";

export interface CategoryMetrics {
	readonly count: number;
	readonly recallAt1: number;
	readonly recallAt3: number;
	readonly recallAt5: number;
	readonly mrr: number;
	readonly ndcgAt5: number;
	/** Multi-hop: fraction of queries retrieving ALL required hops in top-5. */
	readonly bothHopsAt5?: number;
	/** Knowledge update: fraction of queries with ZERO forbidden/stale IDs in top-3. */
	readonly zeroStaleTop3?: number;
	/** Abstention: fraction of unanswerable queries cleanly abstained (empty or top score < threshold). */
	readonly abstentionRate?: number;
}

export interface EngineBenchmarkResult {
	readonly engine: string;
	readonly ingestMs: number;
	readonly warmMs: number;
	readonly queryLatencyP50Ms: number;
	readonly queryLatencyP95Ms: number;
	readonly overallRecallAt1: number;
	readonly overallRecallAt3: number;
	readonly overallRecallAt5: number;
	readonly overallMrr: number;
	readonly overallNdcgAt5: number;
	readonly abstentionPrecision: number;
	readonly knowledgeUpdateFreshness: number;
	readonly multiHopBothAt5: number;
	readonly byCategory: Record<QueryCategory, CategoryMetrics>;
	readonly failedSampleQueries: Array<{ id: string; category: string; query: string; expected: string[]; got: string[]; topScore: number }>;
}

export interface ScaleBenchmarkRow {
	readonly engine: string;
	readonly memoryCount: number;
	readonly ingestTotalMs: number;
	readonly ingestPerOpMs: number;
	readonly searchP50Ms: number;
	readonly searchP95Ms: number;
	readonly searchP99Ms: number;
	readonly recallAt5: number;
	readonly semanticOldestRecallAt5?: number;
}

function percentile(values: readonly number[], p: number): number {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
	return sorted[idx]!;
}

function computeNdcgAt5(rankedIds: readonly string[], goldIds: ReadonlySet<string>): number {
	if (goldIds.size === 0) return 0;
	let dcg = 0;
	const k = Math.min(5, rankedIds.length);
	for (let i = 0; i < k; i++) {
		if (goldIds.has(rankedIds[i]!)) {
			dcg += 1 / Math.log2(i + 2);
		}
	}
	let idcg = 0;
	const idealCount = Math.min(5, goldIds.size);
	for (let i = 0; i < idealCount; i++) {
		idcg += 1 / Math.log2(i + 2);
	}
	return idcg > 0 ? dcg / idcg : 0;
}

/** Extracts benchmark ID tag `[M01]` / `[MH01B]` from chunk/text payloads returned by external engines. */
function extractTagId(text: string): string | undefined {
	const m = /\[([A-Z0-9_]{2,12})\]/.exec(text);
	return m?.[1];
}

// ── Helios Engine Adapter ───────────────────────────────────────────────────────────────────────

async function buildHeliosEngine(
	label: string,
	embedder: EmbeddingProvider | undefined,
	reranker?: Reranker,
): Promise<{ engine: BenchMemoryEngine; ingestMs: number; warmMs: number }> {
	const tWarm0 = performance.now();
	const appDb = AppDatabase.open({ dbPath: ":memory:" });
	let currentNow = BENCHMARK_NOW_MS;
	const memory = new SqliteMemoryService(appDb, {
		now: () => currentNow,
		...(embedder ? { embeddingProvider: embedder } : {}),
		...(reranker ? { reranker } : {}),
		...(process.argv.includes("--rerank-weight") ? { rerankWeight: Number(process.argv[process.argv.indexOf("--rerank-weight") + 1]) } : {}),
	});
	const warmMs = performance.now() - tWarm0;

	const tIngest0 = performance.now();
	const benchIdToMemId = new Map<string, string>();
	const memIdToBenchId = new Map<string, string>();

	for (const m of BENCHMARK_MEMORIES) {
		currentNow = BENCHMARK_NOW_MS - (m.ageDays ?? 0) * DAY_MS;
		let savedId: string | undefined;
		if (m.supersedesId) {
			const oldMemId = benchIdToMemId.get(m.supersedesId);
			if (oldMemId) {
				const res = await memory.applyMemoryMutation(TENANT, USER, {
					op: "SUPERSEDE",
					targetId: oldMemId,
					key: m.key,
					kind: m.kind,
					content: m.content,
					confidence: m.confidence ?? 0.9,
				});
				savedId = res?.id;
			}
		}
		if (!savedId) {
			if (m.pinned) {
				const res = await memory.pinMemory(TENANT, USER, {
					key: m.key,
					content: m.content,
					kind: m.kind,
					confidence: m.confidence ?? 0.95,
				});
				savedId = res.id;
			} else {
				const res = await memory.applyMemoryMutation(TENANT, USER, {
					op: "ADD",
					key: m.key,
					kind: m.kind,
					content: m.content,
					confidence: m.confidence ?? 0.88,
					evidenceCount: m.evidenceCount ?? 1,
				});
				savedId = res?.id;
			}
		}
		if (savedId) {
			benchIdToMemId.set(m.id, savedId);
			memIdToBenchId.set(savedId, m.id);
		}
	}

	// Record unconsolidated mid-term episodes
	const epDbIdToBenchId = new Map<string, string>();
	for (const ep of BENCHMARK_EPISODES) {
		currentNow = BENCHMARK_NOW_MS - (ep.ageMinutes ?? 5) * 60_000;
		const savedEp = await memory.recordEpisode({
			tenantId: TENANT,
			userId: USER,
			conversationId: ep.conversationId,
			runId: ep.runId,
			userMessage: ep.userMessage,
			assistantSummary: ep.assistantSummary,
			toolTrace: [],
			frictionSignals: [],
			frictionScore: 0,
		});
		epDbIdToBenchId.set(savedEp.id, ep.id);
	}

	currentNow = BENCHMARK_NOW_MS;
	if (embedder) {
		await memory.backfillEmbeddings(TENANT, USER, { limit: 1000 });
	}
	const ingestMs = performance.now() - tIngest0;

	const engine: BenchMemoryEngine = {
		name: label,
		supportsHistorical: true,
		supportsEpisodes: true,
		abstentionThreshold: embedder ? 0.30 : 0.15,
		async search(query, topK) {
			const searchOpts: Record<string, unknown> = { limit: topK, includeEpisodes: true };
			const results = await memory.search(TENANT, USER, query, searchOpts as Parameters<typeof memory.search>[3]);
			return results.map((r) => {
				const mappedMem = memIdToBenchId.get(r.memory.id);
				const mappedEp = epDbIdToBenchId.get(r.memory.id);
				return {
					id: mappedMem ?? mappedEp ?? r.memory.id,
					score: r.score ?? -r.rank,
					text: r.memory.content,
					isEpisode: Boolean(mappedEp) || r.memory.id.startsWith("ep_"),
				};
			});
		},
		close() {
			appDb.close();
		},
	};

	return { engine, ingestMs, warmMs };
}

// ── External Real Competitors (cognee-rs via @cognee/cognee-ts & mem0ai/oss) ────────────────────

async function buildExternalCompetitors(
	externalDir: string,
	endpoint: string,
): Promise<Array<{ engine: BenchMemoryEngine; ingestMs: number; warmMs: number }>> {
	const req = createRequire(path.join(externalDir, "package.json"));
	const out: Array<{ engine: BenchMemoryEngine; ingestMs: number; warmMs: number }> = [];

	// Active-only memories for engines without native SUPERSEDE tombstones, plus superseded tagged so
	// we can faithfully test whether they handle knowledge updates cleanly!
	const supersededIds = new Set(BENCHMARK_MEMORIES.map((m) => m.supersedesId).filter(Boolean));

	// 1. Real mem0ai/oss (v3.3.1)
	try {
		const { Memory } = req("mem0ai/oss");
		const tWarm0 = performance.now();
		const mem0 = new Memory({
			version: "v1.1",
			embedder: {
				provider: "openai",
				config: {
					apiKey: "local-bench",
					model: "sentence-transformers/all-MiniLM-L6-v2",
					baseURL: endpoint,
					embeddingDims: 384,
				},
			},
			vectorStore: {
				provider: "memory",
				config: { collectionName: "bench_mem0", dimension: 384, dbPath: ":memory:" },
			},
			llm: {
				provider: "openai",
				config: { apiKey: "local-bench", model: "gpt-4o-mini", baseURL: endpoint },
			},
			disableHistory: true,
		});
		const warmMs = performance.now() - tWarm0;

		const tIngest0 = performance.now();
		const mem0IdByBench = new Map<string, string>();
		for (const m of BENCHMARK_MEMORIES) {
			if (m.supersedesId && mem0IdByBench.has(m.supersedesId)) {
				// Simulate Mem0's update() behavior on known updates (overwrites prior vector)
				const oldId = mem0IdByBench.get(m.supersedesId)!;
				await mem0.update(oldId, `[${m.id}] ${m.key}: ${m.content}`);
				mem0IdByBench.set(m.id, oldId);
			} else {
				const added = await mem0.add(`[${m.id}] ${m.key}: ${m.content}`, { userId: USER, infer: false });
				const firstId = added?.results?.[0]?.id;
				if (firstId) mem0IdByBench.set(m.id, firstId);
			}
		}
		const ingestMs = performance.now() - tIngest0;

		out.push({
			engine: {
				name: "mem0ai/oss v3.3.1 (real)",
				abstentionThreshold: 0.25,
				async search(query, topK) {
					const res = await mem0.search(query, { filters: { user_id: USER }, topK, threshold: 0.0 });
					return (res?.results ?? []).map((r: { memory?: string; score?: number; id?: string }) => ({
						id: extractTagId(r.memory ?? "") ?? String(r.id ?? ""),
						score: Number(r.score ?? 0),
						text: r.memory ?? "",
					}));
				},
			},
			ingestMs,
			warmMs,
		});
	} catch (err) {
		console.warn("[bench] mem0ai/oss skipped:", err instanceof Error ? err.message : String(err));
	}

	// 2. Real cognee-rs via @cognee/cognee-ts v0.2.0 (native Rust Neon engine!)
	try {
		process.env.COGNEE_BINDING_SUPPRESS_LOGS = "1";
		process.env.TELEMETRY_DISABLED = "1";
		const cogMod = req("@cognee/cognee-ts");
		cogMod.init();
		const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "cognee-bench-"));
		const sysDir = path.join(tempRoot, "sys");
		const dataDir = path.join(tempRoot, "data");
		const cacheDir = path.join(tempRoot, "cache");
		fs.mkdirSync(sysDir, { recursive: true });
		fs.mkdirSync(dataDir, { recursive: true });
		fs.mkdirSync(cacheDir, { recursive: true });

		const tWarm0 = performance.now();
		const cognee = new cogMod.Cognee({
			systemRootDirectory: sysDir,
			dataRootDirectory: dataDir,
			cacheRootDirectory: cacheDir,
			relationalDbUrl: `sqlite:${path.join(sysDir, "cognee.db").replace(/\\/g, "/")}?mode=rwc`,
			graphFilePath: path.join(sysDir, "graph.lbug"),
			vectorDbProvider: "lancedb",
			vectorDbUrl: path.join(sysDir, "lancedb"),
			llmProvider: "openai",
			llmModel: "gpt-4o-mini",
			llmApiKey: "local-bench",
			llmEndpoint: endpoint,
			embeddingProvider: "openai",
			embeddingModel: "sentence-transformers/all-MiniLM-L6-v2",
			embeddingDimensions: 384,
			embeddingEndpoint: endpoint,
			embeddingApiKey: "local-bench",
		});
		await cognee.warm();
		const warmMs = performance.now() - tWarm0;

		// Ingest all active memories (Cognee uses update/delete for knowledge updates; keep active set clean for Cognee)
		const tIngest0 = performance.now();
		const datasetName = "helios_bench";
		const inputs = BENCHMARK_MEMORIES.filter((m) => !supersededIds.has(m.id)).map((m) => ({
			type: "text" as const,
			text: `[${m.id}] ${m.key}: ${m.content}`,
		}));
		await cognee.add(inputs, datasetName);
		const cogStats = await cognee.cognify(datasetName, { summarization: false });
		const ingestMs = performance.now() - tIngest0;
		console.error(`[bench] cognee-rs cognify stats (LanceDB+Ladybug+SQLite):`, cogStats);

		const extractCogneeHits = (res: unknown, topK: number, isTripletDistance = false): RankedHit[] => {
			const hits: RankedHit[] = [];
			const seen = new Set<string>();
			const pushText = (txt: string, score: number) => {
				const tag = extractTagId(txt);
				if (tag && !seen.has(tag)) {
					seen.add(tag);
					hits.push({ id: tag, score, text: txt });
				}
			};
			const raw = res as { result?: { data?: unknown }; context?: Record<string, unknown> };
			const items = Array.isArray(raw?.result?.data) ? raw.result.data : [];
			for (let i = 0; i < items.length; i++) {
				const item = items[i] as { score?: number; payload?: Record<string, unknown> };
				const p = item?.payload ?? {};
				let score: number;
				if (typeof item?.score === "number") {
					// GRAPH_COMPLETION returns triplet distance (lower = better; ~0..29.25 with penalty 6.5)
					score = isTripletDistance ? Math.max(0, 1 - item.score / 20) : item.score;
				} else {
					score = Math.max(0.1, 1 - i * 0.08);
				}
				for (const field of ["text", "source_text", "target_text", "description", "source_description", "target_description", "edge_text"]) {
					if (typeof p[field] === "string") pushText(p[field] as string, score);
				}
				if (Array.isArray(p.edges)) {
					for (const e of p.edges) {
						if (typeof e?.text === "string") pushText(e.text, score * 0.85);
					}
				}
			}
			return hits.slice(0, topK);
		};

		const makeCogneeVariant = (
			searchType: string,
			label: string,
			threshold: number,
			options?: { readonly isTripletDistance?: boolean; readonly isLast?: boolean },
		): BenchMemoryEngine => ({
			name: label,
			abstentionThreshold: threshold,
			async search(query, topK) {
				const res = await cognee.search(query, {
					searchType,
					datasets: [datasetName],
					topK,
					onlyContext: true,
					saveInteraction: false,
				});
				return extractCogneeHits(res, topK, options?.isTripletDistance);
			},
			close() {
				if (options?.isLast) {
					try {
						cogMod.shutdown();
					} catch {
						// ignore
					}
					try {
						fs.rmSync(tempRoot, { recursive: true, force: true });
					} catch {
						// ignore Windows file lock
					}
				}
			},
		});

		out.push({ engine: makeCogneeVariant("CHUNKS", "cognee-rs CHUNKS (real)", 0.35), ingestMs, warmMs });
		out.push({
			engine: makeCogneeVariant("GRAPH_COMPLETION", "cognee-rs GRAPH (real)", 0.35, { isTripletDistance: true }),
			ingestMs,
			warmMs,
		});
		out.push({ engine: makeCogneeVariant("HYBRID_COMPLETION", "cognee-rs HYBRID (real)", 0.35, { isLast: true }), ingestMs, warmMs });
	} catch (err) {
		console.warn("[bench] cognee-rs skipped:", err instanceof Error ? err.message : String(err));
	}

	return out;
}

// ── Evaluation Loop ─────────────────────────────────────────────────────────────────────────────

async function evaluateEngine(
	entry: { engine: BenchMemoryEngine; ingestMs: number; warmMs: number },
): Promise<EngineBenchmarkResult> {
	const { engine, ingestMs, warmMs } = entry;
	const latencies: number[] = [];

	const byCatRaw = new Map<QueryCategory, {
		count: number;
		r1: number;
		r3: number;
		r5: number;
		mrr: number;
		ndcg5: number;
		bothHops: number;
		zeroStale: number;
		abstained: number;
	}>();
	const categories: QueryCategory[] = [
		"single_hop",
		"semantic_gap",
		"multi_hop",
		"knowledge_update",
		"temporal_window",
		"historical_audit",
		"abstention",
		"session_recent",
	];
	for (const c of categories) {
		byCatRaw.set(c, { count: 0, r1: 0, r3: 0, r5: 0, mrr: 0, ndcg5: 0, bothHops: 0, zeroStale: 0, abstained: 0 });
	}

	const failures: EngineBenchmarkResult["failedSampleQueries"] = [];

	for (const q of BENCHMARK_QUERIES) {
		const t0 = performance.now();
		let hits: RankedHit[] = [];
		try {
			hits = await engine.search(q.query, 5);
		} catch {
			hits = [];
		}
		latencies.push(performance.now() - t0);

		const bucket = byCatRaw.get(q.category)!;
		bucket.count += 1;
		const ids = hits.map((h) => h.id);
		const topScore = hits[0]?.score ?? 0;

		if (q.expectAbstain) {
			const abstained = hits.length === 0 || topScore < engine.abstentionThreshold;
			if (abstained) bucket.abstained += 1;
			else {
				failures.push({ id: q.id, category: q.category, query: q.query, expected: ["<ABSTAIN>"], got: ids.slice(0, 3), topScore });
			}
			continue;
		}

		const goldList = q.category === "session_recent" ? (q.expectedEpisodeIds ?? []) : q.expectedIds;
		const goldSet = new Set(goldList);
		const firstHitIdx = ids.findIndex((id) => goldSet.has(id));
		const hit1 = firstHitIdx === 0 ? 1 : 0;
		const hit3 = firstHitIdx >= 0 && firstHitIdx < 3 ? 1 : 0;
		const hit5 = firstHitIdx >= 0 && firstHitIdx < 5 ? 1 : 0;
		const rr = firstHitIdx >= 0 ? 1 / (firstHitIdx + 1) : 0;
		const ndcg = computeNdcgAt5(ids, goldSet);

		bucket.r1 += hit1;
		bucket.r3 += hit3;
		bucket.r5 += hit5;
		bucket.mrr += rr;
		bucket.ndcg5 += ndcg;

		if (q.requiredAllIds && q.requiredAllIds.length > 0) {
			const top5Set = new Set(ids.slice(0, 5));
			if (q.requiredAllIds.every((reqId) => top5Set.has(reqId))) {
				bucket.bothHops += 1;
			}
		}

		if (q.forbiddenIds && q.forbiddenIds.length > 0) {
			const top3Set = new Set(ids.slice(0, 3));
			if (q.forbiddenIds.every((badId) => !top3Set.has(badId))) {
				bucket.zeroStale += 1;
			}
		}

		if (hit5 === 0) {
			failures.push({ id: q.id, category: q.category, query: q.query, expected: [...goldList], got: ids.slice(0, 5), topScore });
		}
	}

	const byCategory = {} as Record<QueryCategory, CategoryMetrics>;
	let totalNonAbstain = 0;
	let sumR1 = 0;
	let sumR3 = 0;
	let sumR5 = 0;
	let sumMrr = 0;
	let sumNdcg = 0;

	for (const c of categories) {
		const b = byCatRaw.get(c)!;
		const n = Math.max(1, b.count);
		byCategory[c] = {
			count: b.count,
			recallAt1: b.r1 / n,
			recallAt3: b.r3 / n,
			recallAt5: b.r5 / n,
			mrr: b.mrr / n,
			ndcgAt5: b.ndcg5 / n,
			...(c === "multi_hop" ? { bothHopsAt5: b.bothHops / n } : {}),
			...(c === "knowledge_update" ? { zeroStaleTop3: b.zeroStale / n } : {}),
			...(c === "abstention" ? { abstentionRate: b.abstained / n } : {}),
		};
		if (c !== "abstention") {
			totalNonAbstain += b.count;
			sumR1 += b.r1;
			sumR3 += b.r3;
			sumR5 += b.r5;
			sumMrr += b.mrr;
			sumNdcg += b.ndcg5;
		}
	}

	return {
		engine: engine.name,
		ingestMs,
		warmMs,
		queryLatencyP50Ms: percentile(latencies, 50),
		queryLatencyP95Ms: percentile(latencies, 95),
		overallRecallAt1: sumR1 / Math.max(1, totalNonAbstain),
		overallRecallAt3: sumR3 / Math.max(1, totalNonAbstain),
		overallRecallAt5: sumR5 / Math.max(1, totalNonAbstain),
		overallMrr: sumMrr / Math.max(1, totalNonAbstain),
		overallNdcgAt5: sumNdcg / Math.max(1, totalNonAbstain),
		abstentionPrecision: byCategory.abstention.abstentionRate ?? 0,
		knowledgeUpdateFreshness:
			((byCategory.knowledge_update.recallAt1 + (byCategory.knowledge_update.zeroStaleTop3 ?? 0)) / 2),
		multiHopBothAt5: byCategory.multi_hop.bothHopsAt5 ?? 0,
		byCategory,
		failedSampleQueries: failures.slice(0, 12),
	};
}

// ── Scalability Benchmark (N = 100, 500, 2000 memories) ─────────────────────────────────────────

function deterministicSyntheticVector(seed: number, dim = 384): Float32Array {
	const v = new Float32Array(dim);
	let x = (seed * 2654435761) >>> 0;
	let norm = 0;
	for (let i = 0; i < dim; i++) {
		x ^= x << 13;
		x ^= x >>> 17;
		x ^= x << 5;
		const f = ((x >>> 0) / 0xffffffff) * 2 - 1;
		v[i] = f;
		norm += f * f;
	}
	const inv = 1 / Math.sqrt(norm || 1);
	for (let i = 0; i < dim; i++) v[i] = v[i]! * inv;
	return v;
}

const SEM_TARGET_CODES = [
	"ObsidianVault",
	"CobaltVault",
	"CrimsonVault",
	"EmeraldVault",
	"SaffronVault",
	"IndigoVault",
	"TopazVault",
	"QuartzVault",
	"BerylVault",
	"GarnetVault",
] as const;

const SEM_QUERY_CODES = [
	"OrchidLookup",
	"WillowLookup",
	"CedarLookup",
	"MapleLookup",
	"BirchLookup",
	"AlderLookup",
	"RowanLookup",
	"HazelLookup",
	"LindenLookup",
	"SpruceLookup",
] as const;

const LEX_NEEDLE_NAMES = [
	"AlphaWidget",
	"BravoWidget",
	"CharlieWidget",
	"DeltaWidget",
	"EchoWidget",
	"FoxtrotWidget",
	"GolfWidget",
	"HotelWidget",
	"IndiaWidget",
	"JulietWidget",
] as const;

function makeFastSyntheticEmbedder(): EmbeddingProvider {
	return {
		model: "synthetic-384",
		async embed(texts) {
			return texts.map((t) => {
				// Semantic needle pairing: SEM_QUERY_CODES[k] maps to the exact same basis direction as
				// SEM_TARGET_CODES[k] with ZERO lexical token overlap to test vector recall past 500 memories!
				for (let k = 0; k < SEM_TARGET_CODES.length; k++) {
					if (t.includes(SEM_TARGET_CODES[k]!) || t.includes(SEM_QUERY_CODES[k]!)) {
						return deterministicSyntheticVector(99_000 + k, 384);
					}
				}
				let h = 2166136261;
				for (let i = 0; i < t.length; i++) {
					h ^= t.charCodeAt(i);
					h = Math.imul(h, 16777619);
				}
				return deterministicSyntheticVector(h >>> 0, 384);
			});
		},
	};
}

async function runHeliosScaleBench(counts: readonly number[]): Promise<ScaleBenchmarkRow[]> {
	const rows: ScaleBenchmarkRow[] = [];
	const synthEmbedder = makeFastSyntheticEmbedder();

	for (const mode of ["lexical", "hybrid"] as const) {
		for (const n of counts) {
			const appDb = AppDatabase.open({ dbPath: ":memory:" });
			let now = BENCHMARK_NOW_MS - n * 60_000;
			const mem = new SqliteMemoryService(appDb, {
				now: () => now,
				...(mode === "hybrid" ? { embeddingProvider: synthEmbedder } : {}),
			});

			// Plant 10 lexical needles + 10 pure-semantic needles at OLDEST timestamps
			// (exposes LIMIT-500 vector blind spot when N > 500!)
			const needles: Array<{ id: string; query: string }> = [];
			const semNeedles: Array<{ id: string; query: string }> = [];
			const tIngest0 = performance.now();
			for (let i = 0; i < n; i++) {
				now += 60_000;
				const isLexNeedle = i < 10;
				const isSemNeedle = i >= 10 && i < 20;
				const lexTag = LEX_NEEDLE_NAMES[i] ?? `Widget${i}`;
				const semTag = SEM_TARGET_CODES[i - 10] ?? `Vault${i}`;
				const content = isLexNeedle
					? `Architecture component ${lexTag} runs on dedicated cluster ${lexTag}Node with strict p99 SLA.`
					: isSemNeedle
						? `${semTag} archival blueprint stored during initial onboarding phase.`
						: `Synthetic log fact #${i}: routine telemetry check for service-${i % 40} in zone-${i % 7} batch ${Math.floor(i / 50)}.`;
				const saved = await mem.applyMemoryMutation(TENANT, USER, {
					op: "ADD",
					key: isLexNeedle ? `needle.${lexTag.toLowerCase()}` : isSemNeedle ? `archive.${semTag.toLowerCase()}` : `synthetic.item_${i}`,
					kind: "fact",
					content,
					confidence: 0.88,
				});
				if (isLexNeedle && saved) {
					needles.push({ id: saved.id, query: `${lexTag} ${lexTag}Node cluster` });
				}
				if (isSemNeedle && saved) {
					const qCode = SEM_QUERY_CODES[i - 10]!;
					semNeedles.push({ id: saved.id, query: `${qCode} retrieval` });
				}
			}
			if (mode === "hybrid") {
				// Backfill in batches (legacy backfillEmbeddings capped at 1000 per call, so loop until done)
				let totalBackfilled = 0;
				while (totalBackfilled < n) {
					const written = await mem.backfillEmbeddings(TENANT, USER, { limit: Math.max(n, 2500) });
					if (written === 0) break;
					totalBackfilled += written;
				}
			}
			const ingestTotalMs = performance.now() - tIngest0;

			// Warm 2 queries, then measure 30 queries
			await mem.search(TENANT, USER, needles[0]!.query, { limit: 5 });
			const latencies: number[] = [];
			let hits = 0;
			for (let r = 0; r < 3; r++) {
				for (const needle of needles) {
					const t0 = performance.now();
					const res = await mem.search(TENANT, USER, needle.query, { limit: 5 });
					latencies.push(performance.now() - t0);
					if (r === 0 && res.some((x) => x.memory.id === needle.id)) hits += 1;
				}
			}

			let semHits = 0;
			if (mode === "hybrid") {
				for (const sn of semNeedles) {
					const t0 = performance.now();
					const res = await mem.search(TENANT, USER, sn.query, { limit: 5 });
					latencies.push(performance.now() - t0);
					if (res.some((x) => x.memory.id === sn.id)) semHits += 1;
				}
			}
			appDb.close();

			rows.push({
				engine: `helios-${mode}`,
				memoryCount: n,
				ingestTotalMs,
				ingestPerOpMs: ingestTotalMs / n,
				searchP50Ms: percentile(latencies, 50),
				searchP95Ms: percentile(latencies, 95),
				searchP99Ms: percentile(latencies, 99),
				recallAt5: hits / needles.length,
				...(mode === "hybrid" ? { semanticOldestRecallAt5: semHits / semNeedles.length } : {}),
			});
		}
	}
	return rows;
}

// ── CLI Main ────────────────────────────────────────────────────────────────────────────────────

function pct(v: number): string {
	return `${(v * 100).toFixed(1)}%`;
}

async function main(): Promise<void> {
	const args = process.argv.slice(2);
	const extIdx = args.indexOf("--external");
	const externalDir = extIdx >= 0 ? args[extIdx + 1] : undefined;
	const endpointIdx = args.indexOf("--endpoint");
	const endpoint = (endpointIdx >= 0 ? args[endpointIdx + 1] : undefined) ?? "http://127.0.0.1:18765/v1";
	const jsonOutIdx = args.indexOf("--json");
	const jsonOutPath = jsonOutIdx >= 0 ? args[jsonOutIdx + 1] : undefined;

	const gateIdx = args.indexOf("--gate");
	const gatePath = gateIdx >= 0 ? args[gateIdx + 1] : undefined;
	const inProcIdx = args.indexOf("--in-process");
	const inProcessProfile = inProcIdx >= 0 ? args[inProcIdx + 1] : undefined;
	const lexicalOnly = args.includes("--lexical-only");

	// Embedder resolution: in-process router (--in-process <profile>) > Helios embedding server > legacy MiniLM server.
	let embedder: EmbeddingProvider | undefined;
	let reranker: Reranker | undefined;
	let closeEmbedder: (() => Promise<void>) | undefined;
	if (lexicalOnly) {
		console.error("[bench] --lexical-only: skipping embedding providers");
	} else if (inProcessProfile) {
		const service = buildEmbeddingService(loadEmbedServerConfig({ ...process.env, EMBED_PROFILE: inProcessProfile }));
		embedder = new RouterEmbeddingProvider(service.router);
		if (args.includes("--rerank") && service.reranker) {
			await service.reranker.selfTest();
			reranker = service.reranker;
		}
		closeEmbedder = () => service.close();
		console.error(`[bench] In-process embedding router (${inProcessProfile}): ${embedder.model}`);
	} else {
		try {
			const helios = await HttpEmbeddingProvider.connect({ baseUrl: endpoint, timeoutMs: 120_000 });
			embedder = helios;
			console.error(`[bench] Connected to Helios embedding server at ${endpoint}: ${helios.model}`);
			if (args.includes("--rerank")) {
				reranker = await HttpReranker.connect({ baseUrl: endpoint, timeoutMs: 120_000 });
				console.error(`[bench] Reranker: ${reranker.model} (answer threshold ${reranker.answerThreshold ?? "default"})`);
			}
		} catch {
			try {
				const health = await fetch(endpoint.replace(/\/v1$/, "/health"));
				if (health.ok) {
					embedder = new CachedHttpEmbeddingProvider({ endpoint, model: "sentence-transformers/all-MiniLM-L6-v2" });
					console.error(`[bench] Connected to legacy embedding server at ${endpoint}`);
				}
			} catch {
				console.error(`[bench] Embedding server not reachable at ${endpoint}; running lexical/synthetic fallback`);
			}
		}
	}
	const embedderLabel = embedder ? embedder.model.replace(/^.*\//, "") : "";

	const engines: Array<{ engine: BenchMemoryEngine; ingestMs: number; warmMs: number }> = [];

	// 1. Helios lexical-only (zero dependencies)
	engines.push(await buildHeliosEngine("helios (lexical)", undefined));

	// 2. Helios hybrid (real all-MiniLM-L6-v2)
	if (embedder) {
		engines.push(await buildHeliosEngine(`helios (hybrid ${embedderLabel})`, embedder));
		if (reranker) {
			engines.push(await buildHeliosEngine(`helios (hybrid+rerank ${reranker.model.replace(/^.*\//, "").replace(/-ONNX$/, "")})`, embedder, reranker));
		}

		// 3. Graphiti-Ref (bi-temporal knowledge graph)
		const tG0 = performance.now();
		const graphiti = await GraphitiRefEngine.build(BENCHMARK_MEMORIES, embedder);
		engines.push({ engine: graphiti, ingestMs: performance.now() - tG0, warmMs: 0 });

		// 4. HippoRAG-Ref (entity-graph + Personalized PageRank)
		const tH0 = performance.now();
		const hippo = await HippoRagRefEngine.build(BENCHMARK_MEMORIES, embedder);
		engines.push({ engine: hippo, ingestMs: performance.now() - tH0, warmMs: 0 });
	}

	// 5. Real external competitors (@cognee/cognee-ts native Rust + mem0ai/oss)
	if (externalDir && embedder) {
		const externals = await buildExternalCompetitors(externalDir, endpoint);
		engines.push(...externals);
	}

	const results: EngineBenchmarkResult[] = [];
	for (const entry of engines) {
		console.error(`[bench] Evaluating ${entry.engine.name}...`);
		const res = await evaluateEngine(entry);
		results.push(res);
		await entry.engine.close?.();
	}

	console.error("[bench] Running scalability latency benchmark (N=100, 500, 2000)...");
	const scaleRows = await runHeliosScaleBench([100, 500, 2000]);

	// Print clean Markdown summary table
	console.log(`\n=== QUALITY BENCHMARK [${DATASET_NAME}] (${BENCHMARK_QUERIES.length} queries across 8 categories) ===\n`);
	console.log(
		"| Engine | Overall R@1 | Overall R@5 | MRR | nDCG@5 | 1-Hop R@5 | Semantic R@5 | MultiHop Target@5 | MultiHop Both@5 | Update Fresh@1 | ZeroStale@3 | Temporal R@5 | HistAudit R@5 | Abstain | RecentEp R@5 | p50 (ms) | p95 (ms) |",
	);
	console.log(
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	);
	for (const r of results) {
		const c = r.byCategory;
		const cp = (cat: QueryCategory, v: number): string => (c[cat].count === 0 ? "n/a" : pct(v));
		console.log(
			`| ${r.engine} | ${pct(r.overallRecallAt1)} | ${pct(r.overallRecallAt5)} | ${r.overallMrr.toFixed(3)} | ${r.overallNdcgAt5.toFixed(3)} | ${cp("single_hop", c.single_hop.recallAt5)} | ${cp("semantic_gap", c.semantic_gap.recallAt5)} | ${cp("multi_hop", c.multi_hop.recallAt5)} | ${cp("multi_hop", c.multi_hop.bothHopsAt5 ?? 0)} | ${cp("knowledge_update", c.knowledge_update.recallAt1)} | ${cp("knowledge_update", c.knowledge_update.zeroStaleTop3 ?? 0)} | ${cp("temporal_window", c.temporal_window.recallAt5)} | ${cp("historical_audit", c.historical_audit.recallAt5)} | ${cp("abstention", c.abstention.abstentionRate ?? 0)} | ${cp("session_recent", c.session_recent.recallAt5)} | ${r.queryLatencyP50Ms.toFixed(2)} | ${r.queryLatencyP95Ms.toFixed(2)} |`,
		);
	}

	console.log("\n=== SCALABILITY BENCHMARK (Oldest-Needle Lexical + Semantic Recall & Latency) ===\n");
	console.log("| Engine | N Memories | Ingest (ms) | ms/write | Search p50 (ms) | Search p95 (ms) | Search p99 (ms) | Lex-Needle R@5 | Sem-Oldest R@5 |");
	console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
	for (const s of scaleRows) {
		console.log(
			`| ${s.engine} | ${s.memoryCount} | ${s.ingestTotalMs.toFixed(1)} | ${s.ingestPerOpMs.toFixed(3)} | ${s.searchP50Ms.toFixed(3)} | ${s.searchP95Ms.toFixed(3)} | ${s.searchP99Ms.toFixed(3)} | ${pct(s.recallAt5)} | ${s.semanticOldestRecallAt5 !== undefined ? pct(s.semanticOldestRecallAt5) : "n/a"} |`,
		);
	}

	await closeEmbedder?.();

	if (jsonOutPath) {
		fs.mkdirSync(path.dirname(path.resolve(jsonOutPath)), { recursive: true });
		fs.writeFileSync(jsonOutPath, JSON.stringify({ timestamp: new Date().toISOString(), results, scaleRows }, null, 2));
		console.error(`[bench] Saved JSON report to ${jsonOutPath}`);
	}

	if (gatePath) {
		const failures = checkGate(gatePath, results);
		if (failures.length > 0) {
			console.error(`\n[bench] GATE FAILED (${failures.length}):\n  - ${failures.join("\n  - ")}`);
			process.exitCode = 1;
		} else {
			console.error("\n[bench] Gate passed.");
		}
	}
}

/**
 * Regression gate. The thresholds file maps an engine-name prefix ("helios (lexical)", "helios (hybrid") to minimum
 * metric values (`p95Ms` is an upper bound). Engines that did not run (e.g. no embedding model) are skipped.
 */
function checkGate(file: string, results: readonly EngineBenchmarkResult[]): string[] {
	const spec = JSON.parse(fs.readFileSync(file, "utf8")) as Record<string, Record<string, number> | string>;
	const failures: string[] = [];
	const metric = (r: EngineBenchmarkResult, name: string): number | undefined => {
		const c = r.byCategory;
		const table: Record<string, number | undefined> = {
			overallRecallAt1: r.overallRecallAt1,
			overallRecallAt5: r.overallRecallAt5,
			overallMrr: r.overallMrr,
			overallNdcgAt5: r.overallNdcgAt5,
			singleHopR5: c.single_hop.recallAt5,
			semanticR5: c.semantic_gap.recallAt5,
			multiHopR1: c.multi_hop.recallAt1,
			multiHopTargetR5: c.multi_hop.recallAt5,
			multiHopBothAt5: c.multi_hop.bothHopsAt5,
			updateFreshAt1: c.knowledge_update.recallAt1,
			zeroStaleTop3: c.knowledge_update.zeroStaleTop3,
			temporalR5: c.temporal_window.recallAt5,
			historicalR5: c.historical_audit.recallAt5,
			abstention: c.abstention.abstentionRate,
			sessionRecentR5: c.session_recent.recallAt5,
			p95Ms: r.queryLatencyP95Ms,
		};
		return table[name];
	};
	for (const [prefix, thresholds] of Object.entries(spec)) {
		if (prefix.startsWith("$") || typeof thresholds !== "object") continue;
		const matching = results.filter((r) => r.engine.startsWith(prefix));
		if (matching.length === 0) {
			console.error(`[bench] gate: no engine matching "${prefix}" ran; skipped`);
			continue;
		}
		for (const r of matching) {
			for (const [name, bound] of Object.entries(thresholds)) {
				const value = metric(r, name);
				if (value === undefined) {
					failures.push(`${r.engine}: unknown metric ${name}`);
					continue;
				}
				const upper = name.endsWith("Ms");
				const ok = upper ? value <= bound : value + 1e-9 >= bound;
				if (!ok) failures.push(`${r.engine}: ${name}=${value.toFixed(3)} ${upper ? ">" : "<"} ${bound}`);
			}
		}
	}
	return failures;
}


main().catch((err) => {
	console.error(err);
	process.exit(1);
});
