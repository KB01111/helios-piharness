/**
 * Runs Helios memory retrieval on public long-term-memory benchmarks (LoCoMo, LongMemEval_S).
 *
 *   tsx benchmarks/run-public-bench.ts --dataset locomo|longmemeval [--limit N] [--conversations N]
 *       [--endpoint http://127.0.0.1:18765/v1 | --in-process <profile> | --lexical-only] [--rerank] [--json out.json]
 *
 * Metrics (top-10 retrieval, per category):
 *   R@1 / R@5 / R@10   any gold item retrieved within k (LoCoMo: evidence turns, LongMemEval: answer sessions)
 *   All@10             every gold item within the top 10 (multi-evidence / multi-session questions)
 *   MRR@10
 *   Abstain            unanswerable questions whose top score stays under the engine's abstention threshold
 */

import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import type { EmbeddingProvider, Reranker } from "../src/contracts/app-capabilities.js";
import { loadEmbedServerConfig } from "../src/embeddings/config.js";
import { buildEmbeddingService } from "../src/embeddings/factory.js";
import { HttpEmbeddingProvider, HttpReranker, RouterEmbeddingProvider } from "../src/embeddings/provider.js";
import { AppDatabase } from "../src/memory/db.js";
import { SqliteMemoryService } from "../src/memory/sqlite-memory.js";
import { loadLocomo, loadLongMemEval, type PublicDataset } from "./public-datasets.js";

const TENANT = "public-bench";

function arg(name: string): string | undefined {
	const i = process.argv.indexOf(name);
	return i >= 0 ? process.argv[i + 1] : undefined;
}

interface Bucket {
	n: number;
	r1: number;
	r5: number;
	r10: number;
	all10: number;
	mrr: number;
	abstained: number;
}

interface EngineSpec {
	readonly name: string;
	readonly embedder: EmbeddingProvider | undefined;
	readonly reranker: Reranker | undefined;
	readonly abstentionThreshold: number;
}

async function evaluate(dataset: PublicDataset, spec: EngineSpec) {
	const buckets = new Map<string, Bucket>();
	const latencies: number[] = [];
	let ingestMs = 0;
	const bucket = (c: string): Bucket => {
		let b = buckets.get(c);
		if (!b) buckets.set(c, (b = { n: 0, r1: 0, r5: 0, r10: 0, all10: 0, mrr: 0, abstained: 0 }));
		return b;
	};

	let done = 0;
	for (const scope of dataset.scopes) {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		let now = 0;
		const memory = new SqliteMemoryService(appDb, {
			now: () => now,
			...(spec.embedder ? { embeddingProvider: spec.embedder } : {}),
			...(spec.reranker ? { reranker: spec.reranker } : {}),
			...(arg("--rerank-candidates") ? { rerankCandidates: Number(arg("--rerank-candidates")) } : {}),
		});
		const t0 = performance.now();
		const dbToGold = new Map<string, { id: string; session: string }>();
		for (const m of scope.memories) {
			now = m.timestampMs;
			const saved = await memory.applyMemoryMutation(TENANT, scope.scopeId, {
				op: "ADD",
				key: m.key,
				kind: "episodic",
				content: m.content,
				confidence: 0.8,
			});
			if (saved) dbToGold.set(saved.id, { id: m.id, session: m.sessionId });
		}
		if (spec.embedder) {
			while ((await memory.backfillEmbeddings(TENANT, scope.scopeId, { limit: 10_000 })) > 0) {
				// loop until every memory has a vector
			}
		}
		ingestMs += performance.now() - t0;

		for (const q of scope.queries) {
			now = q.timestampMs;
			const s0 = performance.now();
			const hits = await memory.search(TENANT, scope.scopeId, q.query, { limit: 10 });
			latencies.push(performance.now() - s0);
			const b = bucket(q.category);
			b.n += 1;
			if (q.expectAbstain) {
				if (hits.length === 0 || (hits[0]?.score ?? 0) < spec.abstentionThreshold) b.abstained += 1;
				continue;
			}
			const mapped = hits.map((h) => dbToGold.get(h.memory.id));
			const sessionLevel = q.goldSessionIds.length > 0;
			const gold = new Set(sessionLevel ? q.goldSessionIds : q.goldMemoryIds);
			const keys = mapped.map((m) => (m ? (sessionLevel ? m.session : m.id) : ""));
			const first = keys.findIndex((k) => gold.has(k));
			if (first === 0) b.r1 += 1;
			if (first >= 0 && first < 5) b.r5 += 1;
			if (first >= 0) {
				b.r10 += 1;
				b.mrr += 1 / (first + 1);
			}
			const top10 = new Set(keys);
			if ([...gold].every((g) => top10.has(g))) b.all10 += 1;
		}
		appDb.close();
		done += 1;
		if (done % 10 === 0 || done === dataset.scopes.length) {
			console.error(`[bench] ${spec.name}: ${done}/${dataset.scopes.length} scopes`);
		}
	}
	latencies.sort((a, b) => a - b);
	return {
		engine: spec.name,
		ingestMs,
		p50Ms: latencies[Math.floor(latencies.length * 0.5)] ?? 0,
		p95Ms: latencies[Math.floor(latencies.length * 0.95)] ?? 0,
		buckets: Object.fromEntries(buckets),
	};
}

function pct(n: number, d: number): string {
	return d > 0 ? `${((100 * n) / d).toFixed(1)}%` : "n/a";
}

async function main(): Promise<void> {
	const name = arg("--dataset") ?? "locomo";
	const limit = Number(arg("--limit") ?? 40);
	const dataset =
		name === "locomo"
			? await loadLocomo({ ...(arg("--conversations") ? { limitConversations: Number(arg("--conversations")) } : {}) })
			: name === "longmemeval"
				? await loadLongMemEval({ limit })
				: (() => {
						throw new Error("--dataset must be locomo or longmemeval");
					})();

	let embedder: EmbeddingProvider | undefined;
	let reranker: Reranker | undefined;
	let close: (() => Promise<void>) | undefined;
	const endpoint = arg("--endpoint") ?? "http://127.0.0.1:18765/v1";
	if (!process.argv.includes("--lexical-only")) {
		const profile = arg("--in-process");
		if (profile) {
			const service = buildEmbeddingService(loadEmbedServerConfig({ ...process.env, EMBED_PROFILE: profile, EMBED_CACHE_SIZE: "200000" }));
			close = () => service.close();
			if (service.supervisor && !(await service.supervisor.ensureRunning())) {
				console.error("[bench] FastFlowLM did not start; NPU requests will fall back to the next backend or fail");
			}
			const quality = await service.router.selfTest({ quarantineOnError: true });
			console.error("[bench] embedding self-test:", JSON.stringify(quality));
			if (quality.some((entry) => entry.result?.passed)) {
				embedder = new RouterEmbeddingProvider(service.router);
				if (process.argv.includes("--rerank") && service.reranker) {
					console.error("[bench] rerank self-test:", JSON.stringify(await service.reranker.selfTest()));
					reranker = service.reranker;
				}
			} else {
				console.error("[bench] No embedding backend passed the startup quality gate; running lexical only");
			}
		} else {
			try {
				embedder = await HttpEmbeddingProvider.connect({ baseUrl: endpoint, timeoutMs: 300_000 });
			} catch (error) {
				console.error(`[bench] no embedding server at ${endpoint}: lexical only (${error instanceof Error ? error.message : String(error)})`);
			}
		}
		if (embedder && !reranker && process.argv.includes("--rerank")) {
			reranker = await HttpReranker.connect({ baseUrl: endpoint, timeoutMs: 300_000 });
		}
	}

	const engines: EngineSpec[] = [{ name: "helios (lexical)", embedder: undefined, reranker: undefined, abstentionThreshold: 0.15 }];
	if (embedder) {
		engines.push({ name: `helios (hybrid ${embedder.model.replace(/^.*\//, "")})`, embedder, reranker: undefined, abstentionThreshold: 0.3 });
		if (reranker) {
			engines.push({ name: `helios (hybrid + rerank ${reranker.model.replace(/^.*\//, "")})`, embedder, reranker, abstentionThreshold: 0.3 });
		}
	}

	const results = [];
	for (const spec of engines) results.push(await evaluate(dataset, spec));
	await close?.();

	const categories = [...new Set(dataset.scopes.flatMap((s) => s.queries.map((q) => q.category)))].sort();
	const nQueries = dataset.scopes.reduce((s, x) => s + x.queries.length, 0);
	const nMemories = dataset.scopes.reduce((s, x) => s + x.memories.length, 0);
	console.log(`\n=== ${dataset.name}: ${nQueries} queries over ${dataset.scopes.length} scopes / ${nMemories} memories ===`);
	for (const note of dataset.notes) console.log(`  - ${note}`);
	for (const r of results) {
		console.log(`\n${r.engine}   (ingest ${(r.ingestMs / 1000).toFixed(1)}s, search p50 ${r.p50Ms.toFixed(1)}ms / p95 ${r.p95Ms.toFixed(1)}ms)`);
		console.log("| Category | n | R@1 | R@5 | R@10 | All@10 | MRR@10 | Abstain |");
		console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
		const tot: Bucket = { n: 0, r1: 0, r5: 0, r10: 0, all10: 0, mrr: 0, abstained: 0 };
		for (const c of categories) {
			const b = r.buckets[c] as Bucket | undefined;
			if (!b) continue;
			const isAbs = c === "abstention";
			console.log(
				`| ${c} | ${b.n} | ${isAbs ? "" : pct(b.r1, b.n)} | ${isAbs ? "" : pct(b.r5, b.n)} | ${isAbs ? "" : pct(b.r10, b.n)} | ${isAbs ? "" : pct(b.all10, b.n)} | ${isAbs ? "" : (b.mrr / b.n).toFixed(3)} | ${isAbs ? pct(b.abstained, b.n) : ""} |`,
			);
			if (!isAbs) {
				tot.n += b.n;
				tot.r1 += b.r1;
				tot.r5 += b.r5;
				tot.r10 += b.r10;
				tot.all10 += b.all10;
				tot.mrr += b.mrr;
			}
		}
		console.log(`| **overall (answerable)** | ${tot.n} | ${pct(tot.r1, tot.n)} | ${pct(tot.r5, tot.n)} | ${pct(tot.r10, tot.n)} | ${pct(tot.all10, tot.n)} | ${(tot.mrr / Math.max(1, tot.n)).toFixed(3)} | |`);
	}

	const jsonOut = arg("--json");
	if (jsonOut) {
		fs.mkdirSync(path.dirname(path.resolve(jsonOut)), { recursive: true });
		fs.writeFileSync(jsonOut, JSON.stringify({ dataset: dataset.name, notes: dataset.notes, results }, null, 2));
	}
}

main().catch((error: unknown) => {
	console.error(error);
	process.exit(1);
});
