import { describe, expect, it, vi } from "vitest";
import { loadConfig } from "../src/config.js";
import type { Reranker } from "../src/contracts/app-capabilities.js";
import { loadEmbedServerConfig } from "../src/embeddings/config.js";
import { buildEmbeddingService } from "../src/embeddings/factory.js";
import { HttpReranker } from "../src/embeddings/provider.js";
import {
	HttpRerankBackend,
	OnnxRerankBackend,
	RerankService,
	runRerankSelfTest,
	type RerankBackend,
	type TransformersRerankModule,
} from "../src/embeddings/rerank.js";
import { EmbeddingRouter } from "../src/embeddings/router.js";
import { SELF_TEST_PAIRS } from "../src/embeddings/self-test.js";
import { createEmbedHttpServer } from "../src/embeddings/server.js";
import type { BackendEmbedOptions, EmbeddingBackend } from "../src/embeddings/types.js";
import { AppDatabase } from "../src/memory/db.js";
import { SqliteMemoryService } from "../src/memory/sqlite-memory.js";

class TinyEmbedBackend implements EmbeddingBackend {
	readonly id = "a";
	readonly kind = "openai" as const;
	readonly model = "fake";
	readonly space = "test/space";
	readonly nativeDimensions = 4;
	readonly wantsPrompts = false;
	readonly maxBatchSize = 16;
	async embed(texts: readonly string[], _options: BackendEmbedOptions): Promise<Float32Array[]> {
		return texts.map(() => Float32Array.from([1, 0, 0, 0]));
	}
	async health() {
		return { ok: true };
	}
}

/** Fake cross-encoder scoring documents with `scoreFn`. */
class FakeRerankBackend implements RerankBackend {
	readonly id = "fake-rr";
	readonly kind = "onnx" as const;
	readonly model = "fake-reranker";
	readonly maxBatchSize = 4;
	calls = 0;
	constructor(private readonly scoreFn: (q: string, d: string) => number) {}
	async score(query: string, documents: readonly string[]): Promise<number[]> {
		this.calls += 1;
		return documents.map((d) => this.scoreFn(query, d));
	}
	async health() {
		return { ok: true };
	}
}

function honoFetch(app: { request: (input: string, init?: RequestInit) => Response | Promise<Response> }): typeof fetch {
	return (async (input: string | URL | Request, init?: RequestInit) => {
		const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
		const { pathname, search } = new URL(url);
		return app.request(`${pathname}${search}`, init);
	}) as typeof fetch;
}

const selfTestAware = (q: string, d: string): number => {
	const pair = SELF_TEST_PAIRS.find(([pq]) => pq === q);
	return pair && pair[1] === d ? 0.9 : 0.01;
};

describe("RerankService & self-test", () => {
	it("passes a model whose confident scores are trustworthy and measures the non-answer level", async () => {
		const result = await runRerankSelfTest(async (q, docs) => docs.map((d) => selfTestAware(q, d)));
		expect(result.passed).toBe(true);
		expect(result.answerThreshold).toBeCloseTo(0.01, 3);
		expect(result.positiveMedian).toBeCloseTo(0.9, 3);
	});

	it("quarantines a reranker that scores near-misses as high as answers", async () => {
		const svc = new RerankService({ backend: new FakeRerankBackend(() => 0.85) });
		const result = await svc.selfTest();
		expect(result.passed).toBe(false);
		await expect(svc.rerank("q", ["d"])).rejects.toMatchObject({ code: "UNAVAILABLE" });
		expect((await svc.health()).ok).toBe(false);
	});

	it("caches scores, batches to the backend limit, validates input, and keeps a pinned threshold", async () => {
		const backend = new FakeRerankBackend((q, d) => (d.includes(q) ? 0.9 : 0.05));
		const svc = new RerankService({ backend, maxDocuments: 10, answerThreshold: 0.2 });
		const docs = ["alpha one", "beta", "gamma", "delta", "epsilon alpha"];
		expect(await svc.rerank("alpha", docs)).toEqual([0.9, 0.05, 0.05, 0.05, 0.9]);
		expect(backend.calls).toBe(2); // 5 docs, batch size 4
		await svc.rerank("alpha", docs);
		expect(backend.calls).toBe(2); // all cached
		await expect(svc.rerank(" ", ["x"])).rejects.toMatchObject({ code: "BAD_REQUEST" });
		await expect(svc.rerank("q", new Array(11).fill("x"))).rejects.toMatchObject({ code: "BAD_REQUEST" });
		await svc.selfTest();
		expect(svc.answerThreshold).toBe(0.2);
	});

	it("ONNX backend: sigmoid for single-logit heads, softmax for two-class heads, input order restored", async () => {
		const make = (width: number): TransformersRerankModule => ({
			env: {},
			AutoTokenizer: {
				from_pretrained: async () => (_texts: string[], opts: { text_pair: string[] }) => ({ pairs: opts.text_pair }),
			},
			AutoModelForSequenceClassification: {
				from_pretrained: async () =>
					(async (inputs: { pairs?: string[] }) => {
						const pairs = inputs.pairs ?? [];
						const data = pairs.flatMap((p) => (width === 1 ? [p.length > 5 ? 4 : -4] : [0, p.length > 5 ? 4 : -4]));
						return { logits: { data, dims: [pairs.length, width] } };
					}) as never,
			},
		});
		for (const width of [1, 2]) {
			const backend = new OnnxRerankBackend({ loadTransformers: async () => make(width) });
			const scores = await backend.score("q", ["long document", "tiny", "another long one"]);
			expect(scores[0]).toBeGreaterThan(0.9);
			expect(scores[1]).toBeLessThan(0.1);
			expect(scores[2]).toBeGreaterThan(0.9);
		}
	});

	it("skips canceled ONNX work queued, loading, or between inference batches", async () => {
		const loading = new AbortController();
		const queued = new AbortController();
		const batching = new AbortController();
		const loadStarted = Promise.withResolvers<void>();
		const loadModel = Promise.withResolvers<TransformersRerankModule>();
		const classifier = vi.fn(async () => {
			batching.abort();
			return { logits: { data: new Array<number>(8).fill(0), dims: [8, 1] } };
		});
		const backend = new OnnxRerankBackend({
			loadTransformers: () => {
				loadStarted.resolve();
				return loadModel.promise;
			},
		});
		const first = expect(backend.score("q", ["loading"], loading.signal)).rejects.toMatchObject({ code: "TIMEOUT" });
		await loadStarted.promise;
		const second = expect(backend.score("q", ["queued"], queued.signal)).rejects.toMatchObject({ code: "TIMEOUT" });
		loading.abort();
		queued.abort();
		loadModel.resolve({
			env: {},
			AutoTokenizer: { from_pretrained: async () => () => ({}) },
			AutoModelForSequenceClassification: { from_pretrained: async () => classifier },
		});
		await Promise.all([first, second]);
		expect(classifier).not.toHaveBeenCalled();

		await expect(backend.score("q", Array<string>(9).fill("doc"), batching.signal)).rejects.toMatchObject({ code: "TIMEOUT" });
		expect(classifier).toHaveBeenCalledTimes(1);
	});

	it("HTTP backend parses Cohere/Jina `results` and Voyage `data` responses and clamps scores", async () => {
		for (const field of ["results", "data"]) {
			const fetchImpl = (async (_url: string | URL | Request, init?: RequestInit) => {
				const body = JSON.parse(String(init?.body)) as { documents: string[] };
				return Response.json({ [field]: body.documents.map((_, i) => ({ index: i, relevance_score: i === 0 ? 1.7 : 0.2 })) });
			}) as typeof fetch;
			const backend = new HttpRerankBackend({ baseUrl: "https://api.example/v1", model: "rr", apiKey: "k", fetchImpl });
			expect(await backend.score("q", ["a", "b"])).toEqual([1, 0.2]);
		}
		expect(() => new HttpRerankBackend({ baseUrl: "", model: "x" })).toThrow(/EMBED_RERANK_URL/);
	});
});

describe("rerank HTTP API & client", () => {
	function serverWithReranker() {
		const router = new EmbeddingRouter({ backends: [new TinyEmbedBackend()] });
		const reranker = new RerankService({ backend: new FakeRerankBackend((q, d) => (d.includes(q) ? 0.8 : 0.02)), answerThreshold: 0.05 });
		return createEmbedHttpServer(router, { reranker });
	}

	it("serves Cohere/Jina-style /v1/rerank sorted by relevance, with top_n and {text} documents", async () => {
		const app = serverWithReranker();
		const res = await app.request("/v1/rerank", {
			method: "POST",
			body: JSON.stringify({ query: "cat", documents: ["dog", { text: "a cat" }, "cat food"], top_n: 2, return_documents: true }),
		});
		expect(res.status).toBe(200);
		const body = (await res.json()) as { model: string; results: Array<{ index: number; relevance_score: number; document?: { text: string } }> };
		expect(body.model).toBe("fake-reranker");
		expect(body.results.map((r) => r.index)).toEqual([1, 2]);
		expect(body.results[0]?.document?.text).toBe("a cat");
		const models = (await (await app.request("/v1/models")).json()) as { data: Array<{ type?: string; answer_threshold?: number }> };
		expect(models.data.find((m) => m.type === "rerank")).toMatchObject({ answer_threshold: 0.05 });
		for (const bad of [
			{ query: "", documents: ["x"] },
			{ query: "q", documents: [] },
			{ query: "q", documents: [1] },
			{ query: "q", documents: ["x"], top_n: 0 },
		]) {
			expect((await app.request("/v1/rerank", { method: "POST", body: JSON.stringify(bad) })).status).toBe(400);
		}
	});

	it("answers 404 on /v1/rerank when the server has no reranker", async () => {
		const app = createEmbedHttpServer(new EmbeddingRouter({ backends: [new TinyEmbedBackend()] }));
		expect((await app.request("/v1/rerank", { method: "POST", body: JSON.stringify({ query: "q", documents: ["d"] }) })).status).toBe(404);
	});

	it("HttpReranker discovers the Helios reranker and its threshold; plain endpoints need a model", async () => {
		const rr = await HttpReranker.connect({ baseUrl: "http://embed.local/v1", fetchImpl: honoFetch(serverWithReranker()) });
		expect(rr.helios).toBe(true);
		expect(rr.model).toBe("fake-reranker");
		expect(rr.answerThreshold).toBe(0.05);
		expect(await rr.rerank("cat", ["dog", "cat"])).toEqual([0.02, 0.8]);

		const plain = honoFetch(createEmbedHttpServer(new EmbeddingRouter({ backends: [new TinyEmbedBackend()] })));
		await expect(HttpReranker.connect({ baseUrl: "http://embed.local/v1", fetchImpl: plain })).rejects.toMatchObject({ code: "MISCONFIGURED" });
		const cloud = await HttpReranker.connect({ baseUrl: "http://embed.local/v1", fetchImpl: plain, model: "rerank-v3.5" });
		expect(cloud.helios).toBe(false);
	});

	it("HttpReranker accepts Voyage data, prefers results, and rejects missing scores", async () => {
		const data = [{ index: 1, relevance_score: 0.8 }, { index: 0, relevance_score: 0.2 }];
		let body: object = { data };
		const fetchImpl: typeof fetch = async (input) => String(input).endsWith("/models") ? Response.json({ data: [] }) : Response.json(body);
		const rr = await HttpReranker.connect({ baseUrl: "https://example.test/v1", model: "voyage", fetchImpl });
		expect(await rr.rerank("q", ["a", "b"])).toEqual([0.2, 0.8]);
		body = { data, results: [{ index: 0, relevance_score: 0.6 }, { index: 1, relevance_score: 0.3 }] };
		expect(await rr.rerank("q", ["a", "b"])).toEqual([0.6, 0.3]);
		body = { data: data.slice(0, 1) };
		await expect(rr.rerank("q", ["a", "b"])).rejects.toMatchObject({ code: "UPSTREAM" });
	});

	it("reports degraded health when the reranker ignores its deadline", async () => {
		vi.useFakeTimers();
		try {
			const backend = new FakeRerankBackend(() => 0);
			vi.spyOn(backend, "health").mockImplementation(() => new Promise(() => {}));
			const router = new EmbeddingRouter({ backends: [new TinyEmbedBackend()] });
			const app = createEmbedHttpServer(router, { reranker: new RerankService({ backend }) });
			const response = app.request("/health");
			await vi.advanceTimersByTimeAsync(5_000);
			const res = await response;
			expect(res.status).toBe(200);
			expect(await res.json()).toMatchObject({
				status: "degraded",
				backends: [{ id: "a", ok: true }],
				reranker: { ok: false, model: backend.model, detail: "health probe timed out" },
			});
		} finally {
			vi.useRealTimers();
		}
	});

	it("parses EMBED_RERANK config with profile defaults", () => {
		expect(loadEmbedServerConfig({}).rerank).toMatchObject({ backend: "onnx", model: "onnx-community/bge-reranker-v2-m3-ONNX", dtype: "q8" });
		expect(loadEmbedServerConfig({ EMBED_PROFILE: "openai" }).rerank.backend).toBe("off");
		expect(loadEmbedServerConfig({ EMBED_PROFILE: "npu" }).rerank.backend).toBe("onnx");
		expect(() => loadEmbedServerConfig({ EMBED_RERANK: "gpu" })).toThrow(/EMBED_RERANK/);
		expect(() => loadEmbedServerConfig({ EMBED_RERANK: "http" })).toThrow(/EMBED_RERANK_URL/);
		expect(() => loadEmbedServerConfig({ EMBED_RERANK_ANSWER_THRESHOLD: "1.5" })).toThrow(/\(0, 1\)/);
		const http = loadEmbedServerConfig({
			EMBED_RERANK: "http",
			EMBED_RERANK_URL: "https://api.jina.ai/v1",
			EMBED_RERANK_MODEL: "jina-reranker-v2-base-multilingual",
		});
		expect(buildEmbeddingService(http).reranker?.backend.kind).toBe("http");
		expect(buildEmbeddingService(loadEmbedServerConfig({ EMBED_RERANK: "off" })).reranker).toBeUndefined();
	});
});

describe("memory search with a reranker (positive evidence only)", () => {
	async function seeded(reranker?: Reranker) {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		const memory = new SqliteMemoryService(appDb, { now: () => 1_000, reranker });
		// Two lexical matches for "deploy pipeline"; the second is the real answer.
		await memory.applyMemoryMutation("t", "u", { op: "ADD", key: "ops.deploy.pipeline", kind: "fact", content: "deploy pipeline deploy pipeline overview", confidence: 0.9 });
		await memory.applyMemoryMutation("t", "u", { op: "ADD", key: "ops.release", kind: "fact", content: "The deploy pipeline runs canary releases at 10:00 UTC.", confidence: 0.9 });
		await memory.applyMemoryMutation("t", "u", { op: "ADD", key: "food", kind: "preference", content: "Likes ramen.", confidence: 0.9 });
		return { appDb, memory };
	}

	it("lifts a confident answer above a stronger lexical match and records the rerank signal", async () => {
		const plain = await seeded();
		const before = await plain.memory.search("t", "u", "deploy pipeline");
		expect(before[0]?.memory.key).toBe("ops.deploy.pipeline");
		plain.appDb.close();

		const reranker: Reranker = {
			model: "fake",
			answerThreshold: 0.05,
			rerank: async (_q, docs) => docs.map((d) => (d.includes("canary") ? 0.95 : 0.01)),
		};
		const { appDb, memory } = await seeded(reranker);
		const after = await memory.search("t", "u", "deploy pipeline");
		expect(after[0]?.memory.key).toBe("ops.release");
		expect(after[0]?.matchSignals?.rerank).toBeCloseTo(0.95, 6);
		expect(after.map((h) => h.memory.key).sort()).toEqual(before.map((h) => h.memory.key).sort()); // nothing removed
		expect((await memory.search("t", "u", "deploy pipeline", { rerank: false }))[0]?.memory.key).toBe("ops.deploy.pipeline");
		appDb.close();
	});

	it("preserves the fused scores below a high threshold and still boosts a confident answer", async () => {
		let probability = 0.7;
		const { appDb, memory } = await seeded({
			model: "high-threshold",
			answerThreshold: 0.8,
			rerank: async (_q, docs) => docs.map((d) => d.includes("canary") ? probability : 0),
		});
		try {
			const before = await memory.search("t", "u", "deploy pipeline", { rerank: false });
			const below = await memory.search("t", "u", "deploy pipeline");
			expect(below.map((h) => h.score)).toEqual(before.map((h) => h.score));
			probability = 1;
			const above = await memory.search("t", "u", "deploy pipeline");
			expect(above.find((h) => h.memory.key === "ops.release")!.score).toBeGreaterThan(before.find((h) => h.memory.key === "ops.release")!.score!);
		} finally {
			appDb.close();
		}
	});

	it("keeps a strong fused score with zero rerank weight and records relevance", async () => {
		const { appDb } = await seeded();
		try {
			const memory = new SqliteMemoryService(appDb, {
				rerankWeight: 0,
				now: () => 1_000,
				reranker: { model: "fake", rerank: async (_q, docs) => docs.map(() => 1) },
			});
			await memory.pinMemory("t", "u", { key: "deploy.pipeline", content: "deploy pipeline", kind: "fact" });
			const before = await memory.search("t", "u", "deploy pipeline", { rerank: false });
			expect(before[0]!.score).toBeGreaterThan(0.85);
			const after = await memory.search("t", "u", "deploy pipeline");
			expect(after.map((h) => h.score)).toEqual(before.map((h) => h.score));
			expect(after[0]?.matchSignals?.rerank).toBe(1);
		} finally {
			appDb.close();
		}
	});

	it("never demotes on low scores and falls back to the fused order when the reranker fails", async () => {
		const low = await seeded({ model: "low", rerank: async (_q, docs) => docs.map(() => 0) });
		const plain = await seeded();
		const a = await low.memory.search("t", "u", "deploy pipeline");
		const b = await plain.memory.search("t", "u", "deploy pipeline");
		expect(a.map((h) => h.memory.key)).toEqual(b.map((h) => h.memory.key));
		a.forEach((h, i) => expect(h.score).toBeCloseTo(b[i]?.score ?? Number.NaN, 6)); // no demotion

		const broken = await seeded({
			model: "broken",
			rerank: async () => {
				throw new Error("reranker down");
			},
		});
		expect((await broken.memory.search("t", "u", "deploy pipeline")).map((h) => h.memory.key)).toEqual(b.map((h) => h.memory.key));
		for (const x of [low, plain, broken]) x.appDb.close();
	});
});

describe("harness rerank config", () => {
	it("defaults to auto-discovery and parses RERANK_* overrides", () => {
		expect(loadConfig({})).toMatchObject({ rerankEnabled: true, rerankUrl: undefined, rerankCandidates: 20 });
		expect(
			loadConfig({ RERANK_ENABLED: "false", RERANK_URL: "https://api.jina.ai/v1", RERANK_MODEL: "jina-reranker-v2-base-multilingual", RERANK_CANDIDATES: "500" }),
		).toMatchObject({ rerankEnabled: false, rerankUrl: "https://api.jina.ai/v1", rerankModel: "jina-reranker-v2-base-multilingual", rerankCandidates: 128 });
	});
});
