import { EventEmitter } from "node:events";
import { describe, expect, it } from "vitest";
import { GeminiBackend } from "../src/embeddings/backends/gemini.js";
import { OnnxBackend, type TransformersModule } from "../src/embeddings/backends/onnx.js";
import { OpenAICompatibleBackend } from "../src/embeddings/backends/openai-compatible.js";
import { isLoopbackHost, loadEmbedServerConfig } from "../src/embeddings/config.js";
import { buildEmbeddingService } from "../src/embeddings/factory.js";
import { FlmSupervisor } from "../src/embeddings/flm-supervisor.js";
import { applyTaskPrompt, resolveModelInfo, truncateAndNormalize } from "../src/embeddings/models.js";
import { HttpEmbeddingProvider, RouterEmbeddingProvider } from "../src/embeddings/provider.js";
import { EmbeddingRouter } from "../src/embeddings/router.js";
import { SELF_TEST_DISTRACTORS, SELF_TEST_PAIRS, runEmbeddingSelfTest } from "../src/embeddings/self-test.js";
import { createEmbedHttpServer } from "../src/embeddings/server.js";
import { EmbeddingError, type BackendEmbedOptions, type EmbeddingBackend } from "../src/embeddings/types.js";
import { loadConfig } from "../src/config.js";
import { AppDatabase } from "../src/memory/db.js";
import { calibrateCosine, cosineStats, vectorSalience } from "../src/memory/retrieval.js";
import { SqliteMemoryService } from "../src/memory/sqlite-memory.js";

const DIM = 64;

/** Strips the task prompts the router adds so fakes can recognise their inputs. */
function stripPrompt(text: string): string {
	return text.replace(/^task: [^|]+ \| query: /, "").replace(/^title: [^|]+ \| text: /, "");
}

function axis(i: number, dim = DIM): Float32Array {
	const v = new Float32Array(dim);
	v[i % dim] = 1;
	return v;
}

/** "Semantic" fake: self-test pair i (query and document) share axis i; anything else hashes to its own axis. */
function semanticVector(raw: string): Float32Array {
	const text = stripPrompt(raw);
	const pair = SELF_TEST_PAIRS.findIndex(([q, d]) => q === text || d === text);
	if (pair >= 0) return axis(pair);
	let h = 0;
	for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
	const v = axis(SELF_TEST_PAIRS.length + (h % (DIM - SELF_TEST_PAIRS.length)));
	v[DIM - 1] = 0.3; // shared background component, like real models
	return v;
}

class FakeBackend implements EmbeddingBackend {
	readonly kind = "openai" as const;
	readonly nativeDimensions = DIM;
	readonly maxBatchSize: number;
	readonly wantsPrompts: boolean;
	calls: Array<{ texts: string[]; options: BackendEmbedOptions }> = [];
	fail: EmbeddingError | Error | undefined;
	constructor(
		readonly id: string,
		readonly space = "test/space",
		readonly model = "fake-model",
		options: { maxBatchSize?: number; wantsPrompts?: boolean; vector?: (text: string) => Float32Array } = {},
	) {
		this.maxBatchSize = options.maxBatchSize ?? 64;
		this.wantsPrompts = options.wantsPrompts ?? false;
		if (options.vector) this.vector = options.vector;
	}
	vector: (text: string) => Float32Array = semanticVector;
	async embed(texts: readonly string[], options: BackendEmbedOptions): Promise<Float32Array[]> {
		this.calls.push({ texts: [...texts], options });
		if (this.fail) throw this.fail;
		return texts.map((t) => this.vector(t));
	}
	async health() {
		return this.fail ? { ok: false, detail: this.fail.message } : { ok: true };
	}
}

/** Fetch implementation that routes into a Hono app without opening sockets. */
function honoFetch(app: { request: (input: string, init?: RequestInit) => Response | Promise<Response> }): typeof fetch {
	return (async (input: string | URL | Request, init?: RequestInit) => {
		const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
		const { pathname, search } = new URL(url);
		return app.request(`${pathname}${search}`, init);
	}) as typeof fetch;
}

describe("embedding models", () => {
	it("keeps FastFlowLM's NPU build and Google's reference export in separate embedding spaces", () => {
		expect(resolveModelInfo("embed-gemma:300m").space).toBe("fastflowlm/embed-gemma-300m");
		expect(resolveModelInfo("onnx-community/embeddinggemma-300m-ONNX").space).toBe("google/embeddinggemma-300m");
		expect(resolveModelInfo("onnx-community/embeddinggemma-2-ONNX").space).toBe("google/embeddinggemma-2");
		expect(resolveModelInfo("unsloth/embeddinggemma-2-GGUF").space).toBe("google/embeddinggemma-2");
		expect(resolveModelInfo("text-embedding-3-small").nativeDimensions).toBe(1536);
		expect(resolveModelInfo("my-custom-model").space).toBe("my-custom-model");
		expect(resolveModelInfo("my-custom-model", { space: "shared" }).space).toBe("shared");
	});

	it("applies asymmetric task prompts idempotently", () => {
		expect(applyTaskPrompt("embeddinggemma", "query", "hi")).toBe("task: search result | query: hi");
		expect(applyTaskPrompt("embeddinggemma", "document", "hi")).toBe("title: none | text: hi");
		expect(applyTaskPrompt("embeddinggemma", "query", "task: search result | query: hi")).toBe("task: search result | query: hi");
		expect(applyTaskPrompt("embeddinggemma", "document", "title: Notes | text: hi")).toBe("title: Notes | text: hi");
		expect(applyTaskPrompt("nomic", "document", "x")).toBe("search_document: x");
		expect(applyTaskPrompt("e5", "query", "x")).toBe("query: x");
		expect(applyTaskPrompt("bge", "document", "x")).toBe("x");
		expect(applyTaskPrompt("none", "query", "x")).toBe("x");
	});

	it("truncates (Matryoshka) and renormalizes, zeroing non-finite values", () => {
		const v = truncateAndNormalize([3, 4, 12, Number.NaN], 2);
		expect(Array.from(v)).toEqual([expect.closeTo(0.6, 6), expect.closeTo(0.8, 6)]);
		expect(Array.from(truncateAndNormalize([0, 0], undefined))).toEqual([0, 0]);
		expect(truncateAndNormalize([1, 2, 3], 10)).toHaveLength(3);
	});
});

describe("EmbeddingRouter", () => {
	it("rejects backends from different embedding spaces and duplicate ids", () => {
		expect(() => new EmbeddingRouter({ backends: [new FakeBackend("a", "s1"), new FakeBackend("b", "s2")] })).toThrow(/share one embedding space/);
		expect(() => new EmbeddingRouter({ backends: [new FakeBackend("a"), new FakeBackend("a")] })).toThrow(/duplicate/);
		expect(() => new EmbeddingRouter({ backends: [] })).toThrow(/at least one/);
		expect(() => new EmbeddingRouter({ backends: [new FakeBackend("a")], dimensions: 4096 })).toThrow(/exceeds the native/);
	});

	it("falls back within the space, opens the circuit after repeated failures, and recovers", async () => {
		let now = 0;
		const primary = new FakeBackend("npu");
		const fallback = new FakeBackend("cpu");
		const events: string[] = [];
		const router = new EmbeddingRouter({
			backends: [primary, fallback],
			failureThreshold: 2,
			cooldownMs: 1_000,
			cacheSize: 0,
			now: () => now,
			onEvent: (e) => events.push(e),
		});
		primary.fail = new EmbeddingError("UNAVAILABLE", "npu busy");
		for (let i = 0; i < 2; i++) {
			const res = await router.embed({ texts: [`t${i}`] });
			expect(res.servedBy).toEqual(["cpu"]);
		}
		expect(router.status()[0]).toMatchObject({ id: "npu", circuit: "open", consecutiveFailures: 2 });
		primary.calls = [];
		await router.embed({ texts: ["skip"] });
		expect(primary.calls).toHaveLength(0); // open circuit: not even attempted

		now = 1_500; // half-open: probe succeeds and closes it
		primary.fail = undefined;
		const res = await router.embed({ texts: ["probe"] });
		expect(res.servedBy).toEqual(["npu"]);
		expect(router.status()[0]?.circuit).toBe("closed");
		expect(events).toContain("embedding_backend_recovered");
	});

	it("admits just one recovery probe and releases its slot after abort", async () => {
		const primary = new FakeBackend("primary");
		const router = new EmbeddingRouter({ backends: [primary, new FakeBackend("fallback")], failureThreshold: 1, cooldownMs: 0 });
		primary.fail = new Error("offline");
		await router.embed({ texts: ["first"] });
		const gate = Promise.withResolvers<Float32Array[]>();
		primary.embed = () => gate.promise;
		const controller = new AbortController();
		const probe = router.embed({ texts: ["probe"], signal: controller.signal });
		expect((await router.embed({ texts: ["concurrent"] })).servedBy).toEqual(["fallback"]);
		controller.abort();
		gate.resolve([axis(0)]);
		await expect(probe).rejects.toMatchObject({ code: "TIMEOUT" });
		expect((await router.embed({ texts: ["retry"] })).servedBy).toEqual(["primary"]);
	});

	it("bounds health even while a shared load is pending", async () => {
		const backend = new FakeBackend("slow");
		const gate = Promise.withResolvers<{ ok: boolean }>();
		backend.health = () => gate.promise;
		const router = new EmbeddingRouter({ backends: [backend] });
		expect((await router.health(10)).ok).toBe(false);
		gate.resolve({ ok: true });
		expect((await router.health()).ok).toBe(true);
	});

	it("does not fall back or trip the breaker on client errors", async () => {
		const primary = new FakeBackend("a");
		const fallback = new FakeBackend("b");
		const router = new EmbeddingRouter({ backends: [primary, fallback], failureThreshold: 1 });
		primary.fail = new EmbeddingError("BAD_REQUEST", "input too long");
		await expect(router.embed({ texts: ["x"] })).rejects.toMatchObject({ code: "BAD_REQUEST" });
		expect(fallback.calls).toHaveLength(0);
		expect(router.status()[0]?.circuit).toBe("closed");
	});

	it("reports UNAVAILABLE when every backend fails", async () => {
		const a = new FakeBackend("a");
		a.fail = new Error("boom");
		const router = new EmbeddingRouter({ backends: [a] });
		await expect(router.embed({ texts: ["x"] })).rejects.toMatchObject({ code: "UNAVAILABLE", status: 503 });
	});

	it("caches per input type and dimensions, deduplicates within a request, and batches", async () => {
		const backend = new FakeBackend("a", "test/space", "fake-model", { maxBatchSize: 2 });
		const router = new EmbeddingRouter({ backends: [backend] });
		const first = await router.embed({ texts: ["x", "y", "x", "z"], inputType: "document" });
		expect(first.vectors).toHaveLength(4);
		expect(first.vectors[0]).toBe(first.vectors[2]);
		expect(backend.calls.map((c) => c.texts)).toEqual([["x", "y"], ["z"]]);

		const second = await router.embed({ texts: ["x", "y"], inputType: "document" });
		expect(second.cacheHits).toBe(2);
		expect(second.servedBy).toEqual([]);

		await router.embed({ texts: ["x"], inputType: "query" });
		await router.embed({ texts: ["x"], inputType: "document", dimensions: 16 });
		expect(backend.calls).toHaveLength(4); // other input type / dims are distinct cache entries
	});

	it("applies prompts only for backends that want them and truncates to the requested dimensions", async () => {
		const backend = new FakeBackend("a", "google/embeddinggemma-2", "embeddinggemma-2", { wantsPrompts: true });
		const router = new EmbeddingRouter({ backends: [backend], dimensions: 16 });
		const res = await router.embed({ texts: ["hello"], inputType: "query" });
		expect(backend.calls[0]?.texts).toEqual(["task: search result | query: hello"]);
		expect(res.dimensions).toBe(16);
		expect(router.modelId()).toBe("google/embeddinggemma-2@16");
		await expect(router.embed({ texts: ["a"], dimensions: 4 })).rejects.toMatchObject({ code: "BAD_REQUEST" });
		await expect(router.embed({ texts: [42 as unknown as string] })).rejects.toMatchObject({ code: "BAD_REQUEST" });
	});

	it("self-test measures calibration and quarantines a backend that cannot separate related text", async () => {
		const good = new FakeBackend("good");
		const degenerate = new FakeBackend("broken", "test/space", "fake-model", {
			// Near-isotropic vectors: every text looks alike (the failure mode seen on a broken NPU export).
			vector: (t) => {
				const v = new Float32Array(DIM).fill(1);
				v[t.length % DIM] = 1.05;
				return v;
			},
		});
		const router = new EmbeddingRouter({ backends: [degenerate, good] });
		const results = await router.selfTest();
		expect(results.find((r) => r.id === "good")?.result?.passed).toBe(true);
		expect(results.find((r) => r.id === "broken")?.result?.passed).toBe(false);
		expect(router.status().find((s) => s.id === "broken")?.quarantined).toMatch(/self-test failed/);
		expect(router.calibration).toEqual(results.find((r) => r.id === "good")?.result?.calibration);

		const res = await router.embed({ texts: ["anything"] });
		expect(res.servedBy).toEqual(["good"]);
		const health = await router.health();
		expect(health.ok).toBe(true);
		expect(health.backends.find((b) => b.id === "broken")?.ok).toBe(false);
	});

	it("can quarantine an unreachable backend for the benchmark while allowing server recovery by default", async () => {
		const recovering = new FakeBackend("recovering");
		const router = new EmbeddingRouter({ backends: [recovering, new FakeBackend("good")] });
		recovering.fail = new Error("temporarily unreachable");
		await router.selfTest({ quarantineOnError: true });
		recovering.fail = undefined;
		expect((await router.embed({ texts: ["benchmark query"] })).servedBy).toEqual(["good"]);

		const serverRouter = new EmbeddingRouter({ backends: [recovering, new FakeBackend("good")] });
		recovering.fail = new Error("temporarily unreachable");
		await serverRouter.selfTest();
		recovering.fail = undefined;
		expect((await serverRouter.embed({ texts: ["server query"] })).servedBy).toEqual(["recovering"]);
	});

	it("keeps an explicitly configured calibration over the measured one", async () => {
		const router = new EmbeddingRouter({ backends: [new FakeBackend("a")], calibration: { floor: 0.1, ceiling: 0.9 } });
		await router.selfTest();
		expect(router.calibration).toEqual({ floor: 0.1, ceiling: 0.9 });
	});
});

describe("runEmbeddingSelfTest", () => {
	it("scores a discriminative model as passing with a floor below its ceiling", async () => {
		const result = await runEmbeddingSelfTest(async (texts) => texts.map(semanticVector));
		expect(result.passed).toBe(true);
		expect(result.top1).toBe(1);
		expect(result.calibration!.ceiling).toBeGreaterThan(result.calibration!.floor);
		expect(SELF_TEST_DISTRACTORS.length).toBeGreaterThan(0);
	});
});

describe("embedding HTTP server", () => {
	function makeServer(options: Parameters<typeof createEmbedHttpServer>[1] = {}) {
		const backend = new FakeBackend("a", "google/embeddinggemma-2", "embeddinggemma-2", { wantsPrompts: true });
		const router = new EmbeddingRouter({ backends: [backend] });
		return { backend, router, app: createEmbedHttpServer(router, options) };
	}

	it("serves health, models and OpenAI-compatible embeddings (float + base64, input_type)", async () => {
		const { app, backend } = makeServer();
		const health = await app.request("/health");
		expect(health.status).toBe(200);
		expect(await health.json()).toMatchObject({ status: "ok", space: "google/embeddinggemma-2", model: "google/embeddinggemma-2@768" });

		const models = (await (await app.request("/v1/models")).json()) as { data: Array<Record<string, unknown>> };
		expect(models.data[0]).toMatchObject({ id: "google/embeddinggemma-2@768", space: "google/embeddinggemma-2", matryoshka_dimensions: [768, 512, 256, 128] });

		const res = await app.request("/v1/embeddings", {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({ input: ["a", "b"], input_type: "search_query" }),
		});
		expect(res.status).toBe(200);
		const body = (await res.json()) as { model: string; data: Array<{ index: number; embedding: number[] }>; usage: { total_tokens: number } };
		expect(body.data).toHaveLength(2);
		expect(body.data[1]?.index).toBe(1);
		expect(body.data[0]?.embedding).toHaveLength(DIM);
		expect(backend.calls[0]?.texts[0]).toBe("task: search result | query: a");
		expect(body.usage.total_tokens).toBeGreaterThan(0);

		const b64 = await app.request("/v1/embeddings", {
			method: "POST",
			body: JSON.stringify({ input: "a", encoding_format: "base64", dimensions: 32 }),
		});
		const b64Body = (await b64.json()) as { model: string; data: Array<{ embedding: string }> };
		expect(b64Body.model).toBe("google/embeddinggemma-2@32");
		expect(Buffer.from(b64Body.data[0]!.embedding, "base64").byteLength).toBe(32 * 4);
	});

	it("validates input and enforces limits", async () => {
		const { app } = makeServer({ maxInputs: 2, maxBodyBytes: 2_048 });
		const post = (body: unknown) => app.request("/v1/embeddings", { method: "POST", body: typeof body === "string" ? body : JSON.stringify(body) });
		expect((await post({ input: [] })).status).toBe(400);
		expect((await post({ input: ["a", "b", "c"] })).status).toBe(400);
		expect((await post({ input: [[1, 2]] })).status).toBe(400);
		expect((await post({ input: "a", input_type: "weird" })).status).toBe(400);
		expect((await post({ input: "a", encoding_format: "int8" })).status).toBe(400);
		expect((await post({ input: "a", dimensions: 1.5 })).status).toBe(400);
		expect((await post("not json")).status).toBe(400);
		expect((await post({ input: "x".repeat(4_096) })).status).toBe(413);
	});

	it("cancels an oversized stream before the sender finishes", async () => {
		const { app } = makeServer({ maxBodyBytes: 12 });
		let cancelled = false;
		const body = new ReadableStream({
			start(controller) {
				controller.enqueue(new Uint8Array(8));
				controller.enqueue(new Uint8Array(8));
			},
			cancel() { cancelled = true; },
		});
		const response = await app.request("/v1/embeddings", { method: "POST", body, duplex: "half" } as RequestInit);
		expect(response.status).toBe(413);
		expect(cancelled).toBe(true);
	});

	it("requires a bearer token on /v1 when API keys are configured, keeping /health public", async () => {
		const { app } = makeServer({ apiKeys: ["secret-1"] });
		expect((await app.request("/v1/models")).status).toBe(401);
		expect((await app.request("/v1/models", { headers: { authorization: "Bearer nope" } })).status).toBe(401);
		expect((await app.request("/v1/models", { headers: { authorization: "Bearer secret-1" } })).status).toBe(200);
		expect((await app.request("/health")).status).toBe(200);
	});

	it("returns 503 from /health and the embeddings route when no backend works", async () => {
		const backend = new FakeBackend("a");
		backend.fail = new Error("down");
		const app = createEmbedHttpServer(new EmbeddingRouter({ backends: [backend] }));
		expect((await app.request("/health")).status).toBe(503);
		const res = await app.request("/v1/embeddings", { method: "POST", body: JSON.stringify({ input: "x" }) });
		expect(res.status).toBe(503);
		expect(((await res.json()) as { error: { type: string } }).error.type).toBe("unavailable");
		expect(await (await app.request("/health")).text()).not.toMatch(/lastError|detail|down/);
	});
});

describe("HttpEmbeddingProvider", () => {
	it("detects a Helios server, stores vectors under <space>@<dims>, forwards input_type and adopts its calibration", async () => {
		const backend = new FakeBackend("a", "google/embeddinggemma-2", "embeddinggemma-2", { wantsPrompts: true });
		const router = new EmbeddingRouter({ backends: [backend], calibration: { floor: 0.6, ceiling: 0.8 } });
		const app = createEmbedHttpServer(router);
		const provider = await HttpEmbeddingProvider.connect({ baseUrl: "http://embed.local/v1", fetchImpl: honoFetch(app) });
		expect(provider.helios).toBe(true);
		expect(provider.model).toBe("google/embeddinggemma-2@768");
		expect(provider.similarity).toEqual({ floor: 0.6, ceiling: 0.8 });

		const vecs = await provider.embed(["hi"], { inputType: "query" });
		expect(vecs[0]).toHaveLength(DIM);
		expect(backend.calls[0]?.texts).toEqual(["task: search result | query: hi"]);
	});

	it("discovers native dimensions for an unknown checkpoint before storing its identity", async () => {
		const router = new EmbeddingRouter({ backends: [new FakeBackend("a")] });
		const provider = await HttpEmbeddingProvider.connect({ baseUrl: "http://embed.local/v1", fetchImpl: honoFetch(createEmbedHttpServer(router)) });
		expect(provider.model).toBe(`test/space@${DIM}`);
		expect(await provider.embed(["second request"])).toHaveLength(1);
	});

	it("rejects malformed response indices while accepting reordered rows", async () => {
		let indices = [1, 0];
		const fetchImpl: typeof fetch = async url => String(url).endsWith("/models")
			? Response.json({ data: [] })
			: Response.json({ data: indices.map((index, i) => ({ index, embedding: [i + 1] })) });
		const provider = await HttpEmbeddingProvider.connect({ baseUrl: "http://fixture/v1", model: "fixture", fetchImpl });
		expect((await provider.embed(["a", "b"])).map(row => row[0])).toEqual([2, 1]);
		for (indices of [[0, 0], [0, 2], [0, -1], [0, 0.5]]) {
			await expect(provider.embed(["a", "b"])).rejects.toMatchObject({ code: "UPSTREAM" });
		}
	});

	it("refuses vectors from a server that switched embedding space", async () => {
		let space = "space-a";
		const fetchImpl = (async (input: string | URL | Request) => {
			const url = String(input);
			if (url.endsWith("/models")) return Response.json({ data: [{ id: `${space}@4`, space, dimensions: 4 }] });
			return Response.json({ model: `${space}@4`, data: [{ index: 0, embedding: [1, 0, 0, 0] }] });
		}) as typeof fetch;
		const provider = await HttpEmbeddingProvider.connect({ baseUrl: "http://x/v1", fetchImpl });
		await expect(provider.embed(["a"])).resolves.toHaveLength(1);
		space = "space-b";
		await expect(provider.embed(["a"])).rejects.toMatchObject({ code: "MISCONFIGURED" });
	});

	it("works against plain OpenAI-compatible endpoints when a model is given, and retries transient errors", async () => {
		let calls = 0;
		const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
			const url = String(input);
			if (url.endsWith("/models")) return new Response("not found", { status: 404 });
			calls += 1;
			if (calls === 1) return new Response("overloaded", { status: 503 });
			const body = JSON.parse(String(init?.body)) as { input: string[]; input_type?: string };
			expect(body.input_type).toBeUndefined(); // not sent to non-Helios endpoints
			return Response.json({ data: body.input.map((_, i) => ({ index: i, embedding: [i + 1, 0] })) });
		}) as typeof fetch;
		await expect(HttpEmbeddingProvider.connect({ baseUrl: "http://x/v1", fetchImpl })).rejects.toMatchObject({ code: "MISCONFIGURED" });
		const provider = await HttpEmbeddingProvider.connect({
			baseUrl: "http://x/v1",
			model: "text-embedding-3-small",
			fetchImpl,
			sleep: async () => undefined,
		});
		expect(provider.helios).toBe(false);
		expect(provider.model).toBe("openai/text-embedding-3-small@1536");
		const vecs = await provider.embed(["a", "b"], { inputType: "query" });
		expect(Array.from(vecs[1]!)).toEqual([2, 0]);
		expect(calls).toBe(2);
	});

	it("reports an unreachable endpoint as UNAVAILABLE and a rejected key as MISCONFIGURED", async () => {
		const down = (async () => {
			throw new TypeError("fetch failed");
		}) as typeof fetch;
		await expect(HttpEmbeddingProvider.connect({ baseUrl: "http://x/v1", fetchImpl: down })).rejects.toMatchObject({ code: "UNAVAILABLE" });
		const denied = (async () => new Response("no", { status: 401 })) as typeof fetch;
		await expect(HttpEmbeddingProvider.connect({ baseUrl: "http://x/v1", fetchImpl: denied })).rejects.toMatchObject({ code: "MISCONFIGURED" });
	});

	it("RouterEmbeddingProvider embeds in-process with the router's model id and calibration", async () => {
		const router = new EmbeddingRouter({ backends: [new FakeBackend("a")], calibration: { floor: 0.2, ceiling: 0.7 } });
		const provider = new RouterEmbeddingProvider(router);
		expect(provider.model).toBe("test/space");
		expect(provider.similarity).toEqual({ floor: 0.2, ceiling: 0.7 });
		expect(await provider.embed(["x"], { inputType: "query" })).toHaveLength(1);
	});
});

describe("embedding backends", () => {
	it("OpenAI-compatible backend reorders by index, decodes base64, and validates counts", async () => {
		const seen: Array<{ url: string; headers: Record<string, string>; body: Record<string, unknown> }> = [];
		const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
			seen.push({ url: String(input), headers: init?.headers as Record<string, string>, body: JSON.parse(String(init?.body)) });
			const b64 = Buffer.from(new Float32Array([0.5, 0.25]).buffer).toString("base64");
			return Response.json({ data: [{ index: 1, embedding: b64 }, { index: 0, embedding: [1, 2] }] });
		}) as typeof fetch;
		const backend = new OpenAICompatibleBackend({ id: "npu", kind: "npu", baseUrl: "http://127.0.0.1:52625/v1/", model: "embed-gemma:300m", apiKey: "k", fetchImpl });
		expect(backend.space).toBe("fastflowlm/embed-gemma-300m");
		expect(backend.wantsPrompts).toBe(true);
		const out = await backend.embed(["a", "b"], { inputType: "document", dimensions: 128 });
		expect(Array.from(out[0]!)).toEqual([1, 2]);
		expect(Array.from(out[1]!)).toEqual([0.5, 0.25]);
		expect(seen[0]?.url).toBe("http://127.0.0.1:52625/v1/embeddings");
		expect(seen[0]?.headers.authorization).toBe("Bearer k");
		expect(seen[0]?.body.dimensions).toBeUndefined(); // only sent when sendDimensions

		const short = new OpenAICompatibleBackend({
			id: "x",
			baseUrl: "http://h/v1",
			model: "m",
			fetchImpl: (async () => Response.json({ data: [{ index: 0, embedding: [1] }] })) as typeof fetch,
		});
		await expect(short.embed(["a", "b"], { inputType: "document" })).rejects.toMatchObject({ code: "UPSTREAM" });
		const nan = new OpenAICompatibleBackend({
			id: "x",
			baseUrl: "http://h/v1",
			model: "m",
			fetchImpl: (async () => Response.json({ data: [{ index: 0, embedding: [1, "x"] }] })) as typeof fetch,
		});
		await expect(nan.embed(["a"], { inputType: "document" })).rejects.toMatchObject({ code: "UPSTREAM" });
		const http500 = new OpenAICompatibleBackend({
			id: "x",
			baseUrl: "http://h/v1",
			model: "m",
			fetchImpl: (async () => new Response("bad", { status: 500 })) as typeof fetch,
		});
		await expect(http500.embed(["a"], { inputType: "document" })).rejects.toMatchObject({ code: "UNAVAILABLE" });
	});

	it("Gemini backend uses taskType + outputDimensionality and keeps the key out of the URL", async () => {
		const seen: Array<{ url: string; headers: Record<string, string>; body: { requests: Array<Record<string, unknown>> } }> = [];
		const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
			const body = JSON.parse(String(init?.body)) as { requests: unknown[] };
			seen.push({ url: String(input), headers: init?.headers as Record<string, string>, body: body as never });
			return Response.json({ embeddings: body.requests.map(() => ({ values: [0.1, 0.2] })) });
		}) as typeof fetch;
		const backend = new GeminiBackend({ id: "gemini", apiKey: "g-key", fetchImpl });
		const out = await backend.embed(["q"], { inputType: "query", dimensions: 768 });
		expect(out).toHaveLength(1);
		expect(seen[0]?.url).toBe("https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents");
		expect(seen[0]?.url).not.toContain("g-key");
		expect(seen[0]?.headers["x-goog-api-key"]).toBe("g-key");
		expect(seen[0]?.body.requests[0]).toMatchObject({ taskType: "RETRIEVAL_QUERY", outputDimensionality: 768 });
		await backend.embed(["d"], { inputType: "document" });
		expect(seen[1]?.body.requests[0]).toMatchObject({ taskType: "RETRIEVAL_DOCUMENT" });
		expect(() => new GeminiBackend({ id: "g", apiKey: " " })).toThrow(/API key/);
	});

	it("applies the configured family to an unknown ONNX checkpoint during serving and self-test", async () => {
		const seen: string[] = [];
		const backend = new OnnxBackend({
			id: "custom", model: "custom/checkpoint", promptFamily: "e5",
			loadTransformers: async () => ({
				env: {}, pipeline: async () => async texts => {
					seen.push(...texts);
					return { tolist: () => texts.map(() => [1, 0]) };
				},
			}),
		});
		const router = new EmbeddingRouter({ backends: [backend], promptFamily: "e5" });
		await router.embed({ texts: ["example"] });
		await router.selfTest();
		expect(seen).toContain("passage: example");
		expect(seen).toContain(`query: ${SELF_TEST_PAIRS[0]![0]}`);
		expect(seen).toContain(`passage: ${SELF_TEST_PAIRS[0]![1]}`);
	});

	it("ONNX backend loads lazily once, serializes inference, and explains a missing optional dependency", async () => {
		let loads = 0;
		let active = 0;
		let maxActive = 0;
		const fake: TransformersModule = {
			env: {},
			async pipeline(_task, model, options) {
				loads += 1;
				expect(model).toBe("onnx-community/embeddinggemma-2-ONNX");
				expect(options).toEqual({ device: "cpu", dtype: "q4" });
				const extractor = Object.assign(
					async (texts: string[]) => {
						active += 1;
						maxActive = Math.max(maxActive, active);
						await new Promise((r) => setTimeout(r, 5));
						active -= 1;
						return { tolist: () => texts.map((_, i) => [i, 1]) };
					},
					{ dispose: async () => undefined },
				);
				return extractor;
			},
		};
		const backend = new OnnxBackend({ id: "onnx", cacheDir: "/tmp/models", loadTransformers: async () => fake });
		expect(backend.space).toBe("google/embeddinggemma-2");
		expect((await backend.health()).ok).toBe(true);
		const [a, b] = await Promise.all([backend.embed(["x", "y"], { inputType: "document" }), backend.embed(["z"], { inputType: "query" })]);
		expect(a).toHaveLength(2);
		expect(b).toHaveLength(1);
		expect(loads).toBe(1);
		expect(maxActive).toBe(1);
		expect(fake.env.cacheDir).toBe("/tmp/models");
		await backend.close();

		const missing = new OnnxBackend({
			id: "onnx",
			loadTransformers: async () => {
				throw new Error("Cannot find package '@huggingface/transformers'");
			},
		});
		await expect(missing.embed(["x"], { inputType: "document" })).rejects.toThrow(/npm install @huggingface\/transformers/);
		expect((await missing.health()).ok).toBe(false);
	});
});

describe("embedding server config & factory", () => {
	it("defaults to EmbeddingGemma 2 on ONNX and resolves profiles", () => {
		const def = loadEmbedServerConfig({});
		expect(def).toMatchObject({ host: "127.0.0.1", port: 18765, backends: ["onnx"], selfTest: true });
		expect(def.onnx.model).toBe("onnx-community/embeddinggemma-2-ONNX");
		const npu = loadEmbedServerConfig({ EMBED_PROFILE: "npu", EMBED_NPU_AUTOSTART: "true" });
		expect(npu.backends).toEqual(["npu"]);
		expect(npu.npu).toMatchObject({ model: "embed-gemma:300m", autostart: true, url: "http://127.0.0.1:52625/v1" });
		const cloud = loadEmbedServerConfig({ EMBED_PROFILE: "openai", OPENAI_API_KEY: "sk-test" });
		expect(cloud.openai).toMatchObject({ apiKey: "sk-test", sendDimensions: true, model: "text-embedding-3-small" });
		const router = loadEmbedServerConfig({ EMBED_PROFILE: "openai", EMBED_OPENAI_URL: "https://openrouter.ai/api/v1", OPENROUTER_API_KEY: "or" });
		expect(router.openai).toMatchObject({ apiKey: "or", sendDimensions: false });
		expect(loadEmbedServerConfig({ EMBED_PROFILE: "gemini", GEMINI_API_KEY: "g" }).gemini.apiKey).toBe("g");
	});

	it("rejects invalid configuration loudly", () => {
		expect(() => loadEmbedServerConfig({ EMBED_PROFILE: "tpu" })).toThrow(/unknown/);
		expect(() => loadEmbedServerConfig({ EMBED_BACKENDS: "npu,quantum" })).toThrow(/unknown backend/);
		expect(() => loadEmbedServerConfig({ EMBED_PORT: "abc" })).toThrow(/EMBED_PORT/);
		expect(() => loadEmbedServerConfig({ EMBED_DIMENSIONS: "4" })).toThrow(/EMBED_DIMENSIONS/);
		expect(() => loadEmbedServerConfig({ EMBED_SIMILARITY_FLOOR: "0.5" })).toThrow(/both/);
		expect(() => loadEmbedServerConfig({ EMBED_SIMILARITY_FLOOR: "0.8", EMBED_SIMILARITY_CEILING: "0.5" })).toThrow(/greater/);
		expect(() => loadEmbedServerConfig({ EMBED_ONNX_DEVICE: "npu" })).toThrow(/EMBED_ONNX_DEVICE/);
		expect(() => loadEmbedServerConfig({ EMBED_SELF_TEST: "maybe" })).toThrow(/true or false/);
		expect(() => buildEmbeddingService(loadEmbedServerConfig({ EMBED_PROFILE: "gemini" }))).toThrow(/GEMINI_API_KEY/);
		expect(() => buildEmbeddingService(loadEmbedServerConfig({ EMBED_BACKENDS: "npu,onnx" }))).toThrow(/share one embedding space/);
	});

	it("builds a FastFlowLM supervisor only when NPU autostart is enabled", () => {
		const off = buildEmbeddingService(loadEmbedServerConfig({ EMBED_PROFILE: "npu" }));
		expect(off.supervisor).toBeUndefined();
		const on = buildEmbeddingService(loadEmbedServerConfig({ EMBED_PROFILE: "npu", EMBED_NPU_AUTOSTART: "1", EMBED_NPU_URL: "http://127.0.0.1:52700/v1" }));
		expect(on.supervisor?.baseUrl).toBe("http://127.0.0.1:52700/v1");
	});

	it("recognises loopback hosts", () => {
		for (const h of ["127.0.0.1", "localhost", "::1", "[::1]", "127.1.2.3"]) expect(isLoopbackHost(h)).toBe(true);
		for (const h of ["0.0.0.0", "192.168.1.5", "example.com"]) expect(isLoopbackHost(h)).toBe(false);
	});
});

describe("FlmSupervisor", () => {
	it("reuses a running FastFlowLM, otherwise spawns `flm serve <llm> --embed 1` and stops it", async () => {
		let up = false;
		const fetchImpl = (async () => (up ? new Response("{}") : Promise.reject(new TypeError("refused")))) as typeof fetch;
		const spawned: Array<{ cmd: string; args: readonly string[] }> = [];
		const child = Object.assign(new EventEmitter(), {
			exitCode: null as number | null,
			kill() {
				child.exitCode = 0;
				child.emit("exit", 0);
				return true;
			},
		});
		const spawnImpl = ((cmd: string, args: readonly string[]) => {
			spawned.push({ cmd, args });
			setTimeout(() => {
				up = true;
			}, 10);
			return child;
		}) as never;
		const sup = new FlmSupervisor({ port: 52700, fetchImpl, spawnImpl, startupTimeoutMs: 5_000 });
		expect(await sup.ensureRunning()).toBe(true);
		expect(spawned[0]?.cmd).toBe("flm");
		expect(spawned[0]?.args).toEqual(expect.arrayContaining(["serve", "qwen3:0.6b", "--embed", "1", "--port", "52700"]));
		await sup.stop();
		expect(child.exitCode).toBe(0);

		const again = new FlmSupervisor({ port: 52700, fetchImpl, spawnImpl });
		expect(await again.ensureRunning()).toBe(true);
		expect(spawned).toHaveLength(1); // already reachable: nothing spawned
	});
});

describe("query-relative vector salience", () => {
	it("gates cosines by their z-score against the query's own distribution, only on large scopes", () => {
		const background = Array.from({ length: 50 }, (_, i) => 0.5 + (i % 5) * 0.01);
		const stats = cosineStats([...background, 0.75]);
		expect(stats.count).toBe(51);
		expect(vectorSalience(0.75, stats)).toBe(1); // clear outlier
		expect(vectorSalience(0.52, stats)).toBe(0); // background
		const mid = stats.mean + 2.25 * stats.std;
		expect(vectorSalience(mid, stats)).toBeCloseTo(0.5, 2);
		expect(vectorSalience(0.52, cosineStats([0.5, 0.52, 0.9]))).toBe(1); // tiny scope: no gating
		expect(vectorSalience(0.5, undefined)).toBe(1);
	});

	it("calibrates cosine over a model-specific [floor, ceiling]", () => {
		expect(calibrateCosine(0.6, 0.618, 0.773)).toBe(0);
		expect(calibrateCosine(0.773, 0.618, 0.773)).toBeCloseTo(1, 6);
		expect(calibrateCosine(0.5)).toBeGreaterThan(0); // MiniLM default floor 0.24
		expect(calibrateCosine(0.5, 0.618, 0.773)).toBe(0);
	});

	it("returns a group of strong semantic matches without other evidence", async () => {
		const db = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const service = new SqliteMemoryService(db, { embeddingProvider: {
				model: "group-test",
				async embed(texts, options) {
					return texts.map(text => {
						if (options?.inputType === "query") return [1, 0];
						const cosine = text.includes("amber") ? 0.8 : 0.4;
						return [cosine, Math.sqrt(1 - cosine ** 2)];
					});
				},
			} });
			for (let i = 0; i < 20; i++) {
				await service.pinMemory("t", "u", { key: `item.${i}`, content: `${i < 10 ? "amber" : "blue"} sample ${i}` });
			}
			const hits = await service.search("t", "u", "opaque inquiry", { limit: 20, useGraph: false, temporalMode: "off" });
			expect(hits.map(h => h.memory.key).sort()).toEqual(Array.from({ length: 10 }, (_, i) => `item.${i}`));
		} finally {
			db.close();
		}
	});

	it("SqliteMemoryService honours the provider's declared calibration and embeds queries/documents separately", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const calls: Array<{ texts: string[]; inputType: string | undefined }> = [];
			// Instruction-tuned-like geometry: the editor query sits at cosine 0.8 from the editor memory and 0.55 from
			// the unrelated one; the unrelated query sits at ~0.55 from BOTH memories (pure background similarity).
			const editorDoc = [0.8, 0.6, 0];
			const otherDoc = [0.55, 0, Math.sqrt(1 - 0.55 ** 2)];
			const cross = [0.6 * otherDoc[2]!, -0.8 * otherDoc[2]!, -0.6 * 0.55];
			const crossNorm = Math.hypot(...cross);
			const backgroundQuery = editorDoc.map((v, i) => v + otherDoc[i]! + (1.993 * cross[i]!) / crossNorm);
			const provider = {
				model: "narrow-band@3",
				similarity: { floor: 0.62, ceiling: 0.78 },
				async embed(texts: readonly string[], options?: { inputType?: "query" | "document" }) {
					calls.push({ texts: [...texts], inputType: options?.inputType });
					return texts.map((t) => {
						if (options?.inputType === "query") return /editor/i.test(t) ? [1, 0, 0] : backgroundQuery;
						return /neovim/i.test(t) ? editorDoc : otherDoc;
					});
				},
			};
			const plain = new SqliteMemoryService(appDb);
			await plain.pinMemory("t", "u", { key: "user.tool", content: "Daily driver is neovim" });
			await plain.pinMemory("t", "u", { key: "user.food", content: "Likes ramen" });
			const calibrated = new SqliteMemoryService(appDb, { embeddingProvider: provider });
			const hits = await calibrated.search("t", "u", "text editor");
			expect(hits.map((h) => h.memory.key)).toEqual(["user.tool"]);
			expect(calls.find((c) => c.texts.includes("text editor"))?.inputType).toBe("query");
			expect(calls.some((c) => c.inputType === "document" && c.texts.length === 2)).toBe(true);

			// Without the declared calibration the MiniLM default (floor 0.24) would treat 0.55 as a strong match.
			const uncalibrated = new SqliteMemoryService(appDb, { embeddingProvider: { ...provider, model: "narrow-band-raw@3", similarity: undefined } });
			const noisy = await uncalibrated.search("t", "u", "something unrelated entirely");
			const calibratedNoisy = await calibrated.search("t", "u", "something unrelated entirely");
			expect(calibratedNoisy.length).toBeLessThan(noisy.length);
		} finally {
			appDb.close();
		}
	});
});

describe("harness embedding config", () => {
	it("selects implicit server credentials only for exact provider hosts", () => {
		const config = (url: string) => loadEmbedServerConfig({ EMBED_OPENAI_URL: url, OPENAI_API_KEY: "oa", OPENROUTER_API_KEY: "or" }).openai;
		expect(config("https://api.openai.com/v1").apiKey).toBe("oa");
		expect(config("https://openrouter.ai/v1").apiKey).toBe("or");
		expect(config("https://openrouter.ai.example/v1").apiKey).toBeUndefined();
		expect(config("https://example.org/api.openai.com").apiKey).toBeUndefined();
	});

	it("is off by default and parses EMBEDDINGS_* (OpenAI key fallback only for api.openai.com)", () => {
		expect(loadConfig({}).embeddingsUrl).toBeUndefined();
		const local = loadConfig({ EMBEDDINGS_URL: "http://127.0.0.1:18765/v1", EMBEDDINGS_DIMENSIONS: "256", OPENAI_API_KEY: "sk" });
		expect(local).toMatchObject({ embeddingsUrl: "http://127.0.0.1:18765/v1", embeddingsDimensions: 256, embeddingsTimeoutMs: 15_000, embeddingsRequired: false });
		expect(local.embeddingsApiKey).toBeUndefined(); // never leak the OpenAI key to a third-party endpoint
		const cloud = loadConfig({ EMBEDDINGS_URL: "https://api.openai.com/v1", EMBEDDINGS_MODEL: "text-embedding-3-small", OPENAI_API_KEY: "sk", EMBEDDINGS_REQUIRED: "true" });
		expect(cloud).toMatchObject({ embeddingsApiKey: "sk", embeddingsModel: "text-embedding-3-small", embeddingsRequired: true });
	});
});
