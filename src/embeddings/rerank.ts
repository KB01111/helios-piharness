/**
 * Cross-encoder reranking for the Helios embedding server (`POST /v1/rerank`).
 *
 * A reranker reads (query, document) together and scores whether the document ANSWERS the query, which bi-encoder
 * embeddings cannot: "serial number of my espresso grinder" is *similar* to "likes Ethiopian coffee" but is not
 * answered by it. Memory search uses the score to reorder its top candidates and to abstain.
 *
 * Backends:
 *  - `onnx`: in-process cross-encoder via Transformers.js (default `onnx-community/bge-reranker-v2-m3-ONNX`, q8;
 *    multilingual, 568M). Scores are sigmoid(logit).
 *  - `http`: Cohere / Jina / Voyage-compatible `POST {base}/rerank` (cloud rerankers).
 */

import crypto from "node:crypto";
import { errorText, postJson, probe } from "./backends/http-util.js";
import { SELF_TEST_PAIRS } from "./self-test.js";
import { EmbeddingError, type BackendHealth } from "./types.js";

export interface RerankBackend {
	readonly id: string;
	readonly kind: "onnx" | "http";
	readonly model: string;
	readonly maxBatchSize: number;
	/** Relevance in [0, 1] per document, input order. */
	score(query: string, documents: readonly string[], signal?: AbortSignal): Promise<number[]>;
	health(signal?: AbortSignal): Promise<BackendHealth>;
	close?(): Promise<void>;
}

// ── ONNX cross-encoder ───────────────────────────────────────────────────────────────────────────

interface Tokenized {
	readonly [key: string]: unknown;
}
interface SequenceClassifier {
	(inputs: Tokenized): Promise<{ logits: { data: ArrayLike<number>; dims: readonly number[] } }>;
	dispose?: () => Promise<void>;
}
type Tokenizer = (texts: string[], options: { text_pair: string[]; padding: boolean; truncation: boolean; max_length?: number }) => Tokenized;
export interface TransformersRerankModule {
	AutoTokenizer: { from_pretrained(model: string): Promise<Tokenizer> };
	AutoModelForSequenceClassification: { from_pretrained(model: string, options: { dtype?: string; device?: string }): Promise<SequenceClassifier> };
	env: { cacheDir?: string; allowRemoteModels?: boolean };
}

export interface OnnxRerankBackendOptions {
	readonly id?: string;
	readonly model?: string;
	readonly dtype?: string;
	readonly device?: string;
	readonly cacheDir?: string;
	readonly offline?: boolean;
	/** Max tokens per (query, document) pair (default 512). */
	readonly maxLength?: number;
	/** Test seam replacing the dynamic import. */
	readonly loadTransformers?: () => Promise<TransformersRerankModule>;
}

const RERANK_MICRO_BATCH = 8;

export class OnnxRerankBackend implements RerankBackend {
	readonly id: string;
	readonly kind = "onnx" as const;
	readonly model: string;
	readonly maxBatchSize = 64;
	private readonly options: OnnxRerankBackendOptions;
	private loaded: Promise<{ tokenizer: Tokenizer; classifier: SequenceClassifier }> | undefined;
	private loadError: string | undefined;
	private queue: Promise<unknown> = Promise.resolve();

	constructor(options: OnnxRerankBackendOptions = {}) {
		this.id = options.id ?? "rerank-onnx";
		this.model = options.model?.trim() || "onnx-community/bge-reranker-v2-m3-ONNX";
		this.options = options;
	}

	private load(): Promise<{ tokenizer: Tokenizer; classifier: SequenceClassifier }> {
		if (!this.loaded) {
			this.loaded = (async () => {
				let mod: TransformersRerankModule;
				try {
					mod = this.options.loadTransformers
						? await this.options.loadTransformers()
						: ((await import(/* optional dependency */ "@huggingface/transformers" as string)) as TransformersRerankModule);
				} catch (error) {
					throw new EmbeddingError(
						"MISCONFIGURED",
						`reranker ${this.id}: @huggingface/transformers is not installed (npm install @huggingface/transformers): ${errorText(error)}`,
						{ cause: error },
					);
				}
				if (this.options.cacheDir) mod.env.cacheDir = this.options.cacheDir;
				if (this.options.offline) mod.env.allowRemoteModels = false;
				const [tokenizer, classifier] = await Promise.all([
					mod.AutoTokenizer.from_pretrained(this.model),
					mod.AutoModelForSequenceClassification.from_pretrained(this.model, {
						dtype: this.options.dtype ?? "q8",
						...(this.options.device && this.options.device !== "auto" ? { device: this.options.device } : {}),
					}),
				]);
				return { tokenizer, classifier };
			})();
			this.loaded.then(
				() => {
					this.loadError = undefined;
				},
				(error: unknown) => {
					this.loadError = errorText(error);
					this.loaded = undefined;
				},
			);
		}
		return this.loaded;
	}

	async score(query: string, documents: readonly string[], signal?: AbortSignal): Promise<number[]> {
		if (documents.length === 0) return [];
		const checkAborted = () => {
			if (signal?.aborted) throw new EmbeddingError("TIMEOUT", `reranker ${this.id}: request aborted`);
		};
		checkAborted();
		const run = async (): Promise<number[]> => {
			checkAborted();
			const { tokenizer, classifier } = await this.load();
			checkAborted();
			const order = documents.map((d, i) => ({ i, len: d.length })).sort((a, b) => a.len - b.len);
			const out = new Array<number>(documents.length);
			for (let start = 0; start < order.length; start += RERANK_MICRO_BATCH) {
				checkAborted();
				const slice = order.slice(start, start + RERANK_MICRO_BATCH);
				const inputs = tokenizer(
					slice.map(() => query),
					{ text_pair: slice.map((o) => documents[o.i]!), padding: true, truncation: true, max_length: this.options.maxLength ?? 512 },
				);
				const { logits } = await classifier(inputs);
				const width = logits.dims[logits.dims.length - 1] ?? 1;
				slice.forEach((o, n) => {
					// Single-logit heads: sigmoid. Two-class heads: softmax probability of the "relevant" class.
					const a = Number(logits.data[n * width]);
					const p = width >= 2 ? 1 / (1 + Math.exp(a - Number(logits.data[n * width + 1]))) : 1 / (1 + Math.exp(-a));
					out[o.i] = Number.isFinite(p) ? p : 0;
				});
			}
			return out;
		};
		const result = this.queue.then(run, run);
		this.queue = result.catch(() => undefined);
		return result;
	}

	async health(): Promise<BackendHealth> {
		if (this.loadError) return { ok: false, detail: this.loadError };
		if (!this.loaded) return { ok: true, detail: "model loads on first request" };
		try {
			await this.loaded;
			return { ok: true, detail: this.model };
		} catch (error) {
			return { ok: false, detail: errorText(error) };
		}
	}

	async close(): Promise<void> {
		const pending = this.loaded;
		this.loaded = undefined;
		if (!pending) return;
		try {
			await (await pending).classifier.dispose?.();
		} catch {
			// irrelevant at shutdown
		}
	}
}

// ── HTTP (Cohere / Jina / Voyage) ────────────────────────────────────────────────────────────────

export interface HttpRerankBackendOptions {
	readonly id?: string;
	/** e.g. `https://api.jina.ai/v1`, `https://api.cohere.com/v2`, `https://api.voyageai.com/v1`. */
	readonly baseUrl: string;
	readonly model: string;
	readonly apiKey?: string;
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
}

export class HttpRerankBackend implements RerankBackend {
	readonly id: string;
	readonly kind = "http" as const;
	readonly model: string;
	readonly maxBatchSize = 100;
	private readonly baseUrl: string;
	private readonly headers: Record<string, string>;
	private readonly timeoutMs: number;
	private readonly fetchImpl: typeof fetch | undefined;

	constructor(options: HttpRerankBackendOptions) {
		if (!options.baseUrl.trim() || !options.model.trim()) {
			throw new EmbeddingError("MISCONFIGURED", "http reranker needs EMBED_RERANK_URL and EMBED_RERANK_MODEL");
		}
		this.id = options.id ?? "rerank-http";
		this.model = options.model;
		this.baseUrl = options.baseUrl.replace(/\/+$/, "");
		this.headers = options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {};
		this.timeoutMs = options.timeoutMs ?? 30_000;
		this.fetchImpl = options.fetchImpl;
	}

	async score(query: string, documents: readonly string[], signal?: AbortSignal): Promise<number[]> {
		if (documents.length === 0) return [];
		const json = (await postJson(
			`${this.baseUrl}/rerank`,
			{ model: this.model, query, documents: [...documents], top_n: documents.length, return_documents: false },
			{
				headers: this.headers,
				timeoutMs: this.timeoutMs,
				...(signal ? { signal } : {}),
				...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
			},
		)) as { results?: Array<{ index?: unknown; relevance_score?: unknown }>; data?: Array<{ index?: unknown; relevance_score?: unknown }> };
		const rows = json.results ?? json.data ?? [];
		const out = new Array<number>(documents.length).fill(Number.NaN);
		for (const r of rows) {
			if (typeof r.index === "number" && typeof r.relevance_score === "number" && r.index >= 0 && r.index < documents.length) {
				out[r.index] = Math.max(0, Math.min(1, r.relevance_score));
			}
		}
		if (out.some((s) => !Number.isFinite(s))) throw new EmbeddingError("UPSTREAM", `reranker ${this.id}: response is missing documents`);
		return out;
	}

	async health(signal?: AbortSignal): Promise<BackendHealth> {
		const res = await probe(`${this.baseUrl}/models`, {
			headers: this.headers,
			timeoutMs: 5_000,
			...(signal ? { signal } : {}),
			...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
		});
		// Many rerank APIs have no /models route; reachability (any HTTP answer) is what matters here.
		return res.status > 0 ? { ok: true } : { ok: false, detail: res.detail ?? "unreachable" };
	}
}

// ── Service: cache + self-test + answer threshold ───────────────────────────────────────────────

/** Same-topic pairs whose document does NOT answer the query (neutral general knowledge). */
export const RERANK_NEAR_MISS_PAIRS: ReadonlyArray<readonly [string, string]> = [
	["how tall is the tallest mountain in Africa", SELF_TEST_PAIRS[0]![1]],
	["how many moons does Mars have", SELF_TEST_PAIRS[2]![1]],
	["how fast can a lion run", SELF_TEST_PAIRS[6]![1]],
	["what year did Michelangelo die", SELF_TEST_PAIRS[9]![1]],
	["what currency is used in South Korea", SELF_TEST_PAIRS[11]![1]],
	["how often should I water a fern", SELF_TEST_PAIRS[12]![1]],
	["how to say thank you in French", SELF_TEST_PAIRS[14]![1]],
	["how many players on a basketball team", SELF_TEST_PAIRS[16]![1]],
	["what do koalas eat", SELF_TEST_PAIRS[17]![1]],
	["smallest desert in the world", SELF_TEST_PAIRS[19]![1]],
];

export interface RerankSelfTestResult {
	/**
	 * Highest level non-answers reach (max of near-miss p90 and unrelated p95). Scores above it are trustworthy
	 * "this answers the query" evidence; scores below it carry no information (paraphrased answers land there too).
	 */
	readonly answerThreshold: number;
	readonly positiveMedian: number;
	readonly positiveP10: number;
	readonly nearMissP90: number;
	readonly unrelatedP95: number;
	/** Fraction of true answers scoring above `answerThreshold`. */
	readonly separation: number;
	readonly passed: boolean;
	readonly ms: number;
}

function quantile(values: readonly number[], q: number): number {
	const sorted = [...values].sort((a, b) => a - b);
	if (sorted.length === 0) return 0;
	const pos = (sorted.length - 1) * q;
	const lo = Math.floor(pos);
	const hi = Math.ceil(pos);
	return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
}

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

export async function runRerankSelfTest(score: (query: string, docs: readonly string[]) => Promise<readonly number[]>): Promise<RerankSelfTestResult> {
	const started = performance.now();
	const positives: number[] = [];
	const unrelated: number[] = [];
	for (let i = 0; i < SELF_TEST_PAIRS.length; i++) {
		const [q, d] = SELF_TEST_PAIRS[i]!;
		const other = SELF_TEST_PAIRS[(i + 7) % SELF_TEST_PAIRS.length]![1];
		const [pos, neg] = await score(q, [d, other]);
		positives.push(pos ?? 0);
		unrelated.push(neg ?? 0);
	}
	const near: number[] = [];
	for (const [q, d] of RERANK_NEAR_MISS_PAIRS) near.push((await score(q, [d]))[0] ?? 0);
	const positiveMedian = quantile(positives, 0.5);
	const positiveP10 = quantile(positives, 0.1);
	const nearMissP90 = quantile(near, 0.9);
	const unrelatedP95 = quantile(unrelated, 0.95);
	const answerThreshold = Math.max(nearMissP90, unrelatedP95, 0.01);
	const separation = positives.filter((p) => p > answerThreshold).length / positives.length;
	return {
		answerThreshold: round3(answerThreshold),
		positiveMedian: round3(positiveMedian),
		positiveP10: round3(positiveP10),
		nearMissP90: round3(nearMissP90),
		unrelatedP95: round3(unrelatedP95),
		separation: round3(separation),
		// Usable as positive evidence when non-answers stay low and typical answers clear them comfortably.
		passed: answerThreshold < 0.5 && positiveMedian >= Math.max(0.15, 2 * answerThreshold) && separation >= 0.5,
		ms: Math.round(performance.now() - started),
	};
}

export interface RerankServiceOptions {
	readonly backend: RerankBackend;
	readonly cacheSize?: number;
	readonly maxDocuments?: number;
	readonly maxDocumentChars?: number;
	/** Explicit threshold (EMBED_RERANK_ANSWER_THRESHOLD); otherwise measured by `selfTest()`. */
	readonly answerThreshold?: number;
}

export class RerankService {
	readonly backend: RerankBackend;
	private readonly cache = new Map<string, number>();
	private readonly cacheSize: number;
	readonly maxDocuments: number;
	private readonly maxDocumentChars: number;
	private threshold: number | undefined;
	private readonly thresholdPinned: boolean;
	private quarantined: string | undefined;
	private lastSelfTest: RerankSelfTestResult | undefined;

	constructor(options: RerankServiceOptions) {
		this.backend = options.backend;
		this.cacheSize = Math.max(0, options.cacheSize ?? 50_000);
		this.maxDocuments = Math.max(1, options.maxDocuments ?? 128);
		this.maxDocumentChars = Math.max(64, options.maxDocumentChars ?? 4_000);
		this.threshold = options.answerThreshold;
		this.thresholdPinned = options.answerThreshold !== undefined;
	}

	get model(): string {
		return this.backend.model;
	}

	get answerThreshold(): number | undefined {
		return this.threshold;
	}

	/** Implements the harness `Reranker` contract, so the service can also be injected in-process. */
	async rerank(query: string, documents: readonly string[], options?: { readonly signal?: AbortSignal }): Promise<number[]> {
		const signal = options?.signal;
		if (this.quarantined) throw new EmbeddingError("UNAVAILABLE", `reranker quarantined: ${this.quarantined}`);
		if (typeof query !== "string" || query.trim().length === 0) throw new EmbeddingError("BAD_REQUEST", "query must be a non-empty string");
		if (documents.length > this.maxDocuments) throw new EmbeddingError("BAD_REQUEST", `at most ${this.maxDocuments} documents per request`);
		const docs = documents.map((d) => {
			if (typeof d !== "string") throw new EmbeddingError("BAD_REQUEST", "documents must be strings");
			return d.length > this.maxDocumentChars ? d.slice(0, this.maxDocumentChars) : d;
		});
		const qDigest = crypto.createHash("sha256").update(query).digest("base64url");
		const keys = docs.map((d) => `${qDigest}\0${crypto.createHash("sha256").update(d).digest("base64url")}`);
		const out = new Array<number>(docs.length);
		const missing: number[] = [];
		keys.forEach((k, i) => {
			const hit = this.cacheSize > 0 ? this.cache.get(k) : undefined;
			if (hit !== undefined) out[i] = hit;
			else missing.push(i);
		});
		for (let start = 0; start < missing.length; start += this.backend.maxBatchSize) {
			const idx = missing.slice(start, start + this.backend.maxBatchSize);
			const scores = await this.backend.score(
				query,
				idx.map((i) => docs[i]!),
				signal,
			);
			idx.forEach((i, n) => {
				out[i] = scores[n] ?? 0;
				if (this.cacheSize > 0) {
					this.cache.set(keys[i]!, out[i]!);
					if (this.cache.size > this.cacheSize) this.cache.delete(this.cache.keys().next().value!);
				}
			});
		}
		return out;
	}

	async selfTest(options: { readonly enforce?: boolean } = {}): Promise<RerankSelfTestResult> {
		const result = await runRerankSelfTest((q, docs) => this.backend.score(q, docs));
		this.lastSelfTest = result;
		if (!this.thresholdPinned) this.threshold = result.answerThreshold;
		this.quarantined = !result.passed && (options.enforce ?? true) ? `self-test failed (separation ${result.separation})` : undefined;
		return result;
	}

	async health(): Promise<{ ok: boolean; model: string; answerThreshold?: number; quarantined?: string; selfTest?: RerankSelfTestResult; detail?: string }> {
		const h = await this.backend.health(AbortSignal.timeout(5_000)).catch((e: unknown) => ({ ok: false, detail: errorText(e) }));
		return {
			ok: h.ok && !this.quarantined,
			model: this.model,
			...(this.threshold !== undefined ? { answerThreshold: this.threshold } : {}),
			...(this.quarantined ? { quarantined: this.quarantined } : {}),
			...(this.lastSelfTest ? { selfTest: this.lastSelfTest } : {}),
			...(h.detail ? { detail: h.detail } : {}),
		};
	}

	async close(): Promise<void> {
		await this.backend.close?.();
		this.cache.clear();
	}
}
