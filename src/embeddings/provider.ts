/**
 * Harness-side `EmbeddingProvider` implementations for `SqliteMemoryService` hybrid retrieval.
 *
 *  - `HttpEmbeddingProvider`: talks to the Helios embedding server (`npm run embed:server`) or directly to any
 *    OpenAI-compatible embeddings endpoint (OpenAI, OpenRouter, Lemonade, Ollama, FastFlowLM, TEI, vLLM).
 *  - `RouterEmbeddingProvider`: in-process adapter over an `EmbeddingRouter` (no HTTP hop) for library embedding.
 *
 * Vectors are stored under `model`. For the Helios server that id is `<space>@<dims>`, and every response is
 * checked against it: if the server is reconfigured to another embedding space the provider throws (memory search
 * then degrades to lexical ranking) instead of silently mixing incomparable vectors.
 */

import type { EmbeddingProvider } from "../contracts/app-capabilities.js";
import { errorText, postJson, redactUrl, toFloat32, base64ToFloat32, withTimeout } from "./backends/http-util.js";
import { resolveModelInfo } from "./models.js";
import type { EmbeddingRouter } from "./router.js";
import { EmbeddingError, type SimilarityCalibration } from "./types.js";

export interface HttpEmbeddingProviderOptions {
	/** Base URL including `/v1`, e.g. `http://127.0.0.1:18765/v1`. */
	readonly baseUrl: string;
	readonly apiKey?: string;
	/**
	 * Upstream model name. Required for plain OpenAI-compatible endpoints; optional for the Helios server,
	 * which reports its own model id from `/v1/models`.
	 */
	readonly model?: string;
	/** Matryoshka output size requested from the server (also part of the stored model id). */
	readonly dimensions?: number;
	readonly timeoutMs?: number;
	/** Retries on network/5xx errors (default 2). */
	readonly retries?: number;
	readonly maxBatchSize?: number;
	/** Overrides the calibration announced by the server or derived from the model. */
	readonly similarity?: SimilarityCalibration;
	readonly fetchImpl?: typeof fetch;
	readonly sleep?: (ms: number) => Promise<void>;
}

interface HeliosModelEntry {
	readonly id?: unknown;
	readonly space?: unknown;
	readonly dimensions?: unknown;
	readonly calibration?: { readonly floor?: unknown; readonly ceiling?: unknown } | null;
}

function validCalibration(raw: HeliosModelEntry["calibration"]): SimilarityCalibration | undefined {
	const floor = raw?.floor;
	const ceiling = raw?.ceiling;
	return typeof floor === "number" && typeof ceiling === "number" && ceiling > floor ? { floor, ceiling } : undefined;
}

export class HttpEmbeddingProvider implements EmbeddingProvider {
	readonly model: string;
	readonly similarity?: SimilarityCalibration;
	/** True when the endpoint is a Helios embedding server (supports `input_type`, reports its space). */
	readonly helios: boolean;
	private readonly baseUrl: string;
	private readonly upstreamModel: string;
	private readonly headers: Record<string, string>;
	private readonly dimensions: number | undefined;
	private readonly timeoutMs: number;
	private readonly retries: number;
	private readonly maxBatchSize: number;
	private readonly fetchImpl: typeof fetch | undefined;
	private readonly sleep: (ms: number) => Promise<void>;

	private constructor(init: {
		readonly options: HttpEmbeddingProviderOptions;
		readonly model: string;
		readonly upstreamModel: string;
		readonly helios: boolean;
		readonly similarity: SimilarityCalibration | undefined;
	}) {
		const { options } = init;
		this.baseUrl = options.baseUrl.replace(/\/+$/, "");
		this.model = init.model;
		this.upstreamModel = init.upstreamModel;
		this.helios = init.helios;
		if (init.similarity) this.similarity = init.similarity;
		this.headers = options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {};
		this.dimensions = options.dimensions;
		this.timeoutMs = options.timeoutMs ?? 30_000;
		this.retries = Math.max(0, options.retries ?? 2);
		this.maxBatchSize = Math.max(1, options.maxBatchSize ?? 128);
		this.fetchImpl = options.fetchImpl;
		this.sleep = options.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
	}

	/**
	 * Resolves the endpoint's identity. Helios servers are detected through `/v1/models` (`space` field); other
	 * OpenAI-compatible endpoints need `options.model`. Throws `EmbeddingError` when neither works.
	 */
	static async connect(options: HttpEmbeddingProviderOptions): Promise<HttpEmbeddingProvider> {
		const baseUrl = options.baseUrl.replace(/\/+$/, "");
		let entry: HeliosModelEntry | undefined;
		const { signal, dispose } = withTimeout(undefined, Math.min(options.timeoutMs ?? 30_000, 10_000));
		try {
			const res = await (options.fetchImpl ?? fetch)(`${baseUrl}/models`, {
				headers: options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {},
				signal,
			});
			if (res.ok) {
				const json = (await res.json()) as { data?: HeliosModelEntry[] };
				entry = Array.isArray(json?.data) ? json.data.find((m) => typeof m?.space === "string") : undefined;
			} else {
				await res.body?.cancel().catch(() => undefined);
				if (res.status === 401 || res.status === 403) {
					throw new EmbeddingError("MISCONFIGURED", `embedding endpoint ${redactUrl(baseUrl)} rejected the API key (HTTP ${res.status})`);
				}
			}
		} catch (error) {
			if (error instanceof EmbeddingError) throw error;
			if (!options.model) {
				throw new EmbeddingError("UNAVAILABLE", `embedding endpoint ${redactUrl(baseUrl)} unreachable: ${errorText(error)}`, { cause: error });
			}
		} finally {
			dispose();
		}

		if (entry && typeof entry.space === "string") {
			const space = entry.space;
			const dims = options.dimensions ?? (typeof entry.dimensions === "number" ? entry.dimensions : undefined);
			const model = dims ? `${space}@${dims}` : typeof entry.id === "string" ? entry.id : space;
			return new HttpEmbeddingProvider({
				options,
				model,
				upstreamModel: options.model ?? (typeof entry.id === "string" ? entry.id : space),
				helios: true,
				similarity: options.similarity ?? validCalibration(entry.calibration) ?? resolveModelInfo(space).calibration,
			});
		}
		if (!options.model) {
			throw new EmbeddingError(
				"MISCONFIGURED",
				`${redactUrl(baseUrl)} is not a Helios embedding server; set EMBEDDINGS_MODEL to the upstream embedding model`,
			);
		}
		const info = resolveModelInfo(options.model);
		return new HttpEmbeddingProvider({
			options,
			model: `${info.space}@${options.dimensions ?? info.nativeDimensions ?? "native"}`,
			upstreamModel: options.model,
			helios: false,
			similarity: options.similarity ?? info.calibration,
		});
	}

	async embed(
		texts: readonly string[],
		options?: { readonly signal?: AbortSignal; readonly inputType?: "query" | "document" },
	): Promise<ReadonlyArray<ArrayLike<number>>> {
		const out: Float32Array[] = [];
		for (let start = 0; start < texts.length; start += this.maxBatchSize) {
			out.push(...(await this.embedBatch(texts.slice(start, start + this.maxBatchSize), options)));
		}
		return out;
	}

	private async embedBatch(
		texts: readonly string[],
		options?: { readonly signal?: AbortSignal; readonly inputType?: "query" | "document" },
	): Promise<Float32Array[]> {
		if (texts.length === 0) return [];
		const body: Record<string, unknown> = { model: this.upstreamModel, input: [...texts], encoding_format: "base64" };
		if (this.dimensions) body.dimensions = this.dimensions;
		if (this.helios && options?.inputType) body.input_type = options.inputType;

		let lastError: unknown;
		for (let attempt = 0; attempt <= this.retries; attempt++) {
			if (attempt > 0) await this.sleep(Math.min(2_000, 150 * 2 ** (attempt - 1)));
			try {
				const json = (await postJson(`${this.baseUrl}/embeddings`, body, {
					headers: this.headers,
					timeoutMs: this.timeoutMs,
					...(options?.signal ? { signal: options.signal } : {}),
					...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
				})) as { model?: unknown; data?: Array<{ index?: number; embedding?: unknown }> };
				if (this.helios && typeof json.model === "string" && json.model !== this.model) {
					throw new EmbeddingError(
						"MISCONFIGURED",
						`embedding server now serves ${json.model} but stored vectors use ${this.model}; restart the harness to re-embed`,
					);
				}
				const data = Array.isArray(json.data) ? json.data : [];
				if (data.length !== texts.length) {
					throw new EmbeddingError("UPSTREAM", `expected ${texts.length} embeddings, got ${data.length}`);
				}
				const vectors: Float32Array[] = new Array(texts.length);
				data.forEach((item, pos) => {
					const idx = typeof item.index === "number" ? item.index : pos;
					vectors[idx] =
						typeof item.embedding === "string" ? base64ToFloat32(item.embedding, "embeddings") : toFloat32(item.embedding, "embeddings");
				});
				if (vectors.some((v) => !v)) throw new EmbeddingError("UPSTREAM", "embedding response is missing indices");
				return vectors;
			} catch (error) {
				lastError = error;
				const retryable = error instanceof EmbeddingError ? error.code === "UNAVAILABLE" || error.code === "TIMEOUT" : true;
				if (!retryable || options?.signal?.aborted) break;
			}
		}
		throw lastError;
	}
}

/** In-process `EmbeddingProvider` over an `EmbeddingRouter` (no HTTP server needed). */
export class RouterEmbeddingProvider implements EmbeddingProvider {
	readonly model: string;
	readonly similarity?: SimilarityCalibration;
	private readonly router: EmbeddingRouter;
	private readonly dimensions: number | undefined;

	constructor(router: EmbeddingRouter, options?: { readonly dimensions?: number; readonly similarity?: SimilarityCalibration }) {
		this.router = router;
		this.dimensions = options?.dimensions ?? router.defaultDimensions;
		this.model = router.modelId(this.dimensions);
		const calibration = options?.similarity ?? router.calibration;
		if (calibration) this.similarity = calibration;
	}

	async embed(
		texts: readonly string[],
		options?: { readonly signal?: AbortSignal; readonly inputType?: "query" | "document" },
	): Promise<ReadonlyArray<ArrayLike<number>>> {
		const result = await this.router.embed({
			texts,
			inputType: options?.inputType ?? "document",
			...(this.dimensions !== undefined ? { dimensions: this.dimensions } : {}),
			...(options?.signal ? { signal: options.signal } : {}),
		});
		return result.vectors;
	}
}
