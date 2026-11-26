/**
 * Google Gemini native embedding backend (`models/{model}:batchEmbedContents`).
 *
 * Uses Gemini's own `taskType` (`RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`) instead of text prompts and
 * server-side Matryoshka truncation via `outputDimensionality`. The API key travels in the `x-goog-api-key`
 * header, never in the URL.
 */

import { resolveModelInfo } from "../models.js";
import { EmbeddingError, type BackendEmbedOptions, type BackendHealth, type EmbeddingBackend } from "../types.js";
import { postJson, probe, toFloat32 } from "./http-util.js";

export interface GeminiBackendOptions {
	readonly id: string;
	readonly apiKey: string;
	/** Default `gemini-embedding-001`. */
	readonly model?: string;
	readonly baseUrl?: string;
	readonly timeoutMs?: number;
	readonly space?: string;
	readonly fetchImpl?: typeof fetch;
}

export class GeminiBackend implements EmbeddingBackend {
	readonly id: string;
	readonly kind = "gemini" as const;
	readonly model: string;
	readonly space: string;
	readonly nativeDimensions: number | undefined;
	readonly wantsPrompts = false;
	/** batchEmbedContents accepts at most 100 requests per call. */
	readonly maxBatchSize = 100;
	private readonly apiKey: string;
	private readonly baseUrl: string;
	private readonly timeoutMs: number;
	private readonly fetchImpl: typeof fetch | undefined;

	constructor(options: GeminiBackendOptions) {
		if (!options.apiKey.trim()) throw new EmbeddingError("MISCONFIGURED", `backend ${options.id}: Gemini API key is required`);
		this.id = options.id;
		this.model = (options.model ?? "gemini-embedding-001").replace(/^models\//, "");
		const info = resolveModelInfo(this.model, options.space ? { space: options.space } : undefined);
		this.space = info.space;
		this.nativeDimensions = info.nativeDimensions;
		this.apiKey = options.apiKey;
		this.baseUrl = (options.baseUrl ?? "https://generativelanguage.googleapis.com/v1beta").replace(/\/+$/, "");
		this.timeoutMs = options.timeoutMs ?? 30_000;
		this.fetchImpl = options.fetchImpl;
	}

	async embed(texts: readonly string[], options: BackendEmbedOptions): Promise<Float32Array[]> {
		if (texts.length === 0) return [];
		const taskType = options.inputType === "query" ? "RETRIEVAL_QUERY" : "RETRIEVAL_DOCUMENT";
		const json = (await postJson(
			`${this.baseUrl}/models/${encodeURIComponent(this.model)}:batchEmbedContents`,
			{
				requests: texts.map((text) => ({
					model: `models/${this.model}`,
					content: { parts: [{ text }] },
					taskType,
					...(options.dimensions ? { outputDimensionality: options.dimensions } : {}),
				})),
			},
			{
				headers: { "x-goog-api-key": this.apiKey },
				timeoutMs: this.timeoutMs,
				...(options.signal ? { signal: options.signal } : {}),
				...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
			},
		)) as { embeddings?: ReadonlyArray<{ values?: unknown }> };
		const embeddings = json?.embeddings;
		if (!Array.isArray(embeddings) || embeddings.length !== texts.length) {
			throw new EmbeddingError("UPSTREAM", `backend ${this.id}: expected ${texts.length} embeddings, got ${embeddings?.length ?? 0}`);
		}
		return embeddings.map((e) => toFloat32(e.values, `backend ${this.id}`));
	}

	async health(signal?: AbortSignal): Promise<BackendHealth> {
		const res = await probe(`${this.baseUrl}/models/${encodeURIComponent(this.model)}`, {
			headers: { "x-goog-api-key": this.apiKey },
			timeoutMs: Math.min(this.timeoutMs, 5_000),
			...(signal ? { signal } : {}),
			...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
		});
		return res.ok ? { ok: true } : { ok: false, detail: res.detail ?? `HTTP ${res.status}` };
	}
}
