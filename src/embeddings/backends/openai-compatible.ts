/**
 * OpenAI-compatible `/v1/embeddings` backend.
 *
 * Covers the AMD NPU (FastFlowLM `flm serve <llm> --embed 1`, which exposes `embed-gemma:300m` on XDNA2),
 * AMD Lemonade Server, Ollama, llama.cpp, vLLM, HF TEI, OpenAI and OpenRouter: all speak the same protocol.
 */

import { resolveModelInfo, type PromptFamily } from "../models.js";
import { EmbeddingError, type BackendEmbedOptions, type BackendHealth, type EmbeddingBackend, type EmbeddingBackendKind } from "../types.js";
import { base64ToFloat32, postJson, probe, toFloat32 } from "./http-util.js";

export interface OpenAICompatibleBackendOptions {
	readonly id: string;
	/** `npu` for FastFlowLM/Lemonade on the Ryzen AI NPU, `openai` for any other OpenAI-compatible endpoint. */
	readonly kind?: Extract<EmbeddingBackendKind, "npu" | "openai">;
	/** Base URL including the version segment, e.g. `http://127.0.0.1:52625/v1` or `https://api.openai.com/v1`. */
	readonly baseUrl: string;
	readonly model: string;
	readonly apiKey?: string;
	readonly timeoutMs?: number;
	readonly maxBatchSize?: number;
	/** Send `dimensions` upstream (OpenAI text-embedding-3 supports it; most local servers reject or ignore it). */
	readonly sendDimensions?: boolean;
	/** Override the embedding space / prompt family derived from the model id. */
	readonly space?: string;
	readonly promptFamily?: PromptFamily;
	/** Whether the router should prepend task prompts. Default: true for known instruction-tuned families. */
	readonly wantsPrompts?: boolean;
	readonly extraHeaders?: Record<string, string>;
	readonly fetchImpl?: typeof fetch;
}

interface OpenAIEmbeddingResponse {
	readonly data?: ReadonlyArray<{ readonly index?: number; readonly embedding?: unknown }>;
}

export class OpenAICompatibleBackend implements EmbeddingBackend {
	readonly id: string;
	readonly kind: EmbeddingBackendKind;
	readonly model: string;
	readonly space: string;
	readonly nativeDimensions: number | undefined;
	readonly wantsPrompts: boolean;
	readonly maxBatchSize: number;
	readonly promptFamily: PromptFamily;
	private readonly baseUrl: string;
	private readonly headers: Record<string, string>;
	private readonly timeoutMs: number;
	private readonly sendDimensions: boolean;
	private readonly fetchImpl: typeof fetch | undefined;

	constructor(options: OpenAICompatibleBackendOptions) {
		if (!options.baseUrl.trim()) throw new EmbeddingError("MISCONFIGURED", `backend ${options.id}: baseUrl is required`);
		if (!options.model.trim()) throw new EmbeddingError("MISCONFIGURED", `backend ${options.id}: model is required`);
		const info = resolveModelInfo(options.model, {
			...(options.space ? { space: options.space } : {}),
			...(options.promptFamily ? { promptFamily: options.promptFamily } : {}),
		});
		this.id = options.id;
		this.kind = options.kind ?? "openai";
		this.model = options.model;
		this.space = info.space;
		this.nativeDimensions = info.nativeDimensions;
		this.promptFamily = info.promptFamily;
		this.wantsPrompts = options.wantsPrompts ?? info.promptFamily !== "none";
		this.maxBatchSize = Math.max(1, options.maxBatchSize ?? 64);
		this.baseUrl = options.baseUrl.replace(/\/+$/, "");
		this.timeoutMs = options.timeoutMs ?? 30_000;
		this.sendDimensions = options.sendDimensions ?? false;
		this.fetchImpl = options.fetchImpl;
		this.headers = {
			...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}),
			...options.extraHeaders,
		};
	}

	async embed(texts: readonly string[], options: BackendEmbedOptions): Promise<Float32Array[]> {
		if (texts.length === 0) return [];
		const body: Record<string, unknown> = { model: this.model, input: [...texts], encoding_format: "float" };
		if (this.sendDimensions && options.dimensions) body.dimensions = options.dimensions;
		const json = (await postJson(`${this.baseUrl}/embeddings`, body, {
			headers: this.headers,
			timeoutMs: this.timeoutMs,
			...(options.signal ? { signal: options.signal } : {}),
			...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
		})) as OpenAIEmbeddingResponse;
		const data = Array.isArray(json?.data) ? json.data : undefined;
		if (!data || data.length !== texts.length) {
			throw new EmbeddingError("UPSTREAM", `backend ${this.id}: expected ${texts.length} embeddings, got ${data?.length ?? 0}`);
		}
		const out: Array<Float32Array | undefined> = new Array(texts.length);
		data.forEach((item, position) => {
			const index = item.index === undefined ? position : item.index;
			if (!Number.isInteger(index) || index < 0 || index >= texts.length || out[index]) {
				throw new EmbeddingError("UPSTREAM", `backend ${this.id}: invalid or duplicate embedding index ${index}`);
			}
			out[index] =
				typeof item.embedding === "string"
					? base64ToFloat32(item.embedding, `backend ${this.id}`)
					: toFloat32(item.embedding, `backend ${this.id}`);
		});
		return out as Float32Array[];
	}

	async health(signal?: AbortSignal): Promise<BackendHealth> {
		const res = await probe(`${this.baseUrl}/models`, {
			headers: this.headers,
			timeoutMs: Math.min(this.timeoutMs, 5_000),
			...(signal ? { signal } : {}),
			...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
		});
		return res.ok ? { ok: true } : { ok: false, detail: res.detail ?? `HTTP ${res.status}` };
	}
}
