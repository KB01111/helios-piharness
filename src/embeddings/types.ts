/**
 * Shared types of the Helios embedding service (`src/embeddings/`).
 *
 * An embedding *backend* turns already-formatted texts into vectors on one piece of hardware or one API
 * (AMD NPU via FastFlowLM, in-process ONNX, OpenAI-compatible cloud/local servers, Google Gemini).
 * The *router* in front of them applies task prompts, Matryoshka truncation, caching, circuit breaking and
 * same-space fallback. The HTTP server exposes the router as an OpenAI-compatible `/v1/embeddings` API.
 */

/** Asymmetric retrieval role of a text. Queries and documents get different task prompts on instruction-tuned models. */
export type EmbeddingInputType = "query" | "document";

export type EmbeddingBackendKind = "npu" | "onnx" | "openai" | "gemini";

export interface BackendEmbedOptions {
	readonly inputType: EmbeddingInputType;
	/** Requested output dimensionality; backends that support server-side truncation may use it, others ignore it. */
	readonly dimensions?: number;
	readonly signal?: AbortSignal;
}

export interface BackendHealth {
	readonly ok: boolean;
	readonly detail?: string;
}

export interface EmbeddingBackend {
	/** Unique id within one router (e.g. `npu`, `onnx`, `cloud`). */
	readonly id: string;
	readonly kind: EmbeddingBackendKind;
	/** Model identifier sent to / loaded by the backend (e.g. `embed-gemma:300m`). */
	readonly model: string;
	/**
	 * Embedding space this backend produces. Vectors are only comparable within one space, so the router only
	 * falls back between backends that share it (e.g. FLM `embed-gemma:300m` on the NPU and the ONNX export of
	 * `google/embeddinggemma-300m` on the CPU).
	 */
	readonly space: string;
	/** Native output dimensionality when known up front. */
	readonly nativeDimensions: number | undefined;
	/**
	 * True when the backend expects the router to prepend task prompts (`task: search result | query: ...`).
	 * False when the provider applies its own task handling (Gemini `taskType`, OpenAI models without prompts).
	 */
	readonly wantsPrompts: boolean;
	/** Largest number of texts sent in one backend call; the router splits bigger requests. */
	readonly maxBatchSize: number;
	embed(texts: readonly string[], options: BackendEmbedOptions): Promise<Float32Array[]>;
	health(signal?: AbortSignal): Promise<BackendHealth>;
	close?(): Promise<void>;
}

/** Cosine calibration of one embedding space: `floor` ~ unrelated-text similarity, `ceiling` ~ near-paraphrase. */
export interface SimilarityCalibration {
	readonly floor: number;
	readonly ceiling: number;
}

export class EmbeddingError extends Error {
	readonly code: "BAD_REQUEST" | "UNAVAILABLE" | "UPSTREAM" | "TIMEOUT" | "MISCONFIGURED";
	readonly status: number;
	constructor(code: EmbeddingError["code"], message: string, options?: { readonly cause?: unknown }) {
		super(message, options);
		this.name = "EmbeddingError";
		this.code = code;
		this.status =
			code === "BAD_REQUEST" ? 400 : code === "TIMEOUT" ? 504 : code === "MISCONFIGURED" ? 500 : code === "UPSTREAM" ? 502 : 503;
	}
}
