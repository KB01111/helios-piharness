/**
 * In-process ONNX backend running EmbeddingGemma 2 (default `onnx-community/embeddinggemma-2-ONNX`) through
 * Transformers.js / ONNX Runtime. Works offline after the first model download (cached under `cacheDir`).
 *
 * Devices: `cpu` (default; fastest for small batches on Ryzen AI 300 laptops), `dml` (DirectML on the
 * Radeon iGPU, Windows), `cuda`, `webgpu`. The AMD XDNA NPU is reached through the `npu` backend
 * (FastFlowLM), because ONNX Runtime's VitisAI execution provider is not shipped in onnxruntime-node.
 *
 * `@huggingface/transformers` is an optional dependency and is loaded lazily, so the harness itself never
 * pays for it unless this backend is configured.
 */

import { resolveModelInfo } from "../models.js";
import { EmbeddingError, type BackendEmbedOptions, type BackendHealth, type EmbeddingBackend } from "../types.js";
import { errorText } from "./http-util.js";

export type OnnxDevice = "cpu" | "dml" | "cuda" | "webgpu" | "auto";
export type OnnxDtype = "fp32" | "fp16" | "q8" | "q4" | "q4f16" | "int8" | "uint8";

export interface OnnxBackendOptions {
	readonly id: string;
	readonly model?: string;
	readonly device?: OnnxDevice;
	readonly dtype?: OnnxDtype;
	/** Local model cache directory (Transformers.js `env.cacheDir`). */
	readonly cacheDir?: string;
	/** Refuse network downloads (model must already be in `cacheDir`). */
	readonly offline?: boolean;
	readonly maxBatchSize?: number;
	readonly space?: string;
	/** Test seam: replaces the dynamic `import("@huggingface/transformers")`. */
	readonly loadTransformers?: () => Promise<TransformersModule>;
}

interface FeatureTensor {
	tolist(): number[][];
	dispose?(): void;
}
type FeatureExtractor = ((texts: string[], options: { pooling: "mean"; normalize: boolean }) => Promise<FeatureTensor>) & {
	dispose?: () => Promise<void>;
};
export interface TransformersModule {
	pipeline(task: "feature-extraction", model: string, options: { device?: string; dtype?: string }): Promise<FeatureExtractor>;
	env: { cacheDir?: string; allowRemoteModels?: boolean };
}

const DEFAULT_MODEL = "onnx-community/embeddinggemma-2-ONNX";
/** Texts per ONNX forward pass inside one (length-sorted) backend call. */
const ONNX_MICRO_BATCH = 8;

export class OnnxBackend implements EmbeddingBackend {
	readonly id: string;
	readonly kind = "onnx" as const;
	readonly model: string;
	readonly space: string;
	readonly nativeDimensions: number | undefined;
	readonly wantsPrompts: boolean;
	readonly maxBatchSize: number;
	private readonly device: OnnxDevice;
	private readonly dtype: OnnxDtype;
	private readonly options: OnnxBackendOptions;
	private extractor: Promise<FeatureExtractor> | undefined;
	/** ORT sessions are not re-entrant per pipeline; inference calls are serialized through this chain. */
	private queue: Promise<unknown> = Promise.resolve();
	private loadError: string | undefined;

	constructor(options: OnnxBackendOptions) {
		this.id = options.id;
		this.model = options.model?.trim() || DEFAULT_MODEL;
		const info = resolveModelInfo(this.model, options.space ? { space: options.space } : undefined);
		this.space = info.space;
		this.nativeDimensions = info.nativeDimensions;
		this.wantsPrompts = info.promptFamily !== "none";
		this.device = options.device ?? "cpu";
		this.dtype = options.dtype ?? "q4";
		this.maxBatchSize = Math.max(1, options.maxBatchSize ?? 128);
		this.options = options;
	}

	/** Starts loading the model (download on first use) without waiting for a request. */
	warm(): Promise<void> {
		return this.getExtractor().then(() => undefined);
	}

	private getExtractor(): Promise<FeatureExtractor> {
		if (!this.extractor) {
			this.extractor = (async () => {
				let mod: TransformersModule;
				try {
					mod = this.options.loadTransformers
						? await this.options.loadTransformers()
						: ((await import(/* optional dependency */ "@huggingface/transformers" as string)) as TransformersModule);
				} catch (error) {
					throw new EmbeddingError(
						"MISCONFIGURED",
						`backend ${this.id}: @huggingface/transformers is not installed (npm install @huggingface/transformers): ${errorText(error)}`,
						{ cause: error },
					);
				}
				if (this.options.cacheDir) mod.env.cacheDir = this.options.cacheDir;
				if (this.options.offline) mod.env.allowRemoteModels = false;
				return mod.pipeline("feature-extraction", this.model, {
					...(this.device === "auto" ? {} : { device: this.device }),
					dtype: this.dtype,
				});
			})();
			this.extractor.then(
				() => {
					this.loadError = undefined;
				},
				(error: unknown) => {
					// Allow a later retry (e.g. the network came back) instead of caching the failure forever.
					this.loadError = errorText(error);
					this.extractor = undefined;
				},
			);
		}
		return this.extractor;
	}

	async embed(texts: readonly string[], options: BackendEmbedOptions): Promise<Float32Array[]> {
		if (texts.length === 0) return [];
		if (options.signal?.aborted) throw new EmbeddingError("TIMEOUT", `backend ${this.id}: request aborted`);
		const run = async (): Promise<Float32Array[]> => {
			const extractor = await this.getExtractor();
			// Each micro-batch is padded to its longest text, so length-sorted micro-batches waste far less compute
			// on mixed-length input. Results are restored to input order.
			const order = texts.map((t, i) => ({ i, len: t.length })).sort((a, b) => a.len - b.len);
			const out = new Array<Float32Array>(texts.length);
			for (let start = 0; start < order.length; start += ONNX_MICRO_BATCH) {
				const slice = order.slice(start, start + ONNX_MICRO_BATCH);
				const tensor = await extractor(
					slice.map((o) => texts[o.i]!),
					{ pooling: "mean", normalize: true },
				);
				try {
					const rows = tensor.tolist();
					if (rows.length !== slice.length) {
						throw new EmbeddingError("UPSTREAM", `backend ${this.id}: expected ${slice.length} embeddings, got ${rows.length}`);
					}
					rows.forEach((row, n) => {
						out[slice[n]!.i] = Float32Array.from(row);
					});
				} finally {
					tensor.dispose?.();
				}
			}
			return out;
		};
		const result = this.queue.then(run, run);
		this.queue = result.catch(() => undefined);
		return result;
	}

	async health(): Promise<BackendHealth> {
		if (this.loadError) return { ok: false, detail: this.loadError };
		if (!this.extractor) return { ok: true, detail: "model loads on first request" };
		try {
			await this.extractor;
			return { ok: true, detail: `${this.model} (${this.device}, ${this.dtype})` };
		} catch (error) {
			return { ok: false, detail: errorText(error) };
		}
	}

	async close(): Promise<void> {
		const pending = this.extractor;
		this.extractor = undefined;
		if (!pending) return;
		try {
			await (await pending).dispose?.();
		} catch {
			// Disposal failures are irrelevant at shutdown.
		}
	}
}
