/**
 * EmbeddingRouter: the brain of the embedding service.
 *
 *  - Ordered backends (e.g. NPU first, CPU ONNX second). Falls back ONLY between backends of the same embedding
 *    space, so a query embedded on a fallback is still comparable with documents embedded on the primary.
 *  - Asymmetric task prompts per model family (EmbeddingGemma `task: search result | query:` / `title: none | text:`).
 *  - Matryoshka truncation + L2 renormalization to the requested `dimensions`.
 *  - Bounded LRU cache keyed by (space, inputType, dimensions, text).
 *  - Per-backend circuit breaker (consecutive failures -> open for `cooldownMs`, then a half-open probe).
 *  - Request batching to each backend's `maxBatchSize`.
 */

import crypto from "node:crypto";
import { applyTaskPrompt, resolveModelInfo, truncateAndNormalize, type ModelSpaceInfo, type PromptFamily } from "./models.js";
import { errorText, waitWithSignal } from "./backends/http-util.js";
import { runEmbeddingSelfTest, type SelfTestResult } from "./self-test.js";
import {
	EmbeddingError,
	type EmbeddingBackend,
	type EmbeddingInputType,
	type SimilarityCalibration,
} from "./types.js";

export interface EmbeddingRouterOptions {
	readonly backends: readonly EmbeddingBackend[];
	/** Default output dimensionality (Matryoshka). Undefined keeps the native size. */
	readonly dimensions?: number;
	/** Max cached vectors (default 20000; 0 disables caching). */
	readonly cacheSize?: number;
	readonly failureThreshold?: number;
	readonly cooldownMs?: number;
	/** Max characters per input text before truncation (default 16000, roughly the 8K-token Gemma window). */
	readonly maxInputChars?: number;
	/** Overrides the calibration published for the space. */
	readonly calibration?: SimilarityCalibration;
	/** Overrides the prompt family derived from the primary backend's model. */
	readonly promptFamily?: PromptFamily;
	readonly now?: () => number;
	readonly onEvent?: (event: string, fields: Record<string, unknown>) => void;
}

export interface RouterEmbedRequest {
	readonly texts: readonly string[];
	readonly inputType?: EmbeddingInputType;
	readonly dimensions?: number;
	readonly signal?: AbortSignal;
}

export interface RouterEmbedResult {
	readonly vectors: Float32Array[];
	readonly space: string;
	readonly dimensions: number;
	/** Backend ids that served uncached texts (empty when everything came from cache). */
	readonly servedBy: readonly string[];
	readonly cacheHits: number;
	readonly promptTokensEstimate: number;
}

interface BreakerState {
	failures: number;
	openUntil: number;
	probing?: boolean;
	lastError: string | undefined;
}

export interface BackendStatus {
	readonly id: string;
	readonly kind: string;
	readonly model: string;
	readonly circuit: "closed" | "open" | "half-open";
	readonly consecutiveFailures: number;
	readonly lastError?: string;
	readonly ok?: boolean;
	readonly detail?: string;
	/** Set when the startup self-test found this backend unable to separate related from unrelated text. */
	readonly quarantined?: string;
	readonly selfTest?: SelfTestResult;
}

function validatedRows(rows: Float32Array[], count: number, dimensions?: number): Float32Array[] {
	if (rows.length !== count) throw new EmbeddingError("UPSTREAM", `expected ${count} vectors, got ${rows.length}`);
	for (const row of rows) {
		if (!row || row.length < (dimensions ?? 1)) throw new EmbeddingError("UPSTREAM", "backend returned undersized vector");
	}
	return rows.map(row => truncateAndNormalize(row, dimensions));
}

export class EmbeddingRouter {
	readonly space: string;
	readonly info: ModelSpaceInfo;
	readonly defaultDimensions: number | undefined;
	private calibrationValue: SimilarityCalibration | undefined;
	private readonly calibrationPinned: boolean;
	private readonly quarantine = new Map<string, string>();
	private readonly selfTests = new Map<string, SelfTestResult>();
	private readonly backends: readonly EmbeddingBackend[];
	private readonly promptFamily: PromptFamily;
	private readonly cache = new Map<string, Float32Array>();
	private readonly cacheSize: number;
	private readonly breakers = new Map<string, BreakerState>();
	private readonly failureThreshold: number;
	private readonly cooldownMs: number;
	private readonly maxInputChars: number;
	private readonly now: () => number;
	private readonly onEvent: (event: string, fields: Record<string, unknown>) => void;
	private observedDimensions: number | undefined;

	constructor(options: EmbeddingRouterOptions) {
		if (options.backends.length === 0) throw new EmbeddingError("MISCONFIGURED", "at least one embedding backend is required");
		const ids = new Set<string>();
		for (const b of options.backends) {
			if (ids.has(b.id)) throw new EmbeddingError("MISCONFIGURED", `duplicate embedding backend id "${b.id}"`);
			ids.add(b.id);
		}
		const primary = options.backends[0]!;
		const mismatched = options.backends.filter((b) => b.space !== primary.space);
		if (mismatched.length > 0) {
			throw new EmbeddingError(
				"MISCONFIGURED",
				`embedding backends must share one embedding space so fallback vectors stay comparable: ` +
					`"${primary.id}" is ${primary.space} but ${mismatched.map((b) => `"${b.id}" is ${b.space}`).join(", ")}. ` +
					`Set EMBED_SPACE to assert two model ids are the same space, or run separate embedding servers.`,
			);
		}
		this.backends = options.backends;

		this.info = resolveModelInfo(primary.model, { space: primary.space });
		this.promptFamily = options.promptFamily ?? this.info.promptFamily;
		const effectiveFamily = primary.wantsPrompts ? this.promptFamily : "none";
		if (this.backends.some(b => (b.wantsPrompts ? this.promptFamily : "none") !== effectiveFamily)) {
			throw new EmbeddingError("MISCONFIGURED", "fallback backends must use the same effective prompt family");
		}
		this.space = effectiveFamily === resolveModelInfo(primary.space).promptFamily
			? primary.space : `${primary.space}:prompt=${effectiveFamily}`;
		const native = primary.nativeDimensions ?? this.info.nativeDimensions;
		if (options.dimensions !== undefined) {
			if (!Number.isInteger(options.dimensions) || options.dimensions < 8) {
				throw new EmbeddingError("MISCONFIGURED", `invalid embedding dimensions ${options.dimensions}`);
			}
			if (native !== undefined && options.dimensions > native) {
				throw new EmbeddingError("MISCONFIGURED", `dimensions ${options.dimensions} exceeds the native ${native} of ${this.space}`);
			}
		}
		this.defaultDimensions = options.dimensions;
		this.calibrationValue = options.calibration ?? this.info.calibration;
		this.calibrationPinned = options.calibration !== undefined;
		this.cacheSize = Math.max(0, options.cacheSize ?? 20_000);
		this.failureThreshold = Math.max(1, options.failureThreshold ?? 3);
		this.cooldownMs = Math.max(0, options.cooldownMs ?? 30_000);
		this.maxInputChars = Math.max(64, options.maxInputChars ?? 16_000);
		this.now = options.now ?? (() => Date.now());
		this.onEvent = options.onEvent ?? (() => undefined);
		for (const b of this.backends) this.breakers.set(b.id, { failures: 0, openUntil: 0, lastError: undefined });
	}

	get publishedDimensions(): number | undefined {
		return this.defaultDimensions ?? this.info.nativeDimensions ?? this.observedDimensions;
	}

	/** Model id clients should store vectors under: changes whenever vectors would stop being comparable. */
	modelId(dimensions = this.defaultDimensions): string {
		const dim = dimensions ?? this.info.nativeDimensions ?? this.observedDimensions;
		return dim ? `${this.space}@${dim}` : this.space;
	}

	/** Cosine calibration published to clients: explicit override > measured by `selfTest()` > model table. */
	get calibration(): SimilarityCalibration | undefined {
		return this.calibrationValue;
	}

	/**
	 * Runs the neutral-corpus quality self-test on every backend. Backends that fail are quarantined (skipped by
	 * `embed`) when `enforce` is true. The first passing backend's measured calibration becomes the published one
	 * unless a calibration was configured explicitly. Returns per-backend results (`error` when unreachable).
	 */
	async selfTest(options: { readonly enforce?: boolean } = {}): Promise<Array<{ id: string; result?: SelfTestResult; error?: string }>> {
		const enforce = options.enforce ?? true;
		const out: Array<{ id: string; result?: SelfTestResult; error?: string }> = [];
		for (const backend of this.backends) {
			try {
				const result = await runEmbeddingSelfTest(async (texts, inputType) => {
					const formatted = backend.wantsPrompts ? texts.map((t) => applyTaskPrompt(this.promptFamily, inputType, t)) : texts;
					const vecs: Float32Array[] = [];
					for (let start = 0; start < formatted.length; start += backend.maxBatchSize) {
						const batch = formatted.slice(start, start + backend.maxBatchSize);
						const dimensions = this.defaultDimensions;
						const rows = await backend.embed(batch, { inputType, ...(dimensions !== undefined ? { dimensions } : {}) });
						vecs.push(...validatedRows(rows, batch.length, dimensions));
					}
					return vecs;
				});
				this.selfTests.set(backend.id, result);
				if (!result.passed && enforce) {
					const reason = `self-test failed (separation ${result.separation}, top1 ${result.top1})`;
					this.quarantine.set(backend.id, reason);
					this.onEvent("embedding_backend_quarantined", { backend: backend.id, model: backend.model, ...result });
				} else {
					this.quarantine.delete(backend.id);
				}
				out.push({ id: backend.id, result });
			} catch (error) {
				// Unreachable at startup is not a quality verdict: leave it to the circuit breaker.
				out.push({ id: backend.id, error: errorText(error) });
			}
		}
		if (!this.calibrationPinned) {
			const firstPassing = this.backends.find((b) => this.selfTests.get(b.id)?.passed);
			const measured = firstPassing ? this.selfTests.get(firstPassing.id)?.calibration : undefined;
			if (measured) this.calibrationValue = measured;
		}
		return out;
	}

	get backendIds(): readonly string[] {
		return this.backends.map((b) => b.id);
	}

	async embed(request: RouterEmbedRequest): Promise<RouterEmbedResult> {
		const inputType = request.inputType ?? "document";
		const dimensions = request.dimensions ?? this.defaultDimensions;
		if (dimensions !== undefined) {
			if (!Number.isInteger(dimensions) || dimensions < 8) throw new EmbeddingError("BAD_REQUEST", `invalid dimensions ${dimensions}`);
			const native = this.info.nativeDimensions ?? this.observedDimensions;
			if (native !== undefined && dimensions > native) {
				throw new EmbeddingError("BAD_REQUEST", `dimensions ${dimensions} exceeds the native ${native} of ${this.space}`);
			}
		}
		const texts = request.texts.map((t) => {
			if (typeof t !== "string") throw new EmbeddingError("BAD_REQUEST", "every input must be a string");
			return t.length > this.maxInputChars ? t.slice(0, this.maxInputChars) : t;
		});

		const vectors: Array<Float32Array | undefined> = new Array(texts.length);
		const missing: number[] = [];
		const keys = texts.map((t) => this.cacheKey(inputType, dimensions, t));
		let cacheHits = 0;
		for (let i = 0; i < texts.length; i++) {
			const hit = this.cacheSize > 0 ? this.cache.get(keys[i]!) : undefined;
			if (hit) {
				// refresh LRU position
				this.cache.delete(keys[i]!);
				this.cache.set(keys[i]!, hit);
				vectors[i] = hit;
				cacheHits += 1;
			} else {
				missing.push(i);
			}
		}

		const servedBy = new Set<string>();
		// Deduplicate identical uncached texts inside one request.
		const uniqueByKey = new Map<string, number[]>();
		for (const idx of missing) {
			const list = uniqueByKey.get(keys[idx]!);
			if (list) list.push(idx);
			else uniqueByKey.set(keys[idx]!, [idx]);
		}
		const uniqueIdx = Array.from(uniqueByKey.values()).map((list) => list[0]!);

		if (uniqueIdx.length > 0) {
			const raw = await this.embedWithFallback(
				uniqueIdx.map((i) => texts[i]!),
				inputType,
				dimensions,
				request.signal,
				servedBy,
			);
			raw.forEach((out, n) => {
				const key = keys[uniqueIdx[n]!]!;
				for (const idx of uniqueByKey.get(key)!) vectors[idx] = out;
				this.remember(key, out);
			});
		}

		const dim = vectors[0]?.length ?? dimensions ?? 0;
		for (const v of vectors) {
			if (!v || v.length !== dim) throw new EmbeddingError("UPSTREAM", "embedding backend returned inconsistent vector dimensions");
		}
		if (dimensions === undefined && dim > 0) this.observedDimensions = dim;
		return {
			vectors: vectors as Float32Array[],
			space: this.space,
			dimensions: dim,
			servedBy: Array.from(servedBy),
			cacheHits,
			promptTokensEstimate: texts.reduce((sum, t) => sum + Math.ceil(t.length / 4), 0),
		};
	}

	private async embedWithFallback(
		texts: string[],
		inputType: EmbeddingInputType,
		dimensions: number | undefined,
		signal: AbortSignal | undefined,
		servedBy: Set<string>,
	): Promise<Float32Array[]> {
		const errors: string[] = [];
		let attempted = 0;
		for (const backend of this.backends) {
			const breaker = this.breakers.get(backend.id)!;
			const quarantined = this.quarantine.get(backend.id);
			if (quarantined) {
				errors.push(`${backend.id}: quarantined, ${quarantined}`);
				continue;
			}
			if (breaker.probing || (breaker.failures >= this.failureThreshold && this.now() < breaker.openUntil)) {
				errors.push(`${backend.id}: circuit open`);
				continue;
			}
			const probe = breaker.failures >= this.failureThreshold;
			if (probe) breaker.probing = true;
			attempted += 1;
			try {
				signal?.throwIfAborted();
				const formatted = backend.wantsPrompts ? texts.map((t) => applyTaskPrompt(this.promptFamily, inputType, t)) : texts;
				const out: Float32Array[] = [];
				for (let start = 0; start < formatted.length; start += backend.maxBatchSize) {
					const batch = formatted.slice(start, start + backend.maxBatchSize);
					const vecs = await backend.embed(batch, {
						inputType,
						...(dimensions !== undefined ? { dimensions } : {}),
						...(signal ? { signal } : {}),
					});
					signal?.throwIfAborted();
					out.push(...validatedRows(vecs, batch.length, dimensions));
				}
				const expected = dimensions ?? this.observedDimensions ?? out[0]?.length;
				if (out.some(row => row.length !== expected)) {
					throw new EmbeddingError("UPSTREAM", "embedding backend returned inconsistent vector dimensions");
				}
				if (breaker.failures > 0) this.onEvent("embedding_backend_recovered", { backend: backend.id });
				breaker.failures = 0;
				breaker.openUntil = 0;
				breaker.lastError = undefined;
				servedBy.add(backend.id);
				return out;
			} catch (error) {
				// Client errors (bad input) are not the backend's fault and must not trip its breaker or fall back.
				if (error instanceof EmbeddingError && error.code === "BAD_REQUEST") throw error;
				if (signal?.aborted) throw new EmbeddingError("TIMEOUT", "embedding request aborted", { cause: error });
				breaker.failures += 1;
				breaker.lastError = errorText(error);
				if (breaker.failures >= this.failureThreshold) breaker.openUntil = this.now() + this.cooldownMs;
				errors.push(`${backend.id}: ${breaker.lastError}`);
				this.onEvent("embedding_backend_failed", {
					backend: backend.id,
					consecutiveFailures: breaker.failures,
					circuitOpen: breaker.failures >= this.failureThreshold,
					error: breaker.lastError,
				});
			} finally {
				if (probe) breaker.probing = false;
			}
		}
		throw new EmbeddingError(
			"UNAVAILABLE",
			attempted === 0 ? `no embedding backend is available (${errors.join("; ")})` : `all embedding backends failed (${errors.join("; ")})`,
		);
	}

	private cacheKey(inputType: EmbeddingInputType, dimensions: number | undefined, text: string): string {
		const digest = crypto.createHash("sha256").update(text).digest("base64url");
		return `${inputType}\0${dimensions ?? "native"}\0${digest}`;
	}

	private remember(key: string, vec: Float32Array): void {
		if (this.cacheSize === 0) return;
		this.cache.set(key, vec);
		while (this.cache.size > this.cacheSize) {
			const oldest = this.cache.keys().next().value;
			if (oldest === undefined) break;
			this.cache.delete(oldest);
		}
	}

	status(): BackendStatus[] {
		const nowMs = this.now();
		return this.backends.map((b) => {
			const s = this.breakers.get(b.id)!;
			const circuit = s.failures < this.failureThreshold ? "closed" : nowMs < s.openUntil ? "open" : "half-open";
			return {
				id: b.id,
				kind: b.kind,
				model: b.model,
				circuit,
				consecutiveFailures: s.failures,
				...(s.lastError ? { lastError: s.lastError } : {}),
				...(this.quarantine.has(b.id) ? { quarantined: this.quarantine.get(b.id)! } : {}),
				...(this.selfTests.has(b.id) ? { selfTest: this.selfTests.get(b.id)! } : {}),
			};
		});
	}

	/** Probes every backend (bounded by `timeoutMs`). */
	async health(timeoutMs = 5_000): Promise<{ ok: boolean; backends: BackendStatus[] }> {
		const checks = await Promise.all(
			this.backends.map(async (b) => {
				try {
					const signal = AbortSignal.timeout(timeoutMs);
					return await waitWithSignal(b.health(), signal);
				} catch (error) {
					return { ok: false, detail: errorText(error) };
				}
			}),
		);
		const backends = this.status().map((s, i) => ({
			...s,
			ok: checks[i]!.ok && s.circuit !== "open" && !s.quarantined,
		}));
		return { ok: backends.some((b) => b.ok), backends };
	}

	async close(): Promise<void> {
		await Promise.all(this.backends.map((b) => b.close?.().catch(() => undefined)));
		this.cache.clear();
	}
}
