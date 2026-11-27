/**
 * Model knowledge used by the embedding router: which embedding *space* a backend model id belongs to,
 * which task-prompt family it expects, its native dimensionality, Matryoshka support and a cosine calibration.
 *
 * Calibrations here are fallbacks: the embedding server measures the live model on a neutral corpus at startup
 * (`self-test.ts`: unrelated pairs -> floor = their 95th percentile, paraphrases -> ceiling = their median) and
 * publishes that instead. Entries marked "measured" came from that procedure; the others are approximate
 * starting points. Override with `EMBED_SIMILARITY_FLOOR` / `EMBED_SIMILARITY_CEILING`.
 */

import type { EmbeddingInputType, SimilarityCalibration } from "./types.js";

export type PromptFamily = "embeddinggemma" | "nomic" | "e5" | "bge" | "none";

export interface ModelSpaceInfo {
	/** Canonical space id (vectors from any model id mapping here are mutually comparable). */
	readonly space: string;
	readonly promptFamily: PromptFamily;
	readonly nativeDimensions: number | undefined;
	/** Dimensions the model was trained to be truncated to (Matryoshka Representation Learning). */
	readonly matryoshkaDimensions: readonly number[];
	readonly calibration: SimilarityCalibration | undefined;
}

const GEMMA_MRL = [768, 512, 256, 128] as const;

/** Ordered: first matching pattern wins. Patterns match the lowercased backend model id. */
const KNOWN_MODELS: ReadonlyArray<{ readonly pattern: RegExp; readonly info: Omit<ModelSpaceInfo, "space"> & { readonly space?: string } }> = [
	{
		pattern: /embedding-?gemma-?2|embed-gemma-?2|embeddinggemma2/,
		info: {
			space: "google/embeddinggemma-2",
			promptFamily: "embeddinggemma",
			nativeDimensions: 768,
			matryoshkaDimensions: GEMMA_MRL,
			// measured (self-test.ts, onnx-community q4 on CPU): negatives p95 0.618, positives median 0.773
			calibration: { floor: 0.618, ceiling: 0.773 },
		},
	},
	{
		// FastFlowLM's NPU build (`embed-gemma:300m` tag form). Its vectors are NOT interchangeable with Google's
		// reference export (measured cosine ~0 for identical inputs), so it is its own space. No calibration: the
		// server measures one at startup (and quarantines the backend if it fails the quality self-test).
		pattern: /^embed-gemma:/,
		info: {
			space: "fastflowlm/embed-gemma-300m",
			promptFamily: "embeddinggemma",
			nativeDimensions: 768,
			matryoshkaDimensions: GEMMA_MRL,
			calibration: undefined,
		},
	},
	{
		// onnx-community/embeddinggemma-300m-ONNX, google/embeddinggemma-300m, GGUF ports.
		pattern: /embedding-?gemma|embed-gemma/,
		info: {
			space: "google/embeddinggemma-300m",
			promptFamily: "embeddinggemma",
			nativeDimensions: 768,
			matryoshkaDimensions: GEMMA_MRL,
			calibration: { floor: 0.3, ceiling: 0.75 },
		},
	},
	{
		pattern: /all-minilm-l6-v2/,
		info: {
			space: "sentence-transformers/all-MiniLM-L6-v2",
			promptFamily: "none",
			nativeDimensions: 384,
			matryoshkaDimensions: [],
			calibration: { floor: 0.24, ceiling: 0.74 },
		},
	},
	{
		pattern: /nomic-embed-text/,
		info: {
			promptFamily: "nomic",
			nativeDimensions: 768,
			matryoshkaDimensions: [768, 512, 256, 128, 64],
			calibration: { floor: 0.38, ceiling: 0.8 },
		},
	},
	{
		pattern: /(^|\/)(multilingual-)?e5-/,
		info: { promptFamily: "e5", nativeDimensions: undefined, matryoshkaDimensions: [], calibration: { floor: 0.72, ceiling: 0.9 } },
	},
	{
		pattern: /bge-(small|base|large)/,
		info: { promptFamily: "bge", nativeDimensions: undefined, matryoshkaDimensions: [], calibration: { floor: 0.45, ceiling: 0.85 } },
	},
	{
		pattern: /text-embedding-3-small/,
		info: {
			space: "openai/text-embedding-3-small",
			promptFamily: "none",
			nativeDimensions: 1536,
			matryoshkaDimensions: [1536, 1024, 768, 512, 256],
			calibration: { floor: 0.2, ceiling: 0.68 },
		},
	},
	{
		pattern: /text-embedding-3-large/,
		info: {
			space: "openai/text-embedding-3-large",
			promptFamily: "none",
			nativeDimensions: 3072,
			matryoshkaDimensions: [3072, 2048, 1536, 1024, 768, 512, 256],
			calibration: { floor: 0.18, ceiling: 0.66 },
		},
	},
	{
		pattern: /gemini-embedding/,
		info: {
			promptFamily: "none",
			nativeDimensions: 3072,
			matryoshkaDimensions: [3072, 1536, 768],
			calibration: { floor: 0.5, ceiling: 0.8 },
		},
	},
];

/** Resolves what is known about a backend model id. Unknown models get their own space (`model` itself). */
export function resolveModelInfo(model: string, overrides?: { readonly space?: string; readonly promptFamily?: PromptFamily }): ModelSpaceInfo {
	const lower = model.trim().toLowerCase();
	const known = KNOWN_MODELS.find((entry) => entry.pattern.test(lower))?.info;
	const base = known ?? {
		space: model.trim(),
		promptFamily: "none",
		nativeDimensions: undefined,
		matryoshkaDimensions: [],
		calibration: undefined,
	};
	return {
		...base,
		space: base.space ?? model.trim(),
		...(overrides?.space ? { space: overrides.space } : {}),
		...(overrides?.promptFamily ? { promptFamily: overrides.promptFamily } : {}),
	};
}

export function isPromptFamily(value: string): value is PromptFamily {
	return value === "embeddinggemma" || value === "nomic" || value === "e5" || value === "bge" || value === "none";
}

/**
 * Applies the model family's asymmetric retrieval prompt. Idempotent: a text that already carries the
 * family's prefix is left untouched, so clients that format prompts themselves are not double-prefixed.
 */
export function applyTaskPrompt(family: PromptFamily, inputType: EmbeddingInputType, text: string): string {
	switch (family) {
		case "embeddinggemma": {
			if (/^task: [^|]{1,64} \| query: /.test(text) || /^title: [^|]{0,256} \| text: /.test(text)) return text;
			return inputType === "query" ? `task: search result | query: ${text}` : `title: none | text: ${text}`;
		}
		case "nomic": {
			if (/^search_(query|document): /.test(text)) return text;
			return inputType === "query" ? `search_query: ${text}` : `search_document: ${text}`;
		}
		case "e5": {
			if (/^(query|passage): /.test(text)) return text;
			return inputType === "query" ? `query: ${text}` : `passage: ${text}`;
		}
		case "bge": {
			const prefix = "Represent this sentence for searching relevant passages: ";
			if (inputType !== "query" || text.startsWith(prefix)) return text;
			return `${prefix}${text}`;
		}
		default:
			return text;
	}
}

/** Truncates (Matryoshka) and L2-renormalizes a vector. Returns a fresh Float32Array; zero vectors stay zero. */
export function truncateAndNormalize(vector: ArrayLike<number>, dimensions: number | undefined): Float32Array {
	const dim = dimensions !== undefined && dimensions > 0 ? Math.min(dimensions, vector.length) : vector.length;
	const out = new Float32Array(dim);
	let normSq = 0;
	for (let i = 0; i < dim; i++) {
		const x = Number(vector[i] ?? 0);
		const v = Number.isFinite(x) ? x : 0;
		out[i] = v;
		normSq += v * v;
	}
	if (normSq > 0) {
		const inv = 1 / Math.sqrt(normSq);
		for (let i = 0; i < dim; i++) out[i] = out[i]! * inv;
	}
	return out;
}
