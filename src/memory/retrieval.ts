/**
 * Deterministic retrieval helpers for SqliteMemoryService.search:
 * query tokenization / light stemming / prefix FTS5 queries, a documented blended re-ranker,
 * and vector encode/decode + cosine similarity for the optional EmbeddingProvider path.
 *
 * No clocks (callers pass `nowMs`), no randomness, no network.
 */

const STOPWORDS: ReadonlySet<string> = new Set([
	"a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "has", "have", "how", "i", "in", "is", "it",
	"me", "my", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "who", "with",
	"you", "your",
]);

const VOWELS = /[aeiou]/;

/** Very small English suffix stripper. Non-ASCII-alpha tokens are returned unchanged. */
export function stemToken(token: string): string {
	if (token.length <= 3 || !/^[a-z]+$/.test(token)) return token;
	let t = token;
	if (t.endsWith("sses")) return t.slice(0, -2);
	if (t.endsWith("ies") && t.length > 4) return `${t.slice(0, -3)}y`;
	if (t.endsWith("ing") && t.length > 5 && VOWELS.test(t.slice(0, -3))) {
		t = t.slice(0, -3);
	} else if (t.endsWith("ed") && t.length > 4 && VOWELS.test(t.slice(0, -2))) {
		t = t.slice(0, -2);
	} else if (t.endsWith("ly") && t.length > 5) {
		t = t.slice(0, -2);
	} else if (/(?:s|x|z|ch|sh)es$/.test(t) && t.length > 4) {
		t = t.slice(0, -2);
	} else if (t.endsWith("s") && !t.endsWith("ss") && !t.endsWith("us") && !t.endsWith("is")) {
		t = t.slice(0, -1);
	} else {
		return t;
	}
	// running -> runn -> run
	if (t.length > 3 && t[t.length - 1] === t[t.length - 2] && !/[aeiouls]/.test(t[t.length - 1] ?? "")) {
		t = t.slice(0, -1);
	}
	return t.length >= 2 ? t : token;
}

/** Lowercased, NFKC-normalized letter/digit tokens (unicode aware), stopwords dropped unless nothing else remains. */
export function tokenizeQuery(raw: string, maxTokens = 12): string[] {
	const bounded = raw.length > 2000 ? raw.slice(0, 2000) : raw;
	const all = bounded.normalize("NFKC").toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
	const unique = Array.from(new Set(all));
	const meaningful = unique.filter((t) => !STOPWORDS.has(t));
	return (meaningful.length > 0 ? meaningful : unique).slice(0, maxTokens);
}

/** Terms (stem + original) that should match a document, used for FTS prefix queries and LIKE fallbacks. */
export function expandToken(token: string): string[] {
	const stem = stemToken(token);
	return stem === token || token.startsWith(stem) ? [stem] : [stem, token];
}

/**
 * Builds a safe FTS5 query: every term is quoted (no operator injection) and prefix-matched, OR-combined
 * so BM25 rewards documents matching more terms. Returns null when the query has no usable tokens.
 */
export function buildFtsQuery(rawQuery: string): { fts: string | null; tokens: string[] } {
	const tokens = tokenizeQuery(rawQuery);
	if (tokens.length === 0) return { fts: null, tokens };
	const terms: string[] = [];
	for (const token of tokens) {
		for (const term of expandToken(token)) {
			terms.push(term.length >= 2 ? `"${term}"*` : `"${term}"`);
		}
	}
	return { fts: Array.from(new Set(terms)).join(" OR "), tokens };
}

/** Escapes LIKE wildcards (use with `ESCAPE '\\'`). */
export function escapeLike(text: string): string {
	return text.replace(/[\\%_]/g, (c) => `\\${c}`);
}

// ---------------------------------------------------------------------------------------------------------------------
// Re-ranking
// ---------------------------------------------------------------------------------------------------------------------

/**
 * Blend weights (normalized by their sum, so absolute scale does not matter). Documented defaults:
 *  - bm25       0.55  lexical relevance (BM25 max-normalized within the candidate set)
 *  - recency    0.12  exponential decay of `updatedAt` (half-life `recencyHalfLifeDays`)
 *  - pinned     0.10  explicit user pin
 *  - confidence 0.13  stored confidence (time-decayed for low-confidence, unpinned memories)
 *  - evidence   0.10  log-scaled evidence count (saturates at 8 supporting episodes)
 */
export interface RankingWeights {
	readonly bm25: number;
	readonly recency: number;
	readonly pinned: number;
	readonly confidence: number;
	readonly evidence: number;
}

export const DEFAULT_RANKING_WEIGHTS: RankingWeights = {
	bm25: 0.55,
	recency: 0.12,
	pinned: 0.1,
	confidence: 0.13,
	evidence: 0.1,
};

export interface RankingOptions {
	readonly weights?: Partial<RankingWeights>;
	/** Half-life of the recency boost. Default 30 days. */
	readonly recencyHalfLifeDays?: number;
	/** Memories below this confidence (and not pinned) have their confidence time-decayed. Default 0.7. */
	readonly lowConfidenceThreshold?: number;
	/** Half-life of that decay in days; 0 disables it. Default 90. */
	readonly lowConfidenceDecayHalfLifeDays?: number;
	/** Share of the final score given to embedding cosine similarity when a provider is configured. Default 0.35. */
	readonly vectorWeight?: number;
}

export interface ResolvedRankingOptions {
	readonly weights: RankingWeights;
	readonly recencyHalfLifeDays: number;
	readonly lowConfidenceThreshold: number;
	readonly lowConfidenceDecayHalfLifeDays: number;
	readonly vectorWeight: number;
}

export function resolveRankingOptions(options?: RankingOptions): ResolvedRankingOptions {
	return {
		weights: { ...DEFAULT_RANKING_WEIGHTS, ...options?.weights },
		recencyHalfLifeDays: options?.recencyHalfLifeDays ?? 30,
		lowConfidenceThreshold: options?.lowConfidenceThreshold ?? 0.7,
		lowConfidenceDecayHalfLifeDays: options?.lowConfidenceDecayHalfLifeDays ?? 90,
		vectorWeight: Math.max(0, Math.min(1, options?.vectorWeight ?? 0.35)),
	};
}

export interface RankableMemory {
	readonly confidence: number;
	readonly pinned: boolean;
	readonly evidenceCount: number;
	readonly updatedAt: number;
}

const DAY_MS = 86_400_000;

export function effectiveConfidence(memory: RankableMemory, nowMs: number, opts: ResolvedRankingOptions): number {
	let confidence = Math.max(0, Math.min(1, memory.confidence));
	if (
		!memory.pinned &&
		opts.lowConfidenceDecayHalfLifeDays > 0 &&
		confidence < opts.lowConfidenceThreshold
	) {
		const ageDays = Math.max(0, nowMs - memory.updatedAt) / DAY_MS;
		confidence *= 0.5 ** (ageDays / opts.lowConfidenceDecayHalfLifeDays);
	}
	return confidence;
}

/**
 * Lexical + metadata blend in [0, 1]. `lexical` must already be normalized to [0, 1].
 */
export function blendScore(
	lexical: number,
	memory: RankableMemory,
	nowMs: number,
	opts: ResolvedRankingOptions,
): number {
	const w = opts.weights;
	const sum = w.bm25 + w.recency + w.pinned + w.confidence + w.evidence;
	if (sum <= 0) return lexical;
	const ageDays = Math.max(0, nowMs - memory.updatedAt) / DAY_MS;
	const recency = opts.recencyHalfLifeDays > 0 ? 0.5 ** (ageDays / opts.recencyHalfLifeDays) : 0;
	const evidence = Math.min(1, Math.log(1 + Math.max(0, memory.evidenceCount)) / Math.log(9));
	const total =
		w.bm25 * Math.max(0, Math.min(1, lexical)) +
		w.recency * recency +
		w.pinned * (memory.pinned ? 1 : 0) +
		w.confidence * effectiveConfidence(memory, nowMs, opts) +
		w.evidence * evidence;
	return total / sum;
}

/** Combines the lexical/metadata blend with embedding cosine similarity (clamped to [0, 1]). */
export function hybridScore(blend: number, cosine: number, vectorWeight: number): number {
	const vw = Math.max(0, Math.min(1, vectorWeight));
	return (1 - vw) * blend + vw * Math.max(0, Math.min(1, cosine));
}

// ---------------------------------------------------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------------------------------------------------

export function encodeVector(vector: ArrayLike<number>): Uint8Array {
	const f32 = new Float32Array(vector.length);
	for (let i = 0; i < vector.length; i++) f32[i] = vector[i] ?? 0;
	return new Uint8Array(f32.buffer.slice(0, f32.byteLength));
}

export function decodeVector(blob: Uint8Array): Float32Array {
	const copy = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);
	return new Float32Array(copy);
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
	if (a.length === 0 || a.length !== b.length) return 0;
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		const x = a[i] ?? 0;
		const y = b[i] ?? 0;
		dot += x * y;
		na += x * x;
		nb += y * y;
	}
	if (na === 0 || nb === 0) return 0;
	return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
