/**
 * Deterministic retrieval helpers for SqliteMemoryService.search:
 * query tokenization (camelCase/snake_case/kebab-case aware) / stemming / prefix FTS5 queries,
 * calibrated BM25 + query-coverage scoring, relevance-gated multi-lane fusion (lexical + vector +
 * entity-graph PPR + temporal + metadata modulation), and pre-normalized Float32 vector math.
 *
 * No clocks (callers pass `nowMs`), no randomness, no network.
 */

const STOPWORDS: ReadonlySet<string> = new Set([
	"a", "about", "also", "am", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could", "did", "do", "does",
	"for", "from", "had", "has", "have", "here", "how", "i", "in", "into", "is", "it", "its", "just", "me", "my",
	"of", "on", "or", "our", "should", "than", "that", "the", "their", "them", "then", "there", "these", "they",
	"this", "those", "to", "us", "very", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
	"with", "would", "you", "your",
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

/**
 * Splits camelCase/PascalCase identifiers into subwords while preserving the compound token too
 * (e.g. "maxRetryCount" -> ["maxretrycount", "max", "retry", "count"]).
 */
function splitCamelParts(token: string): string[] {
	if (!/[a-z][A-Z]|[A-Z]{2}[a-z]/.test(token)) return [];
	const parts = token
		.replace(/([a-z0-9])([A-Z])/g, "$1 $2")
		.replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
		.toLowerCase()
		.split(" ")
		.filter((p) => p.length >= 2);
	return parts.length > 1 ? parts : [];
}

/**
 * Lowercased, NFKC-normalized letter/digit tokens (unicode aware, camelCase/snake_case aware),
 * stopwords dropped unless nothing else remains.
 */
export function tokenizeQuery(raw: string, maxTokens = 16): string[] {
	const bounded = raw.length > 2000 ? raw.slice(0, 2000) : raw;
	const rawMatches = bounded.normalize("NFKC").match(/[\p{L}\p{N}]+/gu) ?? [];
	const expanded: string[] = [];
	for (const tok of rawMatches) {
		const lower = tok.toLowerCase();
		expanded.push(lower);
		for (const part of splitCamelParts(tok)) {
			expanded.push(part);
		}
	}
	const unique = Array.from(new Set(expanded));
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

/**
 * Calibrated lexical relevance score in [0, 1] replacing naive per-batch max-normalization.
 *
 * Combines:
 *  - Query-length-adaptive BM25 saturation `mag / (mag + k0)` (eliminating single-hit `1.0` inflation!)
 *  - Length-weighted query-term coverage against key + content (rewarding multi-term & specific matches)
 *  - Exact phrase / key substring bonus
 *  - Mild single-hit dampener when a long query (>= 3 tokens) only hits 1 weak term
 */
export function calibrateLexicalScore(input: {
	readonly bm25Magnitude: number;
	readonly query: string;
	readonly queryTokens: readonly string[];
	readonly docKey: string;
	readonly docContent: string;
}): number {
	const { bm25Magnitude, query, queryTokens, docKey, docContent } = input;
	const haystack = `${docKey} ${docContent}`.normalize("NFKC").toLowerCase();
	const rawQuery = query.trim().normalize("NFKC").toLowerCase();

	if (queryTokens.length === 0) {
		if (rawQuery.length > 0 && haystack.includes(rawQuery)) return 0.6;
		return 0;
	}

	const keyLower = docKey.normalize("NFKC").toLowerCase();
	let matchedWeight = 0;
	let totalWeight = 0;
	let matchedCount = 0;
	let keyMatchedWeight = 0;

	for (const token of queryTokens) {
		const w = Math.min(2.2, Math.max(0.65, token.length / 4));
		totalWeight += w;
		const variants = expandToken(token);
		let inDoc = false;
		let inKey = false;
		for (const v of variants) {
			if (v.length === 0) continue;
			if (!inDoc && haystack.includes(v)) inDoc = true;
			if (!inKey && keyLower.includes(v)) inKey = true;
		}
		if (inDoc) {
			matchedWeight += w;
			matchedCount += 1;
		}
		if (inKey) {
			keyMatchedWeight += w;
		}
	}

	if (matchedCount === 0 && bm25Magnitude <= 0) return 0;

	const coverage = totalWeight > 0 ? matchedWeight / totalWeight : 0;
	const keyBonus = totalWeight > 0 ? Math.min(0.12, 0.14 * (keyMatchedWeight / totalWeight)) : 0;

	// Query-length-adaptive BM25 saturation: longer queries need higher BM25 magnitude to saturate.
	const effectiveTokens = Math.min(queryTokens.length, 8);
	const k0 = 2.6 + 0.7 * effectiveTokens;
	let bm25Sat = bm25Magnitude > 0 ? bm25Magnitude / (bm25Magnitude + k0) : 0;

	// Exact phrase or exact key match bonus
	let exactBonus = 0;
	if (rawQuery.length >= 3 && (haystack.includes(rawQuery) || keyLower === rawQuery)) {
		exactBonus = 0.16;
	}

	// Cap BM25 saturation on incidental single-token overlap in multi-word queries
	// (prevents a 3x-weighted key hit on one generic word like "model" or "policy" from overpowering semantics)
	if (queryTokens.length >= 3 && matchedCount <= 1 && exactBonus === 0) {
		bm25Sat = Math.min(bm25Sat, coverage * 1.25);
	}

	let score = 0.48 * bm25Sat + 0.52 * coverage + keyBonus + exactBonus;

	// Dampen incidental single-token overlap on multi-word queries (e.g. "own" matching "owned")
	if (queryTokens.length >= 3 && matchedCount <= 1 && exactBonus === 0) {
		score *= 0.55;
	} else if (queryTokens.length >= 5 && coverage < 0.28 && exactBonus === 0) {
		score *= 0.68;
	}

	return Math.max(0, Math.min(1, score));
}

// ---------------------------------------------------------------------------------------------------------------------
// Re-ranking & Multi-Lane Fusion
// ---------------------------------------------------------------------------------------------------------------------

/**
 * Blend weights (normalized by their sum, so absolute scale does not matter). Documented defaults:
 *  - bm25       0.55  lexical relevance
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
	/** Share of vector lane in hybrid scoring (used by `hybridScore` and fused scoring). Default 0.45. */
	readonly vectorWeight?: number;
	/** Minimum calibrated score required to return a hit when abstention filtering is active. Default 0.22 (hybrid) / 0.14 (lexical). */
	readonly minScore?: number;
}

export interface ResolvedRankingOptions {
	readonly weights: RankingWeights;
	readonly recencyHalfLifeDays: number;
	readonly lowConfidenceThreshold: number;
	readonly lowConfidenceDecayHalfLifeDays: number;
	readonly vectorWeight: number;
	readonly minScore: number | undefined;
}

export function resolveRankingOptions(options?: RankingOptions): ResolvedRankingOptions {
	return {
		weights: { ...DEFAULT_RANKING_WEIGHTS, ...options?.weights },
		recencyHalfLifeDays: options?.recencyHalfLifeDays ?? 30,
		lowConfidenceThreshold: options?.lowConfidenceThreshold ?? 0.7,
		lowConfidenceDecayHalfLifeDays: options?.lowConfidenceDecayHalfLifeDays ?? 90,
		vectorWeight: Math.max(0, Math.min(1, options?.vectorWeight ?? 0.45)),
		minScore: options?.minScore,
	};
}

export interface RankableMemory {
	readonly confidence: number;
	readonly pinned: boolean;
	readonly evidenceCount: number;
	readonly updatedAt: number;
	readonly feedbackWeight?: number;
	readonly accessCount?: number;
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

/** Computes normalized metadata quality factor in [0, 1] (recency, pin, effective confidence, evidence, feedback). */
export function metadataQualityFactor(
	memory: RankableMemory,
	nowMs: number,
	opts: ResolvedRankingOptions,
): number {
	const w = opts.weights;
	const metaSum = w.recency + w.pinned + w.confidence + w.evidence;
	const ageDays = Math.max(0, nowMs - memory.updatedAt) / DAY_MS;
	const recency = opts.recencyHalfLifeDays > 0 ? 0.5 ** (ageDays / opts.recencyHalfLifeDays) : 0;
	const evidence = Math.min(1, Math.log(1 + Math.max(0, memory.evidenceCount)) / Math.log(9));
	const conf = effectiveConfidence(memory, nowMs, opts);

	const baseMeta =
		metaSum > 0
			? (w.recency * recency + w.pinned * (memory.pinned ? 1 : 0) + w.confidence * conf + w.evidence * evidence) /
				metaSum
			: 0.5;

	// Streaming feedback weight centered at 0.5 shifts quality by at most ±0.12
	const fb = memory.feedbackWeight ?? 0.5;
	const fbDelta = (Math.max(0, Math.min(1, fb)) - 0.5) * 0.24;
	return Math.max(0, Math.min(1, baseMeta + fbDelta));
}

/**
 * Lexical + metadata blend in [0, 1]. `lexical` must already be normalized to [0, 1].
 * Preserved for backward compatibility with callers and unit tests.
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

/**
 * Calibrates raw embedding cosine similarity into [0, 1] above the high-dimensional embedding noise
 * floor (`floor = 0.24`) with a concave curve over `[0.24, 0.74]`. Unrelated texts (cosine <= 0.26)
 * stay cleanly below abstention thresholds while genuine zero-overlap semantic paraphrases (~0.30..0.55)
 * receive strong calibrated relevance.
 */
export function calibrateCosine(cosine: number, floor = 0.24): number {
	if (!Number.isFinite(cosine) || cosine <= floor) return 0;
	const norm = Math.min(1, (cosine - floor) / (0.74 - floor));
	return Math.pow(norm, 0.55);
}

export interface MultiLaneSignals {
	readonly lexical: number;
	/** Raw cosine similarity in [-1, 1], or undefined when no EmbeddingProvider is active. */
	readonly cosine?: number;
	/** Hub-dampened entity-graph PPR activation in [0, 1]. */
	readonly graph: number;
	/** Temporal anchor overlap tightness score in [0, 1]. */
	readonly temporal: number;
	/** True when both query and memory have explicit dates that do NOT overlap. */
	readonly temporalConflict?: boolean;
	/** Optional multiplier for historical-audit queries (e.g. >1 for superseded, <1 for excluded current state). */
	readonly historicalMultiplier?: number;
}

/**
 * Relevance-gated multi-lane fusion combining Lexical + Calibrated Vector + Entity-Graph PPR + Temporal,
 * modulated multiplicatively by metadata quality (pin, confidence, evidence, recency, feedback).
 *
 * Key property: metadata MODULATES genuine relevance (`relevance * (0.78 + 0.40 * meta)`) rather than
 * inventing score out of thin air, guaranteeing clean abstention on unanswerable queries!
 */
export function fuseRetrievalSignals(
	signals: MultiLaneSignals,
	memory: RankableMemory,
	nowMs: number,
	opts: ResolvedRankingOptions,
): number {
	const lex = Math.max(0, Math.min(1, signals.lexical));
	const hasVector = signals.cosine !== undefined;
	const vec = hasVector ? calibrateCosine(signals.cosine!) : 0;
	const graph = Math.max(0, Math.min(1, signals.graph));
	const temporal = Math.max(0, Math.min(1, signals.temporal));

	// 1. Dominant-lane + dual-agreement synergy between lexical & calibrated semantic vector
	let relevance: number;
	if (hasVector) {
		const primary = Math.max(lex, vec);
		const secondary = Math.min(lex, vec);
		const vwShift = (opts.vectorWeight - 0.45) * 0.25;
		const weightedLex = Math.max(0, Math.min(1, lex * (1 - vwShift)));
		const weightedVec = Math.max(0, Math.min(1, vec * (1 + vwShift)));
		relevance = Math.min(1, 0.9 * Math.max(weightedLex, weightedVec) + 0.24 * Math.min(weightedLex, weightedVec));
		if (primary > 0 && relevance === 0) relevance = 0.85 * primary + 0.2 * secondary;
	} else {
		relevance = lex;
	}

	// 2. Entity-graph multi-hop PPR lane (additive residual so bridged 2-hop targets surface beside/above hop-1!)
	if (graph > 0) {
		const graphGain = hasVector ? 0.6 : 0.62;
		relevance = Math.min(1, relevance + graphGain * graph * (1 - 0.3 * relevance));
	}

	// 3. Temporal anchor alignment (tightness-ranked boost; mild dampen on explicit off-year conflict)
	if (temporal > 0) {
		relevance = Math.min(1, relevance + 0.34 * temporal * (1 - 0.3 * relevance));
	} else if (signals.temporalConflict && relevance > 0) {
		relevance *= 0.74;
	}

	if (relevance <= 0) return 0;

	// 4. Relevance-gated multiplicative metadata modulation (preserves pin > strong > weak ordering without fabricating hits)
	const meta = metadataQualityFactor(memory, nowMs, opts);
	let finalScore = relevance * (0.78 + 0.4 * meta);

	// 5. Historical-audit routing adjustment (only applied when query explicitly asks about past/superseded state)
	if (signals.historicalMultiplier !== undefined && signals.historicalMultiplier !== 1) {
		finalScore *= signals.historicalMultiplier;
	}

	return Math.max(0, Math.min(1, finalScore));
}

// ---------------------------------------------------------------------------------------------------------------------
// Vectors (Pre-Normalized Fast Dot Product + Serialization)
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

/** Returns a unit L2-normalized Float32Array (or null if zero/empty/non-finite). */
export function l2Normalize(vector: ArrayLike<number>): Float32Array | null {
	const len = vector.length;
	if (len === 0) return null;
	let normSq = 0;
	for (let i = 0; i < len; i++) {
		const x = vector[i] ?? 0;
		if (!Number.isFinite(x)) return null;
		normSq += x * x;
	}
	if (normSq === 0) return null;
	const inv = 1 / Math.sqrt(normSq);
	const out = new Float32Array(len);
	for (let i = 0; i < len; i++) {
		out[i] = (vector[i] ?? 0) * inv;
	}
	return out;
}

/** Fast unrolled dot product between a normalized query vector and a row inside a packed matrix. */
export function dotProductOffset(query: Float32Array, matrix: Float32Array, offset: number): number {
	const dim = query.length;
	let sum0 = 0;
	let sum1 = 0;
	let sum2 = 0;
	let sum3 = 0;
	const limit = dim - (dim % 4);
	let i = 0;
	for (; i < limit; i += 4) {
		const m = offset + i;
		sum0 += query[i]! * matrix[m]!;
		sum1 += query[i + 1]! * matrix[m + 1]!;
		sum2 += query[i + 2]! * matrix[m + 2]!;
		sum3 += query[i + 3]! * matrix[m + 3]!;
	}
	let total = sum0 + sum1 + sum2 + sum3;
	for (; i < dim; i++) {
		total += query[i]! * matrix[offset + i]!;
	}
	return total;
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
