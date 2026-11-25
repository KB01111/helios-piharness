import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { StatementSync } from "node:sqlite";
import type {
	ArtifactMetadata,
	ArtifactReadResult,
	ArtifactStore,
	EmbeddingProvider,
	FrictionSignal,
	FrictionSignalKind,
	LongTermMemory,
	MemoryKind,
	MemoryMutationOp,
	MemorySearchResult,
	MemoryService,
	MemoryStatus,
	MidTermEpisode,
	PendingEpisodeScope,
	RunTelemetryRecord,
	SkillHealthMetric,
	TelemetryStore,
	ToolInvocationMetric,
	ToolTraceItem,
} from "../contracts/app-capabilities.js";
import type { AppDatabase } from "./db.js";
import {
	buildBipartiteGraph,
	extractEntities,
	extractQueryEntities,
	extractRelations,
	runEntityPpr,
	type BipartiteMemoryGraph,
	type EntityGraphEdge,
} from "./graph.js";
import {
	buildFtsQuery,
	calibrateCosine,
	calibrateLexicalScore,
	decodeVector,
	dotProductOffset,
	encodeVector,
	escapeLike,
	expandToken,
	fuseRetrievalSignals,
	l2Normalize,
	resolveRankingOptions,
	type RankingOptions,
	type ResolvedRankingOptions,
} from "./retrieval.js";
import { sanitizeMemoryKey, sanitizeMemoryText } from "./sanitize.js";
import {
	detectHistoricalIntent,
	extractTemporalAnchors,
	scoreTemporalAlignment,
	type TemporalAnchor,
	type TemporalPrecision,
} from "./temporal.js";

type MemoryRow = {
	id: string;
	tenant_id: string;
	user_id: string;
	kind: string;
	mem_key: string;
	content: string;
	confidence: number;
	pinned: number;
	status: string;
	superseded_by_id: string | null;
	evidence_count: number;
	distinct_conversations: number;
	source_conversation_id: string | null;
	created_at: number;
	updated_at: number;
	valid_from?: number | null;
	valid_until?: number | null;
	access_count?: number | null;
	last_accessed_at?: number | null;
	feedback_weight?: number | null;
};

type EpisodeRow = {
	id: string;
	tenant_id: string;
	user_id: string;
	conversation_id: string;
	run_id: string;
	user_message: string;
	assistant_summary: string;
	tool_trace_json: string;
	friction_signals_json: string;
	friction_score: number;
	consolidated: number;
	created_at: number;
};

type ArtifactRow = {
	id: string;
	tenant_id: string;
	user_id: string;
	content_type: string;
	byte_length: number;
	source_tool: string | null;
	conversation_id: string | null;
	file_path: string | null;
	inline_content: string | null;
	created_at: number;
};

function rowToMemory(row: MemoryRow): LongTermMemory {
	return {
		id: row.id,
		tenantId: row.tenant_id,
		userId: row.user_id,
		kind: row.kind as MemoryKind,
		key: row.mem_key,
		content: row.content,
		confidence: Number(row.confidence),
		pinned: Number(row.pinned) === 1,
		status: row.status as MemoryStatus,
		...(row.superseded_by_id ? { supersededById: row.superseded_by_id } : {}),
		evidenceCount: Number(row.evidence_count),
		distinctConversations: Number(row.distinct_conversations),
		...(row.source_conversation_id ? { sourceConversationId: row.source_conversation_id } : {}),
		createdAt: Number(row.created_at),
		updatedAt: Number(row.updated_at),
		...(row.valid_from !== undefined && row.valid_from !== null ? { validFrom: Number(row.valid_from) } : {}),
		...(row.valid_until !== undefined && row.valid_until !== null ? { validUntil: Number(row.valid_until) } : {}),
		...(row.access_count !== undefined && row.access_count !== null ? { accessCount: Number(row.access_count) } : {}),
		...(row.last_accessed_at !== undefined && row.last_accessed_at !== null
			? { lastAccessedAt: Number(row.last_accessed_at) }
			: {}),
		...(row.feedback_weight !== undefined && row.feedback_weight !== null
			? { feedbackWeight: Number(row.feedback_weight) }
			: {}),
	};
}

function rowToEpisode(row: EpisodeRow): MidTermEpisode {
	return {
		id: row.id,
		tenantId: row.tenant_id,
		userId: row.user_id,
		conversationId: row.conversation_id,
		runId: row.run_id,
		userMessage: row.user_message,
		assistantSummary: row.assistant_summary,
		toolTrace: JSON.parse(row.tool_trace_json) as ToolTraceItem[],
		frictionSignals: JSON.parse(row.friction_signals_json) as FrictionSignal[],
		frictionScore: Number(row.friction_score),
		consolidated: Number(row.consolidated) === 1,
		createdAt: Number(row.created_at),
	};
}

const UPSERT_EMBEDDING_SQL = `INSERT INTO memory_embeddings (memory_id, tenant_id, user_id, model, dim, content_hash, vector, updated_at)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	ON CONFLICT(memory_id) DO UPDATE SET
		tenant_id = excluded.tenant_id, user_id = excluded.user_id, model = excluded.model, dim = excluded.dim,
		content_hash = excluded.content_hash, vector = excluded.vector, updated_at = excluded.updated_at`;

function embeddingText(key: string, content: string): string {
	return `${key}: ${content}`;
}

function hashText(text: string): string {
	return crypto.createHash("sha1").update(text).digest("hex");
}

function normalizeForCompare(text: string): string {
	return text.trim().replace(/\s+/g, " ").toLowerCase();
}

/** A usable embedding is non-empty and entirely finite (NaN/Infinity would poison every cosine score). */
function isUsableVector(vec: ArrayLike<number> | undefined | null): vec is ArrayLike<number> {
	if (!vec || vec.length === 0) return false;
	for (let i = 0; i < vec.length; i++) {
		if (!Number.isFinite(vec[i])) return false;
	}
	return true;
}

/** Decodes a stored vector blob; corrupt/truncated/non-finite blobs yield undefined instead of throwing. */
function tryDecodeVector(blob: Uint8Array): Float32Array | undefined {
	try {
		const vec = decodeVector(blob);
		return isUsableVector(vec) ? vec : undefined;
	} catch {
		return undefined;
	}
}

function resolveBatchTarget(targetId: string, batch: ReadonlyArray<LongTermMemory | null>): string {
	const match = /^batch:(\d{1,6})$/.exec(targetId);
	if (!match) return targetId;
	return batch[Number(match[1])]?.id ?? targetId;
}

interface CachedVectorScope {
	readonly stamp: string;
	readonly dim: number;
	readonly rows: readonly MemoryRow[];
	/** Packed L2-normalized vectors: `rows.length * dim` floats for contiguous fast dot products. */
	readonly matrix: Float32Array;
	readonly stale: ReadonlyArray<{ readonly row: MemoryRow; readonly text: string; readonly hash: string }>;
}

interface CachedGraphScope {
	readonly stamp: string;
	readonly graph: BipartiteMemoryGraph;
	readonly allowedIds: ReadonlySet<string>;
}

export class SqliteMemoryService implements MemoryService {
	private readonly appDb: AppDatabase;
	private readonly now: () => number;
	private readonly embeddingProvider: EmbeddingProvider | undefined;
	private readonly ranking: ResolvedRankingOptions;
	private readonly maxVectorCandidates: number;
	private readonly maxBackfillPerSearch: number;
	private readonly stmts = new Map<string, StatementSync>();
	private readonly vectorCache = new Map<string, CachedVectorScope>();
	private readonly graphCache = new Map<string, CachedGraphScope>();

	constructor(
		appDb: AppDatabase,
		options?: {
			readonly now?: () => number;
			/** Optional injected embedding provider enabling hybrid (lexical + cosine) retrieval. */
			readonly embeddingProvider?: EmbeddingProvider;
			/** Re-ranking weights / decay overrides (see `DEFAULT_RANKING_WEIGHTS` in retrieval.ts). */
			readonly ranking?: RankingOptions;
			/** Max in-scope memories scanned for vector similarity per search. Default 10000. */
			readonly maxVectorCandidates?: number;
			/** Max stale/missing vectors embedded lazily per search. Default 64. */
			readonly maxBackfillPerSearch?: number;
		},
	) {
		this.appDb = appDb;
		this.now = options?.now ?? (() => Date.now());
		this.embeddingProvider = options?.embeddingProvider;
		this.ranking = resolveRankingOptions(options?.ranking);
		this.maxVectorCandidates = Math.max(1, options?.maxVectorCandidates ?? 10_000);
		this.maxBackfillPerSearch = Math.max(1, options?.maxBackfillPerSearch ?? 64);
	}

	private stmt(sql: string): StatementSync {
		let cached = this.stmts.get(sql);
		if (!cached) {
			cached = this.appDb.db.prepare(sql);
			this.stmts.set(sql, cached);
		}
		return cached;
	}

	/**
	 * Monotonic DB change stamp combining same-connection `total_changes()` and cross-connection
	 * `PRAGMA data_version` so in-memory caches stay 100% coherent across sibling instances & raw SQL.
	 */
	private dbStamp(): string {
		const tc = (this.stmt("SELECT total_changes() AS tc").get() as { tc: number }).tc;
		const dv = (this.stmt("PRAGMA data_version").get() as { data_version: number }).data_version;
		return `${dv}:${tc}`;
	}

	async recordEpisode(
		episode: Omit<MidTermEpisode, "id" | "consolidated" | "createdAt"> & { readonly createdAt?: number },
	): Promise<MidTermEpisode> {
		const id = `ep_${crypto.randomUUID()}`;
		const createdAt = episode.createdAt ?? this.now();

		return this.appDb.transaction(() => {
			const res = this.stmt(
				`INSERT INTO midterm_episodes (
					id, tenant_id, user_id, conversation_id, run_id,
					user_message, assistant_summary, tool_trace_json,
					friction_signals_json, friction_score, consolidated, created_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
				ON CONFLICT(tenant_id, user_id, conversation_id, run_id) DO NOTHING`,
			).run(
				id,
				episode.tenantId,
				episode.userId,
				episode.conversationId,
				episode.runId,
				episode.userMessage,
				episode.assistantSummary,
				JSON.stringify(episode.toolTrace),
				JSON.stringify(episode.frictionSignals),
				episode.frictionScore,
				createdAt,
			);

			if (Number(res.changes) > 0) {
				this.stmt(
					`INSERT INTO midterm_episodes_fts (episode_id, tenant_id, user_id, user_message, assistant_summary)
					 VALUES (?, ?, ?, ?, ?)`,
				).run(id, episode.tenantId, episode.userId, episode.userMessage, episode.assistantSummary);
			}

			const existing = this.stmt(
				`SELECT * FROM midterm_episodes
				 WHERE tenant_id = ? AND user_id = ? AND conversation_id = ? AND run_id = ?`,
			).get(episode.tenantId, episode.userId, episode.conversationId, episode.runId) as EpisodeRow;

			return rowToEpisode(existing);
		});
	}

	async listTenantsWithPendingEpisodes(options?: {
		readonly limit?: number;
		readonly minEpisodes?: number;
	}): Promise<PendingEpisodeScope[]> {
		const limit = Math.max(1, Math.min(options?.limit ?? 50, 500));
		const minEpisodes = Math.max(1, options?.minEpisodes ?? 1);
		const scopes = this.stmt(
			`SELECT tenant_id, user_id, COUNT(*) AS pending, MAX(created_at) AS latest
			 FROM midterm_episodes WHERE consolidated = 0
			 GROUP BY tenant_id, user_id
			 HAVING COUNT(*) >= ?
			 ORDER BY latest DESC, tenant_id ASC, user_id ASC
			 LIMIT ?`,
		).all(minEpisodes, limit) as Array<{ tenant_id: string; user_id: string; pending: number; latest: number }>;
		const convStmt = this.stmt(
			`SELECT conversation_id FROM midterm_episodes
			 WHERE tenant_id = ? AND user_id = ? AND consolidated = 0
			 GROUP BY conversation_id ORDER BY MAX(created_at) DESC, conversation_id DESC LIMIT 5`,
		);
		return scopes.map((row) => ({
			tenantId: row.tenant_id,
			userId: row.user_id,
			pendingCount: Number(row.pending),
			latestEpisodeAt: Number(row.latest),
			conversationIds: (convStmt.all(row.tenant_id, row.user_id) as Array<{ conversation_id: string }>).map(
				(c) => c.conversation_id,
			),
		}));
	}

	async listUnconsolidatedEpisodes(tenantId: string, userId: string, limit = 50): Promise<MidTermEpisode[]> {
		const rows = this.stmt(
			`SELECT * FROM midterm_episodes
			 WHERE tenant_id = ? AND user_id = ? AND consolidated = 0
			 ORDER BY created_at ASC, id ASC
			 LIMIT ?`,
		).all(tenantId, userId, limit) as EpisodeRow[];
		return rows.map(rowToEpisode);
	}

	async listEpisodes(tenantId: string, userId: string, limit = 100): Promise<MidTermEpisode[]> {
		const rows = this.stmt(
			`SELECT * FROM midterm_episodes
			 WHERE tenant_id = ? AND user_id = ?
			 ORDER BY created_at DESC, id DESC
			 LIMIT ?`,
		).all(tenantId, userId, limit) as EpisodeRow[];
		return rows.map(rowToEpisode);
	}

	async markEpisodesConsolidated(episodeIds: readonly string[]): Promise<void> {
		if (episodeIds.length === 0) return;
		this.appDb.transaction(() => {
			const mark = this.stmt("UPDATE midterm_episodes SET consolidated = 1 WHERE id = ?");
			const dropFts = this.stmt("DELETE FROM midterm_episodes_fts WHERE episode_id = ?");
			for (const id of episodeIds) {
				mark.run(id);
				dropFts.run(id);
			}
		});
	}

	/**
	 * Applies Cognee-style streaming utility feedback (`w' = clamp(w + alpha * (rating - w), 0, 1)`)
	 * to in-scope memories so consistently helpful memories naturally outrank noisy ones.
	 */
	async applyFeedback(
		tenantId: string,
		userId: string,
		memoryIds: readonly string[],
		rating: number,
		alpha = 0.25,
	): Promise<number> {
		if (memoryIds.length === 0) return 0;
		const r = Math.max(0, Math.min(1, Number.isFinite(rating) ? rating : 0.5));
		const a = Math.max(0.01, Math.min(1, Number.isFinite(alpha) ? alpha : 0.25));
		return this.appDb.transaction(() => {
			const upd = this.stmt(
				`UPDATE longterm_memories
				 SET feedback_weight = MAX(0.0, MIN(1.0, feedback_weight + ? * (? - feedback_weight)))
				 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status != 'deleted'`,
			);
			let changed = 0;
			for (const id of memoryIds) {
				changed += Number(upd.run(a, r, id, tenantId, userId).changes);
			}
			return changed;
		});
	}

	/** Records retrieval access counts & timestamps for in-scope memories. */
	async recordAccess(tenantId: string, userId: string, memoryIds: readonly string[]): Promise<void> {
		if (memoryIds.length === 0) return;
		const ts = this.now();
		this.appDb.transaction(() => {
			const upd = this.stmt(
				`UPDATE longterm_memories
				 SET access_count = access_count + 1, last_accessed_at = ?
				 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status != 'deleted'`,
			);
			for (const id of memoryIds) {
				upd.run(ts, id, tenantId, userId);
			}
		});
	}

	async search(
		tenantId: string,
		userId: string,
		query: string,
		options?: {
			readonly limit?: number;
			readonly includeSuperseded?: boolean;
			readonly kind?: MemoryKind;
			readonly includeEpisodes?: boolean;
			readonly useGraph?: boolean;
			readonly temporalMode?: "auto" | "off";
			readonly minScore?: number;
		},
	): Promise<MemorySearchResult[]> {
		const limit = Math.max(1, Math.min(options?.limit ?? 10, 100));
		const temporalMode = options?.temporalMode ?? "auto";
		const useGraph = options?.useGraph ?? true;
		const includeEpisodes = options?.includeEpisodes ?? false;
		const nowMs = this.now();

		// Detect explicit historical-audit queries (e.g. "before bun?", "originally", "prior to") ONLY when caller
		// did not explicitly pin `includeSuperseded`.
		const historicalIntent =
			temporalMode === "auto" ? detectHistoricalIntent(query) : { isHistorical: false, excludeCurrentHintTokens: [] };
		const includeSuperseded =
			options?.includeSuperseded !== undefined ? options.includeSuperseded : historicalIntent.isHistorical;
		const autoHistoricalRouting = options?.includeSuperseded === undefined && historicalIntent.isHistorical;

		const statusClause = includeSuperseded ? "m.status IN ('active', 'superseded')" : "m.status = 'active'";
		const kindClause = options?.kind ? " AND m.kind = ?" : "";
		const poolSize = Math.max(50, limit * 5);
		const { fts, tokens } = buildFtsQuery(query);
		const queryTemporalAnchors = temporalMode === "auto" ? extractTemporalAnchors(query, nowMs) : [];

		type Candidate = {
			row: MemoryRow;
			lexical: number;
			graph: number;
			temporal: number;
			temporalConflict: boolean;
			snippet: string | undefined;
			historicalLineageBoost: number;
		};
		const candidates = new Map<string, Candidate>();
		const upsertCandidate = (row: MemoryRow): Candidate => {
			let c = candidates.get(row.id);
			if (!c) {
				c = {
					row,
					lexical: 0,
					graph: 0,
					temporal: 0,
					temporalConflict: false,
					snippet: undefined,
					historicalLineageBoost: 0,
				};
				candidates.set(row.id, c);
			}
			return c;
		};

		// ── Lane 1: Calibrated Lexical Search (FTS5 BM25 + query coverage; LIKE fallback) ──
		if (fts) {
			const params: (string | number)[] = [fts, tenantId, userId];
			if (options?.kind) params.push(options.kind);
			params.push(poolSize);

			const sql = `SELECT m.*, bm25(longterm_memories_fts, 0.0, 0.0, 0.0, 3.0, 1.0, 0.5) AS rank,
				        snippet(longterm_memories_fts, 4, '[', ']', '...', 12) AS snippet
				 FROM longterm_memories_fts f
				 JOIN longterm_memories m ON m.id = f.memory_id
				 WHERE longterm_memories_fts MATCH ?
				   AND m.tenant_id = ?
				   AND m.user_id = ?
				   AND ${statusClause}${kindClause}
				 ORDER BY rank ASC, m.pinned DESC, m.confidence DESC, m.updated_at DESC
				 LIMIT ?`;

			const rows = this.stmt(sql).all(...params) as Array<MemoryRow & { rank: number; snippet: string }>;
			for (const row of rows) {
				const lexical = calibrateLexicalScore({
					bm25Magnitude: Math.max(0, -Number(row.rank)),
					query,
					queryTokens: tokens,
					docKey: row.mem_key,
					docContent: row.content,
				});
				const c = upsertCandidate(row);
				c.lexical = Math.max(c.lexical, lexical);
				if (!c.snippet && row.snippet) c.snippet = row.snippet;
			}
		}

		// LIKE fallback (also covers punctuation/non-FTS queries like "%") when FTS found nothing
		if (candidates.size === 0 && query.trim().length > 0) {
			const terms = tokens.length > 0 ? Array.from(new Set(tokens.flatMap(expandToken))) : [];
			const rawNeedle = query.trim().toLowerCase();
			const patterns = terms.length > 0 ? terms : rawNeedle ? [rawNeedle] : [""];
			const likeClause = patterns
				.map(() => "(LOWER(m.mem_key) LIKE ? ESCAPE '\\' OR LOWER(m.content) LIKE ? ESCAPE '\\')")
				.join(" OR ");
			const fallbackParams: (string | number)[] = [tenantId, userId];
			for (const pattern of patterns) {
				const like = `%${escapeLike(pattern)}%`;
				fallbackParams.push(like, like);
			}
			if (options?.kind) fallbackParams.push(options.kind);
			fallbackParams.push(poolSize);

			const fallbackRows = this.appDb.db
				.prepare(
					`SELECT m.*
					 FROM longterm_memories m
					 WHERE m.tenant_id = ?
					   AND m.user_id = ?
					   AND (${likeClause})
					   AND ${statusClause}${kindClause}
					 ORDER BY m.pinned DESC, m.confidence DESC, m.updated_at DESC
					 LIMIT ?`,
				)
				.all(...fallbackParams) as MemoryRow[];

			for (const row of fallbackRows) {
				const lexical = calibrateLexicalScore({
					bm25Magnitude: 0,
					query,
					queryTokens: tokens,
					docKey: row.mem_key,
					docContent: row.content,
				});
				const c = upsertCandidate(row);
				c.lexical = Math.max(c.lexical, lexical);
			}
		}

		// ── Lane 2: Cached Packed-Float32 Vector Similarity Search (scans 100% up to maxVectorCandidates) ──
		let cosines: Map<string, number> | undefined;
		if (this.embeddingProvider && query.trim().length > 0) {
			try {
				const vecResult = await this.vectorCandidates(
					tenantId,
					userId,
					query,
					includeSuperseded,
					options?.kind,
					poolSize,
				);
				cosines = vecResult.allCosines;
				for (const hit of vecResult.topHits) {
					// Only add vector-only candidates above background cosine floor
					if (hit.cosine > 0.24 || candidates.has(hit.row.id)) {
						upsertCandidate(hit.row);
					}
				}
			} catch {
				cosines = undefined; // degrade gracefully to lexical + graph + temporal
			}
		}

		// ── Lane 3: Temporal Window Anchors (interval overlap + tightness scoring) ──
		if (queryTemporalAnchors.length > 0) {
			this.ensureGraphAndTemporalBackfilled(tenantId, userId);
			let minStart = Number.POSITIVE_INFINITY;
			let maxEnd = Number.NEGATIVE_INFINITY;
			for (const qa of queryTemporalAnchors) {
				if (qa.startMs < minStart) minStart = qa.startMs;
				if (qa.endMs > maxEnd) maxEnd = qa.endMs;
			}

			const tParams: (string | number)[] = [tenantId, userId, maxEnd, minStart];
			if (options?.kind) tParams.push(options.kind);
			tParams.push(poolSize);

			const temporalRows = this.stmt(
				`SELECT DISTINCT m.*
				 FROM memory_temporal_anchors a
				 JOIN longterm_memories m ON m.id = a.memory_id
				 WHERE a.tenant_id = ?
				   AND a.user_id = ?
				   AND a.start_ms <= ?
				   AND a.end_ms >= ?
				   AND ${statusClause}${kindClause}
				 ORDER BY m.pinned DESC, m.confidence DESC, m.updated_at DESC
				 LIMIT ?`,
			).all(...tParams) as MemoryRow[];

			for (const row of temporalRows) {
				upsertCandidate(row);
			}

			// Compute exact temporal tightness alignment for every candidate
			const anchorStmt = this.stmt(
				`SELECT start_ms, end_ms, precision, label
				 FROM memory_temporal_anchors
				 WHERE memory_id = ? AND tenant_id = ? AND user_id = ?`,
			);
			for (const cand of candidates.values()) {
				const aRows = anchorStmt.all(cand.row.id, tenantId, userId) as Array<{
					start_ms: number;
					end_ms: number;
					precision: string;
					label: string;
				}>;
				if (aRows.length === 0) continue;
				const memAnchors: TemporalAnchor[] = aRows.map((a) => ({
					startMs: Number(a.start_ms),
					endMs: Number(a.end_ms),
					precision: a.precision as TemporalPrecision,
					label: a.label,
				}));
				const align = scoreTemporalAlignment(queryTemporalAnchors, memAnchors);
				cand.temporal = align.score;
				cand.temporalConflict = align.conflict;
			}
		}

		// ── Lane 4: Entity-Graph Multi-Hop Personalized PageRank (PPR) ──
		if (useGraph && query.trim().length > 0) {
			const graphScope = this.getCachedGraphScope(tenantId, userId, includeSuperseded, options?.kind);
			if (graphScope.graph.knownCanonicals.size > 0) {
				const queryEntities = extractQueryEntities(query, graphScope.graph.knownCanonicals);
				const hasVector = cosines !== undefined;
				const seedFloor = hasVector ? 0.34 : 0.26;

				// Rank current candidates by base relevance to pick high-confidence seeds for 2-hop bridging
				const seedPool: Array<{ memoryId: string; seedScore: number }> = [];
				for (const cand of candidates.values()) {
					const vecCal = cosines ? calibrateCosine(cosines.get(cand.row.id) ?? 0) : 0;
					const baseRel = Math.max(cand.lexical, vecCal, cand.temporal * 0.72);
					if (baseRel >= seedFloor) {
						seedPool.push({ memoryId: cand.row.id, seedScore: baseRel });
					}
				}
				seedPool.sort((a, b) => b.seedScore - a.seedScore);

				const ppr = runEntityPpr(graphScope.graph, {
					queryEntities,
					seedMemories: seedPool.slice(0, 5),
					allowedMemoryIds: graphScope.allowedIds,
				});

				if (ppr.scores.size > 0) {
					const rowByIdStmt = this.stmt(
						`SELECT * FROM longterm_memories WHERE id = ? AND tenant_id = ? AND user_id = ?`,
					);
					for (const [memId, pprScore] of ppr.scores.entries()) {
						const existing = candidates.get(memId);
						if (existing) {
							existing.graph = pprScore;
						} else if (pprScore >= 0.16) {
							const row = rowByIdStmt.get(memId, tenantId, userId) as MemoryRow | undefined;
							if (row) {
								const c = upsertCandidate(row);
								c.graph = pprScore;
							}
						}
					}
				}
			}
		}

		// ── Historical-Audit Lineage Walk (pull superseded predecessors of matched active memories) ──
		if (autoHistoricalRouting && candidates.size > 0) {
			const predStmt = this.stmt(
				`SELECT * FROM longterm_memories
				 WHERE superseded_by_id = ? AND tenant_id = ? AND user_id = ? AND status = 'superseded'`,
			);
			const seedActive = Array.from(candidates.values())
				.map((c) => {
					const vecCal = cosines ? calibrateCosine(cosines.get(c.row.id) ?? 0) : 0;
					return { cand: c, rel: Math.max(c.lexical, vecCal, c.graph * 0.8) };
				})
				.filter((x) => x.rel >= 0.25)
				.sort((a, b) => b.rel - a.rel)
				.slice(0, 5);

			for (const { cand, rel } of seedActive) {
				let curId = cand.row.id;
				let depth = 0;
				while (depth < 5) {
					const pred = predStmt.get(curId, tenantId, userId) as MemoryRow | undefined;
					if (!pred) break;
					if (!options?.kind || pred.kind === options.kind) {
						const pc = upsertCandidate(pred);
						// Inherit relevance along the verified superseded_by_id chain (newest predecessor gets highest boost)
						const inherited = rel * (depth === 0 ? 0.92 : 0.82);
						pc.historicalLineageBoost = Math.max(pc.historicalLineageBoost, inherited);
					}
					curId = pred.id;
					depth += 1;
				}
			}
		}

		// ── Final Multi-Lane Fusion + Abstention Filtering ──
		const hasVector = cosines !== undefined;
		const minScore =
			options?.minScore !== undefined
				? options.minScore
				: this.ranking.minScore !== undefined
					? this.ranking.minScore
					: hasVector
						? 0.24
						: 0.18;

		const scored: Array<{
			memory: LongTermMemory;
			score: number;
			snippet: string | undefined;
			source: "memory" | "episode";
			matchSignals: NonNullable<MemorySearchResult["matchSignals"]>;
		}> = [];

		for (const cand of candidates.values()) {
			const memory = rowToMemory(cand.row);
			const rawCosine = cosines?.get(memory.id);
			const effectiveLexical = Math.max(cand.lexical, cand.historicalLineageBoost);

			let historicalMultiplier = 1;
			if (autoHistoricalRouting) {
				if (memory.status === "superseded") {
					historicalMultiplier = 1.26;
				} else if (historicalIntent.excludeCurrentHintTokens.length > 0) {
					const lowerText = `${memory.key} ${memory.content}`.toLowerCase();
					const mentionsExcludedCurrent = historicalIntent.excludeCurrentHintTokens.some((tok) =>
						lowerText.includes(tok),
					);
					if (mentionsExcludedCurrent) {
						historicalMultiplier = 0.7;
					}
				}
			}

			const score = fuseRetrievalSignals(
				{
					lexical: effectiveLexical,
					...(hasVector ? { cosine: rawCosine ?? 0 } : {}),
					graph: cand.graph,
					temporal: cand.temporal,
					temporalConflict: cand.temporalConflict,
					historicalMultiplier,
				},
				memory,
				nowMs,
				this.ranking,
			);

			if (score >= minScore) {
				scored.push({
					memory,
					score,
					snippet: cand.snippet,
					source: "memory",
					matchSignals: {
						lexical: effectiveLexical,
						...(hasVector ? { vector: rawCosine ?? 0 } : {}),
						...(cand.graph > 0 ? { graph: cand.graph } : {}),
						...(cand.temporal > 0 ? { temporal: cand.temporal } : {}),
					},
				});
			}
		}

		// ── Lane 5: Recent Unconsolidated Mid-Term Episodes (live session recall before sleep cycle) ──
		if (includeEpisodes && (!options?.kind || options.kind === "episodic") && fts) {
			const epRows = this.stmt(
				`SELECT e.*, bm25(midterm_episodes_fts, 0.0, 0.0, 0.0, 1.5, 1.0) AS rank,
				        snippet(midterm_episodes_fts, 3, '[', ']', '...', 14) AS snippet
				 FROM midterm_episodes_fts f
				 JOIN midterm_episodes e ON e.id = f.episode_id
				 WHERE midterm_episodes_fts MATCH ?
				   AND e.tenant_id = ?
				   AND e.user_id = ?
				   AND e.consolidated = 0
				 ORDER BY rank ASC, e.created_at DESC
				 LIMIT ?`,
			).all(fts, tenantId, userId, Math.max(10, limit * 2)) as Array<EpisodeRow & { rank: number; snippet: string }>;

			for (const ep of epRows) {
				const combinedText = `${ep.user_message} | ${ep.assistant_summary}`;
				const epKey = `episode.${ep.conversation_id}.${ep.run_id}`;
				const lex = calibrateLexicalScore({
					bm25Magnitude: Math.max(0, -Number(ep.rank)),
					query,
					queryTokens: tokens,
					docKey: epKey,
					docContent: combinedText,
				});
				// Episodes are live recent turns: keep high precision (require strong lexical match)
				const epMinScore = Math.max(minScore, 0.26);
				const recencyBoost = 0.96 + 0.14 * Math.pow(0.5, Math.max(0, nowMs - Number(ep.created_at)) / 86_400_000);
				const score = Math.min(1, lex * recencyBoost);
				if (score >= epMinScore) {
					const epMemory: LongTermMemory = {
						id: ep.id,
						tenantId: ep.tenant_id,
						userId: ep.user_id,
						kind: "episodic",
						key: epKey,
						content: combinedText,
						confidence: 0.85,
						pinned: false,
						status: "active",
						evidenceCount: 1,
						distinctConversations: 1,
						sourceConversationId: ep.conversation_id,
						createdAt: Number(ep.created_at),
						updatedAt: Number(ep.created_at),
					};
					scored.push({
						memory: epMemory,
						score,
						snippet: ep.snippet || combinedText.slice(0, 160),
						source: "episode",
						matchSignals: { lexical: lex },
					});
				}
			}
		}

		scored.sort(
			(a, b) =>
				b.score - a.score ||
				Number(b.memory.pinned) - Number(a.memory.pinned) ||
				b.memory.updatedAt - a.memory.updatedAt ||
				(a.memory.id < b.memory.id ? -1 : a.memory.id > b.memory.id ? 1 : 0),
		);

		return scored.slice(0, limit).map((entry) => ({
			memory: entry.memory,
			rank: -entry.score,
			score: entry.score,
			snippet: entry.snippet || entry.memory.content.slice(0, 160),
			source: entry.source,
			matchSignals: entry.matchSignals,
		}));
	}

	/**
	 * Loads or returns the version-coherent packed L2-normalized Float32 vector matrix for `(tenantId, userId, model)`.
	 * Zero SQLite reads, zero SHA-1 hashing, and zero Float32 allocations on cache hits!
	 */
	private getCachedVectorScope(
		tenantId: string,
		userId: string,
		model: string,
		includeSuperseded: boolean,
		kind: MemoryKind | undefined,
	): CachedVectorScope {
		const cacheKey = `${tenantId}\0${userId}\0${model}\0${includeSuperseded ? "1" : "0"}\0${kind ?? ""}`;
		const stamp = this.dbStamp();
		const cached = this.vectorCache.get(cacheKey);
		if (cached && cached.stamp === stamp) {
			return cached;
		}

		const statusClause = includeSuperseded ? "m.status IN ('active', 'superseded')" : "m.status = 'active'";
		const params: (string | number)[] = [model, tenantId, userId];
		if (kind) params.push(kind);
		params.push(this.maxVectorCandidates);

		const sql = `SELECT m.*, e.vector AS e_vector, e.content_hash AS e_hash
			 FROM longterm_memories m
			 LEFT JOIN memory_embeddings e
			   ON e.memory_id = m.id AND e.model = ? AND e.tenant_id = m.tenant_id AND e.user_id = m.user_id
			 WHERE m.tenant_id = ? AND m.user_id = ? AND ${statusClause}${kind ? " AND m.kind = ?" : ""}
			 ORDER BY m.pinned DESC, m.updated_at DESC, m.id ASC
			 LIMIT ?`;

		const rows = this.stmt(sql).all(...params) as Array<
			MemoryRow & { e_vector: Uint8Array | null; e_hash: string | null }
		>;

		const validRows: MemoryRow[] = [];
		const validVecs: Float32Array[] = [];
		const stale: Array<{ row: MemoryRow; text: string; hash: string }> = [];
		let dim = 0;

		for (const row of rows) {
			const text = embeddingText(row.mem_key, row.content);
			const hash = hashText(text);
			const decoded = row.e_vector && row.e_hash === hash ? tryDecodeVector(row.e_vector) : undefined;
			const normalized = decoded ? l2Normalize(decoded) : null;
			if (normalized) {
				if (dim === 0) dim = normalized.length;
				if (normalized.length === dim) {
					validRows.push(row);
					validVecs.push(normalized);
					continue;
				}
			}
			stale.push({ row, text, hash });
		}

		const matrix = new Float32Array(validRows.length * dim);
		for (let i = 0; i < validVecs.length; i++) {
			matrix.set(validVecs[i]!, i * dim);
		}

		const scope: CachedVectorScope = { stamp, dim, rows: validRows, matrix, stale };
		if (this.vectorCache.size > 32) this.vectorCache.clear();
		this.vectorCache.set(cacheKey, scope);
		return scope;
	}

	/**
	 * Embeds (lazily, bounded per call) and scores in-scope memories against the query using
	 * the cached packed Float32 matrix for sub-millisecond scans up to 10,000+ memories.
	 */
	private async vectorCandidates(
		tenantId: string,
		userId: string,
		query: string,
		includeSuperseded: boolean,
		kind: MemoryKind | undefined,
		poolSize: number,
	): Promise<{
		readonly topHits: Array<{ row: MemoryRow; cosine: number }>;
		readonly allCosines: Map<string, number>;
	}> {
		const provider = this.embeddingProvider;
		if (!provider) return { topHits: [], allCosines: new Map() };

		const scope = this.getCachedVectorScope(tenantId, userId, provider.model, includeSuperseded, kind);
		const toEmbed = scope.stale.slice(0, this.maxBackfillPerSearch);
		const texts = [query, ...toEmbed.map((entry) => entry.text)];
		const embedded = await provider.embed(texts);
		if (embedded.length !== texts.length) throw new Error("EmbeddingProvider returned a mismatched vector count");
		const queryVector = embedded[0];
		if (!isUsableVector(queryVector)) throw new Error("EmbeddingProvider returned an unusable (empty/non-finite) query vector");
		const normQuery = l2Normalize(queryVector);
		if (!normQuery) throw new Error("EmbeddingProvider returned a zero-norm query vector");

		const fresh: Array<{ id: string; hash: string; vec: ArrayLike<number> }> = [];
		const extraHits: Array<{ row: MemoryRow; cosine: number }> = [];
		const allCosines = new Map<string, number>();

		toEmbed.forEach((entry, idx) => {
			const vec = embedded[idx + 1];
			if (!isUsableVector(vec)) return;
			const normDoc = l2Normalize(vec);
			if (normDoc && normDoc.length === normQuery.length) {
				const cos = dotProductOffset(normQuery, normDoc, 0);
				allCosines.set(entry.row.id, cos);
				extraHits.push({ row: entry.row, cosine: cos });
			}
			fresh.push({ id: entry.row.id, hash: entry.hash, vec });
		});

		if (fresh.length > 0) {
			this.persistEmbeddings(tenantId, userId, provider.model, fresh);
		}

		const hits: Array<{ row: MemoryRow; cosine: number }> = [...extraHits];
		if (scope.dim === normQuery.length && scope.rows.length > 0) {
			const { matrix, rows, dim } = scope;
			for (let i = 0; i < rows.length; i++) {
				const cos = dotProductOffset(normQuery, matrix, i * dim);
				const row = rows[i]!;
				allCosines.set(row.id, cos);
				if (cos > 0.12) {
					hits.push({ row, cosine: cos });
				}
			}
		}

		hits.sort((a, b) => b.cosine - a.cosine || (a.row.id < b.row.id ? -1 : 1));
		return { topHits: hits.slice(0, Math.max(poolSize, 1) * 2), allCosines };
	}

	/**
	 * Computes and stores embeddings for in-scope memories that lack a fresh vector.
	 * No-op without an EmbeddingProvider. Returns the number of vectors written.
	 */
	async backfillEmbeddings(
		tenantId: string,
		userId: string,
		options?: { readonly limit?: number; readonly includeSuperseded?: boolean },
	): Promise<number> {
		const provider = this.embeddingProvider;
		if (!provider) return 0;
		const limit = Math.max(1, Math.min(options?.limit ?? 500, 10_000));
		const statusClause =
			options?.includeSuperseded === false ? "m.status = 'active'" : "m.status IN ('active', 'superseded')";
		const rows = this.stmt(
			`SELECT m.*, e.content_hash AS e_hash
			 FROM longterm_memories m
			 LEFT JOIN memory_embeddings e ON e.memory_id = m.id AND e.model = ?
			 WHERE m.tenant_id = ? AND m.user_id = ? AND ${statusClause}
			 ORDER BY m.pinned DESC, m.updated_at DESC, m.id ASC
			 LIMIT ?`,
		).all(provider.model, tenantId, userId, limit) as Array<MemoryRow & { e_hash: string | null }>;
		const todo = rows
			.map((row) => {
				const text = embeddingText(row.mem_key, row.content);
				return { row, text, hash: hashText(text) };
			})
			.filter((entry) => entry.row.e_hash !== entry.hash);
		if (todo.length === 0) return 0;
		const embedded = await provider.embed(todo.map((entry) => entry.text));
		if (embedded.length !== todo.length) throw new Error("EmbeddingProvider returned a mismatched vector count");
		const fresh: Array<{ id: string; hash: string; vec: ArrayLike<number> }> = [];
		todo.forEach((entry, idx) => {
			const vec = embedded[idx];
			if (isUsableVector(vec)) fresh.push({ id: entry.row.id, hash: entry.hash, vec });
		});
		return this.persistEmbeddings(tenantId, userId, provider.model, fresh);
	}

	/**
	 * Stores embeddings computed OUTSIDE a transaction (the provider call is async). The memory may have been
	 * forgotten or edited meanwhile, so each row is re-checked inside the write transaction: only non-deleted
	 * rows whose current text still hashes to the embedded text are written.
	 */
	private persistEmbeddings(
		tenantId: string,
		userId: string,
		model: string,
		items: ReadonlyArray<{ readonly id: string; readonly hash: string; readonly vec: ArrayLike<number> }>,
	): number {
		if (items.length === 0) return 0;
		const ts = this.now();
		return this.appDb.transaction(() => {
			const current = this.stmt(
				"SELECT mem_key, content, status FROM longterm_memories WHERE id = ? AND tenant_id = ? AND user_id = ?",
			);
			const upsert = this.stmt(UPSERT_EMBEDDING_SQL);
			let written = 0;
			for (const item of items) {
				const row = current.get(item.id, tenantId, userId) as
					| { mem_key: string; content: string; status: string }
					| undefined;
				if (!row || row.status === "deleted") continue;
				if (hashText(embeddingText(row.mem_key, row.content)) !== item.hash) continue;
				upsert.run(item.id, tenantId, userId, model, item.vec.length, item.hash, encodeVector(item.vec), ts);
				written += 1;
			}
			return written;
		});
	}

	async getCoreMemory(tenantId: string, userId: string, limit = 20): Promise<LongTermMemory[]> {
		const rows = this.stmt(
			`SELECT * FROM longterm_memories
			 WHERE tenant_id = ?
			   AND user_id = ?
			   AND status = 'active'
			   AND (pinned = 1 OR confidence >= 0.7)
			 ORDER BY pinned DESC,
			          (confidence * 0.55 + MIN(1.0, evidence_count / 8.0) * 0.20 + MIN(1.0, distinct_conversations / 4.0) * 0.10 + feedback_weight * 0.15) DESC,
			          created_at ASC,
			          id ASC
			 LIMIT ?`,
		).all(tenantId, userId, limit) as MemoryRow[];
		return rows.map(rowToMemory);
	}

	async getMemoryById(tenantId: string, userId: string, id: string): Promise<LongTermMemory | undefined> {
		const row = this.stmt("SELECT * FROM longterm_memories WHERE id = ? AND tenant_id = ? AND user_id = ?").get(
			id,
			tenantId,
			userId,
		) as MemoryRow | undefined;
		return row ? rowToMemory(row) : undefined;
	}

	async listMemories(
		tenantId: string,
		userId: string,
		options?: {
			readonly includeSuperseded?: boolean;
			readonly includeDeleted?: boolean;
			readonly kind?: MemoryKind;
			readonly limit?: number;
		},
	): Promise<LongTermMemory[]> {
		const statuses: string[] = ["'active'"];
		if (options?.includeSuperseded) statuses.push("'superseded'");
		if (options?.includeDeleted) statuses.push("'deleted'");

		const params: (string | number)[] = [tenantId, userId];
		const kindClause = options?.kind ? " AND kind = ?" : "";
		if (options?.kind) params.push(options.kind);
		params.push(options?.limit ?? 200);

		const rows = this.stmt(
			`SELECT * FROM longterm_memories
			 WHERE tenant_id = ?
			   AND user_id = ?
			   AND status IN (${statuses.join(", ")})${kindClause}
			 ORDER BY pinned DESC, updated_at DESC, id ASC
			 LIMIT ?`,
		).all(...params) as MemoryRow[];
		return rows.map(rowToMemory);
	}

	async pinMemory(
		tenantId: string,
		userId: string,
		input: {
			readonly key: string;
			readonly content: string;
			readonly kind?: MemoryKind;
			readonly confidence?: number;
			readonly sourceConversationId?: string;
		},
	): Promise<LongTermMemory> {
		const normalizedKey = sanitizeMemoryKey(input.key);
		const content = sanitizeMemoryText(input.content).text;
		if (!normalizedKey) throw new Error("Memory key is empty after sanitization");
		if (!content) throw new Error("Memory content is empty after sanitization");
		const kind: MemoryKind = input.kind ?? "preference";
		const confidence = Math.max(0, Math.min(1, input.confidence ?? 1.0));

		return this.appDb.transaction(() => {
			const existing = this.stmt(
				`SELECT * FROM longterm_memories
				 WHERE tenant_id = ? AND user_id = ? AND mem_key = ? AND status = 'active'
				 ORDER BY updated_at DESC LIMIT 1`,
			).get(tenantId, userId, normalizedKey) as MemoryRow | undefined;

			if (existing && existing.content.trim() === content) {
				const ts = this.now();
				this.stmt(
					`UPDATE longterm_memories
					 SET pinned = 1, confidence = MAX(confidence, ?), updated_at = ?
					 WHERE id = ?`,
				).run(confidence, ts, existing.id);
				const updated = this.stmt("SELECT * FROM longterm_memories WHERE id = ?").get(existing.id) as MemoryRow;
				return rowToMemory(updated);
			}

			const newId = `mem_${crypto.randomUUID()}`;
			const ts = this.now();
			this.insertMemoryRow({
				id: newId,
				tenantId,
				userId,
				kind,
				key: normalizedKey,
				content,
				confidence,
				pinned: true,
				evidenceCount: existing ? Number(existing.evidence_count) + 1 : 1,
				distinctConversations: existing ? Number(existing.distinct_conversations) : 1,
				sourceConversationId: input.sourceConversationId,
				timestamp: ts,
			});

			if (existing) {
				this.copyEvidence(existing.id, newId);
				this.stmt(
					`UPDATE longterm_memories
					 SET status = 'superseded', superseded_by_id = ?, valid_until = ?, updated_at = ?
					 WHERE id = ?`,
				).run(newId, ts, ts, existing.id);
				this.stmt("UPDATE memory_relations SET valid_until = ? WHERE memory_id = ? AND valid_until IS NULL").run(
					ts,
					existing.id,
				);
			}

			const created = this.stmt("SELECT * FROM longterm_memories WHERE id = ?").get(newId) as MemoryRow;
			return rowToMemory(created);
		});
	}

	async applyMemoryMutation(tenantId: string, userId: string, op: MemoryMutationOp): Promise<LongTermMemory | null> {
		if (op.op === "NOOP") return null;
		return this.appDb.transaction(() => this.applyMutationTx(tenantId, userId, op, []));
	}

	/**
	 * Atomically applies consolidation mutations and marks the supporting episodes consolidated in ONE
	 * transaction. A crash/throw anywhere rolls back everything (episodes stay unconsolidated); episodes that
	 * are already consolidated are skipped so a re-run is a no-op.
	 */
	async applyConsolidation(
		tenantId: string,
		userId: string,
		input: { readonly ops: readonly MemoryMutationOp[]; readonly episodeIds: readonly string[] },
	): Promise<{ readonly applied: boolean; readonly results: ReadonlyArray<LongTermMemory | null> }> {
		return this.appDb.transaction(() => {
			const pending = this.filterPendingEpisodeIds(tenantId, userId, input.episodeIds);
			// All-or-nothing: the ops were planned from the whole batch. If any episode has since been consolidated or
			// scrubbed (e.g. its memory was forgotten), applying them to the rest could resurrect that content, so the
			// consolidator must re-plan from what is still pending.
			if (pending.length === 0 || pending.length < new Set(input.episodeIds).size) {
				return { applied: false, results: input.ops.map(() => null) };
			}

			const results: Array<LongTermMemory | null> = [];
			for (const op of input.ops) {
				if (op.op === "NOOP") {
					results.push(null);
					continue;
				}
				const withEvidence: MemoryMutationOp = { ...op, evidenceEpisodeIds: op.evidenceEpisodeIds ?? pending };
				results.push(this.applyMutationTx(tenantId, userId, withEvidence, results));
			}

			this.afterConsolidationOps(pending);

			const mark = this.stmt(
				"UPDATE midterm_episodes SET consolidated = 1 WHERE id = ? AND tenant_id = ? AND user_id = ?",
			);
			const dropEpFts = this.stmt("DELETE FROM midterm_episodes_fts WHERE episode_id = ?");
			for (const id of pending) {
				mark.run(id, tenantId, userId);
				dropEpFts.run(id);
			}
			return { applied: true, results };
		});
	}

	/**
	 * Hook invoked inside the consolidation transaction after all memory ops were written and before the
	 * episodes are marked. Throwing here rolls the whole batch back (used to simulate crashes in tests).
	 */
	protected afterConsolidationOps(_episodeIds: readonly string[]): void {}

	private filterPendingEpisodeIds(tenantId: string, userId: string, episodeIds: readonly string[]): string[] {
		const stmt = this.stmt(
			"SELECT 1 AS ok FROM midterm_episodes WHERE id = ? AND tenant_id = ? AND user_id = ? AND consolidated = 0",
		);
		const seen = new Set<string>();
		const pending: string[] = [];
		for (const id of episodeIds) {
			if (seen.has(id)) continue;
			seen.add(id);
			if (stmt.get(id, tenantId, userId)) pending.push(id);
		}
		return pending;
	}

	private freshEvidenceIds(memoryId: string, episodeIds: readonly string[] | undefined): string[] {
		if (!episodeIds) return [];
		const stmt = this.stmt("SELECT 1 AS ok FROM memory_evidence WHERE memory_id = ? AND episode_id = ?");
		const fresh: string[] = [];
		const seen = new Set<string>();
		for (const id of episodeIds) {
			if (seen.has(id)) continue;
			seen.add(id);
			if (!stmt.get(memoryId, id)) fresh.push(id);
		}
		return fresh;
	}

	private recordEvidence(memoryId: string, episodeIds: readonly string[] | undefined, timestamp: number): void {
		if (!episodeIds || episodeIds.length === 0) return;
		const stmt = this.stmt(
			"INSERT OR IGNORE INTO memory_evidence (memory_id, episode_id, created_at) VALUES (?, ?, ?)",
		);
		for (const id of episodeIds) stmt.run(memoryId, id, timestamp);
	}

	private copyEvidence(fromMemoryId: string, toMemoryId: string): void {
		this.stmt(
			`INSERT OR IGNORE INTO memory_evidence (memory_id, episode_id, created_at)
			 SELECT ?, episode_id, created_at FROM memory_evidence WHERE memory_id = ?`,
		).run(toMemoryId, fromMemoryId);
	}

	/** Must run inside `appDb.transaction`. `batch` resolves `batch:<n>` target ids to earlier results. */
	private applyMutationTx(
		tenantId: string,
		userId: string,
		op: Exclude<MemoryMutationOp, { op: "NOOP" }>,
		batch: ReadonlyArray<LongTermMemory | null>,
	): LongTermMemory | null {
		const ts = this.now();
		const content = sanitizeMemoryText(op.content).text;
		if (!content) throw new Error("Memory content is empty after sanitization");

		if (op.op === "ADD") {
			const key = sanitizeMemoryKey(op.key);
			if (!key) throw new Error("Memory key is empty after sanitization");

			// Idempotent re-apply: same key + same content already active -> only record not-yet-counted evidence.
			if (op.evidenceEpisodeIds) {
				const duplicate = this.stmt(
					`SELECT * FROM longterm_memories
					 WHERE tenant_id = ? AND user_id = ? AND mem_key = ? AND status = 'active'
					 ORDER BY updated_at DESC, id ASC`,
				).all(tenantId, userId, key) as MemoryRow[];
				const same = duplicate.find((row) => normalizeForCompare(row.content) === normalizeForCompare(content));
				if (same) {
					const fresh = this.freshEvidenceIds(same.id, op.evidenceEpisodeIds);
					if (fresh.length > 0) {
						this.recordEvidence(same.id, fresh, ts);
						this.stmt("UPDATE longterm_memories SET evidence_count = evidence_count + ? WHERE id = ?").run(
							fresh.length,
							same.id,
						);
					}
					const current = this.stmt("SELECT * FROM longterm_memories WHERE id = ?").get(same.id) as MemoryRow;
					return rowToMemory(current);
				}
			}

			const newId = `mem_${crypto.randomUUID()}`;
			this.insertMemoryRow({
				id: newId,
				tenantId,
				userId,
				kind: op.kind,
				key,
				content,
				confidence: Math.max(0, Math.min(1, op.confidence)),
				pinned: op.pinned ?? false,
				evidenceCount: op.evidenceCount ?? 1,
				distinctConversations: op.distinctConversations ?? 1,
				sourceConversationId: op.sourceConversationId,
				timestamp: ts,
			});
			this.recordEvidence(newId, op.evidenceEpisodeIds, ts);
			const created = this.stmt("SELECT * FROM longterm_memories WHERE id = ?").get(newId) as MemoryRow;
			return rowToMemory(created);
		}

		const targetId = resolveBatchTarget(op.targetId, batch);

		if (op.op === "UPDATE") {
			const existing = this.stmt(
				`SELECT * FROM longterm_memories
				 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'`,
			).get(targetId, tenantId, userId) as MemoryRow | undefined;
			if (!existing) return null;

			const fresh = op.evidenceEpisodeIds ? this.freshEvidenceIds(existing.id, op.evidenceEpisodeIds) : undefined;
			const nextConfidence = Math.max(0, Math.min(1, op.confidence));
			const evidenceDelta = fresh ? fresh.length : (op.evidenceCountDelta ?? 1);
			const distinctDelta = fresh && fresh.length === 0 ? 0 : (op.distinctConversationsDelta ?? 0);
			const nextEvidence = Number(existing.evidence_count) + evidenceDelta;
			const nextDistinct = Number(existing.distinct_conversations) + distinctDelta;

			this.stmt(
				`UPDATE longterm_memories
				 SET content = ?, confidence = ?, evidence_count = ?, distinct_conversations = ?, updated_at = ?
				 WHERE id = ?`,
			).run(content, nextConfidence, nextEvidence, nextDistinct, ts, existing.id);
			this.recordEvidence(existing.id, fresh, ts);

			this.syncFts(existing.id, tenantId, userId, existing.mem_key, content, existing.kind);
			this.syncGraphAndTemporal(
				existing.id,
				tenantId,
				userId,
				existing.mem_key,
				content,
				Number(existing.valid_from ?? existing.created_at),
				null,
				ts,
			);

			const updated = this.stmt("SELECT * FROM longterm_memories WHERE id = ?").get(existing.id) as MemoryRow;
			return rowToMemory(updated);
		}

		// SUPERSEDE: create new active row and tombstone the old row with superseded_by_id & valid_until
		const key = sanitizeMemoryKey(op.key);
		if (!key) throw new Error("Memory key is empty after sanitization");
		const existing = this.stmt(
			`SELECT * FROM longterm_memories
			 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'`,
		).get(targetId, tenantId, userId) as MemoryRow | undefined;

		if (!existing && op.evidenceEpisodeIds) {
			const prior = this.stmt(
				`SELECT superseded_by_id FROM longterm_memories
				 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'superseded'`,
			).get(targetId, tenantId, userId) as { superseded_by_id: string | null } | undefined;
			if (prior?.superseded_by_id) {
				const successor = this.stmt(
					`SELECT * FROM longterm_memories
					 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'`,
				).get(prior.superseded_by_id, tenantId, userId) as MemoryRow | undefined;
				if (successor && normalizeForCompare(successor.content) === normalizeForCompare(content)) {
					return rowToMemory(successor);
				}
			}
		}

		if (!existing) return null;

		const fresh = op.evidenceEpisodeIds ? this.freshEvidenceIds(existing.id, op.evidenceEpisodeIds) : undefined;
		const evidenceDelta = fresh ? fresh.length : 1;
		const distinctDelta = fresh && fresh.length === 0 ? 0 : 1;

		const newId = `mem_${crypto.randomUUID()}`;
		this.insertMemoryRow({
			id: newId,
			tenantId,
			userId,
			kind: op.kind,
			key,
			content,
			confidence: Math.max(0, Math.min(1, op.confidence)),
			// Never inherit `pinned`: a pin is an explicit user act (the consolidator refuses pinned targets).
			pinned: false,
			evidenceCount: Number(existing.evidence_count) + evidenceDelta,
			distinctConversations: Number(existing.distinct_conversations) + distinctDelta,
			sourceConversationId: op.sourceConversationId,
			timestamp: ts,
		});

		this.copyEvidence(existing.id, newId);
		this.stmt(
			`UPDATE longterm_memories
			 SET status = 'superseded', superseded_by_id = ?, valid_until = ?, updated_at = ?
			 WHERE id = ?`,
		).run(newId, ts, ts, existing.id);
		this.stmt("UPDATE memory_relations SET valid_until = ? WHERE memory_id = ? AND valid_until IS NULL").run(
			ts,
			existing.id,
		);
		this.recordEvidence(newId, op.evidenceEpisodeIds, ts);

		const created = this.stmt("SELECT * FROM longterm_memories WHERE id = ?").get(newId) as MemoryRow;
		return rowToMemory(created);
	}

	async forgetMemory(tenantId: string, userId: string, memoryId: string): Promise<boolean> {
		return this.appDb.transaction(() => {
			const ts = this.now();
			const result = this.stmt(
				`UPDATE longterm_memories
				 SET status = 'deleted', content = '', valid_until = COALESCE(valid_until, ?), updated_at = ?
				 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status != 'deleted'`,
			).run(ts, ts, memoryId, tenantId, userId);
			if (Number(result.changes) === 0) return false;

			// Purge the superseded predecessors' text & auxiliary indexes too (the whole key chain that led to this memory).
			const chain = this.stmt(
				`WITH RECURSIVE chain(id) AS (
					SELECT id FROM longterm_memories
					 WHERE superseded_by_id = ? AND tenant_id = ? AND user_id = ?
					UNION
					SELECT m.id FROM longterm_memories m JOIN chain c ON m.superseded_by_id = c.id
					 WHERE m.tenant_id = ? AND m.user_id = ?
				)
				SELECT id FROM chain`,
			).all(memoryId, tenantId, userId, tenantId, userId) as Array<{ id: string }>;
			const purgeIds = [memoryId, ...chain.map((row) => row.id)];

			const blank = this.stmt(
				`UPDATE longterm_memories SET status = 'deleted', content = '', valid_until = COALESCE(valid_until, ?), updated_at = ?
				 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status != 'deleted'`,
			);
			const dropFts = this.stmt("DELETE FROM longterm_memories_fts WHERE memory_id = ?");
			const dropEmbedding = this.stmt(
				"DELETE FROM memory_embeddings WHERE memory_id = ? AND tenant_id = ? AND user_id = ?",
			);
			const dropEntityLinks = this.stmt(
				"DELETE FROM memory_entity_links WHERE memory_id = ? AND tenant_id = ? AND user_id = ?",
			);
			const dropRelations = this.stmt(
				"DELETE FROM memory_relations WHERE memory_id = ? AND tenant_id = ? AND user_id = ?",
			);
			const dropTemporal = this.stmt(
				"DELETE FROM memory_temporal_anchors WHERE memory_id = ? AND tenant_id = ? AND user_id = ?",
			);

			// The raw episodes that evidenced a forgotten memory still hold the verbatim user message and tool arguments.
			// Scrub their text (keeping the row so the evidence ledger and capture idempotency stay intact) and mark
			// them consolidated so a later sweep cannot re-derive the memory from them.
			const scrubEpisodes = this.stmt(
				`UPDATE midterm_episodes
				 SET user_message = '', assistant_summary = '', tool_trace_json = '[]', friction_signals_json = '[]',
				     consolidated = 1
				 WHERE tenant_id = ? AND user_id = ?
				   AND id IN (SELECT episode_id FROM memory_evidence WHERE memory_id = ?)`,
			);
			const dropEpisodeFts = this.stmt(
				`DELETE FROM midterm_episodes_fts
				 WHERE episode_id IN (SELECT episode_id FROM memory_evidence WHERE memory_id = ?)`,
			);

			for (const id of purgeIds) {
				if (id !== memoryId) blank.run(ts, ts, id, tenantId, userId);
				scrubEpisodes.run(tenantId, userId, id);
				dropEpisodeFts.run(id);
				dropFts.run(id);
				dropEmbedding.run(id, tenantId, userId);
				dropEntityLinks.run(id, tenantId, userId);
				dropRelations.run(id, tenantId, userId);
				dropTemporal.run(id, tenantId, userId);
			}

			// Garbage-collect orphaned entities in this tenant+user scope so forgotten entity names leave zero residue.
			this.stmt(
				`DELETE FROM memory_entities
				 WHERE tenant_id = ? AND user_id = ?
				   AND NOT EXISTS (
				       SELECT 1 FROM memory_entity_links l
				       WHERE l.entity_id = memory_entities.id
				   )`,
			).run(tenantId, userId);

			return true;
		});
	}

	private insertMemoryRow(input: {
		id: string;
		tenantId: string;
		userId: string;
		kind: MemoryKind;
		key: string;
		content: string;
		confidence: number;
		pinned: boolean;
		evidenceCount: number;
		distinctConversations: number;
		sourceConversationId?: string;
		timestamp: number;
	}): void {
		this.stmt(
			`INSERT INTO longterm_memories (
				id, tenant_id, user_id, kind, mem_key, content, confidence, pinned,
				status, superseded_by_id, evidence_count, distinct_conversations,
				source_conversation_id, created_at, updated_at, valid_from, valid_until,
				access_count, last_accessed_at, feedback_weight
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?, ?, ?, ?, ?, NULL, 0, NULL, 0.5)`,
		).run(
			input.id,
			input.tenantId,
			input.userId,
			input.kind,
			input.key,
			input.content,
			input.confidence,
			input.pinned ? 1 : 0,
			input.evidenceCount,
			input.distinctConversations,
			input.sourceConversationId ?? null,
			input.timestamp,
			input.timestamp,
			input.timestamp,
		);
		this.syncFts(input.id, input.tenantId, input.userId, input.key, input.content, input.kind);
		this.syncGraphAndTemporal(
			input.id,
			input.tenantId,
			input.userId,
			input.key,
			input.content,
			input.timestamp,
			null,
			input.timestamp,
		);
	}

	private syncFts(
		id: string,
		tenantId: string,
		userId: string,
		key: string,
		content: string,
		kind: string,
	): void {
		this.stmt("DELETE FROM longterm_memories_fts WHERE memory_id = ?").run(id);
		this.stmt(
			`INSERT INTO longterm_memories_fts (memory_id, tenant_id, user_id, mem_key, content, kind)
			 VALUES (?, ?, ?, ?, ?, ?)`,
		).run(id, tenantId, userId, key, content, kind);
	}

	/**
	 * Atomically extracts & syncs entities, entity links, lightweight relations, and temporal anchors
	 * for one memory. Must be called inside an active transaction.
	 */
	private syncGraphAndTemporal(
		memoryId: string,
		tenantId: string,
		userId: string,
		key: string,
		content: string,
		validFrom: number,
		validUntil: number | null,
		timestamp: number,
	): void {
		this.stmt("DELETE FROM memory_entity_links WHERE memory_id = ?").run(memoryId);
		this.stmt("DELETE FROM memory_relations WHERE memory_id = ?").run(memoryId);
		this.stmt("DELETE FROM memory_temporal_anchors WHERE memory_id = ?").run(memoryId);

		// 1. Entities & Links
		const entities = extractEntities(key, content);
		if (entities.length > 0) {
			const upsertEntity = this.stmt(
				`INSERT INTO memory_entities (id, tenant_id, user_id, canonical_name, display_name, entity_type, updated_at)
				 VALUES (?, ?, ?, ?, ?, ?, ?)
				 ON CONFLICT(tenant_id, user_id, canonical_name) DO UPDATE SET
				   display_name = excluded.display_name,
				   updated_at = excluded.updated_at`,
			);
			const getEntityId = this.stmt(
				`SELECT id FROM memory_entities WHERE tenant_id = ? AND user_id = ? AND canonical_name = ?`,
			);
			const insertLink = this.stmt(
				`INSERT OR REPLACE INTO memory_entity_links (memory_id, entity_id, tenant_id, user_id, weight)
				 VALUES (?, ?, ?, ?, ?)`,
			);

			const canonicalToId = new Map<string, string>();
			for (const ent of entities) {
				const candidateId = `ent_${crypto.randomUUID()}`;
				upsertEntity.run(candidateId, tenantId, userId, ent.canonicalName, ent.displayName, ent.type, timestamp);
				const row = getEntityId.get(tenantId, userId, ent.canonicalName) as { id: string } | undefined;
				if (!row) continue;
				canonicalToId.set(ent.canonicalName, row.id);
				insertLink.run(memoryId, row.id, tenantId, userId, ent.weight);
			}

			// 2. Co-occurrence / contextual relations
			const relations = extractRelations(entities, content);
			if (relations.length > 0) {
				const insertRel = this.stmt(
					`INSERT INTO memory_relations (
						id, tenant_id, user_id, source_entity_id, target_entity_id,
						relation, memory_id, valid_from, valid_until, weight
					) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
				);
				for (const rel of relations) {
					const srcId = canonicalToId.get(rel.source);
					const dstId = canonicalToId.get(rel.target);
					if (!srcId || !dstId || srcId === dstId) continue;
					insertRel.run(
						`rel_${crypto.randomUUID()}`,
						tenantId,
						userId,
						srcId,
						dstId,
						rel.relation,
						memoryId,
						validFrom,
						validUntil,
						rel.weight,
					);
				}
			}
		}

		// 3. Temporal anchors
		const anchors = extractTemporalAnchors(content, timestamp);
		if (anchors.length > 0) {
			const insertAnchor = this.stmt(
				`INSERT INTO memory_temporal_anchors (
					id, memory_id, tenant_id, user_id, start_ms, end_ms, precision, label
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			);
			for (const a of anchors) {
				insertAnchor.run(
					`tmp_${crypto.randomUUID()}`,
					memoryId,
					tenantId,
					userId,
					a.startMs,
					a.endMs,
					a.precision,
					a.label,
				);
			}
		}
	}

	/** Lazily backfills graph & temporal tables for rows migrated from v1/v2 databases. */
	private ensureGraphAndTemporalBackfilled(tenantId: string, userId: string): void {
		const hasLinks = this.stmt(
			"SELECT 1 AS ok FROM memory_entity_links WHERE tenant_id = ? AND user_id = ? LIMIT 1",
		).get(tenantId, userId);
		if (hasLinks) return;

		const rows = this.stmt(
			`SELECT id, mem_key, content, created_at, updated_at, valid_from, valid_until, status
			 FROM longterm_memories
			 WHERE tenant_id = ? AND user_id = ? AND status IN ('active', 'superseded') AND content != ''
			 LIMIT 2000`,
		).all(tenantId, userId) as Array<{
			id: string;
			mem_key: string;
			content: string;
			created_at: number;
			updated_at: number;
			valid_from: number | null;
			valid_until: number | null;
			status: string;
		}>;
		if (rows.length === 0) return;

		this.appDb.transaction(() => {
			for (const r of rows) {
				this.syncGraphAndTemporal(
					r.id,
					tenantId,
					userId,
					r.mem_key,
					r.content,
					Number(r.valid_from ?? r.created_at),
					r.valid_until !== null ? Number(r.valid_until) : r.status === "superseded" ? Number(r.updated_at) : null,
					Number(r.updated_at),
				);
			}
		});
	}

	/** Loads or returns the version-coherent in-memory bipartite graph for `(tenantId, userId, status, kind)`. */
	private getCachedGraphScope(
		tenantId: string,
		userId: string,
		includeSuperseded: boolean,
		kind: MemoryKind | undefined,
	): CachedGraphScope {
		this.ensureGraphAndTemporalBackfilled(tenantId, userId);
		const cacheKey = `${tenantId}\0${userId}\0${includeSuperseded ? "1" : "0"}\0${kind ?? ""}`;
		const stamp = this.dbStamp();
		const cached = this.graphCache.get(cacheKey);
		if (cached && cached.stamp === stamp) {
			return cached;
		}

		const statusClause = includeSuperseded ? "m.status IN ('active', 'superseded')" : "m.status = 'active'";
		const params: (string | number)[] = [tenantId, userId];
		if (kind) params.push(kind);

		const edgeRows = this.stmt(
			`SELECT l.memory_id, l.entity_id, e.canonical_name, l.weight
			 FROM memory_entity_links l
			 JOIN memory_entities e ON e.id = l.entity_id
			 JOIN longterm_memories m ON m.id = l.memory_id
			 WHERE l.tenant_id = ?
			   AND l.user_id = ?
			   AND ${statusClause}${kind ? " AND m.kind = ?" : ""}`,
		).all(...params) as Array<{
			memory_id: string;
			entity_id: string;
			canonical_name: string;
			weight: number;
		}>;

		const edges: EntityGraphEdge[] = [];
		const allowedIds = new Set<string>();
		for (const r of edgeRows) {
			allowedIds.add(r.memory_id);
			edges.push({
				memoryId: r.memory_id,
				entityId: r.entity_id,
				canonicalName: r.canonical_name,
				weight: Number(r.weight),
			});
		}

		const scope: CachedGraphScope = {
			stamp,
			graph: buildBipartiteGraph(edges),
			allowedIds,
		};
		if (this.graphCache.size > 32) this.graphCache.clear();
		this.graphCache.set(cacheKey, scope);
		return scope;
	}
}

export class SqliteArtifactStore implements ArtifactStore {
	private readonly appDb: AppDatabase;
	private readonly artifactsDir: string | undefined;
	private readonly now: () => number;

	constructor(appDb: AppDatabase, options?: { readonly artifactsDir?: string; readonly now?: () => number }) {
		this.appDb = appDb;
		this.artifactsDir = options?.artifactsDir;
		this.now = options?.now ?? (() => Date.now());
	}

	async put(
		tenantId: string,
		userId: string,
		content: string,
		meta?: {
			readonly contentType?: string;
			readonly sourceTool?: string;
			readonly conversationId?: string;
		},
	): Promise<ArtifactMetadata> {
		const id = `art_${crypto.randomUUID()}`;
		const contentType = meta?.contentType ?? "text/plain; charset=utf-8";
		const byteLength = Buffer.byteLength(content, "utf8");
		const createdAt = this.now();

		let filePath: string | null = null;
		let inlineContent: string | null = null;

		if (this.artifactsDir && this.appDb.dbPath !== ":memory:") {
			fs.mkdirSync(this.artifactsDir, { recursive: true });
			filePath = path.join(this.artifactsDir, `${id}.txt`);
			fs.writeFileSync(filePath, content, "utf8");
		} else {
			inlineContent = content;
		}

		this.appDb.db
			.prepare(
				`INSERT INTO artifacts (
					id, tenant_id, user_id, content_type, byte_length,
					source_tool, conversation_id, file_path, inline_content, created_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			)
			.run(
				id,
				tenantId,
				userId,
				contentType,
				byteLength,
				meta?.sourceTool ?? null,
				meta?.conversationId ?? null,
				filePath,
				inlineContent,
				createdAt,
			);

		return {
			id,
			tenantId,
			userId,
			contentType,
			byteLength,
			...(meta?.sourceTool ? { sourceTool: meta.sourceTool } : {}),
			...(meta?.conversationId ? { conversationId: meta.conversationId } : {}),
			createdAt,
		};
	}

	async get(
		tenantId: string,
		userId: string,
		artifactId: string,
		range?: { readonly offset?: number; readonly limit?: number },
	): Promise<ArtifactReadResult | undefined> {
		const row = this.appDb.db
			.prepare("SELECT * FROM artifacts WHERE id = ? AND tenant_id = ? AND user_id = ?")
			.get(artifactId, tenantId, userId) as ArtifactRow | undefined;

		if (!row) return undefined;

		const fullText =
			row.file_path && fs.existsSync(row.file_path)
				? fs.readFileSync(row.file_path, "utf8")
				: (row.inline_content ?? "");

		const offset = Math.max(0, Math.floor(range?.offset ?? 0));
		const limit = Math.max(1, Math.min(Math.floor(range?.limit ?? 4096), 32_768));
		const chunk = fullText.slice(offset, offset + limit);
		const truncated = offset + chunk.length < fullText.length;

		return {
			metadata: {
				id: row.id,
				tenantId: row.tenant_id,
				userId: row.user_id,
				contentType: row.content_type,
				byteLength: Number(row.byte_length),
				...(row.source_tool ? { sourceTool: row.source_tool } : {}),
				...(row.conversation_id ? { conversationId: row.conversation_id } : {}),
				createdAt: Number(row.created_at),
			},
			chunk,
			offset,
			limit,
			totalBytes: Number(row.byte_length),
			truncated,
		};
	}

	async delete(tenantId: string, userId: string, artifactId: string): Promise<boolean> {
		const row = this.appDb.db
			.prepare("SELECT file_path FROM artifacts WHERE id = ? AND tenant_id = ? AND user_id = ?")
			.get(artifactId, tenantId, userId) as { file_path: string | null } | undefined;
		if (!row) return false;

		if (row.file_path && fs.existsSync(row.file_path)) {
			fs.rmSync(row.file_path, { force: true });
		}
		const result = this.appDb.db
			.prepare("DELETE FROM artifacts WHERE id = ? AND tenant_id = ? AND user_id = ?")
			.run(artifactId, tenantId, userId);
		return Number(result.changes) > 0;
	}
}

export class SqliteTelemetryStore implements TelemetryStore {
	private readonly appDb: AppDatabase;
	private readonly now: () => number;

	constructor(appDb: AppDatabase, options?: { readonly now?: () => number }) {
		this.appDb = appDb;
		this.now = options?.now ?? (() => Date.now());
	}

	async recordToolInvocation(metric: ToolInvocationMetric): Promise<void> {
		this.appDb.transaction(() => {
			this.appDb.db
				.prepare(
					`INSERT INTO tool_invocations (
						tenant_id, user_id, conversation_id, call_id, tool_name,
						skill_name, is_error, diagnostic_code, duration_ms, created_at
					) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
				)
				.run(
					metric.tenantId,
					metric.userId,
					metric.conversationId,
					metric.callId,
					metric.toolName,
					metric.skillName ?? null,
					metric.isError ? 1 : 0,
					metric.diagnosticCode ?? null,
					metric.durationMs ?? null,
					metric.createdAt,
				);

			this.upsertSkillMetric(metric.tenantId, metric.toolName, "tool", metric.isError, metric.durationMs ?? 0, metric.createdAt);
			if (metric.skillName) {
				this.upsertSkillMetric(
					metric.tenantId,
					metric.skillName,
					"skill",
					metric.isError,
					metric.durationMs ?? 0,
					metric.createdAt,
				);
			}
		});
	}

	async recordRunTelemetry(record: RunTelemetryRecord): Promise<void> {
		this.appDb.db
			.prepare(
				`INSERT INTO run_telemetry (
					tenant_id, user_id, conversation_id, run_id,
					tool_names_json, friction_score, friction_kinds_json, created_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(tenant_id, user_id, conversation_id, run_id) DO NOTHING`,
			)
			.run(
				record.tenantId,
				record.userId,
				record.conversationId,
				record.runId,
				JSON.stringify(record.toolNames),
				record.frictionScore,
				JSON.stringify(record.frictionKinds),
				record.createdAt,
			);
	}

	async listToolInvocations(tenantId: string, limit = 200): Promise<ToolInvocationMetric[]> {
		const rows = this.appDb.db
			.prepare(
				`SELECT tenant_id, user_id, conversation_id, call_id, tool_name,
				        skill_name, is_error, diagnostic_code, duration_ms, created_at
				 FROM tool_invocations
				 WHERE tenant_id = ?
				 ORDER BY created_at DESC, id DESC
				 LIMIT ?`,
			)
			.all(tenantId, limit) as Array<{
			tenant_id: string;
			user_id: string;
			conversation_id: string;
			call_id: string;
			tool_name: string;
			skill_name: string | null;
			is_error: number;
			diagnostic_code: string | null;
			duration_ms: number | null;
			created_at: number;
		}>;

		return rows.map((row) => ({
			tenantId: row.tenant_id,
			userId: row.user_id,
			conversationId: row.conversation_id,
			callId: row.call_id,
			toolName: row.tool_name,
			...(row.skill_name ? { skillName: row.skill_name } : {}),
			isError: Number(row.is_error) === 1,
			...(row.diagnostic_code ? { diagnosticCode: row.diagnostic_code } : {}),
			...(row.duration_ms !== null ? { durationMs: Number(row.duration_ms) } : {}),
			createdAt: Number(row.created_at),
		}));
	}

	async getRecentToolSequences(tenantId: string, userId?: string, limit = 200): Promise<string[][]> {
		const rows = userId
			? (this.appDb.db
					.prepare(
						`SELECT tool_names_json FROM run_telemetry
						 WHERE tenant_id = ? AND user_id = ?
						 ORDER BY created_at DESC, id DESC
						 LIMIT ?`,
					)
					.all(tenantId, userId, limit) as Array<{ tool_names_json: string }>)
			: (this.appDb.db
					.prepare(
						`SELECT tool_names_json FROM run_telemetry
						 WHERE tenant_id = ?
						 ORDER BY created_at DESC, id DESC
						 LIMIT ?`,
					)
					.all(tenantId, limit) as Array<{ tool_names_json: string }>);

		return rows
			.map((row) => JSON.parse(row.tool_names_json) as string[])
			.filter((seq) => Array.isArray(seq) && seq.length > 0);
	}

	async getSkillMetrics(tenantId: string, kind?: "tool" | "skill"): Promise<SkillHealthMetric[]> {
		const rows = kind
			? (this.appDb.db
					.prepare(
						`SELECT m.*, d.reason AS disabled_reason
						 FROM skill_metrics m
						 LEFT JOIN disabled_skills d
						   ON d.tenant_id = m.tenant_id AND d.skill_name = m.name
						 WHERE m.tenant_id = ? AND m.kind = ?
						 ORDER BY m.calls DESC, m.name ASC`,
					)
					.all(tenantId, kind) as Array<{
					tenant_id: string;
					name: string;
					kind: "tool" | "skill";
					calls: number;
					successes: number;
					errors: number;
					total_duration_ms: number;
					last_used_at: number;
					disabled_reason: string | null;
				}>)
			: (this.appDb.db
					.prepare(
						`SELECT m.*, d.reason AS disabled_reason
						 FROM skill_metrics m
						 LEFT JOIN disabled_skills d
						   ON d.tenant_id = m.tenant_id AND d.skill_name = m.name
						 WHERE m.tenant_id = ?
						 ORDER BY m.calls DESC, m.name ASC`,
					)
					.all(tenantId) as Array<{
					tenant_id: string;
					name: string;
					kind: "tool" | "skill";
					calls: number;
					successes: number;
					errors: number;
					total_duration_ms: number;
					last_used_at: number;
					disabled_reason: string | null;
				}>);

		return rows.map((row) => {
			const calls = Number(row.calls);
			const errors = Number(row.errors);
			return {
				tenantId: row.tenant_id,
				name: row.name,
				kind: row.kind,
				calls,
				successes: Number(row.successes),
				errors,
				errorRate: calls > 0 ? errors / calls : 0,
				avgDurationMs: calls > 0 ? Number(row.total_duration_ms) / calls : 0,
				lastUsedAt: Number(row.last_used_at),
				disabled: row.disabled_reason !== null,
				...(row.disabled_reason ? { disabledReason: row.disabled_reason } : {}),
			};
		});
	}

	async listDisabledSkills(tenantId: string): Promise<string[]> {
		const rows = this.appDb.db
			.prepare("SELECT skill_name FROM disabled_skills WHERE tenant_id = ? ORDER BY skill_name ASC")
			.all(tenantId) as Array<{ skill_name: string }>;
		return rows.map((row) => row.skill_name);
	}

	async setSkillDisabled(
		tenantId: string,
		skillName: string,
		disabled: boolean,
		reason = "disabled",
		nowMs?: number,
	): Promise<void> {
		if (disabled) {
			this.appDb.db
				.prepare(
					`INSERT INTO disabled_skills (tenant_id, skill_name, reason, disabled_at)
					 VALUES (?, ?, ?, ?)
					 ON CONFLICT(tenant_id, skill_name) DO UPDATE SET reason = excluded.reason, disabled_at = excluded.disabled_at`,
				)
				.run(tenantId, skillName, reason, nowMs ?? this.now());
		} else {
			this.appDb.db
				.prepare("DELETE FROM disabled_skills WHERE tenant_id = ? AND skill_name = ?")
				.run(tenantId, skillName);
			this.appDb.db
				.prepare(
					`UPDATE skill_metrics
					 SET calls = 0, successes = 0, errors = 0, total_duration_ms = 0, last_used_at = ?
					 WHERE tenant_id = ? AND name = ? AND kind = 'skill'`,
				)
				.run(nowMs ?? this.now(), tenantId, skillName);
		}
	}

	private upsertSkillMetric(
		tenantId: string,
		name: string,
		kind: "tool" | "skill",
		isError: boolean,
		durationMs: number,
		timestamp: number,
	): void {
		this.appDb.db
			.prepare(
				`INSERT INTO skill_metrics (
					tenant_id, name, kind, calls, successes, errors, total_duration_ms, last_used_at
				) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
				ON CONFLICT(tenant_id, name, kind) DO UPDATE SET
					calls = calls + 1,
					successes = successes + excluded.successes,
					errors = errors + excluded.errors,
					total_duration_ms = total_duration_ms + excluded.total_duration_ms,
					last_used_at = MAX(last_used_at, excluded.last_used_at)`,
			)
			.run(tenantId, name, kind, isError ? 0 : 1, isError ? 1 : 0, durationMs, timestamp);
	}
}

export type { FrictionSignalKind };
