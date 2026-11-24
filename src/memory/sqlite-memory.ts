import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
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
	blendScore,
	buildFtsQuery,
	cosineSimilarity,
	decodeVector,
	encodeVector,
	escapeLike,
	expandToken,
	hybridScore,
	resolveRankingOptions,
	type RankingOptions,
	type ResolvedRankingOptions,
} from "./retrieval.js";
import { sanitizeMemoryKey, sanitizeMemoryText } from "./sanitize.js";

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

export class SqliteMemoryService implements MemoryService {
	private readonly appDb: AppDatabase;
	private readonly now: () => number;
	private readonly embeddingProvider: EmbeddingProvider | undefined;
	private readonly ranking: ResolvedRankingOptions;
	private readonly maxVectorCandidates: number;
	private readonly maxBackfillPerSearch: number;

	constructor(
		appDb: AppDatabase,
		options?: {
			readonly now?: () => number;
			/** Optional injected embedding provider enabling hybrid (lexical + cosine) retrieval. */
			readonly embeddingProvider?: EmbeddingProvider;
			/** Re-ranking weights / decay overrides (see `DEFAULT_RANKING_WEIGHTS` in retrieval.ts). */
			readonly ranking?: RankingOptions;
			/** Max in-scope memories scanned for vector similarity per search. Default 500. */
			readonly maxVectorCandidates?: number;
			/** Max stale/missing vectors embedded lazily per search. Default 64. */
			readonly maxBackfillPerSearch?: number;
		},
	) {
		this.appDb = appDb;
		this.now = options?.now ?? (() => Date.now());
		this.embeddingProvider = options?.embeddingProvider;
		this.ranking = resolveRankingOptions(options?.ranking);
		this.maxVectorCandidates = Math.max(1, options?.maxVectorCandidates ?? 500);
		this.maxBackfillPerSearch = Math.max(1, options?.maxBackfillPerSearch ?? 64);
	}

	async recordEpisode(
		episode: Omit<MidTermEpisode, "id" | "consolidated" | "createdAt"> & { readonly createdAt?: number },
	): Promise<MidTermEpisode> {
		const id = `ep_${crypto.randomUUID()}`;
		const createdAt = episode.createdAt ?? this.now();

		this.appDb.db
			.prepare(
				`INSERT INTO midterm_episodes (
					id, tenant_id, user_id, conversation_id, run_id,
					user_message, assistant_summary, tool_trace_json,
					friction_signals_json, friction_score, consolidated, created_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
				ON CONFLICT(tenant_id, user_id, conversation_id, run_id) DO NOTHING`,
			)
			.run(
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

		const existing = this.appDb.db
			.prepare(
				`SELECT * FROM midterm_episodes
				 WHERE tenant_id = ? AND user_id = ? AND conversation_id = ? AND run_id = ?`,
			)
			.get(episode.tenantId, episode.userId, episode.conversationId, episode.runId) as EpisodeRow;

		return rowToEpisode(existing);
	}

	async listTenantsWithPendingEpisodes(options?: {
		readonly limit?: number;
		readonly minEpisodes?: number;
	}): Promise<PendingEpisodeScope[]> {
		const limit = Math.max(1, Math.min(options?.limit ?? 50, 500));
		const minEpisodes = Math.max(1, options?.minEpisodes ?? 1);
		const scopes = this.appDb.db
			.prepare(
				`SELECT tenant_id, user_id, COUNT(*) AS pending, MAX(created_at) AS latest
				 FROM midterm_episodes WHERE consolidated = 0
				 GROUP BY tenant_id, user_id
				 HAVING COUNT(*) >= ?
				 ORDER BY latest DESC, tenant_id ASC, user_id ASC
				 LIMIT ?`,
			)
			.all(minEpisodes, limit) as Array<{ tenant_id: string; user_id: string; pending: number; latest: number }>;
		const convStmt = this.appDb.db.prepare(
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
		const rows = this.appDb.db
			.prepare(
				`SELECT * FROM midterm_episodes
				 WHERE tenant_id = ? AND user_id = ? AND consolidated = 0
				 ORDER BY created_at ASC, id ASC
				 LIMIT ?`,
			)
			.all(tenantId, userId, limit) as EpisodeRow[];
		return rows.map(rowToEpisode);
	}

	async listEpisodes(tenantId: string, userId: string, limit = 100): Promise<MidTermEpisode[]> {
		const rows = this.appDb.db
			.prepare(
				`SELECT * FROM midterm_episodes
				 WHERE tenant_id = ? AND user_id = ?
				 ORDER BY created_at DESC, id DESC
				 LIMIT ?`,
			)
			.all(tenantId, userId, limit) as EpisodeRow[];
		return rows.map(rowToEpisode);
	}

	async markEpisodesConsolidated(episodeIds: readonly string[]): Promise<void> {
		if (episodeIds.length === 0) return;
		this.appDb.transaction(() => {
			const stmt = this.appDb.db.prepare("UPDATE midterm_episodes SET consolidated = 1 WHERE id = ?");
			for (const id of episodeIds) {
				stmt.run(id);
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
		},
	): Promise<MemorySearchResult[]> {
		const limit = Math.max(1, Math.min(options?.limit ?? 10, 100));
		const includeSuperseded = options?.includeSuperseded ?? false;
		const statusClause = includeSuperseded ? "m.status IN ('active', 'superseded')" : "m.status = 'active'";
		const kindClause = options?.kind ? " AND m.kind = ?" : "";
		const nowMs = this.now();
		const poolSize = Math.max(50, limit * 5);
		const { fts, tokens } = buildFtsQuery(query);

		type Candidate = { row: MemoryRow; lexical: number; snippet: string | undefined };
		const candidates = new Map<string, Candidate>();

		// 1) Lexical candidates: FTS5 BM25 over prefix/stemmed terms (key weighted above content above kind).
		if (fts) {
			const params: (string | number)[] = [fts, tenantId, userId];
			if (options?.kind) params.push(options.kind);
			params.push(poolSize);

			const rows = this.appDb.db
				.prepare(
					`SELECT m.*, bm25(longterm_memories_fts, 0.0, 0.0, 0.0, 3.0, 1.0, 0.5) AS rank,
					        snippet(longterm_memories_fts, 4, '[', ']', '...', 12) AS snippet
					 FROM longterm_memories_fts f
					 JOIN longterm_memories m ON m.id = f.memory_id
					 WHERE longterm_memories_fts MATCH ?
					   AND m.tenant_id = ?
					   AND m.user_id = ?
					   AND ${statusClause}${kindClause}
					 ORDER BY rank ASC, m.pinned DESC, m.confidence DESC, m.updated_at DESC
					 LIMIT ?`,
				)
				.all(...params) as Array<MemoryRow & { rank: number; snippet: string }>;

			// bm25() is negative (more negative = better); max-normalize the magnitudes into [0, 1].
			const magnitudes = rows.map((row) => Math.max(0, -Number(row.rank)));
			const maxMagnitude = magnitudes.reduce((a, b) => Math.max(a, b), 0);
			rows.forEach((row, idx) => {
				const lexical = maxMagnitude > 0 ? (magnitudes[idx] ?? 0) / maxMagnitude : 1;
				candidates.set(row.id, { row, lexical, snippet: row.snippet || undefined });
			});
		}

		// 2) LIKE fallback (also covers queries with no FTS-usable tokens) when FTS found nothing.
		if (candidates.size === 0) {
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
				const haystack = `${row.mem_key} ${row.content}`.toLowerCase();
				const matched = tokens.filter((token) => expandToken(token).some((term) => haystack.includes(term))).length;
				const lexical = tokens.length > 0 ? matched / tokens.length : 0;
				candidates.set(row.id, { row, lexical, snippet: undefined });
			}
		}

		// 3) Optional embedding provider: hybrid-rank with cosine over stored vectors (never throws).
		let cosines: Map<string, number> | undefined;
		if (this.embeddingProvider && query.trim().length > 0) {
			try {
				const vector = await this.vectorCandidates(tenantId, userId, query, statusClause, options?.kind, poolSize);
				cosines = new Map();
				for (const hit of vector) {
					cosines.set(hit.row.id, hit.cosine);
					if (!candidates.has(hit.row.id) && hit.cosine > 0) {
						candidates.set(hit.row.id, { row: hit.row, lexical: 0, snippet: undefined });
					}
				}
			} catch {
				cosines = undefined; // degrade to purely lexical ranking
			}
		}

		const scored = Array.from(candidates.values()).map((candidate) => {
			const memory = rowToMemory(candidate.row);
			const blend = blendScore(candidate.lexical, memory, nowMs, this.ranking);
			const cosine = cosines?.get(memory.id);
			const score = cosines ? hybridScore(blend, cosine ?? 0, this.ranking.vectorWeight) : blend;
			return { memory, score, snippet: candidate.snippet };
		});

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
		}));
	}

	/**
	 * Embeds (lazily, bounded per call) and scores in-scope memories against the query.
	 * Stale vectors (content changed or different model) are recomputed. Tenant+user scoping is enforced in SQL.
	 */
	private async vectorCandidates(
		tenantId: string,
		userId: string,
		query: string,
		statusClause: string,
		kind: MemoryKind | undefined,
		poolSize: number,
	): Promise<Array<{ row: MemoryRow; cosine: number }>> {
		const provider = this.embeddingProvider;
		if (!provider) return [];

		const params: (string | number)[] = [provider.model, tenantId, userId];
		if (kind) params.push(kind);
		params.push(this.maxVectorCandidates);
		const rows = this.appDb.db
			.prepare(
				`SELECT m.*, e.vector AS e_vector, e.content_hash AS e_hash
				 FROM longterm_memories m
				 LEFT JOIN memory_embeddings e
				   ON e.memory_id = m.id AND e.model = ? AND e.tenant_id = m.tenant_id AND e.user_id = m.user_id
				 WHERE m.tenant_id = ? AND m.user_id = ? AND ${statusClause}${kind ? " AND m.kind = ?" : ""}
				 ORDER BY m.pinned DESC, m.updated_at DESC, m.id ASC
				 LIMIT ?`,
			)
			.all(...params) as Array<MemoryRow & { e_vector: Uint8Array | null; e_hash: string | null }>;

		const vectors = new Map<string, Float32Array>();
		const stale: Array<{ row: MemoryRow; text: string; hash: string }> = [];
		for (const row of rows) {
			const text = embeddingText(row.mem_key, row.content);
			const hash = hashText(text);
			// An undecodable stored vector is treated as stale (re-embedded within the backfill budget) rather
			// than throwing, which would disable vector search for the whole query.
			const stored = row.e_vector && row.e_hash === hash ? tryDecodeVector(row.e_vector) : undefined;
			if (stored) {
				vectors.set(row.id, stored);
			} else {
				stale.push({ row, text, hash });
			}
		}

		const toEmbed = stale.slice(0, this.maxBackfillPerSearch);
		const texts = [query, ...toEmbed.map((entry) => entry.text)];
		const embedded = await provider.embed(texts);
		if (embedded.length !== texts.length) throw new Error("EmbeddingProvider returned a mismatched vector count");
		const queryVector = embedded[0];
		if (!isUsableVector(queryVector)) throw new Error("EmbeddingProvider returned an unusable (empty/non-finite) query vector");

		const fresh: Array<{ id: string; hash: string; vec: ArrayLike<number> }> = [];
		toEmbed.forEach((entry, idx) => {
			const vec = embedded[idx + 1];
			if (!isUsableVector(vec)) return; // ignore bad vectors: that memory simply has no cosine this search
			vectors.set(entry.row.id, Float32Array.from(vec));
			fresh.push({ id: entry.row.id, hash: entry.hash, vec });
		});
		this.persistEmbeddings(tenantId, userId, provider.model, fresh);

		const hits: Array<{ row: MemoryRow; cosine: number }> = [];
		for (const row of rows) {
			const vec = vectors.get(row.id);
			if (!vec) continue;
			hits.push({ row, cosine: cosineSimilarity(queryVector, vec) });
		}
		hits.sort((a, b) => b.cosine - a.cosine || (a.row.id < b.row.id ? -1 : 1));
		return hits.slice(0, Math.max(poolSize, 1) * 2);
	}

	/**
	 * Computes and stores embeddings for in-scope active memories that lack a fresh vector.
	 * No-op without an EmbeddingProvider. Returns the number of vectors written.
	 */
	async backfillEmbeddings(tenantId: string, userId: string, options?: { readonly limit?: number }): Promise<number> {
		const provider = this.embeddingProvider;
		if (!provider) return 0;
		const limit = Math.max(1, Math.min(options?.limit ?? 200, 1000));
		const rows = this.appDb.db
			.prepare(
				`SELECT m.*, e.content_hash AS e_hash
				 FROM longterm_memories m
				 LEFT JOIN memory_embeddings e ON e.memory_id = m.id AND e.model = ?
				 WHERE m.tenant_id = ? AND m.user_id = ? AND m.status = 'active'
				 ORDER BY m.pinned DESC, m.updated_at DESC, m.id ASC
				 LIMIT ?`,
			)
			.all(provider.model, tenantId, userId, limit) as Array<MemoryRow & { e_hash: string | null }>;
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
	 * superseded, forgotten, or edited meanwhile, so each row is re-checked inside the write transaction: only
	 * still-active rows whose current text still hashes to the embedded text are written.
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
			const current = this.appDb.db.prepare(
				"SELECT mem_key, content, status FROM longterm_memories WHERE id = ? AND tenant_id = ? AND user_id = ?",
			);
			const upsert = this.appDb.db.prepare(UPSERT_EMBEDDING_SQL);
			let written = 0;
			for (const item of items) {
				const row = current.get(item.id, tenantId, userId) as
					| { mem_key: string; content: string; status: string }
					| undefined;
				if (!row || row.status !== "active") continue;
				if (hashText(embeddingText(row.mem_key, row.content)) !== item.hash) continue;
				upsert.run(item.id, tenantId, userId, model, item.vec.length, item.hash, encodeVector(item.vec), ts);
				written += 1;
			}
			return written;
		});
	}

	async getCoreMemory(tenantId: string, userId: string, limit = 20): Promise<LongTermMemory[]> {
		const rows = this.appDb.db
			.prepare(
				`SELECT * FROM longterm_memories
				 WHERE tenant_id = ?
				   AND user_id = ?
				   AND status = 'active'
				   AND (pinned = 1 OR confidence >= 0.7)
				 ORDER BY pinned DESC, confidence DESC, created_at ASC, id ASC
				 LIMIT ?`,
			)
			.all(tenantId, userId, limit) as MemoryRow[];
		return rows.map(rowToMemory);
	}

	async getMemoryById(tenantId: string, userId: string, id: string): Promise<LongTermMemory | undefined> {
		const row = this.appDb.db
			.prepare("SELECT * FROM longterm_memories WHERE id = ? AND tenant_id = ? AND user_id = ?")
			.get(id, tenantId, userId) as MemoryRow | undefined;
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

		const rows = this.appDb.db
			.prepare(
				`SELECT * FROM longterm_memories
				 WHERE tenant_id = ?
				   AND user_id = ?
				   AND status IN (${statuses.join(", ")})${kindClause}
				 ORDER BY pinned DESC, updated_at DESC, id ASC
				 LIMIT ?`,
			)
			.all(...params) as MemoryRow[];
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
			const existing = this.appDb.db
				.prepare(
					`SELECT * FROM longterm_memories
					 WHERE tenant_id = ? AND user_id = ? AND mem_key = ? AND status = 'active'
					 ORDER BY updated_at DESC LIMIT 1`,
				)
				.get(tenantId, userId, normalizedKey) as MemoryRow | undefined;

			if (existing && existing.content.trim() === content) {
				const ts = this.now();
				this.appDb.db
					.prepare(
						`UPDATE longterm_memories
						 SET pinned = 1, confidence = MAX(confidence, ?), updated_at = ?
						 WHERE id = ?`,
					)
					.run(confidence, ts, existing.id);
				const updated = this.appDb.db
					.prepare("SELECT * FROM longterm_memories WHERE id = ?")
					.get(existing.id) as MemoryRow;
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
				this.appDb.db
					.prepare(
						`UPDATE longterm_memories
						 SET status = 'superseded', superseded_by_id = ?, updated_at = ?
						 WHERE id = ?`,
					)
					.run(newId, ts, existing.id);
			}

			const created = this.appDb.db
				.prepare("SELECT * FROM longterm_memories WHERE id = ?")
				.get(newId) as MemoryRow;
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
			if (pending.length === 0) {
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

			const mark = this.appDb.db.prepare(
				"UPDATE midterm_episodes SET consolidated = 1 WHERE id = ? AND tenant_id = ? AND user_id = ?",
			);
			for (const id of pending) mark.run(id, tenantId, userId);
			return { applied: true, results };
		});
	}

	/**
	 * Hook invoked inside the consolidation transaction after all memory ops were written and before the
	 * episodes are marked. Throwing here rolls the whole batch back (used to simulate crashes in tests).
	 */
	protected afterConsolidationOps(_episodeIds: readonly string[]): void {}

	private filterPendingEpisodeIds(tenantId: string, userId: string, episodeIds: readonly string[]): string[] {
		const stmt = this.appDb.db.prepare(
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
		const stmt = this.appDb.db.prepare("SELECT 1 AS ok FROM memory_evidence WHERE memory_id = ? AND episode_id = ?");
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
		const stmt = this.appDb.db.prepare(
			"INSERT OR IGNORE INTO memory_evidence (memory_id, episode_id, created_at) VALUES (?, ?, ?)",
		);
		for (const id of episodeIds) stmt.run(memoryId, id, timestamp);
	}

	private copyEvidence(fromMemoryId: string, toMemoryId: string): void {
		this.appDb.db
			.prepare(
				`INSERT OR IGNORE INTO memory_evidence (memory_id, episode_id, created_at)
				 SELECT ?, episode_id, created_at FROM memory_evidence WHERE memory_id = ?`,
			)
			.run(toMemoryId, fromMemoryId);
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
				const duplicate = this.appDb.db
					.prepare(
						`SELECT * FROM longterm_memories
						 WHERE tenant_id = ? AND user_id = ? AND mem_key = ? AND status = 'active'
						 ORDER BY updated_at DESC, id ASC`,
					)
					.all(tenantId, userId, key) as MemoryRow[];
				const same = duplicate.find((row) => normalizeForCompare(row.content) === normalizeForCompare(content));
				if (same) {
					const fresh = this.freshEvidenceIds(same.id, op.evidenceEpisodeIds);
					if (fresh.length > 0) {
						this.recordEvidence(same.id, fresh, ts);
						this.appDb.db
							.prepare("UPDATE longterm_memories SET evidence_count = evidence_count + ? WHERE id = ?")
							.run(fresh.length, same.id);
					}
					const current = this.appDb.db
						.prepare("SELECT * FROM longterm_memories WHERE id = ?")
						.get(same.id) as MemoryRow;
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
			const created = this.appDb.db
				.prepare("SELECT * FROM longterm_memories WHERE id = ?")
				.get(newId) as MemoryRow;
			return rowToMemory(created);
		}

		const targetId = resolveBatchTarget(op.targetId, batch);

		if (op.op === "UPDATE") {
			const existing = this.appDb.db
				.prepare(
					`SELECT * FROM longterm_memories
					 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'`,
				)
				.get(targetId, tenantId, userId) as MemoryRow | undefined;
			if (!existing) return null;

			const fresh = op.evidenceEpisodeIds ? this.freshEvidenceIds(existing.id, op.evidenceEpisodeIds) : undefined;
			const nextConfidence = Math.max(0, Math.min(1, op.confidence));
			const evidenceDelta = fresh ? fresh.length : (op.evidenceCountDelta ?? 1);
			const distinctDelta = fresh && fresh.length === 0 ? 0 : (op.distinctConversationsDelta ?? 0);
			const nextEvidence = Number(existing.evidence_count) + evidenceDelta;
			const nextDistinct = Number(existing.distinct_conversations) + distinctDelta;

			this.appDb.db
				.prepare(
					`UPDATE longterm_memories
					 SET content = ?, confidence = ?, evidence_count = ?, distinct_conversations = ?, updated_at = ?
					 WHERE id = ?`,
				)
				.run(content, nextConfidence, nextEvidence, nextDistinct, ts, existing.id);
			this.recordEvidence(existing.id, fresh, ts);

			this.syncFts(existing.id, tenantId, userId, existing.mem_key, content, existing.kind);

			const updated = this.appDb.db
				.prepare("SELECT * FROM longterm_memories WHERE id = ?")
				.get(existing.id) as MemoryRow;
			return rowToMemory(updated);
		}

		// SUPERSEDE: create new active row and tombstone the old row with superseded_by_id
		const key = sanitizeMemoryKey(op.key);
		if (!key) throw new Error("Memory key is empty after sanitization");
		const existing = this.appDb.db
			.prepare(
				`SELECT * FROM longterm_memories
				 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'`,
			)
			.get(targetId, tenantId, userId) as MemoryRow | undefined;

		if (!existing && op.evidenceEpisodeIds) {
			// Idempotent re-apply: the target was already superseded by an identical active successor.
			const prior = this.appDb.db
				.prepare(
					`SELECT superseded_by_id FROM longterm_memories
					 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'superseded'`,
				)
				.get(targetId, tenantId, userId) as { superseded_by_id: string | null } | undefined;
			if (prior?.superseded_by_id) {
				const successor = this.appDb.db
					.prepare(
						`SELECT * FROM longterm_memories
						 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'`,
					)
					.get(prior.superseded_by_id, tenantId, userId) as MemoryRow | undefined;
				if (successor && normalizeForCompare(successor.content) === normalizeForCompare(content)) {
					return rowToMemory(successor);
				}
			}
		}

		// Target no longer active and no identical successor: nothing to supersede. Inserting here would leave a
		// second active row for the key, so this is a no-op (the planner re-reads state on the next cycle).
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
		this.appDb.db
			.prepare(
				`UPDATE longterm_memories
				 SET status = 'superseded', superseded_by_id = ?, updated_at = ?
				 WHERE id = ?`,
			)
			.run(newId, ts, existing.id);
		this.recordEvidence(newId, op.evidenceEpisodeIds, ts);

		const created = this.appDb.db
			.prepare("SELECT * FROM longterm_memories WHERE id = ?")
			.get(newId) as MemoryRow;
		return rowToMemory(created);
	}

	async forgetMemory(tenantId: string, userId: string, memoryId: string): Promise<boolean> {
		return this.appDb.transaction(() => {
			const ts = this.now();
			// Tombstone: keep id/key/kind/timestamps/links but blank the text itself.
			const result = this.appDb.db
				.prepare(
					`UPDATE longterm_memories
					 SET status = 'deleted', content = '', updated_at = ?
					 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status != 'deleted'`,
				)
				.run(ts, memoryId, tenantId, userId);
			// Nothing in this scope matched: do NOT touch the index/embeddings (they may belong to another scope).
			if (Number(result.changes) === 0) return false;

			// Purge the superseded predecessors' text too (the whole key chain that led to this memory).
			const chain = this.appDb.db
				.prepare(
					`WITH RECURSIVE chain(id) AS (
						SELECT id FROM longterm_memories
						 WHERE superseded_by_id = ? AND tenant_id = ? AND user_id = ?
						UNION
						SELECT m.id FROM longterm_memories m JOIN chain c ON m.superseded_by_id = c.id
						 WHERE m.tenant_id = ? AND m.user_id = ?
					)
					SELECT id FROM chain`,
				)
				.all(memoryId, tenantId, userId, tenantId, userId) as Array<{ id: string }>;
			const purgeIds = [memoryId, ...chain.map((row) => row.id)];

			const blank = this.appDb.db.prepare(
				`UPDATE longterm_memories SET status = 'deleted', content = '', updated_at = ?
				 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status != 'deleted'`,
			);
			const dropFts = this.appDb.db.prepare("DELETE FROM longterm_memories_fts WHERE memory_id = ?");
			const dropEmbedding = this.appDb.db.prepare(
				"DELETE FROM memory_embeddings WHERE memory_id = ? AND tenant_id = ? AND user_id = ?",
			);
			for (const id of purgeIds) {
				if (id !== memoryId) blank.run(ts, id, tenantId, userId);
				dropFts.run(id);
				dropEmbedding.run(id, tenantId, userId);
			}
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
		this.appDb.db
			.prepare(
				`INSERT INTO longterm_memories (
					id, tenant_id, user_id, kind, mem_key, content, confidence, pinned,
					status, superseded_by_id, evidence_count, distinct_conversations,
					source_conversation_id, created_at, updated_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?, ?, ?, ?)`,
			)
			.run(
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
			);
		this.syncFts(input.id, input.tenantId, input.userId, input.key, input.content, input.kind);
	}

	private syncFts(
		id: string,
		tenantId: string,
		userId: string,
		key: string,
		content: string,
		kind: string,
	): void {
		this.appDb.db.prepare("DELETE FROM longterm_memories_fts WHERE memory_id = ?").run(id);
		this.appDb.db
			.prepare(
				`INSERT INTO longterm_memories_fts (memory_id, tenant_id, user_id, mem_key, content, kind)
				 VALUES (?, ?, ?, ?, ?, ?)`,
			)
			.run(id, tenantId, userId, key, content, kind);
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
