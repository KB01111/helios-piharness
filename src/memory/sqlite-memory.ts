import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type {
	ArtifactMetadata,
	ArtifactReadResult,
	ArtifactStore,
	FrictionSignal,
	FrictionSignalKind,
	LongTermMemory,
	MemoryKind,
	MemoryMutationOp,
	MemorySearchResult,
	MemoryService,
	MemoryStatus,
	MidTermEpisode,
	RunTelemetryRecord,
	SkillHealthMetric,
	TelemetryStore,
	ToolInvocationMetric,
	ToolTraceItem,
} from "../contracts/app-capabilities.js";
import type { AppDatabase } from "./db.js";

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

function buildSafeFtsQuery(rawQuery: string): string | null {
	const tokens = rawQuery
		.toLowerCase()
		.match(/[a-z0-9_]+/g)
		?.filter((token) => token.length > 0);
	if (!tokens || tokens.length === 0) return null;
	const unique = Array.from(new Set(tokens)).slice(0, 12);
	return unique.map((token) => `"${token}"*`).join(" OR ");
}

export class SqliteMemoryService implements MemoryService {
	private readonly appDb: AppDatabase;
	private readonly now: () => number;

	constructor(appDb: AppDatabase, options?: { readonly now?: () => number }) {
		this.appDb = appDb;
		this.now = options?.now ?? (() => Date.now());
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

		const ftsQuery = buildSafeFtsQuery(query);
		if (ftsQuery) {
			const params: (string | number)[] = [ftsQuery, tenantId, userId];
			if (options?.kind) params.push(options.kind);
			params.push(limit);

			const rows = this.appDb.db
				.prepare(
					`SELECT m.*, bm25(longterm_memories_fts) AS rank,
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

			if (rows.length > 0) {
				return rows.map((row) => ({
					memory: rowToMemory(row),
					rank: Number(row.rank),
					snippet: row.snippet || row.content.slice(0, 160),
				}));
			}
		}

		const likePattern = `%${query.trim()}%`;
		const fallbackParams: (string | number)[] = [tenantId, userId, likePattern, likePattern];
		if (options?.kind) fallbackParams.push(options.kind);
		fallbackParams.push(limit);

		const fallbackRows = this.appDb.db
			.prepare(
				`SELECT m.*
				 FROM longterm_memories m
				 WHERE m.tenant_id = ?
				   AND m.user_id = ?
				   AND (m.mem_key LIKE ? OR m.content LIKE ?)
				   AND ${statusClause}${kindClause}
				 ORDER BY m.pinned DESC, m.confidence DESC, m.updated_at DESC
				 LIMIT ?`,
			)
			.all(...fallbackParams) as MemoryRow[];

		return fallbackRows.map((row, idx) => ({
			memory: rowToMemory(row),
			rank: idx + 1,
			snippet: row.content.slice(0, 160),
		}));
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
		const normalizedKey = input.key.trim();
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

			if (existing && existing.content.trim() === input.content.trim()) {
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
				content: input.content.trim(),
				confidence,
				pinned: true,
				evidenceCount: existing ? Number(existing.evidence_count) + 1 : 1,
				distinctConversations: existing ? Number(existing.distinct_conversations) : 1,
				sourceConversationId: input.sourceConversationId,
				timestamp: ts,
			});

			if (existing) {
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

		return this.appDb.transaction(() => {
			const ts = this.now();
			if (op.op === "ADD") {
				const newId = `mem_${crypto.randomUUID()}`;
				this.insertMemoryRow({
					id: newId,
					tenantId,
					userId,
					kind: op.kind,
					key: op.key.trim(),
					content: op.content.trim(),
					confidence: Math.max(0, Math.min(1, op.confidence)),
					pinned: op.pinned ?? false,
					evidenceCount: op.evidenceCount ?? 1,
					distinctConversations: op.distinctConversations ?? 1,
					sourceConversationId: op.sourceConversationId,
					timestamp: ts,
				});
				const created = this.appDb.db
					.prepare("SELECT * FROM longterm_memories WHERE id = ?")
					.get(newId) as MemoryRow;
				return rowToMemory(created);
			}

			if (op.op === "UPDATE") {
				const existing = this.appDb.db
					.prepare(
						`SELECT * FROM longterm_memories
						 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'`,
					)
					.get(op.targetId, tenantId, userId) as MemoryRow | undefined;
				if (!existing) return null;

				const nextContent = op.content.trim();
				const nextConfidence = Math.max(0, Math.min(1, op.confidence));
				const nextEvidence = Number(existing.evidence_count) + (op.evidenceCountDelta ?? 1);
				const nextDistinct = Number(existing.distinct_conversations) + (op.distinctConversationsDelta ?? 0);

				this.appDb.db
					.prepare(
						`UPDATE longterm_memories
						 SET content = ?, confidence = ?, evidence_count = ?, distinct_conversations = ?, updated_at = ?
						 WHERE id = ?`,
					)
					.run(nextContent, nextConfidence, nextEvidence, nextDistinct, ts, existing.id);

				this.syncFts(existing.id, tenantId, userId, existing.mem_key, nextContent, existing.kind);

				const updated = this.appDb.db
					.prepare("SELECT * FROM longterm_memories WHERE id = ?")
					.get(existing.id) as MemoryRow;
				return rowToMemory(updated);
			}

			// SUPERSEDE: create new active row and tombstone the old row with superseded_by_id
			const existing = this.appDb.db
				.prepare(
					`SELECT * FROM longterm_memories
					 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'`,
				)
				.get(op.targetId, tenantId, userId) as MemoryRow | undefined;

			const newId = `mem_${crypto.randomUUID()}`;
			this.insertMemoryRow({
				id: newId,
				tenantId,
				userId,
				kind: op.kind,
				key: op.key.trim(),
				content: op.content.trim(),
				confidence: Math.max(0, Math.min(1, op.confidence)),
				pinned: existing ? Number(existing.pinned) === 1 : false,
				evidenceCount: existing ? Number(existing.evidence_count) + 1 : 1,
				distinctConversations: existing ? Number(existing.distinct_conversations) + 1 : 1,
				sourceConversationId: op.sourceConversationId,
				timestamp: ts,
			});

			if (existing) {
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

	async forgetMemory(tenantId: string, userId: string, memoryId: string): Promise<boolean> {
		return this.appDb.transaction(() => {
			const result = this.appDb.db
				.prepare(
					`UPDATE longterm_memories
					 SET status = 'deleted', updated_at = ?
					 WHERE id = ? AND tenant_id = ? AND user_id = ? AND status != 'deleted'`,
				)
				.run(this.now(), memoryId, tenantId, userId);

			this.appDb.db.prepare("DELETE FROM longterm_memories_fts WHERE memory_id = ?").run(memoryId);
			return Number(result.changes) > 0;
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
