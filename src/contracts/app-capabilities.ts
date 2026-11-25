/**
 * Outbound dependency-injection contracts for the harness.
 *
 * Zero Pi Durable / Pi AI / Chord types may appear here so embedding apps
 * can supply their own storage, memory, artifact, or model implementations,
 * or mount an MCP adapter over MemoryService later.
 */

export type MemoryKind = "fact" | "preference" | "procedure" | "constraint" | "episodic";

export type MemoryStatus = "active" | "superseded" | "deleted";

export interface LongTermMemory {
	readonly id: string;
	readonly tenantId: string;
	readonly userId: string;
	readonly kind: MemoryKind;
	readonly key: string;
	readonly content: string;
	readonly confidence: number;
	readonly pinned: boolean;
	readonly status: MemoryStatus;
	readonly supersededById?: string;
	readonly evidenceCount: number;
	readonly distinctConversations: number;
	readonly sourceConversationId?: string;
	readonly createdAt: number;
	readonly updatedAt: number;
}

export interface ToolTraceItem {
	readonly callId: string;
	readonly toolName: string;
	readonly argumentsJson: string;
	readonly isError: boolean;
	readonly diagnosticCodes: readonly string[];
	readonly durationMs?: number;
	/**
	 * Optional model-visible result text (output text and/or diagnostic messages, e.g. "Tool call blocked: Authorization denied ...").
	 * Used by friction detection to recognise guard blocks even when no `blocked` diagnostic code was captured.
	 */
	readonly resultText?: string;
}

export type FrictionSignalKind =
	| "tool_error"
	| "repeated_tool_call"
	| "auth_blocked"
	| "output_truncated"
	| "user_correction"
	| "high_turn_count"
	| "user_repetition"
	| "abandoned_tool_loop"
	| "slow_run";

export interface FrictionSignal {
	readonly kind: FrictionSignalKind;
	readonly severity: "low" | "medium" | "high";
	readonly detail: string;
	readonly toolName?: string;
}

export interface MidTermEpisode {
	readonly id: string;
	readonly tenantId: string;
	readonly userId: string;
	readonly conversationId: string;
	readonly runId: string;
	readonly userMessage: string;
	readonly assistantSummary: string;
	readonly toolTrace: readonly ToolTraceItem[];
	readonly frictionSignals: readonly FrictionSignal[];
	readonly frictionScore: number;
	readonly consolidated: boolean;
	readonly createdAt: number;
}

export interface MemorySearchResult {
	readonly memory: LongTermMemory;
	/** Ordering key: LOWER is better (negated blended relevance score). */
	readonly rank: number;
	readonly snippet: string;
	/** Blended relevance score in roughly [0, 1] (higher is better). Present when produced by the re-ranker. */
	readonly score?: number;
}

/**
 * Optional, injected text-embedding provider for hybrid (lexical + vector) memory retrieval.
 * The harness never performs network calls itself; embedding apps supply an implementation.
 */
export interface EmbeddingProvider {
	/** Stable model identifier. Stored with vectors; vectors from a different model id are ignored. */
	readonly model: string;
	embed(texts: readonly string[], options?: { readonly signal?: AbortSignal }): Promise<ReadonlyArray<ArrayLike<number>>>;
}

export type MemoryMutationOp =
	| {
			readonly op: "ADD";
			readonly key: string;
			readonly kind: MemoryKind;
			readonly content: string;
			readonly confidence: number;
			readonly pinned?: boolean;
			readonly evidenceCount?: number;
			readonly distinctConversations?: number;
			readonly sourceConversationId?: string;
			/** Episodes supporting this memory; recorded in an evidence ledger so re-applying is idempotent. */
			readonly evidenceEpisodeIds?: readonly string[];
	  }
	| {
			readonly op: "UPDATE";
			readonly targetId: string;
			readonly content: string;
			readonly confidence: number;
			readonly evidenceCountDelta?: number;
			readonly distinctConversationsDelta?: number;
			/**
			 * Episodes supporting this update. When set, evidence is counted once per not-yet-recorded
			 * episode id (idempotent re-apply) instead of using `evidenceCountDelta`.
			 */
			readonly evidenceEpisodeIds?: readonly string[];
	  }
	| {
			readonly op: "SUPERSEDE";
			readonly targetId: string;
			readonly key: string;
			readonly kind: MemoryKind;
			readonly content: string;
			readonly confidence: number;
			readonly reason?: string;
			readonly sourceConversationId?: string;
			readonly evidenceEpisodeIds?: readonly string[];
	  }
	| {
			readonly op: "NOOP";
			readonly reason?: string;
	  };

/** A (tenant, user) scope that has episodes awaiting consolidation. */
export interface PendingEpisodeScope {
	readonly tenantId: string;
	readonly userId: string;
	readonly pendingCount: number;
	readonly latestEpisodeAt: number;
	/** Conversations with pending episodes, most recent first (a sleep cycle can be hosted by any of them). */
	readonly conversationIds: readonly string[];
}

export interface MemoryService {
	/**
	 * Optional: enumerate scopes with unconsolidated episodes (used by the SleepCycleScheduler).
	 * Implementations that cannot enumerate may omit it; the scheduler then stays idle.
	 */
	listTenantsWithPendingEpisodes?(options?: {
		readonly limit?: number;
		readonly minEpisodes?: number;
	}): Promise<PendingEpisodeScope[]>;

	recordEpisode(
		episode: Omit<MidTermEpisode, "id" | "consolidated" | "createdAt"> & { readonly createdAt?: number },
	): Promise<MidTermEpisode>;
	listUnconsolidatedEpisodes(tenantId: string, userId: string, limit?: number): Promise<MidTermEpisode[]>;
	listEpisodes(tenantId: string, userId: string, limit?: number): Promise<MidTermEpisode[]>;
	markEpisodesConsolidated(episodeIds: readonly string[]): Promise<void>;

	search(
		tenantId: string,
		userId: string,
		query: string,
		options?: {
			readonly limit?: number;
			readonly includeSuperseded?: boolean;
			readonly kind?: MemoryKind;
		},
	): Promise<MemorySearchResult[]>;

	getCoreMemory(tenantId: string, userId: string, limit?: number): Promise<LongTermMemory[]>;
	getMemoryById(tenantId: string, userId: string, id: string): Promise<LongTermMemory | undefined>;
	listMemories(
		tenantId: string,
		userId: string,
		options?: {
			readonly includeSuperseded?: boolean;
			readonly includeDeleted?: boolean;
			readonly kind?: MemoryKind;
			readonly limit?: number;
		},
	): Promise<LongTermMemory[]>;

	pinMemory(
		tenantId: string,
		userId: string,
		input: {
			readonly key: string;
			readonly content: string;
			readonly kind?: MemoryKind;
			readonly confidence?: number;
			readonly sourceConversationId?: string;
		},
	): Promise<LongTermMemory>;

	applyMemoryMutation(tenantId: string, userId: string, op: MemoryMutationOp): Promise<LongTermMemory | null>;
	forgetMemory(tenantId: string, userId: string, memoryId: string): Promise<boolean>;

	/**
	 * Optional: atomically apply a batch of consolidation mutations AND mark the supporting episodes
	 * consolidated in ONE transaction (all-or-nothing). Episodes already consolidated are skipped, so
	 * re-running after a crash is a no-op (`applied: false`). A `targetId` of `batch:<n>` refers to the
	 * memory produced by op index `n` of the same batch. `results[i]` is the memory produced by `ops[i]`.
	 */
	applyConsolidation?(
		tenantId: string,
		userId: string,
		input: { readonly ops: readonly MemoryMutationOp[]; readonly episodeIds: readonly string[] },
	): Promise<{ readonly applied: boolean; readonly results: ReadonlyArray<LongTermMemory | null> }>;
}

export interface ArtifactMetadata {
	readonly id: string;
	readonly tenantId: string;
	readonly userId: string;
	readonly contentType: string;
	readonly byteLength: number;
	readonly sourceTool?: string;
	readonly conversationId?: string;
	readonly createdAt: number;
}

export interface ArtifactReadResult {
	readonly metadata: ArtifactMetadata;
	readonly chunk: string;
	readonly offset: number;
	readonly limit: number;
	readonly totalBytes: number;
	readonly truncated: boolean;
}

export interface ArtifactStore {
	put(
		tenantId: string,
		userId: string,
		content: string,
		meta?: {
			readonly contentType?: string;
			readonly sourceTool?: string;
			readonly conversationId?: string;
		},
	): Promise<ArtifactMetadata>;

	get(
		tenantId: string,
		userId: string,
		artifactId: string,
		range?: {
			readonly offset?: number;
			readonly limit?: number;
		},
	): Promise<ArtifactReadResult | undefined>;

	delete(tenantId: string, userId: string, artifactId: string): Promise<boolean>;
}

export interface ToolInvocationMetric {
	readonly tenantId: string;
	readonly userId: string;
	readonly conversationId: string;
	readonly callId: string;
	readonly toolName: string;
	readonly skillName?: string;
	readonly isError: boolean;
	readonly diagnosticCode?: string;
	readonly durationMs?: number;
	readonly createdAt: number;
}

export interface RunTelemetryRecord {
	readonly tenantId: string;
	readonly userId: string;
	readonly conversationId: string;
	readonly runId: string;
	readonly toolNames: readonly string[];
	readonly frictionScore: number;
	readonly frictionKinds: readonly FrictionSignalKind[];
	readonly createdAt: number;
}

export interface SkillHealthMetric {
	readonly tenantId: string;
	readonly name: string;
	readonly kind: "tool" | "skill";
	readonly calls: number;
	readonly successes: number;
	readonly errors: number;
	readonly errorRate: number;
	readonly avgDurationMs: number;
	readonly lastUsedAt: number;
	readonly disabled: boolean;
	readonly disabledReason?: string;
}

export interface TelemetryStore {
	recordToolInvocation(metric: ToolInvocationMetric): Promise<void>;
	recordRunTelemetry(record: RunTelemetryRecord): Promise<void>;
	listToolInvocations(tenantId: string, limit?: number): Promise<ToolInvocationMetric[]>;
	getRecentToolSequences(tenantId: string, userId?: string, limit?: number): Promise<string[][]>;
	getSkillMetrics(tenantId: string, kind?: "tool" | "skill"): Promise<SkillHealthMetric[]>;
	listDisabledSkills(tenantId: string): Promise<string[]>;
	setSkillDisabled(
		tenantId: string,
		skillName: string,
		disabled: boolean,
		reason?: string,
		nowMs?: number,
	): Promise<void>;
}

/**
 * Transport- and SDK-agnostic fast-model JSON completion interface used by
 * MemoryConsolidator so `src/memory/` has zero dependency on Pi AI.
 */
export interface FastModelClient {
	completeJson<T = unknown>(
		systemPrompt: string,
		userPrompt: string,
		options?: { readonly signal?: AbortSignal },
	): Promise<{ readonly rawText: string; readonly parsed: T | null }>;
}

export interface AppCapabilities {
	readonly memory: MemoryService;
	readonly artifacts: ArtifactStore;
	readonly telemetry: TelemetryStore;
	readonly fastModel?: FastModelClient;
	readonly skillsDir: string;
	readonly maxToolOutputBytes: number;
	readonly staleSkillDays: number;
}
