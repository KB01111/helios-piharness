import type { McpServerSummary } from "../mcp/mcp-client.js";
import type { PermissionAuditEntry } from "../security/permission-policy.js";
import type { TenantContext } from "./tenant-context.js";

/**
 * Inbound transport-agnostic API for the durable agent harness.
 *
 * Both direct TypeScript embedders and the Hono HTTP/SSE transport talk
 * strictly to HarnessPort. Zero Pi Durable / Pi AI / Chord types cross this
 * boundary.
 */

export interface ModelSpec {
	readonly provider: string;
	readonly modelId: string;
}

/** Runtime execution mode: `"plan"` deterministically blocks all mutating tools in `beforeTool`. */
export type ConversationMode = "execute" | "plan";

export interface SideQuestionAnswer {
	readonly conversationId: string;
	readonly answerText: string;
	/** True when the provider request reused the conversation's sessionId prompt-cache prefix. */
	readonly reusedCachePrefix: boolean;
	readonly usage?: UsageBucketView;
}

export interface CreateConversationInput {
	readonly tenant: TenantContext;
	readonly model?: ModelSpec;
	readonly instructions?: string;
	readonly enabledToolPacks?: readonly string[];
	readonly maxTurns?: number;
	/** Initial conversation mode (default `"execute"`). */
	readonly mode?: ConversationMode;
}

export interface ConversationInfo {
	readonly conversationId: string;
	readonly tenant: TenantContext;
	readonly createdAt: number;
	readonly forkedFrom?: {
		readonly conversationId: string;
		readonly entryId: string;
	};
	readonly enabledToolPacks: readonly string[];
	readonly mode?: ConversationMode;
}

export interface SubmitMessageInput {
	readonly conversationId: string;
	readonly content: string;
	readonly requestId?: string;
	readonly whenBusy?: "steer" | "followUp" | "reject";
}

export interface SubmissionHandle {
	readonly submissionId: string;
	readonly conversationId: string;
	readonly requestId?: string;
}

export interface SettledSubmission {
	readonly submissionId: string;
	readonly conversationId: string;
	readonly status: "done" | "unanswered";
	readonly answerText?: string;
	readonly answerEntryId?: string;
	readonly reason?: string;
	readonly detail?: unknown;
}

export interface SignalDeliveryInput {
	readonly taskId: string;
	readonly signalName: string;
	readonly key: string;
	readonly payload: unknown;
}

export interface SignalDeliveryResult {
	readonly status: "delivered" | "duplicate" | "invalid" | "closed" | "not_found";
}

export interface PendingApproval {
	readonly taskId: string;
	readonly conversationId: string;
	readonly callId: string;
	readonly toolName: string;
	readonly action: string;
	readonly summary: string;
	readonly signalName: string;
	readonly signalKey: string;
	readonly createdAt: number;
}

export interface TranscriptEntryView {
	readonly id: string;
	readonly kind: string;
	readonly role?: "system" | "user" | "assistant" | "toolResult";
	readonly text?: string;
	readonly toolName?: string;
	readonly toolCallId?: string;
	readonly isError?: boolean;
	readonly data?: unknown;
}

export interface ToolSlotView {
	readonly callId: string;
	readonly taskId?: string;
	readonly toolName: string;
	readonly status: "pending" | "running" | "sleeping" | "waiting" | "done";
	readonly output?: string;
	readonly entryId?: string;
}

export interface LiveStatusView {
	readonly busy: boolean;
	readonly turns: number;
	readonly streamingText?: string;
	readonly tools: readonly ToolSlotView[];
	readonly runFailure?: {
		readonly reason: string;
		readonly detail?: string;
	};
}

export interface UsageBucketView {
	readonly input: number;
	readonly output: number;
	readonly cacheRead: number;
	readonly cacheWrite: number;
	readonly totalTokens: number;
	readonly totalCost: number;
	/** Prefix cache efficiency ratio: `cacheRead / (input + cacheRead)` when input+cacheRead > 0, else 0. */
	readonly cacheHitRatio?: number;
}

export interface UsageSummaryView {
	readonly models: Readonly<Record<string, UsageBucketView>>;
	readonly tools: Readonly<Record<string, UsageBucketView>>;
}

export interface ToolPackSummary {
	readonly name: string;
	readonly enabled: boolean;
	readonly tools: readonly string[];
}

export interface TaskSettlementView {
	readonly taskId: string;
	readonly status: "completed" | "failed" | "aborted" | "orphaned" | "faulted";
	readonly result?: unknown;
	/** Generic, client-safe message. Internal error text is logged server-side and never returned. */
	readonly error?: string;
	/** Stable machine-readable code (e.g. `task_failed`) present whenever `error` is. */
	readonly errorCode?: string;
}

export type HarnessEvent =
	| {
			readonly type: "snapshot";
			readonly conversationId: string;
			readonly entries: readonly TranscriptEntryView[];
			readonly live: LiveStatusView;
			readonly usage: UsageSummaryView;
	  }
	| {
			readonly type: "text_delta";
			readonly conversationId: string;
			readonly delta: string;
			readonly fullText: string;
	  }
	| {
			readonly type: "entry_appended";
			readonly conversationId: string;
			readonly entry: TranscriptEntryView;
	  }
	| {
			readonly type: "tool_update";
			readonly conversationId: string;
			readonly slot: ToolSlotView;
	  }
	| {
			readonly type: "approval_requested";
			readonly conversationId: string;
			readonly approval: PendingApproval;
	  }
	| {
			readonly type: "run_status";
			readonly conversationId: string;
			readonly busy: boolean;
			readonly turns: number;
			readonly failure?: {
				readonly reason: string;
				readonly detail?: string;
			};
	  }
	| {
			readonly type: "delta_ops";
			readonly conversationId: string;
			readonly opCount: number;
			readonly replaced: boolean;
	  }
	| {
			readonly type: "stream_closed";
			readonly conversationId: string;
			readonly reason: StreamCloseReason;
	  };

/**
 * Why a watch stream ended. `overflow` means the consumer was too slow and the bounded event queue
 * overflowed: the client should reconnect (resuming via `afterEntryId` / `Last-Event-ID`).
 */
export type StreamCloseReason = "stopped" | "aborted" | "ended" | "listener_error" | "failed" | "overflow";

/** Options for `HarnessPort.watchConversation`. */
export interface WatchConversationOptions {
	/**
	 * Resume cursor: the initial `snapshot` only carries transcript entries whose id is greater than this
	 * (live status + usage are always included). Unparseable or unknown ids fall back to a full snapshot.
	 * Transports should use `entry_appended` entry ids as SSE ids and feed `Last-Event-ID` back here.
	 */
	readonly afterEntryId?: string;
	/** Max buffered events per stream before it is closed with reason "overflow" (default 1000). */
	readonly maxQueueSize?: number;
}

export interface HarnessPort {
	createConversation(input: CreateConversationInput): Promise<ConversationInfo>;
	getConversation(conversationId: string): Promise<ConversationInfo | undefined>;
	/**
	 * Forks a conversation. `overrides.tenant` is the CALLER's authenticated context: the child is owned by (and runs
	 * with the permissions of) the caller, never the parent's stored identity, and the caller must be allowed to
	 * access the parent (same tenant and same user, or `admin:conversations`). When omitted (trusted embedders such
	 * as eval runners) the child inherits the parent's tenant.
	 */
	forkConversation(
		conversationId: string,
		atEntryId: string,
		overrides?: Partial<CreateConversationInput>,
	): Promise<ConversationInfo>;

	submit(input: SubmitMessageInput): Promise<SubmissionHandle>;
	waitSubmission(submissionId: string, signal?: AbortSignal): Promise<SettledSubmission>;
	submitAndWait(input: SubmitMessageInput, signal?: AbortSignal): Promise<SettledSubmission>;

	abortConversation(conversationId: string): Promise<void>;
	resetConversation(conversationId: string, handoff?: string): Promise<void>;
	compactConversation(
		conversationId: string,
		instructions?: string,
		signal?: AbortSignal,
	): Promise<{ readonly taskId: string; readonly compacted: boolean }>;
	/** Non-blocking compaction: starts the compaction task and returns its id immediately. */
	startCompaction(conversationId: string, instructions?: string): Promise<{ readonly taskId: string }>;

	deliverSignal(input: SignalDeliveryInput): Promise<SignalDeliveryResult>;
	/**
	 * `options.tenant` (the authenticated caller) restricts results to conversations the caller may access
	 * (same tenant and same user, or `admin:conversations`). `options.tenantId` restricts to a whole tenant (trusted
	 * embedders). Unscoped listing across the all-conversation scan is capped at `options.limit` (default 500).
	 */
	listPendingApprovals(
		conversationId?: string,
		options?: { readonly tenantId?: string; readonly tenant?: TenantContext; readonly limit?: number },
	): Promise<PendingApproval[]>;

	/**
	 * Ownership guards. Throw `NotFoundError` (never a distinct "forbidden") when the conversation/task does not
	 * exist, belongs to another tenant, or belongs to another user of the same tenant (unless the caller holds
	 * `admin:conversations` / `admin:*` / `*`), so existence is not leaked.
	 */
	assertConversationOwner(caller: TenantContext, conversationId: string): Promise<void>;
	assertTaskOwner(caller: TenantContext, taskId: string): Promise<void>;

	triggerSleepCycle(
		conversationId: string,
		options?: { readonly reason?: string },
	): Promise<{ readonly taskId: string }>;
	waitTask(taskId: string, signal?: AbortSignal): Promise<TaskSettlementView>;

	watchConversation(
		conversationId: string,
		signal?: AbortSignal,
		options?: WatchConversationOptions,
	): AsyncIterable<HarnessEvent>;

	setToolPackEnabled(packName: string, enabled: boolean): void;
	listToolPacks(): readonly ToolPackSummary[];

	/** Switches a conversation between `"plan"` (read-only) and `"execute"` modes. */
	setConversationMode(
		conversationId: string,
		mode: ConversationMode,
	): Promise<{ readonly conversationId: string; readonly mode: ConversationMode }>;
	getConversationMode(conversationId: string): Promise<ConversationMode>;

	/**
	 * Ephemeral `/btw` side question: answers using the conversation's committed context
	 * and prompt-cache prefix without invoking tools or writing to the durable transcript.
	 */
	askSideQuestion(conversationId: string, question: string, signal?: AbortSignal): Promise<SideQuestionAnswer>;

	/** Lists configured MCP servers and their lazy discovery state (never credentials/headers). */
	listMcpServers(): readonly McpServerSummary[];

	/** Returns recent permission-policy audit records scoped to `tenantId`. */
	listPermissionAudit(
		tenantId: string,
		options?: { readonly conversationId?: string; readonly limit?: number },
	): readonly PermissionAuditEntry[];

	getTranscript(conversationId: string): Promise<readonly TranscriptEntryView[]>;
	getUsage(): Promise<UsageSummaryView>;
	waitForIdle(conversationId?: string): Promise<void>;
	close(): Promise<void>;
}
