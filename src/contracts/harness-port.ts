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

export interface CreateConversationInput {
	readonly tenant: TenantContext;
	readonly model?: ModelSpec;
	readonly instructions?: string;
	readonly enabledToolPacks?: readonly string[];
	readonly maxTurns?: number;
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
	readonly error?: string;
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
			readonly reason: string;
	  };

export interface HarnessPort {
	createConversation(input: CreateConversationInput): Promise<ConversationInfo>;
	getConversation(conversationId: string): Promise<ConversationInfo | undefined>;
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
	): Promise<{ readonly taskId: string; readonly compacted: boolean }>;

	deliverSignal(input: SignalDeliveryInput): Promise<SignalDeliveryResult>;
	listPendingApprovals(conversationId?: string): Promise<PendingApproval[]>;

	triggerSleepCycle(
		conversationId: string,
		options?: { readonly reason?: string },
	): Promise<{ readonly taskId: string }>;
	waitTask(taskId: string, signal?: AbortSignal): Promise<TaskSettlementView>;

	watchConversation(conversationId: string, signal?: AbortSignal): AsyncIterable<HarnessEvent>;

	setToolPackEnabled(packName: string, enabled: boolean): void;
	listToolPacks(): readonly ToolPackSummary[];

	getTranscript(conversationId: string): Promise<readonly TranscriptEntryView[]>;
	getUsage(): Promise<UsageSummaryView>;
	waitForIdle(conversationId?: string): Promise<void>;
	close(): Promise<void>;
}
