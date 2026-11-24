import { BACKGROUND_CONTEXT, withAbortSignal } from "@earendil-works/chord/context";
import { contentText, type AssistantMessage } from "@earendil-works/pi-ai";
import {
	AssistantEntry,
	type Conversation,
	type ConversationId,
	type Cursor,
	type EntryId,
	type Harness,
	type SignalToken,
	type SubmissionId,
	type Task,
	type TaskId,
} from "@earendil-works/pi-durable";
import type { AppCapabilities } from "../contracts/app-capabilities.js";
import type {
	ConversationInfo,
	CreateConversationInput,
	HarnessEvent,
	HarnessPort,
	ModelSpec,
	PendingApproval,
	SettledSubmission,
	SignalDeliveryInput,
	SignalDeliveryResult,
	SubmissionHandle,
	SubmitMessageInput,
	TaskSettlementView,
	ToolPackSummary,
	TranscriptEntryView,
	UsageSummaryView,
	WatchConversationOptions,
} from "../contracts/harness-port.js";
import { NotFoundError, ValidationError } from "../contracts/errors.js";
import { canAccessConversation, validateTenantContext, type TenantContext } from "../contracts/tenant-context.js";
import { createLogger, serializeError, type Logger } from "../logger.js";
import { formatFrozenSkillsIndex, loadSkillCatalog } from "../extensions/skills-loader.ext.js";
import type { SleepCycleCheckpoint, SleepCycleInput, SleepCycleResult } from "../extensions/sleep-cycle.ext.js";
import type { HostRuntime } from "../runtime/host-runtime.js";
import { entryToView, usageToView, watchConversationEvents } from "../runtime/stream-adapter.js";
import {
	ConversationMetaDoc,
	type ConversationMetaState,
	PendingApprovalsDoc,
	deserializeTenantFromDoc,
	formatFrozenCoreMemory,
	listApprovalsFromDoc,
	serializeTenantForDoc,
} from "./app-documents.js";
import type { HarnessRegistryManager } from "./registry.js";

function parseId<T extends number>(raw: string, label: string): T {
	const parsed = Number.parseInt(raw, 10);
	if (!Number.isSafeInteger(parsed) || parsed <= 0 || String(parsed) !== raw.trim()) {
		throw new ValidationError(`Invalid ${label} "${raw}"`);
	}
	return parsed as T;
}

/** Reasons are stable snake_case codes; anything else (free text) is collapsed so it cannot leak internals. */
const SAFE_REASON_PATTERN = /^[a-z][a-z0-9_.-]{0,63}$/;
/** Budget vetoes are intentionally client-visible (they tell the caller which limit to raise). */
const CLIENT_SAFE_DETAIL_PREFIX = "BUDGET_EXCEEDED ";
const MAX_CLIENT_DETAIL_CHARS = 500;
/** Cap on the unscoped (all-conversation) approvals listing. */
const DEFAULT_APPROVALS_LIMIT = 500;

function clientSafeReason(reason: string): string {
	return SAFE_REASON_PATTERN.test(reason) ? reason : "unanswered";
}

function clientSafeDetail(detail: unknown): string | undefined {
	if (typeof detail === "string" && detail.startsWith(CLIENT_SAFE_DETAIL_PREFIX)) {
		return detail.slice(0, MAX_CLIENT_DETAIL_CHARS);
	}
	return undefined;
}

export interface ConversationServiceOptions {
	/** Server-side diagnostics sink for error details that are deliberately withheld from clients. */
	readonly logger?: Logger;
	readonly hostRuntime: HostRuntime;
	readonly registryManager: HarnessRegistryManager;
	readonly capabilities: AppCapabilities;
	readonly defaultModel: ModelSpec;
	readonly sleepCycleTask: Task<SleepCycleInput, SleepCycleCheckpoint, SleepCycleResult, object>;
	readonly signals?: ReadonlyArray<SignalToken<unknown>>;
	readonly now?: () => number;
	readonly onClose?: () => Promise<void> | void;
}

/**
 * Implements the transport-agnostic `HarnessPort` over Pi Durable's `Harness`.
 *
 * All Pi Durable / Chord types stay encapsulated inside `src/harness/` and
 * `src/runtime/`; external callers interact exclusively via plain JSON and
 * opaque string IDs.
 */
export class ConversationService implements HarnessPort {
	private readonly hostRuntime: HostRuntime;
	private readonly harness: Harness;
	private readonly registryManager: HarnessRegistryManager;
	private readonly capabilities: AppCapabilities;
	private readonly defaultModel: ModelSpec;
	private readonly sleepCycleTask: Task<SleepCycleInput, SleepCycleCheckpoint, SleepCycleResult, object>;
	private readonly signalMap = new Map<string, SignalToken<unknown>>();
	private readonly now: () => number;
	private readonly onClose?: () => Promise<void> | void;
	private readonly logger: Logger;

	constructor(options: ConversationServiceOptions) {
		this.logger = options.logger ?? createLogger({ level: "warn", base: { component: "conversation-service" } });
		this.hostRuntime = options.hostRuntime;
		this.harness = options.hostRuntime.harness;
		this.registryManager = options.registryManager;
		this.capabilities = options.capabilities;
		this.defaultModel = options.defaultModel;
		this.sleepCycleTask = options.sleepCycleTask;
		this.now = options.now ?? (() => Date.now());
		this.onClose = options.onClose;

		if (options.signals) {
			for (const token of options.signals) {
				this.registerSignal(token);
			}
		}
	}

	registerSignal<T>(token: SignalToken<T>): void {
		this.signalMap.set(token.name, token as SignalToken<unknown>);
	}

	async createConversation(input: CreateConversationInput): Promise<ConversationInfo> {
		validateTenantContext(input.tenant);
		const createdAt = this.now();

		// Freeze session-start Core Memory and enabled SKILL.md index atomically into ConversationMetaDoc
		const coreMemories = await this.capabilities.memory.getCoreMemory(input.tenant.tenantId, input.tenant.userId);
		const frozenCoreMemory = formatFrozenCoreMemory(coreMemories);

		const disabledSkills = await this.capabilities.telemetry.listDisabledSkills(input.tenant.tenantId);
		const skillsCatalog = loadSkillCatalog(this.capabilities.skillsDir, disabledSkills);
		const frozenSkillsIndex = formatFrozenSkillsIndex(skillsCatalog);

		const selectedExtensions = this.registryManager.resolveConversationExtensionSelection(input.enabledToolPacks);
		const enabledToolPacks = input.enabledToolPacks
			? [...input.enabledToolPacks]
			: this.registryManager.getEnabledPackNames();

		const model = input.model ?? this.defaultModel;

		const conversation = await this.harness.createConversation(
			{
				ownership: { kind: "ownerless" },
				agent: {
					model: { provider: model.provider, modelId: model.modelId },
					...(input.instructions !== undefined ? { instructions: input.instructions } : {}),
					...(selectedExtensions !== undefined ? { extensions: selectedExtensions } : {}),
					...(input.maxTurns !== undefined ? { maxTurns: input.maxTurns } : {}),
				},
				init: async (tx, conversationId) => {
					const doc = await tx.doc(ConversationMetaDoc, conversationId);
					doc.tenant = serializeTenantForDoc(input.tenant);
					doc.createdAt = createdAt;
					doc.frozenCoreMemory = frozenCoreMemory;
					doc.frozenCoreMemoryAt = createdAt;
					doc.frozenSkillsIndex = frozenSkillsIndex;
					doc.enabledToolPacks = [...enabledToolPacks];
				},
			},
			BACKGROUND_CONTEXT,
		);

		return {
			conversationId: String(conversation.id),
			tenant: input.tenant,
			createdAt,
			enabledToolPacks,
		};
	}

	async getConversation(conversationId: string): Promise<ConversationInfo | undefined> {
		let id: ConversationId;
		try {
			id = parseId<ConversationId>(conversationId, "conversationId");
		} catch {
			return undefined;
		}
		const conv = await this.harness.conversation(id, BACKGROUND_CONTEXT);
		if (!conv) return undefined;
		const meta = await this.harness.snapshot(ConversationMetaDoc, id, BACKGROUND_CONTEXT);
		if (!meta) return undefined;

		return {
			conversationId: String(id),
			tenant: deserializeTenantFromDoc(meta),
			createdAt: meta.createdAt,
			...(meta.forkedFrom ? { forkedFrom: { ...meta.forkedFrom } } : {}),
			enabledToolPacks: [...meta.enabledToolPacks],
		};
	}

	async forkConversation(
		conversationId: string,
		atEntryId: string,
		overrides?: Partial<CreateConversationInput>,
	): Promise<ConversationInfo> {
		// `overrides.tenant` is the authenticated CALLER. It must be allowed to access the parent (same tenant and
		// same user, or admin:conversations) and becomes the child's identity. The parent's stored permissions are
		// never copied to a different principal.
		if (overrides?.tenant) {
			validateTenantContext(overrides.tenant);
			await this.assertConversationOwner(overrides.tenant, conversationId);
		}
		const parent = await this.requireConversation(conversationId);
		const entryId = parseId<EntryId>(atEntryId, "atEntryId");
		const parentMeta = await this.harness.snapshot(ConversationMetaDoc, parent.id, BACKGROUND_CONTEXT);
		if (!parentMeta) {
			throw new NotFoundError(`Conversation "${conversationId}" not found`);
		}
		// No caller supplied (trusted embedders such as the eval replay runner): inherit the parent's identity.
		const childTenant = serializeTenantForDoc(overrides?.tenant ?? deserializeTenantFromDoc(parentMeta));

		const selectedExtensions = overrides?.enabledToolPacks
			? this.registryManager.resolveConversationExtensionSelection(overrides.enabledToolPacks)
			: undefined;

		const createdAt = this.now();
		const fork = await parent.fork(
			entryId,
			{
				ownership: { kind: "ownerless" },
				agent: {
					...(overrides?.model
						? { model: { provider: overrides.model.provider, modelId: overrides.model.modelId } }
						: {}),
					...(overrides?.instructions !== undefined ? { instructions: overrides.instructions } : {}),
					...(selectedExtensions !== undefined ? { extensions: selectedExtensions } : {}),
					...(overrides?.maxTurns !== undefined ? { maxTurns: overrides.maxTurns } : {}),
				},
				init: async (tx, childId) => {
					const doc = await tx.doc(ConversationMetaDoc, childId);
					doc.createdAt = createdAt;
					doc.forkedFrom = {
						conversationId,
						entryId: atEntryId,
					};
					doc.tenant = childTenant;
					if (overrides?.enabledToolPacks) {
						doc.enabledToolPacks = [...overrides.enabledToolPacks];
					}
				},
			},
			BACKGROUND_CONTEXT,
		);

		const childMeta = await this.harness.snapshot(ConversationMetaDoc, fork.id, BACKGROUND_CONTEXT);
		const tenant = childMeta
			? deserializeTenantFromDoc(childMeta)
			: parentMeta
				? deserializeTenantFromDoc(parentMeta)
				: { tenantId: "default", userId: "anonymous", permissions: [] };

		return {
			conversationId: String(fork.id),
			tenant,
			createdAt,
			forkedFrom: {
				conversationId,
				entryId: atEntryId,
			},
			enabledToolPacks: childMeta ? [...childMeta.enabledToolPacks] : [],
		};
	}

	async submit(input: SubmitMessageInput): Promise<SubmissionHandle> {
		const conv = await this.requireConversation(input.conversationId);
		const submission = await conv.submit(
			{
				type: "input",
				content: input.content,
				...(input.requestId !== undefined ? { requestId: input.requestId } : {}),
				...(input.whenBusy !== undefined ? { whenBusy: input.whenBusy } : {}),
			},
			BACKGROUND_CONTEXT,
		);

		return {
			submissionId: String(submission.id),
			conversationId: input.conversationId,
			...(input.requestId !== undefined ? { requestId: input.requestId } : {}),
		};
	}

	async waitSubmission(submissionId: string, signal?: AbortSignal): Promise<SettledSubmission> {
		const ctx = signal ? withAbortSignal(signal, BACKGROUND_CONTEXT) : BACKGROUND_CONTEXT;
		const id = parseId<SubmissionId>(submissionId, "submissionId");
		const submission = await this.harness.submission(id, ctx);
		if (!submission) {
			throw new NotFoundError(`Submission "${submissionId}" not found`);
		}

		const settled = await submission.wait(ctx);
		const conversationId = String(settled.conversationId);

		if (settled.status === "done" && (settled.type === "input" || settled.type === "continue")) {
			const answerEntry = await this.harness.commit(
				(tx) => tx.entry(AssistantEntry, settled.answer),
				BACKGROUND_CONTEXT,
			);
			const assistantMsg = answerEntry?.model?.[0] as AssistantMessage | undefined;
			const answerText = assistantMsg ? contentText(assistantMsg.content) : "";
			return {
				submissionId: String(settled.id),
				conversationId,
				status: "done",
				answerText,
				answerEntryId: String(settled.answer),
			};
		}

		if (settled.status === "unanswered") {
			// Free-text reason/detail can carry provider errors, paths or SQL: log them, return a stable code only.
			this.logger.warn("submission_unanswered", {
				submissionId: String(settled.id),
				conversationId,
				reason: settled.reason,
				detail: settled.detail,
			});
			const detail = clientSafeDetail(settled.detail);
			return {
				submissionId: String(settled.id),
				conversationId,
				status: "unanswered",
				reason: clientSafeReason(settled.reason),
				...(detail !== undefined ? { detail } : {}),
			};
		}
		return {
			submissionId: String(settled.id),
			conversationId,
			status: settled.status,
		};
	}

	async submitAndWait(input: SubmitMessageInput, signal?: AbortSignal): Promise<SettledSubmission> {
		const handle = await this.submit(input);
		return this.waitSubmission(handle.submissionId, signal);
	}

	async abortConversation(conversationId: string): Promise<void> {
		const conv = await this.requireConversation(conversationId);
		await conv.abort(BACKGROUND_CONTEXT);
	}

	async resetConversation(conversationId: string, handoff?: string): Promise<void> {
		const conv = await this.requireConversation(conversationId);
		await conv.reset(handoff, BACKGROUND_CONTEXT);
	}

	async startCompaction(conversationId: string, instructions?: string): Promise<{ readonly taskId: string }> {
		const conv = await this.requireConversation(conversationId);
		const taskId = await conv.compact(instructions, BACKGROUND_CONTEXT);
		return { taskId: String(taskId) };
	}

	async compactConversation(
		conversationId: string,
		instructions?: string,
		signal?: AbortSignal,
	): Promise<{ readonly taskId: string; readonly compacted: boolean }> {
		const ctx = signal ? withAbortSignal(signal, BACKGROUND_CONTEXT) : BACKGROUND_CONTEXT;
		const conv = await this.requireConversation(conversationId);
		const taskId = await conv.compact(instructions, ctx);
		const settled = await this.harness.waitForTask(taskId, ctx);
		if (settled.state.outcome.status !== "completed") {
			return { taskId: String(taskId), compacted: false };
		}
		const result = settled.state.outcome.result;
		if (result.submissionId !== undefined) {
			const sub = await this.harness.submission(result.submissionId, ctx);
			if (sub) {
				const placed = await sub.wait(ctx);
				return { taskId: String(taskId), compacted: placed.status === "done" };
			}
		}
		return { taskId: String(taskId), compacted: result.entryId !== undefined };
	}

	async deliverSignal(input: SignalDeliveryInput): Promise<SignalDeliveryResult> {
		const token = this.signalMap.get(input.signalName);
		if (!token) {
			return { status: "invalid" };
		}
		const taskId = parseId<TaskId>(input.taskId, "taskId");
		const status = await this.harness.signal(taskId, token, input.key, input.payload, BACKGROUND_CONTEXT);
		return { status };
	}

	async listPendingApprovals(
		conversationId?: string,
		options?: { readonly tenantId?: string; readonly tenant?: TenantContext; readonly limit?: number },
	): Promise<PendingApproval[]> {
		const caller = options?.tenant;
		const tenantId = caller?.tenantId ?? options?.tenantId;
		const limit = Math.max(1, options?.limit ?? DEFAULT_APPROVALS_LIMIT);
		/** Tenant check always precedes loading the approvals doc, so other tenants' docs are never read. */
		const mayAccess = (meta: Readonly<ConversationMetaState>): boolean => {
			if (caller) return canAccessConversation(caller, meta.tenant);
			return tenantId === undefined || meta.tenant.tenantId === tenantId;
		};

		if (conversationId !== undefined) {
			if (caller) await this.assertConversationOwner(caller, conversationId);
			else if (tenantId !== undefined) await this.assertTenantOwnsConversation(tenantId, conversationId);
			const id = parseId<ConversationId>(conversationId, "conversationId");
			const doc = await this.harness.snapshot(PendingApprovalsDoc, id, BACKGROUND_CONTEXT);
			return listApprovalsFromDoc(doc).slice(0, limit);
		}

		const results: PendingApproval[] = [];
		let cursor: Cursor | undefined;
		scan: do {
			const after = cursor;
			const page = await this.harness.commit(
				(tx) => tx.scanConversations({ order: "ascending" }, 200, after),
				BACKGROUND_CONTEXT,
			);
			for (const conv of page.items) {
				if (tenantId !== undefined) {
					const meta = await this.harness.snapshot(ConversationMetaDoc, conv.id, BACKGROUND_CONTEXT);
					if (!meta || !mayAccess(meta)) continue;
				}
				const doc = await this.harness.snapshot(PendingApprovalsDoc, conv.id, BACKGROUND_CONTEXT);
				results.push(...listApprovalsFromDoc(doc));
				if (results.length >= limit) break scan;
			}
			cursor = page.next;
		} while (cursor !== undefined);
		results.sort((a, b) => a.createdAt - b.createdAt);
		return results.slice(0, limit);
	}

	async assertConversationOwner(caller: TenantContext, conversationId: string): Promise<void> {
		const meta = await this.loadMetaIfExists(conversationId);
		if (!meta || !canAccessConversation(caller, meta.tenant)) {
			throw new NotFoundError(`Conversation "${conversationId}" not found`);
		}
	}

	/** Tenant-wide ownership check for trusted embedders that only know a tenant id (no user). */
	private async assertTenantOwnsConversation(tenantId: string, conversationId: string): Promise<void> {
		const meta = await this.loadMetaIfExists(conversationId);
		if (!meta || meta.tenant.tenantId !== tenantId) {
			throw new NotFoundError(`Conversation "${conversationId}" not found`);
		}
	}

	private async loadMetaIfExists(conversationId: string): Promise<Readonly<ConversationMetaState> | undefined> {
		let id: ConversationId;
		try {
			id = parseId<ConversationId>(conversationId, "conversationId");
		} catch {
			return undefined;
		}
		const meta = await this.harness.snapshot(ConversationMetaDoc, id, BACKGROUND_CONTEXT);
		if (!meta) return undefined;
		const conv = await this.harness.conversation(id, BACKGROUND_CONTEXT);
		return conv ? meta : undefined;
	}

	async assertTaskOwner(caller: TenantContext, taskId: string): Promise<void> {
		let id: TaskId;
		try {
			id = parseId<TaskId>(taskId, "taskId");
		} catch {
			throw new NotFoundError(`Task "${taskId}" not found`);
		}
		const task = await this.harness.getTask(id, BACKGROUND_CONTEXT);
		if (!task) throw new NotFoundError(`Task "${taskId}" not found`);
		await this.assertConversationOwner(caller, String(task.conversationId)).catch(() => {
			throw new NotFoundError(`Task "${taskId}" not found`);
		});
	}

	async triggerSleepCycle(
		conversationId: string,
		options?: { readonly reason?: string },
	): Promise<{ readonly taskId: string }> {
		const conv = await this.requireConversation(conversationId);
		const taskId = await conv.commit(
			(tx) =>
				tx.createTask(
					this.sleepCycleTask,
					{ reason: options?.reason ?? "manual", staleSkillDays: null },
					{
						ownership: { kind: "conversation" },
						background: true,
					},
				),
			BACKGROUND_CONTEXT,
		);
		this.hostRuntime.resume();
		return { taskId: String(taskId) };
	}

	async waitTask(taskId: string, signal?: AbortSignal): Promise<TaskSettlementView> {
		const ctx = signal ? withAbortSignal(signal, BACKGROUND_CONTEXT) : BACKGROUND_CONTEXT;
		const id = parseId<TaskId>(taskId, "taskId");
		const settled = await this.harness.waitForTask(id, ctx);
		const outcome = settled.state.outcome;
		if (outcome.status === "completed") {
			return {
				taskId: String(settled.id),
				status: "completed",
				result: outcome.result,
			};
		}
		// Internal error text (provider errors, paths, SQL) is logged, never returned to clients.
		if (outcome.status === "failed" || outcome.status === "faulted") {
			this.logger.error("task_settled_with_error", {
				taskId: String(settled.id),
				status: outcome.status,
				error: serializeError(outcome.error),
			});
			return {
				taskId: String(settled.id),
				status: outcome.status,
				error: outcome.status === "failed" ? "Task failed" : "Task faulted",
				errorCode: `task_${outcome.status}`,
			};
		}
		if (outcome.reason) {
			this.logger.warn("task_settled_without_completion", {
				taskId: String(settled.id),
				status: outcome.status,
				reason: outcome.reason,
			});
		}
		return {
			taskId: String(settled.id),
			status: outcome.status,
			error: outcome.status === "aborted" ? "Task aborted" : "Task orphaned",
			errorCode: `task_${outcome.status}`,
		};
	}

	watchConversation(
		conversationId: string,
		signal?: AbortSignal,
		options?: WatchConversationOptions,
	): AsyncIterable<HarnessEvent> {
		const self = this;
		return {
			async *[Symbol.asyncIterator](): AsyncGenerator<HarnessEvent> {
				const conv = await self.requireConversation(conversationId);
				for await (const event of watchConversationEvents(conv, signal, options)) {
					yield event;
				}
			},
		};
	}

	setToolPackEnabled(packName: string, enabled: boolean): void {
		this.registryManager.setToolPackEnabled(packName, enabled);
	}

	listToolPacks(): readonly ToolPackSummary[] {
		return this.registryManager.listToolPacks();
	}

	async getTranscript(conversationId: string): Promise<readonly TranscriptEntryView[]> {
		const conv = await this.requireConversation(conversationId);
		const entries: TranscriptEntryView[] = [];
		let cursor: Awaited<ReturnType<Conversation["entries"]>>["next"];
		do {
			const page = await conv.entries({ order: "ascending" }, 256, cursor, BACKGROUND_CONTEXT);
			for (const entry of page.items) {
				entries.push(entryToView(entry));
			}
			cursor = page.next;
		} while (cursor !== undefined);
		return entries;
	}

	async getUsage(): Promise<UsageSummaryView> {
		const usage = await this.harness.usage(BACKGROUND_CONTEXT);
		return usageToView(usage);
	}

	async waitForIdle(conversationId?: string): Promise<void> {
		if (conversationId !== undefined) {
			const conv = await this.requireConversation(conversationId);
			await conv.waitForIdle(BACKGROUND_CONTEXT);
			return;
		}
		await this.harness.waitForIdle(BACKGROUND_CONTEXT);
	}

	async close(): Promise<void> {
		await this.hostRuntime.close(BACKGROUND_CONTEXT);
		await this.onClose?.();
	}

	private async requireConversation(conversationId: string): Promise<Conversation> {
		const id = parseId<ConversationId>(conversationId, "conversationId");
		const conv = await this.harness.conversation(id, BACKGROUND_CONTEXT);
		if (!conv) {
			throw new NotFoundError(`Conversation "${conversationId}" not found`);
		}
		return conv;
	}
}
