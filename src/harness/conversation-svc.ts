import { BACKGROUND_CONTEXT, withAbortSignal } from "@earendil-works/chord/context";
import { contentText, type AssistantMessage, type Message } from "@earendil-works/pi-ai";
import type { Models } from "@earendil-works/pi-ai/models";
import { getCurrentSystemPrompt } from "@earendil-works/pi-ai/utils/transcript";
import {
	AssistantEntry,
	ProviderDoc,
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
	ConversationMode,
	CreateConversationInput,
	HarnessEvent,
	HarnessPort,
	ModelSpec,
	PendingApproval,
	SettledSubmission,
	SideQuestionAnswer,
	SignalDeliveryInput,
	SignalDeliveryResult,
	SubmissionHandle,
	SubmitMessageInput,
	TaskSettlementView,
	ToolPackSummary,
	TranscriptEntryView,
	UsageBucketView,
	UsageSummaryView,
	WatchConversationOptions,
} from "../contracts/harness-port.js";
import { NotFoundError, ValidationError } from "../contracts/errors.js";
import {
	ADMIN_PERMISSIONS,
	canAccessConversation,
	hasPermission,
	validateTenantContext,
	type TenantContext,
} from "../contracts/tenant-context.js";
import { createLogger, serializeError, type Logger } from "../logger.js";
import { formatFrozenSkillsIndex, loadSkillCatalog } from "../extensions/skills-loader.ext.js";
import type { SleepCycleCheckpoint, SleepCycleInput, SleepCycleResult } from "../extensions/sleep-cycle.ext.js";
import type { McpRegistry, McpServerSummary } from "../mcp/mcp-client.js";
import { usageCostUsd, usageTokens, type BudgetStore } from "../runtime/budget.js";
import { clientSafeDetail, clientSafeReason } from "../runtime/client-safe.js";
import type { HostRuntime } from "../runtime/host-runtime.js";
import { entryToView, usageToView, watchConversationEvents } from "../runtime/stream-adapter.js";
import type { PermissionAuditEntry, PermissionAuditStore } from "../security/permission-policy.js";
import {
	ConversationMetaDoc,
	ConversationModeDoc,
	type ConversationMetaState,
	PendingApprovalsDoc,
	deserializeTenantFromDoc,
	formatFrozenCoreMemory,
	listApprovalsFromDoc,
	refreshFrozenCoreMemory,
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

/** Cap on the unscoped (all-conversation) approvals listing. */
const DEFAULT_APPROVALS_LIMIT = 500;

export interface ConversationServiceOptions {
	/** Server-side diagnostics sink for error details that are deliberately withheld from clients. */
	readonly logger?: Logger;
	readonly hostRuntime: HostRuntime;
	readonly registryManager: HarnessRegistryManager;
	readonly capabilities: AppCapabilities;
	readonly defaultModel: ModelSpec;
	readonly models?: Models;
	readonly budgetStore?: BudgetStore;
	readonly mcpRegistry?: McpRegistry;
	readonly permissionAuditStore?: PermissionAuditStore;
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
	private readonly models?: Models;
	private readonly budgetStore?: BudgetStore;
	private readonly mcpRegistry?: McpRegistry;
	private readonly permissionAuditStore?: PermissionAuditStore;
	private readonly sleepCycleTask: Task<SleepCycleInput, SleepCycleCheckpoint, SleepCycleResult, object>;
	private readonly signalMap = new Map<string, SignalToken<unknown>>();
	private readonly now: () => number;
	private readonly onClose?: () => Promise<void> | void;
	private readonly logger: Logger;
	/** Conversation id -> owning tenant, built incrementally for tenant-scoped approval listings. */
	private readonly tenantIndex = new Map<number, ConversationMetaState["tenant"]>();
	private tenantIndexHighWater = 0;
	private tenantIndexSync: Promise<void> | undefined;

	constructor(options: ConversationServiceOptions) {
		this.logger = options.logger ?? createLogger({ level: "warn", base: { component: "conversation-service" } });
		this.hostRuntime = options.hostRuntime;
		this.harness = options.hostRuntime.harness;
		this.registryManager = options.registryManager;
		this.capabilities = options.capabilities;
		this.defaultModel = options.defaultModel;
		this.models = options.models;
		this.budgetStore = options.budgetStore;
		this.mcpRegistry = options.mcpRegistry;
		this.permissionAuditStore = options.permissionAuditStore;
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
		const initialMode: ConversationMode = input.mode ?? "execute";

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
					doc.subagentDepth = 0;
					doc.parentConversationId = null;

					if (initialMode !== "execute") {
						const modeDoc = await tx.doc(ConversationModeDoc, conversationId);
						modeDoc.mode = initialMode;
						modeDoc.updatedAt = createdAt;
					}
				},
			},
			BACKGROUND_CONTEXT,
		);

		return {
			conversationId: String(conversation.id),
			tenant: input.tenant,
			createdAt,
			enabledToolPacks,
			mode: initialMode,
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
		const modeDoc = await this.harness.snapshot(ConversationModeDoc, id, BACKGROUND_CONTEXT);

		return {
			conversationId: String(id),
			tenant: deserializeTenantFromDoc(meta),
			createdAt: meta.createdAt,
			...(meta.forkedFrom ? { forkedFrom: { ...meta.forkedFrom } } : {}),
			enabledToolPacks: [...meta.enabledToolPacks],
			mode: modeDoc?.mode ?? "execute",
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
		await this.refreshFrozenCore(conv);
		await conv.reset(handoff, BACKGROUND_CONTEXT);
	}

	/**
	 * Reset and compaction already rewrite the visible history (so the prompt-cache prefix is rebuilt anyway):
	 * the cheapest moment to pick up memories consolidated since the conversation started. A no-op when the
	 * rendered core memory is unchanged, and never allowed to fail the reset/compaction itself.
	 */
	private async refreshFrozenCore(conv: Conversation): Promise<void> {
		try {
			const meta = await this.harness.snapshot(ConversationMetaDoc, conv.id, BACKGROUND_CONTEXT);
			if (!meta) return;
			const patch = await refreshFrozenCoreMemory(this.capabilities.memory, meta.tenant, meta, this.now());
			if (!patch.changed) return;
			await conv.commit(async (tx) => {
				const doc = await tx.doc(ConversationMetaDoc, conv.id);
				doc.frozenCoreMemory = patch.frozenCoreMemory;
				doc.frozenCoreMemoryAt = patch.frozenCoreMemoryAt;
			}, BACKGROUND_CONTEXT);
		} catch (error) {
			this.logger.warn("frozen_core_memory_refresh_failed", {
				conversationId: String(conv.id),
				error: serializeError(error),
			});
		}
	}

	async startCompaction(conversationId: string, instructions?: string): Promise<{ readonly taskId: string }> {
		const conv = await this.requireConversation(conversationId);
		await this.refreshFrozenCore(conv);
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
		await this.refreshFrozenCore(conv);
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
		const mayAccess = (meta: { readonly tenant: ConversationMetaState["tenant"] }): boolean => {
			if (caller) return canAccessConversation(caller, meta.tenant);
			return tenantId === undefined || meta.tenant.tenantId === tenantId;
		};

		const mergeWithGateApprovals = (
			base: PendingApproval[],
			gateFilter: { readonly conversationId?: string; readonly tenantId?: string; readonly userId?: string },
		): PendingApproval[] => {
			if (!this.permissionAuditStore) {
				base.sort((a, b) => a.createdAt - b.createdAt);
				return base.slice(0, limit);
			}
			const seen = new Set(base.map((item) => item.taskId));
			const gateItems = this.permissionAuditStore.listPendingGateApprovals({ ...gateFilter, limit });
			for (const item of gateItems) {
				if (!seen.has(item.taskId)) {
					seen.add(item.taskId);
					base.push(item);
				}
			}
			base.sort((a, b) => a.createdAt - b.createdAt);
			return base.slice(0, limit);
		};

		const callerIsTenantAdmin = caller !== undefined && hasPermission(caller, ADMIN_PERMISSIONS.conversations);

		if (conversationId !== undefined) {
			if (caller) await this.assertConversationOwner(caller, conversationId);
			else if (tenantId !== undefined) await this.assertTenantOwnsConversation(tenantId, conversationId);
			const id = parseId<ConversationId>(conversationId, "conversationId");
			const doc = await this.harness.snapshot(PendingApprovalsDoc, id, BACKGROUND_CONTEXT);
			return mergeWithGateApprovals(listApprovalsFromDoc(doc), { conversationId });
		}

		const results: PendingApproval[] = [];

		if (tenantId !== undefined) {
			// Tenant-scoped: consult the incremental conversation->tenant index so only this tenant's conversations
			// have their approvals document read (no per-request meta read of every conversation in the system).
			await this.syncTenantIndex();
			// Oldest conversation first, matching the unscoped scan order.
			for (const rawId of [...this.tenantIndex.keys()].sort((a, b) => a - b)) {
				const owner = this.tenantIndex.get(rawId);
				if (!owner || !mayAccess({ tenant: owner })) continue;
				const doc = await this.harness.snapshot(PendingApprovalsDoc, rawId as ConversationId, BACKGROUND_CONTEXT);
				results.push(...listApprovalsFromDoc(doc));
				if (results.length >= limit) break;
			}
			return mergeWithGateApprovals(results, {
				tenantId,
				...(caller && !callerIsTenantAdmin ? { userId: caller.userId } : {}),
			});
		}

		// Trusted embedders with no tenant filter: scan everything.
		let cursor: Cursor | undefined;
		scan: do {
			const after = cursor;
			const page = await this.harness.commit(
				(tx) => tx.scanConversations({ order: "ascending" }, 200, after),
				BACKGROUND_CONTEXT,
			);
			for (const conv of page.items) {
				const doc = await this.harness.snapshot(PendingApprovalsDoc, conv.id, BACKGROUND_CONTEXT);
				results.push(...listApprovalsFromDoc(doc));
				if (results.length >= limit) break scan;
			}
			cursor = page.next;
		} while (cursor !== undefined);
		return mergeWithGateApprovals(results, {});
	}

	/**
	 * Brings the in-memory conversation->tenant index up to date. Conversation ids only grow, so a newest-first scan
	 * stops at the first id that is already indexed: steady-state cost is one page read, not a full scan. Concurrent
	 * callers share one in-flight sync.
	 */
	private syncTenantIndex(): Promise<void> {
		this.tenantIndexSync ??= this.runTenantIndexSync().finally(() => {
			this.tenantIndexSync = undefined;
		});
		return this.tenantIndexSync;
	}

	private async runTenantIndexSync(): Promise<void> {
		const known = this.tenantIndexHighWater;
		let newest = known;
		let cursor: Cursor | undefined;
		scan: do {
			const after = cursor;
			const page = await this.harness.commit(
				(tx) => tx.scanConversations({ order: "descending" }, 200, after),
				BACKGROUND_CONTEXT,
			);
			for (const conv of page.items) {
				const id = Number(conv.id);
				if (id <= known) break scan;
				const meta = await this.harness.snapshot(ConversationMetaDoc, conv.id, BACKGROUND_CONTEXT);
				if (meta) this.tenantIndex.set(id, meta.tenant);
				newest = Math.max(newest, id);
			}
			cursor = page.next;
		} while (cursor !== undefined);
		this.tenantIndexHighWater = newest;
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
		// Durable single-flight: the lookup and the create share one transaction, so concurrent callers (and other
		// processes / restarts) reuse a live cycle for this conversation instead of starting a duplicate.
		const taskId = await conv.commit(async (tx) => {
			const recent = await tx.scanTasks(
				{ conversationId: conv.id, kind: this.sleepCycleTask.definition.name, order: "descending" },
				10,
			);
			const live = recent.items.find(
				(task) => task.state.status !== "terminal" && task.state.status !== "completing",
			);
			if (live) return live.id;
			return tx.createTask(
				this.sleepCycleTask,
				{ reason: options?.reason ?? "manual", staleSkillDays: null },
				{
					ownership: { kind: "conversation" },
					background: true,
				},
			);
		}, BACKGROUND_CONTEXT);
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

	async setConversationMode(
		conversationId: string,
		mode: ConversationMode,
	): Promise<{ readonly conversationId: string; readonly mode: ConversationMode }> {
		if (mode !== "execute" && mode !== "plan") {
			throw new ValidationError(`Invalid conversation mode "${String(mode)}"`);
		}
		const conv = await this.requireConversation(conversationId);
		const updatedAt = this.now();
		await conv.commit(async (tx) => {
			const doc = await tx.doc(ConversationModeDoc, conv.id);
			doc.mode = mode;
			doc.updatedAt = updatedAt;
		}, BACKGROUND_CONTEXT);
		return { conversationId, mode };
	}

	async getConversationMode(conversationId: string): Promise<ConversationMode> {
		const conv = await this.requireConversation(conversationId);
		const doc = await this.harness.snapshot(ConversationModeDoc, conv.id, BACKGROUND_CONTEXT);
		return doc?.mode ?? "execute";
	}

	/**
	 * Cached ephemeral `/btw` side question:
	 * Reads the conversation's committed ContextView + ProviderDoc.sessionId and issues a
	 * single tool-free completion reusing the exact leading system prompt prefix without
	 * writing anything to the durable conversation transcript.
	 */
	async askSideQuestion(
		conversationId: string,
		question: string,
		signal?: AbortSignal,
	): Promise<SideQuestionAnswer> {
		const trimmed = question.trim();
		if (!trimmed) {
			throw new ValidationError("Side question must be a non-empty string");
		}
		if (!this.models) {
			throw new Error("Side questions require model catalog access on ConversationService");
		}

		const ctx = signal ? withAbortSignal(signal, BACKGROUND_CONTEXT) : BACKGROUND_CONTEXT;
		const conv = await this.requireConversation(conversationId);
		const [view, agent, providerDoc, meta] = await Promise.all([
			conv.context(ctx),
			conv.agent(ctx),
			this.harness.snapshot(ProviderDoc, conv.id, ctx),
			this.harness.snapshot(ConversationMetaDoc, conv.id, ctx),
		]);

		const modelRef = agent.model ?? this.defaultModel;
		const model = this.models.getModel(modelRef.provider, modelRef.modelId);
		if (!model) {
			throw new Error(`Model "${modelRef.provider}:${modelRef.modelId}" is not registered`);
		}

		const systemPrompt = getCurrentSystemPrompt(view.messages as Message[]);
		const historyMessages = view.messages.filter((m) => m.role !== "system");
		const sidePrompt = [
			"[Ephemeral /btw side question — answer concisely from the conversation context above without calling any tools or modifying state.]",
			trimmed,
		].join("\n\n");

		const sessionId = providerDoc?.sessionId;
		const response = await this.models.completeSimple(
			model,
			{
				...(systemPrompt ? { systemPrompt } : {}),
				messages: [...historyMessages, { role: "user", content: sidePrompt, timestamp: this.now() }],
			},
			{
				...(signal ? { signal } : {}),
				...(sessionId ? { sessionId } : {}),
			},
		);

		if (response.stopReason === "error") {
			throw new Error(response.errorMessage ?? "Side question model call failed");
		}

		let usageView: UsageBucketView | undefined;
		if (response.usage) {
			const tokens = usageTokens(response.usage);
			const cost = usageCostUsd(response.usage);
			if (this.budgetStore && (tokens > 0 || cost > 0)) {
				try {
					this.budgetStore.recordUsage(meta?.tenant.tenantId ?? "default", tokens, cost, this.now());
				} catch {
					// metering must never break a completed side answer
				}
			}
			const denom = (response.usage.input ?? 0) + (response.usage.cacheRead ?? 0);
			usageView = {
				input: response.usage.input ?? 0,
				output: response.usage.output ?? 0,
				cacheRead: response.usage.cacheRead ?? 0,
				cacheWrite: response.usage.cacheWrite ?? 0,
				totalTokens: response.usage.totalTokens ?? tokens,
				totalCost: cost,
				cacheHitRatio: denom > 0 ? Number(((response.usage.cacheRead ?? 0) / denom).toFixed(4)) : 0,
			};
		}

		return {
			conversationId,
			answerText: contentText(response.content).trim(),
			reusedCachePrefix: Boolean(sessionId),
			...(usageView ? { usage: usageView } : {}),
		};
	}

	listMcpServers(): readonly McpServerSummary[] {
		return this.mcpRegistry ? this.mcpRegistry.listServers() : [];
	}

	listPermissionAudit(
		tenantId: string,
		options?: { readonly conversationId?: string; readonly limit?: number },
	): readonly PermissionAuditEntry[] {
		return this.permissionAuditStore ? this.permissionAuditStore.listAudit(tenantId, options) : [];
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
