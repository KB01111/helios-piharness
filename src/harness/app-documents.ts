import type { JsonObject } from "@earendil-works/pi-durable";
import { defineDoc } from "@earendil-works/pi-durable";
import type { LongTermMemory, MemoryService } from "../contracts/app-capabilities.js";
import type { PendingApproval } from "../contracts/harness-port.js";
import type { TenantContext } from "../contracts/tenant-context.js";
import { formatMemoryLine, renderUntrustedMemoryBlock } from "../memory/sanitize.js";

/**
 * Typed Pi Durable Documents committed atomically alongside the conversation
 * transcript. All app-level durable metadata lives here so restarts and forks
 * inherit exact state atomically.
 */

export type ConversationMetaState = {
	tenant: {
		tenantId: string;
		userId: string;
		permissions: string[];
		workspaceId: string | null;
		metadata: Record<string, string | number | boolean | null> | null;
	};
	createdAt: number;
	/**
	 * Session-start frozen snapshot of active Core Memory.
	 * Never mutated during an active session so the system prompt prefix
	 * stays byte-identical across turns (preserving provider KV caches).
	 */
	frozenCoreMemory: string;
	frozenCoreMemoryAt: number;
	/**
	 * Session-start frozen snapshot of enabled progressive-disclosure skills index.
	 */
	frozenSkillsIndex: string;
	enabledToolPacks: string[];
	forkedFrom: {
		conversationId: string;
		entryId: string;
	} | null;
};

export const ConversationMetaDoc = defineDoc<ConversationMetaState>({
	kind: "app.conversation-meta",
	version: 1,
	scope: "conversation",
	history: "rewindable",
	fork: "asOf",
	initial: () => ({
		tenant: {
			tenantId: "default",
			userId: "anonymous",
			permissions: [],
			workspaceId: null,
			metadata: null,
		},
		createdAt: 0,
		frozenCoreMemory: "",
		frozenCoreMemoryAt: 0,
		frozenSkillsIndex: "",
		enabledToolPacks: [],
		forkedFrom: null,
	}),
	checkpointWhen: (_value, _ops, info) => info.deltasSinceBase >= 16,
});

export interface PendingApprovalsState extends JsonObject {
	items: Record<
		string,
		{
			taskId: string;
			conversationId: string;
			callId: string;
			toolName: string;
			action: string;
			summary: string;
			signalName: string;
			signalKey: string;
			createdAt: number;
		}
	>;
}

export const PendingApprovalsDoc = defineDoc<PendingApprovalsState>({
	kind: "app.pending-approvals",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ items: {} }),
	checkpointWhen: (_value, _ops, info) => info.deltasSinceBase >= 16,
});

export interface RunTelemetryDocState extends JsonObject {
	totalRunsCaptured: number;
	lastCapturedEntryId: number;
	sleepCyclesCompleted: number;
	lastSleepCycleAt: number;
}

export const RunTelemetryDoc = defineDoc<RunTelemetryDocState>({
	kind: "app.run-telemetry",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({
		totalRunsCaptured: 0,
		lastCapturedEntryId: 0,
		sleepCyclesCompleted: 0,
		lastSleepCycleAt: 0,
	}),
	checkpointWhen: (_value, _ops, info) => info.deltasSinceBase >= 32,
});

export interface SideEffectRecord extends JsonObject {
	idempotencyKey: string;
	recordId: string;
	value: string;
	status: "staged" | "committed";
	updatedAt: number;
}

export interface PendingSideEffectsState extends JsonObject {
	records: Record<string, SideEffectRecord>;
}

export const PendingSideEffectsDoc = defineDoc<PendingSideEffectsState>({
	kind: "app.side-effects",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ records: {} }),
	checkpointWhen: (_value, _ops, info) => info.deltasSinceBase >= 16,
});

/**
 * Deterministically renders session-start Core Memory bullets.
 * Order follows the input (callers pass `getCoreMemory`'s stable ordering); formatting is stable so repeated
 * sessions with identical memory state render byte-identically.
 *
 * Memory text is untrusted (it originates from user text / assistant summaries), so every field is
 * sanitized at render time and the bullets are wrapped in a delimited "untrusted data, not instructions" block.
 */
export function formatFrozenCoreMemory(memories: readonly LongTermMemory[]): string {
	if (memories.length === 0) return "";
	return renderUntrustedMemoryBlock(
		memories.map((m) => formatMemoryLine(m)),
		[
			"Loaded at session start (frozen for prompt cache stability).",
			"Use `memory_search` to retrieve additional or recently updated memories on demand.",
		],
	);
}

export interface FrozenCoreMemoryPatch {
	readonly changed: boolean;
	readonly frozenCoreMemory: string;
	readonly frozenCoreMemoryAt: number;
}

/**
 * Pure helper to rebuild the frozen core memory (e.g. on conversation reset / compaction) so long
 * conversations do not run on a stale session-start snapshot.
 *
 * Idempotent and cache-friendly: when the rendered text is byte-identical to `current`, the patch is
 * `changed: false` and the ORIGINAL timestamp is kept, so callers can skip the document write and the
 * system prompt prefix stays untouched.
 */
export function buildFrozenCoreMemoryPatch(
	current: Pick<ConversationMetaState, "frozenCoreMemory" | "frozenCoreMemoryAt">,
	memories: readonly LongTermMemory[],
	nowMs: number,
): FrozenCoreMemoryPatch {
	const rendered = formatFrozenCoreMemory(memories);
	if (rendered === current.frozenCoreMemory) {
		return {
			changed: false,
			frozenCoreMemory: current.frozenCoreMemory,
			frozenCoreMemoryAt: current.frozenCoreMemoryAt,
		};
	}
	return { changed: true, frozenCoreMemory: rendered, frozenCoreMemoryAt: nowMs };
}

/** Loads the tenant's current core memory and builds the refresh patch (see `buildFrozenCoreMemoryPatch`). */
export async function refreshFrozenCoreMemory(
	memory: Pick<MemoryService, "getCoreMemory">,
	tenant: Pick<TenantContext, "tenantId" | "userId">,
	current: Pick<ConversationMetaState, "frozenCoreMemory" | "frozenCoreMemoryAt">,
	nowMs: number,
	options?: { readonly limit?: number },
): Promise<FrozenCoreMemoryPatch> {
	const memories = await memory.getCoreMemory(tenant.tenantId, tenant.userId, options?.limit);
	return buildFrozenCoreMemoryPatch(current, memories, nowMs);
}

export function serializeTenantForDoc(tenant: TenantContext): ConversationMetaState["tenant"] {
	return {
		tenantId: tenant.tenantId,
		userId: tenant.userId,
		permissions: [...tenant.permissions],
		workspaceId: tenant.workspaceId ?? null,
		metadata: tenant.metadata !== undefined ? { ...tenant.metadata } : null,
	};
}

export function deserializeTenantFromDoc(doc: Readonly<ConversationMetaState>): TenantContext {
	return {
		tenantId: doc.tenant.tenantId,
		userId: doc.tenant.userId,
		permissions: [...doc.tenant.permissions],
		...(doc.tenant.workspaceId !== null ? { workspaceId: doc.tenant.workspaceId } : {}),
		...(doc.tenant.metadata !== null ? { metadata: { ...doc.tenant.metadata } } : {}),
	};
}

export function listApprovalsFromDoc(doc: Readonly<PendingApprovalsState> | undefined): PendingApproval[] {
	if (!doc) return [];
	return Object.values(doc.items)
		.map((item) => ({ ...item }))
		.sort((a, b) => a.createdAt - b.createdAt);
}
