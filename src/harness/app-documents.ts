import type { JsonObject } from "@earendil-works/pi-durable";
import { defineDoc } from "@earendil-works/pi-durable";
import type { LongTermMemory } from "../contracts/app-capabilities.js";
import type { PendingApproval } from "../contracts/harness-port.js";
import type { TenantContext } from "../contracts/tenant-context.js";

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
 * Sorting and formatting are stable so repeated sessions with identical
 * memory state render byte-identically.
 */
export function formatFrozenCoreMemory(memories: readonly LongTermMemory[]): string {
	if (memories.length === 0) return "";
	const lines = memories.map((m) => {
		const pinTag = m.pinned ? "[pinned] " : "";
		return `- (${m.kind}) ${pinTag}${m.key}: ${m.content}`;
	});
	return [
		"The following user context was loaded at session start (frozen for prompt cache stability).",
		"Use `memory_search` to retrieve additional or recently updated memories on demand.",
		...lines,
	].join("\n");
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
