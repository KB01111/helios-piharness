import { defineSignal } from "@earendil-works/pi-durable";
import { bool, int, obj, opt, str } from "@earendil-works/pi-schema";
import { PendingApprovalsDoc, PendingSideEffectsDoc } from "../harness/app-documents.js";
import type { ToolPackDefinition } from "../harness/registry.js";
import { createTenantScopedTool, defineToolPack } from "./tool-pack.js";

/**
 * Typed Pi Durable signal delivered externally to resume a parked approval tool.
 */
export const ApprovalDecisionSignal = defineSignal({
	name: "app.approval.decision",
	version: 1,
	schema: obj({
		approved: bool(),
		reviewer: opt(str()),
		note: opt(str()),
	}),
});

export interface DomainRecord {
	readonly tenantId: string;
	readonly recordId: string;
	readonly value: string;
	readonly updatedAt: number;
}

export interface DomainStore {
	lookup(tenantId: string, recordId: string, signal?: AbortSignal): Promise<DomainRecord | undefined>;
	mutate(
		tenantId: string,
		recordId: string,
		value: string,
		idempotencyKey: string,
		signal?: AbortSignal,
	): Promise<DomainRecord>;
	getLookupCount(recordId?: string): number;
	getMutationCount(recordId?: string): number;
}

export interface InMemoryDomainStoreHooks {
	/** Optional test hook invoked when lookup starts (e.g. to simulate a mid-call crash). */
	onLookupStart?: (recordId: string, attempt: number, signal?: AbortSignal) => Promise<void> | void;
	/** Optional test hook invoked when mutate starts (e.g. to simulate a mid-call crash). */
	onMutateStart?: (recordId: string, attempt: number, signal?: AbortSignal) => Promise<void> | void;
}

export class InMemoryDomainStore implements DomainStore {
	private readonly records = new Map<string, DomainRecord>();
	private readonly committedKeys = new Map<string, DomainRecord>();
	private readonly lookupCounts = new Map<string, number>();
	private readonly mutationCounts = new Map<string, number>();
	private hooks: InMemoryDomainStoreHooks;

	constructor(initialRecords?: readonly DomainRecord[], hooks?: InMemoryDomainStoreHooks) {
		this.hooks = hooks ?? {};
		if (initialRecords) {
			for (const rec of initialRecords) {
				this.records.set(`${rec.tenantId}:${rec.recordId}`, rec);
			}
		}
	}

	setHooks(hooks: InMemoryDomainStoreHooks): void {
		this.hooks = hooks;
	}

	seed(record: DomainRecord): void {
		this.records.set(`${record.tenantId}:${record.recordId}`, record);
	}

	async lookup(tenantId: string, recordId: string, signal?: AbortSignal): Promise<DomainRecord | undefined> {
		const nextCount = (this.lookupCounts.get(recordId) ?? 0) + 1;
		this.lookupCounts.set(recordId, nextCount);
		await this.hooks.onLookupStart?.(recordId, nextCount, signal);
		if (signal?.aborted) {
			throw new Error("Lookup aborted");
		}
		return this.records.get(`${tenantId}:${recordId}`);
	}

	async mutate(
		tenantId: string,
		recordId: string,
		value: string,
		idempotencyKey: string,
		signal?: AbortSignal,
	): Promise<DomainRecord> {
		const nextCount = (this.mutationCounts.get(recordId) ?? 0) + 1;
		this.mutationCounts.set(recordId, nextCount);
		await this.hooks.onMutateStart?.(recordId, nextCount, signal);
		if (signal?.aborted) {
			throw new Error("Mutation aborted");
		}
		const already = this.committedKeys.get(idempotencyKey);
		if (already) return already;

		const updated: DomainRecord = {
			tenantId,
			recordId,
			value,
			updatedAt: Date.now(),
		};
		this.records.set(`${tenantId}:${recordId}`, updated);
		this.committedKeys.set(idempotencyKey, updated);
		return updated;
	}

	getLookupCount(recordId?: string): number {
		if (recordId !== undefined) return this.lookupCounts.get(recordId) ?? 0;
		let total = 0;
		for (const c of this.lookupCounts.values()) total += c;
		return total;
	}

	getMutationCount(recordId?: string): number {
		if (recordId !== undefined) return this.mutationCounts.get(recordId) ?? 0;
		let total = 0;
		for (const c of this.mutationCounts.values()) total += c;
		return total;
	}
}

const LookupParams = obj({
	recordId: str({ description: "ID of the domain record to look up." }),
});

const MutateParams = obj({
	recordId: str({ description: "ID of the domain record to create or update." }),
	value: str({ description: "New value to write to the record." }),
});

const ApprovalParams = obj({
	action: str({ description: "Short machine-readable action name requiring human approval." }),
	summary: str({ description: "Human-readable explanation of what will happen if approved." }),
	timeoutMs: opt(int({ minimum: 1000, description: "Optional timeout in ms (default 24h)." })),
});

type ApprovalCheckpoint = {
	action: string;
	summary: string;
	signalKey: string;
};

const DEFAULT_APPROVAL_TIMEOUT_MS = 24 * 60 * 60 * 1000;

/**
 * Example domain tool pack demonstrating all three Pi Durable tool patterns:
 * 1. `domain_lookup_record`: `replay: "safe"` idempotent read tool
 * 2. `domain_mutate_record`: two-phase `replay: "unsafe"` side-effect tool
 * 3. `domain_request_approval`: parked `{ sleep: { signals, until } }` approval tool
 */
export function createExampleDomainToolPack(options?: { readonly store?: DomainStore }): {
	readonly pack: ToolPackDefinition;
	readonly store: DomainStore;
} {
	const store = options?.store ?? new InMemoryDomainStore();

	// 1. Replay-safe read tool
	const lookupTool = createTenantScopedTool({
		name: "domain_lookup_record",
		description: "Read a tenant-scoped domain record by ID. Idempotent and safe to replay across crashes.",
		parameters: LookupParams,
		replay: "safe",
		execute: async (args, tenant, _api, context) => {
			const record = await store.lookup(tenant.tenantId, args.recordId, context.abortSignal);
			if (!record) {
				return {
					isError: true,
					output: [{ type: "text", text: `Record "${args.recordId}" not found.` }],
				};
			}
			return {
				output: [{ type: "text", text: JSON.stringify({ recordId: record.recordId, value: record.value }) }],
				details: { recordId: record.recordId },
			};
		},
	});

	// 2. Two-phase side-effect tool (replay: "unsafe")
	const mutateTool = createTenantScopedTool({
		name: "domain_mutate_record",
		description:
			"Stage and commit a tenant-scoped domain record mutation using a two-phase idempotency log. Not automatically replayed on crash.",
		parameters: MutateParams,
		replay: "unsafe",
		execute: async (args, tenant, api, context) => {
			// Phase 1: Durably stage intent + idempotency key in conversation document
			const idempotencyKey = await api.memo("side_effect.key", `se:${api.conversationId}:${api.callId}`, context);
			await api.commit(async (tx) => {
				const doc = await tx.doc(PendingSideEffectsDoc, api.conversationId);
				doc.records[idempotencyKey] = {
					idempotencyKey,
					recordId: args.recordId,
					value: args.value,
					status: "staged",
					updatedAt: api.now(),
				};
			}, context);

			// Phase 2: Execute external mutation and mark committed
			const updated = await store.mutate(
				tenant.tenantId,
				args.recordId,
				args.value,
				idempotencyKey,
				context.abortSignal,
			);

			await api.commit(async (tx) => {
				const doc = await tx.doc(PendingSideEffectsDoc, api.conversationId);
				doc.records[idempotencyKey] = {
					idempotencyKey,
					recordId: updated.recordId,
					value: updated.value,
					status: "committed",
					updatedAt: api.now(),
				};
			}, context);

			return {
				output: [
					{
						type: "text",
						text: `Committed record "${updated.recordId}" (idempotencyKey=${idempotencyKey}) with value="${updated.value}".`,
					},
				],
				details: { recordId: updated.recordId, idempotencyKey, status: "committed" },
			};
		},
	});

	// 3. Parked human-approval tool ({ sleep: { signals, until, checkpoint } })
	const approvalTool = createTenantScopedTool<typeof ApprovalParams, Record<string, string | number | boolean>, ApprovalCheckpoint>({
		name: "domain_request_approval",
		description:
			"Pause execution and wait durably for an external human approval decision signal. Survives restarts without holding a process.",
		parameters: ApprovalParams,
		replay: "safe",
		execute: async (args, _tenant, api, context) => {
			const signalKey =
				api.checkpoint?.signalKey ?? (await api.memo("approval.key", `approval_${api.callId}`, context));

			const state = await api.readSignal(ApprovalDecisionSignal, signalKey, context);

			if (state !== undefined) {
				await api.commit(async (tx) => {
					const doc = await tx.doc(PendingApprovalsDoc, api.conversationId);
					delete doc.items[String(api.taskId)];
				}, context);

				if ("expired" in state) {
					return {
						isError: true,
						output: [{ type: "text", text: `Approval request for "${args.action}" expired without a decision.` }],
						details: { action: args.action, status: "expired", reviewer: "" },
					};
				}

				const decision = state.value;
				const reviewer = decision.reviewer ?? "reviewer";
				const notePart = decision.note ? ` Note: ${decision.note}` : "";
				if (!decision.approved) {
					return {
						isError: true,
						output: [{ type: "text", text: `Approval denied by ${reviewer} for "${args.action}".${notePart}` }],
						details: { action: args.action, status: "denied", reviewer },
					};
				}
				return {
					output: [{ type: "text", text: `Approval granted by ${reviewer} for "${args.action}".${notePart}` }],
					details: { action: args.action, status: "approved", reviewer },
				};
			}

			// Fresh call: persist pending approval and park durably on ApprovalDecisionSignal
			const createdAt = api.now();
			const until = createdAt + (args.timeoutMs ?? DEFAULT_APPROVAL_TIMEOUT_MS);

			await api.commit(async (tx) => {
				const doc = await tx.doc(PendingApprovalsDoc, api.conversationId);
				doc.items[String(api.taskId)] = {
					taskId: String(api.taskId),
					conversationId: String(api.conversationId),
					callId: api.callId,
					toolName: "domain_request_approval",
					action: args.action,
					summary: args.summary,
					signalName: ApprovalDecisionSignal.name,
					signalKey,
					createdAt,
				};
			}, context);

			await api.details(
				{
					kind: "approval_request",
					action: args.action,
					summary: args.summary,
					signalName: ApprovalDecisionSignal.name,
					signalKey,
					createdAt,
				},
				context,
			);

			return {
				sleep: {
					checkpoint: {
						action: args.action,
						summary: args.summary,
						signalKey,
					},
					signals: [ApprovalDecisionSignal.key(signalKey)],
					until,
				},
			};
		},
		cancel: async (_args, _tenant, api, context) => {
			await api.commit(async (tx) => {
				const doc = await tx.doc(PendingApprovalsDoc, api.conversationId);
				delete doc.items[String(api.taskId)];
			}, context);
		},
	});

	const pack = defineToolPack({
		name: "example-domain",
		description: "Example domain tools: replay-safe read, two-phase side-effect, and parked approval.",
		tools: [lookupTool, mutateTool, approvalTool],
	});

	return { pack, store };
}
