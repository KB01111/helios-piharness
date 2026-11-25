import {
	AgentDoc,
	defineExtension,
	GenerationTask,
	hook,
	ProviderDoc,
	UsageDoc,
	type ConversationId,
	type Extension,
	type Tx,
} from "@earendil-works/pi-durable";
import { ConversationMetaDoc } from "../harness/app-documents.js";
import {
	BudgetGate,
	evaluateBudget,
	hasAnyBudgetLimit,
	sumUsageState,
	usageCostUsd,
	usageTokens,
	type BudgetLimits,
	type BudgetStore,
	type BudgetViolation,
	type UsageRecord,
} from "../runtime/budget.js";

/** A budget block, emitted to `onDiagnostic` and persisted in the `budget_events` table. */
export interface BudgetDiagnostic {
	readonly type: "budget_blocked";
	readonly code: string;
	readonly scope: BudgetViolation["scope"];
	readonly metric: BudgetViolation["metric"];
	readonly limit: number;
	readonly used: number;
	readonly conversationId: string;
	readonly tenantId: string;
	readonly message: string;
	readonly at: number;
}

export interface BudgetExtensionOptions {
	readonly store: BudgetStore;
	/** Shared with `createBudgetGatedModels` (installed around the models handed to the harness). */
	readonly gate: BudgetGate;
	readonly limits: BudgetLimits;
	readonly now?: () => number;
	/**
	 * Dollar cost of one usage record. Default: the cost Pi AI computed from the model catalog
	 * (`usage.cost.total`), which is 0 for models without pricing; supply a price table to meter those.
	 */
	readonly estimateCostUsd?: (usage: UsageRecord) => number;
	readonly onDiagnostic?: (diagnostic: BudgetDiagnostic) => void;
	/** Failures of the budget machinery itself (the check fails open). */
	readonly onError?: (error: unknown) => void;
	/**
	 * Operational warnings (e.g. a request without a provider session id, which cannot be gated). Emitted at most
	 * once per distinct warning. Default: one JSON line on stderr.
	 */
	readonly onWarning?: (message: string) => void;
}

/**
 * Budget enforcement extension.
 *
 * - `beforeRequest`: totals this conversation's `pi.usage` and the tenant's daily spend; when a limit is
 *   reached it parks a veto in the gate keyed by the conversation's provider session id. The gated `Models`
 *   then answers the request locally with a non-retryable error message, so the run ends `unanswered`
 *   (`reason: "model_error"`, `detail: "BUDGET_EXCEEDED ..."`) without calling the provider.
 * - `afterResponse`: adds the response's usage to the tenant's daily ledger.
 *
 * Install after guard-shaping. Limits of `0` are unlimited.
 *
 * Scope: only the main conversation's generations are metered and gated. Fast-model spend (memory
 * consolidation and other `FastModelClient` calls) is NOT counted toward, or blocked by, these limits.
 */
export function createBudgetExtension(options: BudgetExtensionOptions): Extension {
	const { store, gate, limits } = options;
	const now = options.now ?? (() => Date.now());
	const enforcing = hasAnyBudgetLimit(limits);
	const costOf = options.estimateCostUsd ?? ((usage: UsageRecord) => usageCostUsd(usage));
	let warnedMissingSession = false;
	const warn =
		options.onWarning ??
		((message: string): void => {
			process.stderr.write(`${JSON.stringify({ level: "warn", event: "budget_enforcement_skipped", message })}\n`);
		});

	return defineExtension({
		name: "budget",
		hooks: [
			hook(GenerationTask, {
				beforeRequest: async (_request, api, context) => {
					if (!enforcing) return undefined;
					let sessionId: string | undefined;
					try {
						const provider = await api.snapshot(ProviderDoc, api.conversationId, context);
						sessionId = provider?.sessionId;
						if (sessionId === undefined) {
							// The gate is keyed by provider session id; without one this request cannot be vetoed.
							if (!warnedMissingSession) {
								warnedMissingSession = true;
								warn(
									"Budget limits are configured but a request has no provider sessionId, so budget enforcement is skipped for it (usage is still recorded).",
								);
							}
							return undefined;
						}
						gate.clear(sessionId);

						const usage = await api.snapshot(UsageDoc, api.conversationId, context);
						const conversation = sumUsageState(usage, costOf);
						const meta = await api.snapshot(ConversationMetaDoc, api.conversationId, context);
						const tenantId = meta?.tenant.tenantId ?? "default";
						const tenantDaily = store.getTenantDaily(tenantId, now());

						const violation = evaluateBudget({ limits, conversation, tenantDaily });
						if (!violation) return undefined;

						gate.block(sessionId, violation);
						const diagnostic: BudgetDiagnostic = {
							type: "budget_blocked",
							code: violation.code,
							scope: violation.scope,
							metric: violation.metric,
							limit: violation.limit,
							used: violation.used,
							conversationId: String(api.conversationId),
							tenantId,
							message: violation.message,
							at: now(),
						};
						try {
							store.recordEvent({ tenantId, conversationId: diagnostic.conversationId, violation });
						} catch (error) {
							options.onError?.(error);
						}
						options.onDiagnostic?.(diagnostic);
					} catch (error) {
						// Fail open: a broken budget check must not take the whole runtime down.
						if (sessionId !== undefined) gate.clear(sessionId);
						options.onError?.(error);
					}
					return undefined;
				},

				afterResponse: async (message, api, context) => {
					try {
						const tokens = usageTokens(message.usage);
						const costUsd = costOf(message.usage);
						if (!(tokens > 0) && !(costUsd > 0)) return;
						const meta = await api.snapshot(ConversationMetaDoc, api.conversationId, context);
						store.recordUsage(meta?.tenant.tenantId ?? "default", tokens, costUsd, now());
					} catch (error) {
						options.onError?.(error);
					}
				},
			}),
		],
	});
}

/**
 * `HarnessOptions.conversationCreated` callback applying a default `maxTurns` to every new or forked
 * conversation that has none, so no run can loop through tool rounds without bound. An explicit
 * `createConversation({ maxTurns })` still wins because the creating commit applies `agent` afterwards.
 */
export function createDefaultMaxTurnsInitializer(
	defaultMaxTurns: number,
): ((tx: Tx, conversation: { readonly id: ConversationId }) => Promise<void>) | undefined {
	if (!Number.isInteger(defaultMaxTurns) || defaultMaxTurns < 1) return undefined;
	return async (tx, conversation) => {
		const agent = await tx.doc(AgentDoc, conversation.id);
		if (agent.maxTurns === undefined) agent.maxTurns = defaultMaxTurns;
	};
}
