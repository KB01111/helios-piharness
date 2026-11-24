/**
 * Cost / token budget primitives (Pi-free apart from the gated-Models wrapper).
 *
 * - `evaluateBudget` decides whether the next generation may start.
 * - `BudgetStore` keeps per-tenant daily spend in the app SQLite DB (own tables, created lazily).
 * - Scope: only the MAIN conversation's provider calls are metered and gated. Fast-model spend (memory
 *   consolidation, other `FastModelClient.completeJson` calls) is neither counted toward these limits nor
 *   blocked by them; bound it separately (fast-model timeouts/retries, scheduler limits).
 * - `BudgetGate` + `createBudgetGatedModels` let the budget extension veto a provider call cleanly:
 *   the vetoed call yields an error assistant message, which Pi Durable settles as an `unanswered`
 *   submission with reason `model_error` and the budget message as `detail` (never retried, never a crash).
 */

import type { DatabaseSync } from "node:sqlite";
import {
	createAssistantMessageEventStream,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Usage,
} from "@earendil-works/pi-ai";
import type { Models } from "@earendil-works/pi-ai/models";

// ─── Limits ──────────────────────────────────────────────────────────────────

/** `0` (or any non-positive value) means unlimited. */
export interface BudgetLimits {
	readonly maxConversationTokens: number;
	readonly maxConversationCostUsd: number;
	readonly maxTenantDailyTokens: number;
	readonly maxTenantDailyCostUsd: number;
}

export const UNLIMITED_BUDGET: BudgetLimits = {
	maxConversationTokens: 0,
	maxConversationCostUsd: 0,
	maxTenantDailyTokens: 0,
	maxTenantDailyCostUsd: 0,
};

export function budgetLimitsFromConfig(config: Partial<BudgetLimits>): BudgetLimits {
	return {
		maxConversationTokens: config.maxConversationTokens ?? 0,
		maxConversationCostUsd: config.maxConversationCostUsd ?? 0,
		maxTenantDailyTokens: config.maxTenantDailyTokens ?? 0,
		maxTenantDailyCostUsd: config.maxTenantDailyCostUsd ?? 0,
	};
}

export function hasAnyBudgetLimit(limits: BudgetLimits): boolean {
	return (
		limits.maxConversationTokens > 0 ||
		limits.maxConversationCostUsd > 0 ||
		limits.maxTenantDailyTokens > 0 ||
		limits.maxTenantDailyCostUsd > 0
	);
}

// ─── Evaluation ──────────────────────────────────────────────────────────────

export type BudgetScope = "conversation" | "tenant_daily";
export type BudgetMetric = "tokens" | "cost_usd";

export interface SpendTotals {
	readonly tokens: number;
	readonly costUsd: number;
}

export interface BudgetViolation {
	readonly scope: BudgetScope;
	readonly metric: BudgetMetric;
	/** Stable machine-readable code, e.g. `conversation_tokens`, `tenant_daily_cost_usd`. */
	readonly code: string;
	readonly limit: number;
	readonly used: number;
	/** Client-facing text, also used as the model-error detail of the blocked submission. */
	readonly message: string;
}

const SCOPE_LABEL: Record<BudgetScope, string> = {
	conversation: "this conversation",
	tenant_daily: "this tenant today (UTC)",
};

const ENV_HINT: Record<string, string> = {
	conversation_tokens: "MAX_CONVERSATION_TOKENS",
	conversation_cost_usd: "MAX_CONVERSATION_COST_USD",
	tenant_daily_tokens: "MAX_TENANT_DAILY_TOKENS",
	tenant_daily_cost_usd: "MAX_TENANT_DAILY_COST_USD",
};

function formatAmount(metric: BudgetMetric, value: number): string {
	return metric === "tokens" ? `${Math.round(value)} tokens` : `$${value.toFixed(4)}`;
}

export function buildViolation(scope: BudgetScope, metric: BudgetMetric, limit: number, used: number): BudgetViolation {
	const code = `${scope}_${metric}`;
	// The phrase "out of budget" keeps pi-ai's retry classifier from retrying the (deterministic) block.
	const message =
		`BUDGET_EXCEEDED ${code}: out of budget for ${SCOPE_LABEL[scope]} ` +
		`(used ${formatAmount(metric, used)} of ${formatAmount(metric, limit)} allowed). ` +
		`The request was not sent to the model; raise ${ENV_HINT[code] ?? "the budget"} or ${
			scope === "conversation" ? "start a new conversation" : "retry after the daily window resets"
		}.`;
	return { scope, metric, code, limit, used, message };
}

export interface BudgetEvaluationInput {
	readonly limits: BudgetLimits;
	readonly conversation: SpendTotals;
	readonly tenantDaily?: SpendTotals;
}

/**
 * First exhausted limit, or undefined. A limit is exhausted once usage has *reached* it, so the generation
 * that crosses the line completes and the next one is blocked (a request cannot be cut off mid-flight).
 */
export function evaluateBudget(input: BudgetEvaluationInput): BudgetViolation | undefined {
	const { limits, conversation, tenantDaily } = input;
	if (limits.maxConversationTokens > 0 && conversation.tokens >= limits.maxConversationTokens) {
		return buildViolation("conversation", "tokens", limits.maxConversationTokens, conversation.tokens);
	}
	if (limits.maxConversationCostUsd > 0 && conversation.costUsd >= limits.maxConversationCostUsd) {
		return buildViolation("conversation", "cost_usd", limits.maxConversationCostUsd, conversation.costUsd);
	}
	if (tenantDaily) {
		if (limits.maxTenantDailyTokens > 0 && tenantDaily.tokens >= limits.maxTenantDailyTokens) {
			return buildViolation("tenant_daily", "tokens", limits.maxTenantDailyTokens, tenantDaily.tokens);
		}
		if (limits.maxTenantDailyCostUsd > 0 && tenantDaily.costUsd >= limits.maxTenantDailyCostUsd) {
			return buildViolation("tenant_daily", "cost_usd", limits.maxTenantDailyCostUsd, tenantDaily.costUsd);
		}
	}
	return undefined;
}

/** The subset of a Pi AI `Usage` the budget reads (so callers need no Pi types). */
export type UsageRecord = Partial<Omit<Usage, "cost">> & { readonly cost?: { readonly total?: number } };
type UsageLike = UsageRecord;

export function usageTokens(usage: UsageLike | undefined): number {
	if (!usage) return 0;
	if (typeof usage.totalTokens === "number" && Number.isFinite(usage.totalTokens)) return usage.totalTokens;
	return (usage.input ?? 0) + (usage.output ?? 0) + (usage.cacheRead ?? 0) + (usage.cacheWrite ?? 0);
}

export function usageCostUsd(usage: UsageLike | undefined): number {
	const total = usage?.cost?.total;
	return typeof total === "number" && Number.isFinite(total) ? total : 0;
}

/** Sums a conversation's `pi.usage` ledger (model buckets + tool buckets). */
export function sumUsageState(
	state: { readonly models?: Readonly<Record<string, unknown>>; readonly tools?: Readonly<Record<string, unknown>> } | undefined,
	estimateCostUsd: (usage: UsageLike) => number = usageCostUsd,
): SpendTotals {
	let tokens = 0;
	let costUsd = 0;
	for (const bucket of [state?.models, state?.tools]) {
		if (!bucket) continue;
		for (const usage of Object.values(bucket) as UsageLike[]) {
			tokens += usageTokens(usage);
			costUsd += estimateCostUsd(usage);
		}
	}
	return { tokens, costUsd };
}

// ─── Persistence ─────────────────────────────────────────────────────────────

export interface BudgetEventRecord {
	readonly id: number;
	readonly createdAt: number;
	readonly tenantId: string;
	readonly conversationId: string;
	readonly code: string;
	readonly scope: BudgetScope;
	readonly metric: BudgetMetric;
	readonly limit: number;
	readonly used: number;
	readonly message: string;
}

/** UTC calendar day key, `YYYY-MM-DD`. */
export function utcDayKey(atMs: number): string {
	return new Date(atMs).toISOString().slice(0, 10);
}

/**
 * Per-tenant daily spend ledger + budget-block event log. Owns two tables in the app database and creates
 * them lazily, so no shared migration is needed.
 */
export class BudgetStore {
	private readonly db: Pick<DatabaseSync, "exec" | "prepare">;
	private readonly now: () => number;

	constructor(db: Pick<DatabaseSync, "exec" | "prepare">, options: { readonly now?: () => number } = {}) {
		this.db = db;
		this.now = options.now ?? (() => Date.now());
		this.db.exec(`
			CREATE TABLE IF NOT EXISTS budget_tenant_daily (
				tenant_id TEXT NOT NULL,
				day TEXT NOT NULL,
				tokens INTEGER NOT NULL DEFAULT 0,
				cost_usd REAL NOT NULL DEFAULT 0,
				updated_at INTEGER NOT NULL,
				PRIMARY KEY (tenant_id, day)
			);
			CREATE TABLE IF NOT EXISTS budget_events (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				created_at INTEGER NOT NULL,
				tenant_id TEXT NOT NULL,
				conversation_id TEXT NOT NULL,
				code TEXT NOT NULL,
				scope TEXT NOT NULL,
				metric TEXT NOT NULL,
				limit_value REAL NOT NULL,
				used_value REAL NOT NULL,
				message TEXT NOT NULL
			);
			CREATE INDEX IF NOT EXISTS idx_budget_events_created ON budget_events (created_at);
		`);
	}

	recordUsage(tenantId: string, tokens: number, costUsd: number, atMs: number = this.now()): void {
		if (!(tokens > 0) && !(costUsd > 0)) return;
		this.db
			.prepare(
				`INSERT INTO budget_tenant_daily (tenant_id, day, tokens, cost_usd, updated_at)
				 VALUES (?, ?, ?, ?, ?)
				 ON CONFLICT (tenant_id, day) DO UPDATE SET
					tokens = tokens + excluded.tokens,
					cost_usd = cost_usd + excluded.cost_usd,
					updated_at = excluded.updated_at`,
			)
			.run(tenantId, utcDayKey(atMs), Math.round(tokens), costUsd, atMs);
	}

	getTenantDaily(tenantId: string, atMs: number = this.now()): SpendTotals {
		const row = this.db
			.prepare("SELECT tokens, cost_usd FROM budget_tenant_daily WHERE tenant_id = ? AND day = ?")
			.get(tenantId, utcDayKey(atMs)) as { tokens: number; cost_usd: number } | undefined;
		return { tokens: row?.tokens ?? 0, costUsd: row?.cost_usd ?? 0 };
	}

	recordEvent(input: {
		readonly tenantId: string;
		readonly conversationId: string;
		readonly violation: BudgetViolation;
	}): void {
		const { violation } = input;
		this.db
			.prepare(
				`INSERT INTO budget_events
					(created_at, tenant_id, conversation_id, code, scope, metric, limit_value, used_value, message)
				 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			)
			.run(
				this.now(),
				input.tenantId,
				input.conversationId,
				violation.code,
				violation.scope,
				violation.metric,
				violation.limit,
				violation.used,
				violation.message,
			);
	}

	listEvents(limit = 50): BudgetEventRecord[] {
		const rows = this.db
			.prepare("SELECT * FROM budget_events ORDER BY id DESC LIMIT ?")
			.all(Math.max(1, Math.floor(limit))) as Array<Record<string, unknown>>;
		return rows.map((row) => ({
			id: Number(row.id),
			createdAt: Number(row.created_at),
			tenantId: String(row.tenant_id),
			conversationId: String(row.conversation_id),
			code: String(row.code),
			scope: row.scope as BudgetScope,
			metric: row.metric as BudgetMetric,
			limit: Number(row.limit_value),
			used: Number(row.used_value),
			message: String(row.message),
		}));
	}
}

// ─── Gate: veto a provider call ──────────────────────────────────────────────

/**
 * Hand-off between the budget extension (which knows the conversation) and the gated `Models`
 * (which only sees the provider session id of a request). One entry is consumed per vetoed request.
 */
export class BudgetGate {
	private readonly blocked = new Map<string, BudgetViolation>();

	block(sessionId: string, violation: BudgetViolation): void {
		this.blocked.set(sessionId, violation);
	}

	clear(sessionId: string): void {
		this.blocked.delete(sessionId);
	}

	/** Returns and removes the pending veto for `sessionId`. */
	consume(sessionId: string | undefined): BudgetViolation | undefined {
		if (sessionId === undefined) return undefined;
		const violation = this.blocked.get(sessionId);
		if (violation) this.blocked.delete(sessionId);
		return violation;
	}

	get size(): number {
		return this.blocked.size;
	}
}

/** Fresh object per blocked message: consumers may mutate usage, which must never leak across messages. */
function zeroUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function blockedStream(
	model: { readonly api: AssistantMessage["api"]; readonly provider: string; readonly id: string },
	violation: BudgetViolation,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	const error: AssistantMessage = {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: zeroUsage(),
		stopReason: "error",
		errorMessage: violation.message,
		timestamp: Date.now(),
	};
	stream.push({ type: "error", reason: "error", error });
	stream.end();
	return stream;
}

/**
 * Wraps `Models` so a request whose provider session has a pending budget veto is answered locally with an
 * error message instead of reaching the provider. All other members delegate to `models` untouched.
 */
export function createBudgetGatedModels(models: Models, gate: BudgetGate): Models {
	type StreamArgs = Parameters<Models["streamSimple"]>;
	const gatedStream = (original: Models["streamSimple"] | Models["stream"]) => {
		return (...args: StreamArgs): AssistantMessageEventStream => {
			const [model, , streamOptions] = args;
			const sessionId = (streamOptions as { sessionId?: string } | undefined)?.sessionId;
			const violation = gate.consume(sessionId);
			if (violation) return blockedStream(model, violation);
			return (original as (...a: StreamArgs) => AssistantMessageEventStream)(...args);
		};
	};

	return new Proxy(models, {
		get(target, property) {
			if (property === "streamSimple") return gatedStream(target.streamSimple.bind(target));
			if (property === "stream") return gatedStream(target.stream.bind(target) as Models["stream"]);
			if (property === "completeSimple") {
				const stream = gatedStream(target.streamSimple.bind(target));
				return (...args: StreamArgs): Promise<AssistantMessage> => stream(...args).result();
			}
			if (property === "complete") {
				const stream = gatedStream(target.stream.bind(target) as Models["stream"]);
				return (...args: StreamArgs): Promise<AssistantMessage> => stream(...args).result();
			}
			const value = Reflect.get(target, property, target) as unknown;
			return typeof value === "function" ? (value as (...a: unknown[]) => unknown).bind(target) : value;
		},
	});
}
