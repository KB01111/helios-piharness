import type { DatabaseSync } from "node:sqlite";
import type { PendingApproval } from "../contracts/harness-port.js";

/**
 * Three-tier deterministic tool permission policy (`allow` | `ask` | `deny`)
 * plus SQLite-backed permission audit log and pending tool-gate approval store.
 *
 * Inspired by `@gotgenes/pi-permission-system` and `pi-verdict`:
 * - Rules are evaluated deterministically before a tool executes (never left to prompts).
 * - `ask` durably parks the ToolTask on a typed signal until a human approves or denies it.
 * - Every policy evaluation and human decision is written to `permission_audit_log`.
 */

export type PermissionTier = "allow" | "ask" | "deny";

export interface PermissionRule {
	/** Exact tool name, prefix wildcard (e.g. `"mcp_github__*"`), or `"*"`. */
	readonly tool: string;
	readonly tier: PermissionTier;
	readonly reason?: string;
	/** Optional timeout in ms when `tier === "ask"`. Defaults to policy `approvalTimeoutMs` (24h). */
	readonly approvalTimeoutMs?: number;
}

export interface PermissionPolicyConfig {
	/** Fallback tier when no rule matches. Default: `"allow"` (RBAC still applies first). */
	readonly defaultTier?: PermissionTier;
	/** Ordered rules; the first matching rule wins. */
	readonly rules?: readonly PermissionRule[];
	/** Default approval timeout for `"ask"` gates in milliseconds. Default: 24h. */
	readonly approvalTimeoutMs?: number;
}

export interface PermissionDecision {
	readonly tier: PermissionTier;
	readonly matchedRule?: string;
	readonly reason?: string;
	readonly approvalTimeoutMs: number;
}

export interface PermissionAuditEntry {
	readonly id: number;
	readonly tenantId: string;
	readonly userId: string;
	readonly conversationId: string;
	readonly taskId: string;
	readonly callId: string;
	readonly toolName: string;
	readonly tier: PermissionTier;
	readonly decision: "allowed" | "blocked" | "approval_requested" | "approved" | "denied" | "expired";
	readonly reason?: string;
	readonly reviewer?: string;
	readonly createdAt: number;
}

export const DEFAULT_TOOL_GATE_APPROVAL_TIMEOUT_MS = 24 * 60 * 60 * 1000;

export function matchesToolPattern(pattern: string, toolName: string): boolean {
	if (pattern === "*") return true;
	if (!pattern.includes("*")) return pattern === toolName;
	if (pattern.endsWith("*")) {
		const prefix = pattern.slice(0, -1);
		if (!prefix.includes("*")) return toolName.startsWith(prefix);
	}
	// Bounded glob matching without catastrophic backtracking
	const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^\\s]{0,128}");
	return new RegExp(`^${escaped}$`).test(toolName);
}

export function evaluatePermissionPolicy(
	toolName: string,
	policy?: PermissionPolicyConfig,
): PermissionDecision {
	const defaultTimeout = policy?.approvalTimeoutMs ?? DEFAULT_TOOL_GATE_APPROVAL_TIMEOUT_MS;
	if (policy?.rules) {
		for (const rule of policy.rules) {
			if (matchesToolPattern(rule.tool, toolName)) {
				return {
					tier: rule.tier,
					matchedRule: rule.tool,
					...(rule.reason !== undefined ? { reason: rule.reason } : {}),
					approvalTimeoutMs: rule.approvalTimeoutMs ?? defaultTimeout,
				};
			}
		}
	}
	return {
		tier: policy?.defaultTier ?? "allow",
		approvalTimeoutMs: defaultTimeout,
	};
}

/** Parses comma-separated tool names / globs (e.g. `TOOL_APPROVAL_REQUIRED=domain_mutate_record,mcp_prod__*`). */
export function buildPermissionPolicyFromEnv(options: {
	readonly approvalRequiredTools?: readonly string[];
	readonly deniedTools?: readonly string[];
	readonly approvalTimeoutMs?: number;
}): PermissionPolicyConfig {
	const rules: PermissionRule[] = [];
	for (const tool of options.deniedTools ?? []) {
		const trimmed = tool.trim();
		if (trimmed) {
			rules.push({ tool: trimmed, tier: "deny", reason: `Tool "${trimmed}" is denied by policy.` });
		}
	}
	for (const tool of options.approvalRequiredTools ?? []) {
		const trimmed = tool.trim();
		if (trimmed) {
			rules.push({
				tool: trimmed,
				tier: "ask",
				reason: `Tool "${trimmed}" requires explicit human approval before execution.`,
			});
		}
	}
	return {
		defaultTier: "allow",
		rules,
		...(options.approvalTimeoutMs !== undefined ? { approvalTimeoutMs: options.approvalTimeoutMs } : {}),
	};
}

/**
 * SQLite-backed store for permission audit records and pending `beforeTool` gate approvals.
 * Shares `AppDatabase`'s `DatabaseSync` connection (tables created in migration v4).
 */
export class PermissionAuditStore {
	private readonly db: DatabaseSync;
	private readonly now: () => number;

	constructor(db: DatabaseSync, options?: { readonly now?: () => number }) {
		this.db = db;
		this.now = options?.now ?? (() => Date.now());
	}

	recordAudit(input: {
		readonly tenantId: string;
		readonly userId: string;
		readonly conversationId: string;
		readonly taskId: string;
		readonly callId: string;
		readonly toolName: string;
		readonly tier: PermissionTier;
		readonly decision: PermissionAuditEntry["decision"];
		readonly reason?: string;
		readonly reviewer?: string;
		readonly createdAt?: number;
	}): void {
		this.db
			.prepare(
				`INSERT INTO permission_audit_log
					(tenant_id, user_id, conversation_id, task_id, call_id, tool_name, tier, decision, reason, reviewer, created_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			)
			.run(
				input.tenantId,
				input.userId,
				input.conversationId,
				input.taskId,
				input.callId,
				input.toolName,
				input.tier,
				input.decision,
				input.reason ?? null,
				input.reviewer ?? null,
				input.createdAt ?? this.now(),
			);
	}

	listAudit(
		tenantId: string,
		options?: { readonly conversationId?: string; readonly limit?: number },
	): PermissionAuditEntry[] {
		const limit = Math.max(1, Math.min(500, options?.limit ?? 100));
		const rows = options?.conversationId
			? (this.db
					.prepare(
						`SELECT id, tenant_id, user_id, conversation_id, task_id, call_id, tool_name, tier, decision, reason, reviewer, created_at
						 FROM permission_audit_log
						 WHERE tenant_id = ? AND conversation_id = ?
						 ORDER BY id DESC LIMIT ?`,
					)
					.all(tenantId, options.conversationId, limit) as Array<Record<string, unknown>>)
			: (this.db
					.prepare(
						`SELECT id, tenant_id, user_id, conversation_id, task_id, call_id, tool_name, tier, decision, reason, reviewer, created_at
						 FROM permission_audit_log
						 WHERE tenant_id = ?
						 ORDER BY id DESC LIMIT ?`,
					)
					.all(tenantId, limit) as Array<Record<string, unknown>>);

		return rows.map((row) => ({
			id: Number(row.id),
			tenantId: String(row.tenant_id),
			userId: String(row.user_id),
			conversationId: String(row.conversation_id),
			taskId: String(row.task_id),
			callId: String(row.call_id),
			toolName: String(row.tool_name),
			tier: String(row.tier) as PermissionTier,
			decision: String(row.decision) as PermissionAuditEntry["decision"],
			...(typeof row.reason === "string" ? { reason: row.reason } : {}),
			...(typeof row.reviewer === "string" ? { reviewer: row.reviewer } : {}),
			createdAt: Number(row.created_at),
		}));
	}

	upsertPendingGateApproval(input: PendingApproval & { readonly tenantId: string; readonly userId: string }): void {
		this.db
			.prepare(
				`INSERT INTO pending_tool_gate_approvals
					(task_id, tenant_id, user_id, conversation_id, call_id, tool_name, action, summary, signal_name, signal_key, created_at)
				 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				 ON CONFLICT(task_id) DO UPDATE SET
					summary = excluded.summary,
					signal_name = excluded.signal_name,
					signal_key = excluded.signal_key`,
			)
			.run(
				input.taskId,
				input.tenantId,
				input.userId,
				input.conversationId,
				input.callId,
				input.toolName,
				input.action,
				input.summary,
				input.signalName,
				input.signalKey,
				input.createdAt,
			);
	}

	deletePendingGateApproval(taskId: string): void {
		this.db.prepare("DELETE FROM pending_tool_gate_approvals WHERE task_id = ?").run(taskId);
	}

	listPendingGateApprovals(filter?: {
		readonly conversationId?: string;
		readonly tenantId?: string;
		readonly userId?: string;
		readonly limit?: number;
	}): PendingApproval[] {
		const clauses: string[] = [];
		const params: Array<string | number> = [];
		if (filter?.conversationId !== undefined) {
			clauses.push("conversation_id = ?");
			params.push(filter.conversationId);
		}
		if (filter?.tenantId !== undefined) {
			clauses.push("tenant_id = ?");
			params.push(filter.tenantId);
		}
		if (filter?.userId !== undefined) {
			clauses.push("user_id = ?");
			params.push(filter.userId);
		}
		const where = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
		const limit = Math.max(1, filter?.limit ?? 500);
		params.push(limit);

		const rows = this.db
			.prepare(
				`SELECT task_id, conversation_id, call_id, tool_name, action, summary, signal_name, signal_key, created_at
				 FROM pending_tool_gate_approvals
				 ${where}
				 ORDER BY created_at ASC
				 LIMIT ?`,
			)
			.all(...params) as Array<Record<string, unknown>>;

		return rows.map((row) => ({
			taskId: String(row.task_id),
			conversationId: String(row.conversation_id),
			callId: String(row.call_id),
			toolName: String(row.tool_name),
			action: String(row.action),
			summary: String(row.summary),
			signalName: String(row.signal_name),
			signalKey: String(row.signal_key),
			createdAt: Number(row.created_at),
		}));
	}
}
