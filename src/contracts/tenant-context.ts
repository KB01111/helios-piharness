/**
 * Tenant, user, and authorization context bound to a conversation.
 *
 * Security invariant: TenantContext is committed to the conversation metadata
 * document at creation time and injected into tool handlers via closures.
 * LLM tool schemas must NEVER accept tenantId or userId as parameters.
 */
export interface TenantContext {
	readonly tenantId: string;
	readonly userId: string;
	readonly permissions: readonly string[];
	readonly workspaceId?: string;
	readonly metadata?: Readonly<Record<string, string | number | boolean | null>>;
}

export const DEFAULT_PERMISSIONS = [
	"memory:read",
	"memory:write",
	"artifacts:read",
	"skills:read",
	"files:read",
	"domain:read",
	"domain:write",
	"domain:approve_request",
	"subagent:spawn",
	"advisor:consult",
	"mcp:read",
	"mcp:write",
] as const;

/** Permissions gating process-global operations. `*` and `admin:*` also satisfy them. */
export const ADMIN_PERMISSIONS = {
	toolPacks: "admin:tool-packs",
	usage: "admin:usage",
	/** Run the self-improvement loop and approve, reject or roll back its prompt changes. */
	improvements: "admin:improvements",
	/** Lets a caller read/act on other users' conversations inside the same tenant. */
	conversations: "admin:conversations",
	/** Lets a caller inspect the permission audit log for their tenant. */
	audit: "admin:audit",
} as const;

export function hasPermission(tenant: TenantContext, required: string): boolean {
	if (tenant.permissions.includes("*")) return true;
	if (tenant.permissions.includes(required)) return true;
	const colonIndex = required.indexOf(":");
	if (colonIndex > 0) {
		const domainWildcard = `${required.slice(0, colonIndex)}:*`;
		if (tenant.permissions.includes(domainWildcard)) return true;
	}
	return false;
}

/**
 * Conversation access rule: same tenant AND (same user OR caller holds `*` / `admin:*` / `admin:conversations`).
 * Tenant is always required: an admin permission never crosses tenants.
 */
export function canAccessConversation(
	caller: Pick<TenantContext, "tenantId" | "userId" | "permissions">,
	owner: Pick<TenantContext, "tenantId" | "userId">,
): boolean {
	if (caller.tenantId !== owner.tenantId) return false;
	if (caller.userId === owner.userId) return true;
	return hasPermission(caller as TenantContext, ADMIN_PERMISSIONS.conversations);
}

export function validateTenantContext(tenant: TenantContext): void {
	if (!tenant.tenantId || tenant.tenantId.trim().length === 0) {
		throw new Error("TenantContext.tenantId must be a non-empty string");
	}
	if (!tenant.userId || tenant.userId.trim().length === 0) {
		throw new Error("TenantContext.userId must be a non-empty string");
	}
	if (!Array.isArray(tenant.permissions)) {
		throw new Error("TenantContext.permissions must be an array");
	}
}
