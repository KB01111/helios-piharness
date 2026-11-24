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
	"domain:read",
	"domain:write",
	"domain:approve_request",
] as const;

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
