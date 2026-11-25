import { createHash, timingSafeEqual } from "node:crypto";
import { DEFAULT_PERMISSIONS, type TenantContext } from "../contracts/tenant-context.js";
import { createLogger, type Logger } from "../logger.js";

/**
 * Resolves the authenticated identity for a request. Return `null` when the
 * request carries no valid credentials (the transport answers 401). Resolvers
 * must derive the whole TenantContext from trusted material; client-supplied
 * body fields are never consulted.
 */
export type AuthResolver = (request: Request) => TenantContext | null | Promise<TenantContext | null>;

export interface ApiKeyEntry {
	readonly tenantId: string;
	readonly userId: string;
	readonly permissions?: readonly string[];
	readonly workspaceId?: string;
}

function sha256(value: string): Buffer {
	return createHash("sha256").update(value).digest();
}

function nonEmptyString(value: unknown): value is string {
	return typeof value === "string" && value.trim().length > 0;
}

function parseBearer(request: Request): string | undefined {
	const header = request.headers.get("authorization");
	if (!header) return undefined;
	const match = /^Bearer\s+(.+)$/i.exec(header.trim());
	return match?.[1]?.trim() || undefined;
}

/** Static bearer-token resolver backed by an in-memory token -> identity map. */
export function bearerTokenAuth(keys: Readonly<Record<string, ApiKeyEntry>>): AuthResolver {
	const entries = Object.entries(keys).map(([token, entry]) => {
		if (!nonEmptyString(token)) throw new Error("API key tokens must be non-empty strings");
		if (!entry || !nonEmptyString(entry.tenantId) || !nonEmptyString(entry.userId)) {
			throw new Error("Each API key entry requires non-empty tenantId and userId");
		}
		if (entry.permissions !== undefined && !(Array.isArray(entry.permissions) && entry.permissions.every((p) => typeof p === "string"))) {
			throw new Error("API key entry permissions must be an array of strings");
		}
		const tenant: TenantContext = {
			tenantId: entry.tenantId,
			userId: entry.userId,
			permissions: [...(entry.permissions ?? DEFAULT_PERMISSIONS)],
			...(nonEmptyString(entry.workspaceId) ? { workspaceId: entry.workspaceId } : {}),
		};
		return { digest: sha256(token), tenant };
	});

	return (request) => {
		const presented = parseBearer(request);
		if (!presented) return null;
		const digest = sha256(presented);
		let match: TenantContext | null = null;
		// Compare against every key so timing does not reveal which (if any) matched.
		for (const entry of entries) {
			if (timingSafeEqual(entry.digest, digest)) match = entry.tenant;
		}
		return match;
	};
}

/**
 * DEV ONLY. Trusts x-tenant-id / x-user-id / x-permissions headers verbatim.
 * Anyone who can reach the server can impersonate any tenant, so this must be
 * opted into explicitly (never a default). x-tenant-id is required.
 */
export function trustedHeadersAuth(): AuthResolver {
	const resolver: AuthResolver = (request) => {
		const tenantId = request.headers.get("x-tenant-id")?.trim();
		if (!tenantId) return null;
		const userId = request.headers.get("x-user-id")?.trim() || "anonymous";
		const rawPerms = request.headers.get("x-permissions");
		const permissions = rawPerms
			? rawPerms
					.split(",")
					.map((p) => p.trim())
					.filter(Boolean)
			: [...DEFAULT_PERMISSIONS];
		return { tenantId, userId, permissions };
	};
	TRUSTED_HEADER_RESOLVERS.add(resolver);
	return resolver;
}

/** Resolvers built by `trustedHeadersAuth()`; `chainAuth` never consults them when a credential was presented. */
const TRUSTED_HEADER_RESOLVERS = new WeakSet<AuthResolver>();

/**
 * Tries each resolver in order and returns the first identity found.
 * A request that presents an `Authorization` header is judged by the credential resolvers only: if the credential is
 * invalid the chain answers `null` and never falls through to trusted-headers identity.
 */
export function chainAuth(...resolvers: readonly AuthResolver[]): AuthResolver {
	return async (request) => {
		const credentialPresented = request.headers.has("authorization");
		for (const resolver of resolvers) {
			if (credentialPresented && TRUSTED_HEADER_RESOLVERS.has(resolver)) continue;
			const tenant = await resolver(request);
			if (tenant) return tenant;
		}
		return null;
	};
}

/** Resolver that rejects everything (used when no auth is configured). */
export const denyAllAuth: AuthResolver = () => null;

function parseKeyMap(raw: string): Record<string, ApiKeyEntry> {
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch {
		throw new Error("HELIOS_API_KEYS must be valid JSON");
	}
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error("HELIOS_API_KEYS must be a JSON object mapping token -> { tenantId, userId, permissions? }");
	}
	return parsed as Record<string, ApiKeyEntry>;
}

/**
 * Builds the auth resolver from environment:
 * - `HELIOS_API_KEYS`: JSON map `{ "<token>": { "tenantId": "...", "userId": "...", "permissions": [...] } }`
 * - `HELIOS_AUTH_TRUSTED_HEADERS=true`: explicit dev opt-in to header-based identity.
 * With neither set every request except `/health` receives 401.
 */
export function authFromEnv(
	env: Readonly<Record<string, string | undefined>> = process.env,
	logger?: Logger,
): AuthResolver {
	const resolvers: AuthResolver[] = [];
	const rawKeys = env.HELIOS_API_KEYS?.trim();
	const trusted = env.HELIOS_AUTH_TRUSTED_HEADERS?.trim().toLowerCase();
	const trustedEnabled = trusted === "true" || trusted === "1" || trusted === "yes";
	if (rawKeys && trustedEnabled) {
		// Header identity would let any client that omits a (or sends a bad) token impersonate any tenant.
		const message =
			"HELIOS_AUTH_TRUSTED_HEADERS cannot be enabled while HELIOS_API_KEYS is set: trusted-header identity is " +
			"dev-only and bypasses API key authentication. Unset one of them.";
		(logger ?? createLogger({ level: "error", base: { component: "auth" } })).error("auth_config_invalid", { message });
		throw new Error(message);
	}
	if (rawKeys) resolvers.push(bearerTokenAuth(parseKeyMap(rawKeys)));
	if (trustedEnabled) resolvers.push(trustedHeadersAuth());
	if (resolvers.length === 0) return denyAllAuth;
	return resolvers.length === 1 ? (resolvers[0] as AuthResolver) : chainAuth(...resolvers);
}
