import { randomUUID } from "node:crypto";
import { Hono, type Context } from "hono";
import { streamSSE } from "hono/streaming";
import {
	ForbiddenError,
	HarnessError,
	NotFoundError,
	PayloadTooLargeError,
	RateLimitError,
	TooManyStreamsError,
	UnauthorizedError,
	ValidationError,
} from "../contracts/errors.js";
import type { CreateConversationInput, HarnessPort, SubmitMessageInput } from "../contracts/harness-port.js";
import { ADMIN_PERMISSIONS, hasPermission, type TenantContext } from "../contracts/tenant-context.js";
import type { ImprovementService } from "../learning/self-improve/service.js";
import { createLogger, serializeError, type Logger } from "../logger.js";
import { authFromEnv, type AuthResolver } from "./auth.js";
import { RateLimiter } from "./rate-limit.js";
import {
	optionalBoolean,
	optionalEnum,
	optionalModelSpec,
	optionalPositiveInt,
	optionalString,
	optionalStringArray,
	parseJsonObject,
	requiredString,
	type JsonRecord,
} from "./validation.js";

export interface RateLimitOptions {
	/** Max requests per tenant per window. */
	readonly limit: number;
	readonly windowMs: number;
}

export interface HttpServerOptions {
	/**
	 * Resolves the caller identity. Defaults to `authFromEnv()`: `HELIOS_API_KEYS` bearer tokens and/or the explicit
	 * `HELIOS_AUTH_TRUSTED_HEADERS=true` dev opt-in. With neither configured every route except the public ones is 401.
	 */
	readonly auth?: AuthResolver;
	/** Max request body size in bytes (default `HELIOS_MAX_BODY_BYTES` or 1 MiB). */
	readonly maxBodyBytes?: number;
	/** Per-tenant limiter (default `HELIOS_RATE_LIMIT_PER_MINUTE` or 600/min); `false` disables it. */
	readonly rateLimit?: RateLimitOptions | false;
	/**
	 * Per-remote-address limiter applied only to failed authentications (default `HELIOS_AUTH_FAIL_LIMIT_PER_MINUTE`
	 * or 60/min): once exceeded, further failures answer 429 instead of 401. Keyed by socket address; skipped when
	 * the runtime exposes none. `false` disables it.
	 */
	readonly authFailureLimit?: RateLimitOptions | false;
	/** Max concurrently open SSE streams per tenant (default `HELIOS_MAX_STREAMS_PER_TENANT` or 20). */
	readonly maxStreamsPerTenant?: number;
	/** Max concurrently open SSE streams process-wide (default `HELIOS_MAX_STREAMS` or 1000). */
	readonly maxStreams?: number;
	/** SSE keepalive comment interval in ms (default `HELIOS_SSE_KEEPALIVE_MS` or 15000; 0 disables). */
	readonly keepAliveMs?: number;
	/** Gated self-improvement service; its admin routes (`/improvements`) are only mounted when provided. */
	readonly improvements?: ImprovementService;
	readonly logger?: Logger;
	/** Paths served without authentication (default `["/health"]`). */
	readonly publicPaths?: readonly string[];
	/** Hook to register extra unauthenticated routes (e.g. liveness/readiness probes). List their paths in `publicPaths`. */
	readonly registerPublicRoutes?: (app: Hono<HttpEnv>) => void;
}

export interface HttpEnv {
	Variables: {
		requestId: string;
		tenant: TenantContext;
	};
}

const DEFAULT_MAX_BODY_BYTES = 1024 * 1024;
const DEFAULT_RATE_LIMIT_PER_MINUTE = 600;
const DEFAULT_KEEPALIVE_MS = 15_000;
const DEFAULT_AUTH_FAIL_PER_MINUTE = 60;
const DEFAULT_MAX_STREAMS_PER_TENANT = 20;
const DEFAULT_MAX_STREAMS = 1000;
/** Permission required to start a sleep cycle (it consolidates and rewrites memory). */
const SLEEP_CYCLE_PERMISSION = "memory:write";
const MAX_CONTENT_CHARS = 1_000_000;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9._:-]{1,128}$/;
const NUMERIC_ID_PATTERN = /^\d{1,20}$/;

function envInt(raw: string | undefined, fallback: number, allowZero = false): number {
	if (raw === undefined || raw.trim() === "") return fallback;
	const parsed = Number.parseInt(raw, 10);
	if (!Number.isFinite(parsed) || parsed < 0 || (parsed === 0 && !allowZero)) return fallback;
	return parsed;
}

/** Reads the raw body as text, enforcing `maxBytes` both via content-length and while streaming. */
async function readBodyText(c: Context<HttpEnv>, maxBytes: number): Promise<string> {
	const declared = c.req.header("content-length");
	if (declared !== undefined) {
		const length = Number.parseInt(declared, 10);
		if (Number.isFinite(length) && length > maxBytes) throw new PayloadTooLargeError();
	}
	const body = c.req.raw.body;
	if (!body) return "";
	const reader = body.getReader();
	const chunks: Uint8Array[] = [];
	let total = 0;
	try {
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			total += value.byteLength;
			if (total > maxBytes) {
				await reader.cancel().catch(() => undefined);
				throw new PayloadTooLargeError();
			}
			chunks.push(value);
		}
	} finally {
		reader.releaseLock();
	}
	return Buffer.concat(chunks).toString("utf8");
}

function legacyStatus(err: unknown): { status: number; code: string; message: string } | undefined {
	// Defensive mapping for plain Errors thrown by collaborators that predate typed errors. The status is inferred
	// from the message shape, but the message itself is never echoed: clients get a fixed string per code.
	if (!(err instanceof Error)) return undefined;
	if (/^Invalid [A-Za-z]+ "/.test(err.message)) return { status: 400, code: "invalid_request", message: "Invalid request" };
	if (/^[A-Za-z ]+ "[^"]*" not found$/.test(err.message)) return { status: 404, code: "not_found", message: "Not found" };
	return undefined;
}

function remoteAddress(c: Context<HttpEnv>): string | undefined {
	const incoming = (c.env as { incoming?: { socket?: { remoteAddress?: unknown } } } | undefined)?.incoming;
	const address = incoming?.socket?.remoteAddress;
	return typeof address === "string" && address.length > 0 ? address : undefined;
}

/**
 * Thin Hono HTTP + native SSE transport wrapping ONLY `HarnessPort`.
 * Zero Pi Durable / Pi AI / Chord imports or leakage.
 *
 * Security model: every route except `publicPaths` requires an authenticated TenantContext produced by
 * `options.auth`; client-supplied tenant/permissions in bodies or headers are never trusted (unless the dev-only
 * trusted-headers resolver is explicitly enabled). Conversations, approvals and tasks are tenant-scoped and other
 * tenants' resources surface as 404.
 */
export function createHttpServer(harness: HarnessPort, options: HttpServerOptions = {}): Hono<HttpEnv> {
	const app = new Hono<HttpEnv>();
	const logger = options.logger ?? createLogger({ level: "warn", base: { component: "http" } });
	const auth = options.auth ?? authFromEnv(process.env, logger);
	const maxBodyBytes = options.maxBodyBytes ?? envInt(process.env.HELIOS_MAX_BODY_BYTES, DEFAULT_MAX_BODY_BYTES);
	const keepAliveMs = options.keepAliveMs ?? envInt(process.env.HELIOS_SSE_KEEPALIVE_MS, DEFAULT_KEEPALIVE_MS, true);
	const publicPaths = new Set(options.publicPaths ?? ["/health"]);

	const rateLimitConfig: RateLimitOptions | false =
		options.rateLimit ??
		(envInt(process.env.HELIOS_RATE_LIMIT_PER_MINUTE, DEFAULT_RATE_LIMIT_PER_MINUTE, true) === 0
			? false
			: {
					limit: envInt(process.env.HELIOS_RATE_LIMIT_PER_MINUTE, DEFAULT_RATE_LIMIT_PER_MINUTE),
					windowMs: 60_000,
				});
	const limiter = rateLimitConfig ? new RateLimiter(rateLimitConfig) : undefined;

	const authFailureConfig: RateLimitOptions | false =
		options.authFailureLimit ??
		(envInt(process.env.HELIOS_AUTH_FAIL_LIMIT_PER_MINUTE, DEFAULT_AUTH_FAIL_PER_MINUTE, true) === 0
			? false
			: {
					limit: envInt(process.env.HELIOS_AUTH_FAIL_LIMIT_PER_MINUTE, DEFAULT_AUTH_FAIL_PER_MINUTE),
					windowMs: 60_000,
				});
	const authFailureLimiter = authFailureConfig ? new RateLimiter(authFailureConfig) : undefined;

	// --- SSE stream caps (per tenant + global) -------------------------------------------------------------------
	const maxStreamsPerTenant =
		options.maxStreamsPerTenant ?? envInt(process.env.HELIOS_MAX_STREAMS_PER_TENANT, DEFAULT_MAX_STREAMS_PER_TENANT);
	const maxStreams = options.maxStreams ?? envInt(process.env.HELIOS_MAX_STREAMS, DEFAULT_MAX_STREAMS);
	const streamsByTenant = new Map<string, number>();
	let openStreams = 0;
	/** Reserves a stream slot or throws 429. The returned release function is idempotent. */
	const acquireStreamSlot = (tenantId: string): (() => void) => {
		const perTenant = streamsByTenant.get(tenantId) ?? 0;
		if (openStreams >= maxStreams || perTenant >= maxStreamsPerTenant) throw new TooManyStreamsError();
		openStreams++;
		streamsByTenant.set(tenantId, perTenant + 1);
		let released = false;
		return () => {
			if (released) return;
			released = true;
			openStreams--;
			const remaining = (streamsByTenant.get(tenantId) ?? 1) - 1;
			if (remaining <= 0) streamsByTenant.delete(tenantId);
			else streamsByTenant.set(tenantId, remaining);
		};
	};

	// --- request id -------------------------------------------------------------------------------------------
	app.use("*", async (c, next) => {
		const incoming = c.req.header("x-request-id");
		const requestId = incoming && REQUEST_ID_PATTERN.test(incoming) ? incoming : randomUUID();
		c.set("requestId", requestId);
		c.header("x-request-id", requestId);
		await next();
		// This is a JSON/SSE API: never sniffed, cached by shared caches, or leaking referrers. Set after the handler so
		// responses that build their own headers (streamSSE replaces Cache-Control) cannot drop them.
		c.res.headers.set("x-content-type-options", "nosniff");
		c.res.headers.set("cache-control", "no-store");
		c.res.headers.set("referrer-policy", "no-referrer");
	});

	app.onError((err, c) => {
		const requestId = c.get("requestId") ?? randomUUID();
		const headers: Record<string, string> = { "x-request-id": requestId };
		if (err instanceof RateLimitError) headers["retry-after"] = String(err.retryAfterSeconds);
		if (err instanceof UnauthorizedError) headers["www-authenticate"] = "Bearer";

		// Typed 4xx errors carry intentionally client-facing messages. 5xx and untyped errors never echo their text.
		const typed =
			err instanceof HarnessError
				? err.status < 500
					? { status: err.status, code: err.code, message: err.message }
					: undefined
				: legacyStatus(err);
		if (typed) {
			return c.json({ error: typed.message, code: typed.code, requestId }, typed.status as 400, headers);
		}
		if (c.req.raw.signal.aborted) {
			// The client already went away (e.g. disconnected during wait=true); nothing to report loudly.
			logger.debug("http_request_aborted", { requestId, method: c.req.method, path: c.req.path });
			return c.body(null, 499 as 400, headers);
		}
		logger.error("http_unhandled_error", {
			requestId,
			method: c.req.method,
			path: c.req.path,
			error: serializeError(err),
		});
		return c.json({ error: "Internal server error", code: "internal_error", requestId }, 500, headers);
	});

	app.notFound((c) => c.json({ error: "Not found", code: "not_found", requestId: c.get("requestId") }, 404));

	// --- public routes (before auth) ---------------------------------------------------------------------------
	app.get("/health", (c) => {
		return c.json({ status: "ok", timestamp: Date.now() });
	});
	options.registerPublicRoutes?.(app);

	// --- authentication + rate limiting ------------------------------------------------------------------------
	app.use("*", async (c, next) => {
		if (publicPaths.has(c.req.path)) return next();
		let tenant: TenantContext | null;
		try {
			tenant = await auth(c.req.raw);
		} catch (error) {
			logger.error("http_auth_resolver_failed", { requestId: c.get("requestId"), error: serializeError(error) });
			tenant = null;
		}
		if (!tenant) {
			const address = authFailureLimiter ? remoteAddress(c) : undefined;
			if (authFailureLimiter && address !== undefined) {
				const decision = authFailureLimiter.check(address);
				if (!decision.allowed) throw new RateLimitError(decision.retryAfterSeconds);
			}
			throw new UnauthorizedError();
		}
		c.set("tenant", tenant);

		if (limiter) {
			const decision = limiter.check(tenant.tenantId);
			if (!decision.allowed) throw new RateLimitError(decision.retryAfterSeconds);
		}
		return next();
	});

	// --- helpers -----------------------------------------------------------------------------------------------
	const readJson = async (c: Context<HttpEnv>, opts: { required?: boolean } = {}): Promise<JsonRecord> =>
		parseJsonObject(await readBodyText(c, maxBodyBytes), opts);

	const requirePermission = (tenant: TenantContext, permission: string): void => {
		if (!hasPermission(tenant, permission)) throw new ForbiddenError(`Missing permission "${permission}"`);
	};

	/** Throws NotFoundError unless the conversation exists and belongs to the caller's tenant. */
	const ownedConversationId = async (c: Context<HttpEnv>): Promise<string> => {
		const id = c.req.param("id") ?? "";
		await harness.assertConversationOwner(c.get("tenant"), id);
		return id;
	};

	// --- conversations -----------------------------------------------------------------------------------------
	app.post("/conversations", async (c) => {
		const body = await readJson(c);
		const model = optionalModelSpec(body, "model");
		const instructions = optionalString(body, "instructions", { maxLength: MAX_CONTENT_CHARS });
		const enabledToolPacks = optionalStringArray(body, "enabledToolPacks");
		const maxTurns = optionalPositiveInt(body, "maxTurns", 10_000);
		const mode = optionalEnum(body, "mode", ["execute", "plan"] as const);
		// body.tenant is deliberately ignored: identity and permissions come only from the authenticated caller.
		const created = await harness.createConversation({
			tenant: c.get("tenant"),
			...(model ? { model } : {}),
			...(instructions !== undefined ? { instructions } : {}),
			...(enabledToolPacks ? { enabledToolPacks } : {}),
			...(maxTurns !== undefined ? { maxTurns } : {}),
			...(mode !== undefined ? { mode } : {}),
		});
		return c.json(created, 201);
	});

	app.get("/conversations/:id", async (c) => {
		const id = await ownedConversationId(c);
		const info = await harness.getConversation(id);
		if (!info) throw new NotFoundError(`Conversation "${id}" not found`);
		const transcript = await harness.getTranscript(id);
		return c.json({ conversation: info, transcript });
	});

	app.post("/conversations/:id/fork", async (c) => {
		const id = await ownedConversationId(c);
		const body = await readJson(c, { required: true });
		if (body.tenant !== undefined) {
			throw new ValidationError("tenant cannot be set when forking; the child belongs to the authenticated caller");
		}
		const atEntryId = requiredString(body, "atEntryId", { maxLength: 40 });
		if (!NUMERIC_ID_PATTERN.test(atEntryId)) throw new ValidationError("atEntryId must be a numeric entry id");
		const model = optionalModelSpec(body, "model");
		const instructions = optionalString(body, "instructions", { maxLength: MAX_CONTENT_CHARS });
		// The child belongs to the authenticated caller (not the parent's stored identity/permissions).
		const forked = await harness.forkConversation(id, atEntryId, {
			tenant: c.get("tenant"),
			...(model ? { model } : {}),
			...(instructions !== undefined ? { instructions } : {}),
		});
		return c.json(forked, 201);
	});

	app.post("/conversations/:id/messages", async (c) => {
		const conversationId = await ownedConversationId(c);
		const body = await readJson(c, { required: true });
		const content = requiredString(body, "content", { maxLength: MAX_CONTENT_CHARS });
		const requestId = optionalString(body, "requestId", { maxLength: 200 });
		const whenBusy = optionalEnum(body, "whenBusy", ["steer", "followUp", "reject"] as const);
		const wait = optionalBoolean(body, "wait");

		const submitInput: SubmitMessageInput = {
			conversationId,
			content,
			...(requestId ? { requestId } : {}),
			...(whenBusy ? { whenBusy } : {}),
		};

		if (wait === true) {
			const settled = await harness.submitAndWait(submitInput, c.req.raw.signal);
			return c.json(settled);
		}

		const handle = await harness.submit(submitInput);
		return c.json(handle, 202);
	});

	app.get("/conversations/:id/stream", async (c) => {
		const conversationId = await ownedConversationId(c);

		const queryAfter = c.req.query("afterEntryId");
		if (queryAfter !== undefined && !NUMERIC_ID_PATTERN.test(queryAfter)) {
			throw new ValidationError("afterEntryId must be a numeric entry id");
		}
		const headerAfter = c.req.header("last-event-id");
		const afterEntryId = queryAfter ?? (headerAfter && NUMERIC_ID_PATTERN.test(headerAfter) ? headerAfter : undefined);
		const requestId = c.get("requestId");
		const releaseSlot = acquireStreamSlot(c.get("tenant").tenantId);

		return streamSSE(
			c,
			async (stream) => {
				const abortController = new AbortController();
				let keepAlive: ReturnType<typeof setInterval> | undefined;
				const cleanup = (): void => {
					if (keepAlive) clearInterval(keepAlive);
					keepAlive = undefined;
				};
				stream.onAbort(() => {
					cleanup();
					releaseSlot();
					abortController.abort();
				});
				// Client disconnect also surfaces on the request signal.
				c.req.raw.signal.addEventListener("abort", () => abortController.abort(), { once: true });

				if (keepAliveMs > 0) {
					keepAlive = setInterval(() => {
						if (stream.aborted || stream.closed) return cleanup();
						void stream.write(": ping\n\n").catch(cleanup);
					}, keepAliveMs);
					keepAlive.unref?.();
				}

				try {
					for await (const event of harness.watchConversation(conversationId, abortController.signal, {
						...(afterEntryId !== undefined ? { afterEntryId } : {}),
					})) {
						// SSE ids are always transcript entry ids so Last-Event-ID can be fed straight back as
						// `afterEntryId`. Events without an entry omit `id`, which leaves the client's last id unchanged.
						let id: string | undefined;
						if (event.type === "entry_appended") {
							id = event.entry.id;
						} else if (event.type === "snapshot" && event.entries.length > 0) {
							id = event.entries[event.entries.length - 1]?.id;
						}
						await stream.writeSSE({
							...(id !== undefined ? { id } : {}),
							event: event.type,
							data: JSON.stringify(event),
						});
					}
				} finally {
					cleanup();
					releaseSlot();
				}
			},
			async (err, stream) => {
				logger.error("http_stream_error", { requestId, conversationId, error: serializeError(err) });
				await stream.writeSSE({
					event: "error",
					data: JSON.stringify({ error: "Stream failed", code: "internal_error", requestId }),
				});
			},
		);
	});

	app.post("/conversations/:id/abort", async (c) => {
		await harness.abortConversation(await ownedConversationId(c));
		return c.json({ aborted: true });
	});

	app.post("/conversations/:id/reset", async (c) => {
		const id = await ownedConversationId(c);
		const body = await readJson(c);
		const handoff = optionalString(body, "handoff", { maxLength: MAX_CONTENT_CHARS });
		await harness.resetConversation(id, handoff);
		return c.json({ reset: true });
	});

	app.post("/conversations/:id/compact", async (c) => {
		const id = await ownedConversationId(c);
		const body = await readJson(c);
		const instructions = optionalString(body, "instructions", { maxLength: MAX_CONTENT_CHARS });
		const wait = optionalBoolean(body, "wait");
		if (wait === true) {
			// Forward the request signal so a client disconnect stops waiting on the compaction.
			return c.json(await harness.compactConversation(id, instructions, c.req.raw.signal));
		}
		const { taskId } = await harness.startCompaction(id, instructions);
		return c.json({ taskId }, 202);
	});

	app.post("/conversations/:id/sleep-cycle", async (c) => {
		const id = await ownedConversationId(c);
		requirePermission(c.get("tenant"), SLEEP_CYCLE_PERMISSION);
		const body = await readJson(c);
		const reason = optionalString(body, "reason", { maxLength: 1000 });
		const wait = optionalBoolean(body, "wait");

		// Single-flight is durable and enforced by the harness: a repeat request (from any process, across restarts)
		// receives the live task for this conversation instead of starting a duplicate.
		const { taskId } = await harness.triggerSleepCycle(id, { reason });
		if (wait === true) {
			return c.json(await harness.waitTask(taskId, c.req.raw.signal));
		}
		return c.json({ taskId }, 202);
	});

	app.post("/conversations/:id/mode", async (c) => {
		const id = await ownedConversationId(c);
		const body = await readJson(c, { required: true });
		const mode = optionalEnum(body, "mode", ["execute", "plan"] as const);
		if (!mode) throw new ValidationError('mode must be "execute" or "plan"');
		return c.json(await harness.setConversationMode(id, mode));
	});

	app.post("/conversations/:id/btw", async (c) => {
		const id = await ownedConversationId(c);
		const body = await readJson(c, { required: true });
		const question = requiredString(body, "question", { maxLength: MAX_CONTENT_CHARS });
		return c.json(await harness.askSideQuestion(id, question, c.req.raw.signal));
	});

	// --- approvals / signals -----------------------------------------------------------------------------------
	app.get("/approvals", async (c) => {
		const conversationId = c.req.query("conversationId");
		const approvals = await harness.listPendingApprovals(conversationId, { tenant: c.get("tenant") });
		return c.json({ approvals });
	});

	app.post("/signals", async (c) => {
		const body = await readJson(c, { required: true });
		const taskId = requiredString(body, "taskId", { maxLength: 40 });
		const signalName = requiredString(body, "signalName", { maxLength: 200 });
		const key = requiredString(body, "key", { maxLength: 500 });
		await harness.assertTaskOwner(c.get("tenant"), taskId);
		// The approver recorded in the audit trail is the authenticated caller, never a client-supplied name.
		const payload =
			typeof body.payload === "object" && body.payload !== null && !Array.isArray(body.payload)
				? { ...(body.payload as JsonRecord), reviewer: c.get("tenant").userId }
				: (body.payload ?? null);
		const result = await harness.deliverSignal({ taskId, signalName, key, payload });
		return c.json(result);
	});

	// --- tool packs / usage (process-global => admin permissions) ---------------------------------------------
	app.get("/tool-packs", (c) => {
		return c.json({ toolPacks: harness.listToolPacks() });
	});

	app.post("/tool-packs/:name", async (c) => {
		requirePermission(c.get("tenant"), ADMIN_PERMISSIONS.toolPacks);
		const name = c.req.param("name");
		const body = await readJson(c, { required: true });
		const enabled = optionalBoolean(body, "enabled");
		if (enabled === undefined) throw new ValidationError("enabled boolean is required");
		if (!harness.listToolPacks().some((pack) => pack.name === name)) {
			throw new NotFoundError(`Tool pack "${name}" not found`);
		}
		harness.setToolPackEnabled(name, enabled);
		return c.json({ toolPacks: harness.listToolPacks() });
	});

	app.get("/usage", async (c) => {
		// Usage is aggregated process-wide (all tenants), so it is admin-only.
		requirePermission(c.get("tenant"), ADMIN_PERMISSIONS.usage);
		return c.json(await harness.getUsage());
	});

	app.get("/mcp/servers", (c) => {
		// Upstream error text is operator-level diagnostics; ordinary callers only see connection state.
		const detailed = hasPermission(c.get("tenant"), ADMIN_PERMISSIONS.usage);
		const servers = harness.listMcpServers().map((server) => {
			if (detailed) return server;
			const { lastError: _omitted, ...rest } = server;
			return rest;
		});
		return c.json({ servers });
	});

	app.get("/audit/permissions", (c) => {
		const tenant = c.get("tenant");
		if (!hasPermission(tenant, ADMIN_PERMISSIONS.audit) && !hasPermission(tenant, ADMIN_PERMISSIONS.usage)) {
			throw new ForbiddenError(`Missing permission "${ADMIN_PERMISSIONS.audit}"`);
		}
		const conversationId = c.req.query("conversationId");
		const limitRaw = c.req.query("limit");
		const limit = limitRaw ? Number.parseInt(limitRaw, 10) : undefined;
		const entries = harness.listPermissionAudit(tenant.tenantId, {
			...(conversationId ? { conversationId } : {}),
			...(limit !== undefined && Number.isFinite(limit) && limit > 0 ? { limit } : {}),
		});
		return c.json({ entries });
	});

	// --- gated self-improvement (process-global => admin:improvements; the service re-checks the permission) -------
	const improvements = options.improvements;
	if (improvements) {
		app.get("/improvements", (c) => c.json({ ...improvements.list(c.get("tenant")), running: improvements.isRunning() }));

		app.post("/improvements/run", async (c) => {
			// Long-running (evaluates every candidate against the whole eval set); the loop is single-flight.
			const report = await improvements.run(c.get("tenant"));
			return c.json(report);
		});

		app.get("/improvements/:id", (c) => c.json(improvements.get(c.get("tenant"), c.req.param("id"))));

		for (const action of ["approve", "reject", "rollback"] as const) {
			app.post(`/improvements/:id/${action}`, async (c) => {
				const body = await readJson(c, { required: false });
				const note = optionalString(body ?? {}, "note", { maxLength: 500 });
				return c.json(improvements[action](c.get("tenant"), c.req.param("id"), note));
			});
		}
	}

	return app;
}
