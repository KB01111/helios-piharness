import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { bootstrapHarness, DEFAULT_PERMISSIONS, type HarnessEvent, type HarnessPort } from "../src/index.js";
import { HarnessError } from "../src/contracts/errors.js";
import { authFromEnv, bearerTokenAuth, chainAuth, trustedHeadersAuth } from "../src/transport/auth.js";
import { createHttpServer } from "../src/transport/http-server.js";
import { RateLimiter } from "../src/transport/rate-limit.js";

const KEYS = {
	"tok-a": { tenantId: "tenant-a", userId: "alice", permissions: [...DEFAULT_PERMISSIONS] },
	"tok-b": { tenantId: "tenant-b", userId: "bob", permissions: [...DEFAULT_PERMISSIONS] },
	// Same tenant as tok-a, different user, same (default) permissions.
	"tok-a2": { tenantId: "tenant-a", userId: "carol", permissions: [...DEFAULT_PERMISSIONS] },
	// Same tenant, may act on other users' conversations.
	"tok-a-convadmin": {
		tenantId: "tenant-a",
		userId: "dave",
		permissions: [...DEFAULT_PERMISSIONS, "admin:conversations"],
	},
	// Super-user of ANOTHER tenant: `*` must never cross tenants.
	"tok-b-super": { tenantId: "tenant-b", userId: "eve", permissions: ["*"] },
	// Same user as tok-a but read-only.
	"tok-a-ro": { tenantId: "tenant-a", userId: "alice", permissions: ["memory:read"] },
};

const QUIET = {
	debug() {},
	info() {},
	warn() {},
	error() {},
	child() {
		return QUIET;
	},
	level: "error",
	isEnabled: () => false,
} as any;

function headers(token?: string, extra: Record<string, string> = {}): Record<string, string> {
	return {
		"content-type": "application/json",
		...(token ? { authorization: `Bearer ${token}` } : {}),
		...extra,
	};
}

describe("conversation access is per user, not just per tenant", () => {
	let dir: string;
	let runtime: Awaited<ReturnType<typeof bootstrapHarness>>;
	let app: ReturnType<typeof createHttpServer>;

	beforeAll(async () => {
		dir = fs.mkdtempSync(path.join(os.tmpdir(), "helios-hardening-test-"));
		runtime = await bootstrapHarness({
			config: {
				piDbPath: path.join(dir, "pi.sqlite"),
				appDbPath: path.join(dir, "app.sqlite"),
				useMockModels: true,
				skillsDir: path.join(dir, "skills"),
			},
		});
		app = createHttpServer(runtime.harness, { auth: bearerTokenAuth(KEYS), rateLimit: false, logger: QUIET });
	});

	afterAll(async () => {
		await runtime.close();
		fs.rmSync(dir, { recursive: true, force: true });
	});

	async function createConv(token: string): Promise<string> {
		const res = await app.request("/conversations", { method: "POST", headers: headers(token), body: "{}" });
		expect(res.status).toBe(201);
		return ((await res.json()) as { conversationId: string }).conversationId;
	}

	async function lastEntryId(conversationId: string): Promise<string> {
		runtime.testModel.enqueueMain({ type: "text", text: "reply" });
		const msg = await app.request(`/conversations/${conversationId}/messages`, {
			method: "POST",
			headers: headers("tok-a"),
			body: JSON.stringify({ content: "hello", wait: true }),
		});
		expect(msg.status).toBe(200);
		const transcript = await runtime.harness.getTranscript(conversationId);
		return transcript[transcript.length - 1]!.id;
	}

	it("hides another user's conversation (same tenant, same permissions) on every route", async () => {
		const id = await createConv("tok-a");
		const routes: Array<[string, string, unknown?]> = [
			["GET", `/conversations/${id}`],
			["POST", `/conversations/${id}/fork`, { atEntryId: "1" }],
			["POST", `/conversations/${id}/messages`, { content: "hi" }],
			["GET", `/conversations/${id}/stream`],
			["POST", `/conversations/${id}/abort`, {}],
			["POST", `/conversations/${id}/reset`, {}],
			["POST", `/conversations/${id}/compact`, {}],
			["POST", `/conversations/${id}/sleep-cycle`, {}],
			["GET", `/approvals?conversationId=${id}`],
		];
		for (const [method, url, body] of routes) {
			const res = await app.request(url, {
				method,
				headers: headers("tok-a2"),
				...(body !== undefined ? { body: JSON.stringify(body) } : {}),
			});
			expect(res.status, `carol ${method} ${url}`).toBe(404);
		}
		// `*` in another tenant is still another tenant.
		expect((await app.request(`/conversations/${id}`, { headers: headers("tok-b-super") })).status).toBe(404);
		// Owner and a same-tenant admin:conversations holder get through.
		expect((await app.request(`/conversations/${id}`, { headers: headers("tok-a") })).status).toBe(200);
		expect((await app.request(`/conversations/${id}`, { headers: headers("tok-a-convadmin") })).status).toBe(200);
	});

	it("does not let another user in the same tenant touch the owner's tasks via /signals", async () => {
		const id = await createConv("tok-a");
		const compact = await app.request(`/conversations/${id}/compact`, { method: "POST", headers: headers("tok-a"), body: "{}" });
		const { taskId } = (await compact.json()) as { taskId: string };
		const body = JSON.stringify({ taskId, signalName: "nonexistent.signal", key: "k", payload: null });
		expect((await app.request("/signals", { method: "POST", headers: headers("tok-a2"), body })).status).toBe(404);
		expect((await app.request("/signals", { method: "POST", headers: headers("tok-a"), body })).status).toBe(200);
	});

	it("fork: the child belongs to the CALLER and a lower-privileged user cannot fork at all", async () => {
		const id = await createConv("tok-a");
		const entryId = await lastEntryId(id);

		const denied = await app.request(`/conversations/${id}/fork`, {
			method: "POST",
			headers: headers("tok-a2"),
			body: JSON.stringify({ atEntryId: entryId }),
		});
		expect(denied.status).toBe(404);

		// Same user with fewer permissions: the child carries the caller's (smaller) permission set, not the parent's.
		const narrowed = await app.request(`/conversations/${id}/fork`, {
			method: "POST",
			headers: headers("tok-a-ro"),
			body: JSON.stringify({ atEntryId: entryId }),
		});
		expect(narrowed.status).toBe(201);
		const narrowedChild = (await narrowed.json()) as { conversationId: string; tenant: TenantLike };
		expect(narrowedChild.tenant.permissions).toEqual(["memory:read"]);
		expect((await runtime.harness.getConversation(narrowedChild.conversationId))?.tenant.permissions).toEqual(["memory:read"]);

		// An admin:conversations holder may fork, but then owns the child under its own identity.
		const adminFork = await app.request(`/conversations/${id}/fork`, {
			method: "POST",
			headers: headers("tok-a-convadmin"),
			body: JSON.stringify({ atEntryId: entryId }),
		});
		expect(adminFork.status).toBe(201);
		const adminChild = (await adminFork.json()) as { conversationId: string; tenant: TenantLike };
		expect(adminChild.tenant).toMatchObject({ tenantId: "tenant-a", userId: "dave" });
		expect((await app.request(`/conversations/${adminChild.conversationId}`, { headers: headers("tok-a") })).status).toBe(404);
		expect((await app.request(`/conversations/${adminChild.conversationId}`, { headers: headers("tok-a-convadmin") })).status).toBe(200);
	}, 30_000);

	it("embedders that pass no caller keep the old behaviour: the child inherits the parent's identity", async () => {
		const id = await createConv("tok-a");
		const entryId = await lastEntryId(id);
		const child = await runtime.harness.forkConversation(id, entryId);
		expect(child.tenant).toMatchObject({ tenantId: "tenant-a", userId: "alice" });
	});

	it("scopes the approvals listing to the caller's own conversations unless admin:conversations", async () => {
		await createConv("tok-a");
		const own = await runtime.harness.listPendingApprovals(undefined, {
			tenant: { tenantId: "tenant-a", userId: "carol", permissions: [...DEFAULT_PERMISSIONS] },
		});
		expect(own).toEqual([]);
		await expect(
			runtime.harness.listPendingApprovals("999999", { tenant: { tenantId: "tenant-a", userId: "carol", permissions: [] } }),
		).rejects.toBeInstanceOf(HarnessError);
	});

	it("never returns raw provider error text for an unanswered submission", async () => {
		const id = await createConv("tok-a");
		runtime.testModel.enqueueMain({ type: "error", errorMessage: "SECRET provider exploded at /var/db/helios.sqlite line 42" });
		const res = await app.request(`/conversations/${id}/messages`, {
			method: "POST",
			headers: headers("tok-a"),
			body: JSON.stringify({ content: "boom", wait: true }),
		});
		const text = await res.text();
		expect(res.status).toBe(200);
		expect(text).not.toContain("SECRET");
		expect(text).not.toContain("helios.sqlite");
		expect(JSON.parse(text)).toMatchObject({ status: "unanswered" });
	}, 60_000);
});

interface TenantLike {
	tenantId: string;
	userId: string;
	permissions: string[];
}

/** A harness stub where every conversation is owned by the caller. */
function stubHarness(overrides: Record<string, unknown>): HarnessPort {
	return { assertConversationOwner: async () => undefined, ...overrides } as unknown as HarnessPort;
}

describe("SSE stream caps", () => {
	function streamingHarness(): HarnessPort {
		return stubHarness({
			watchConversation: (_id: string, signal?: AbortSignal) =>
				(async function* (): AsyncGenerator<HarnessEvent> {
					await new Promise<void>((resolve) => {
						if (signal?.aborted) return resolve();
						signal?.addEventListener("abort", () => resolve(), { once: true });
					});
				})(),
		});
	}

	it("429s beyond the per-tenant and global caps and frees slots when a stream is cancelled", async () => {
		const app = createHttpServer(streamingHarness(), {
			auth: bearerTokenAuth(KEYS),
			rateLimit: false,
			logger: QUIET,
			keepAliveMs: 0,
			maxStreamsPerTenant: 2,
			maxStreams: 3,
		});
		const open = (token: string) => app.request("/conversations/1/stream", { headers: headers(token) });

		const a1 = await open("tok-a");
		const a2 = await open("tok-a");
		expect(a1.status).toBe(200);
		expect(a2.status).toBe(200);

		const a3 = await open("tok-a");
		expect(a3.status).toBe(429);
		expect(((await a3.json()) as { code: string }).code).toBe("too_many_streams");
		expect(a3.headers.get("retry-after")).toBeTruthy();

		const b1 = await open("tok-b");
		expect(b1.status).toBe(200);
		// Global cap (3) reached even though tenant-b is under its own cap.
		expect((await open("tok-b")).status).toBe(429);

		await a1.body!.cancel();
		await vi.waitFor(async () => {
			const again = await open("tok-a");
			expect(again.status).toBe(200);
			await again.body!.cancel();
		});

		await a2.body!.cancel();
		await b1.body!.cancel();
	});

	it("releases the slot when the stream ends normally", async () => {
		const finite = stubHarness({
			watchConversation: (id: string) =>
				(async function* (): AsyncGenerator<HarnessEvent> {
					yield { type: "stream_closed", conversationId: id, reason: "ended" };
				})(),
		});
		const app = createHttpServer(finite, {
			auth: bearerTokenAuth(KEYS),
			rateLimit: false,
			logger: QUIET,
			keepAliveMs: 0,
			maxStreamsPerTenant: 1,
		});
		for (let i = 0; i < 3; i++) {
			const res = await app.request("/conversations/1/stream", { headers: headers("tok-a") });
			expect(res.status).toBe(200);
			await res.text();
		}
	});
});

describe("error text never reaches clients", () => {
	it("does not echo legacy plain-Error messages or 5xx typed messages", async () => {
		const legacy = stubHarness({
			getConversation: async () => {
				throw new Error('Invalid conversationId "SECRET-/var/db/x"');
			},
		});
		const legacyApp = createHttpServer(legacy, { auth: bearerTokenAuth(KEYS), rateLimit: false, logger: QUIET });
		const res = await legacyApp.request("/conversations/1", { headers: headers("tok-a") });
		expect(res.status).toBe(400);
		const text = await res.text();
		expect(text).not.toContain("SECRET");
		expect(JSON.parse(text)).toMatchObject({ code: "invalid_request", error: "Invalid request" });

		const upstream = stubHarness({
			getConversation: async () => {
				throw new HarnessError("SECRET upstream detail", 502, "bad_gateway");
			},
		});
		const upstreamApp = createHttpServer(upstream, { auth: bearerTokenAuth(KEYS), rateLimit: false, logger: QUIET });
		const res2 = await upstreamApp.request("/conversations/1", { headers: headers("tok-a") });
		expect(res2.status).toBe(500);
		const text2 = await res2.text();
		expect(text2).not.toContain("SECRET");
		expect(JSON.parse(text2)).toMatchObject({ code: "internal_error" });
	});
});

describe("compact?wait=true honours client disconnect", () => {
	it("passes the request abort signal through to compactConversation", async () => {
		let received: AbortSignal | undefined;
		const harness = stubHarness({
			compactConversation: (_id: string, _instructions: string | undefined, signal?: AbortSignal) => {
				received = signal;
				return new Promise((_resolve, reject) => {
					signal?.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
				});
			},
		});
		const app = createHttpServer(harness, { auth: bearerTokenAuth(KEYS), rateLimit: false, logger: QUIET });
		const controller = new AbortController();
		const pending = app.request("/conversations/1/compact", {
			method: "POST",
			headers: headers("tok-a"),
			body: JSON.stringify({ wait: true }),
			signal: controller.signal,
		});
		await vi.waitFor(() => expect(received).toBeDefined());
		expect(received?.aborted).toBe(false);
		controller.abort();
		await Promise.resolve(pending).catch(() => undefined);
		expect(received?.aborted).toBe(true);
	});
});

describe("approvals listing", () => {
	it("passes the authenticated tenant context so the service can check the owner before reading approvals", async () => {
		let seen: unknown;
		const harness = stubHarness({
			listPendingApprovals: async (_id: string | undefined, options: unknown) => {
				seen = options;
				return [];
			},
		});
		const app = createHttpServer(harness, { auth: bearerTokenAuth(KEYS), rateLimit: false, logger: QUIET });
		const res = await app.request("/approvals", { headers: headers("tok-a2") });
		expect(res.status).toBe(200);
		expect(seen).toMatchObject({ tenant: { tenantId: "tenant-a", userId: "carol" } });
	});
});

describe("POST /sleep-cycle", () => {
	function sleepHarness() {
		const calls: string[] = [];
		const settle: Array<() => void> = [];
		const harness = stubHarness({
			triggerSleepCycle: async (id: string) => {
				calls.push(id);
				return { taskId: String(100 + calls.length) };
			},
			waitTask: (taskId: string) =>
				new Promise((resolve) => {
					settle.push(() => resolve({ taskId, status: "completed" }));
				}),
		});
		return { harness, calls, settle };
	}

	it("requires memory:write", async () => {
		const { harness, calls } = sleepHarness();
		const app = createHttpServer(harness, { auth: bearerTokenAuth(KEYS), rateLimit: false, logger: QUIET });
		const denied = await app.request("/conversations/1/sleep-cycle", { method: "POST", headers: headers("tok-a-ro"), body: "{}" });
		expect(denied.status).toBe(403);
		expect(calls).toHaveLength(0);
		const ok = await app.request("/conversations/1/sleep-cycle", { method: "POST", headers: headers("tok-a"), body: "{}" });
		expect(ok.status).toBe(202);
	});

	it("delegates single-flight to the harness (durable, see learning-wiring) and returns its task id", async () => {
		const { harness, calls, settle } = sleepHarness();
		const app = createHttpServer(harness, { auth: bearerTokenAuth(KEYS), rateLimit: false, logger: QUIET });
		const post = (query = "") =>
			app.request(`/conversations/1/sleep-cycle${query}`, { method: "POST", headers: headers("tok-a"), body: "{}" });

		const first = await post();
		expect(first.status).toBe(202);
		expect(((await first.json()) as { taskId: string }).taskId).toBe("101");
		expect(calls).toEqual(["1"]);

		// `wait: true` settles through waitTask on the task the harness returned.
		const waiting = app.request("/conversations/1/sleep-cycle", {
			method: "POST",
			headers: headers("tok-a"),
			body: JSON.stringify({ wait: true }),
		});
		await vi.waitFor(() => expect(settle.length).toBe(1));
		settle.splice(0).forEach((fn) => fn());
		const settled = await waiting;
		expect(settled.status).toBe(200);
		expect(await settled.json()).toMatchObject({ taskId: "102", status: "completed" });
	});
});

describe("auth hardening", () => {
	it("chainAuth never falls through to trusted headers when an Authorization header is present", async () => {
		const chain = chainAuth(bearerTokenAuth(KEYS), trustedHeadersAuth());
		const spoof = { "x-tenant-id": "tenant-a", "x-permissions": "*" };
		expect(await chain(new Request("http://h/", { headers: { authorization: "Bearer wrong", ...spoof } }))).toBeNull();
		expect(await chain(new Request("http://h/", { headers: { authorization: "Basic abc", ...spoof } }))).toBeNull();
		// Without a credential the dev opt-in still works, and a valid token still wins.
		expect(await chain(new Request("http://h/", { headers: spoof }))).toMatchObject({ tenantId: "tenant-a" });
		expect(await chain(new Request("http://h/", { headers: { authorization: "Bearer tok-b" } }))).toMatchObject({ tenantId: "tenant-b" });
	});

	it("refuses to enable trusted headers together with API keys", () => {
		expect(() =>
			authFromEnv({
				HELIOS_API_KEYS: JSON.stringify({ t: { tenantId: "x", userId: "y" } }),
				HELIOS_AUTH_TRUSTED_HEADERS: "true",
			}, QUIET),
		).toThrow(/TRUSTED_HEADERS/);
	});

	it("answers 429 instead of 401 once a remote address keeps failing authentication", async () => {
		const app = createHttpServer(stubHarness({}), {
			auth: bearerTokenAuth(KEYS),
			rateLimit: false,
			authFailureLimit: { limit: 2, windowMs: 60_000 },
			logger: QUIET,
		});
		const attempt = (token: string, address?: string) =>
			app.request(
				"/usage",
				{ headers: headers(token) },
				address ? { incoming: { socket: { remoteAddress: address } } } : undefined,
			);
		expect((await attempt("bad", "10.0.0.1")).status).toBe(401);
		expect((await attempt("bad", "10.0.0.1")).status).toBe(401);
		const limited = await attempt("bad", "10.0.0.1");
		expect(limited.status).toBe(429);
		expect(limited.headers.get("retry-after")).toBeTruthy();
		// Other addresses and requests without a socket address are unaffected.
		expect((await attempt("bad", "10.0.0.2")).status).toBe(401);
		for (let i = 0; i < 5; i++) expect((await attempt("bad")).status).toBe(401);
	});
});

describe("RateLimiter bucket cap", () => {
	it("evicts the oldest keys beyond maxBuckets", () => {
		let now = 0;
		const limiter = new RateLimiter({ limit: 1, windowMs: 60_000, maxBuckets: 3, now: () => now });
		for (const key of ["a", "b", "c", "d", "e"]) {
			now += 1;
			limiter.check(key);
		}
		expect(limiter.size).toBe(3);
		// "a" was evicted, so it starts from a fresh window (allowed again); "e" is still counted.
		expect(limiter.check("a").allowed).toBe(true);
		expect(limiter.check("e").allowed).toBe(false);
	});
});
