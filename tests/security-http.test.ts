import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { bootstrapHarness, DEFAULT_PERMISSIONS, type HarnessEvent, type HarnessPort } from "../src/index.js";
import { NotFoundError } from "../src/contracts/errors.js";
import type { TenantContext } from "../src/contracts/tenant-context.js";
import { authFromEnv, bearerTokenAuth, trustedHeadersAuth } from "../src/transport/auth.js";
import { createHttpServer } from "../src/transport/http-server.js";

const KEYS = {
	"tok-a": { tenantId: "tenant-a", userId: "alice", permissions: [...DEFAULT_PERMISSIONS] },
	"tok-b": { tenantId: "tenant-b", userId: "bob", permissions: [...DEFAULT_PERMISSIONS] },
	"tok-admin": {
		tenantId: "tenant-a",
		userId: "root",
		permissions: [...DEFAULT_PERMISSIONS, "admin:tool-packs", "admin:usage", "admin:conversations"],
	},
};

function headers(token?: string, extra: Record<string, string> = {}): Record<string, string> {
	return {
		"content-type": "application/json",
		...(token ? { authorization: `Bearer ${token}` } : {}),
		...extra,
	};
}

describe("Security + HTTP hardening", () => {
	let dir: string;
	let runtime: Awaited<ReturnType<typeof bootstrapHarness>>;
	let app: ReturnType<typeof createHttpServer>;

	beforeAll(async () => {
		dir = fs.mkdtempSync(path.join(os.tmpdir(), "helios-security-test-"));
		runtime = await bootstrapHarness({
			config: {
				piDbPath: path.join(dir, "pi.sqlite"),
				appDbPath: path.join(dir, "app.sqlite"),
				useMockModels: true,
				skillsDir: path.join(dir, "skills"),
			},
		});
		app = createHttpServer(runtime.harness, { auth: bearerTokenAuth(KEYS), rateLimit: false });
	});

	afterAll(async () => {
		await runtime.close();
		fs.rmSync(dir, { recursive: true, force: true });
	});

	async function createConv(token: string, body: unknown = {}): Promise<{ conversationId: string; tenant: any }> {
		const res = await app.request("/conversations", { method: "POST", headers: headers(token), body: JSON.stringify(body) });
		expect(res.status).toBe(201);
		return (await res.json()) as { conversationId: string; tenant: any };
	}

	describe("authentication", () => {
		it("returns 401 without credentials, with bad tokens, but keeps /health public", async () => {
			expect((await app.request("/health")).status).toBe(200);
			expect((await app.request("/conversations", { method: "POST", headers: headers(), body: "{}" })).status).toBe(401);
			expect((await app.request("/usage")).status).toBe(401);
			const bad = await app.request("/conversations", { method: "POST", headers: headers("nope"), body: "{}" });
			expect(bad.status).toBe(401);
			expect(bad.headers.get("www-authenticate")).toBe("Bearer");
		});

		it("does not honour x-tenant-id headers unless trusted-headers mode is explicitly enabled", async () => {
			const res = await app.request("/conversations", {
				method: "POST",
				headers: headers(undefined, { "x-tenant-id": "tenant-a", "x-permissions": "*" }),
				body: "{}",
			});
			expect(res.status).toBe(401);
		});

		it("denies everything by default (no env configuration)", async () => {
			vi.stubEnv("HELIOS_API_KEYS", "");
			vi.stubEnv("HELIOS_AUTH_TRUSTED_HEADERS", "");
			try {
				const defaultApp = createHttpServer(runtime.harness);
				const res = await defaultApp.request("/conversations", {
					method: "POST",
					headers: headers(undefined, { "x-tenant-id": "tenant-a" }),
					body: "{}",
				});
				expect(res.status).toBe(401);
			} finally {
				vi.unstubAllEnvs();
			}
		});

		it("builds resolvers from env: HELIOS_API_KEYS and explicit trusted-headers opt-in", async () => {
			const bearer = authFromEnv({ HELIOS_API_KEYS: JSON.stringify({ t1: { tenantId: "x", userId: "y" } }) });
			const tenant = await bearer(new Request("http://h/", { headers: { authorization: "Bearer t1" } }));
			expect(tenant).toMatchObject({ tenantId: "x", userId: "y" });
			expect(await bearer(new Request("http://h/", { headers: { "x-tenant-id": "x" } }))).toBeNull();

			const dev = authFromEnv({ HELIOS_AUTH_TRUSTED_HEADERS: "true" });
			expect(await dev(new Request("http://h/", { headers: { "x-tenant-id": "x" } }))).toMatchObject({ tenantId: "x" });
			expect(() => authFromEnv({ HELIOS_API_KEYS: "not json" })).toThrow(/HELIOS_API_KEYS/);

			const trusted = trustedHeadersAuth();
			expect(await trusted(new Request("http://h/"))).toBeNull();
		});
	});

	describe("tenant spoofing", () => {
		it("ignores body tenant and headers; identity comes only from the authenticated key", async () => {
			const conv = await createConv("tok-a", {
				tenant: { tenantId: "tenant-b", userId: "bob", permissions: ["*"] },
			});
			expect(conv.tenant.tenantId).toBe("tenant-a");
			expect(conv.tenant.userId).toBe("alice");
			expect(conv.tenant.permissions).not.toContain("*");

			const res = await app.request("/conversations", {
				method: "POST",
				headers: headers("tok-a", { "x-tenant-id": "tenant-b", "x-permissions": "*" }),
				body: "{}",
			});
			expect(((await res.json()) as { tenant: { tenantId: string } }).tenant.tenantId).toBe("tenant-a");
		});
	});

	describe("ownership", () => {
		it("returns 404 for every cross-tenant conversation route", async () => {
			const conv = await createConv("tok-a");
			const id = conv.conversationId;

			expect((await app.request(`/conversations/${id}`, { headers: headers("tok-a") })).status).toBe(200);

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
					headers: headers("tok-b"),
					...(body !== undefined ? { body: JSON.stringify(body) } : {}),
				});
				expect(res.status, `${method} ${url}`).toBe(404);
			}

			// Nonexistent id is indistinguishable from another tenant's id.
			expect((await app.request("/conversations/999999", { headers: headers("tok-b") })).status).toBe(404);
		});

		it("scopes /approvals to the caller tenant and checks signal task ownership", async () => {
			const conv = await createConv("tok-a");
			const compactRes = await app.request(`/conversations/${conv.conversationId}/compact`, {
				method: "POST",
				headers: headers("tok-a"),
				body: "{}",
			});
			const { taskId } = (await compactRes.json()) as { taskId: string };

			const listB = await app.request("/approvals", { headers: headers("tok-b") });
			expect(listB.status).toBe(200);
			expect(((await listB.json()) as { approvals: unknown[] }).approvals).toEqual([]);

			const signalBody = JSON.stringify({ taskId, signalName: "nonexistent.signal", key: "k", payload: null });
			const crossTenant = await app.request("/signals", { method: "POST", headers: headers("tok-b"), body: signalBody });
			expect(crossTenant.status).toBe(404);
			const owner = await app.request("/signals", { method: "POST", headers: headers("tok-a"), body: signalBody });
			expect(owner.status).toBe(200);
			expect(await owner.json()).toEqual({ status: "invalid" });

			const missing = await app.request("/signals", {
				method: "POST",
				headers: headers("tok-a"),
				body: JSON.stringify({ taskId: "987654", signalName: "s", key: "k" }),
			});
			expect(missing.status).toBe(404);
		});

		it("paginates across all conversations when listing a tenant's approvals", async () => {
			for (let i = 0; i < 205; i++) {
				await runtime.harness.createConversation({
					tenant: { tenantId: "tenant-bulk", userId: "u", permissions: [] },
				});
			}
			await expect(
				runtime.harness.listPendingApprovals(undefined, { tenantId: "tenant-bulk" }),
			).resolves.toEqual([]);
		}, 60_000);
	});

	describe("admin gating", () => {
		it("requires admin:tool-packs to toggle packs and admin:usage to read usage", async () => {
			const packs = runtime.harness.listToolPacks();
			const name = packs[0]!.name;
			const original = packs[0]!.enabled;

			const denied = await app.request(`/tool-packs/${name}`, {
				method: "POST",
				headers: headers("tok-a"),
				body: JSON.stringify({ enabled: !original }),
			});
			expect(denied.status).toBe(403);
			expect(runtime.harness.listToolPacks().find((p) => p.name === name)?.enabled).toBe(original);

			const ok = await app.request(`/tool-packs/${name}`, {
				method: "POST",
				headers: headers("tok-admin"),
				body: JSON.stringify({ enabled: original }),
			});
			expect(ok.status).toBe(200);

			const unknown = await app.request("/tool-packs/does-not-exist", {
				method: "POST",
				headers: headers("tok-admin"),
				body: JSON.stringify({ enabled: true }),
			});
			expect(unknown.status).toBe(404);

			expect((await app.request("/tool-packs", { headers: headers("tok-a") })).status).toBe(200);
			expect((await app.request("/usage", { headers: headers("tok-a") })).status).toBe(403);
			expect((await app.request("/usage", { headers: headers("tok-admin") })).status).toBe(200);
		});
	});

	describe("fork", () => {
		it("cannot change tenant; child inherits the parent's tenant", async () => {
			const conv = await createConv("tok-a");
			runtime.testModel.enqueueMain({ type: "text", text: "fork me" });
			const msg = await app.request(`/conversations/${conv.conversationId}/messages`, {
				method: "POST",
				headers: headers("tok-a"),
				body: JSON.stringify({ content: "hello", wait: true }),
			});
			expect(msg.status).toBe(200);
			const transcript = await runtime.harness.getTranscript(conv.conversationId);
			const entryId = transcript[transcript.length - 1]!.id;

			const spoof = await app.request(`/conversations/${conv.conversationId}/fork`, {
				method: "POST",
				headers: headers("tok-a"),
				body: JSON.stringify({ atEntryId: entryId, tenant: { tenantId: "tenant-b", userId: "bob", permissions: ["*"] } }),
			});
			expect(spoof.status).toBe(400);

			// A caller from another tenant (even with `*`) cannot fork: the parent surfaces as not found.
			await expect(
				runtime.harness.forkConversation(conv.conversationId, entryId, {
					tenant: { tenantId: "tenant-b", userId: "bob", permissions: ["*"] },
				}),
			).rejects.toBeInstanceOf(NotFoundError);

			const ok = await app.request(`/conversations/${conv.conversationId}/fork`, {
				method: "POST",
				headers: headers("tok-a"),
				body: JSON.stringify({ atEntryId: entryId }),
			});
			expect(ok.status).toBe(201);
			const child = (await ok.json()) as { conversationId: string; tenant: { tenantId: string; userId: string } };
			expect(child.tenant).toMatchObject({ tenantId: "tenant-a", userId: "alice" });
			const info = await runtime.harness.getConversation(child.conversationId);
			expect(info?.tenant.tenantId).toBe("tenant-a");
			expect(info?.tenant.permissions).toEqual(conv.tenant.permissions);
		});
	});

	describe("input handling", () => {
		it("returns 400 (never 500) for malformed JSON and invalid shapes", async () => {
			const conv = await createConv("tok-a");
			const cases: Array<[string, string]> = [
				["/conversations", "{not json"],
				["/conversations", "[1,2]"],
				["/conversations", JSON.stringify({ model: "gpt" })],
				["/conversations", JSON.stringify({ maxTurns: -3 })],
				["/conversations", JSON.stringify({ enabledToolPacks: "x" })],
				[`/conversations/${conv.conversationId}/messages`, "{"],
				[`/conversations/${conv.conversationId}/messages`, JSON.stringify({})],
				[`/conversations/${conv.conversationId}/messages`, JSON.stringify({ content: 5 })],
				[`/conversations/${conv.conversationId}/messages`, JSON.stringify({ content: "x", whenBusy: "later" })],
				[`/conversations/${conv.conversationId}/fork`, "nope"],
				[`/conversations/${conv.conversationId}/fork`, JSON.stringify({ atEntryId: "abc" })],
				[`/conversations/${conv.conversationId}/compact`, "{bad"],
				[`/conversations/${conv.conversationId}/reset`, JSON.stringify({ handoff: 1 })],
				["/signals", "{"],
				["/signals", JSON.stringify({ taskId: "1" })],
				["/tool-packs/example-domain", JSON.stringify({ enabled: "yes" })],
			];
			for (const [url, body] of cases) {
				const res = await app.request(url, { method: "POST", headers: headers("tok-admin"), body });
				expect(res.status, `${url} ${body}`).toBe(400);
				const json = (await res.json()) as { error: string; requestId: string };
				expect(json.error.length).toBeGreaterThan(0);
				expect(json.requestId).toBeTruthy();
			}
			// Non-numeric conversation ids are a 404, not a crash.
			expect((await app.request("/conversations/abc", { headers: headers("tok-a") })).status).toBe(404);
		});

		it("enforces the body size limit", async () => {
			const small = createHttpServer(runtime.harness, { auth: bearerTokenAuth(KEYS), maxBodyBytes: 64, rateLimit: false });
			const res = await small.request("/conversations", {
				method: "POST",
				headers: headers("tok-a"),
				body: JSON.stringify({ instructions: "x".repeat(500) }),
			});
			expect(res.status).toBe(413);
		});

		it("echoes or generates x-request-id", async () => {
			const echoed = await app.request("/health", { headers: { "x-request-id": "req-123" } });
			expect(echoed.headers.get("x-request-id")).toBe("req-123");
			const generated = await app.request("/health", { headers: { "x-request-id": "bad id with spaces" } });
			expect(generated.headers.get("x-request-id")).toMatch(/^[0-9a-f-]{36}$/);
		});

		it("hides internal error details on 500 and returns the request id", async () => {
			const broken = {
				assertConversationOwner: async () => undefined,
				getConversation: async () => {
					throw new Error("SECRET /var/db/path exploded at line 42");
				},
			} as unknown as HarnessPort;
			const quiet = { debug() {}, info() {}, warn() {}, error() {}, child() { return quiet; }, level: "error", isEnabled: () => false } as any;
			const brokenApp = createHttpServer(broken, { auth: bearerTokenAuth(KEYS), rateLimit: false, logger: quiet });
			const res = await brokenApp.request("/conversations/1", { headers: headers("tok-a", { "x-request-id": "r-1" }) });
			expect(res.status).toBe(500);
			const text = await res.text();
			expect(text).not.toContain("SECRET");
			expect(text).not.toContain("line 42");
			expect(JSON.parse(text)).toMatchObject({ error: "Internal server error", requestId: "r-1" });
		});

		it("applies a per-tenant rate limit", async () => {
			const limited = createHttpServer(runtime.harness, {
				auth: bearerTokenAuth(KEYS),
				rateLimit: { limit: 2, windowMs: 60_000 },
			});
			expect((await limited.request("/tool-packs", { headers: headers("tok-a") })).status).toBe(200);
			expect((await limited.request("/tool-packs", { headers: headers("tok-a") })).status).toBe(200);
			const third = await limited.request("/tool-packs", { headers: headers("tok-a") });
			expect(third.status).toBe(429);
			expect(third.headers.get("retry-after")).toBeTruthy();
			// Another tenant is unaffected.
			expect((await limited.request("/tool-packs", { headers: headers("tok-b") })).status).toBe(200);
		});
	});

	describe("compact", () => {
		it("returns 202 {taskId} by default and blocks only when wait is true", async () => {
			const conv = await createConv("tok-a");
			const res = await app.request(`/conversations/${conv.conversationId}/compact`, {
				method: "POST",
				headers: headers("tok-a"),
				body: "{}",
			});
			expect(res.status).toBe(202);
			const body = (await res.json()) as { taskId: string; compacted?: boolean };
			expect(body.taskId).toMatch(/^\d+$/);
			expect(body.compacted).toBeUndefined();

			const waited = await app.request(`/conversations/${conv.conversationId}/compact`, {
				method: "POST",
				headers: headers("tok-a"),
				body: JSON.stringify({ wait: true }),
			});
			expect(waited.status).toBe(200);
			expect(typeof ((await waited.json()) as { compacted: boolean }).compacted).toBe("boolean");
		}, 30_000);
	});

	describe("SSE stream", () => {
		it("passes Last-Event-ID as afterEntryId, uses entry ids as SSE ids, and rejects other tenants", async () => {
			const seen: Array<{ id: string; afterEntryId?: string }> = [];
			const fake = {
				assertConversationOwner: async (caller: TenantContext) => {
					if (caller.tenantId !== "tenant-a") throw new NotFoundError('Conversation "1" not found');
				},
				watchConversation: (id: string, _signal?: AbortSignal, options?: { afterEntryId?: string }) => {
					seen.push({ id, ...(options?.afterEntryId ? { afterEntryId: options.afterEntryId } : {}) });
					return (async function* (): AsyncGenerator<HarnessEvent> {
						yield {
							type: "entry_appended",
							conversationId: id,
							entry: { id: "42", kind: "message", role: "assistant", text: "hi" },
						};
						yield { type: "stream_closed", conversationId: id, reason: "ended" };
					})();
				},
			} as unknown as HarnessPort;
			const sseApp = createHttpServer(fake, { auth: bearerTokenAuth(KEYS), rateLimit: false, keepAliveMs: 10 });

			const res = await sseApp.request("/conversations/1/stream", {
				headers: headers("tok-a", { "last-event-id": "41" }),
			});
			expect(res.status).toBe(200);
			expect(res.headers.get("content-type")).toContain("text/event-stream");
			const text = await res.text();
			expect(seen).toEqual([{ id: "1", afterEntryId: "41" }]);
			expect(text).toContain("id: 42");
			expect(text).toContain("event: entry_appended");

			const viaQuery = await sseApp.request("/conversations/1/stream?afterEntryId=7", { headers: headers("tok-a") });
			await viaQuery.text();
			expect(seen[1]).toEqual({ id: "1", afterEntryId: "7" });

			expect((await sseApp.request("/conversations/1/stream?afterEntryId=x", { headers: headers("tok-a") })).status).toBe(400);
			expect((await sseApp.request("/conversations/1/stream", { headers: headers("tok-b") })).status).toBe(404);
		});
	});
});
