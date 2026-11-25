import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AppDatabase, SqliteMemoryService, DEFAULT_PERMISSIONS } from "../src/index.js";
import { readBoundedText } from "../src/mcp/mcp-client.js";
import { JsonlTraceExporter } from "../src/runtime/trace-export.js";
import { redactSecrets } from "../src/security/secret-redactor.js";
import { bearerTokenAuth } from "../src/transport/auth.js";
import { createHttpServer } from "../src/transport/http-server.js";

describe("secret redactor: JSON and vendor-token gaps", () => {
	it("redacts JSON-quoted keys, which a bare `key=value` pattern misses", () => {
		const out = redactSecrets(JSON.stringify({ password: "hunter2hunter2", apiKey: "abcd1234efgh5678", note: "ok" }));
		expect(out.text).not.toContain("hunter2hunter2");
		expect(out.text).not.toContain("abcd1234efgh5678");
		expect(out.text).toContain('"note":"ok"');
		expect(out.categories).toContain("secret_assignment");
	});

	it("still redacts unquoted and single-quoted assignments", () => {
		expect(redactSecrets("password=hunter2hunter2").text).not.toContain("hunter2hunter2");
		expect(redactSecrets("token: 'abcdefgh12345678'").text).not.toContain("abcdefgh12345678");
	});

	it("redacts quoted values longer than the old 128-char limit", () => {
		const long = "x".repeat(300);
		expect(redactSecrets(`{"secret":"${long}"}`).text).not.toContain("xxxxxxxxxx");
	});

	it("does not touch ordinary prose or short values", () => {
		const text = "The token count was high, tokens: 5 and the secret garden was lovely.";
		expect(redactSecrets(text).text).toBe(text);
	});

	it("redacts common vendor tokens", () => {
		const samples = [
			`glpat-${"abcdefghij"}1234567890`,
			`npm_${"a".repeat(36)}`,
			`ya29.${"b".repeat(40)}`,
			`SG.${"c".repeat(22)}.${"d".repeat(43)}`,
		];
		for (const sample of samples) {
			const out = redactSecrets(`value ${sample} end`);
			expect(out.text, sample).not.toContain(sample);
			expect(out.redactedCount).toBeGreaterThan(0);
		}
	});
});

describe("forgetMemory scrubs the episodes that evidenced the memory", () => {
	it("blanks verbatim episode text and its FTS row, keeps the ledger row, and spares unrelated episodes", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const memory = new SqliteMemoryService(appDb);
			const mem = (await memory.applyMemoryMutation("t", "u", {
				op: "ADD",
				key: "user.address",
				kind: "fact",
				content: "Lives at 12 Privet Drive.",
				confidence: 0.9,
			}))!;

			const insertEpisode = (id: string, text: string): void => {
				appDb.db
					.prepare(
						`INSERT INTO midterm_episodes (id, tenant_id, user_id, conversation_id, run_id, user_message, assistant_summary,
						   tool_trace_json, friction_signals_json, created_at)
						 VALUES (?, 't', 'u', 'c', ?, ?, 'summary', '[{"tool":"x"}]', '[]', 1)`,
					)
					.run(id, id, text);
				appDb.db
					.prepare(
						"INSERT INTO midterm_episodes_fts (episode_id, tenant_id, user_id, user_message, assistant_summary) VALUES (?, 't', 'u', ?, 'summary')",
					)
					.run(id, text);
			};
			insertEpisode("ep-secret", "my address is 12 Privet Drive");
			insertEpisode("ep-other", "unrelated chatter");
			appDb.db.prepare("INSERT INTO memory_evidence (memory_id, episode_id, created_at) VALUES (?, 'ep-secret', 1)").run(mem.id);

			expect(await memory.forgetMemory("t", "u", mem.id)).toBe(true);

			const secret = appDb.db.prepare("SELECT * FROM midterm_episodes WHERE id = 'ep-secret'").get() as Record<string, unknown>;
			expect(secret.user_message).toBe("");
			expect(secret.assistant_summary).toBe("");
			expect(secret.tool_trace_json).toBe("[]");
			expect(secret.consolidated).toBe(1);
			expect(appDb.db.prepare("SELECT COUNT(*) AS n FROM midterm_episodes_fts WHERE episode_id = 'ep-secret'").get()).toEqual({ n: 0 });

			const other = appDb.db.prepare("SELECT user_message, consolidated FROM midterm_episodes WHERE id = 'ep-other'").get() as Record<string, unknown>;
			expect(other.user_message).toBe("unrelated chatter");
			expect(other.consolidated).toBe(0);
			expect(appDb.db.prepare("SELECT COUNT(*) AS n FROM memory_evidence WHERE memory_id = ?").get(mem.id)).toEqual({ n: 1 });
		} finally {
			appDb.close();
		}
	});
});

describe("MCP bounded response reader", () => {
	it("returns small bodies and rejects oversized ones, declared or streamed", async () => {
		expect(await readBoundedText(new Response('{"ok":true}'), 1024)).toBe('{"ok":true}');
		await expect(readBoundedText(new Response("x".repeat(2048)), 1024)).rejects.toThrow(/too large/);

		const streamed = new Response(
			new ReadableStream<Uint8Array>({
				start(controller) {
					for (let i = 0; i < 4; i++) controller.enqueue(new TextEncoder().encode("y".repeat(512)));
					controller.close();
				},
			}),
		);
		await expect(readBoundedText(streamed, 1024)).rejects.toThrow(/too large/);
	});
});

describe("trace exporter rotation", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const dir of dirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
	});

	it("rotates without leaving a stale file descriptor and keeps writing", () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "trace-rot-"));
		dirs.push(dir);
		const file = path.join(dir, "trace.jsonl");
		const errors: unknown[] = [];
		const exporter = new JsonlTraceExporter({ filePath: file, maxBytes: 300, onError: (e) => errors.push(e) });
		const span = { traceId: "t", spanId: "s", name: "pi.durable.phase", startTimeMs: 1, endTimeMs: 2, attributes: {}, events: [], links: [], status: { status: "ok" } } as never;
		for (let i = 0; i < 10; i++) exporter.write(span);
		exporter.close();
		expect(errors).toEqual([]);
		expect(fs.existsSync(`${file}.1`)).toBe(true);
		expect(fs.readFileSync(file, "utf8").trim().split("\n").length).toBeGreaterThan(0);
	});
});

describe("HTTP hardening", () => {
	const KEYS = {
		user: { tenantId: "t", userId: "alice", permissions: [...DEFAULT_PERMISSIONS] },
		ops: { tenantId: "t", userId: "ops", permissions: [...DEFAULT_PERMISSIONS, "admin:usage"] },
	};
	const harness = {
		listMcpServers: () => [
			{ name: "docs", connected: false, discoveredTools: 0, readOnly: true, lastError: "boom from https://internal.example/secret" },
		],
	} as never;
	const app = createHttpServer(harness, { auth: bearerTokenAuth(KEYS), rateLimit: false });
	const get = (token: string, url: string) => app.request(url, { headers: { authorization: `Bearer ${token}` } });

	it("sets defensive response headers", async () => {
		const res = await get("user", "/mcp/servers");
		expect(res.headers.get("x-content-type-options")).toBe("nosniff");
		expect(res.headers.get("cache-control")).toBe("no-store");
		expect(res.headers.get("referrer-policy")).toBe("no-referrer");
		const health = await app.request("/health");
		expect(health.headers.get("x-content-type-options")).toBe("nosniff");
	});

	it("shows upstream MCP error text only to operators", async () => {
		const user = (await (await get("user", "/mcp/servers")).json()) as { servers: Array<Record<string, unknown>> };
		expect(user.servers[0]).not.toHaveProperty("lastError");
		expect(user.servers[0]).toMatchObject({ name: "docs", connected: false });

		const ops = (await (await get("ops", "/mcp/servers")).json()) as { servers: Array<Record<string, unknown>> };
		expect(ops.servers[0]?.lastError).toContain("boom");
	});

	it("records the authenticated caller, not a client-supplied name, as the approver", async () => {
		const delivered: Array<Record<string, unknown>> = [];
		const signalHarness = {
			assertTaskOwner: async () => undefined,
			deliverSignal: async (input: Record<string, unknown>) => {
				delivered.push(input);
				return { delivered: true };
			},
		} as never;
		const signalApp = createHttpServer(signalHarness, { auth: bearerTokenAuth(KEYS), rateLimit: false });
		const res = await signalApp.request("/signals", {
			method: "POST",
			headers: { authorization: "Bearer user", "content-type": "application/json" },
			body: JSON.stringify({
				taskId: "7",
				signalName: "app.tool_gate.decision",
				key: "gate_1",
				payload: { approved: true, reviewer: "security-team" },
			}),
		});
		expect(res.status).toBe(200);
		expect(delivered[0]?.payload).toEqual({ approved: true, reviewer: "alice" });
	});
});

describe("review follow-ups", () => {
	it("redacts quoted passphrases that contain spaces and values past 512 chars", () => {
		const phrase = redactSecrets('{"password": "correct horse battery staple"}');
		expect(phrase.text).not.toContain("horse battery");
		expect(redactSecrets("passphrase='two words here'").text).not.toContain("two words");
		const long = "z".repeat(1500);
		expect(redactSecrets(`token=${long}`).text).not.toContain("zzzzzzzzzz");
	});

	it("does not apply a stale consolidation plan once one of its episodes is gone", async () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const memory = new SqliteMemoryService(appDb);
			const insert = appDb.db.prepare(
				`INSERT INTO midterm_episodes (id, tenant_id, user_id, conversation_id, run_id, user_message, assistant_summary,
				   tool_trace_json, friction_signals_json, consolidated, created_at)
				 VALUES (?, 't', 'u', 'c', ?, 'm', 's', '[]', '[]', ?, 1)`,
			);
			insert.run("ep-live", "r1", 0);
			insert.run("ep-scrubbed", "r2", 1);
			const result = await memory.applyConsolidation("t", "u", {
				ops: [{ op: "ADD", key: "k.resurrected", kind: "fact", content: "forgotten fact", confidence: 0.9 }],
				episodeIds: ["ep-live", "ep-scrubbed"],
			});
			expect(result.applied).toBe(false);
			expect(await memory.search("t", "u", "forgotten fact")).toEqual([]);
			// The live episode stays pending so the consolidator can re-plan it.
			expect(appDb.db.prepare("SELECT consolidated FROM midterm_episodes WHERE id = 'ep-live'").get()).toEqual({ consolidated: 0 });
		} finally {
			appDb.close();
		}
	});
});
