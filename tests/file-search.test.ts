import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	DEFAULT_PERMISSIONS,
	FallbackFileSearchService,
	LocalWorkspaceFileSearchService,
	normalizePathPrefix,
	sanitizeXerjIndexSlug,
	validateXerjBaseUrl,
	XerjFileSearchService,
	type FileSearchScope,
	type TenantContext,
} from "../src/index.js";

const TENANT_A: TenantContext = {
	tenantId: "acme_corp",
	userId: "alice",
	permissions: [...DEFAULT_PERMISSIONS],
};

const TENANT_B: TenantContext = {
	tenantId: "beta-inc",
	userId: "bob",
	workspaceId: "payments",
	permissions: [...DEFAULT_PERMISSIONS],
};

const NO_FILES_PERM_TENANT: TenantContext = {
	tenantId: "acme_corp",
	userId: "restricted",
	permissions: ["memory:read", "artifacts:read"],
};

interface RecordedRequest {
	readonly method: string;
	readonly url: string;
	readonly headers: http.IncomingHttpHeaders;
	readonly body: unknown;
}

async function startMockXerjServer(
	handler: (req: RecordedRequest, res: http.ServerResponse) => void,
): Promise<{
	readonly url: string;
	readonly requests: RecordedRequest[];
	close(): Promise<void>;
}> {
	const requests: RecordedRequest[] = [];
	const server = http.createServer((req, res) => {
		const chunks: Buffer[] = [];
		req.on("data", (chunk) => chunks.push(chunk as Buffer));
		req.on("end", () => {
			const raw = Buffer.concat(chunks).toString("utf8");
			let parsed: unknown = undefined;
			if (raw.trim()) {
				try {
					parsed = JSON.parse(raw);
				} catch {
					parsed = raw;
				}
			}
			const record: RecordedRequest = {
				method: req.method ?? "GET",
				url: req.url ?? "/",
				headers: req.headers,
				body: parsed,
			};
			requests.push(record);
			handler(record, res);
		});
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
	const addr = server.address();
	if (!addr || typeof addr === "string") throw new Error("Failed to bind mock XERJ server");
	return {
		url: `http://127.0.0.1:${addr.port}`,
		requests,
		close: () => new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve()))),
	};
}

describe("XERJ & Indexed File Search Integration", () => {
	const cleanups: Array<() => Promise<void> | void> = [];
	afterEach(async () => {
		while (cleanups.length > 0) {
			await cleanups.pop()!();
		}
	});

	it("validates URLs, sanitizes index slugs, and blocks path traversal", () => {
		expect(validateXerjBaseUrl("http://127.0.0.1:9200/")).toBe("http://127.0.0.1:9200");
		expect(() => validateXerjBaseUrl("ftp://127.0.0.1:9200")).toThrow(/Unsupported XERJ URL protocol/);
		expect(() => validateXerjBaseUrl("http://user:pass@127.0.0.1:9200")).toThrow(/must not embed credentials/);

		expect(sanitizeXerjIndexSlug("My_Proj-*")).toBe("my-proj");
		expect(sanitizeXerjIndexSlug("  acme/corp  ")).toBe("acme-corp");
		expect(() => sanitizeXerjIndexSlug("***")).toThrow(/Invalid XERJ/);

		expect(normalizePathPrefix("./src\\memory//")).toBe("src/memory/");
		expect(normalizePathPrefix("   ")).toBeUndefined();
		expect(() => normalizePathPrefix("../etc/passwd")).toThrow(/traversal/);
		expect(() => normalizePathPrefix("src/../../secret")).toThrow(/traversal/);
	});

	it("enforces tenant-scoped index resolution and sends projected _source + _passage + defs boost to XERJ", async () => {
		const mock = await startMockXerjServer((req, res) => {
			if (req.method === "GET" && req.url === "/") {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ name: "xerj-test", version: { number: "8.13.0" }, tagline: "You Know, for Search" }));
				return;
			}
			if (req.method === "POST" && req.url.includes("/_search")) {
				if (req.url.startsWith("/autoindex-catalog/")) {
					res.writeHead(200, { "content-type": "application/json" });
					res.end(
						JSON.stringify({
							took: 2,
							hits: {
								total: { value: 2 },
								hits: [
									{
										_id: "ds:ax-acme-corp:docs",
										_source: {
											doc_kind: "dataset",
											index_name: "ax-acme-corp-docs",
											slug: "docs",
											record_count: 142,
											formats: ["code", "json"],
										},
									},
									{
										_id: "file:ax-acme-corp:1",
										_source: {
											// Per-file catalog entry must be skipped by getIndexMap
											doc_kind: "file",
											path: "src/runtime/budget.ts",
										},
									},
									{
										_id: "ds:ax-other-tenant:docs",
										_source: {
											// Another tenant's dataset must be filtered out!
											doc_kind: "dataset",
											index_name: "ax-other-tenant-docs",
											slug: "docs",
											record_count: 999,
											formats: ["code"],
										},
									},
								],
							},
						}),
					);
					return;
				}
				res.writeHead(200, { "content-type": "application/json" });
				res.end(
					JSON.stringify({
						took: 4,
						_xerj: { hints: [{ code: "index_stale", message: "2 files modified since last autoindex" }] },
						hits: {
							total: { value: 2 },
							hits: [
								{
									_id: "doc-1",
									_score: 14.5,
									match_line: 42,
									_source: {
										title: "budget.ts",
										ax_path: "src/runtime/budget.ts",
										language: "typescript",
										ax_format: "ts",
										ax_dataset: "code",
										symbol_count: 6,
										symbols: [
											{ name: "BudgetGate", kind: "class", line: 42 },
											{ name: "evaluateBudget", kind: "function", line: 108 },
										],
									},
									fields: {
										_passage: [
											{
												field: "body",
												start_line: 42,
												text: "export class BudgetGate { check(tokens: number) { /* ghp_1234567890abcdef1234567890abcdef12 */ } }",
											},
										],
									},
								},
								{
									_id: "doc-2",
									_score: 4.1,
									_source: {
										title: "index.ts",
										ax_path: "src/index.ts",
										language: "typescript",
										ax_format: "ts",
										symbol_count: 2,
										symbols: [{ name: "bootstrapHarness", kind: "function", line: 392 }],
									},
									fields: {
										_passage: [{ field: "body", start_line: 507, text: "const budgetGate = new BudgetGate();" }],
									},
								},
							],
						},
					}),
				);
				return;
			}
			res.writeHead(404, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: { type: "index_not_found_exception" } }));
		});
		cleanups.push(() => mock.close());

		const xerj = new XerjFileSearchService({
			baseUrl: mock.url,
			apiKey: "secret-xerj-admin-key",
			indexPrefix: "ax",
			scopeMode: "tenant",
		});

		const scopeA: FileSearchScope = { tenantId: TENANT_A.tenantId, userId: TENANT_A.userId };
		const scopeB: FileSearchScope = {
			tenantId: TENANT_B.tenantId,
			userId: TENANT_B.userId,
			workspaceId: TENANT_B.workspaceId,
		};

		expect(xerj.resolveIndex(scopeA)).toBe("ax-acme-corp-*");
		expect(xerj.resolveIndex(scopeB)).toBe("ax-beta-inc-payments-*");

		const health = await xerj.healthCheck();
		expect(health.ok).toBe(true);
		expect(health.version).toBe("8.13.0");

		// 1. file_find projects _source WITHOUT body and scopes to tenant index
		const found = await xerj.findFiles(scopeA, { query: "budget", extension: "ts", pathPrefix: "src/" });
		expect(found.hits).toHaveLength(2);
		expect(found.hits[0]!.path).toBe("src/runtime/budget.ts");
		expect(found.notes?.[0]).toContain("index_stale");

		const findReq = mock.requests[1]!;
		expect(findReq.url).toBe(`/${encodeURIComponent("ax-acme-corp-*")}/_search`);
		expect(findReq.headers.authorization).toBe("ApiKey secret-xerj-admin-key");
		const findBody = findReq.body as {
			_source: string[];
			fields?: string[];
			query: { bool: { filter: Array<Record<string, unknown>>; must_not: Array<Record<string, unknown>> } };
		};
		expect(findBody._source).not.toContain("body");
		expect(findBody.fields).toBeUndefined();
		// Extension filter targets `ax_path` wildcard (`*.ts`), NOT `ax_format` (which holds content family in XERJ).
		expect(findBody.query.bool.filter).toEqual(
			expect.arrayContaining([expect.objectContaining({ wildcard: { ax_path: { value: "*.ts", case_insensitive: true } } })]),
		);
		expect(findBody.query.bool.must_not).toEqual([{ exists: { field: "kind" } }]);

		// 2. searchContent requests fields: ["_passage"] and uses defs match_phrase boost:4 (no body in _source)
		const searched = await xerj.searchContent(scopeA, { query: "BudgetGate" });
		expect(searched.hits[0]!.path).toBe("src/runtime/budget.ts");
		expect(searched.hits[0]!.line).toBe(42);
		expect(searched.hits[0]!.snippet).toContain("export class BudgetGate");

		const searchBody = mock.requests[2]!.body as {
			_source: string[];
			fields: string[];
			query: { bool: { should: Array<Record<string, unknown>> } };
		};
		expect(searchBody._source).not.toContain("body");
		expect(searchBody.fields).toEqual(["_passage"]);
		expect(searchBody.query.bool.should).toEqual(
			expect.arrayContaining([expect.objectContaining({ match_phrase: { defs: { query: "BudgetGate", boost: 4 } } })]),
		);

		// 2b. Hybrid search omits `fields: ["_passage"]` (forbidden by XERJ on hybrid) and uses `highlight` instead
		const hybridRes = await xerj.searchContent(scopeA, { query: "BudgetGate", mode: "hybrid" });
		expect(hybridRes.mode).toBe("hybrid");
		const hybridBody = mock.requests[3]!.body as { fields?: string[]; highlight?: Record<string, unknown> };
		expect(hybridBody.fields).toBeUndefined();
		expect(hybridBody.highlight).toBeDefined();

		// 3. findSymbolDefinitions ranks exact symbol definition first with line number
		const defs = await xerj.findSymbolDefinitions(scopeA, { symbol: "BudgetGate", kind: "class" });
		expect(defs.hits).toHaveLength(1);
		expect(defs.hits[0]!.path).toBe("src/runtime/budget.ts");
		expect(defs.hits[0]!.line).toBe(42);
		expect(defs.hits[0]!.symbols?.[0]).toEqual({ name: "BudgetGate", kind: "class", line: 42 });

		// 4. getIndexMap strictly filters autoindex-catalog dataset rows to the caller's tenant prefix
		const map = await xerj.getIndexMap(scopeA);
		expect(map.datasets).toHaveLength(1);
		expect(map.datasets[0]!.index).toBe("ax-acme-corp-docs");
		expect(map.totalDocs).toBe(142);
	});

	it("degrades semantic/hybrid search cleanly to lexical on 400 mapping error, handles 404 empty index, and trips circuit breaker on 5xx", async () => {
		let mode: "semantic_400" | "missing_404" | "fail_503" = "semantic_400";
		let requestCount = 0;

		const mock = await startMockXerjServer((req, res) => {
			requestCount += 1;
			if (mode === "semantic_400") {
				const body = req.body as { query?: { semantic?: unknown } };
				if (body?.query?.semantic) {
					res.writeHead(400, { "content-type": "application/json" });
					res.end(JSON.stringify({ error: { type: "illegal_argument_exception", reason: "field body is not mapped as semantic_text" } }));
					return;
				}
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ hits: { total: { value: 1 }, hits: [{ _id: "1", _score: 3, _source: { ax_path: "README.md" } }] } }));
				return;
			}
			if (mode === "missing_404") {
				res.writeHead(404, { "content-type": "application/json" });
				res.end(JSON.stringify({ error: { type: "index_not_found_exception" } }));
				return;
			}
			res.writeHead(503, { "content-type": "application/json" });
			res.end("service unavailable");
		});
		cleanups.push(() => mock.close());

		let nowMs = 1_000;
		const xerj = new XerjFileSearchService({
			baseUrl: mock.url,
			scopeMode: "shared",
			circuitBreaker: { failureThreshold: 2, cooldownMs: 10_000 },
			now: () => nowMs,
		});
		const scope: FileSearchScope = { tenantId: "t", userId: "u" };

		// Semantic fallback -> lexical
		const semRes = await xerj.searchContent(scope, { query: "startup", mode: "semantic" });
		expect(semRes.mode).toBe("lexical");
		expect(semRes.hits).toHaveLength(1);
		expect(semRes.notes?.[0]).toContain("fell back to definition-first lexical");

		// 404 returns clean empty result (does NOT trip circuit breaker)
		mode = "missing_404";
		const emptyRes = await xerj.findFiles(scope, { query: "foo" });
		expect(emptyRes.hits).toEqual([]);
		expect(emptyRes.notes?.[0]).toContain("No indexed files found");

		// 503 trips circuit breaker after 2 failures; 3rd call fails fast without hitting network
		mode = "fail_503";
		await expect(xerj.findFiles(scope, { query: "a" })).rejects.toThrow(/503/);
		await expect(xerj.findFiles(scope, { query: "b" })).rejects.toThrow(/503/);
		const countBeforeOpen = requestCount;
		await expect(xerj.findFiles(scope, { query: "c" })).rejects.toThrow(/circuit breaker is open/i);
		expect(requestCount).toBe(countBeforeOpen);

		// After cooldown, half-open probe hits network again
		nowMs += 11_000;
		mode = "missing_404";
		const recovered = await xerj.findFiles(scope, { query: "d" });
		expect(recovered.hits).toEqual([]);
		expect(requestCount).toBe(countBeforeOpen + 1);
	});

	it("searches a local workspace safely, ignores build dirs, finds symbol definitions, and falls back when XERJ is down", async () => {
		const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "helios-filesearch-"));
		cleanups.push(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

		fs.mkdirSync(path.join(tmpDir, "src", "core"), { recursive: true });
		fs.mkdirSync(path.join(tmpDir, "node_modules", "ignored-pkg"), { recursive: true });
		fs.writeFileSync(
			path.join(tmpDir, "src", "core", "rotator.ts"),
			[
				"export interface WalConfig { segmentBytes: number; }",
				"export function rotateWalSegment(cfg: WalConfig): boolean {",
				"  return cfg.segmentBytes > 0;",
				"}",
			].join("\n"),
			"utf8",
		);
		fs.writeFileSync(
			path.join(tmpDir, "src", "core", "caller.ts"),
			`import { rotateWalSegment } from "./rotator.js";\nconsole.log(rotateWalSegment({ segmentBytes: 1024 }));\n`,
			"utf8",
		);
		fs.writeFileSync(
			path.join(tmpDir, "node_modules", "ignored-pkg", "rotator.ts"),
			"export function rotateWalSegment() { return false; }\n",
			"utf8",
		);

		const local = new LocalWorkspaceFileSearchService({ workspaceRoot: tmpDir });
		const scope: FileSearchScope = { tenantId: "t", userId: "u" };

		// Skips node_modules
		const found = await local.findFiles(scope, { query: "rotator" });
		expect(found.hits.map((h) => h.path)).toEqual(["src/core/rotator.ts"]);

		// Ranks defining file ABOVE caller file!
		const searched = await local.searchContent(scope, { query: "rotateWalSegment" });
		expect(searched.hits.length).toBe(2);
		expect(searched.hits[0]!.path).toBe("src/core/rotator.ts");
		expect(searched.hits[0]!.line).toBe(2);

		// Symbol definition lookup finds exact line and kind
		const defs = await local.findSymbolDefinitions(scope, { symbol: "rotateWalSegment", kind: "function" });
		expect(defs.hits).toHaveLength(1);
		expect(defs.hits[0]!.path).toBe("src/core/rotator.ts");
		expect(defs.hits[0]!.line).toBe(2);

		// FallbackFileSearchService fails over cleanly from broken XERJ to local workspace
		const deadXerj = new XerjFileSearchService({
			baseUrl: "http://127.0.0.1:1",
			timeoutMs: 500,
			fetchImpl: async () => {
				throw new Error("ECONNREFUSED");
			},
		});
		const fallbackEvents: string[] = [];
		const resilient = new FallbackFileSearchService({
			primary: deadXerj,
			fallback: local,
			onFallback: (op) => fallbackEvents.push(op),
		});

		const res = await resilient.findSymbolDefinitions(scope, { symbol: "WalConfig", kind: "interface" });
		expect(res.hits).toHaveLength(1);
		expect(res.hits[0]!.path).toBe("src/core/rotator.ts");
		expect(res.engine).toBe("local");
		expect(res.notes?.some((n) => n.includes('fallback "local"'))).toBe(true);
		expect(fallbackEvents).toEqual(["findSymbolDefinitions"]);
	});

	it("wires file-search into bootstrapHarness: Plan Mode, Researcher subagent, secret redaction, RBAC, tenant isolation, and deferred tool_search", async () => {
		const seenIndices: string[] = [];
		const mock = await startMockXerjServer((req, res) => {
			if (req.url.includes("/_search")) {
				const idxMatch = req.url.match(/^\/([^/]+)\/_search/);
				if (idxMatch?.[1]) seenIndices.push(decodeURIComponent(idxMatch[1]));
				res.writeHead(200, { "content-type": "application/json" });
				res.end(
					JSON.stringify({
						took: 3,
						hits: {
							total: { value: 1 },
							hits: [
								{
									_id: "1",
									_score: 12.2,
									match_line: 18,
									_source: {
										title: "auth.ts",
										ax_path: "src/transport/auth.ts",
										language: "typescript",
										ax_format: "ts",
										symbol_count: 3,
										symbols: [{ name: "authFromEnv", kind: "function", line: 18 }],
									},
									fields: {
										_passage: [
											{
												field: "body",
												start_line: 18,
												text: "export function authFromEnv() { const leaked = 'ghp_1234567890abcdef1234567890abcdef12'; }",
											},
										],
									},
								},
							],
						},
					}),
				);
				return;
			}
			res.writeHead(200, { "content-type": "application/json" });
			res.end(JSON.stringify({ version: { number: "8.13.0" } }));
		});
		cleanups.push(() => mock.close());

		const xerj = new XerjFileSearchService({ baseUrl: mock.url, scopeMode: "tenant" });

		// 1. Default harness without fileSearch does NOT register file-search pack (zero overhead)
		const bare = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
			sleepCycleScheduler: false,
		});
		try {
			expect(bare.harness.listToolPacks().some((p) => p.name === "file-search")).toBe(false);
		} finally {
			await bare.close();
		}

		// 2. Harness with XERJ fileSearch registers `file-search`, works in Plan Mode, redacts secrets, enforces RBAC & tenant isolation
		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
			fileSearch: xerj,
			sleepCycleScheduler: false,
		});
		cleanups.push(() => rt.close());

		expect(rt.harness.listToolPacks().some((p) => p.name === "file-search" && p.enabled)).toBe(true);

		// Plan Mode allows file_search + file_symbol_def, and guard-shaping redacts leaked secrets from snippets!
		const convPlan = await rt.harness.createConversation({ tenant: TENANT_A, mode: "plan" });
		rt.testModel.enqueueMainMany([
			{
				type: "tool_calls",
				calls: [
					{ id: "fs1", name: "file_search", arguments: { query: "authFromEnv" } },
					{ id: "fs2", name: "file_symbol_def", arguments: { symbol: "authFromEnv", kind: "function" } },
				],
			},
			{ type: "text", text: "authFromEnv is defined at src/transport/auth.ts:18." },
		]);
		const settledPlan = await rt.harness.submitAndWait({
			conversationId: convPlan.conversationId,
			content: "Find where authFromEnv is defined",
		});
		expect(settledPlan.status).toBe("done");
		const transcript = await rt.harness.getTranscript(convPlan.conversationId);
		const fs1 = transcript.find((e) => e.toolCallId === "fs1");
		expect(fs1?.isError).toBe(false);
		expect(fs1?.text).toContain("src/transport/auth.ts:18");
		expect(fs1?.text).not.toContain("ghp_1234567890");
		expect(fs1?.text).toContain("[REDACTED:github_token]");
		expect(fs1?.text).toContain("UNTRUSTED FILE SEARCH RESULTS");

		// Tenant B hits a completely separate index prefix (`ax-beta-inc-payments-*`)
		const convB = await rt.harness.createConversation({ tenant: TENANT_B });
		rt.testModel.enqueueMainMany([
			{ type: "tool_calls", calls: [{ id: "fsb", name: "file_find", arguments: { query: "auth" } }] },
			{ type: "text", text: "Found auth.ts." },
		]);
		await rt.harness.submitAndWait({ conversationId: convB.conversationId, content: "Find auth file" });
		expect(seenIndices).toContain("ax-acme-corp-*");
		expect(seenIndices).toContain("ax-beta-inc-payments-*");

		// RBAC: tenant lacking `files:read` is blocked deterministically
		const convDenied = await rt.harness.createConversation({ tenant: NO_FILES_PERM_TENANT });
		rt.testModel.enqueueMainMany([
			{ type: "tool_calls", calls: [{ id: "fsd", name: "file_find", arguments: { query: "auth" } }] },
			{ type: "text", text: "Blocked by missing files:read permission." },
		]);
		await rt.harness.submitAndWait({ conversationId: convDenied.conversationId, content: "Find auth" });
		const deniedTranscript = await rt.harness.getTranscript(convDenied.conversationId);
		const fsd = deniedTranscript.find((e) => e.toolCallId === "fsd");
		expect(fsd?.isError).toBe(true);
		expect(fsd?.text).toContain('missing permission "files:read"');

		// Researcher subagent inherits narrowed read-only `files:read` and can call `file_find`!
		const convSub = await rt.harness.createConversation({ tenant: TENANT_A });
		let parentTurn = 0;
		let childTurn = 0;
		rt.testModel.setDefaultMainHandler((req) => {
			if (req.effectiveSystemPrompt.includes("RESEARCHER subagent")) {
				childTurn += 1;
				if (childTurn === 1) {
					return { type: "tool_calls", calls: [{ id: "sub_f", name: "file_find", arguments: { query: "auth" } }] };
				}
				return { type: "text", text: "Researcher found src/transport/auth.ts." };
			}
			parentTurn += 1;
			if (parentTurn === 1) {
				return {
					type: "tool_calls",
					calls: [{ id: "p_f", name: "subagent_delegate", arguments: { profile: "researcher", task: "Find auth file" } }],
				};
			}
			return { type: "text", text: "Delegated search found src/transport/auth.ts." };
		});
		const settledSub = await rt.harness.submitAndWait({
			conversationId: convSub.conversationId,
			content: "Use researcher to find auth file",
		});
		expect(settledSub.status).toBe("done");
		expect(settledSub.answerText).toContain("src/transport/auth.ts");

		// 3. Deferred mode hides file-search tools until `tool_search` discovers them
		const rtDeferred = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, fileSearchDeferred: true },
			fileSearch: xerj,
			sleepCycleScheduler: false,
		});
		cleanups.push(() => rtDeferred.close());

		const convDef = await rtDeferred.harness.createConversation({ tenant: TENANT_A });
		rtDeferred.testModel.enqueueMainMany([
			{ type: "tool_calls", calls: [{ id: "td1", name: "tool_search", arguments: { query: "symbol definition" } }] },
			{ type: "tool_calls", calls: [{ id: "td2", name: "file_symbol_def", arguments: { symbol: "authFromEnv" } }] },
			{ type: "text", text: "Found via deferred tool_search." },
		]);
		const settledDef = await rtDeferred.harness.submitAndWait({
			conversationId: convDef.conversationId,
			content: "Find symbol definition for authFromEnv",
		});
		expect(settledDef.status).toBe("done");
		const reqs = rtDeferred.testModel.getRecordedRequests({ modelId: "main" });
		expect(reqs[0]!.offeredToolNames).not.toContain("file_symbol_def");
		expect(reqs[1]!.offeredToolNames).toContain("file_symbol_def");
	});
});
