import { describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	createHttpServer,
	DEFAULT_PERMISSIONS,
	formatMcpToolName,
	type McpJsonRpcRequest,
	type McpJsonRpcResponse,
	type McpTransport,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-mcp",
	userId: "user-mcp",
	permissions: [...DEFAULT_PERMISSIONS],
};

function createMockMcpTransport(): {
	readonly transport: McpTransport;
	readonly calls: McpJsonRpcRequest[];
} {
	const calls: McpJsonRpcRequest[] = [];
	const transport: McpTransport = {
		async send(request): Promise<McpJsonRpcResponse> {
			calls.push(request);
			if (request.method === "initialize") {
				return {
					jsonrpc: "2.0",
					id: request.id,
					result: { protocolVersion: "2025-03-26", serverInfo: { name: "mock-docs", version: "1.0.0" } },
				};
			}
			if (request.method === "tools/list") {
				return {
					jsonrpc: "2.0",
					id: request.id,
					result: {
						tools: [
							{
								name: "search_docs",
								description: "Search internal engineering documentation by keyword.",
								inputSchema: {
									type: "object",
									properties: { q: { type: "string", description: "Search query" } },
									required: ["q"],
								},
								annotations: { readOnlyHint: true },
							},
							{
								name: "delete_page",
								description: "Permanently delete a documentation page.",
								inputSchema: {
									type: "object",
									properties: { pageId: { type: "string", description: "Page ID" } },
									required: ["pageId"],
								},
								annotations: { readOnlyHint: false, destructiveHint: true },
							},
						],
					},
				};
			}
			if (request.method === "tools/call") {
				const params = request.params as { name: string; arguments: Record<string, unknown> };
				if (params.name === "search_docs") {
					return {
						jsonrpc: "2.0",
						id: request.id,
						result: {
							content: [
								{
									type: "text",
									text: `Docs match for "${params.arguments.q}": use pi-durable! token=ghp_1234567890abcdef1234567890abcdef12`,
								},
							],
						},
					};
				}
				return {
					jsonrpc: "2.0",
					id: request.id,
					result: { content: [{ type: "text", text: `Deleted ${params.arguments.pageId}` }] },
				};
			}
			return { jsonrpc: "2.0", id: request.id, error: { code: -32601, message: "Method not found" } };
		},
	};
	return { transport, calls };
}

describe("Lazy MCP Adapter & Deferred Tool Discovery", () => {
	it("stays lazy at startup, discovers deferred tools on demand, and routes calls through guard-shaping (plan mode + secret redaction + RBAC)", async () => {
		const mock = createMockMcpTransport();
		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
			mcpServers: [
				{
					name: "docs",
					description: "Internal docs server",
					transport: mock.transport,
					headers: { Authorization: "Bearer super-secret-mcp-token" },
				},
			],
		});

		try {
			// 1. Lazy startup: zero network/RPC calls made until discovery!
			expect(mock.calls.length).toBe(0);
			expect(rt.harness.listMcpServers()).toEqual([
				expect.objectContaining({ name: "docs", connected: false, discoveredTools: 0 }),
			]);
			// Credentials are never exposed via HTTP `/mcp/servers`
			const app = createHttpServer(rt.harness, { auth: async () => TENANT });
			const serversRes = await app.request("/mcp/servers");
			expect(serversRes.status).toBe(200);
			const serversText = await serversRes.text();
			expect(serversText).not.toContain("super-secret-mcp-token");

			// 2. Start conversation in PLAN mode: discover MCP tools lazily, call read-only search_docs (allowed + redacted), and verify delete_page is blocked!
			const conv = await rt.harness.createConversation({ tenant: TENANT, mode: "plan" });
			const searchToolName = formatMcpToolName("docs", "search_docs");
			const deleteToolName = formatMcpToolName("docs", "delete_page");

			rt.testModel.enqueueMainMany([
				{ type: "tool_calls", calls: [{ id: "m1", name: "mcp_discover", arguments: { server: "docs" } }] },
				{
					type: "tool_calls",
					calls: [
						{ id: "m2", name: searchToolName, arguments: { q: "durable" } },
						{ id: "m3", name: deleteToolName, arguments: { pageId: "p-1" } },
					],
				},
				{ type: "text", text: "Found durable docs; deletion blocked in plan mode." },
			]);

			const settled = await rt.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "Discover docs tools and search for durable",
			});
			expect(settled.status).toBe("done");

			// First request did NOT bloat offered tools with undiscovered MCP schemas; second request offered them after mcp_discover!
			const reqs = rt.testModel.getRecordedRequests({ modelId: "main" });
			expect(reqs[0]!.offeredToolNames).not.toContain(searchToolName);
			expect(reqs[1]!.offeredToolNames).toContain(searchToolName);

			const transcript = await rt.harness.getTranscript(conv.conversationId);
			const searchResult = transcript.find((e) => e.toolCallId === "m2");
			expect(searchResult?.isError).toBe(false);
			expect(searchResult?.text).toContain("use pi-durable!");
			// Secret from MCP server output was redacted before reaching the LLM!
			expect(searchResult?.text).not.toContain("ghp_1234567890");
			expect(searchResult?.text).toContain("[REDACTED:github_token]");

			// Destructive MCP tool (`readOnlyHint: false`) was deterministically blocked by Plan Mode!
			const deleteResult = transcript.find((e) => e.toolCallId === "m3");
			expect(deleteResult?.isError).toBe(true);
			expect(deleteResult?.text).toContain("Plan mode active");

			// 3. Verify native `tool_search` BM25 discovery loads deferred MCP tools in a fresh conversation without calling `tools/list` again (metadata cached!)
			const callsBeforeSecondConv = mock.calls.length;
			const conv2 = await rt.harness.createConversation({ tenant: TENANT });
			rt.testModel.clearRecordedRequests();
			rt.testModel.enqueueMainMany([
				{ type: "tool_calls", calls: [{ id: "ts1", name: "tool_search", arguments: { query: "engineering documentation" } }] },
				{ type: "tool_calls", calls: [{ id: "ts2", name: searchToolName, arguments: { q: "cached" } }] },
				{ type: "text", text: "Searched via tool_search." },
			]);
			const settled2 = await rt.harness.submitAndWait({
				conversationId: conv2.conversationId,
				content: "Find docs tool via tool_search",
			});
			expect(settled2.status).toBe("done");
			const reqs2 = rt.testModel.getRecordedRequests({ modelId: "main" });
			expect(reqs2[0]!.offeredToolNames).toContain("tool_search");
			expect(reqs2[0]!.offeredToolNames).not.toContain(searchToolName);
			expect(reqs2[1]!.offeredToolNames).toContain(searchToolName);
			// Only 1 new RPC (`tools/call`), no repeat `initialize` or `tools/list`
			expect(mock.calls.length).toBe(callsBeforeSecondConv + 1);
		} finally {
			await rt.close();
		}
	});
});
