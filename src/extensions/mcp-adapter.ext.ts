import { defineExtension, defineTool, type Extension, type ToolRegistration } from "@earendil-works/pi-durable";
import { createToolSearchExtension } from "@earendil-works/pi-durable/tool-search";
import { bool, int, obj, opt, str } from "@earendil-works/pi-schema";
import type { HarnessRegistryManager } from "../harness/registry.js";
import {
	formatMcpToolName,
	type McpRegistry,
	type McpToolDescriptor,
} from "../mcp/mcp-client.js";

/**
 * Lazy MCP Adapter + Native Deferred Tool Search Extension.
 *
 * Inspired by `pi-mcp-adapter` and `@earendil-works/pi-durable/tool-search`:
 * - Connecting many MCP servers does NOT bloat the initial system prompt or tool schema list.
 * - Discovered MCP tools are dynamically registered into `HarnessRegistryManager` with
 *   `deferred: true` so they stay hidden until discovered via `tool_search` or `mcp_discover`
 *   (which loads them into the conversation's `modelTools` via `control.addTools`).
 * - Every MCP tool execution flows through `guard-shaping` (tenant-spoof block, plan mode
 *   via `readOnlyHint`, RBAC `mcp:read`/`mcp:write`, three-tier permission policy, secret
 *   redaction, and 6KB artifact offloading).
 */

export interface McpAdapterOptions {
	readonly mcpRegistry: McpRegistry;
	readonly registryManager: HarnessRegistryManager;
}

const McpListServersParams = obj({});

const McpDiscoverParams = obj({
	server: opt(str({ description: "Optional MCP server name to discover. Omit to discover all configured servers." })),
	query: opt(str({ description: "Optional keyword filter to immediately load matching tools into this conversation." })),
	limit: opt(int({ minimum: 1, maximum: 20, description: "Max matching tools to load immediately (default 8)." })),
	forceRefresh: opt(bool({ description: "Bypass cached tool metadata and re-query the MCP server." })),
});

function buildDeferredMcpTool(
	mcpRegistry: McpRegistry,
	serverName: string,
	descriptor: McpToolDescriptor,
	serverReadOnly: boolean,
): ToolRegistration {
	const registeredName = formatMcpToolName(serverName, descriptor.name);
	const isReadOnly = descriptor.annotations?.readOnlyHint === true || serverReadOnly;
	const isIdempotent = descriptor.annotations?.idempotentHint === true;

	return defineTool({
		name: registeredName,
		description: `[MCP:${serverName}] ${descriptor.description}`,
		parameters: descriptor.inputSchema,
		deferred: true,
		replay: isReadOnly || isIdempotent ? "safe" : "unsafe",
		execute: async (args, _api, context) => {
			const callResult = await mcpRegistry.callTool(
				serverName,
				descriptor.name,
				(args ?? {}) as Record<string, unknown>,
				context.abortSignal,
			);

			const textParts = callResult.content
				.map((item) => {
					if (item.type === "text" && typeof item.text === "string") return item.text;
					if (item.type === "resource" && typeof item.text === "string") return item.text;
					return JSON.stringify(item);
				})
				.filter((part) => part.length > 0);

			const text =
				textParts.join("\n") ||
				(callResult.structuredContent !== undefined ? JSON.stringify(callResult.structuredContent) : "(empty MCP response)");

			return {
				...(callResult.isError ? { isError: true } : {}),
				output: [{ type: "text", text }],
				details: {
					mcpServer: serverName,
					mcpTool: descriptor.name,
					readOnly: isReadOnly,
				},
			};
		},
	});
}

/** Synchronizes one MCP server's discovered tools into `HarnessRegistryManager` as deferred tools. */
export function syncMcpServerToolsToRegistry(
	mcpRegistry: McpRegistry,
	registryManager: HarnessRegistryManager,
	serverName: string,
	tools: readonly McpToolDescriptor[],
): string[] {
	const serverSummary = mcpRegistry.listServers().find((s) => s.name === serverName);
	const serverReadOnly = serverSummary?.readOnly === true;
	const registered = tools.map((tool) => buildDeferredMcpTool(mcpRegistry, serverName, tool, serverReadOnly));

	const extension = defineExtension({
		name: `mcp:${serverName}`,
		namespace: {
			name: `mcp_${serverName}`,
			description: serverSummary?.description ?? `Deferred MCP tools from server "${serverName}".`,
		},
		tools: registered,
	});
	registryManager.replaceExtension(extension);
	return registered.map((t) => t.name);
}

/**
 * Creates the lightweight MCP discovery extension (`mcp_list_servers`, `mcp_discover`)
 * that lazily connects to configured MCP servers and installs their tools as deferred tools.
 */
export function createMcpAdapterExtension(options: McpAdapterOptions): Extension {
	const { mcpRegistry, registryManager } = options;

	const listServersTool = defineTool({
		name: "mcp_list_servers",
		description:
			"List configured external MCP servers and how many tools have been discovered from each. Call mcp_discover to load a server's tools on demand.",
		parameters: McpListServersParams,
		replay: "safe",
		execute: async () => {
			const servers = mcpRegistry.listServers();
			if (servers.length === 0) {
				return { output: [{ type: "text", text: "No external MCP servers are configured." }] };
			}
			const lines = servers.map((s) => {
				const status = s.connected ? `connected (${s.discoveredTools} tools)` : "lazy (not yet discovered)";
				const ro = s.readOnly ? " [read-only]" : "";
				const desc = s.description ? ` — ${s.description}` : "";
				return `- ${s.name}: ${status}${ro}${desc}`;
			});
			return {
				output: [{ type: "text", text: lines.join("\n") }],
				details: { serverCount: servers.length },
			};
		},
	});

	const discoverTool = defineTool({
		name: "mcp_discover",
		description:
			"Lazily connect to an MCP server (or all configured servers), register its tools as deferred tools, and optionally load matching tools into this conversation immediately.",
		parameters: McpDiscoverParams,
		replay: "safe",
		callers: ["model"],
		execute: async (args, _api, context) => {
			const targets = args.server
				? [args.server]
				: mcpRegistry.listServers().map((s) => s.name);

			if (targets.length === 0) {
				return { output: [{ type: "text", text: "No MCP servers configured to discover." }] };
			}

			const discoveredAll: Array<{ server: string; registeredName: string; description: string }> = [];
			const errors: string[] = [];

			for (const serverName of targets) {
				try {
					const descriptors = await mcpRegistry.discoverTools(serverName, {
						forceRefresh: args.forceRefresh === true,
						signal: context.abortSignal,
					});
					const registeredNames = syncMcpServerToolsToRegistry(mcpRegistry, registryManager, serverName, descriptors);
					for (let i = 0; i < descriptors.length; i++) {
						discoveredAll.push({
							server: serverName,
							registeredName: registeredNames[i]!,
							description: descriptors[i]!.description,
						});
					}
				} catch (error) {
					errors.push(`${serverName}: ${error instanceof Error ? error.message : String(error)}`);
				}
			}

			const queryLower = args.query?.trim().toLowerCase() ?? "";
			const maxLoad = args.limit ?? 8;
			const matched = queryLower
				? discoveredAll.filter(
						(item) =>
							item.registeredName.toLowerCase().includes(queryLower) ||
							item.description.toLowerCase().includes(queryLower),
					)
				: discoveredAll;

			const toLoad = matched.slice(0, maxLoad).map((m) => m.registeredName);
			const lines: string[] = [];
			if (discoveredAll.length > 0) {
				lines.push(`Discovered ${discoveredAll.length} MCP tool(s). Loaded ${toLoad.length} into this conversation:`);
				for (const item of matched.slice(0, maxLoad)) {
					lines.push(`- ${item.registeredName}: ${item.description.split(/\r?\n/)[0]}`);
				}
				if (matched.length > maxLoad) {
					lines.push(`(${matched.length - maxLoad} more available via \`tool_search\`.)`);
				}
			} else {
				lines.push("No MCP tools discovered.");
			}
			if (errors.length > 0) {
				lines.push("Errors:", ...errors.map((e) => `- ${e}`));
			}

			return {
				...(discoveredAll.length === 0 && errors.length > 0 ? { isError: true } : {}),
				output: [{ type: "text", text: lines.join("\n") }],
				details: { discovered: discoveredAll.length, loaded: toLoad },
				...(toLoad.length > 0 ? { control: { addTools: toLoad } } : {}),
			};
		},
	});

	return defineExtension({
		name: "mcp-adapter",
		tools: [listServersTool, discoverTool],
	});
}

/** Re-export Pi Durable's native BM25 deferred `tool_search` extension. */
export { createToolSearchExtension };
