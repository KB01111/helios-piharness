/**
 * Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 client & lazy server registry.
 *
 * Inspired by `pi-mcp-adapter` and `antiproton`:
 * - Lazy server startup & metadata caching (no startup blocking, minimal prompt overhead).
 * - Credentials (`headers`) stay inside the transport and are NEVER surfaced to the model
 *   or returned by status/list endpoints.
 * - Pluggable `McpTransport` supports Streamable HTTP (JSON + SSE) out of the box as well
 *   as custom in-process / stdio / test transports.
 */

export interface McpJsonRpcRequest {
	readonly jsonrpc: "2.0";
	readonly id: number | string;
	readonly method: string;
	readonly params?: Record<string, unknown>;
}

export interface McpJsonRpcResponse {
	readonly jsonrpc: "2.0";
	readonly id: number | string | null;
	readonly result?: unknown;
	readonly error?: {
		readonly code: number;
		readonly message: string;
		readonly data?: unknown;
	};
}

export interface McpTransport {
	send(request: McpJsonRpcRequest, signal?: AbortSignal): Promise<McpJsonRpcResponse>;
	close?(): Promise<void> | void;
}

export interface McpToolAnnotations {
	readonly readOnlyHint?: boolean;
	readonly destructiveHint?: boolean;
	readonly idempotentHint?: boolean;
	readonly openWorldHint?: boolean;
}

export interface McpToolDescriptor {
	readonly name: string;
	readonly description: string;
	readonly inputSchema: Record<string, unknown>;
	readonly annotations?: McpToolAnnotations;
}

export interface McpContentItem {
	readonly type: string;
	readonly text?: string;
	readonly data?: string;
	readonly mimeType?: string;
}

export interface McpToolCallResult {
	readonly content: readonly McpContentItem[];
	readonly isError?: boolean;
	readonly structuredContent?: unknown;
}

export interface McpServerConfig {
	/** Stable identifier used to namespace tools as `mcp_<name>__<tool>`. */
	readonly name: string;
	/** Streamable HTTP / JSON-RPC endpoint URL (required unless custom `transport` is supplied). */
	readonly url?: string;
	/** Optional custom headers (e.g. Authorization). Never logged or exposed to the LLM. */
	readonly headers?: Readonly<Record<string, string>>;
	/** Custom transport (stdio wrapper, in-process server, or test double). */
	readonly transport?: McpTransport;
	/** Per-request timeout in milliseconds (default 30_000). */
	readonly timeoutMs?: number;
	/** Tool metadata cache TTL in milliseconds (default 300_000). */
	readonly cacheTtlMs?: number;
	/** Mark all tools from this server as read-only when server omits `annotations.readOnlyHint`. */
	readonly readOnly?: boolean;
	/** Human description shown in `mcp_list_servers`. */
	readonly description?: string;
}

export interface McpServerSummary {
	readonly name: string;
	readonly description?: string;
	readonly connected: boolean;
	readonly discoveredTools: number;
	readonly readOnly: boolean;
	readonly lastDiscoveredAt?: number;
	readonly lastError?: string;
}

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_CACHE_TTL_MS = 5 * 60_000;
const SAFE_SERVER_NAME = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}$/;
const SAFE_TOOL_NAME = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/;

export function formatMcpToolName(serverName: string, toolName: string): string {
	const cleanServer = serverName.replace(/[^a-zA-Z0-9_-]+/g, "_");
	const cleanTool = toolName.replace(/[^a-zA-Z0-9_-]+/g, "_");
	return `mcp_${cleanServer}__${cleanTool}`;
}

export function validateMcpServerUrl(raw: string): string {
	let parsed: URL;
	try {
		parsed = new URL(raw);
	} catch {
		throw new Error(`Invalid MCP server URL "${raw}"`);
	}
	if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
		throw new Error(`Unsupported MCP server protocol "${parsed.protocol}"; expected http: or https:`);
	}
	return parsed.toString();
}

/** Parses SSE `data:` frames when a Streamable HTTP MCP server replies with `text/event-stream`. */
function parseSseJsonRpc(bodyText: string, expectedId: number | string): McpJsonRpcResponse {
	const lines = bodyText.split(/\r?\n/);
	let fallback: McpJsonRpcResponse | undefined;
	for (const line of lines) {
		if (!line.startsWith("data:")) continue;
		const payload = line.slice(5).trim();
		if (!payload || payload === "[DONE]") continue;
		try {
			const parsed = JSON.parse(payload) as McpJsonRpcResponse;
			if (parsed && parsed.jsonrpc === "2.0") {
				if (parsed.id === expectedId) return parsed;
				fallback = parsed;
			}
		} catch {
			// Skip non-JSON SSE comments/heartbeats
		}
	}
	if (fallback) return fallback;
	throw new Error("MCP Streamable HTTP SSE response contained no JSON-RPC payload");
}

/**
 * Standard MCP Streamable HTTP JSON-RPC 2.0 transport over Node's built-in `fetch`.
 */
export class HttpMcpTransport implements McpTransport {
	private readonly url: string;
	private readonly headers: Readonly<Record<string, string>>;
	private readonly timeoutMs: number;
	private sessionId: string | undefined;

	constructor(options: {
		readonly url: string;
		readonly headers?: Readonly<Record<string, string>>;
		readonly timeoutMs?: number;
	}) {
		this.url = validateMcpServerUrl(options.url);
		this.headers = options.headers ? { ...options.headers } : {};
		this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
	}

	async send(request: McpJsonRpcRequest, signal?: AbortSignal): Promise<McpJsonRpcResponse> {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(new Error(`MCP request timed out after ${this.timeoutMs}ms`)), this.timeoutMs);
		timer.unref?.();

		const onAbort = () => controller.abort(signal?.reason);
		if (signal) {
			if (signal.aborted) {
				clearTimeout(timer);
				throw signal.reason instanceof Error ? signal.reason : new Error("MCP request aborted");
			}
			signal.addEventListener("abort", onAbort, { once: true });
		}

		try {
			const res = await fetch(this.url, {
				method: "POST",
				headers: {
					"content-type": "application/json",
					accept: "application/json, text/event-stream",
					...(this.sessionId ? { "mcp-session-id": this.sessionId } : {}),
					...this.headers,
				},
				body: JSON.stringify(request),
				signal: controller.signal,
			});

			const nextSession = res.headers.get("mcp-session-id");
			if (nextSession) this.sessionId = nextSession;

			if (!res.ok) {
				throw new Error(`MCP HTTP ${res.status}: ${res.statusText || "request failed"}`);
			}

			const contentType = (res.headers.get("content-type") ?? "").toLowerCase();
			const bodyText = await res.text();
			if (contentType.includes("text/event-stream")) {
				return parseSseJsonRpc(bodyText, request.id);
			}
			return JSON.parse(bodyText) as McpJsonRpcResponse;
		} finally {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		}
	}
}

interface ManagedMcpServer {
	readonly config: McpServerConfig;
	readonly transport: McpTransport;
	initialized: boolean;
	tools: McpToolDescriptor[];
	lastDiscoveredAt?: number;
	lastError?: string;
	inFlightDiscovery?: Promise<readonly McpToolDescriptor[]>;
}

/**
 * Manages multiple lazily-initialized MCP servers with metadata caching and single-flight discovery.
 */
export class McpRegistry {
	private readonly servers = new Map<string, ManagedMcpServer>();
	private readonly now: () => number;
	private nextRpcId = 1;

	constructor(options?: { readonly now?: () => number; readonly servers?: readonly McpServerConfig[] }) {
		this.now = options?.now ?? (() => Date.now());
		for (const cfg of options?.servers ?? []) {
			this.registerServer(cfg);
		}
	}

	registerServer(config: McpServerConfig): void {
		if (!SAFE_SERVER_NAME.test(config.name)) {
			throw new Error(`Invalid MCP server name "${config.name}". Must match ${SAFE_SERVER_NAME.source}.`);
		}
		const transport =
			config.transport ??
			(config.url
				? new HttpMcpTransport({
						url: config.url,
						headers: config.headers,
						timeoutMs: config.timeoutMs,
					})
				: undefined);
		if (!transport) {
			throw new Error(`MCP server "${config.name}" requires either "url" or "transport".`);
		}
		this.servers.set(config.name, {
			config,
			transport,
			initialized: false,
			tools: [],
		});
	}

	hasServers(): boolean {
		return this.servers.size > 0;
	}

	listServers(): McpServerSummary[] {
		return Array.from(this.servers.values())
			.map((entry) => ({
				name: entry.config.name,
				...(entry.config.description ? { description: entry.config.description } : {}),
				connected: entry.initialized,
				discoveredTools: entry.tools.length,
				readOnly: entry.config.readOnly === true,
				...(entry.lastDiscoveredAt !== undefined ? { lastDiscoveredAt: entry.lastDiscoveredAt } : {}),
				...(entry.lastError ? { lastError: entry.lastError } : {}),
			}))
			.sort((a, b) => a.name.localeCompare(b.name));
	}

	getCachedTools(serverName?: string): Array<{ readonly serverName: string; readonly tool: McpToolDescriptor }> {
		const out: Array<{ readonly serverName: string; readonly tool: McpToolDescriptor }> = [];
		for (const [name, entry] of this.servers.entries()) {
			if (serverName !== undefined && name !== serverName) continue;
			for (const tool of entry.tools) {
				out.push({ serverName: name, tool });
			}
		}
		return out;
	}

	isToolReadOnly(registeredToolName: string): boolean {
		for (const [serverName, entry] of this.servers.entries()) {
			for (const tool of entry.tools) {
				if (formatMcpToolName(serverName, tool.name) === registeredToolName) {
					return tool.annotations?.readOnlyHint === true || entry.config.readOnly === true;
				}
			}
		}
		return false;
	}

	async discoverTools(
		serverName: string,
		options?: { readonly forceRefresh?: boolean; readonly signal?: AbortSignal },
	): Promise<readonly McpToolDescriptor[]> {
		const entry = this.servers.get(serverName);
		if (!entry) {
			throw new Error(`Unknown MCP server "${serverName}"`);
		}

		const ttl = entry.config.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS;
		if (
			!options?.forceRefresh &&
			entry.lastDiscoveredAt !== undefined &&
			this.now() - entry.lastDiscoveredAt < ttl
		) {
			return entry.tools;
		}

		if (entry.inFlightDiscovery) {
			return entry.inFlightDiscovery;
		}

		const run = (async () => {
			try {
				await this.ensureInitialized(entry, options?.signal);
				const resp = await this.rpc(entry, "tools/list", {}, options?.signal);
				const rawTools = (resp as { tools?: unknown } | undefined)?.tools;
				const parsed: McpToolDescriptor[] = [];
				if (Array.isArray(rawTools)) {
					for (const item of rawTools) {
						if (!item || typeof item !== "object") continue;
						const rec = item as Record<string, unknown>;
						if (typeof rec.name !== "string" || !SAFE_TOOL_NAME.test(rec.name)) continue;
						const inputSchema =
							rec.inputSchema && typeof rec.inputSchema === "object" && !Array.isArray(rec.inputSchema)
								? (rec.inputSchema as Record<string, unknown>)
								: { type: "object", properties: {} };
						const ann =
							rec.annotations && typeof rec.annotations === "object" && !Array.isArray(rec.annotations)
								? (rec.annotations as McpToolAnnotations)
								: undefined;
						parsed.push({
							name: rec.name,
							description: typeof rec.description === "string" ? rec.description : `MCP tool ${rec.name}`,
							inputSchema,
							...(ann ? { annotations: ann } : {}),
						});
					}
				}
				entry.tools = parsed;
				entry.lastDiscoveredAt = this.now();
				entry.lastError = undefined;
				return entry.tools;
			} catch (error) {
				entry.lastError = error instanceof Error ? error.message : String(error);
				throw error;
			} finally {
				entry.inFlightDiscovery = undefined;
			}
		})();

		entry.inFlightDiscovery = run;
		return run;
	}

	async callTool(
		serverName: string,
		toolName: string,
		args: Record<string, unknown>,
		signal?: AbortSignal,
	): Promise<McpToolCallResult> {
		const entry = this.servers.get(serverName);
		if (!entry) {
			throw new Error(`Unknown MCP server "${serverName}"`);
		}
		await this.ensureInitialized(entry, signal);
		const result = await this.rpc(
			entry,
			"tools/call",
			{
				name: toolName,
				arguments: args,
			},
			signal,
		);
		const rec = (result && typeof result === "object" ? result : {}) as Record<string, unknown>;
		const rawContent = Array.isArray(rec.content) ? (rec.content as McpContentItem[]) : [];
		return {
			content: rawContent,
			...(rec.isError === true ? { isError: true } : {}),
			...(rec.structuredContent !== undefined ? { structuredContent: rec.structuredContent } : {}),
		};
	}

	async close(): Promise<void> {
		for (const entry of this.servers.values()) {
			await entry.transport.close?.();
		}
	}

	private async ensureInitialized(entry: ManagedMcpServer, signal?: AbortSignal): Promise<void> {
		if (entry.initialized) return;
		await this.rpc(
			entry,
			"initialize",
			{
				protocolVersion: "2025-03-26",
				capabilities: {},
				clientInfo: { name: "helios-piharness", version: "0.1.0" },
			},
			signal,
		);
		entry.initialized = true;
	}

	private async rpc(
		entry: ManagedMcpServer,
		method: string,
		params: Record<string, unknown>,
		signal?: AbortSignal,
	): Promise<unknown> {
		const request: McpJsonRpcRequest = {
			jsonrpc: "2.0",
			id: this.nextRpcId++,
			method,
			params,
		};
		const response = await entry.transport.send(request, signal);
		if (response.error) {
			throw new Error(`MCP error (${response.error.code}): ${response.error.message}`);
		}
		return response.result;
	}
}

/** Parses `MCP_SERVERS` JSON env var (`[{ name, url, headers?, timeoutMs?, readOnly?, description? }]`). */
export function parseMcpServersEnv(raw: string | undefined): McpServerConfig[] {
	if (!raw || raw.trim() === "") return [];
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch (error) {
		throw new Error(`Invalid MCP_SERVERS JSON: ${error instanceof Error ? error.message : String(error)}`);
	}
	if (!Array.isArray(parsed)) {
		throw new Error("MCP_SERVERS must be a JSON array of server objects");
	}
	const out: McpServerConfig[] = [];
	for (const item of parsed) {
		if (!item || typeof item !== "object" || Array.isArray(item)) {
			throw new Error("Each MCP_SERVERS entry must be an object");
		}
		const rec = item as Record<string, unknown>;
		if (typeof rec.name !== "string" || typeof rec.url !== "string") {
			throw new Error("Each MCP_SERVERS entry requires string 'name' and 'url'");
		}
		const headers: Record<string, string> | undefined =
			rec.headers && typeof rec.headers === "object" && !Array.isArray(rec.headers)
				? Object.fromEntries(
						Object.entries(rec.headers as Record<string, unknown>)
							.filter(([, v]) => typeof v === "string")
							.map(([k, v]) => [k, v as string]),
					)
				: undefined;
		out.push({
			name: rec.name,
			url: validateMcpServerUrl(rec.url),
			...(headers ? { headers } : {}),
			...(typeof rec.timeoutMs === "number" && rec.timeoutMs > 0 ? { timeoutMs: rec.timeoutMs } : {}),
			...(typeof rec.cacheTtlMs === "number" && rec.cacheTtlMs > 0 ? { cacheTtlMs: rec.cacheTtlMs } : {}),
			...(typeof rec.readOnly === "boolean" ? { readOnly: rec.readOnly } : {}),
			...(typeof rec.description === "string" ? { description: rec.description } : {}),
		});
	}
	return out;
}
