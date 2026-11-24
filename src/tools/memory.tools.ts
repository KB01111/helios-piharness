import { int, obj, oneOf, opt, str } from "@earendil-works/pi-schema";
import type { MemoryService } from "../contracts/app-capabilities.js";
import type { ToolPackDefinition } from "../harness/registry.js";
import {
	formatMemoryLine,
	renderUntrustedMemoryBlock,
	sanitizeInlineLabel,
	sanitizeMemoryKey,
	sanitizeMemoryText,
} from "../memory/sanitize.js";
import { createTenantScopedTool, defineToolPack } from "./tool-pack.js";

const MemoryKindSchema = oneOf(["fact", "preference", "procedure", "constraint", "episodic"] as const);

const MemorySearchParams = obj({
	query: str({ description: "Search keywords or phrase to match against long-term memory." }),
	limit: opt(int({ minimum: 1, maximum: 50, description: "Max results to return (default 8)." })),
	kind: opt(MemoryKindSchema),
});

const MemoryPinParams = obj({
	key: str({ description: "Short stable dotted key, e.g. user.editor.theme or user.package_manager." }),
	content: str({ description: "Concise fact, preference, or constraint to pin." }),
	kind: opt(MemoryKindSchema),
});

const MemoryForgetParams = obj({
	memoryId: str({ description: "ID of the long-term memory record to forget." }),
});

/**
 * Native Pi tools wrapping MemoryService (tenant-scoped via closure + ConversationMetaDoc).
 *
 * Notice: `memory_search` returns memories purely as tool output — it NEVER
 * mutates system prompt sections, preserving provider KV-cache prefix stability.
 */
export function createMemoryToolPack(memory: MemoryService): ToolPackDefinition {
	const memorySearch = createTenantScopedTool({
		name: "memory_search",
		description: "Search long-term user memories using full-text search. Safe to replay.",
		parameters: MemorySearchParams,
		replay: "safe",
		execute: async (args, tenant) => {
			const results = await memory.search(tenant.tenantId, tenant.userId, args.query, {
				limit: args.limit ?? 8,
				...(args.kind ? { kind: args.kind } : {}),
			});
			if (results.length === 0) {
				return {
					output: [{ type: "text", text: `No memories matched query "${sanitizeInlineLabel(args.query, 80)}".` }],
					details: { count: 0, ids: [] as string[] },
				};
			}
			const lines = results.map((r, i) => {
				const id = r.memory.id.replace(/[^A-Za-z0-9_-]/g, "");
				return `${i + 1}. [${id}] (conf=${r.memory.confidence.toFixed(2)}) ${formatMemoryLine(r.memory).slice(2)}`;
			});
			return {
				output: [{ type: "text", text: renderUntrustedMemoryBlock(lines) }],
				details: { count: results.length, ids: results.map((r) => r.memory.id) },
			};
		},
	});

	const memoryPin = createTenantScopedTool({
		name: "memory_pin",
		description: "Pin an explicit user fact, preference, or constraint into long-term memory.",
		parameters: MemoryPinParams,
		replay: "unsafe",
		execute: async (args, tenant, api) => {
			// Write-time screen: normalize the key and neutralize instruction-like / role-marker content.
			const key = sanitizeMemoryKey(args.key);
			const content = sanitizeMemoryText(args.content);
			if (!key || !content.text) {
				return {
					isError: true,
					output: [{ type: "text", text: "Refused to pin memory: key and content must be non-empty plain facts." }],
				};
			}
			const saved = await memory.pinMemory(tenant.tenantId, tenant.userId, {
				key,
				content: content.text,
				kind: args.kind ?? "preference",
				sourceConversationId: String(api.conversationId),
			});
			return {
				output: [
					{
						type: "text",
						text: `Pinned memory [${saved.id}] (${saved.kind}) ${saved.key}: ${saved.content}${content.flagged ? " (instruction-like text was neutralized)" : ""}`,
					},
				],
				details: { memoryId: saved.id, key: saved.key },
			};
		},
	});

	const memoryForget = createTenantScopedTool({
		name: "memory_forget",
		description: "Forget (soft-delete) a long-term memory record by ID.",
		parameters: MemoryForgetParams,
		replay: "unsafe",
		execute: async (args, tenant) => {
			const removed = await memory.forgetMemory(tenant.tenantId, tenant.userId, args.memoryId);
			if (!removed) {
				return {
					isError: true,
					output: [{ type: "text", text: `Memory "${args.memoryId}" was not found or already deleted.` }],
				};
			}
			return {
				output: [{ type: "text", text: `Forgot memory "${args.memoryId}".` }],
				details: { memoryId: args.memoryId, deleted: true },
			};
		},
	});

	return defineToolPack({
		name: "memory",
		description: "Tenant-scoped long-term memory search, pinning, and deletion.",
		tools: [memorySearch, memoryPin, memoryForget],
	});
}
