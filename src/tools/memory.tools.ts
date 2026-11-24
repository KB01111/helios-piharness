import { int, obj, oneOf, opt, str } from "@earendil-works/pi-schema";
import type { MemoryService } from "../contracts/app-capabilities.js";
import type { ToolPackDefinition } from "../harness/registry.js";
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
					output: [{ type: "text", text: `No memories matched query "${args.query}".` }],
					details: { count: 0, ids: [] as string[] },
				};
			}
			const formatted = results
				.map(
					(r, i) =>
						`${i + 1}. [${r.memory.id}] (${r.memory.kind}${r.memory.pinned ? ", pinned" : ""}, conf=${r.memory.confidence.toFixed(2)}) ${r.memory.key}: ${r.memory.content}`,
				)
				.join("\n");
			return {
				output: [{ type: "text", text: formatted }],
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
			const saved = await memory.pinMemory(tenant.tenantId, tenant.userId, {
				key: args.key,
				content: args.content,
				kind: args.kind ?? "preference",
				sourceConversationId: String(api.conversationId),
			});
			return {
				output: [
					{
						type: "text",
						text: `Pinned memory [${saved.id}] (${saved.kind}) ${saved.key}: ${saved.content}`,
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
