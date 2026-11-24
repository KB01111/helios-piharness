import path from "node:path";
import { describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	DEFAULT_PERMISSIONS,
	InMemoryDomainStore,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-kv",
	userId: "user-kv",
	permissions: [...DEFAULT_PERMISSIONS],
};

describe("KV-cache system prompt stability, guardrails, and 6KB artifact offloading", () => {
	it("keeps the leading system prompt prefix byte-identical across multi-turn chats and memory_search calls", async () => {
		const runtime = await bootstrapHarness({
			config: {
				piDbPath: ":memory:",
				appDbPath: ":memory:",
				useMockModels: true,
				skillsDir: path.resolve("skills"),
			},
		});

		try {
			// Seed initial Core Memory BEFORE starting the conversation
			await runtime.capabilities.memory.pinMemory(TENANT.tenantId, TENANT.userId, {
				key: "user.editor",
				content: "Prefers Neovim with Lua config",
				kind: "preference",
			});

			const conv = await runtime.harness.createConversation({ tenant: TENANT });

			// Turn 1: triggers memory_search, then answers
			runtime.testModel.enqueueMainMany([
				{
					type: "tool_calls",
					calls: [{ id: "c1", name: "memory_search", arguments: { query: "Neovim" } }],
				},
				{ type: "text", text: "You prefer Neovim with Lua config." },
			]);
			const r1 = await runtime.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "What editor do I prefer?",
			});
			expect(r1.status).toBe("done");

			// Mutate SQLite long-term memory mid-session (outside & via tool)
			await runtime.capabilities.memory.pinMemory(TENANT.tenantId, TENANT.userId, {
				key: "user.language",
				content: "Prefers TypeScript strict mode",
				kind: "preference",
			});

			// Turn 2: calls memory_pin and memory_search to confirm live search works while prompt stays frozen
			runtime.testModel.enqueueMainMany([
				{
					type: "tool_calls",
					calls: [
						{ id: "c2", name: "memory_pin", arguments: { key: "user.shell", content: "zsh", kind: "preference" } },
						{ id: "c3", name: "memory_search", arguments: { query: "TypeScript" } },
					],
				},
				{ type: "text", text: "Pinned zsh and found your TypeScript preference." },
			]);
			const r2 = await runtime.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "Remember I use zsh and check my language preference.",
			});
			expect(r2.status).toBe("done");

			// Turn 3: another normal turn
			runtime.testModel.enqueueMain({ type: "text", text: "All set!" });
			const r3 = await runtime.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "Thanks!",
			});
			expect(r3.status).toBe("done");

			const requests = runtime.testModel.getRecordedRequests({ modelId: "main" });
			expect(requests.length).toBeGreaterThanOrEqual(5);

			const baselinePrefix = requests[0]!.leadingSystemPrompt;
			expect(baselinePrefix.length).toBeGreaterThan(50);
			expect(baselinePrefix).toContain("Neovim");
			expect(baselinePrefix).toContain("incident-triage");
			// Mid-session additions must NOT mutate the frozen prefix of this conversation
			expect(baselinePrefix).not.toContain("TypeScript strict mode");
			expect(baselinePrefix).not.toContain("zsh");

			for (const req of requests) {
				expect(req.leadingSystemPrompt).toBe(baselinePrefix);
				expect(req.effectiveSystemPrompt).toBe(baselinePrefix);
				expect(req.systemMessageCount).toBe(1);
			}

			// Verify live memory_search DID return the mid-session TypeScript memory in the tool result
			const transcript = await runtime.harness.getTranscript(conv.conversationId);
			const tsSearch = transcript.find(
				(e) => e.role === "toolResult" && e.toolCallId === "c3" && e.text?.includes("TypeScript strict mode"),
			);
			expect(tsSearch).toBeDefined();

			// A brand-new conversation created afterward freezes the updated Core Memory
			const nextConv = await runtime.harness.createConversation({ tenant: TENANT });
			runtime.testModel.clearRecordedRequests();
			runtime.testModel.enqueueMain({ type: "text", text: "Hello again" });
			await runtime.harness.submitAndWait({ conversationId: nextConv.conversationId, content: "Hi" });
			const nextReq = runtime.testModel.getRecordedRequests({ modelId: "main" })[0]!;
			expect(nextReq.leadingSystemPrompt).toContain("TypeScript strict mode");
			expect(nextReq.leadingSystemPrompt).toContain("zsh");
		} finally {
			await runtime.close();
		}
	});

	it("caps tool outputs over 6KB, offloads full content to artifact store, and enforces authz blocks", async () => {
		const bigPayload = "X".repeat(12_000) + "::END_MARKER_999";
		const domainStore = new InMemoryDomainStore([
			{ tenantId: TENANT.tenantId, recordId: "huge-doc", value: bigPayload, updatedAt: 1_000 },
		]);

		const runtime = await bootstrapHarness({
			config: {
				piDbPath: ":memory:",
				appDbPath: ":memory:",
				useMockModels: true,
				maxToolOutputBytes: 6144,
			},
			domainStore,
		});

		try {
			const readOnlyTenant: TenantContext = {
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				permissions: ["domain:read", "artifacts:read"],
			};
			const conv = await runtime.harness.createConversation({ tenant: readOnlyTenant });

			let capturedArtifactId = "";

			runtime.testModel.enqueueMainMany([
				{
					type: "tool_calls",
					calls: [
						{ id: "c_big", name: "domain_lookup_record", arguments: { recordId: "huge-doc" } },
						{ id: "c_denied", name: "domain_mutate_record", arguments: { recordId: "huge-doc", value: "hack" } },
					],
				},
				async () => {
					const entries = await runtime.harness.getTranscript(conv.conversationId);
					const bigResult = entries.find((e) => e.toolCallId === "c_big");
					const match = bigResult?.text?.match(/artifact "(art_[^"]+)"/);
					capturedArtifactId = match?.[1] ?? "";
					return {
						type: "tool_calls",
						calls: [
							{
								id: "c_art",
								name: "artifact_read",
								arguments: { artifactId: capturedArtifactId, offset: 11_900, limit: 500 },
							},
						],
					};
				},
				{ type: "text", text: "Read tail of artifact and respected auth denial." },
			]);

			const settled = await runtime.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "Inspect huge-doc and try writing it",
			});

			expect(settled.status).toBe("done");
			expect(capturedArtifactId).toMatch(/^art_/);

			const transcript = await runtime.harness.getTranscript(conv.conversationId);
			const bigToolResult = transcript.find((e) => e.toolCallId === "c_big");
			expect(bigToolResult).toBeDefined();
			expect(Buffer.byteLength(bigToolResult!.text ?? "", "utf8")).toBeLessThanOrEqual(6144 + 512);
			expect(bigToolResult!.text).toContain("Output truncated");

			const deniedResult = transcript.find((e) => e.toolCallId === "c_denied");
			expect(deniedResult?.isError).toBe(true);
			expect(deniedResult?.text).toContain("Authorization denied");

			const artReadResult = transcript.find((e) => e.toolCallId === "c_art");
			expect(artReadResult?.isError).toBe(false);
			expect(artReadResult?.text).toContain("END_MARKER_999");
		} finally {
			await runtime.close();
		}
	});
});
