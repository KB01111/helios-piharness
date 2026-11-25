import type { Message } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	createHttpServer,
	DEFAULT_PERMISSIONS,
	InMemoryDomainStore,
	pruneMessagesForContext,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-ctx",
	userId: "user-ctx",
	permissions: [...DEFAULT_PERMISSIONS],
};

describe("Context Pruning, Structured Compaction, and Cached /btw Side Questions", () => {
	it("pruneMessagesForContext leaves messages untouched below threshold and prunes stale/duplicate tool outputs above threshold", () => {
		const sysMsg = { role: "system", content: "STATIC SYSTEM PROMPT", timestamp: 1 } as unknown as Message;
		const oldTool = {
			role: "toolResult",
			toolCallId: "c1",
			toolName: "domain_lookup_record",
			isError: false,
			content: [
				{
					type: "text",
					text: `First line summary\n${"X".repeat(1200)}\nFull output saved as artifact "art_abc12345".`,
				},
			],
			timestamp: 2,
		} as unknown as Message;
		const recentTool = {
			role: "toolResult",
			toolCallId: "c2",
			toolName: "domain_lookup_record",
			isError: false,
			content: [{ type: "text", text: "Recent fresh output".repeat(40) }],
			timestamp: 3,
		} as unknown as Message;

		const messages: Message[] = [sysMsg, oldTool, recentTool];

		// Below threshold -> exact same array reference (zero KV-cache disruption)
		const untouched = pruneMessagesForContext(messages, { thresholdChars: 50_000, keepRecentToolResults: 1 });
		expect(untouched.prunedCount).toBe(0);
		expect(untouched.messages).toBe(messages);

		// Above threshold -> prunes oldTool while keeping sysMsg byte-identical and recentTool verbatim
		const pruned = pruneMessagesForContext(messages, {
			thresholdChars: 500,
			keepRecentToolResults: 1,
			minPrunableChars: 200,
		});
		expect(pruned.prunedCount).toBe(1);
		expect(pruned.bytesSaved).toBeGreaterThan(800);
		expect(pruned.messages[0]).toBe(sysMsg);
		expect(pruned.messages[2]).toBe(recentTool);
		const prunedText = ((pruned.messages[1] as { content: Array<{ text: string }> }).content[0]!).text;
		expect(prunedText).toContain("Pruned earlier domain_lookup_record result");
		expect(prunedText).toContain("art_abc12345");
	});

	it("preserves active PlanDoc steps and offloaded artifact IDs across compaction via beforeCompact hook", async () => {
		const domainStore = new InMemoryDomainStore([
			{ tenantId: TENANT.tenantId, recordId: "big-log", value: "L".repeat(8000), updatedAt: 1000 },
		]);

		const rt = await bootstrapHarness({
			config: {
				piDbPath: ":memory:",
				appDbPath: ":memory:",
				useMockModels: true,
				maxToolOutputBytes: 4096,
				structuredCompactionEnabled: true,
			},
			domainStore,
		});

		try {
			const conv = await rt.harness.createConversation({ tenant: TENANT, mode: "plan" });

			// Turn 1: create a plan and trigger an artifact offload
			rt.testModel.enqueueMainMany([
				{
					type: "tool_calls",
					calls: [
						{ id: "p1", name: "plan_set", arguments: { steps: ["Read big-log", "Analyze errors", "Write fix"] } },
						{ id: "p2", name: "domain_lookup_record", arguments: { recordId: "big-log" } },
					],
				},
				{ type: "text", text: "Started plan and inspected big-log." },
			]);
			await rt.harness.submitAndWait({ conversationId: conv.conversationId, content: "Investigate big-log" });

			// Add enough filler turns to exceed Pi Durable's default keepRecentTokens (20,000 tokens ≈ 80,000 chars)
			for (let i = 0; i < 8; i++) {
				rt.testModel.enqueueMain({ type: "text", text: `Detailed follow-up notes ${i}: ${"analysis-detail ".repeat(700)}` });
				await rt.harness.submitAndWait({
					conversationId: conv.conversationId,
					content: `Follow-up question ${i}: ${"context-block ".repeat(500)}`,
				});
			}

			rt.testModel.enqueueFastJson({
				goal: "Investigate big-log errors and prepare fix",
				decisions: ["Stay in plan mode until root cause is confirmed"],
				openItems: ["Finish step s2 and s3"],
			});

			const compacted = await rt.harness.compactConversation(conv.conversationId, "Preserve plan and artifacts");
			expect(compacted.compacted).toBe(true);

			const transcript = await rt.harness.getTranscript(conv.conversationId);
			const summaryEntry = transcript.find((e) => e.kind === "pi.compaction");
			expect(summaryEntry).toBeDefined();
			const summaryText = JSON.stringify(summaryEntry);
			expect(summaryText).toContain("Active Plan");
			expect(summaryText).toContain("Analyze errors");
			expect(summaryText).toContain("PLAN mode");
			expect(summaryText).toContain("art_");
		} finally {
			await rt.close();
		}
	});

	it("answers /btw side questions using the conversation prompt-cache prefix without running tools or mutating the transcript", async () => {
		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
		});
		try {
			const conv = await rt.harness.createConversation({ tenant: TENANT });
			rt.testModel.enqueueMain({ type: "text", text: "The deployment target is eu-west-1." });
			await rt.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "Where are we deploying?",
			});

			const transcriptBefore = await rt.harness.getTranscript(conv.conversationId);
			const reqBefore = rt.testModel.getRecordedRequests({ modelId: "main" })[0]!;

			// Ask a /btw side question via HTTP endpoint
			rt.testModel.enqueueMain({ type: "text", text: "Side answer: eu-west-1 is in Ireland." });
			const app = createHttpServer(rt.harness, { auth: async () => TENANT });
			const btwRes = await app.request(`/conversations/${conv.conversationId}/btw`, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ question: "Which country is eu-west-1 in?" }),
			});
			expect(btwRes.status).toBe(200);
			const btwBody = (await btwRes.json()) as {
				answerText: string;
				reusedCachePrefix: boolean;
				usage?: { cacheHitRatio?: number };
			};
			expect(btwBody.answerText).toContain("Ireland");
			expect(btwBody.reusedCachePrefix).toBe(true);
			expect(btwBody.usage?.cacheHitRatio).toBeTypeOf("number");

			// Transcript MUST be completely untouched (zero pollution!)
			const transcriptAfter = await rt.harness.getTranscript(conv.conversationId);
			expect(transcriptAfter.length).toBe(transcriptBefore.length);

			// Side request offered zero tools and reused the exact leading system prompt & sessionId!
			const reqs = rt.testModel.getRecordedRequests({ modelId: "main" });
			const btwReq = reqs[reqs.length - 1]!;
			expect(btwReq.offeredToolNames).toEqual([]);
			expect(btwReq.sessionId).toBe(reqBefore.sessionId);
			expect(btwReq.leadingSystemPrompt).toBe(reqBefore.leadingSystemPrompt);

			// Usage endpoint also reports cacheHitRatio
			const usage = await rt.harness.getUsage();
			for (const bucket of Object.values(usage.models)) {
				expect(bucket.cacheHitRatio).toBeTypeOf("number");
			}
		} finally {
			await rt.close();
		}
	});
});
