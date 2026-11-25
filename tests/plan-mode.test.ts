import { describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	createHttpServer,
	DEFAULT_PERMISSIONS,
	InMemoryDomainStore,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-plan",
	userId: "user-plan",
	permissions: [...DEFAULT_PERMISSIONS],
};

describe("Runtime-enforced read-only Plan Mode", () => {
	it("blocks mutating tools in plan mode, allows read-only + planning tools, and preserves KV-cache prefix across mode switches", async () => {
		const domainStore = new InMemoryDomainStore([
			{ tenantId: TENANT.tenantId, recordId: "rec-1", value: "initial", updatedAt: 1000 },
		]);

		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
			domainStore,
		});
		try {
			const conv = await rt.harness.createConversation({ tenant: TENANT, mode: "plan" });
			expect(conv.mode).toBe("plan");
			expect(await rt.harness.getConversationMode(conv.conversationId)).toBe("plan");

			// Turn 1 in PLAN mode: read + plan_set succeed, mutation is blocked deterministically
			rt.testModel.enqueueMainMany([
				{
					type: "tool_calls",
					calls: [
						{ id: "c_read", name: "domain_lookup_record", arguments: { recordId: "rec-1" } },
						{ id: "c_plan", name: "plan_set", arguments: { steps: ["Inspect rec-1", "Update rec-1 after approval"] } },
						{ id: "c_mut", name: "domain_mutate_record", arguments: { recordId: "rec-1", value: "hacked" } },
						{ id: "c_pin", name: "memory_pin", arguments: { key: "k", content: "v", kind: "fact" } },
					],
				},
				{ type: "text", text: "Plan created. Waiting for execute mode." },
			]);

			const r1 = await rt.harness.submitAndWait({ conversationId: conv.conversationId, content: "Plan an update to rec-1" });
			expect(r1.status).toBe("done");
			expect(domainStore.getMutationCount("rec-1")).toBe(0);

			const t1 = await rt.harness.getTranscript(conv.conversationId);
			expect(t1.find((e) => e.toolCallId === "c_read")?.isError).toBe(false);
			expect(t1.find((e) => e.toolCallId === "c_plan")?.isError).toBe(false);
			expect(t1.find((e) => e.toolCallId === "c_mut")?.isError).toBe(true);
			expect(t1.find((e) => e.toolCallId === "c_mut")?.text).toContain("Plan mode active");
			expect(t1.find((e) => e.toolCallId === "c_pin")?.isError).toBe(true);

			// Fork while still in plan mode -> fork inherits "plan" mode
			const forkPoint = t1[t1.length - 1]!.id;
			const forked = await rt.harness.forkConversation(conv.conversationId, forkPoint);
			expect(await rt.harness.getConversationMode(forked.conversationId)).toBe("plan");

			// Switch parent to EXECUTE mode via HTTP route
			const app = createHttpServer(rt.harness, { auth: async () => TENANT });
			const modeRes = await app.request(`/conversations/${conv.conversationId}/mode`, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ mode: "execute" }),
			});
			expect(modeRes.status).toBe(200);
			expect(await rt.harness.getConversationMode(conv.conversationId)).toBe("execute");
			// Fork remains in plan mode independently
			expect(await rt.harness.getConversationMode(forked.conversationId)).toBe("plan");

			// Turn 2 in EXECUTE mode: mutation now succeeds!
			rt.testModel.enqueueMainMany([
				{
					type: "tool_calls",
					calls: [{ id: "c_mut_ok", name: "domain_mutate_record", arguments: { recordId: "rec-1", value: "updated" } }],
				},
				{ type: "text", text: "Updated rec-1." },
			]);
			const r2 = await rt.harness.submitAndWait({ conversationId: conv.conversationId, content: "Proceed with the plan" });
			expect(r2.status).toBe("done");
			expect((await domainStore.lookup(TENANT.tenantId, "rec-1"))?.value).toBe("updated");

			// KV-cache prefix stayed 100% byte-identical across the plan -> execute mode switch!
			const reqs = rt.testModel.getRecordedRequests({ modelId: "main" });
			expect(reqs.length).toBeGreaterThanOrEqual(4);
			for (const req of reqs) {
				expect(req.leadingSystemPrompt).toBe(reqs[0]!.leadingSystemPrompt);
				expect(req.systemMessageCount).toBe(1);
			}
		} finally {
			await rt.close();
		}
	});
});
