import { describe, expect, it } from "vitest";
import { bootstrapHarness, DEFAULT_PERMISSIONS, PLAN_LIMITS, type TenantContext } from "../src/index.js";

const TENANT: TenantContext = { tenantId: "t-plan", userId: "u-plan", permissions: [...DEFAULT_PERMISSIONS] };

async function boot() {
	return bootstrapHarness({
		config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
		sleepCycleScheduler: false,
		improvements: false,
	});
}

type Rt = Awaited<ReturnType<typeof boot>>;

/** Runs one scripted tool call and returns the text of its result. */
async function call(rt: Rt, conversationId: string, name: string, args: Record<string, unknown>): Promise<{ text: string; isError: boolean }> {
	rt.testModel.enqueueMain({ type: "tool_calls", calls: [{ name, arguments: args }] });
	rt.testModel.enqueueMain({ type: "text", text: "ok" });
	await rt.harness.submitAndWait({ conversationId, content: `run ${name}` });
	const entries = await rt.harness.getTranscript(conversationId);
	const last = [...entries].reverse().find((e) => e.role === "toolResult" && e.toolName === name)!;
	return { text: last.text ?? "", isError: last.isError === true };
}

describe("planning tool pack", () => {
	it("sets, updates and views a durable plan, and rejects misuse", async () => {
		const rt = await boot();
		try {
			const { conversationId } = await rt.harness.createConversation({ tenant: TENANT });
			expect((await call(rt, conversationId, "plan_view", {})).text).toContain("No plan set");

			const set = await call(rt, conversationId, "plan_set", { steps: ["Find the record", "Update it", "Confirm"] });
			expect(set.text).toContain("[~] s1. Find the record");
			expect(set.text).toContain("[ ] s2. Update it");
			expect(set.text).toContain("revision 1");

			const done = await call(rt, conversationId, "plan_update", { stepId: "s1", status: "done", note: "found rec-1" });
			expect(done.text).toContain("[x] s1. Find the record — found rec-1");
			expect(done.text).toContain("Next: s2");

			const noReason = await call(rt, conversationId, "plan_update", { stepId: "s2", status: "blocked" });
			expect(noReason.isError).toBe(true);
			expect(noReason.text).toContain("needs a note");

			const blocked = await call(rt, conversationId, "plan_update", { stepId: "s2", status: "blocked", note: "write denied" });
			expect(blocked.text).toContain("[!] s2. Update it — write denied");
			expect(blocked.text).toContain("re-plan with plan_set");

			const missing = await call(rt, conversationId, "plan_update", { stepId: "s9", status: "done" });
			expect(missing.isError).toBe(true);
			expect(missing.text).toContain('No step "s9"');

			const view = await call(rt, conversationId, "plan_view", {});
			expect(view.text).toContain("revision 3");
		} finally {
			await rt.close();
		}
	});

	it("bounds plan size and isolates plans per conversation and per fork point", async () => {
		const rt = await boot();
		try {
			const a = await rt.harness.createConversation({ tenant: TENANT });
			const b = await rt.harness.createConversation({ tenant: TENANT });
			const many = Array.from({ length: PLAN_LIMITS.maxSteps }, (_, i) => `step ${i} ${"x".repeat(400)}`);
			const set = await call(rt, a.conversationId, "plan_set", { steps: many });
			expect(set.text.split("\n").filter((l) => /^\[.\] s\d+\./.test(l))).toHaveLength(PLAN_LIMITS.maxSteps);
			expect(Math.max(...set.text.split("\n").map((l) => l.length))).toBeLessThan(PLAN_LIMITS.maxTitleChars + 40);

			expect((await call(rt, b.conversationId, "plan_view", {})).text).toContain("No plan set");

			// A fork sees the plan as of its fork point and then diverges.
			const entries = await rt.harness.getTranscript(a.conversationId);
			const fork = await rt.harness.forkConversation(a.conversationId, entries.at(-1)!.id);
			expect((await call(rt, fork.conversationId, "plan_view", {})).text).toContain("s1.");
			await call(rt, fork.conversationId, "plan_set", { steps: ["only one"] });
			expect((await call(rt, a.conversationId, "plan_view", {})).text).toContain("s12.");
		} finally {
			await rt.close();
		}
	});

	it("is a toggleable tool pack and is reachable from codemode scripts", async () => {
		const rt = await boot();
		try {
			expect(rt.harness.listToolPacks().some((p) => p.name === "planning" && p.enabled)).toBe(true);
			const { conversationId } = await rt.harness.createConversation({ tenant: TENANT });
			const script = `await tools.plan_set({ steps: ["a", "b"] }); const v = await tools.plan_view({}); text(String(v).includes("s2") ? "has-s2" : JSON.stringify(v));`;
			const out = await call(rt, conversationId, "codemode", { code: script });
			expect(out.text).toContain("has-s2");
		} finally {
			await rt.close();
		}
	});
});
