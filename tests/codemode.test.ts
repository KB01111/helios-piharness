import { describe, expect, it } from "vitest";
import { bootstrapHarness, DEFAULT_PERMISSIONS, type TenantContext } from "../src/index.js";

const TENANT: TenantContext = { tenantId: "t-code", userId: "u-code", permissions: [...DEFAULT_PERMISSIONS] };
const READONLY: TenantContext = { tenantId: "t-ro", userId: "u-ro", permissions: ["memory:read"] };

async function boot(config: Record<string, unknown> = {}) {
	return bootstrapHarness({
		config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, ...config },
		sleepCycleScheduler: false,
	});
}

async function runScript(rt: Awaited<ReturnType<typeof boot>>, tenant: TenantContext, code: string) {
	const conv = await rt.harness.createConversation({ tenant });
	rt.testModel.enqueueMain({ type: "tool_calls", calls: [{ id: "c1", name: "codemode", arguments: { code } }] });
	rt.testModel.enqueueMain({ type: "text", text: "done" });
	const sub = await rt.harness.submit({ conversationId: conv.conversationId, content: "go", requestId: `r-${Math.random()}` });
	const settled = await rt.harness.waitSubmission(sub.submissionId);
	const transcript = await rt.harness.getTranscript(conv.conversationId);
	return { settled, transcript: JSON.stringify(transcript) };
}

describe("codemode", () => {
	it("is installed by default and executes a script that calls tools", async () => {
		const rt = await boot();
		try {
			expect(rt.codemode).toBeDefined();
			const { settled, transcript } = await runScript(
				rt,
				TENANT,
				`const r = await tools.memory_search({ query: "anything" }); text("ran:" + (r === undefined ? "undef" : "ok"));`,
			);
			expect(settled.status).toBe("done");
			expect(transcript).toContain("ran:");
		} finally {
			await rt.close();
		}
	});

	it("keeps tenant permission guards active for nested calls", async () => {
		const rt = await boot();
		try {
			const { transcript } = await runScript(
				rt,
				READONLY,
				`try { await tools.memory_pin({ content: "x" }); text("PINNED"); } catch (e) { text("denied:" + String(e)); }`,
			);
			expect(transcript).not.toContain("PINNED");
			expect(transcript).toContain("denied:");
		} finally {
			await rt.close();
		}
	});

	it("does not expose the human-approval tool to scripts", async () => {
		const rt = await boot();
		try {
			const { transcript } = await runScript(rt, TENANT, `await tools.domain_request_approval({ action: "x", summary: "y" });`);
			expect(transcript).toContain("tools.domain_request_approval does not exist");
		} finally {
			await rt.close();
		}
	});

	it("can be disabled", async () => {
		const rt = await boot({ codemodeEnabled: false });
		try {
			expect(rt.codemode).toBeUndefined();
		} finally {
			await rt.close();
		}
	});
});
