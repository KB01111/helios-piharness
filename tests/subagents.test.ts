import { describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	DEFAULT_PERMISSIONS,
	InMemoryDomainStore,
	narrowSubagentPermissions,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-sub",
	userId: "user-sub",
	permissions: [...DEFAULT_PERMISSIONS, "admin:usage", "admin:tool-packs"],
};

describe("Durable Subagents & Advisor", () => {
	it("narrowSubagentPermissions never widens permissions and strips admin privileges", () => {
		const readOnlyParent: TenantContext = {
			tenantId: "t",
			userId: "u",
			permissions: ["domain:read", "subagent:spawn"],
		};
		expect(narrowSubagentPermissions(readOnlyParent, "researcher", false)).toEqual(["domain:read"]);
		expect(narrowSubagentPermissions(readOnlyParent, "worker", false)).toEqual(["domain:read"]);

		const adminParent: TenantContext = {
			tenantId: "t",
			userId: "u",
			permissions: ["*"],
		};
		const workerPerms = narrowSubagentPermissions(adminParent, "worker", false);
		expect(workerPerms).not.toContain("*");
		expect(workerPerms).not.toContain("subagent:spawn");
		expect(workerPerms.every((p) => !p.startsWith("admin:"))).toBe(true);
	});

	it("delegates to a durable read-only researcher subagent that cannot mutate state and returns a distilled answer", async () => {
		const domainStore = new InMemoryDomainStore([
			{ tenantId: TENANT.tenantId, recordId: "doc-42", value: "Status is GREEN with key sk-ant-api03-secretsecretsecret1234567890", updatedAt: 1000 },
		]);

		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, subagentMaxDepth: 1 },
			domainStore,
		});

		try {
			const conv = await rt.harness.createConversation({ tenant: TENANT });

			// Route mock main model responses cleanly by checking whether the request is the parent or child
			let parentTurn = 0;
			let childTurn = 0;
			rt.testModel.setDefaultMainHandler((req) => {
				const isResearcher = req.effectiveSystemPrompt.includes("RESEARCHER subagent");
				if (isResearcher) {
					childTurn += 1;
					if (childTurn === 1) {
						return {
							type: "tool_calls",
							calls: [
								{ id: "sub_r", name: "domain_lookup_record", arguments: { recordId: "doc-42" } },
								// Attempt a forbidden mutation AND recursive subagent spawn (both must be blocked!)
								{ id: "sub_w", name: "domain_mutate_record", arguments: { recordId: "doc-42", value: "tampered" } },
								{
									id: "sub_recurse",
									name: "subagent_delegate",
									arguments: { profile: "researcher", task: "nested spawn" },
								},
							],
						};
					}
					return { type: "text", text: "Research findings: doc-42 status is GREEN." };
				}

				parentTurn += 1;
				if (parentTurn === 1) {
					return {
						type: "tool_calls",
						calls: [
							{
								id: "p_del",
								name: "subagent_delegate",
								arguments: { profile: "researcher", task: "Inspect doc-42 and report status" },
							},
						],
					};
				}
				return { type: "text", text: "Subagent confirmed doc-42 is GREEN." };
			});

			const settled = await rt.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "Delegate research on doc-42",
			});

			expect(settled.status).toBe("done");
			expect(settled.answerText).toContain("GREEN");
			// Child researcher could NOT mutate domainStore!
			expect(domainStore.getMutationCount("doc-42")).toBe(0);
			expect(domainStore.getLookupCount("doc-42")).toBe(1);

			const transcript = await rt.harness.getTranscript(conv.conversationId);
			const delegateResult = transcript.find((e) => e.toolCallId === "p_del");
			expect(delegateResult?.isError).toBe(false);
			expect(delegateResult?.text).toContain("[Subagent:researcher");
			expect(delegateResult?.text).toContain("GREEN");
			// Secrets never leak back to parent
			expect(delegateResult?.text).not.toContain("sk-ant-api03-");
		} finally {
			await rt.close();
		}
	});

	it("runs subagent_parallel fan-out concurrently and collects ordered results", async () => {
		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
		});
		try {
			const conv = await rt.harness.createConversation({ tenant: TENANT });
			let parentTurn = 0;

			rt.testModel.setDefaultMainHandler((req) => {
				if (req.effectiveSystemPrompt.includes("RESEARCHER subagent")) {
					return { type: "text", text: `Fact for: ${req.lastUserText}` };
				}
				if (req.effectiveSystemPrompt.includes("REVIEWER subagent")) {
					return { type: "text", text: `Review for: ${req.lastUserText}` };
				}
				parentTurn += 1;
				if (parentTurn === 1) {
					return {
						type: "tool_calls",
						calls: [
							{
								id: "p_par",
								name: "subagent_parallel",
								arguments: {
									tasks: [
										{ label: "branch-a", profile: "researcher", task: "Check A" },
										{ label: "branch-b", profile: "reviewer", task: "Verify B" },
									],
								},
							},
						],
					};
				}
				return { type: "text", text: "Combined both branches." };
			});

			const settled = await rt.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "Run two parallel checks",
			});

			expect(settled.status).toBe("done");
			const transcript = await rt.harness.getTranscript(conv.conversationId);
			const parResult = transcript.find((e) => e.toolCallId === "p_par");
			expect(parResult?.isError).toBe(false);
			expect(parResult?.text).toContain("[branch-a]");
			expect(parResult?.text).toContain("Fact for: Check A");
			expect(parResult?.text).toContain("[branch-b]");
			expect(parResult?.text).toContain("Review for: Verify B");
		} finally {
			await rt.close();
		}
	});

	it("supports advisor_consult for fast read-only second opinions even in plan mode", async () => {
		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
		});
		try {
			const conv = await rt.harness.createConversation({ tenant: TENANT, mode: "plan" });
			rt.testModel.enqueueFastJson({
				recommendation: "revise",
				summary: "Add an idempotency check before mutating.",
				risks: ["Duplicate writes on retry"],
				alternatives: ["Use two-phase staging"],
			});
			rt.testModel.enqueueMainMany([
				{
					type: "tool_calls",
					calls: [
						{
							id: "c_adv",
							name: "advisor_consult",
							arguments: {
								question: "Should I overwrite billing records directly?",
								proposedAction: "Direct UPDATE without idempotency key",
							},
						},
					],
				},
				{ type: "text", text: "Advisor recommended revising to add an idempotency check." },
			]);

			const res = await rt.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "Double-check my billing plan",
			});
			expect(res.status).toBe("done");

			const transcript = await rt.harness.getTranscript(conv.conversationId);
			const advEntry = transcript.find((e) => e.toolCallId === "c_adv");
			expect(advEntry?.isError).toBe(false);
			expect(advEntry?.text).toContain("REVISE");
			expect(advEntry?.text).toContain("Duplicate writes on retry");
		} finally {
			await rt.close();
		}
	});
});
