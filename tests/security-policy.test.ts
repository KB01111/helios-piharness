import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	createHttpServer,
	DEFAULT_PERMISSIONS,
	InMemoryDomainStore,
	redactSecrets,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-sec",
	userId: "user-sec",
	permissions: [...DEFAULT_PERMISSIONS, "admin:audit"],
};

const OTHER_TENANT: TenantContext = {
	tenantId: "tenant-other",
	userId: "user-other",
	permissions: [...DEFAULT_PERMISSIONS],
};

async function waitFor<T>(fn: () => Promise<T | undefined>, timeoutMs = 4000): Promise<T> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		const val = await fn();
		if (val !== undefined) return val;
		await new Promise((r) => setTimeout(r, 25));
	}
	throw new Error("Timed out waiting for condition");
}

describe("Secret redaction and three-tier permission policy", () => {
	const tmpDirs: string[] = [];
	afterEach(() => {
		for (const dir of tmpDirs.splice(0)) {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	it("redacts API keys, JWTs, PEM private keys, bearer headers, DB URLs, and generic secret assignments", () => {
		const raw = [
			"anthropic=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890",
			"github=ghp_1234567890abcdefghij1234567890abcd",
			"aws=AKIAIOSFODNN7EXAMPLE",
			"auth: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
			"db=postgres://admin:superSecretPass99@db.internal:5432/prod",
			"api_key=my_custom_secret_value_12345",
			"-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAtestkeydata12345\n-----END RSA PRIVATE KEY-----",
		].join("\n");

		const res = redactSecrets(raw);
		expect(res.redactedCount).toBeGreaterThanOrEqual(6);
		expect(res.categories).toEqual(
			expect.arrayContaining(["anthropic_key", "github_token", "aws_access_key", "db_credentials", "private_key"]),
		);
		expect(res.text).not.toContain("sk-ant-api03-");
		expect(res.text).not.toContain("ghp_1234567890");
		expect(res.text).not.toContain("AKIAIOSFODNN7EXAMPLE");
		expect(res.text).not.toContain("superSecretPass99");
		expect(res.text).not.toContain("my_custom_secret_value_12345");
		expect(res.text).not.toContain("MIIEpAIBAAKCAQEAtestkeydata12345");

		// Idempotent: re-redacting already redacted text changes nothing
		const second = redactSecrets(res.text);
		expect(second.redactedCount).toBe(0);
		expect(second.text).toBe(res.text);
	});

	it("redacts secrets in tool outputs BEFORE storing to offloaded artifacts and returning to the model", async () => {
		const secretToken = "ghp_ABCDEF1234567890abcdef1234567890ABCD";
		const bigSecretDoc = `${"A".repeat(4000)} token=${secretToken} ${"B".repeat(4000)}`;
		const domainStore = new InMemoryDomainStore([
			{ tenantId: TENANT.tenantId, recordId: "secret-doc", value: bigSecretDoc, updatedAt: 1000 },
		]);

		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, maxToolOutputBytes: 4096 },
			domainStore,
		});
		try {
			const conv = await rt.harness.createConversation({ tenant: TENANT });
			rt.testModel.enqueueMainMany([
				{ type: "tool_calls", calls: [{ id: "c_sec", name: "domain_lookup_record", arguments: { recordId: "secret-doc" } }] },
				{ type: "text", text: "Inspected secret-doc safely." },
			]);

			const res = await rt.harness.submitAndWait({ conversationId: conv.conversationId, content: "Read secret-doc" });
			expect(res.status).toBe("done");

			const transcript = await rt.harness.getTranscript(conv.conversationId);
			const toolEntry = transcript.find((e) => e.toolCallId === "c_sec");
			expect(toolEntry).toBeDefined();
			expect(toolEntry!.text).not.toContain(secretToken);

			// Verify offloaded artifact ALSO contains the redacted text (never raw secret)
			const match = toolEntry!.text?.match(/artifact "(art_[^"]+)"/);
			expect(match?.[1]).toBeTruthy();
			const artifactSlice = await rt.capabilities.artifacts.get(TENANT.tenantId, TENANT.userId, match![1]!, {
				offset: 0,
				limit: 20_000,
			});
			expect(artifactSlice?.chunk).not.toContain(secretToken);
			expect(artifactSlice?.chunk).toContain("[REDACTED:github_token]");
		} finally {
			await rt.close();
		}
	});

	it("enforces three-tier allow/ask/deny policy with durable beforeTool approval parking across restarts and audit log", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "helios-gate-"));
		tmpDirs.push(dir);
		const piDbPath = path.join(dir, "pi.sqlite");
		const appDbPath = path.join(dir, "app.sqlite");

		const domainStore = new InMemoryDomainStore([
			{ tenantId: TENANT.tenantId, recordId: "r1", value: "v1", updatedAt: 1000 },
		]);

		const permissionPolicy = {
			rules: [
				{ tool: "memory_forget", tier: "deny" as const, reason: "Forgetting memories is disabled by compliance policy." },
				{ tool: "domain_mutate_record", tier: "ask" as const, reason: "Mutations require human sign-off." },
			],
		};

		const rt1 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, sleepCycleSchedulerEnabled: false },
			domainStore,
			permissionPolicy,
		});

		let conversationId = "";
		let submissionId = "";
		let pendingApprovalTaskId = "";
		let pendingSignalName = "";
		let pendingSignalKey = "";

		try {
			const conv = await rt1.harness.createConversation({ tenant: TENANT });
			conversationId = conv.conversationId;

			// 1. Test `deny` tier blocks unconditionally
			rt1.testModel.enqueueMainMany([
				{ type: "tool_calls", calls: [{ id: "c_deny", name: "memory_forget", arguments: { memoryId: "mem-1" } }] },
				{ type: "text", text: "Blocked by compliance." },
			]);
			const deniedRun = await rt1.harness.submitAndWait({ conversationId, content: "Forget k1" });
			expect(deniedRun.status).toBe("done");
			const t1 = await rt1.harness.getTranscript(conversationId);
			expect(t1.find((e) => e.toolCallId === "c_deny")?.text).toContain("compliance policy");

			// 2. Test `ask` tier durably parks tool execution BEFORE mutating domainStore
			rt1.testModel.enqueueMain({
				type: "tool_calls",
				calls: [{ id: "c_ask", name: "domain_mutate_record", arguments: { recordId: "r1", value: "v2" } }],
			});
			const sub = await rt1.harness.submit({ conversationId, content: "Update r1 to v2" });
			submissionId = sub.submissionId;

			const pending = await waitFor(async () => {
				const list = await rt1.harness.listPendingApprovals(conversationId, { tenant: TENANT });
				return list.find((a) => a.toolName === "domain_mutate_record");
			});
			pendingApprovalTaskId = pending.taskId;
			pendingSignalName = pending.signalName;
			pendingSignalKey = pending.signalKey;

			// Mutation must NOT have run yet!
			expect(domainStore.getMutationCount("r1")).toBe(0);
			// Other tenants cannot see our pending gate approval
			expect(await rt1.harness.listPendingApprovals(undefined, { tenant: OTHER_TENANT })).toEqual([]);
		} finally {
			// Simulate process crash / restart while parked at beforeTool gate
			await rt1.close();
		}

		// Reopen on the same SQLite files
		const rt2 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, sleepCycleSchedulerEnabled: false },
			domainStore,
			permissionPolicy,
		});
		try {
			// Pending gate approval survives restart
			const afterReopen = await rt2.harness.listPendingApprovals(conversationId, { tenant: TENANT });
			expect(afterReopen.some((a) => a.taskId === pendingApprovalTaskId)).toBe(true);

			// Deliver approval signal and let the tool execute + finish the run
			rt2.testModel.enqueueMain({ type: "text", text: "Record r1 updated to v2 after approval." });
			const delivered = await rt2.harness.deliverSignal({
				taskId: pendingApprovalTaskId,
				signalName: pendingSignalName,
				key: pendingSignalKey,
				payload: { approved: true, reviewer: "alice@ops", note: "LGTM" },
			});
			expect(delivered.status).toBe("delivered");

			const settled = await rt2.harness.waitSubmission(submissionId);
			expect(settled.status).toBe("done");
			expect(domainStore.getMutationCount("r1")).toBe(1);
			expect((await domainStore.lookup(TENANT.tenantId, "r1"))?.value).toBe("v2");

			// Pending gate approval is cleared after completion
			expect(await rt2.harness.listPendingApprovals(conversationId, { tenant: TENANT })).toEqual([]);

			// Audit log recorded deny, approval_requested, and approved
			const audit = rt2.harness.listPermissionAudit(TENANT.tenantId, { conversationId });
			expect(audit.some((a) => a.toolName === "memory_forget" && a.decision === "blocked")).toBe(true);
			expect(audit.some((a) => a.toolName === "domain_mutate_record" && a.decision === "approval_requested")).toBe(true);
			expect(audit.some((a) => a.toolName === "domain_mutate_record" && a.decision === "approved" && a.reviewer === "alice@ops")).toBe(true);

			// HTTP audit route enforces admin:audit permission and tenant scoping
			const app = createHttpServer(rt2.harness, {
				auth: async (req) => (req.headers.get("authorization") === "Bearer admin" ? TENANT : OTHER_TENANT),
			});
			const forbidden = await app.request("/audit/permissions", { headers: { authorization: "Bearer other" } });
			expect(forbidden.status).toBe(403);

			const ok = await app.request(`/audit/permissions?conversationId=${conversationId}`, {
				headers: { authorization: "Bearer admin" },
			});
			expect(ok.status).toBe(200);
			const body = (await ok.json()) as { entries: Array<{ decision: string }> };
			expect(body.entries.length).toBeGreaterThanOrEqual(3);
		} finally {
			await rt2.close();
		}
	});
});
