import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	DEFAULT_PERMISSIONS,
	ManualWakeScheduler,
	type PendingApproval,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-approval",
	userId: "user-approval",
	permissions: [...DEFAULT_PERMISSIONS],
};

function makeTempDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "helios-parked-test-"));
}

async function waitForApproval(
	listFn: () => Promise<PendingApproval[]>,
	timeoutMs = 5_000,
): Promise<PendingApproval> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		const approvals = await listFn();
		if (approvals.length > 0) return approvals[0]!;
		await new Promise((r) => setTimeout(r, 25));
	}
	throw new Error("Timed out waiting for parked approval request");
}

describe("Parked approval tool survives Harness reopen and resumes on signal delivery", () => {
	const tempDirs: string[] = [];

	afterEach(() => {
		for (const dir of tempDirs.splice(0)) {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	it("parks on ApprovalDecisionSignal, survives close/reopen, and finishes cleanly on signal delivery", async () => {
		const dir = makeTempDir();
		tempDirs.push(dir);
		const piDbPath = path.join(dir, "pi.sqlite");
		const appDbPath = path.join(dir, "app.sqlite");

		const wake1 = new ManualWakeScheduler();
		const runtime1 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, skillsDir: path.join(dir, "skills") },
			wakeScheduler: wake1,
		});

		const conv = await runtime1.harness.createConversation({ tenant: TENANT });
		runtime1.testModel.enqueueMain({
			type: "tool_calls",
			calls: [
				{
					id: "call_appr_1",
					name: "domain_request_approval",
					arguments: {
						action: "deploy_prod",
						summary: "Deploy release v2.4 to production",
						timeoutMs: 60 * 60 * 1000,
					},
				},
			],
		});

		const submission = await runtime1.harness.submit({
			conversationId: conv.conversationId,
			content: "Please deploy release v2.4 after getting approval.",
			requestId: "req-deploy-1",
		});

		const pending1 = await waitForApproval(() => runtime1.harness.listPendingApprovals(conv.conversationId));
		expect(pending1.action).toBe("deploy_prod");
		expect(pending1.signalName).toBe("app.approval.decision");

		// Wait briefly for the sleep state to publish into wakeScheduler
		await new Promise((r) => setTimeout(r, 50));
		expect(wake1.currentWake().active).toBe(false);
		expect(wake1.currentWake().at).toBeTypeOf("number");

		// Close Harness completely while tool is parked
		await runtime1.close();

		// Reopen fresh Harness over the exact same SQLite files
		const wake2 = new ManualWakeScheduler();
		const runtime2 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, skillsDir: path.join(dir, "skills") },
			wakeScheduler: wake2,
		});

		try {
			// Pending approval survives reopen intact
			const pendingAfterReopen = await runtime2.harness.listPendingApprovals(conv.conversationId);
			expect(pendingAfterReopen).toHaveLength(1);
			expect(pendingAfterReopen[0]!.taskId).toBe(pending1.taskId);
			expect(pendingAfterReopen[0]!.signalKey).toBe(pending1.signalKey);

			// Enqueue assistant turn after approval wakes up
			runtime2.testModel.enqueueMain({
				type: "text",
				text: "Deployment v2.4 completed after Alice's approval.",
			});

			// Deliver signal externally via HarnessPort
			const delivered = await runtime2.harness.deliverSignal({
				taskId: pending1.taskId,
				signalName: pending1.signalName,
				key: pending1.signalKey,
				payload: {
					approved: true,
					reviewer: "Alice",
					note: "Looks good, ship it!",
				},
			});
			expect(delivered.status).toBe("delivered");

			// Duplicate delivery to the same key is reported cleanly
			const duplicate = await runtime2.harness.deliverSignal({
				taskId: pending1.taskId,
				signalName: pending1.signalName,
				key: pending1.signalKey,
				payload: {
					approved: true,
					reviewer: "Alice",
				},
			});
			expect(["duplicate", "closed"]).toContain(duplicate.status);

			// Original submission now settles with final answer
			const settled = await runtime2.harness.waitSubmission(submission.submissionId);
			expect(settled.status).toBe("done");
			expect(settled.answerText).toBe("Deployment v2.4 completed after Alice's approval.");

			// Pending approval doc was cleaned up upon resume
			const remaining = await runtime2.harness.listPendingApprovals(conv.conversationId);
			expect(remaining).toHaveLength(0);

			// Transcript records approval grant
			const transcript = await runtime2.harness.getTranscript(conv.conversationId);
			const approvalResult = transcript.find(
				(e) => e.role === "toolResult" && e.toolName === "domain_request_approval",
			);
			expect(approvalResult?.isError).toBe(false);
			expect(approvalResult?.text).toContain("Approval granted by Alice");
		} finally {
			await runtime2.close();
		}
	});
});
