import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	DEFAULT_PERMISSIONS,
	InMemoryDomainStore,
	SingleWriterLockError,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-crash",
	userId: "user-1",
	permissions: [...DEFAULT_PERMISSIONS],
};

function makeTempDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "helios-crash-test-"));
}

function deferred<T = void>(): { promise: Promise<T>; resolve: (value: T) => void } {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((res) => {
		resolve = res;
	});
	return { promise, resolve };
}

describe("Crash recovery & single-writer durability", () => {
	const tempDirs: string[] = [];

	afterEach(() => {
		for (const dir of tempDirs.splice(0)) {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	it("reruns a replay:'safe' tool after closing and reopening the Harness mid-execution", async () => {
		const dir = makeTempDir();
		tempDirs.push(dir);
		const piDbPath = path.join(dir, "pi.sqlite");
		const appDbPath = path.join(dir, "app.sqlite");

		const domainStore = new InMemoryDomainStore([
			{ tenantId: TENANT.tenantId, recordId: "rec-1", value: "alpha-42", updatedAt: 1_000 },
		]);

		const lookupStarted = deferred<void>();
		domainStore.setHooks({
			onLookupStart: async (_recordId, attempt, signal) => {
				if (attempt === 1) {
					lookupStarted.resolve();
					// Hang until Harness close aborts this invocation
					await new Promise<void>((_resolve, reject) => {
						if (signal?.aborted) {
							reject(new Error("Simulated process crash"));
							return;
						}
						signal?.addEventListener("abort", () => reject(new Error("Simulated process crash")), { once: true });
					});
				}
			},
		});

		const runtime1 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, skillsDir: path.join(dir, "skills") },
			domainStore,
		});

		const conv = await runtime1.harness.createConversation({ tenant: TENANT });
		runtime1.testModel.enqueueMain({
			type: "tool_calls",
			calls: [{ id: "call_lookup_1", name: "domain_lookup_record", arguments: { recordId: "rec-1" } }],
		});

		const submission = await runtime1.harness.submit({
			conversationId: conv.conversationId,
			content: "Look up rec-1",
			requestId: "req-safe-1",
		});

		await lookupStarted.promise;
		// Close mid-tool-execution (simulating abrupt stop before result commit)
		await runtime1.close();

		expect(domainStore.getLookupCount("rec-1")).toBe(1);

		// Reopen over the exact same SQLite files with normal execution restored
		domainStore.setHooks({});
		const runtime2 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, skillsDir: path.join(dir, "skills") },
			domainStore,
			autoResume: false,
		});

		try {
			runtime2.testModel.enqueueMain({
				type: "text",
				text: "Recovered value: alpha-42",
			});

			// Idempotent retry with same requestId returns the exact same submission
			const duplicate = await runtime2.harness.submit({
				conversationId: conv.conversationId,
				content: "Look up rec-1",
				requestId: "req-safe-1",
			});
			expect(duplicate.submissionId).toBe(submission.submissionId);

			const settled = await runtime2.harness.waitSubmission(submission.submissionId);
			expect(settled.status).toBe("done");
			expect(settled.answerText).toBe("Recovered value: alpha-42");

			// Safe tool was replayed cleanly on reopen
			expect(domainStore.getLookupCount("rec-1")).toBe(2);

			const transcript = await runtime2.harness.getTranscript(conv.conversationId);
			const toolResult = transcript.find((e) => e.role === "toolResult" && e.toolName === "domain_lookup_record");
			expect(toolResult).toBeDefined();
			expect(toolResult?.isError).toBe(false);
			expect(toolResult?.text).toContain("alpha-42");
		} finally {
			await runtime2.close();
		}
	});

	it("does NOT re-execute a replay:'unsafe' side-effect tool after reopen and reports interruption", async () => {
		const dir = makeTempDir();
		tempDirs.push(dir);
		const piDbPath = path.join(dir, "pi.sqlite");
		const appDbPath = path.join(dir, "app.sqlite");

		const domainStore = new InMemoryDomainStore();
		const mutateStarted = deferred<void>();

		domainStore.setHooks({
			onMutateStart: async (_recordId, attempt, signal) => {
				if (attempt === 1) {
					mutateStarted.resolve();
					await new Promise<void>((_resolve, reject) => {
						if (signal?.aborted) {
							reject(new Error("Simulated crash mid side-effect"));
							return;
						}
						signal?.addEventListener(
							"abort",
							() => reject(new Error("Simulated crash mid side-effect")),
							{ once: true },
						);
					});
				}
			},
		});

		const runtime1 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, skillsDir: path.join(dir, "skills") },
			domainStore,
		});

		const conv = await runtime1.harness.createConversation({ tenant: TENANT });
		runtime1.testModel.enqueueMain({
			type: "tool_calls",
			calls: [
				{
					id: "call_mut_1",
					name: "domain_mutate_record",
					arguments: { recordId: "rec-unsafe", value: "charge-100" },
				},
			],
		});

		const submission = await runtime1.harness.submit({
			conversationId: conv.conversationId,
			content: "Update rec-unsafe",
			requestId: "req-unsafe-1",
		});

		await mutateStarted.promise;
		await runtime1.close();

		expect(domainStore.getMutationCount("rec-unsafe")).toBe(1);

		domainStore.setHooks({});
		const runtime2 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, skillsDir: path.join(dir, "skills") },
			domainStore,
			autoResume: false,
		});

		try {
			runtime2.testModel.enqueueMain({
				type: "text",
				text: "Detected interrupted mutation; did not retry blindly.",
			});

			const settled = await runtime2.harness.waitSubmission(submission.submissionId);
			expect(settled.status).toBe("done");
			expect(settled.answerText).toBe("Detected interrupted mutation; did not retry blindly.");

			// Unsafe tool MUST NOT re-execute automatically on recovery!
			expect(domainStore.getMutationCount("rec-unsafe")).toBe(1);

			const transcript = await runtime2.harness.getTranscript(conv.conversationId);
			const toolResult = transcript.find((e) => e.role === "toolResult" && e.toolName === "domain_mutate_record");
			expect(toolResult).toBeDefined();
			expect(toolResult?.isError).toBe(true);
			expect(toolResult?.text?.toLowerCase()).toContain("interrupted");
		} finally {
			await runtime2.close();
		}
	});

	it("enforces single-writer lock on the same SQLite DB path", async () => {
		const dir = makeTempDir();
		tempDirs.push(dir);
		const piDbPath = path.join(dir, "locked.sqlite");
		const appDbPath = path.join(dir, "app.sqlite");

		const runtime1 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, skillsDir: path.join(dir, "skills") },
		});

		try {
			await expect(
				bootstrapHarness({
					config: { piDbPath, appDbPath: path.join(dir, "app2.sqlite"), useMockModels: true },
				}),
			).rejects.toThrow(SingleWriterLockError);
		} finally {
			await runtime1.close();
		}

		// Lock is released on close, so reopening succeeds
		const runtime2 = await bootstrapHarness({
			config: { piDbPath, appDbPath, useMockModels: true, skillsDir: path.join(dir, "skills") },
		});
		await runtime2.close();
	});
});
