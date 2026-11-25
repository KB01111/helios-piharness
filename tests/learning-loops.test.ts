import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	AppDatabase,
	bootstrapHarness,
	comparePareto,
	createHttpServer,
	DEFAULT_PERMISSIONS,
	detectFrictionSignals,
	loadSkillCatalog,
	ReplayEvalRunner,
	SimLog,
	SkillDojo,
	SkillForge,
	SqliteTelemetryStore,
	type EvalCase,
	type TenantContext,
} from "../src/index.js";
import { trustedHeadersAuth } from "../src/transport/auth.js";

const TENANT: TenantContext = {
	tenantId: "tenant-learn",
	userId: "user-learn",
	permissions: [...DEFAULT_PERMISSIONS],
};

describe("Learning loops (Friction, Skill Dojo, Skill Forge, SIM Log, Replay Evals, HTTP)", () => {
	const tempDirs: string[] = [];

	afterEach(() => {
		for (const dir of tempDirs.splice(0)) {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	it("detects deterministic pre-LLM friction signals and captures them passively via Loop 1", async () => {
		const analysis = detectFrictionSignals({
			userMessage: "No, actually don't use npm — please remember to use pnpm instead of npm.",
			assistantSummary: "Retried tool.",
			toolTrace: [
				{
					callId: "c1",
					toolName: "domain_lookup_record",
					argumentsJson: '{"recordId":"a"}',
					isError: true,
					diagnosticCodes: ["tool_error"],
				},
				{
					callId: "c2",
					toolName: "domain_lookup_record",
					argumentsJson: '{"recordId":"a"}',
					isError: false,
					diagnosticCodes: ["artifact_offloaded"],
				},
				{
					callId: "c3",
					toolName: "domain_mutate_record",
					argumentsJson: '{"recordId":"a","value":"x"}',
					isError: true,
					diagnosticCodes: ["blocked"],
				},
			],
		});

		const kinds = new Set(analysis.signals.map((s) => s.kind));
		expect(kinds.has("user_correction")).toBe(true);
		expect(kinds.has("tool_error")).toBe(true);
		expect(kinds.has("repeated_tool_call")).toBe(true);
		expect(kinds.has("output_truncated")).toBe(true);
		expect(kinds.has("auth_blocked")).toBe(true);
		expect(analysis.score).toBeGreaterThanOrEqual(10);

		// End-to-end passive capture during a live harness run
		const runtime = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
		});
		try {
			const conv = await runtime.harness.createConversation({ tenant: TENANT });
			runtime.testModel.enqueueMainMany([
				{
					type: "tool_calls",
					calls: [{ id: "tc1", name: "domain_lookup_record", arguments: { recordId: "missing-id" } }],
				},
				{ type: "text", text: "Record missing-id was not found." },
			]);

			const res = await runtime.harness.submitAndWait({
				conversationId: conv.conversationId,
				content: "No, actually check missing-id instead of old-id.",
			});
			expect(res.status).toBe("done");
			await runtime.harness.waitForIdle(conv.conversationId);

			const episodes = await runtime.capabilities.memory.listEpisodes(TENANT.tenantId, TENANT.userId);
			expect(episodes).toHaveLength(1);
			expect(episodes[0]!.frictionSignals.some((s) => s.kind === "user_correction")).toBe(true);
			expect(episodes[0]!.frictionSignals.some((s) => s.kind === "tool_error")).toBe(true);
		} finally {
			await runtime.close();
		}
	});

	it("computes Skill Dojo health metrics, auto-disables stale (>30d) & degraded skills, and blocks them in skills-loader", async () => {
		const skillsDir = fs.mkdtempSync(path.join(os.tmpdir(), "helios-skills-"));
		tempDirs.push(skillsDir);

		fs.mkdirSync(path.join(skillsDir, "active-skill"), { recursive: true });
		fs.writeFileSync(
			path.join(skillsDir, "active-skill", "SKILL.md"),
			"---\nname: active-skill\ndescription: Healthy active skill\ntags: [core]\n---\n# Active\nDo good things.\n",
		);
		fs.mkdirSync(path.join(skillsDir, "stale-skill"), { recursive: true });
		fs.writeFileSync(
			path.join(skillsDir, "stale-skill", "SKILL.md"),
			"---\nname: stale-skill\ndescription: Old forgotten skill\ntags: [legacy]\n---\n# Stale\nOld steps.\n",
		);

		const DAY = 24 * 60 * 60 * 1000;
		const nowMs = Date.UTC(2026, 10, 24);

		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const telemetry = new SqliteTelemetryStore(appDb, { now: () => nowMs });
			const dojo = new SkillDojo(telemetry, { staleSkillDays: 30, maxErrorRate: 0.6, minErrorSamples: 3 });

			// Recent healthy skill (2 days ago)
			await telemetry.recordToolInvocation({
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				conversationId: "c1",
				callId: "1",
				toolName: "skill_read",
				skillName: "active-skill",
				isError: false,
				durationMs: 12,
				createdAt: nowMs - 2 * DAY,
			});

			// Stale skill (45 days ago > 30d threshold)
			await telemetry.recordToolInvocation({
				tenantId: TENANT.tenantId,
				userId: TENANT.userId,
				conversationId: "c1",
				callId: "2",
				toolName: "skill_read",
				skillName: "stale-skill",
				isError: false,
				durationMs: 15,
				createdAt: nowMs - 45 * DAY,
			});

			// Degraded skill (3/3 errors)
			for (let i = 0; i < 3; i++) {
				await telemetry.recordToolInvocation({
					tenantId: TENANT.tenantId,
					userId: TENANT.userId,
					conversationId: "c1",
					callId: `deg_${i}`,
					toolName: "skill_read",
					skillName: "flaky-skill",
					isError: true,
					durationMs: 20,
					createdAt: nowMs - DAY,
				});
			}

			const evalRes = await dojo.evaluateHealth(TENANT.tenantId, nowMs);
			expect(evalRes.allDisabled).toContain("stale-skill");
			expect(evalRes.allDisabled).toContain("flaky-skill");
			expect(evalRes.allDisabled).not.toContain("active-skill");

			// Progressive-disclosure catalog excludes disabled skills
			const enabledCatalog = loadSkillCatalog(skillsDir, evalRes.allDisabled);
			expect(enabledCatalog.map((s) => s.name)).toEqual(["active-skill"]);
		} finally {
			appDb.close();
		}
	});

	it("mines recurring 3-to-7 tool n-gram workflows deterministically in Skill Forge", () => {
		const forge = new SkillForge({ minLength: 3, maxLength: 7, minSupport: 2 });
		const mined = forge.mine([
			["memory_search", "domain_lookup_record", "domain_request_approval", "domain_mutate_record"],
			["artifact_read", "memory_search", "domain_lookup_record", "domain_request_approval", "domain_mutate_record"],
			["memory_search", "domain_lookup_record", "domain_request_approval", "domain_mutate_record", "memory_pin"],
			["memory_search", "artifact_read"],
		]);

		expect(mined.length).toBeGreaterThanOrEqual(1);
		const top = mined[0]!;
		expect(top.sequence).toEqual([
			"memory_search",
			"domain_lookup_record",
			"domain_request_approval",
			"domain_mutate_record",
		]);
		expect(top.length).toBe(4);
		expect(top.support).toBe(3);
		expect(top.draftSkillMarkdown).toContain("---");
		expect(top.draftSkillMarkdown).toContain("domain_mutate_record");
	});

	it("persists both wins and failed edits in append-only SimLog (Loop 3)", () => {
		const appDb = AppDatabase.open({ dbPath: ":memory:" });
		try {
			const simLog = new SimLog(appDb);
			const win = simLog.append({
				tenantId: TENANT.tenantId,
				targetModule: "src/extensions/core-prompt.ext.ts",
				hypothesis: "Clarify tool selection rule for memory_search before guessing",
				patchSummary: "Added explicit bullet #2 in DEFAULT_STATIC_SYSTEM_PROMPT",
				outcome: "win",
				evolveDelta: { passRate: 0.15, toolErrorRate: -0.05 },
				heldOutDelta: { passRate: 0.1, toolErrorRate: -0.04 },
				regressionDetected: false,
			});

			const fail = simLog.append({
				tenantId: TENANT.tenantId,
				targetModule: "src/extensions/core-prompt.ext.ts",
				hypothesis: "Remove skills_index instructions to save tokens",
				patchSummary: "Omitted skill_read hint",
				outcome: "failed",
				evolveDelta: { passRate: -0.2 },
				heldOutDelta: { passRate: -0.25 },
				regressionDetected: true,
				notes: "Model stopped reading relevant skills; do not retry.",
			});

			expect(win.id).toMatch(/^sim_/);
			expect(fail.id).toMatch(/^sim_/);

			const all = simLog.list({ tenantId: TENANT.tenantId });
			expect(all).toHaveLength(2);

			const failedForPrompt = simLog.getFailedHypothesesForModule(
				TENANT.tenantId,
				"src/extensions/core-prompt.ext.ts",
			);
			expect(failedForPrompt).toHaveLength(1);
			expect(failedForPrompt[0]!.outcome).toBe("failed");

			const summary = simLog.summarize(TENANT.tenantId);
			expect(summary).toEqual({
				total: 2,
				wins: 1,
				failed: 1,
				reverted: 0,
				inconclusive: 0,
				regressions: 1,
			});
		} finally {
			appDb.close();
		}
	});

	it("runs trace-replay evals with conversation forking (evolve vs held_out splits) and serves HTTP endpoints", async () => {
		const runtime = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
		});

		try {
			const runner = new ReplayEvalRunner({
				harness: runtime.harness,
				testModel: runtime.testModel,
			});

			const cases: EvalCase[] = [
				{
					id: "ev-1",
					name: "evolve split case with fork",
					split: "evolve",
					tenant: TENANT,
					seedTurns: [
						{
							userPrompt: "Seed context: project is Helios",
							scriptedResponses: [{ type: "text", text: "Noted project Helios." }],
						},
					],
					evalPrompt: "What project are we working on?",
					scriptedResponses: [{ type: "text", text: "We are working on Helios." }],
					expectedSubstrings: ["Helios"],
					forbiddenSubstrings: ["SECRET_LEAK"],
				},
				{
					id: "ho-1",
					name: "held_out split case",
					split: "held_out",
					tenant: TENANT,
					evalPrompt: "Pin a preference",
					scriptedResponses: [
						{
							type: "tool_calls",
							calls: [{ name: "memory_pin", arguments: { key: "eval.pref", content: "concise", kind: "preference" } }],
						},
						{ type: "text", text: "Pinned your concise preference." },
					],
					expectedToolNames: ["memory_pin"],
					expectedSubstrings: ["Pinned"],
				},
			];

			const suite = await runner.runSuite(cases);
			expect(suite.evolveMetrics.caseCount).toBe(1);
			expect(suite.evolveMetrics.passRate).toBe(1);
			expect(suite.heldOutMetrics.caseCount).toBe(1);
			expect(suite.heldOutMetrics.passRate).toBe(1);

			const cmp = comparePareto(suite.heldOutMetrics, {
				caseCount: 1,
				passRate: 0.5,
				avgLatencyMs: 500,
				toolErrorRate: 0.2,
				safetyViolations: 0,
			});
			expect(cmp.dominates).toBe(true);
			expect(cmp.nonRegressing).toBe(true);

			// Thin Hono HTTP transport smoke test (wraps HarnessPort only)
			const app = createHttpServer(runtime.harness, { auth: trustedHeadersAuth() });
			const authHeaders = {
				"content-type": "application/json",
				"x-tenant-id": TENANT.tenantId,
				"x-user-id": TENANT.userId,
				"x-permissions": [...TENANT.permissions, "admin:tool-packs"].join(","),
			};
			const healthRes = await app.request("/health");
			expect(healthRes.status).toBe(200);

			const createRes = await app.request("/conversations", {
				method: "POST",
				headers: authHeaders,
				body: JSON.stringify({ tenant: TENANT }),
			});
			expect(createRes.status).toBe(201);
			const created = (await createRes.json()) as { conversationId: string };

			runtime.testModel.enqueueMain({ type: "text", text: "HTTP response OK" });
			const msgRes = await app.request(`/conversations/${created.conversationId}/messages`, {
				method: "POST",
				headers: authHeaders,
				body: JSON.stringify({ content: "Ping over HTTP", wait: true, requestId: "http-1" }),
			});
			expect(msgRes.status).toBe(200);
			const msgBody = (await msgRes.json()) as { status: string; answerText: string };
			expect(msgBody.status).toBe("done");
			expect(msgBody.answerText).toBe("HTTP response OK");

			// ToolPack toggling over HTTP
			const toggleRes = await app.request("/tool-packs/example-domain", {
				method: "POST",
				headers: authHeaders,
				body: JSON.stringify({ enabled: false }),
			});
			expect(toggleRes.status).toBe(200);
			const packs = ((await toggleRes.json()) as { toolPacks: Array<{ name: string; enabled: boolean }> }).toolPacks;
			expect(packs.find((p) => p.name === "example-domain")?.enabled).toBe(false);

			// Stream adapter emits snapshot + entry_appended + text_delta over watchConversation
			const abort = new AbortController();
			const seenEvents: string[] = [];
			const watchPromise = (async () => {
				for await (const ev of runtime.harness.watchConversation(created.conversationId, abort.signal)) {
					seenEvents.push(ev.type);
					if (ev.type === "entry_appended" && ev.entry.role === "assistant") {
						abort.abort();
						break;
					}
				}
			})();

			runtime.testModel.enqueueMain({ type: "text", text: "Streamed delta hello world!" });
			await runtime.harness.submitAndWait({
				conversationId: created.conversationId,
				content: "Stream me an answer",
			});
			await watchPromise;
			expect(seenEvents[0]).toBe("snapshot");
			expect(seenEvents).toContain("entry_appended");
		} finally {
			await runtime.close();
		}
	});
});
