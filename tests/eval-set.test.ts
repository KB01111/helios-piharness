import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	bootstrapHarness,
	buildEvalCaseFromTranscript,
	checkAgainstBaseline,
	EvalSetError,
	loadEvalSet,
	parseEvalCases,
	readBaseline,
	redactForEval,
	ReplayEvalRunner,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = { tenantId: "t", userId: "u", permissions: ["memory:read"] };
const metrics = (passRate: number, toolErrorRate = 0, safetyViolations = 0, caseCount = 4) => ({
	caseCount,
	passRate,
	avgLatencyMs: 10,
	toolErrorRate,
	safetyViolations,
});

describe("eval set tooling", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
	});
	const tmp = () => {
		const d = fs.mkdtempSync(path.join(os.tmpdir(), "helios-evalset-"));
		dirs.push(d);
		return d;
	};

	it("rejects malformed cases with a precise path", () => {
		expect(() => parseEvalCases({ id: "x" }, "f.json")).toThrow(EvalSetError);
		expect(() =>
			parseEvalCases({ id: "x", name: "n", split: "train", tenant: TENANT, evalPrompt: "p" }, "f.json"),
		).toThrow(/split/);
		expect(() =>
			parseEvalCases(
				{ id: "x", name: "n", split: "evolve", tenant: TENANT, evalPrompt: "p", scriptedResponses: [{ type: "tool_calls", calls: [{ name: 1 }] }] },
				"f.json",
			),
		).toThrow(/scriptedResponses\[0\]\.calls\[0\]/);
	});

	it("loads a directory deterministically and rejects duplicate ids and bad JSON", () => {
		const dir = tmp();
		const ok = { id: "a", name: "A", split: "evolve", tenant: TENANT, evalPrompt: "hi" };
		fs.writeFileSync(path.join(dir, "b.json"), JSON.stringify([{ ...ok, id: "b" }]));
		fs.writeFileSync(path.join(dir, "a.json"), JSON.stringify(ok));
		expect(loadEvalSet(dir).cases.map((c) => c.id)).toEqual(["a", "b"]);
		expect(loadEvalSet(path.join(dir, "missing")).cases).toEqual([]);

		fs.writeFileSync(path.join(dir, "c.json"), JSON.stringify(ok));
		expect(() => loadEvalSet(dir)).toThrow(/duplicate eval case id "a"/);
		fs.rmSync(path.join(dir, "c.json"));
		fs.writeFileSync(path.join(dir, "d.json"), "{nope");
		expect(() => loadEvalSet(dir)).toThrow(/invalid JSON/);
	});

	it("redacts PII-shaped text and builds a case from a transcript", () => {
		expect(redactForEval("mail bob@example.com key sk-abcdefghijklmnopqrstuvwx call +1 415 555 0100")).toBe(
			"mail [email] key [secret] call [phone]",
		);
		const evalCase = buildEvalCaseFromTranscript(
			[
				{ id: "1", kind: "pi.user", role: "user", text: "first, my email is a@b.co" },
				{ id: "2", kind: "pi.assistant", role: "assistant", text: "ok" },
				{ id: "3", kind: "pi.user", role: "user", text: "pin my tea preference" },
				{ id: "4", kind: "pi.tool-result", role: "toolResult", toolName: "memory_pin", isError: false },
				{ id: "5", kind: "pi.tool-result", role: "toolResult", toolName: "bad_tool", isError: true },
				{ id: "6", kind: "pi.assistant", role: "assistant", text: "done" },
			],
			{ id: "c1", name: "tea", split: "held_out", tenant: TENANT, expectedSubstrings: ["tea"] },
		);
		expect(evalCase.evalPrompt).toBe("pin my tea preference");
		expect(evalCase.seedTurns).toEqual([{ userPrompt: "first, my email is [email]" }]);
		expect(evalCase.expectedToolNames).toEqual(["memory_pin"]);
		expect(evalCase.scriptedResponses).toBeUndefined();
		expect(() => buildEvalCaseFromTranscript([], { id: "x", name: "x", split: "evolve", tenant: TENANT })).toThrow(EvalSetError);
	});

	it("baseline check flags pass-rate, tool-error, safety and shrinkage regressions but ignores latency", () => {
		const baseline = { version: 1 as const, updatedAt: "", evolve: metrics(1), heldOut: metrics(1) };
		expect(checkAgainstBaseline({ evolve: { ...metrics(1), avgLatencyMs: 9_999 }, heldOut: metrics(1) }, baseline).ok).toBe(true);
		const bad = checkAgainstBaseline(
			{ evolve: metrics(0.75), heldOut: metrics(1, 0.2, 1, 3) },
			baseline,
		);
		expect(bad.ok).toBe(false);
		expect(bad.problems.join("\n")).toMatch(/evolve: pass rate fell/);
		expect(bad.problems.join("\n")).toMatch(/heldOut: eval set shrank/);
		expect(bad.problems.join("\n")).toMatch(/heldOut: safety violations rose/);
		expect(bad.problems.join("\n")).toMatch(/heldOut: tool error rate rose/);
	});
});

/**
 * CI regression gate: the committed eval set must still pass and must not regress against the committed baseline.
 * If you intentionally change behaviour, run `npm run eval:baseline` and review the baseline diff in the PR.
 */
describe("CI eval regression gate", () => {
	it("committed eval set does not regress against evals/baseline.json", async () => {
		const evalsDir = path.resolve("evals");
		const { cases } = loadEvalSet(path.join(evalsDir, "cases"));
		const baseline = readBaseline(path.join(evalsDir, "baseline.json"));
		expect(cases.length).toBeGreaterThanOrEqual(8);
		expect(baseline).toBeDefined();

		const runtime = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true },
			improvements: false,
			applyActiveImprovements: false,
			sleepCycleScheduler: false,
		});
		try {
			const suite = await new ReplayEvalRunner({ harness: runtime.harness, testModel: runtime.testModel }).runSuite(cases);
			const failed = suite.results.filter((r) => !r.passed).map((r) => `${r.caseId}: ${r.failures.join("; ")}`);
			expect(failed).toEqual([]);
			const report = checkAgainstBaseline({ evolve: suite.evolveMetrics, heldOut: suite.heldOutMetrics }, baseline!);
			expect(report.problems).toEqual([]);
		} finally {
			await runtime.close();
		}
	});
});
