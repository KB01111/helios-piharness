import path from "node:path";
import { bootstrapHarness } from "../index.js";
import { checkAgainstBaseline, loadEvalSet, readBaseline, writeBaseline } from "./eval-set.js";
import { ReplayEvalRunner } from "./replay-runner.js";

/**
 * `npm run eval`            -> run the committed eval set and fail (exit 1) on any regression vs `evals/baseline.json`.
 * `npm run eval:baseline`   -> run it and (over)write the baseline. Review the diff before committing it.
 *
 * `npm run eval:live`       -> run the same cases against REAL models (needs a provider key; costs tokens). The scripted
 *                              responses are ignored, so the model decides; the verifier, planner, codemode and guards are
 *                              all live. No baseline comparison (the baseline is a mock-model artifact); exits 1 only on
 *                              safety violations. Prints per-case outcome, tools used and total token usage.
 *
 * Without `--live` it runs against the offline TestModel (deterministic, free).
 */
async function main(): Promise<void> {
	const update = process.argv.includes("--update-baseline");
	const live = process.argv.includes("--live");
	if (live && !process.env.ANTHROPIC_API_KEY && !process.env.OPENAI_API_KEY && !process.env.GEMINI_API_KEY) {
		throw new Error("--live needs a provider key (ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY); refusing to run on the mock model");
	}
	const evalsDir = path.resolve(process.env.EVALS_DIR ?? "./evals");
	const { cases } = loadEvalSet(path.join(evalsDir, "cases"));
	if (cases.length === 0) throw new Error(`No eval cases found in ${path.join(evalsDir, "cases")}`);

	const runtime = await bootstrapHarness({
		config: {
			piDbPath: ":memory:",
			appDbPath: ":memory:",
			sleepCycleSchedulerEnabled: false,
			...(live ? { useMockModels: false } : {}),
		},
		improvements: false,
		applyActiveImprovements: false,
		sleepCycleScheduler: false,
	});
	try {
		const suite = await new ReplayEvalRunner({ harness: runtime.harness, testModel: runtime.testModel }).runSuite(cases);
		const current = { evolve: suite.evolveMetrics, heldOut: suite.heldOutMetrics };
		for (const r of suite.results.filter((r) => !r.passed)) {
			process.stdout.write(`FAIL ${r.caseId}: ${r.failures.join("; ")}\n`);
		}
		if (live) {
			for (const r of suite.results) {
				process.stdout.write(
					`${r.passed ? "PASS" : "FAIL"} ${r.caseId} [${r.split}] tools=[${r.toolCalls.join(",")}] ${r.latencyMs}ms answer="${r.answerText.replace(/\s+/g, " ").slice(0, 120)}"
`,
				);
			}
			const usage = await runtime.harness.getUsage();
			process.stdout.write(`live: ${JSON.stringify({ evolve: current.evolve, heldOut: current.heldOut })}
usage: ${JSON.stringify(usage)}
`);
			if (current.evolve.safetyViolations + current.heldOut.safetyViolations > 0) process.exitCode = 1;
			return;
		}
		const baselineFile = path.join(evalsDir, "baseline.json");
		if (update) {
			writeBaseline(baselineFile, current);
			process.stdout.write(`Baseline written to ${baselineFile}\n`);
			return;
		}
		const baseline = readBaseline(baselineFile);
		if (!baseline) throw new Error(`No baseline at ${baselineFile}; run \`npm run eval:baseline\` first`);
		const report = checkAgainstBaseline(current, baseline);
		for (const problem of report.problems) process.stdout.write(`REGRESSION ${problem}\n`);
		process.stdout.write(
			`evolve ${(current.evolve.passRate * 100).toFixed(0)}% (${current.evolve.caseCount}) | held-out ${(current.heldOut.passRate * 100).toFixed(0)}% (${current.heldOut.caseCount})\n`,
		);
		if (!report.ok) process.exitCode = 1;
	} finally {
		await runtime.close();
	}
}

main().catch((error: unknown) => {
	process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
	process.exit(1);
});
