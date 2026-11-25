import path from "node:path";
import { bootstrapHarness } from "../index.js";
import { checkAgainstBaseline, loadEvalSet, readBaseline, writeBaseline } from "./eval-set.js";
import { ReplayEvalRunner } from "./replay-runner.js";

/**
 * `npm run eval`            -> run the committed eval set and fail (exit 1) on any regression vs `evals/baseline.json`.
 * `npm run eval:baseline`   -> run it and (over)write the baseline. Review the diff before committing it.
 *
 * Runs against the offline TestModel by default (deterministic, free). Set real provider keys and USE_MOCK_MODELS=false
 * to evaluate a real model; scripted cases then act as smoke tests of the harness around it.
 */
async function main(): Promise<void> {
	const update = process.argv.includes("--update-baseline");
	const evalsDir = path.resolve(process.env.EVALS_DIR ?? "./evals");
	const { cases } = loadEvalSet(path.join(evalsDir, "cases"));
	if (cases.length === 0) throw new Error(`No eval cases found in ${path.join(evalsDir, "cases")}`);

	const runtime = await bootstrapHarness({
		config: { piDbPath: ":memory:", appDbPath: ":memory:", sleepCycleSchedulerEnabled: false },
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
