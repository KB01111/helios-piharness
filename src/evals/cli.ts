import path from "node:path";
import { bootstrapHarness } from "../index.js";
import { checkAgainstBaseline, loadEvalSet, readBaseline, writeBaseline } from "./eval-set.js";
import { ReplayEvalRunner } from "./replay-runner.js";
import { aggregateParetoMetrics, type EvalCaseResult } from "./types.js";

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
	if (live && !process.env.ANTHROPIC_API_KEY && !process.env.OPENAI_API_KEY && !process.env.GEMINI_API_KEY && !process.env.OPENROUTER_API_KEY) {
		throw new Error("--live needs a provider key (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or OPENROUTER_API_KEY); refusing to run on the mock model");
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
	// A durable retry backoff is an unref'd timer; without a live handle Node would exit silently mid-run.
	const keepAlive = setInterval(() => undefined, 1_000);
	try {
		const runner = new ReplayEvalRunner({ harness: runtime.harness, testModel: runtime.testModel });
		if (live) {
			// Stream each case as it finishes (free/real models are slow) and bound every case with a timeout.
			const caseTimeoutMs = Number(process.env.EVAL_CASE_TIMEOUT_MS ?? 240_000);
			const only = process.env.EVAL_ONLY?.split(",").map((id) => id.trim()).filter(Boolean);
			const selected = only?.length ? cases.filter((c) => only.includes(c.id)) : cases;
			const results: EvalCaseResult[] = [];
			for (const evalCase of selected) {
				const timeout = new Promise<EvalCaseResult>((resolve) =>
					setTimeout(
						() =>
							resolve({
								caseId: evalCase.id,
								name: evalCase.name,
								split: evalCase.split,
								passed: false,
								answerText: "",
								latencyMs: caseTimeoutMs,
								toolCalls: [],
								toolErrors: 0,
								safetyViolations: 0,
								failures: [`timed out after ${caseTimeoutMs}ms`],
							}),
						caseTimeoutMs,
					).unref(),
				);
				const r = await Promise.race([runner.runCase(evalCase), timeout]);
				results.push(r);
				process.stdout.write(
					`${r.passed ? "PASS" : "FAIL"} ${r.caseId} [${r.split}] tools=[${r.toolCalls.join(",")}] ${r.latencyMs}ms${r.passed ? "" : ` why="${r.failures.join("; ").slice(0, 200)}"`} answer="${r.answerText.replace(/\s+/g, " ").slice(0, 120)}"
`,
				);
			}
			const evolve = aggregateParetoMetrics(results.filter((r) => r.split === "evolve"));
			const heldOut = aggregateParetoMetrics(results.filter((r) => r.split === "held_out"));
			const usage = await runtime.harness.getUsage();
			process.stdout.write(`live: ${JSON.stringify({ evolve, heldOut })}
usage: ${JSON.stringify(usage)}
`);
			if (evolve.safetyViolations + heldOut.safetyViolations > 0) process.exitCode = 1;
			return;
		}
		const suite = await runner.runSuite(cases);
		const current = { evolve: suite.evolveMetrics, heldOut: suite.heldOutMetrics };
		for (const r of suite.results.filter((r) => !r.passed)) {
			process.stdout.write(`FAIL ${r.caseId}: ${r.failures.join("; ")}
`);
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
		clearInterval(keepAlive);
		await runtime.close();
	}
}

main().catch((error: unknown) => {
	process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
	process.exit(1);
});
