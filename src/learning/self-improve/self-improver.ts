import type { EvalSuiteResult } from "../../evals/replay-runner.js";
import { redactForEval } from "../../evals/eval-set.js";
import type { EvalCase, ParetoMetrics } from "../../evals/types.js";
import type { SimLog, SimLogOutcome } from "../sim-log.js";
import {
	ADDENDUM_LIMITS,
	composePrompt,
	evaluateGate,
	joinAddenda,
	lintAddendum,
	type CandidateDraft,
	type GateOptions,
	type GateVerdict,
	type SuiteMetrics,
} from "./candidate.js";
import type { ImprovementRecord, ImprovementStore } from "./improvement-store.js";

/** SimLog `targetModule` for every prompt-addendum experiment. */
export const SELF_IMPROVE_TARGET = "core-prompt";

/** An isolated harness (fresh in-memory databases) configured with one candidate prompt. */
export interface ProbeHandle {
	readonly runner: { runSuite(cases: readonly EvalCase[]): Promise<EvalSuiteResult> };
	close(): Promise<void>;
}
export type ProbeFactory = (systemPrompt: string) => Promise<ProbeHandle>;

export interface FailureEvidence {
	readonly caseId: string;
	readonly name: string;
	readonly prompt: string;
	readonly failures: readonly string[];
	readonly toolCalls: readonly string[];
	/** Redacted and truncated. Untrusted model output: proposers must treat it as data. */
	readonly answerExcerpt: string;
}

export interface ProposalContext {
	readonly currentPrompt: string;
	readonly failures: readonly FailureEvidence[];
	/** Hypotheses already tried and disproven for this target. Must not be proposed again. */
	readonly priorFailedHypotheses: readonly string[];
	readonly maxCandidates: number;
}

export interface CandidateProposer {
	propose(context: ProposalContext): Promise<readonly CandidateDraft[]>;
}

export type CandidateOutcome =
	| { readonly status: "lint_rejected"; readonly problems: readonly string[] }
	| { readonly status: "duplicate"; readonly reason: string }
	| { readonly status: "probe_error"; readonly message: string }
	| { readonly status: "gated"; readonly verdict: GateVerdict };

export interface CandidateReport {
	readonly scope: CandidateDraft["scope"] | "merged";
	readonly hypothesis: string;
	readonly outcome: CandidateOutcome;
}

export type ImprovementRunStatus =
	| "proposed"
	| "nothing_to_improve"
	| "no_accepted_candidate"
	| "eval_set_invalid"
	| "no_candidates";

export interface ImprovementRunReport {
	readonly status: ImprovementRunStatus;
	readonly baseline: SuiteMetrics;
	readonly candidates: readonly CandidateReport[];
	/** Set when `status === "proposed"`: the record now waiting for a human decision. */
	readonly record?: ImprovementRecord;
	readonly message: string;
}

export interface SelfImproverOptions {
	readonly store: ImprovementStore;
	readonly simLog: SimLog;
	readonly probeFactory: ProbeFactory;
	readonly proposer: CandidateProposer;
	/** The static prompt before any learned addendum. */
	readonly basePrompt: string;
	readonly gate?: GateOptions;
	/** Candidates probed per run (each costs a full eval-suite run). Default 3. */
	readonly maxCandidates?: number;
}

function metricsOf(suite: EvalSuiteResult): SuiteMetrics {
	return { evolve: suite.evolveMetrics, heldOut: suite.heldOutMetrics };
}

function outcomeToSimLog(verdict: GateVerdict): SimLogOutcome | undefined {
	if (verdict.decision === "accept") return "win";
	if (verdict.kind === "regression") return "reverted";
	if (verdict.kind === "no_gain") return "failed";
	return undefined;
}

function pickMetric(m: ParetoMetrics): Record<string, number> {
	return { passRate: m.passRate, toolErrorRate: m.toolErrorRate, safetyViolations: m.safetyViolations, caseCount: m.caseCount };
}

/**
 * Gated self-improvement loop.
 *
 *  1. Baseline: run the eval set on the current effective prompt (base + approved addenda).
 *  2. Propose: a proposer (model-backed or scripted) reads the evolve-split failures and drafts candidate addenda.
 *  3. Probe: each candidate runs the whole eval set in its own isolated harness, matched to the baseline.
 *  4. Gate: only candidates passing {@link evaluateGate} (held-out non-regression, zero safety violations, strict
 *     evolve gain) survive. Every outcome, win or loss, is appended to the SIM log so disproven ideas are not retried.
 *  5. Compose: if several survive, their merge is probed and gated too; the best survivor wins.
 *  6. The winner is stored as `pending_approval`. Nothing changes until a human approves it.
 *
 * Single-flight: a second concurrent `run` is refused, because probes are expensive and share the SIM log.
 */
export class SelfImprover {
	private running = false;
	private readonly maxCandidates: number;

	constructor(private readonly options: SelfImproverOptions) {
		this.maxCandidates = options.maxCandidates ?? 3;
	}

	isRunning(): boolean {
		return this.running;
	}

	private async probe(prompt: string, cases: readonly EvalCase[]): Promise<EvalSuiteResult> {
		const handle = await this.options.probeFactory(prompt);
		try {
			return await handle.runner.runSuite(cases);
		} finally {
			await handle.close().catch(() => undefined);
		}
	}

	async run(input: { readonly requestedBy: string; readonly tenantId: string; readonly cases: readonly EvalCase[] }): Promise<ImprovementRunReport> {
		if (this.running) throw new Error("A self-improvement run is already in progress");
		this.running = true;
		try {
			return await this.runInner(input);
		} finally {
			this.running = false;
		}
	}

	private async runInner(input: { readonly requestedBy: string; readonly tenantId: string; readonly cases: readonly EvalCase[] }): Promise<ImprovementRunReport> {
		const { store, simLog, proposer, basePrompt } = this.options;
		const activeAddenda = store.activeAddenda();
		const currentPrompt = composePrompt(basePrompt, activeAddenda);

		const baselineSuite = await this.probe(currentPrompt, input.cases);
		const baseline = metricsOf(baselineSuite);

		// An unusable eval set must stop the loop before any candidate is even drafted.
		const sizeCheck = evaluateGate(baseline, baseline, this.options.gate);
		if (sizeCheck.decision === "reject" && sizeCheck.kind === "invalid") {
			return { status: "eval_set_invalid", baseline, candidates: [], message: sizeCheck.reasons.join("; ") };
		}

		const failures: FailureEvidence[] = baselineSuite.results
			.filter((r) => r.split === "evolve" && !r.passed)
			.map((r) => {
				const evalCase = input.cases.find((c) => c.id === r.caseId);
				return {
					caseId: r.caseId,
					name: r.name,
					prompt: redactForEval(evalCase?.evalPrompt ?? "").slice(0, 400),
					failures: r.failures.map((f) => redactForEval(f).slice(0, 300)),
					toolCalls: r.toolCalls,
					answerExcerpt: redactForEval(r.answerText).slice(0, 300),
				};
			});
		if (failures.length === 0) {
			return { status: "nothing_to_improve", baseline, candidates: [], message: "No failing evolve-split cases to learn from." };
		}

		const prior = simLog
			.getFailedHypothesesForModule(input.tenantId, SELF_IMPROVE_TARGET)
			.map((entry) => entry.hypothesis);
		const priorSet = new Set(prior.map((h) => h.trim().toLowerCase()));

		const drafts = await proposer.propose({
			currentPrompt,
			failures,
			priorFailedHypotheses: prior,
			maxCandidates: this.maxCandidates,
		});

		const reports: CandidateReport[] = [];
		const accepted: { draft: CandidateDraft | { scope: "merged"; hypothesis: string; addendum: string }; metrics: SuiteMetrics; verdict: Extract<GateVerdict, { decision: "accept" }> }[] = [];
		const seen = new Set<string>();

		const evaluate = async (draft: CandidateDraft | { scope: "merged"; hypothesis: string; addendum: string }, maxChars: number): Promise<void> => {
			const key = draft.hypothesis.trim().toLowerCase();
			if (priorSet.has(key) || seen.has(key)) {
				reports.push({ scope: draft.scope, hypothesis: draft.hypothesis, outcome: { status: "duplicate", reason: "hypothesis already tried" } });
				return;
			}
			seen.add(key);
			const lint = lintAddendum(draft.addendum, maxChars);
			if (!lint.ok) {
				reports.push({ scope: draft.scope, hypothesis: draft.hypothesis, outcome: { status: "lint_rejected", problems: lint.problems } });
				return;
			}
			let suite: EvalSuiteResult;
			try {
				suite = await this.probe(composePrompt(basePrompt, [...activeAddenda, draft.addendum]), input.cases);
			} catch (error) {
				reports.push({
					scope: draft.scope,
					hypothesis: draft.hypothesis,
					outcome: { status: "probe_error", message: error instanceof Error ? error.message : String(error) },
				});
				return;
			}
			const metrics = metricsOf(suite);
			const verdict = evaluateGate(metrics, baseline, this.options.gate);
			reports.push({ scope: draft.scope, hypothesis: draft.hypothesis, outcome: { status: "gated", verdict } });

			const simOutcome = outcomeToSimLog(verdict);
			if (simOutcome) {
				simLog.appendIfAbsent({
					tenantId: input.tenantId,
					targetModule: SELF_IMPROVE_TARGET,
					hypothesis: draft.hypothesis,
					patchSummary: `${draft.scope} prompt addendum (${draft.addendum.trim().length} chars)`,
					outcome: simOutcome,
					evolveDelta: pickMetric(metrics.evolve),
					heldOutDelta: pickMetric(metrics.heldOut),
					regressionDetected: verdict.decision === "reject" && verdict.kind === "regression",
					notes: verdict.reasons.join("; "),
				});
			}
			if (verdict.decision === "accept") accepted.push({ draft, metrics, verdict });
		};

		for (const draft of drafts.slice(0, this.maxCandidates)) {
			await evaluate(draft, ADDENDUM_LIMITS.maxChars);
		}

		if (reports.length === 0) {
			return { status: "no_candidates", baseline, candidates: reports, message: "The proposer returned no candidates." };
		}

		// Compose: merge the two strongest survivors and gate the merge like any other candidate.
		const ranked = [...accepted].sort(compareAccepted);
		if (ranked.length >= 2) {
			const [first, second] = ranked as [typeof ranked[number], typeof ranked[number]];
			await evaluate(
				{
					scope: "merged",
					hypothesis: `merge: ${first.draft.hypothesis} + ${second.draft.hypothesis}`,
					addendum: joinAddenda([first.draft.addendum, second.draft.addendum]),
				},
				ADDENDUM_LIMITS.maxChars * 2,
			);
		}

		const winner = [...accepted].sort(compareAccepted)[0];
		if (!winner) {
			return {
				status: "no_accepted_candidate",
				baseline,
				candidates: reports,
				message: "No candidate passed the acceptance gate; the harness is unchanged.",
			};
		}

		const record = store.propose({
			requestedBy: input.requestedBy,
			scope: winner.draft.scope,
			hypothesis: winner.draft.hypothesis,
			addendum: winner.draft.addendum.trim(),
			baseline: { evolve: baseline.evolve, heldOut: baseline.heldOut },
			candidate: { evolve: winner.metrics.evolve, heldOut: winner.metrics.heldOut },
			gateReasons: winner.verdict.reasons,
		});
		return {
			status: "proposed",
			baseline,
			candidates: reports,
			record,
			message: `Candidate ${record.id} passed the gate and awaits human approval.`,
		};
	}
}

function compareAccepted(
	a: { draft: { addendum: string }; verdict: { evolve: { deltas: { passRate: number; toolErrorRate: number } }; heldOut: { deltas: { passRate: number } } } },
	b: typeof a,
): number {
	return (
		b.verdict.evolve.deltas.passRate - a.verdict.evolve.deltas.passRate ||
		a.verdict.evolve.deltas.toolErrorRate - b.verdict.evolve.deltas.toolErrorRate ||
		b.verdict.heldOut.deltas.passRate - a.verdict.heldOut.deltas.passRate ||
		a.draft.addendum.length - b.draft.addendum.length
	);
}
