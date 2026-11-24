import type { TenantContext } from "../contracts/tenant-context.js";
import type { ScriptedTurnOutput } from "../runtime/model-catalog.js";

/**
 * Frozen eval case schema with explicit evolve vs held_out splits and
 * Pareto multi-objective comparison metrics (Loop 3).
 */

export type EvalSplit = "evolve" | "held_out";

export interface EvalSeedTurn {
	readonly userPrompt: string;
	readonly scriptedResponses?: readonly ScriptedTurnOutput[];
}

export interface EvalCase {
	readonly id: string;
	readonly name: string;
	readonly split: EvalSplit;
	readonly tenant: TenantContext;
	/** Optional seed turns executed on a base conversation before forking. */
	readonly seedTurns?: readonly EvalSeedTurn[];
	/** Prompt evaluated on the forked branch. */
	readonly evalPrompt: string;
	/** Scripted deterministic responses for offline TestModel replay. */
	readonly scriptedResponses?: readonly ScriptedTurnOutput[];
	/** Assertions checked against the final answer and transcript. */
	readonly expectedSubstrings?: readonly string[];
	readonly forbiddenSubstrings?: readonly string[];
	readonly expectedToolNames?: readonly string[];
	readonly maxTurns?: number;
	readonly maxLatencyMs?: number;
}

export interface EvalCaseResult {
	readonly caseId: string;
	readonly name: string;
	readonly split: EvalSplit;
	readonly passed: boolean;
	readonly answerText: string;
	readonly latencyMs: number;
	readonly toolCalls: readonly string[];
	readonly toolErrors: number;
	readonly safetyViolations: number;
	readonly failures: readonly string[];
}

export interface ParetoMetrics {
	readonly caseCount: number;
	/** Higher is better (0..1). */
	readonly passRate: number;
	/** Lower is better. */
	readonly avgLatencyMs: number;
	/** Lower is better (0..1). */
	readonly toolErrorRate: number;
	/** Lower is better (must stay 0 for safety gate). */
	readonly safetyViolations: number;
}

export interface ParetoComparison {
	readonly dominates: boolean;
	readonly nonRegressing: boolean;
	readonly deltas: {
		readonly passRate: number;
		readonly avgLatencyMs: number;
		readonly toolErrorRate: number;
		readonly safetyViolations: number;
	};
}

export function aggregateParetoMetrics(results: readonly EvalCaseResult[]): ParetoMetrics {
	if (results.length === 0) {
		return {
			caseCount: 0,
			passRate: 0,
			avgLatencyMs: 0,
			toolErrorRate: 0,
			safetyViolations: 0,
		};
	}

	let passed = 0;
	let totalLatency = 0;
	let totalTools = 0;
	let totalToolErrors = 0;
	let totalSafety = 0;

	for (const r of results) {
		if (r.passed) passed += 1;
		totalLatency += r.latencyMs;
		totalTools += r.toolCalls.length;
		totalToolErrors += r.toolErrors;
		totalSafety += r.safetyViolations;
	}

	return {
		caseCount: results.length,
		passRate: passed / results.length,
		avgLatencyMs: totalLatency / results.length,
		toolErrorRate: totalTools > 0 ? totalToolErrors / totalTools : 0,
		safetyViolations: totalSafety,
	};
}

/**
 * Compares candidate metrics against baseline metrics on the Pareto frontier.
 * Safety violations may never regress; passRate may never drop.
 */
export function comparePareto(candidate: ParetoMetrics, baseline: ParetoMetrics): ParetoComparison {
	const deltas = {
		passRate: candidate.passRate - baseline.passRate,
		avgLatencyMs: candidate.avgLatencyMs - baseline.avgLatencyMs,
		toolErrorRate: candidate.toolErrorRate - baseline.toolErrorRate,
		safetyViolations: candidate.safetyViolations - baseline.safetyViolations,
	};

	const nonRegressing =
		deltas.safetyViolations <= 0 &&
		deltas.passRate >= 0 &&
		deltas.toolErrorRate <= 0;

	const strictlyBetterOnAtLeastOne =
		deltas.passRate > 0 ||
		deltas.toolErrorRate < 0 ||
		deltas.safetyViolations < 0 ||
		deltas.avgLatencyMs < 0;

	return {
		dominates: nonRegressing && strictlyBetterOnAtLeastOne,
		nonRegressing,
		deltas,
	};
}
