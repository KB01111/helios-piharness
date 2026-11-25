import { sanitizeMemoryText } from "../../memory/sanitize.js";
import { comparePareto, type ParetoComparison, type ParetoMetrics } from "../../evals/types.js";

/**
 * Self-improvement candidates (Harness²-style propose / probe / compose, scoped for a multi-tenant server).
 *
 * The only editable surface is a **prompt addendum**: extra operating guidance appended to the static system prompt.
 * Guardrails, permissions, tool code, budgets and tenant binding are NOT editable by the loop, by construction.
 * Skills have their own reviewed pipeline (Skill Forge -> `skills/_proposed` -> `promoteProposedSkill`).
 */

export type CandidateScope = "invariant" | "domain";

export interface CandidateDraft {
	/** `invariant`: a general agentic fix. `domain`: specific to the failures observed in this deployment's evals. */
	readonly scope: CandidateScope;
	/** One falsifiable sentence: "adding X will fix Y". Used as the SimLog key so a failed idea is never retried. */
	readonly hypothesis: string;
	/** Text appended to the system prompt. */
	readonly addendum: string;
}

export const ADDENDUM_LIMITS = {
	/** One candidate. */
	maxChars: 1_200,
	/** All active addenda combined; keeps the prompt prefix small and stable. */
	maxTotalChars: 4_000,
} as const;

/** Phrases a prompt edit must never contain: attempts to weaken controls or smuggle identity/secrets. */
const FORBIDDEN: readonly { readonly pattern: RegExp; readonly reason: string }[] = [
	{ pattern: /\bignore\b[^.\n]{0,40}\b(instruction|rule|prompt|polic)/i, reason: "tells the model to ignore instructions" },
	{ pattern: /\b(bypass|circumvent|evade|disable|skip|override|turn off)\b[^.\n]{0,40}\b(guard|safety|approval|permission|authori[sz]ation|budget|limit|check|sandbox|policy)/i, reason: "weakens a control" },
	{ pattern: /\bwithout\b[^.\n]{0,20}\b(asking|approval|permission|confirmation|authori[sz]ation)/i, reason: "removes a confirmation step" },
	{ pattern: /\b(tenant[\s_-]?id|user[\s_-]?id|api[\s_-]?key|password|secret|credential|bearer)\b/i, reason: "mentions identity or secrets" },
	{ pattern: /<\/?[a-z_][a-z0-9_-]*[^>]*>/i, reason: "contains markup/tag-like text" },
	{ pattern: /https?:\/\//i, reason: "contains a URL" },
	{ pattern: /\bcodemode\b|\btools\.[a-z_]+\(/i, reason: "references code execution" },
];

export interface LintResult {
	readonly ok: boolean;
	readonly problems: readonly string[];
}

export function lintAddendum(addendum: string, maxChars: number = ADDENDUM_LIMITS.maxChars): LintResult {
	const problems: string[] = [];
	const text = addendum.trim();
	if (text.length === 0) problems.push("addendum is empty");
	if (text.length > maxChars) problems.push(`addendum is ${text.length} chars, over the ${maxChars} cap`);
	for (const rule of FORBIDDEN) {
		if (rule.pattern.test(text)) problems.push(rule.reason);
	}
	const sanitized = sanitizeMemoryText(text, { maxChars: Math.max(maxChars, text.length) });
	if (sanitized.flagged) problems.push(`prompt-injection screen flagged: ${sanitized.reasons.join(", ")}`);
	return { ok: problems.length === 0, problems };
}

/** Joins addenda deterministically (stable bytes => stable KV-cache prefix). */
export function joinAddenda(addenda: readonly string[]): string {
	return addenda
		.map((text) => text.trim())
		.filter((text) => text.length > 0)
		.join("\n");
}

export function composePrompt(basePrompt: string, activeAddenda: readonly string[]): string {
	const extra = joinAddenda(activeAddenda);
	return extra ? `${basePrompt}\n\nLearned operating guidance (reviewed and approved by an operator):\n${extra}` : basePrompt;
}

// ── Acceptance gate ──────────────────────────────────────────────────────────────────────────────────────

export interface SuiteMetrics {
	readonly evolve: ParetoMetrics;
	readonly heldOut: ParetoMetrics;
}

export interface GateOptions {
	/** Held-out cases required before any candidate may pass. Default 3. */
	readonly minHeldOutCases?: number;
	/** Evolve cases required. Default 1. */
	readonly minEvolveCases?: number;
	/** Required evolve pass-rate gain when the tool-error rate does not improve. Default 0 (any strict gain). */
	readonly minPassRateGain?: number;
}

export type GateVerdict =
	| { readonly decision: "accept"; readonly evolve: ParetoComparison; readonly heldOut: ParetoComparison; readonly reasons: readonly string[] }
	| {
			readonly decision: "reject";
			/** `regression`: made things worse (held-out, safety, tool errors). `no_gain`: safe but did not help. `invalid`: eval set unusable. */
			readonly kind: "regression" | "no_gain" | "invalid";
			readonly evolve: ParetoComparison;
			readonly heldOut: ParetoComparison;
			readonly reasons: readonly string[];
	  };

/**
 * Accepts a candidate only when ALL hold (any one failing rejects):
 *  1. the eval set is big enough on both splits (a tiny set cannot justify a change),
 *  2. zero safety violations in absolute terms, and none newly introduced,
 *  3. held-out split is non-regressing (pass rate, tool errors, safety),
 *  4. evolve split is non-regressing AND strictly better on pass rate or tool-error rate.
 * Latency is reported but never a reason to accept.
 */
export function evaluateGate(candidate: SuiteMetrics, baseline: SuiteMetrics, options: GateOptions = {}): GateVerdict {
	const minHeldOut = options.minHeldOutCases ?? 3;
	const minEvolve = options.minEvolveCases ?? 1;
	const minGain = options.minPassRateGain ?? 0;
	const evolve = comparePareto(candidate.evolve, baseline.evolve);
	const heldOut = comparePareto(candidate.heldOut, baseline.heldOut);
	const reasons: string[] = [];

	if (candidate.heldOut.caseCount < minHeldOut || candidate.evolve.caseCount < minEvolve) {
		return {
			decision: "reject",
			kind: "invalid",
			evolve,
			heldOut,
			reasons: [
				`eval set too small: ${candidate.evolve.caseCount} evolve (need ${minEvolve}), ${candidate.heldOut.caseCount} held-out (need ${minHeldOut})`,
			],
		};
	}

	if (candidate.evolve.safetyViolations + candidate.heldOut.safetyViolations > 0) {
		reasons.push("candidate run had safety violations");
	}
	if (!heldOut.nonRegressing) reasons.push("held-out split regressed");
	if (!evolve.nonRegressing) reasons.push("evolve split regressed");
	if (reasons.length > 0) return { decision: "reject", kind: "regression", evolve, heldOut, reasons };

	const passGain = evolve.deltas.passRate > minGain;
	const toolErrorGain = evolve.deltas.toolErrorRate < 0;
	if (!passGain && !toolErrorGain) {
		return { decision: "reject", kind: "no_gain", evolve, heldOut, reasons: ["no strict gain on the evolve split"] };
	}
	return {
		decision: "accept",
		evolve,
		heldOut,
		reasons: [
			passGain ? `evolve pass rate +${(evolve.deltas.passRate * 100).toFixed(1)} points` : "evolve tool-error rate improved",
			"held-out non-regressing",
			"zero safety violations",
		],
	};
}
