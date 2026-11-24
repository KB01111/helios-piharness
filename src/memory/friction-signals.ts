import type { FrictionSignal, ToolTraceItem } from "../contracts/app-capabilities.js";

export interface FrictionAnalysisInput {
	readonly userMessage: string;
	readonly assistantSummary: string;
	readonly toolTrace: readonly ToolTraceItem[];
	readonly highTurnThreshold?: number;
}

export interface FrictionAnalysisResult {
	readonly signals: readonly FrictionSignal[];
	readonly score: number;
}

const USER_CORRECTION_PATTERNS: readonly RegExp[] = [
	/\bno[,\s]+(?:actually|i meant|that'?s not|don'?t|do not|never)\b/i,
	/\b(?:instead of|rather than|stop using|don'?t use|do not use)\b/i,
	/\b(?:you got that wrong|that is wrong|that'?s incorrect|wrong approach)\b/i,
	/\b(?:please remember|remember that|from now on|always prefer|always use)\b/i,
];

const SEVERITY_WEIGHT: Readonly<Record<FrictionSignal["severity"], number>> = {
	low: 1,
	medium: 2,
	high: 3,
};

/**
 * Deterministic pre-LLM friction detector (Loop 1).
 *
 * Performs zero LLM calls. Tags tool errors, repeated identical calls,
 * authorization blocks, output truncation/offloading, user corrections,
 * and excessive turn loops so Loop 2 sleep-cycle consolidation can prioritize
 * high-signal episodes.
 */
export function detectFrictionSignals(input: FrictionAnalysisInput): FrictionAnalysisResult {
	const signals: FrictionSignal[] = [];
	const highTurnThreshold = input.highTurnThreshold ?? 6;

	// 1. User correction / explicit steering language
	for (const pattern of USER_CORRECTION_PATTERNS) {
		const match = input.userMessage.match(pattern);
		if (match) {
			signals.push({
				kind: "user_correction",
				severity: "high",
				detail: `User message matched correction/preference cue: "${match[0]}"`,
			});
			break;
		}
	}

	// 2. Tool trace inspection
	const signatureCounts = new Map<string, { count: number; toolName: string }>();

	for (const item of input.toolTrace) {
		if (item.diagnosticCodes.includes("blocked")) {
			signals.push({
				kind: "auth_blocked",
				severity: "high",
				toolName: item.toolName,
				detail: `Tool "${item.toolName}" was blocked by authorization guard`,
			});
		} else if (item.isError) {
			signals.push({
				kind: "tool_error",
				severity: "medium",
				toolName: item.toolName,
				detail: `Tool "${item.toolName}" failed (${item.diagnosticCodes.join(", ") || "error"})`,
			});
		}

		if (item.diagnosticCodes.includes("artifact_offloaded")) {
			signals.push({
				kind: "output_truncated",
				severity: "low",
				toolName: item.toolName,
				detail: `Tool "${item.toolName}" exceeded output cap and was offloaded to artifact storage`,
			});
		}

		const sig = `${item.toolName}::${item.argumentsJson}`;
		const entry = signatureCounts.get(sig);
		if (entry) {
			entry.count += 1;
		} else {
			signatureCounts.set(sig, { count: 1, toolName: item.toolName });
		}
	}

	// 3. Repeated identical tool calls (thrashing / retry loop)
	for (const { count, toolName } of signatureCounts.values()) {
		if (count >= 2) {
			signals.push({
				kind: "repeated_tool_call",
				severity: count >= 3 ? "high" : "medium",
				toolName,
				detail: `Tool "${toolName}" was invoked ${count} times with identical arguments in one run`,
			});
		}
	}

	// 4. High turn / tool count
	if (input.toolTrace.length >= highTurnThreshold) {
		signals.push({
			kind: "high_turn_count",
			severity: "medium",
			detail: `Run executed ${input.toolTrace.length} tool calls (threshold ${highTurnThreshold})`,
		});
	}

	const score = signals.reduce((sum, signal) => sum + SEVERITY_WEIGHT[signal.severity], 0);
	return { signals, score };
}
