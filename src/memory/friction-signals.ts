import type { FrictionSignal, ToolTraceItem } from "../contracts/app-capabilities.js";

export interface FrictionAnalysisInput {
	readonly userMessage: string;
	readonly assistantSummary: string;
	readonly toolTrace: readonly ToolTraceItem[];
	readonly highTurnThreshold?: number;
	/**
	 * Earlier user messages from the same conversation (oldest first). Enables repeated-message / rephrase
	 * detection via token overlap. Optional: without it that detector is skipped.
	 */
	readonly previousUserMessages?: readonly string[];
	/** Wall-clock duration of the whole run, when known (else the sum of tool durations is used). */
	readonly durationMs?: number;
}

export interface FrictionAnalysisResult {
	readonly signals: readonly FrictionSignal[];
	readonly score: number;
}

export type SeverityWeights = Readonly<Record<FrictionSignal["severity"], number>>;

export interface FrictionOptions {
	/** Score weight per severity. Default: low 1, medium 2, high 3. */
	readonly severityWeights?: Partial<SeverityWeights>;
	/** Overrides `input.highTurnThreshold`. Default 6 tool calls. */
	readonly highTurnThreshold?: number;
	/** Run (or summed tool) duration at/above which a `slow_run` signal fires. Default 60_000 ms. */
	readonly slowRunMs?: number;
	/** Jaccard token overlap at/above which the user message counts as a rephrase of a previous one. Default 0.6. */
	readonly repetitionOverlap?: number;
	/** Trailing consecutive failing tool calls that count as an abandoned loop. Default 2. */
	readonly abandonedLoopMinErrors?: number;
	/** Only the last N previous user messages are compared. Default 5. */
	readonly maxPreviousMessages?: number;
}

export const DEFAULT_SEVERITY_WEIGHTS: SeverityWeights = { low: 1, medium: 2, high: 3 };

/** Legacy regex cues (all linear-time: bounded alternations, no nested quantifiers). */
const USER_CORRECTION_PATTERNS: readonly RegExp[] = [
	/\bno[,\s]+(?:actually|i meant|that'?s not|don'?t|do not|never)\b/i,
	/\b(?:instead of|rather than|stop using|don'?t use|do not use)\b/i,
	/\b(?:you got that wrong|that is wrong|that'?s incorrect|wrong approach)\b/i,
	/\b(?:please remember|remember that|from now on|always prefer|always use)\b/i,
];

/**
 * Substring cues (checked with `includes` on a lowercased, apostrophe-normalized message: no regex, so no ReDoS
 * surface). Covers English plus a few cheap multi-language / CJK equivalents.
 */
const CORRECTION_CUES: readonly string[] = [
	"that's wrong",
	"thats wrong",
	"that is wrong",
	"that's not right",
	"that isn't right",
	"not what i asked",
	"not what i wanted",
	"not what i meant",
	"not what i said",
	"you forgot",
	"you missed",
	"you ignored",
	"you misunderstood",
	"i already told you",
	"i told you",
	"still wrong",
	"wrong again",
	"that didn't work",
	"it didn't work",
	"doesn't work",
	"does not work",
	// es / pt / fr / de / it
	"eso está mal",
	"eso esta mal",
	"no es lo que pedí",
	"no es lo que pedi",
	"está errado",
	"esta errado",
	"não é o que pedi",
	"c'est faux",
	"ce n'est pas ce que",
	"das ist falsch",
	"nicht was ich",
	"non è quello che",
	// CJK
	"不对",
	"不是我要的",
	"错了",
	"錯了",
	"間違",
	"違う",
	"違います",
	"틀렸",
];

const RETRY_CUES: readonly string[] = [
	"try again",
	"do it again",
	"once more",
	"redo it",
	"retry",
	"start over",
	"inténtalo de nuevo",
	"intentalo de nuevo",
	"tente novamente",
	"réessaie",
	"reessaie",
	"versuch es noch mal",
	"versuche es erneut",
	"もう一度",
	"重新",
	"再试一次",
	"다시 해",
];

const AUTH_BLOCK_PREFIXES: readonly string[] = [
	"authorization denied",
	"security violation",
	"missing conversation tenant metadata",
];

function normalizeText(text: string): string {
	return text
		.slice(0, 4000)
		.normalize("NFKC")
		.toLowerCase()
		.replace(/[‘’ʼ`]/g, "'");
}

/** Letter/number tokens; CJK runs are split into character bigrams so overlap works without whitespace. */
function tokenSet(text: string): Set<string> {
	const normalized = normalizeText(text);
	const out = new Set<string>();
	const words = normalized.match(/[\p{L}\p{N}]+/gu) ?? [];
	for (const word of words) {
		if (/[぀-ヿ㐀-鿿가-힯]/u.test(word)) {
			const chars = Array.from(word);
			if (chars.length === 1) out.add(chars[0]!);
			for (let i = 0; i + 1 < chars.length; i++) out.add(`${chars[i]}${chars[i + 1]}`);
		} else if (word.length > 1) {
			out.add(word);
		}
	}
	return out;
}

function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
	if (a.size === 0 || b.size === 0) return 0;
	let intersection = 0;
	const [small, large] = a.size <= b.size ? [a, b] : [b, a];
	for (const token of small) if (large.has(token)) intersection += 1;
	return intersection / (a.size + b.size - intersection);
}

function isAuthBlocked(item: ToolTraceItem): boolean {
	if (item.diagnosticCodes.some((code) => code.toLowerCase() === "blocked")) return true;
	const text = item.resultText;
	if (!text) return false;
	let head = text.trimStart().slice(0, 160).toLowerCase();
	// Pi Durable renders beforeTool `{ block }` as "Tool call blocked: <reason>" with diagnostic code "blocked".
	if (head.startsWith("tool call blocked:")) head = head.slice("tool call blocked:".length).trimStart();
	return AUTH_BLOCK_PREFIXES.some((prefix) => head.startsWith(prefix));
}

/**
 * Deterministic pre-LLM friction detector (Loop 1).
 *
 * Performs zero LLM calls. Tags tool errors, repeated identical calls, authorization blocks, output
 * truncation/offloading, user corrections and retry requests, repeated/rephrased user messages, abandoned
 * failing tool loops, slow runs, and excessive turn loops so Loop 2 sleep-cycle consolidation can prioritize
 * high-signal episodes.
 */
export function detectFrictionSignals(input: FrictionAnalysisInput, options?: FrictionOptions): FrictionAnalysisResult {
	const signals: FrictionSignal[] = [];
	const highTurnThreshold = options?.highTurnThreshold ?? input.highTurnThreshold ?? 6;
	const weights: SeverityWeights = { ...DEFAULT_SEVERITY_WEIGHTS, ...options?.severityWeights };
	const slowRunMs = options?.slowRunMs ?? 60_000;
	const repetitionOverlap = options?.repetitionOverlap ?? 0.6;
	const abandonedMin = Math.max(1, options?.abandonedLoopMinErrors ?? 2);
	const maxPrevious = Math.max(1, options?.maxPreviousMessages ?? 5);

	// 1. User correction / explicit steering language (strongest cue wins; one signal per message)
	const normalizedMessage = normalizeText(input.userMessage);
	const correctionCue = CORRECTION_CUES.find((cue) => normalizedMessage.includes(cue));
	let correctionDetail: string | undefined = correctionCue ? `"${correctionCue}"` : undefined;
	if (!correctionDetail) {
		for (const pattern of USER_CORRECTION_PATTERNS) {
			const match = input.userMessage.slice(0, 4000).match(pattern);
			if (match) {
				correctionDetail = `"${match[0]}"`;
				break;
			}
		}
	}
	if (correctionDetail) {
		signals.push({
			kind: "user_correction",
			severity: "high",
			detail: `User message matched correction/preference cue: ${correctionDetail}`,
		});
	} else {
		const retryCue = RETRY_CUES.find((cue) => normalizedMessage.includes(cue));
		if (retryCue) {
			signals.push({
				kind: "user_correction",
				severity: "medium",
				detail: `User message asked to retry: "${retryCue}"`,
			});
		}
	}

	// 1b. Repeated / rephrased user message across turns
	if (input.previousUserMessages && input.previousUserMessages.length > 0) {
		const current = tokenSet(input.userMessage);
		let best = 0;
		for (const previous of input.previousUserMessages.slice(-maxPrevious)) {
			best = Math.max(best, jaccard(current, tokenSet(previous)));
		}
		// Very short messages ("ok", "yes") overlap trivially; require a few tokens.
		if (current.size >= 3 && best >= repetitionOverlap) {
			signals.push({
				kind: "user_repetition",
				severity: best >= 0.9 ? "high" : "medium",
				detail: `User message repeats/rephrases an earlier request (token overlap ${best.toFixed(2)})`,
			});
		}
	}

	// 2. Tool trace inspection
	const signatureCounts = new Map<string, { count: number; toolName: string }>();

	for (const item of input.toolTrace) {
		if (isAuthBlocked(item)) {
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

	// 3b. Abandoned tool loop: the run ended while the trailing calls were still failing
	let trailingErrors = 0;
	for (let i = input.toolTrace.length - 1; i >= 0; i--) {
		const item = input.toolTrace[i]!;
		if (!item.isError) break;
		trailingErrors += 1;
	}
	if (trailingErrors >= abandonedMin) {
		const last = input.toolTrace[input.toolTrace.length - 1]!;
		signals.push({
			kind: "abandoned_tool_loop",
			severity: "high",
			toolName: last.toolName,
			detail: `Run ended with ${trailingErrors} consecutive failing tool calls (last: "${last.toolName}")`,
		});
	}

	// 4. High turn / tool count
	if (input.toolTrace.length >= highTurnThreshold) {
		signals.push({
			kind: "high_turn_count",
			severity: "medium",
			detail: `Run executed ${input.toolTrace.length} tool calls (threshold ${highTurnThreshold})`,
		});
	}

	// 5. Long latency
	const toolDurationMs = input.toolTrace.reduce((sum, item) => sum + (item.durationMs ?? 0), 0);
	const totalMs = Math.max(input.durationMs ?? 0, toolDurationMs);
	if (slowRunMs > 0 && totalMs >= slowRunMs) {
		signals.push({
			kind: "slow_run",
			severity: totalMs >= slowRunMs * 3 ? "medium" : "low",
			detail: `Run took ${Math.round(totalMs)}ms (threshold ${slowRunMs}ms)`,
		});
	}

	const score = signals.reduce((sum, signal) => sum + weights[signal.severity], 0);
	return { signals, score };
}
