import fs from "node:fs";
import path from "node:path";
import type { TranscriptEntryView } from "../contracts/harness-port.js";
import type { TenantContext } from "../contracts/tenant-context.js";
import type { ScriptedTurnOutput } from "../runtime/model-catalog.js";
import {
	comparePareto,
	type EvalCase,
	type EvalSeedTurn,
	type EvalSplit,
	type ParetoMetrics,
} from "./types.js";

/**
 * Eval-set tooling: validated loading of `evals/cases/*.json`, building cases from real conversation
 * transcripts (with PII redaction), and a baseline file that CI compares every change against.
 */

export class EvalSetError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "EvalSetError";
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown, where: string): string[] | undefined {
	if (value === undefined) return undefined;
	if (!Array.isArray(value) || value.some((v) => typeof v !== "string")) {
		throw new EvalSetError(`${where} must be an array of strings`);
	}
	return value as string[];
}

function parseScripted(value: unknown, where: string): ScriptedTurnOutput[] | undefined {
	if (value === undefined) return undefined;
	if (!Array.isArray(value)) throw new EvalSetError(`${where} must be an array`);
	return value.map((raw, i) => {
		const at = `${where}[${i}]`;
		if (!isRecord(raw)) throw new EvalSetError(`${at} must be an object`);
		if (raw.type === "text") {
			if (typeof raw.text !== "string") throw new EvalSetError(`${at}.text must be a string`);
			return { type: "text", text: raw.text };
		}
		if (raw.type === "error") {
			if (typeof raw.errorMessage !== "string") throw new EvalSetError(`${at}.errorMessage must be a string`);
			return { type: "error", errorMessage: raw.errorMessage };
		}
		if (raw.type === "tool_calls") {
			if (!Array.isArray(raw.calls)) throw new EvalSetError(`${at}.calls must be an array`);
			const calls = raw.calls.map((call, j) => {
				if (!isRecord(call) || typeof call.name !== "string" || !isRecord(call.arguments)) {
					throw new EvalSetError(`${at}.calls[${j}] needs a string name and an object arguments`);
				}
				return {
					...(typeof call.id === "string" ? { id: call.id } : {}),
					name: call.name,
					arguments: call.arguments,
				};
			});
			return { type: "tool_calls", calls, ...(typeof raw.text === "string" ? { text: raw.text } : {}) };
		}
		throw new EvalSetError(`${at}.type must be "text", "tool_calls" or "error"`);
	});
}

function parseTenant(value: unknown, where: string): TenantContext {
	if (
		!isRecord(value) ||
		typeof value.tenantId !== "string" ||
		typeof value.userId !== "string" ||
		!value.tenantId ||
		!value.userId
	) {
		throw new EvalSetError(`${where} needs non-empty string tenantId and userId`);
	}
	const permissions = stringArray(value.permissions, `${where}.permissions`);
	if (!permissions) throw new EvalSetError(`${where}.permissions is required`);
	return { tenantId: value.tenantId, userId: value.userId, permissions };
}

/** Validates one untrusted JSON value as an {@link EvalCase}. Throws {@link EvalSetError} with the path of the problem. */
export function parseEvalCase(raw: unknown, where = "case"): EvalCase {
	if (!isRecord(raw)) throw new EvalSetError(`${where} must be an object`);
	for (const key of ["id", "name", "evalPrompt"] as const) {
		if (typeof raw[key] !== "string" || !(raw[key] as string).trim()) {
			throw new EvalSetError(`${where}.${key} must be a non-empty string`);
		}
	}
	if (raw.split !== "evolve" && raw.split !== "held_out") {
		throw new EvalSetError(`${where}.split must be "evolve" or "held_out"`);
	}
	let seedTurns: EvalSeedTurn[] | undefined;
	if (raw.seedTurns !== undefined) {
		if (!Array.isArray(raw.seedTurns)) throw new EvalSetError(`${where}.seedTurns must be an array`);
		seedTurns = raw.seedTurns.map((seed, i) => {
			if (!isRecord(seed) || typeof seed.userPrompt !== "string") {
				throw new EvalSetError(`${where}.seedTurns[${i}].userPrompt must be a string`);
			}
			const scripted = parseScripted(seed.scriptedResponses, `${where}.seedTurns[${i}].scriptedResponses`);
			return { userPrompt: seed.userPrompt, ...(scripted ? { scriptedResponses: scripted } : {}) };
		});
	}
	for (const key of ["maxTurns", "maxLatencyMs"] as const) {
		if (raw[key] !== undefined && (typeof raw[key] !== "number" || !Number.isFinite(raw[key]) || (raw[key] as number) <= 0)) {
			throw new EvalSetError(`${where}.${key} must be a positive number`);
		}
	}
	const scripted = parseScripted(raw.scriptedResponses, `${where}.scriptedResponses`);
	const expectedSubstrings = stringArray(raw.expectedSubstrings, `${where}.expectedSubstrings`);
	const forbiddenSubstrings = stringArray(raw.forbiddenSubstrings, `${where}.forbiddenSubstrings`);
	const expectedToolNames = stringArray(raw.expectedToolNames, `${where}.expectedToolNames`);
	return {
		id: raw.id as string,
		name: raw.name as string,
		split: raw.split as EvalSplit,
		tenant: parseTenant(raw.tenant, `${where}.tenant`),
		...(seedTurns ? { seedTurns } : {}),
		evalPrompt: raw.evalPrompt as string,
		...(scripted ? { scriptedResponses: scripted } : {}),
		...(expectedSubstrings ? { expectedSubstrings } : {}),
		...(forbiddenSubstrings ? { forbiddenSubstrings } : {}),
		...(expectedToolNames ? { expectedToolNames } : {}),
		...(typeof raw.maxTurns === "number" ? { maxTurns: raw.maxTurns } : {}),
		...(typeof raw.maxLatencyMs === "number" ? { maxLatencyMs: raw.maxLatencyMs } : {}),
	};
}

/** A case file holds one case object or an array of them. */
export function parseEvalCases(raw: unknown, where = "cases"): EvalCase[] {
	const list = Array.isArray(raw) ? raw : [raw];
	return list.map((item, i) => parseEvalCase(item, Array.isArray(raw) ? `${where}[${i}]` : where));
}

export interface LoadedEvalSet {
	readonly cases: readonly EvalCase[];
	readonly files: readonly string[];
}

/**
 * Loads every `*.json` under `<dir>` (non-recursive, sorted for determinism). Case ids must be unique across files.
 * A missing directory yields an empty set; a malformed file throws, so a broken eval set can never pass silently.
 */
export function loadEvalSet(dir: string): LoadedEvalSet {
	if (!fs.existsSync(dir)) return { cases: [], files: [] };
	const files = fs
		.readdirSync(dir)
		.filter((name) => name.endsWith(".json"))
		.sort();
	const cases: EvalCase[] = [];
	const seen = new Set<string>();
	for (const name of files) {
		const file = path.join(dir, name);
		let json: unknown;
		try {
			json = JSON.parse(fs.readFileSync(file, "utf8"));
		} catch (error) {
			throw new EvalSetError(`${name}: invalid JSON (${error instanceof Error ? error.message : String(error)})`);
		}
		for (const evalCase of parseEvalCases(json, name)) {
			if (seen.has(evalCase.id)) throw new EvalSetError(`${name}: duplicate eval case id "${evalCase.id}"`);
			seen.add(evalCase.id);
			cases.push(evalCase);
		}
	}
	return { cases, files };
}

// ── Transcript → eval case ───────────────────────────────────────────────────────────────────────────────

const REDACTIONS: readonly { readonly pattern: RegExp; readonly replacement: string }[] = [
	{ pattern: /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g, replacement: "[email]" },
	{ pattern: /\b(?:sk|pk|rk|ghp|gho|xox[abp])[-_][A-Za-z0-9_-]{16,}\b/g, replacement: "[secret]" },
	{ pattern: /\bBearer\s+[A-Za-z0-9._~+/=-]{16,}/gi, replacement: "Bearer [secret]" },
	{ pattern: /\b(?:\d[ -]?){13,19}\b/g, replacement: "[number]" },
	{ pattern: /\+?\d[\d\s().-]{8,}\d/g, replacement: "[phone]" },
];

/** Strips emails, API-key-shaped tokens, bearer tokens, card/phone-like digit runs. Not a substitute for review. */
export function redactForEval(text: string): string {
	let out = text;
	for (const rule of REDACTIONS) out = out.replace(rule.pattern, rule.replacement);
	return out;
}

export interface TranscriptCaseOptions {
	readonly id: string;
	readonly name: string;
	readonly split: EvalSplit;
	readonly tenant: TenantContext;
	/** Substrings a reviewer confirmed a good answer must contain. Never invented from the model's own answer. */
	readonly expectedSubstrings?: readonly string[];
	readonly forbiddenSubstrings?: readonly string[];
	readonly maxTurns?: number;
}

/**
 * Builds a live (non-scripted) eval case from a recorded conversation: earlier user turns become seed turns, the
 * last user message becomes the eval prompt, and the successful tool calls after it become `expectedToolNames`.
 * All text is passed through {@link redactForEval}. Scripted responses are deliberately not recorded: such a case
 * is meant for a real model, and a human reviews the file before it is committed.
 */
export function buildEvalCaseFromTranscript(
	entries: readonly TranscriptEntryView[],
	options: TranscriptCaseOptions,
): EvalCase {
	let lastUser = -1;
	for (let i = entries.length - 1; i >= 0; i--) {
		if (entries[i]!.role === "user" && entries[i]!.text) {
			lastUser = i;
			break;
		}
	}
	if (lastUser < 0) throw new EvalSetError("Transcript has no user message to build an eval case from");

	const seedTurns: EvalSeedTurn[] = entries
		.slice(0, lastUser)
		.filter((e) => e.role === "user" && e.text)
		.map((e) => ({ userPrompt: redactForEval(e.text!) }));

	const tools: string[] = [];
	for (const entry of entries.slice(lastUser + 1)) {
		if (entry.role === "toolResult" && entry.toolName && !entry.isError && !tools.includes(entry.toolName)) {
			tools.push(entry.toolName);
		}
	}

	return {
		id: options.id,
		name: options.name,
		split: options.split,
		tenant: options.tenant,
		...(seedTurns.length > 0 ? { seedTurns } : {}),
		evalPrompt: redactForEval(entries[lastUser]!.text!),
		...(options.expectedSubstrings?.length ? { expectedSubstrings: [...options.expectedSubstrings] } : {}),
		...(options.forbiddenSubstrings?.length ? { forbiddenSubstrings: [...options.forbiddenSubstrings] } : {}),
		...(tools.length > 0 ? { expectedToolNames: tools } : {}),
		...(options.maxTurns !== undefined ? { maxTurns: options.maxTurns } : {}),
	};
}

// ── Baseline + regression gate ───────────────────────────────────────────────────────────────────────────

export interface EvalBaseline {
	readonly version: 1;
	readonly updatedAt: string;
	readonly evolve: ParetoMetrics;
	readonly heldOut: ParetoMetrics;
}

export function readBaseline(file: string): EvalBaseline | undefined {
	if (!fs.existsSync(file)) return undefined;
	const raw: unknown = JSON.parse(fs.readFileSync(file, "utf8"));
	if (!isRecord(raw) || raw.version !== 1 || !isRecord(raw.evolve) || !isRecord(raw.heldOut)) {
		throw new EvalSetError(`${file}: not a version 1 eval baseline`);
	}
	return raw as unknown as EvalBaseline;
}

export function writeBaseline(
	file: string,
	metrics: { readonly evolve: ParetoMetrics; readonly heldOut: ParetoMetrics },
	now: Date = new Date(),
): EvalBaseline {
	const baseline: EvalBaseline = { version: 1, updatedAt: now.toISOString(), ...metrics };
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, `${JSON.stringify(baseline, null, 2)}\n`, "utf8");
	return baseline;
}

export interface RegressionReport {
	readonly ok: boolean;
	readonly problems: readonly string[];
}

/**
 * CI gate: the current run must not regress against the committed baseline on either split (pass rate, tool-error
 * rate, safety violations). Latency is intentionally ignored: it is machine-dependent and would make CI flaky.
 * Case counts may only grow; a shrunken eval set is reported so tests cannot be "fixed" by deleting them.
 */
export function checkAgainstBaseline(
	current: { readonly evolve: ParetoMetrics; readonly heldOut: ParetoMetrics },
	baseline: EvalBaseline,
): RegressionReport {
	const problems: string[] = [];
	for (const split of ["evolve", "heldOut"] as const) {
		const now = current[split];
		const was = baseline[split];
		if (now.caseCount < was.caseCount) {
			problems.push(`${split}: eval set shrank from ${was.caseCount} to ${now.caseCount} cases`);
		}
		const cmp = comparePareto(now, was);
		if (cmp.deltas.safetyViolations > 0) problems.push(`${split}: safety violations rose by ${cmp.deltas.safetyViolations}`);
		if (cmp.deltas.passRate < 0) problems.push(`${split}: pass rate fell by ${(-cmp.deltas.passRate * 100).toFixed(1)} points`);
		if (cmp.deltas.toolErrorRate > 0) problems.push(`${split}: tool error rate rose by ${(cmp.deltas.toolErrorRate * 100).toFixed(1)} points`);
	}
	return { ok: problems.length === 0, problems };
}
