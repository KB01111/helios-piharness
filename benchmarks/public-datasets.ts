/**
 * Public long-term-memory benchmarks, normalised to "scopes of timestamped memories + queries with gold ids".
 *
 *  - LoCoMo (Maharana et al., 2024; snap-research/locomo): 10 long multi-session conversations, 1,986 QA with
 *    evidence dialogue ids. Categories: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
 *    Category 5 is excluded from retrieval scoring (its "evidence" belongs to the other speaker by design).
 *  - LongMemEval_S (Wu et al., 2025; MIT): 500 questions, each with its own ~115k-token chat history (~50
 *    sessions). Gold = answer session ids; `_abs` questions are unanswerable (abstention).
 *
 * The files are downloaded on first use into `data/benchmarks/` (gitignored, never vendored) and verified
 * against pinned SHA-256 digests.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

export interface PublicMemory {
	readonly id: string;
	readonly key: string;
	readonly content: string;
	readonly timestampMs: number;
	/** Grouping id used for session-level scoring (LongMemEval). */
	readonly sessionId: string;
}

export interface PublicQuery {
	readonly id: string;
	readonly category: string;
	readonly query: string;
	readonly timestampMs: number;
	/** Gold memory ids (turn level). Empty for abstention. */
	readonly goldMemoryIds: readonly string[];
	/** Gold session ids (LongMemEval session-level scoring). */
	readonly goldSessionIds: readonly string[];
	readonly expectAbstain: boolean;
}

export interface PublicScope {
	readonly scopeId: string;
	readonly memories: readonly PublicMemory[];
	readonly queries: readonly PublicQuery[];
}

export interface PublicDataset {
	readonly name: string;
	readonly scopes: readonly PublicScope[];
	readonly notes: readonly string[];
}

const SOURCES = {
	locomo: {
		file: "locomo10.json",
		url: "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
		sha256: "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
	},
	longmemeval: {
		file: "longmemeval_s_cleaned.json",
		url: "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
		sha256: "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
	},
} as const;

function sha256File(file: string): string {
	return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

/** Returns the local path of a dataset, downloading and verifying it on first use. */
export async function ensureDataset(name: keyof typeof SOURCES, dir = path.resolve("data/benchmarks")): Promise<string> {
	const src = SOURCES[name];
	const file = path.join(dir, src.file);
	if (!fs.existsSync(file)) {
		fs.mkdirSync(dir, { recursive: true });
		console.error(`[bench] downloading ${src.url}`);
		const res = await fetch(src.url);
		if (!res.ok) throw new Error(`download failed: HTTP ${res.status} for ${src.url}`);
		const tmp = `${file}.part`;
		fs.writeFileSync(tmp, Buffer.from(await res.arrayBuffer()));
		fs.renameSync(tmp, file);
	}
	const digest = sha256File(file);
	if (digest !== src.sha256) {
		throw new Error(`${file} has sha256 ${digest}, expected ${src.sha256}; delete it to re-download`);
	}
	return file;
}

const MONTHS: Record<string, number> = {
	january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11,
};

/** LoCoMo session stamps look like "1:56 pm on 8 May, 2023". */
export function parseLocomoDate(raw: string): number {
	const m = /(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})/i.exec(raw);
	if (!m) throw new Error(`unparseable LoCoMo date: ${raw}`);
	let hour = Number(m[1]) % 12;
	if (m[3]!.toLowerCase() === "pm") hour += 12;
	const month = MONTHS[m[5]!.toLowerCase()];
	if (month === undefined) throw new Error(`unknown month in ${raw}`);
	return Date.UTC(Number(m[6]), month, Number(m[4]), hour, Number(m[2]));
}

/** LongMemEval stamps look like "2023/05/20 (Sat) 02:21". */
export function parseLongMemEvalDate(raw: string): number {
	const m = /(\d{4})\/(\d{2})\/(\d{2})\s*\([A-Za-z]+\)\s*(\d{2}):(\d{2})/.exec(raw);
	if (!m) throw new Error(`unparseable LongMemEval date: ${raw}`);
	return Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]));
}

function isoDay(ms: number): string {
	return new Date(ms).toISOString().slice(0, 10);
}

const LOCOMO_CATEGORY: Record<number, string> = { 1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial" };

interface LocomoTurn {
	readonly speaker: string;
	readonly dia_id: string;
	readonly text: string;
	readonly blip_caption?: string;
}

export async function loadLocomo(options: { readonly limitConversations?: number } = {}): Promise<PublicDataset> {
	const raw = JSON.parse(fs.readFileSync(await ensureDataset("locomo"), "utf8")) as Array<{
		sample_id: string;
		conversation: Record<string, unknown>;
		qa: Array<{ question: string; evidence?: string[]; category: number }>;
	}>;
	const scopes: PublicScope[] = [];
	let excluded = 0;
	for (const conv of raw.slice(0, options.limitConversations ?? raw.length)) {
		const memories: PublicMemory[] = [];
		const byDia = new Map<string, string>();
		let lastTs = 0;
		for (let k = 1; k < 100; k++) {
			const turns = conv.conversation[`session_${k}`] as LocomoTurn[] | undefined;
			if (!turns) continue;
			const ts = parseLocomoDate(String(conv.conversation[`session_${k}_date_time`]));
			lastTs = Math.max(lastTs, ts);
			turns.forEach((t, i) => {
				const id = `${conv.sample_id}:${t.dia_id}`;
				const photo = t.blip_caption ? ` (shares a photo: ${t.blip_caption})` : "";
				memories.push({
					id,
					key: `dialogue.session_${k}.turn_${i + 1}`,
					content: `[${isoDay(ts)}] ${t.speaker}: ${t.text}${photo}`,
					// Turns within a session are a minute apart so ordering is preserved.
					timestampMs: ts + i * 60_000,
					sessionId: `session_${k}`,
				});
				byDia.set(t.dia_id, id);
			});
		}
		const queries: PublicQuery[] = [];
		conv.qa.forEach((qa, i) => {
			if (qa.category === 5) {
				excluded += 1;
				return;
			}
			const gold = (qa.evidence ?? [])
				.flatMap((e) => String(e).split(/[;,\s]+/))
				.map((e) => byDia.get(e.trim()))
				.filter((x): x is string => Boolean(x));
			if (gold.length === 0) return; // malformed evidence: unscorable
			queries.push({
				id: `${conv.sample_id}:q${i}`,
				category: LOCOMO_CATEGORY[qa.category] ?? `cat_${qa.category}`,
				query: qa.question,
				timestampMs: lastTs + 86_400_000,
				goldMemoryIds: gold,
				goldSessionIds: [],
				expectAbstain: false,
			});
		});
		scopes.push({ scopeId: conv.sample_id, memories, queries });
	}
	return {
		name: "locomo",
		scopes,
		notes: [`category 5 (adversarial, ${excluded} questions) excluded from retrieval scoring`, "memory unit = one dialogue turn, prefixed with its session date"],
	};
}

interface LmeQuestion {
	question_id: string;
	question_type: string;
	question: string;
	question_date: string;
	answer_session_ids: string[];
	haystack_dates: string[];
	haystack_session_ids: string[];
	haystack_sessions: Array<Array<{ role: string; content: string; has_answer?: boolean }>>;
}

/**
 * Deterministic stratified subset: ~20% abstention questions, the rest proportional to question type,
 * each stratum taken in question_id order. `limit >= 500` loads everything.
 */
export function selectLongMemEval(all: readonly LmeQuestion[], limit: number): LmeQuestion[] {
	if (limit >= all.length) return [...all];
	const sorted = [...all].sort((a, b) => (a.question_id < b.question_id ? -1 : 1));
	const abs = sorted.filter((q) => q.question_id.endsWith("_abs"));
	const rest = sorted.filter((q) => !q.question_id.endsWith("_abs"));
	const absTake = Math.min(abs.length, Math.max(1, Math.round(limit * 0.2)));
	const restTake = limit - absTake;
	const byType = new Map<string, LmeQuestion[]>();
	for (const q of rest) byType.set(q.question_type, [...(byType.get(q.question_type) ?? []), q]);
	const picked: LmeQuestion[] = abs.slice(0, absTake);
	const types = [...byType.keys()].sort();
	const quotas = types.map((t) => ({ t, n: Math.max(1, Math.floor((restTake * byType.get(t)!.length) / rest.length)) }));
	let total = quotas.reduce((s, q) => s + q.n, 0);
	for (let i = 0; total < restTake; i = (i + 1) % quotas.length, total++) quotas[i]!.n += 1;
	for (const { t, n } of quotas) picked.push(...byType.get(t)!.slice(0, n));
	return picked.slice(0, limit);
}

export async function loadLongMemEval(options: { readonly limit?: number; readonly maxTurnChars?: number } = {}): Promise<PublicDataset> {
	const all = JSON.parse(fs.readFileSync(await ensureDataset("longmemeval"), "utf8")) as LmeQuestion[];
	const selected = selectLongMemEval(all, options.limit ?? 40);
	const maxChars = options.maxTurnChars ?? 1_500;
	const scopes: PublicScope[] = selected.map((q) => {
		const memories: PublicMemory[] = [];
		q.haystack_sessions.forEach((session, s) => {
			const sid = q.haystack_session_ids[s]!;
			const ts = parseLongMemEvalDate(q.haystack_dates[s]!);
			session.forEach((turn, t) => {
				memories.push({
					id: `${sid}#${t}`,
					key: `chat.${turn.role}`,
					content: `[${isoDay(ts)}] ${turn.role}: ${turn.content.slice(0, maxChars)}`,
					timestampMs: ts + t * 30_000,
					sessionId: sid,
				});
			});
		});
		const abstain = q.question_id.endsWith("_abs");
		return {
			scopeId: q.question_id,
			memories,
			queries: [
				{
					id: q.question_id,
					category: abstain ? "abstention" : q.question_type,
					query: q.question,
					timestampMs: parseLongMemEvalDate(q.question_date),
					goldMemoryIds: [],
					goldSessionIds: abstain ? [] : q.answer_session_ids,
					expectAbstain: abstain,
				},
			],
		};
	});
	return {
		name: "longmemeval_s",
		scopes,
		notes: [
			`${selected.length}/${all.length} questions (deterministic stratified subset; --limit 500 for all)`,
			"memory unit = one chat turn (truncated to " + maxChars + " chars); scored at session level",
		],
	};
}
