/**
 * Deterministic, ReDoS-safe temporal extraction, tightness scoring, relative temporal hints,
 * and historical-audit intent routing for Helios memory.
 *
 * Inspired by Cognee-RS's `TemporalHybridRetriever` (`tightness_rank`, `temporal_hints`)
 * and Graphiti's bi-temporal valid-time intervals. Zero external deps, zero LLM calls.
 */

export type TemporalPrecision = "day" | "month" | "season" | "year" | "range";

export interface TemporalAnchor {
	readonly startMs: number;
	readonly endMs: number;
	readonly precision: TemporalPrecision;
	readonly label: string;
}

const MAX_TEMPORAL_CHARS = 2000;
const DAY_MS = 86_400_000;

const MONTHS: Readonly<Record<string, number>> = {
	january: 0, jan: 0,
	february: 1, feb: 1,
	march: 2, mar: 2,
	april: 3, apr: 3,
	may: 4,
	june: 5, jun: 5,
	july: 6, jul: 6,
	august: 7, aug: 7,
	september: 8, sep: 8, sept: 8,
	october: 9, oct: 9,
	november: 10, nov: 10,
	december: 11, dec: 11,
};

const SEASONS: Readonly<Record<string, readonly [number, number]>> = {
	spring: [2, 4], // Mar-May
	summer: [5, 7], // Jun-Aug
	autumn: [8, 10], // Sep-Nov
	fall: [8, 10], // Sep-Nov
	winter: [0, 1], // Jan-Feb (plus Dec of same/prior)
};

function startOfDayUtc(year: number, month: number, day: number): number {
	return Date.UTC(year, month, day, 0, 0, 0, 0);
}

function endOfDayUtc(year: number, month: number, day: number): number {
	return Date.UTC(year, month, day, 23, 59, 59, 999);
}

function startOfMonthUtc(year: number, month: number): number {
	return Date.UTC(year, month, 1, 0, 0, 0, 0);
}

function endOfMonthUtc(year: number, month: number): number {
	return Date.UTC(year, month + 1, 0, 23, 59, 59, 999);
}

function isValidYear(y: number): boolean {
	return Number.isInteger(y) && y >= 1900 && y <= 2100;
}

function isValidDay(year: number, month: number, day: number): boolean {
	if (!Number.isInteger(day) || day < 1 || day > 31) return false;
	const d = new Date(Date.UTC(year, month, day));
	return d.getUTCFullYear() === year && d.getUTCMonth() === month && d.getUTCDate() === day;
}

const MONTH_NAME_GROUP =
	"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)";

/**
 * Extracts explicit temporal intervals (ISO dates, Month DD YYYY, Month YYYY, Season YYYY,
 * year ranges, before/after YYYY, and bare years) from memory text or a query.
 */
export function extractTemporalAnchors(text: string, referenceNowMs?: number): TemporalAnchor[] {
	const bounded = text.length > MAX_TEMPORAL_CHARS ? text.slice(0, MAX_TEMPORAL_CHARS) : text;
	const anchors: TemporalAnchor[] = [];
	const seen = new Set<string>();
	const coveredSpans: Array<[number, number]> = [];

	const isCovered = (start: number, end: number): boolean =>
		coveredSpans.some(([s, e]) => start >= s && end <= e);

	const push = (anchor: TemporalAnchor, matchStart: number, matchEnd: number): void => {
		const key = `${anchor.startMs}:${anchor.endMs}:${anchor.precision}`;
		if (seen.has(key)) return;
		seen.add(key);
		coveredSpans.push([matchStart, matchEnd]);
		anchors.push(anchor);
	};

	// 1. ISO dates: YYYY-MM-DD
	const isoRe = /\b((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b/g;
	for (const m of bounded.matchAll(isoRe)) {
		const year = Number(m[1]);
		const month = Number(m[2]) - 1;
		const day = Number(m[3]);
		if (!isValidYear(year) || !isValidDay(year, month, day)) continue;
		const idx = m.index ?? 0;
		push(
			{ startMs: startOfDayUtc(year, month, day), endMs: endOfDayUtc(year, month, day), precision: "day", label: m[0] },
			idx,
			idx + m[0].length,
		);
	}

	// 2. Named Month DD, YYYY (e.g. "August 11, 1999", "January 20, 2025")
	const mdyRe = new RegExp(`\\b(${MONTH_NAME_GROUP})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,\\s*|\\s+)((?:19|20)\\d{2})\\b`, "gi");
	for (const m of bounded.matchAll(mdyRe)) {
		const idx = m.index ?? 0;
		if (isCovered(idx, idx + m[0].length)) continue;
		const month = MONTHS[m[1]!.toLowerCase()];
		const day = Number(m[2]);
		const year = Number(m[3]);
		if (month === undefined || !isValidYear(year) || !isValidDay(year, month, day)) continue;
		push(
			{ startMs: startOfDayUtc(year, month, day), endMs: endOfDayUtc(year, month, day), precision: "day", label: m[0] },
			idx,
			idx + m[0].length,
		);
	}

	// 3. DD Month YYYY (e.g. "11 August 1999")
	const dmyRe = new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(${MONTH_NAME_GROUP})\\.?\\s+((?:19|20)\\d{2})\\b`, "gi");
	for (const m of bounded.matchAll(dmyRe)) {
		const idx = m.index ?? 0;
		if (isCovered(idx, idx + m[0].length)) continue;
		const day = Number(m[1]);
		const month = MONTHS[m[2]!.toLowerCase()];
		const year = Number(m[3]);
		if (month === undefined || !isValidYear(year) || !isValidDay(year, month, day)) continue;
		push(
			{ startMs: startOfDayUtc(year, month, day), endMs: endOfDayUtc(year, month, day), precision: "day", label: m[0] },
			idx,
			idx + m[0].length,
		);
	}

	// 4. Year ranges: "between 2015 and 2016", "from 2019 to 2021", "2015-2016"
	const rangeRe = /\b(?:between\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})|from\s+((?:19|20)\d{2})\s+(?:to|through|until|-)\s+((?:19|20)\d{2})|((?:19|20)\d{2})\s*[–-]\s*((?:19|20)\d{2}))\b/gi;
	for (const m of bounded.matchAll(rangeRe)) {
		const idx = m.index ?? 0;
		if (isCovered(idx, idx + m[0].length)) continue;
		const y1 = Number(m[1] ?? m[3] ?? m[5]);
		const y2 = Number(m[2] ?? m[4] ?? m[6]);
		if (!isValidYear(y1) || !isValidYear(y2) || y2 < y1) continue;
		push(
			{
				startMs: startOfMonthUtc(y1, 0),
				endMs: endOfMonthUtc(y2, 11),
				precision: "range",
				label: `${y1}-${y2}`,
			},
			idx,
			idx + m[0].length,
		);
	}

	// 5. Month YYYY (e.g. "October 2019", "March 2018")
	const myRe = new RegExp(`\\b(${MONTH_NAME_GROUP})\\.?\\s+((?:19|20)\\d{2})\\b`, "gi");
	for (const m of bounded.matchAll(myRe)) {
		const idx = m.index ?? 0;
		if (isCovered(idx, idx + m[0].length)) continue;
		const month = MONTHS[m[1]!.toLowerCase()];
		const year = Number(m[2]);
		if (month === undefined || !isValidYear(year)) continue;
		push(
			{ startMs: startOfMonthUtc(year, month), endMs: endOfMonthUtc(year, month), precision: "month", label: m[0] },
			idx,
			idx + m[0].length,
		);
	}

	// 6. Season YYYY (e.g. "Spring 2021")
	const seasonRe = /\b(spring|summer|autumn|fall|winter)\s+((?:19|20)\d{2})\b/gi;
	for (const m of bounded.matchAll(seasonRe)) {
		const idx = m.index ?? 0;
		if (isCovered(idx, idx + m[0].length)) continue;
		const bounds = SEASONS[m[1]!.toLowerCase()];
		const year = Number(m[2]);
		if (!bounds || !isValidYear(year)) continue;
		push(
			{
				startMs: startOfMonthUtc(year, bounds[0]),
				endMs: endOfMonthUtc(year, bounds[1]),
				precision: "season",
				label: m[0],
			},
			idx,
			idx + m[0].length,
		);
	}

	// 7. Open-ended year bounds ("before 2018", "since 2020", "after 2020")
	const boundRe = /\b(before|prior\s+to|after|since)\s+((?:19|20)\d{2})\b/gi;
	for (const m of bounded.matchAll(boundRe)) {
		const idx = m.index ?? 0;
		if (isCovered(idx, idx + m[0].length)) continue;
		const dir = m[1]!.toLowerCase();
		const year = Number(m[2]);
		if (!isValidYear(year)) continue;
		if (dir.startsWith("before") || dir.startsWith("prior")) {
			push(
				{
					startMs: startOfMonthUtc(1900, 0),
					endMs: endOfMonthUtc(year - 1, 11),
					precision: "range",
					label: `<${year}`,
				},
				idx,
				idx + m[0].length,
			);
		} else {
			const endMs = referenceNowMs ? Math.max(referenceNowMs, endOfMonthUtc(year, 11)) : endOfMonthUtc(2100, 11);
			push(
				{
					startMs: startOfMonthUtc(year, 0),
					endMs,
					precision: "range",
					label: `>=${year}`,
				},
				idx,
				idx + m[0].length,
			);
		}
	}

	// 8. Standalone years (1900..2099) not already inside a more specific match
	const yearRe = /\b((?:19|20)\d{2})\b/g;
	for (const m of bounded.matchAll(yearRe)) {
		const idx = m.index ?? 0;
		if (isCovered(idx, idx + m[0].length)) continue;
		const year = Number(m[1]);
		if (!isValidYear(year)) continue;
		// Skip port numbers or technical codes like "port 2000" / "max_client_conn 2000" / "ISO 27001"
		const prefixWindow = bounded.slice(Math.max(0, idx - 22), idx).toLowerCase();
		if (/(?:port|conn|limit|budget|iso|sec|ticket|error|code|status|max|min|credits)\s*[:#-]?\s*$/.test(prefixWindow)) {
			continue;
		}
		push(
			{
				startMs: startOfMonthUtc(year, 0),
				endMs: endOfMonthUtc(year, 11),
				precision: "year",
				label: String(year),
			},
			idx,
			idx + m[0].length,
		);
	}

	return anchors;
}

const PRECISION_TIGHTNESS: Readonly<Record<TemporalPrecision, number>> = {
	day: 1.0,
	month: 0.94,
	season: 0.88,
	year: 0.84,
	range: 0.78,
};

/**
 * Scores how well a memory's temporal anchors match a query's temporal anchors.
 * Returns:
 *  - `matched: true` and `score` in (0, 1] (tightness-ranked like Cognee's `tightness_rank`) when overlapping.
 *  - `matched: false, conflict: true` when both query and memory specify explicit dates that do NOT overlap.
 *  - `matched: false, conflict: false` when either side has no temporal anchors.
 */
export function scoreTemporalAlignment(
	queryAnchors: readonly TemporalAnchor[],
	memoryAnchors: readonly TemporalAnchor[],
): { readonly matched: boolean; readonly score: number; readonly conflict: boolean } {
	if (queryAnchors.length === 0 || memoryAnchors.length === 0) {
		return { matched: false, score: 0, conflict: false };
	}

	let best = 0;
	for (const qa of queryAnchors) {
		const qSpan = Math.max(DAY_MS, qa.endMs - qa.startMs);
		for (const ma of memoryAnchors) {
			if (ma.startMs <= qa.endMs && ma.endMs >= qa.startMs) {
				const overlapStart = Math.max(qa.startMs, ma.startMs);
				const overlapEnd = Math.min(qa.endMs, ma.endMs);
				const overlapSpan = Math.max(DAY_MS, overlapEnd - overlapStart);
				const mSpan = Math.max(DAY_MS, ma.endMs - ma.startMs);
				// Containment ratio: memory anchor contained inside query window scores high;
				// coarser anchor spanning way beyond query window scores slightly lower.
				const containment = Math.min(1, overlapSpan / mSpan);
				const coverage = Math.min(1, overlapSpan / qSpan);
				const tightness = PRECISION_TIGHTNESS[ma.precision] ?? 0.8;
				const s = Math.min(1, tightness * (0.65 * containment + 0.35 * Math.sqrt(coverage)));
				if (s > best) best = s;
			}
		}
	}

	if (best > 0) {
		return { matched: true, score: best, conflict: false };
	}
	return { matched: false, score: 0, conflict: true };
}

/**
 * High-precision detector for queries explicitly asking about superseded / historical state
 * (e.g. "What package manager did I use before bun?", "originally run in", "previously used",
 * "prior to Incident.io", "used to use").
 *
 * Deliberately does NOT trigger on present-tense procedural ordering ("before committing",
 * "before refactoring") so active-only search invariants remain strict!
 */
export interface HistoricalIntent {
	readonly isHistorical: boolean;
	/** Optional lowercase tokens naming the *current* state referenced after "before/prior to" (e.g. "bun", "obsidian"). */
	readonly excludeCurrentHintTokens: readonly string[];
}

const STRONG_HISTORICAL_RE =
	/\b(?:previously|formerly|originally|prior\s+to|used\s+to|in\s+the\s+past|before\s+(?:switching|migrating|moving|upgrading|replacing|changing)\s+to|what\s+(?:[\p{L}\p{N}.-]+\s+){1,5}did\s+(?:i|we|you|they|it)\s+(?:use|run|have|prefer|host|store)\s+before|what\s+was\s+(?:my|our|the)\s+(?:old|previous|original|prior|former)\b)/iu;

const AFTER_PIVOT_RE =
	/\b(?:before\s+(?:switching\s+to|migrating\s+to|moving\s+to|upgrading\s+to)?|prior\s+to)\s+([A-Za-z0-9._-]+(?:\s+[A-Za-z0-9._-]+)?)/i;

export function detectHistoricalIntent(query: string): HistoricalIntent {
	const bounded = query.length > MAX_TEMPORAL_CHARS ? query.slice(0, MAX_TEMPORAL_CHARS) : query;
	if (!STRONG_HISTORICAL_RE.test(bounded)) {
		return { isHistorical: false, excludeCurrentHintTokens: [] };
	}

	const hints: string[] = [];
	const m = AFTER_PIVOT_RE.exec(bounded);
	if (m && m[1]) {
		const toks = m[1]
			.toLowerCase()
			.split(/[^a-z0-9]+/)
			.filter((t) => t.length >= 2 && !/^(?:the|to|in|on|at|for|my|our|we|i)$/.test(t));
		hints.push(...toks);
	}

	return { isHistorical: true, excludeCurrentHintTokens: hints };
}

/**
 * Generates deterministic relative-date resolution hints for consolidation prompts (Cognee-style
 * `temporal_hints`), converting words like "yesterday", "today", "last week", "last month" into
 * exact ISO dates anchored to the episode's `createdAt` timestamp.
 */
export function buildRelativeTemporalHints(text: string, referenceMs: number): string[] {
	const bounded = text.length > MAX_TEMPORAL_CHARS ? text.slice(0, MAX_TEMPORAL_CHARS) : text;
	const lower = bounded.toLowerCase();
	const hints: string[] = [];
	const fmt = (ms: number): string => new Date(ms).toISOString().slice(0, 10);

	if (/\btoday\b/.test(lower)) {
		hints.push(`"today" = ${fmt(referenceMs)}`);
	}
	if (/\byesterday\b/.test(lower)) {
		hints.push(`"yesterday" = ${fmt(referenceMs - DAY_MS)}`);
	}
	if (/\btomorrow\b/.test(lower)) {
		hints.push(`"tomorrow" = ${fmt(referenceMs + DAY_MS)}`);
	}
	if (/\blast\s+week\b/.test(lower)) {
		hints.push(`"last week" ≈ ${fmt(referenceMs - 7 * DAY_MS)}..${fmt(referenceMs - DAY_MS)}`);
	}
	if (/\blast\s+month\b/.test(lower)) {
		const d = new Date(referenceMs);
		const prev = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() - 1, 1));
		hints.push(`"last month" = ${prev.toISOString().slice(0, 7)}`);
	}
	if (/\bthis\s+year\b/.test(lower)) {
		hints.push(`"this year" = ${new Date(referenceMs).getUTCFullYear()}`);
	}
	if (/\blast\s+year\b/.test(lower)) {
		hints.push(`"last year" = ${new Date(referenceMs).getUTCFullYear() - 1}`);
	}
	return hints;
}
