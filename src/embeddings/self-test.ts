/**
 * Embedding quality self-test + automatic cosine calibration on a small NEUTRAL corpus (general-knowledge
 * paraphrase pairs and distractors; no overlap with any benchmark or user data).
 *
 *   floor       = 95th percentile of query->document cosines between UNRELATED texts
 *   ceiling     = median cosine of matching (paraphrase) pairs
 *   separation  = fraction of matching pairs scoring above `floor`
 *   top1        = fraction of queries whose own document ranks first among all documents
 *
 * A healthy retrieval model scores separation >= 0.8 and top1 >= 0.8. A backend below the bar (e.g. a broken
 * NPU export whose vectors are near-isotropic) is quarantined by the server instead of polluting memory.
 */

import type { EmbeddingInputType, SimilarityCalibration } from "./types.js";

/** [query, matching document] — paraphrases with little lexical overlap. */
export const SELF_TEST_PAIRS: ReadonlyArray<readonly [string, string]> = [
	["how tall is the tallest mountain on earth", "Mount Everest rises 8,849 metres above sea level."],
	["best way to keep bread from going stale", "Store loaves in a cloth bag at room temperature and freeze what you cannot eat in two days."],
	["which planet has the most moons", "Saturn holds the record with well over one hundred confirmed natural satellites."],
	["symptoms of not drinking enough water", "Dehydration causes headaches, dizziness, dark urine and fatigue."],
	["how do I make my laptop battery last longer", "Lower screen brightness and close background apps to extend notebook runtime."],
	["what language is spoken in Brazil", "Portuguese is the official tongue of the Brazilian people."],
	["fastest land animal", "The cheetah can sprint at roughly 110 kilometres per hour."],
	["why is the sky blue during the day", "Rayleigh scattering spreads short wavelengths of sunlight across the atmosphere."],
	["tips for falling asleep quickly", "Keep a cool dark bedroom and avoid screens for an hour before bed."],
	["who painted the ceiling of the Sistine Chapel", "Michelangelo spent four years on the frescoes above the papal chapel."],
	["how to remove a red wine stain from carpet", "Blot the spill, then apply a paste of baking soda and cold water before vacuuming."],
	["currency used in Japan", "Payments in Tokyo are made in yen."],
	["how often should I water a cactus", "Succulents in the desert family need soaking only every two to three weeks."],
	["what causes ocean tides", "The Moon's gravitational pull lifts the seas into twice-daily bulges."],
	["how to say thank you in German", "Germans express gratitude with the word danke."],
	["boiling point of water at sea level", "At one atmosphere, water turns to steam at 100 degrees Celsius."],
	["how many players on a soccer team", "Each football side fields eleven people including the goalkeeper."],
	["what do pandas eat", "Giant pandas feed almost entirely on bamboo shoots and leaves."],
	["how to speed up a slow wifi connection", "Move the router to a central spot and switch to a less crowded channel."],
	["largest desert in the world", "Antarctica is the biggest desert by area, ahead of the Sahara."],
];

/** Extra unrelated documents that widen the negative distribution. */
export const SELF_TEST_DISTRACTORS: readonly string[] = [
	"The orchestra tuned to an A of 440 hertz before the overture.",
	"Quarterly revenue grew eight percent on strong subscription renewals.",
	"Knit the sleeve in the round using four double-pointed needles.",
	"The ferry to the island departs every ninety minutes in summer.",
	"Replace the furnace filter at the start of every heating season.",
	"Volcanic soil makes the valley ideal for growing grapes.",
	"The library extended weekend opening hours for exam season.",
	"A torque wrench prevents over-tightening the wheel lug nuts.",
	"Medieval scribes used iron gall ink on vellum manuscripts.",
	"The marathon route passes three bridges and the old harbour.",
];

export interface SelfTestResult {
	readonly calibration: SimilarityCalibration;
	readonly separation: number;
	readonly top1: number;
	readonly negativeMedian: number;
	readonly positiveMedian: number;
	readonly passed: boolean;
	readonly ms: number;
}

export const SELF_TEST_MIN_SEPARATION = 0.8;
export const SELF_TEST_MIN_TOP1 = 0.8;

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
	let s = 0;
	let na = 0;
	let nb = 0;
	const n = Math.min(a.length, b.length);
	for (let i = 0; i < n; i++) {
		s += a[i]! * b[i]!;
		na += a[i]! * a[i]!;
		nb += b[i]! * b[i]!;
	}
	return na > 0 && nb > 0 ? s / Math.sqrt(na * nb) : 0;
}

function quantile(values: readonly number[], q: number): number {
	const sorted = [...values].sort((x, y) => x - y);
	if (sorted.length === 0) return 0;
	const pos = (sorted.length - 1) * q;
	const lo = Math.floor(pos);
	const hi = Math.ceil(pos);
	return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
}

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

/** Runs the self-test through any `embed(texts, inputType)` function (router, backend or provider). */
export async function runEmbeddingSelfTest(
	embed: (texts: readonly string[], inputType: EmbeddingInputType) => Promise<ReadonlyArray<ArrayLike<number>>>,
): Promise<SelfTestResult> {
	const started = performance.now();
	const queries = SELF_TEST_PAIRS.map((p) => p[0]);
	const docs = [...SELF_TEST_PAIRS.map((p) => p[1]), ...SELF_TEST_DISTRACTORS];
	const qv = await embed(queries, "query");
	const dv = await embed(docs, "document");
	const pos: number[] = [];
	const neg: number[] = [];
	let top1 = 0;
	for (let i = 0; i < queries.length; i++) {
		let best = -Infinity;
		let bestIdx = -1;
		for (let j = 0; j < docs.length; j++) {
			const c = cosine(qv[i]!, dv[j]!);
			(i === j ? pos : neg).push(c);
			if (c > best) {
				best = c;
				bestIdx = j;
			}
		}
		if (bestIdx === i) top1 += 1;
	}
	const floor = quantile(neg, 0.95);
	const positiveMedian = quantile(pos, 0.5);
	const separation = pos.filter((p) => p > floor).length / pos.length;
	const top1Rate = top1 / queries.length;
	// Keep a usable span even for degenerate models (ceiling must exceed floor).
	const ceiling = Math.max(positiveMedian, floor + 0.05);
	return {
		calibration: { floor: round3(floor), ceiling: round3(Math.min(1, ceiling)) },
		separation: round3(separation),
		top1: round3(top1Rate),
		negativeMedian: round3(quantile(neg, 0.5)),
		positiveMedian: round3(positiveMedian),
		passed: separation >= SELF_TEST_MIN_SEPARATION && top1Rate >= SELF_TEST_MIN_TOP1,
		ms: Math.round(performance.now() - started),
	};
}
