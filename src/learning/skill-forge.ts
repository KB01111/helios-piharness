export interface SkillForgeOptions {
	/** Minimum tool sequence n-gram length. Default: 3 */
	readonly minLength?: number;
	/** Maximum tool sequence n-gram length. Default: 7 */
	readonly maxLength?: number;
	/** Minimum distinct runs containing the sequence. Default: 2 */
	readonly minSupport?: number;
	/** Collapse consecutive duplicate calls to the same tool before mining. Default: false */
	readonly collapseImmediateRepeats?: boolean;
}

export interface MinedWorkflow {
	readonly id: string;
	readonly sequence: readonly string[];
	readonly length: number;
	/** Number of distinct runs/sessions in which this workflow occurred. */
	readonly support: number;
	/** Total occurrences across all runs. */
	readonly totalOccurrences: number;
	readonly suggestedSkillName: string;
	readonly draftSkillMarkdown: string;
}

function canonicalizeSequence(seq: readonly string[], collapseRepeats: boolean): string[] {
	const cleaned = seq.map((s) => s.trim()).filter((s) => s.length > 0);
	if (!collapseRepeats) return cleaned;
	const out: string[] = [];
	for (const tool of cleaned) {
		if (out[out.length - 1] !== tool) {
			out.push(tool);
		}
	}
	return out;
}

function buildSkillName(sequence: readonly string[]): string {
	const slug = sequence
		.slice(0, 4)
		.map((s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, ""))
		.join("-then-");
	return `workflow-${slug}`.slice(0, 64);
}

function renderSkillMarkdown(name: string, sequence: readonly string[], support: number): string {
	const steps = sequence.map((tool, i) => `${i + 1}. Call \`${tool}\` with scoped parameters.`).join("\n");
	return [
		"---",
		`name: ${name}`,
		`description: Mined recurring workflow (${sequence.join(" -> ")}) observed in ${support} runs.`,
		"version: 1",
		`tools: [${sequence.join(", ")}]`,
		"tags: [mined, workflow]",
		"---",
		"",
		`# ${name}`,
		"",
		"## When to Use",
		`Use this workflow when a task requires the recurring sequence: \`${sequence.join(" -> ")}\`.`,
		"",
		"## Steps",
		steps,
		"",
	].join("\n");
}

/**
 * Deterministic n-gram workflow miner (Loop 2: Skill Forge).
 *
 * Mines recurring contiguous 3-to-7 tool sequences across recorded runs
 * without any LLM calls, prunes strictly dominated sub-grams, and synthesizes
 * progressive-disclosure SKILL.md candidate drafts.
 */
export class SkillForge {
	private readonly minLength: number;
	private readonly maxLength: number;
	private readonly minSupport: number;
	private readonly collapseImmediateRepeats: boolean;

	constructor(options?: SkillForgeOptions) {
		this.minLength = Math.max(2, options?.minLength ?? 3);
		this.maxLength = Math.max(this.minLength, Math.min(options?.maxLength ?? 7, 12));
		this.minSupport = Math.max(2, options?.minSupport ?? 2);
		this.collapseImmediateRepeats = options?.collapseImmediateRepeats ?? false;
	}

	mine(sequences: readonly (readonly string[])[]): MinedWorkflow[] {
		const counts = new Map<
			string,
			{
				sequence: string[];
				runIndices: Set<number>;
				totalOccurrences: number;
			}
		>();

		sequences.forEach((rawSeq, runIndex) => {
			const seq = canonicalizeSequence(rawSeq, this.collapseImmediateRepeats);
			if (seq.length < this.minLength) return;

			const maxLen = Math.min(this.maxLength, seq.length);
			for (let len = this.minLength; len <= maxLen; len++) {
				for (let start = 0; start <= seq.length - len; start++) {
					const slice = seq.slice(start, start + len);
					const key = slice.join("\u001f");
					const existing = counts.get(key);
					if (existing) {
						existing.totalOccurrences += 1;
						existing.runIndices.add(runIndex);
					} else {
						counts.set(key, {
							sequence: slice,
							runIndices: new Set([runIndex]),
							totalOccurrences: 1,
						});
					}
				}
			}
		});

		const frequent = Array.from(counts.entries())
			.filter(([, entry]) => entry.runIndices.size >= this.minSupport)
			.map(([key, entry]) => ({
				key,
				sequence: entry.sequence,
				support: entry.runIndices.size,
				totalOccurrences: entry.totalOccurrences,
			}));

		// Maximal pruning: drop a shorter n-gram if a strictly longer frequent n-gram
		// contains it as a contiguous subsequence with the same (or greater) support.
		const maximal = frequent.filter((candidate) => {
			return !frequent.some(
				(other) =>
					other.sequence.length > candidate.sequence.length &&
					other.support >= candidate.support &&
					other.key.includes(candidate.key),
			);
		});

		maximal.sort((a, b) => {
			if (b.support !== a.support) return b.support - a.support;
			if (b.sequence.length !== a.sequence.length) return b.sequence.length - a.sequence.length;
			return a.key.localeCompare(b.key);
		});

		return maximal.map((item, idx) => {
			const suggestedSkillName = buildSkillName(item.sequence);
			return {
				id: `wf_${idx + 1}_${suggestedSkillName}`,
				sequence: item.sequence,
				length: item.sequence.length,
				support: item.support,
				totalOccurrences: item.totalOccurrences,
				suggestedSkillName,
				draftSkillMarkdown: renderSkillMarkdown(suggestedSkillName, item.sequence, item.support),
			};
		});
	}
}
