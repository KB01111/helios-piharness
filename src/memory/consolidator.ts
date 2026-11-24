import type {
	FastModelClient,
	LongTermMemory,
	MemoryKind,
	MemoryMutationOp,
	MemoryService,
	MidTermEpisode,
} from "../contracts/app-capabilities.js";

export interface ConsolidationGateConfig {
	/** Minimum confidence required to promote/mutate a long-term memory. Default: 0.7 */
	readonly minConfidence: number;
	/** Minimum supporting episodes required for non-explicit ADD promotions. Default: 2 */
	readonly minEvidenceCount: number;
	/** Alternative to evidence count: distinct conversations supporting the fact. Default: 2 */
	readonly minDistinctConversations: number;
}

export const DEFAULT_CONSOLIDATION_GATES: ConsolidationGateConfig = {
	minConfidence: 0.7,
	minEvidenceCount: 2,
	minDistinctConversations: 2,
};

export interface ProposedConsolidationOp {
	readonly op: "ADD" | "UPDATE" | "SUPERSEDE" | "NOOP";
	readonly targetId?: string;
	readonly key?: string;
	readonly kind?: MemoryKind;
	readonly content?: string;
	readonly confidence?: number;
	readonly explicitUserPreference?: boolean;
	readonly evidenceCount?: number;
	readonly distinctConversations?: number;
	readonly reason?: string;
}

export interface ConsolidationPlan {
	readonly operations: readonly ProposedConsolidationOp[];
}

export interface RejectedProposal {
	readonly proposal: ProposedConsolidationOp;
	readonly reason: string;
}

export interface ConsolidationReport {
	readonly processedEpisodeIds: readonly string[];
	readonly proposedCount: number;
	readonly addedIds: readonly string[];
	readonly updatedIds: readonly string[];
	readonly supersededOldIds: readonly string[];
	readonly supersededNewIds: readonly string[];
	readonly noopCount: number;
	readonly rejectedByGate: readonly RejectedProposal[];
}

const CONSOLIDATOR_SYSTEM_PROMPT = `You are a memory consolidation engine (ACE delta mode).
Given existing active long-term memories and recent mid-term episodes, emit ONLY a strict JSON object:
{
  "operations": [
    { "op": "ADD", "key": "short.dotted.key", "kind": "preference|fact|procedure|constraint", "content": "...", "confidence": 0.85, "explicitUserPreference": true, "evidenceCount": 2, "distinctConversations": 1 },
    { "op": "UPDATE", "targetId": "mem_...", "content": "...", "confidence": 0.9 },
    { "op": "SUPERSEDE", "targetId": "mem_...", "key": "short.dotted.key", "kind": "preference", "content": "...", "confidence": 0.95, "reason": "contradicts older preference" },
    { "op": "NOOP", "reason": "no durable signal" }
  ]
}
Rules:
- Never delete conflicting preferences; emit SUPERSEDE targeting the older active memory ID.
- Only propose durable, reusable user facts, preferences, constraints, or workflows.
- Return valid JSON only with no markdown fences.`;

function normalizeKey(raw: string): string {
	return raw
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9._-]+/g, "_")
		.replace(/^_+|_+$/g, "");
}

function normalizeContent(raw: string): string {
	return raw.trim().replace(/\s+/g, " ").toLowerCase();
}

function isValidKind(kind: unknown): kind is MemoryKind {
	return (
		kind === "fact" ||
		kind === "preference" ||
		kind === "procedure" ||
		kind === "constraint" ||
		kind === "episodic"
	);
}

/**
 * Deterministic fallback extractor when no fast LLM is available.
 * Extracts explicit user preferences of the form "prefer X", "always use X",
 * "switch to X", etc.
 */
function deterministicFallbackPlan(
	episodes: readonly MidTermEpisode[],
	activeMemories: readonly LongTermMemory[],
): ConsolidationPlan {
	const ops: ProposedConsolidationOp[] = [];
	const byKey = new Map<string, LongTermMemory>();
	for (const mem of activeMemories) {
		byKey.set(normalizeKey(mem.key), mem);
	}

	const keyCounts = new Map<string, { content: string; conversations: Set<string>; count: number; explicit: boolean }>();

	for (const ep of episodes) {
		const match = ep.userMessage.match(
			/\b(?:i\s+prefer|always\s+use|from\s+now\s+on\s+use|switch\s+to|remember\s+that\s+i\s+prefer)\s+([^.!?\n]{3,120})/i,
		);
		if (!match || !match[1]) continue;
		const content = match[1].trim();
		const key = "user.preference.general";
		const existingEntry = keyCounts.get(key);
		if (existingEntry) {
			existingEntry.content = content;
			existingEntry.count += 1;
			existingEntry.conversations.add(ep.conversationId);
			existingEntry.explicit = true;
		} else {
			keyCounts.set(key, {
				content,
				conversations: new Set([ep.conversationId]),
				count: 1,
				explicit: true,
			});
		}
	}

	for (const [key, item] of keyCounts.entries()) {
		const existing = byKey.get(key);
		if (!existing) {
			ops.push({
				op: "ADD",
				key,
				kind: "preference",
				content: item.content,
				confidence: 0.9,
				explicitUserPreference: item.explicit,
				evidenceCount: item.count,
				distinctConversations: item.conversations.size,
			});
		} else if (normalizeContent(existing.content) === normalizeContent(item.content)) {
			ops.push({ op: "NOOP", reason: "Preference unchanged" });
		} else {
			ops.push({
				op: "SUPERSEDE",
				targetId: existing.id,
				key,
				kind: "preference",
				content: item.content,
				confidence: 0.95,
				explicitUserPreference: true,
				reason: "User updated preference",
			});
		}
	}

	if (ops.length === 0) {
		ops.push({ op: "NOOP", reason: "No durable memory updates detected" });
	}
	return { operations: ops };
}

/**
 * Fast-model structured diff engine (ACE ADD / UPDATE / SUPERSEDE / NOOP)
 * with deterministic promotion gates and contradiction tombstoning.
 */
export class MemoryConsolidator {
	private readonly memory: MemoryService;
	private readonly fastModel?: FastModelClient;
	private readonly gates: ConsolidationGateConfig;

	constructor(options: {
		readonly memory: MemoryService;
		readonly fastModel?: FastModelClient;
		readonly gates?: Partial<ConsolidationGateConfig>;
	}) {
		this.memory = options.memory;
		this.fastModel = options.fastModel;
		this.gates = { ...DEFAULT_CONSOLIDATION_GATES, ...options.gates };
	}

	async consolidate(
		tenantId: string,
		userId: string,
		options?: { readonly limit?: number; readonly signal?: AbortSignal },
	): Promise<ConsolidationReport> {
		const episodes = await this.memory.listUnconsolidatedEpisodes(tenantId, userId, options?.limit ?? 50);
		if (episodes.length === 0) {
			return {
				processedEpisodeIds: [],
				proposedCount: 0,
				addedIds: [],
				updatedIds: [],
				supersededOldIds: [],
				supersededNewIds: [],
				noopCount: 0,
				rejectedByGate: [],
			};
		}

		const activeMemories = await this.memory.listMemories(tenantId, userId, {
			includeSuperseded: false,
			includeDeleted: false,
			limit: 200,
		});

		const plan = await this.proposePlan(episodes, activeMemories, options?.signal);
		const report = await this.applyPlan(tenantId, userId, episodes, activeMemories, plan);

		await this.memory.markEpisodesConsolidated(episodes.map((ep) => ep.id));
		return {
			...report,
			processedEpisodeIds: episodes.map((ep) => ep.id),
		};
	}

	private async proposePlan(
		episodes: readonly MidTermEpisode[],
		activeMemories: readonly LongTermMemory[],
		signal?: AbortSignal,
	): Promise<ConsolidationPlan> {
		if (!this.fastModel) {
			return deterministicFallbackPlan(episodes, activeMemories);
		}

		const userPrompt = JSON.stringify(
			{
				activeMemories: activeMemories.map((m) => ({
					id: m.id,
					key: m.key,
					kind: m.kind,
					content: m.content,
					confidence: m.confidence,
					pinned: m.pinned,
					evidenceCount: m.evidenceCount,
				})),
				episodes: episodes.map((e) => ({
					id: e.id,
					conversationId: e.conversationId,
					userMessage: e.userMessage,
					assistantSummary: e.assistantSummary,
					frictionSignals: e.frictionSignals,
					frictionScore: e.frictionScore,
				})),
			},
			null,
			2,
		);

		const response = await this.fastModel.completeJson<ConsolidationPlan>(CONSOLIDATOR_SYSTEM_PROMPT, userPrompt, {
			signal,
		});
		if (response.parsed && Array.isArray(response.parsed.operations)) {
			return response.parsed;
		}
		return deterministicFallbackPlan(episodes, activeMemories);
	}

	private async applyPlan(
		tenantId: string,
		userId: string,
		episodes: readonly MidTermEpisode[],
		initialActiveMemories: readonly LongTermMemory[],
		plan: ConsolidationPlan,
	): Promise<Omit<ConsolidationReport, "processedEpisodeIds">> {
		const activeById = new Map<string, LongTermMemory>();
		const activeByKey = new Map<string, LongTermMemory>();
		for (const mem of initialActiveMemories) {
			activeById.set(mem.id, mem);
			activeByKey.set(normalizeKey(mem.key), mem);
		}

		const distinctEpisodeConversations = new Set(episodes.map((e) => e.conversationId)).size;
		const latestConversationId = episodes[episodes.length - 1]?.conversationId;

		const addedIds: string[] = [];
		const updatedIds: string[] = [];
		const supersededOldIds: string[] = [];
		const supersededNewIds: string[] = [];
		const rejectedByGate: RejectedProposal[] = [];
		let noopCount = 0;

		for (const proposal of plan.operations) {
			if (proposal.op === "NOOP") {
				noopCount += 1;
				continue;
			}

			const confidence = typeof proposal.confidence === "number" ? proposal.confidence : 0;
			if (confidence < this.gates.minConfidence) {
				rejectedByGate.push({
					proposal,
					reason: `confidence ${confidence.toFixed(2)} below gate ${this.gates.minConfidence.toFixed(2)}`,
				});
				continue;
			}

			if (proposal.op === "UPDATE") {
				const target = proposal.targetId ? activeById.get(proposal.targetId) : undefined;
				if (!target) {
					rejectedByGate.push({ proposal, reason: `targetId "${proposal.targetId ?? ""}" not found among active memories` });
					continue;
				}
				if (!proposal.content || proposal.content.trim().length === 0) {
					rejectedByGate.push({ proposal, reason: "UPDATE requires non-empty content" });
					continue;
				}
				if (normalizeContent(target.content) === normalizeContent(proposal.content)) {
					noopCount += 1;
					continue;
				}
				const mutation: MemoryMutationOp = {
					op: "UPDATE",
					targetId: target.id,
					content: proposal.content.trim(),
					confidence,
					evidenceCountDelta: 1,
					distinctConversationsDelta: distinctEpisodeConversations > 1 ? 1 : 0,
				};
				const updated = await this.memory.applyMemoryMutation(tenantId, userId, mutation);
				if (updated) {
					activeById.set(updated.id, updated);
					activeByKey.set(normalizeKey(updated.key), updated);
					updatedIds.push(updated.id);
				}
				continue;
			}

			if (proposal.op === "SUPERSEDE") {
				const normalizedKey = proposal.key ? normalizeKey(proposal.key) : "";
				const target =
					(proposal.targetId ? activeById.get(proposal.targetId) : undefined) ??
					(normalizedKey ? activeByKey.get(normalizedKey) : undefined);

				if (!target) {
					rejectedByGate.push({
						proposal,
						reason: `SUPERSEDE target "${proposal.targetId ?? normalizedKey}" not found among active memories`,
					});
					continue;
				}
				if (!proposal.content || proposal.content.trim().length === 0) {
					rejectedByGate.push({ proposal, reason: "SUPERSEDE requires non-empty content" });
					continue;
				}
				if (normalizeContent(target.content) === normalizeContent(proposal.content)) {
					noopCount += 1;
					continue;
				}
				const kind: MemoryKind = isValidKind(proposal.kind) ? proposal.kind : target.kind;
				const key = normalizedKey || normalizeKey(target.key);
				const created = await this.memory.applyMemoryMutation(tenantId, userId, {
					op: "SUPERSEDE",
					targetId: target.id,
					key,
					kind,
					content: proposal.content.trim(),
					confidence,
					reason: proposal.reason,
					sourceConversationId: latestConversationId,
				});
				if (created) {
					activeById.delete(target.id);
					activeById.set(created.id, created);
					activeByKey.set(normalizeKey(created.key), created);
					supersededOldIds.push(target.id);
					supersededNewIds.push(created.id);
				}
				continue;
			}

			// ADD
			const key = proposal.key ? normalizeKey(proposal.key) : "";
			const content = proposal.content?.trim() ?? "";
			const kind: MemoryKind = isValidKind(proposal.kind) ? proposal.kind : "fact";
			if (!key || !content) {
				rejectedByGate.push({ proposal, reason: "ADD requires non-empty key and content" });
				continue;
			}

			const existingSameKey = activeByKey.get(key);
			if (existingSameKey) {
				if (normalizeContent(existingSameKey.content) === normalizeContent(content)) {
					noopCount += 1;
					continue;
				}
				// Contradictory ADD on the same key automatically upgrades to SUPERSEDE tombstone
				const created = await this.memory.applyMemoryMutation(tenantId, userId, {
					op: "SUPERSEDE",
					targetId: existingSameKey.id,
					key,
					kind,
					content,
					confidence,
					reason: "Conflicting ADD on existing active key auto-promoted to SUPERSEDE",
					sourceConversationId: latestConversationId,
				});
				if (created) {
					activeById.delete(existingSameKey.id);
					activeById.set(created.id, created);
					activeByKey.set(key, created);
					supersededOldIds.push(existingSameKey.id);
					supersededNewIds.push(created.id);
				}
				continue;
			}

			// Promotion gate for brand-new ADD memories:
			// Must either be an explicit user preference/constraint OR satisfy evidence/repetition threshold.
			const evidenceCount = proposal.evidenceCount ?? 1;
			const distinctConversations = proposal.distinctConversations ?? distinctEpisodeConversations;
			const passesRepetitionGate =
				proposal.explicitUserPreference === true ||
				evidenceCount >= this.gates.minEvidenceCount ||
				distinctConversations >= this.gates.minDistinctConversations;

			if (!passesRepetitionGate) {
				rejectedByGate.push({
					proposal,
					reason: `promotion gate rejected single-occurrence non-explicit ADD (evidence=${evidenceCount}, distinctConversations=${distinctConversations})`,
				});
				continue;
			}

			const created = await this.memory.applyMemoryMutation(tenantId, userId, {
				op: "ADD",
				key,
				kind,
				content,
				confidence,
				evidenceCount,
				distinctConversations,
				sourceConversationId: latestConversationId,
			});
			if (created) {
				activeById.set(created.id, created);
				activeByKey.set(key, created);
				addedIds.push(created.id);
			}
		}

		return {
			proposedCount: plan.operations.length,
			addedIds,
			updatedIds,
			supersededOldIds,
			supersededNewIds,
			noopCount,
			rejectedByGate,
		};
	}
}
