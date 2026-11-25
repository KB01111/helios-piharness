/**
 * Deterministic, ReDoS-safe entity & relation extraction plus hub-dampened
 * Personalized PageRank (PPR) over a tenant-scoped bipartite Entity <-> Memory graph.
 *
 * Zero external dependencies, zero LLM calls, zero network, linear-time bounded input.
 * Inspired by HippoRAG's associative entity-passage PPR and Mem0/Cognee entity linking.
 */

export type ExtractedEntityType = "entity" | "identifier" | "acronym" | "topic";

export interface ExtractedEntity {
	readonly canonicalName: string;
	readonly displayName: string;
	readonly type: ExtractedEntityType;
	readonly weight: number;
}

export interface ExtractedRelation {
	readonly source: string;
	readonly target: string;
	readonly relation: string;
	readonly weight: number;
}

const MAX_EXTRACT_CHARS = 2000;
const MAX_ENTITIES_PER_MEMORY = 16;
const MAX_RELATIONS_PER_MEMORY = 16;

const GENERIC_NON_ENTITIES: ReadonlySet<string> = new Set([
	"a", "am", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does", "done",
	"for", "from", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "i", "if", "in",
	"into", "is", "it", "its", "may", "me", "might", "my", "no", "not", "now", "of", "on", "one",
	"only", "or", "our", "ours", "out", "over", "pm", "she", "should", "so", "some", "that", "the",
	"their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "too",
	"two", "up", "us", "was", "we", "were", "what", "when", "where", "which", "while", "who",
	"whom", "whose", "why", "will", "with", "would", "you", "your", "yours",
	// Timezones / clock abbreviations
	"cet", "cest", "utc", "gmt", "est", "edt", "pst", "pdt", "mst", "cst",
	// Common sentence-initial verbs/adverbs/adjectives that appear capitalized at start of memories
	"after", "all", "also", "always", "annual", "attended", "based", "before", "between", "built",
	"completed", "currently", "daily", "discovered", "during", "enrolled", "every", "feed",
	"first", "follow", "formerly", "general", "heads", "heavy", "hiked", "incident", "keep",
	"learned", "listens", "local", "major", "migrated", "moved", "never", "new", "next", "noted",
	"originally", "personal", "planted", "please", "prefers", "previous", "previously", "primary",
	"prior", "production", "read", "recorded", "recovered", "refactored", "relocated", "remember",
	"replaced", "rotate", "run", "runs", "salt", "scheduled", "senior", "set", "shipped", "since",
	"spoke", "spotted", "stores", "studied", "switched", "team", "temporary", "transaction",
	"under", "understood", "upgraded", "use", "used", "user", "uses", "using", "visited",
	"volunteered", "walked", "watched", "weekly", "work", "wrote", "yes",
	// Calendar words (handled by temporal.ts rather than entity graph)
	"january", "february", "march", "april", "june", "july", "august", "september", "october",
	"november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
	"sunday", "spring", "summer", "autumn", "fall", "winter", "today", "yesterday", "tomorrow",
]);

const GENERIC_KEY_SEGMENTS: ReadonlySet<string> = new Set([
	"user", "org", "team", "person", "service", "infra", "project", "data", "ops", "security",
	"repo", "memory", "history", "general", "fact", "preference", "procedure", "constraint",
	"episodic", "contact", "travel", "career", "baking", "cooking", "pet", "design", "search",
	"ml", "finops", "release", "devops", "homelab", "backup", "ci", "git", "db", "agent",
	"primary", "default", "config", "setup", "rule", "style", "mode", "version", "time",
	"location", "name", "lead", "owner", "scope", "budget", "guide", "channel", "app", "tool",
	"notes", "note", "item", "synthetic", "archive", "needle", "recent", "episode",
	"model", "policy", "favorite", "spot", "formatter", "shell", "editor", "proxy", "edge",
	"auth", "ingress", "paging", "styling", "feed", "vet", "clinic", "stack", "cluster",
	"warehouse", "ui", "api", "sdk", "token", "canary", "pattern", "recovery", "workflow",
	"timezone", "technique", "deploy", "testing", "runner", "retry", "repro", "pooler",
	"region", "standup", "engine", "algo", "transport", "serialization", "communication",
	"status", "program", "system", "network", "database", "audit", "launch", "outage",
	"concert", "books", "study", "demo", "refactor", "volunteer", "commute", "sunrise",
]);

const HONORIFIC_PREFIX = /^(?:dr|mr|mrs|ms|prof|sir)\.?\s+/i;

const TITLE_WORD_RE = /^[A-Z][\p{L}\p{N}.-]*$/u;
const ACRONYM_RE = /^[A-Z][A-Z0-9-]{1,11}$/;
const MIXED_CASE_TECH_RE = /^(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+\d+[A-Za-z0-9.-]*)$/;
const CODE_IDENT_RE = /^(?:[a-zA-Z][a-zA-Z0-9]*(?:[_-][a-zA-Z0-9]+)+|@[a-z0-9-]+\/[a-z0-9._-]+|v\d+\.\d+(?:\.\d+)?)$/;

/** Normalizes an entity mention to a stable canonical lookup key. */
export function canonicalizeEntityName(raw: string): string {
	const cleaned = raw
		.normalize("NFKC")
		.trim()
		.replace(/['’]s$/i, "")
		.replace(/^[^A-Za-z0-9@#]+|[^A-Za-z0-9+#]+$/g, "")
		.replace(HONORIFIC_PREFIX, "")
		.replace(/[_]+/g, " ")
		.replace(/\s+/g, " ")
		.toLowerCase();
	if (cleaned.length < 2 || cleaned.length > 80) return "";
	if (/^\d+$/.test(cleaned)) return "";
	if (GENERIC_NON_ENTITIES.has(cleaned)) return "";
	return cleaned;
}

function addEntity(
	map: Map<string, ExtractedEntity>,
	displayName: string,
	type: ExtractedEntityType,
	weight: number,
): void {
	const canonical = canonicalizeEntityName(displayName);
	if (!canonical) return;
	const existing = map.get(canonical);
	if (!existing || weight > existing.weight) {
		map.set(canonical, {
			canonicalName: canonical,
			displayName: displayName.trim().replace(/['’]s$/i, "").slice(0, 80),
			type,
			weight,
		});
	}
}

/**
 * Extracts entities from a memory's key + content (or a user query) in strict linear time.
 * Captures multi-word proper nouns, single proper nouns, acronyms, camelCase/snake_case/kebab-case
 * technical identifiers, and meaningful key segments.
 */
export function extractEntities(key: string, content: string): ExtractedEntity[] {
	const boundedContent = content.length > MAX_EXTRACT_CHARS ? content.slice(0, MAX_EXTRACT_CHARS) : content;
	const boundedKey = key.length > 200 ? key.slice(0, 200) : key;
	const entities = new Map<string, ExtractedEntity>();

	// 1. Key segments (e.g. "team.marcus_chen.telemetry_db" -> "marcus chen")
	// Only extract multi-word named key segments (e.g. marcus_chen, horizon_cluster) to avoid generic single-word noise.
	if (boundedKey) {
		const segments = boundedKey.split(/[.:/]+/);
		for (const seg of segments) {
			if (!seg || seg.length < 3) continue;
			const spaced = seg.replace(/[_-]+/g, " ").trim();
			const canon = canonicalizeEntityName(spaced);
			if (!canon || GENERIC_KEY_SEGMENTS.has(canon) || GENERIC_NON_ENTITIES.has(canon)) continue;
			const parts = canon.split(" ");
			if (parts.length >= 2) {
				// Skip generic pairs like "default model", "favorite spot", "prod region", "policy rule"
				if (parts.every((p) => GENERIC_KEY_SEGMENTS.has(p) || GENERIC_NON_ENTITIES.has(p))) continue;
				addEntity(entities, spaced, "entity", 1.0);
			}
		}
	}

	// 2. Scan sentences/clauses linearly to detect capitalized multi-word spans & technical identifiers
	const sentences = boundedContent.split(/(?<=[.!?\n])\s+/);
	for (const sentence of sentences) {
		const rawTokens = sentence.split(/\s+/).filter(Boolean);
		let span: string[] = [];

		const flushSpan = (): void => {
			if (span.length === 0) return;
			// Trim leading generic words if span has >1 word (e.g. "Our shelter" or "In March" won't leak)
			while (span.length > 0) {
				const head = span[0]!.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9.]+$/g, "");
				const low = head.toLowerCase();
				if (GENERIC_NON_ENTITIES.has(low) && !/^(?:dr|mr|mrs|ms|prof)\.?$/i.test(head)) {
					span.shift();
				} else {
					break;
				}
			}
			while (span.length > 0) {
				const tail = span[span.length - 1]!.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "").toLowerCase();
				if (GENERIC_NON_ENTITIES.has(tail)) {
					span.pop();
				} else {
					break;
				}
			}
			if (span.length === 0) return;

			const joined = span.join(" ");
			if (span.length >= 2) {
				addEntity(entities, joined, "entity", 1.15);
				// Also emit honorific-stripped form and strong sub-entities if length >= 2
				const stripped = joined.replace(HONORIFIC_PREFIX, "");
				if (stripped !== joined) {
					addEntity(entities, stripped, "entity", 1.15);
				}
			} else {
				const single = span[0]!;
				const isAcronym = ACRONYM_RE.test(single);
				addEntity(entities, single, isAcronym ? "acronym" : "entity", isAcronym ? 1.0 : 0.92);
			}
			span = [];
		};

		for (let i = 0; i < rawTokens.length; i++) {
			const rawTok = rawTokens[i]!;
			const hasHardBreakAfter = /[,;:!?)\]]$/.test(rawTok);
			let cleaned = rawTok.replace(/^[\s"'([{]+|[\s"',;:!?)\]}]+$/g, "");
			if (cleaned.endsWith(".") && !/^(?:Dr|Mr|Mrs|Ms|Prof)\.$/i.test(cleaned)) {
				cleaned = cleaned.replace(/\.+$/, "");
			}
			if (!cleaned) {
				flushSpan();
				continue;
			}
			const noPossessive = cleaned.replace(/['’]s$/i, "");

			// Technical identifiers (snake_case, kebab-case, @scope/pkg, v1.6.0, SEC-4819, us-east-1, maxRetryCount)
			if (CODE_IDENT_RE.test(noPossessive) || MIXED_CASE_TECH_RE.test(noPossessive)) {
				addEntity(entities, noPossessive, "identifier", 1.05);
				// Also allow mixed-case brand/tech names (e.g. MobileNetV3, ClickHouse, HashiCorp, PostgreSQL) to stay in spans
				if (/^[A-Z]/.test(noPossessive) && !noPossessive.includes("_")) {
					span.push(noPossessive);
					if (hasHardBreakAfter) flushSpan();
					continue;
				}
				flushSpan();
				continue;
			}

			// Honorifics like "Dr." or "Mrs."
			if (/^(?:Dr|Mr|Mrs|Ms|Prof)\.$/i.test(cleaned)) {
				span.push(cleaned);
				continue;
			}

			// Version/roman tokens continuing an active multi-word entity (e.g. "Type II", "CSS v4", "Go 1.23", "ISO 27001")
			if (
				span.length > 0 &&
				(/^(?:v\d+(?:\.\d+)*|\d+(?:\.\d+)+|27001|II|III|IV)$/i.test(noPossessive) ||
					(noPossessive === "du" || noPossessive === "de" || noPossessive === "of"))
			) {
				if (noPossessive === "du" || noPossessive === "de" || noPossessive === "of") {
					// Only keep connector if next token is capitalized
					const nextClean = rawTokens[i + 1]?.replace(/^[\s"'([{]+|[\s"',;:!?)\]}]+$/g, "") ?? "";
					if (TITLE_WORD_RE.test(nextClean)) {
						span.push(noPossessive);
						continue;
					}
				} else {
					span.push(noPossessive);
					if (hasHardBreakAfter) flushSpan();
					continue;
				}
			}

			const isTitle = TITLE_WORD_RE.test(noPossessive) && noPossessive.length >= 2;
			const isAcronym = ACRONYM_RE.test(noPossessive);

			if (isTitle || isAcronym) {
				const low = noPossessive.toLowerCase();
				// Sentence-initial single common word should not start an entity unless followed by another capitalized token
				if (i === 0 && span.length === 0 && GENERIC_NON_ENTITIES.has(low)) {
					continue;
				}
				if (GENERIC_NON_ENTITIES.has(low) && span.length === 0) {
					continue;
				}
				span.push(noPossessive);
				if (hasHardBreakAfter) flushSpan();
			} else {
				flushSpan();
			}
		}
		flushSpan();
	}

	return Array.from(entities.values())
		.sort((a, b) => b.weight - a.weight || (a.canonicalName < b.canonicalName ? -1 : 1))
		.slice(0, MAX_ENTITIES_PER_MEMORY);
}

/**
 * Extracts lightweight relations between entities co-occurring in the same memory.
 * Uses short linking verb/preposition spans between mentions when present, otherwise "related_to".
 */
export function extractRelations(
	entities: readonly ExtractedEntity[],
	content: string,
): ExtractedRelation[] {
	if (entities.length < 2) return [];
	const bounded = content.length > MAX_EXTRACT_CHARS ? content.slice(0, MAX_EXTRACT_CHARS) : content;
	const lower = bounded.toLowerCase();

	// Order entities by earliest occurrence in content (entities only from key go at the end)
	const located = entities
		.slice(0, 8)
		.map((e) => ({
			entity: e,
			pos: lower.indexOf(e.canonicalName),
		}))
		.sort((a, b) => (a.pos < 0 ? 1e9 : a.pos) - (b.pos < 0 ? 1e9 : b.pos));

	const relations: ExtractedRelation[] = [];
	const seen = new Set<string>();

	for (let i = 0; i < located.length && relations.length < MAX_RELATIONS_PER_MEMORY; i++) {
		for (let j = i + 1; j < located.length && relations.length < MAX_RELATIONS_PER_MEMORY; j++) {
			const a = located[i]!;
			const b = located[j]!;
			if (a.entity.canonicalName === b.entity.canonicalName) continue;
			const pairKey = `${a.entity.canonicalName}\0${b.entity.canonicalName}`;
			if (seen.has(pairKey)) continue;
			seen.add(pairKey);

			let label = "related_to";
			if (a.pos >= 0 && b.pos > a.pos + a.entity.canonicalName.length) {
				const between = bounded
					.slice(a.pos + a.entity.canonicalName.length, Math.min(b.pos, a.pos + a.entity.canonicalName.length + 56))
					.replace(/^[^A-Za-z]+|[^A-Za-z]+$/g, "")
					.replace(/\s+/g, " ")
					.trim()
					.toLowerCase();
				if (between.length >= 2 && between.length <= 40 && !/[.!?]/.test(between)) {
					label = between.split(" ").slice(0, 4).join("_").replace(/[^a-z0-9_]/g, "") || "related_to";
				}
			}

			relations.push({
				source: a.entity.canonicalName,
				target: b.entity.canonicalName,
				relation: label,
				weight: Math.min(1, (a.entity.weight + b.entity.weight) / 2.2),
			});
		}
	}

	return relations;
}

/**
 * Extracts query entities both structurally (capitalized/identifiers) AND by case-insensitive
 * 1..3-gram matching against the tenant's known entity vocabulary.
 */
export function extractQueryEntities(
	query: string,
	knownCanonicalEntities?: ReadonlySet<string>,
): ExtractedEntity[] {
	const found = new Map<string, ExtractedEntity>();
	for (const e of extractEntities("", query)) {
		found.set(e.canonicalName, e);
	}

	if (knownCanonicalEntities && knownCanonicalEntities.size > 0) {
		const bounded = query.length > MAX_EXTRACT_CHARS ? query.slice(0, MAX_EXTRACT_CHARS) : query;
		const tokens = bounded.normalize("NFKC").toLowerCase().match(/[\p{L}\p{N}.-]+/gu) ?? [];
		const maxN = Math.min(tokens.length, 32);
		for (let i = 0; i < maxN; i++) {
			for (let span = 3; span >= 1; span--) {
				if (i + span > maxN) continue;
				const candidate = tokens
					.slice(i, i + span)
					.join(" ")
					.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, "");
				if (!candidate || candidate.length < 2 || GENERIC_NON_ENTITIES.has(candidate)) continue;
				if (knownCanonicalEntities.has(candidate) && !found.has(candidate)) {
					found.set(candidate, {
						canonicalName: candidate,
						displayName: candidate,
						type: span >= 2 ? "entity" : "topic",
						weight: span >= 2 ? 1.1 : 0.85,
					});
				}
			}
		}
	}

	return Array.from(found.values());
}

// ---------------------------------------------------------------------------------------------------------------------
// In-Memory Tenant-Scoped Bipartite Graph + Hub-Dampened Personalized PageRank (PPR)
// ---------------------------------------------------------------------------------------------------------------------

export interface EntityGraphEdge {
	readonly memoryId: string;
	readonly entityId: string;
	readonly canonicalName: string;
	readonly weight: number;
}

export interface BipartiteMemoryGraph {
	readonly memoryToEntities: ReadonlyMap<string, ReadonlyArray<{ readonly entityId: string; readonly canonicalName: string; readonly weight: number }>>;
	readonly entityToMemories: ReadonlyMap<string, ReadonlyArray<{ readonly memoryId: string; readonly weight: number }>>;
	readonly canonicalToEntityId: ReadonlyMap<string, string>;
	readonly knownCanonicals: ReadonlySet<string>;
}

/** Builds an immutable bipartite Entity <-> Memory adjacency structure for fast PPR traversal. */
export function buildBipartiteGraph(edges: readonly EntityGraphEdge[]): BipartiteMemoryGraph {
	const memoryToEntities = new Map<string, Array<{ entityId: string; canonicalName: string; weight: number }>>();
	const entityToMemories = new Map<string, Array<{ memoryId: string; weight: number }>>();
	const canonicalToEntityId = new Map<string, string>();
	const knownCanonicals = new Set<string>();

	for (const edge of edges) {
		canonicalToEntityId.set(edge.canonicalName, edge.entityId);
		knownCanonicals.add(edge.canonicalName);

		let mList = memoryToEntities.get(edge.memoryId);
		if (!mList) {
			mList = [];
			memoryToEntities.set(edge.memoryId, mList);
		}
		mList.push({ entityId: edge.entityId, canonicalName: edge.canonicalName, weight: edge.weight });

		let eList = entityToMemories.get(edge.entityId);
		if (!eList) {
			eList = [];
			entityToMemories.set(edge.entityId, eList);
		}
		eList.push({ memoryId: edge.memoryId, weight: edge.weight });
	}

	return {
		memoryToEntities,
		entityToMemories,
		canonicalToEntityId,
		knownCanonicals,
	};
}

export interface PprResult {
	/** Normalized PPR activation score in [0, 1] per memoryId. */
	readonly scores: ReadonlyMap<string, number>;
	/** Number of distinct query entities matched directly in the graph. */
	readonly matchedQueryEntityCount: number;
}

/**
 * Runs 2-hop hub-dampened Personalized PageRank over the bipartite Entity <-> Memory graph.
 *
 * Seeds come from:
 *  1. Direct query entities that exist in the graph (weighted by inverse degree specificity).
 *  2. High-confidence seed memories from lexical/vector retrieval (so even when the bridge entity
 *     is NOT named in the query — e.g. "Billing Service" -> "Elena Rostova" -> "Go 1.23 & gRPC" —
 *     the bridge entity activates and pulls the 2-hop target memory cleanly into top-K!).
 */
export function runEntityPpr(
	graph: BipartiteMemoryGraph,
	input: {
		readonly queryEntities: readonly ExtractedEntity[];
		/** Seed memory IDs with their base relevance scores in [0, 1]. Only strong seeds should be passed. */
		readonly seedMemories: ReadonlyArray<{ readonly memoryId: string; readonly seedScore: number }>;
		/** Optional filter of allowed memory IDs (e.g. active-only vs including superseded). */
		readonly allowedMemoryIds?: ReadonlySet<string>;
		/** Damping factor for multi-hop propagation. Default 0.72. */
		readonly damping?: number;
	},
): PprResult {
	const damping = input.damping ?? 0.72;
	const entityMass = new Map<string, number>();
	const directSeedIds = new Set<string>();
	let matchedQueryEntityCount = 0;

	// Inverse-degree hub dampening (combines HippoRAG 1/sqrt(deg) and Mem0 1/(1 + 0.02*(deg-1)^2))
	const hubDampen = (degree: number): number => {
		if (degree <= 1) return 1.0;
		return 1 / (Math.sqrt(degree) * (1 + 0.02 * (degree - 1)));
	};

	// 1) Inject mass from explicit query entities present in the graph
	for (const qe of input.queryEntities) {
		const entityId = graph.canonicalToEntityId.get(qe.canonicalName);
		if (!entityId) continue;
		const neighbors = graph.entityToMemories.get(entityId);
		if (!neighbors || neighbors.length === 0) continue;
		matchedQueryEntityCount += 1;
		const spec = hubDampen(neighbors.length);
		entityMass.set(entityId, (entityMass.get(entityId) ?? 0) + qe.weight * spec * 1.15);
	}

	// 2) Inject mass from top seed memories into their linked entities (enables unseen bridge discovery!)
	for (const seed of input.seedMemories) {
		if (seed.seedScore <= 0) continue;
		directSeedIds.add(seed.memoryId);
		const linked = graph.memoryToEntities.get(seed.memoryId);
		if (!linked || linked.length === 0) continue;
		const perEntityShare = seed.seedScore / Math.sqrt(linked.length);
		for (const edge of linked) {
			const deg = graph.entityToMemories.get(edge.entityId)?.length ?? 1;
			// Prefer bridge entities (degree >= 2) while dampening ultra-generic hubs
			const bridgeFactor = deg >= 2 ? 1.25 * hubDampen(deg) : 0.35;
			entityMass.set(edge.entityId, (entityMass.get(edge.entityId) ?? 0) + perEntityShare * edge.weight * bridgeFactor);
		}
	}

	if (entityMass.size === 0) {
		return { scores: new Map(), matchedQueryEntityCount };
	}

	// 3) Propagate Entities -> Memories (Hop 1 from query entities / Hop 2 from seed memories!)
	const rawScores = new Map<string, number>();
	for (const [entityId, mass] of entityMass.entries()) {
		if (mass <= 1e-6) continue;
		const neighbors = graph.entityToMemories.get(entityId);
		if (!neighbors || neighbors.length === 0) continue;
		const degNorm = 1 / Math.sqrt(neighbors.length);
		for (const edge of neighbors) {
			if (input.allowedMemoryIds && !input.allowedMemoryIds.has(edge.memoryId)) continue;
			// Strongly favor newly-discovered 2-hop neighbor memories over self-echo on the original seed memory
			const hopBonus = directSeedIds.has(edge.memoryId) ? 0.35 : 1.25;
			const delta = damping * mass * edge.weight * degNorm * hopBonus;
			rawScores.set(edge.memoryId, (rawScores.get(edge.memoryId) ?? 0) + delta);
		}
	}

	if (rawScores.size === 0) {
		return { scores: new Map(), matchedQueryEntityCount };
	}

	// 4) Soft-saturate raw PPR scores into [0, 1] without single-hit inflation (1 - exp(-k * x))
	const scores = new Map<string, number>();
	for (const [memId, val] of rawScores.entries()) {
		const normalized = 1 - Math.exp(-2.1 * val);
		if (normalized > 0.02) {
			scores.set(memId, Math.min(1, normalized));
		}
	}

	return { scores, matchedQueryEntityCount };
}
