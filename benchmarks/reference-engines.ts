/**
 * Reference architectures and shared embedding providers for apples-to-apples memory benchmarking:
 *  - LocalOpenAIEmbeddingProvider: calls local all-MiniLM-L6-v2 server (384-dim real neural embeddings)
 *  - GraphitiRefEngine: Bi-temporal knowledge graph (valid_from/valid_until invalidation + entity graph + BM25+Vector RRF)
 *  - HippoRagRefEngine: Hippocampal entity-memory bipartite graph + Personalized PageRank (PPR) multi-hop retrieval
 */

import type { EmbeddingProvider } from "../src/contracts/app-capabilities.js";
import type { BenchmarkMemory } from "./memory-dataset.js";
import { BENCHMARK_NOW_MS } from "./memory-dataset.js";

const DAY_MS = 86_400_000;

export interface RankedHit {
	readonly id: string;
	readonly score: number;
	readonly text: string;
	readonly isEpisode?: boolean;
}

export interface BenchMemoryEngine {
	readonly name: string;
	readonly supportsHistorical?: boolean;
	readonly supportsEpisodes?: boolean;
	readonly abstentionThreshold: number;
	search(query: string, topK: number): Promise<RankedHit[]>;
	close?(): Promise<void> | void;
}

/**
 * Calls an OpenAI-compatible `/v1/embeddings` endpoint (our local all-MiniLM-L6-v2 server)
 * with an in-memory cache so repeated texts across engines incur zero extra compute.
 */
export class CachedHttpEmbeddingProvider implements EmbeddingProvider {
	readonly model: string;
	private readonly endpoint: string;
	private readonly apiKey: string | undefined;
	private readonly cache = new Map<string, Float32Array>();

	constructor(options: { readonly endpoint: string; readonly apiKey?: string; readonly model: string }) {
		this.apiKey = options.apiKey;
		this.endpoint = options.endpoint.replace(/\/+$/, "");
		this.model = options.model;
	}

	async embed(texts: readonly string[]): Promise<ReadonlyArray<ArrayLike<number>>> {
		const out: Array<Float32Array | undefined> = new Array(texts.length);
		const missing: Array<{ idx: number; text: string }> = [];
		for (let i = 0; i < texts.length; i++) {
			const text = texts[i] ?? "";
			const hit = this.cache.get(text);
			if (hit) out[i] = hit;
			else missing.push({ idx: i, text });
		}
		if (missing.length > 0) {
			// Batch in chunks of 64
			for (let start = 0; start < missing.length; start += 64) {
				const slice = missing.slice(start, start + 64);
				const res = await fetch(`${this.endpoint}/embeddings`, {
					method: "POST",
					headers: { "Content-Type": "application/json", ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}) },
					body: JSON.stringify({ model: this.model, input: slice.map((s) => s.text) }),
				});
				if (!res.ok) throw new Error(`Embedding HTTP ${res.status}: ${await res.text()}`);
				const body = (await res.json()) as { data: Array<{ index: number; embedding: number[] }> };
				for (const item of body.data) {
					const entry = slice[item.index];
					if (!entry) continue;
					const vec = Float32Array.from(item.embedding);
					this.cache.set(entry.text, vec);
					out[entry.idx] = vec;
				}
			}
		}
		return out.map((v) => v ?? new Float32Array(384));
	}
}

// ── Shared lightweight text / entity / vector math for reference architectures ──────────────────

const STOP = new Set([
	"a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "did", "for", "from", "has", "have", "how",
	"i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "that", "the", "their", "this", "to", "us",
	"was", "we", "what", "when", "where", "which", "who", "why", "with", "you", "your",
]);

function tokenize(text: string): string[] {
	return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => t.length > 1 && !STOP.has(t));
}

function dotCosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		const x = a[i] ?? 0;
		const y = b[i] ?? 0;
		dot += x * y;
		na += x * x;
		nb += y * y;
	}
	return na > 0 && nb > 0 ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

const STOP_CAPS = new Set([
	"The", "This", "That", "When", "Where", "What", "Who", "Why", "How", "In", "On", "At", "By", "For",
	"From", "With", "After", "Before", "During", "Since", "Until", "My", "Our", "We", "He", "She", "They",
	"It", "I", "A", "An", "But", "And", "Or", "If", "No", "Yes", "Always", "Never", "Uses", "Used", "User",
	"Switched", "Migrated", "Moved", "Updated", "Replaced", "Adopted", "Daily", "Keep", "Run", "Set", "Now",
	"Production", "Annual", "Senior", "Severity", "Feed", "Watched", "Attended", "Visited", "Read", "Learned",
]);

function extractSimpleEntities(text: string): string[] {
	const out = new Set<string>();
	// Multi-word capitalized phrases & proper nouns
	const properRe = /\b([A-Z][a-zA-Z0-9+#.-]*(?:\s+(?:of|the|du|de|la)?\s*[A-Z][a-zA-Z0-9+#.-]*)*)\b/g;
	for (const m of text.matchAll(properRe)) {
		const words = m[1]!.split(/\s+/);
		while (words.length > 0 && STOP_CAPS.has(words[0]!)) words.shift();
		if (words.length === 0) continue;
		const phrase = words.join(" ").toLowerCase();
		if (phrase.length >= 2) out.add(phrase);
		// Also add individual proper tokens for partial entity match
		for (const w of words) {
			if (w.length >= 3 && !STOP_CAPS.has(w)) out.add(w.toLowerCase());
		}
	}
	// Technical identifiers (us-east-1, pg_bouncer, maxRetryCount, v1.6.0)
	for (const m of text.matchAll(/\b([a-z][a-z0-9]*(?:[-_./][a-z0-9]+)+|[a-z]+[A-Z][a-zA-Z0-9]*)\b/g)) {
		out.add(m[1]!.toLowerCase());
	}
	return Array.from(out);
}

function extractYears(text: string): number[] {
	const years: number[] = [];
	for (const m of text.matchAll(/\b((?:19|20)\d{2})\b/g)) {
		years.push(Number(m[1]));
	}
	return years;
}

function computeBm25Map(docs: ReadonlyArray<{ id: string; tokens: string[] }>, queryTokens: string[]): Map<string, number> {
	const N = docs.length;
	if (N === 0 || queryTokens.length === 0) return new Map();
	const avgLen = docs.reduce((s, d) => s + d.tokens.length, 0) / N || 1;
	const k1 = 1.5;
	const b = 0.75;
	const qUnique = Array.from(new Set(queryTokens));
	const df = new Map<string, number>();
	for (const qt of qUnique) {
		let c = 0;
		for (const d of docs) if (d.tokens.includes(qt)) c++;
		df.set(qt, c);
	}
	const scores = new Map<string, number>();
	for (const d of docs) {
		let score = 0;
		for (const qt of qUnique) {
			let tf = 0;
			for (const t of d.tokens) if (t === qt) tf++;
			if (tf === 0) continue;
			const idf = Math.log(1 + (N - (df.get(qt) ?? 0) + 0.5) / ((df.get(qt) ?? 0) + 0.5));
			score += (idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * d.tokens.length) / avgLen));
		}
		if (score > 0) scores.set(d.id, score);
	}
	return scores;
}

// ── 1. Graphiti-Style Bi-Temporal Knowledge Graph Reference ────────────────────────────────────

interface GraphitiNode {
	readonly id: string;
	readonly text: string;
	readonly tokens: string[];
	readonly entities: string[];
	readonly years: number[];
	readonly vector: Float32Array;
	readonly validFrom: number;
	validUntil: number | null;
}

export class GraphitiRefEngine implements BenchMemoryEngine {
	readonly name = "graphiti-ref (bi-temporal+RRF)";
	readonly supportsHistorical = true;
	readonly abstentionThreshold = 0.28;

	private readonly nodes: GraphitiNode[] = [];
	private readonly entityToNodes = new Map<string, Set<string>>();
	private readonly embedder: EmbeddingProvider;

	private constructor(embedder: EmbeddingProvider) {
		this.embedder = embedder;
	}

	static async build(memories: readonly BenchmarkMemory[], embedder: EmbeddingProvider): Promise<GraphitiRefEngine> {
		const engine = new GraphitiRefEngine(embedder);
		const texts = memories.map((m) => `${m.key}: ${m.content}`);
		const vecs = await embedder.embed(texts, { inputType: "document" });
		const byId = new Map<string, GraphitiNode>();

		for (let i = 0; i < memories.length; i++) {
			const m = memories[i]!;
			const validFrom = BENCHMARK_NOW_MS - (m.ageDays ?? 0) * DAY_MS;
			if (m.supersedesId) {
				const prior = byId.get(m.supersedesId);
				if (prior) prior.validUntil = validFrom;
			}
			const node: GraphitiNode = {
				id: m.id,
				text: m.content,
				tokens: tokenize(`${m.key} ${m.content}`),
				entities: extractSimpleEntities(`${m.key} ${m.content}`),
				years: extractYears(m.content),
				vector: Float32Array.from(vecs[i]!),
				validFrom,
				validUntil: null,
			};
			engine.nodes.push(node);
			byId.set(m.id, node);
			for (const ent of node.entities) {
				let set = engine.entityToNodes.get(ent);
				if (!set) {
					set = new Set();
					engine.entityToNodes.set(ent, set);
				}
				set.add(node.id);
			}
		}
		return engine;
	}

	async search(query: string, topK: number): Promise<RankedHit[]> {
		const isHistorical = /\b(before|previously|prior to|originally|used to|earlier)\b/i.test(query);
		const activeNodes = isHistorical ? this.nodes : this.nodes.filter((n) => n.validUntil === null);
		const [qVec] = await this.embedder.embed([query], { inputType: "query" });
		const qTokens = tokenize(query);
		const qEntities = extractSimpleEntities(query);
		const qYears = extractYears(query);

		// 1) Dense vector rank
		const vecRanked = activeNodes
			.map((n) => ({ id: n.id, cos: dotCosine(qVec!, n.vector) }))
			.sort((a, b) => b.cos - a.cos);
		const vecCos = new Map(vecRanked.map((r) => [r.id, r.cos]));

		// 2) BM25 rank
		const bm25 = computeBm25Map(activeNodes, qTokens);
		const bm25Ranked = Array.from(bm25.entries()).sort((a, b) => b[1] - a[1]);

		// 3) 1-hop entity BFS neighborhood from query entities + top-2 vector seeds
		const activeSet = new Set(activeNodes.map((n) => n.id));
		const graphBoost = new Map<string, number>();
		const seedEntities = new Set<string>(qEntities);
		for (const top of vecRanked.slice(0, 2)) {
			if (top.cos >= 0.42) {
				const n = activeNodes.find((x) => x.id === top.id);
				if (n) for (const e of n.entities) seedEntities.add(e);
			}
		}
		for (const ent of seedEntities) {
			const linked = this.entityToNodes.get(ent);
			if (!linked) continue;
			const w = 1 / Math.sqrt(linked.size);
			for (const id of linked) {
				if (activeSet.has(id)) graphBoost.set(id, (graphBoost.get(id) ?? 0) + w);
			}
		}
		const graphRanked = Array.from(graphBoost.entries()).sort((a, b) => b[1] - a[1]);

		// RRF fusion (k=60) + bi-temporal/year bonus
		const K = 60;
		const rrf = new Map<string, number>();
		vecRanked.slice(0, 30).forEach((r, idx) => {
			if (r.cos > 0.25) rrf.set(r.id, (rrf.get(r.id) ?? 0) + 1 / (K + idx + 1));
		});
		bm25Ranked.slice(0, 30).forEach(([id], idx) => {
			rrf.set(id, (rrf.get(id) ?? 0) + 1 / (K + idx + 1));
		});
		graphRanked.slice(0, 20).forEach(([id], idx) => {
			rrf.set(id, (rrf.get(id) ?? 0) + 0.7 / (K + idx + 1));
		});

		const maxRrf = 1 / (K + 1) + 1 / (K + 1) + 0.7 / (K + 1);
		const scored: RankedHit[] = [];
		for (const n of activeNodes) {
			const cos = vecCos.get(n.id) ?? 0;
			const bm = bm25.get(n.id) ?? 0;
			const gb = graphBoost.get(n.id) ?? 0;
			const yearHit = qYears.length > 0 && n.years.some((y) => qYears.includes(y)) ? 0.15 : 0;
			const histBoost = isHistorical && n.validUntil !== null ? 0.12 : 0;
			// Scale by cosine/bm25 strength so abstention stays calibrated
			const baseRel = Math.max(cos, bm > 0 ? Math.min(1, bm / 8) : 0, gb > 0 ? 0.4 : 0);
			const rrfNorm = (rrf.get(n.id) ?? 0) / maxRrf;
			const score = 0.55 * baseRel + 0.45 * rrfNorm + yearHit + histBoost;
			scored.push({ id: n.id, score, text: n.text });
		}
		scored.sort((a, b) => b.score - a.score);
		return scored.slice(0, topK);
	}
}

// ── 2. HippoRAG-Style Entity-Memory Bipartite Graph + Personalized PageRank (PPR) ──────────────

interface HippoDoc {
	readonly id: string;
	readonly text: string;
	readonly entities: string[];
	readonly vector: Float32Array;
	readonly active: boolean;
}

export class HippoRagRefEngine implements BenchMemoryEngine {
	readonly name = "hipporag-ref (entity-PPR)";
	readonly abstentionThreshold = 0.26;

	private readonly docs: HippoDoc[] = [];
	private readonly docById = new Map<string, HippoDoc>();
	private readonly entToDocs = new Map<string, string[]>();
	private readonly embedder: EmbeddingProvider;

	private constructor(embedder: EmbeddingProvider) {
		this.embedder = embedder;
	}

	static async build(memories: readonly BenchmarkMemory[], embedder: EmbeddingProvider): Promise<HippoRagRefEngine> {
		const engine = new HippoRagRefEngine(embedder);
		const superseded = new Set(memories.map((m) => m.supersedesId).filter(Boolean));
		const texts = memories.map((m) => `${m.key}: ${m.content}`);
		const vecs = await embedder.embed(texts, { inputType: "document" });

		for (let i = 0; i < memories.length; i++) {
			const m = memories[i]!;
			const active = !superseded.has(m.id);
			const entities = extractSimpleEntities(`${m.key} ${m.content}`);
			const doc: HippoDoc = {
				id: m.id,
				text: m.content,
				entities,
				vector: Float32Array.from(vecs[i]!),
				active,
			};
			engine.docs.push(doc);
			engine.docById.set(m.id, doc);
			if (active) {
				for (const ent of entities) {
					let arr = engine.entToDocs.get(ent);
					if (!arr) {
						arr = [];
						engine.entToDocs.set(ent, arr);
					}
					arr.push(m.id);
				}
			}
		}
		return engine;
	}

	async search(query: string, topK: number): Promise<RankedHit[]> {
		const activeDocs = this.docs.filter((d) => d.active);
		const [qVec] = await this.embedder.embed([query], { inputType: "query" });
		const qEntities = extractSimpleEntities(query);

		// 1) Initial dense passage similarities
		const cosByDoc = new Map<string, number>();
		const ranked = activeDocs
			.map((d) => {
				const cos = dotCosine(qVec!, d.vector);
				cosByDoc.set(d.id, cos);
				return { doc: d, cos };
			})
			.sort((a, b) => b.cos - a.cos);

		// 2) Build restart personalization vector over entities & top passages (HippoRAG 2 style)
		const entScore = new Map<string, number>();
		for (const qe of qEntities) {
			if (this.entToDocs.has(qe)) entScore.set(qe, (entScore.get(qe) ?? 0) + 1.0);
		}
		for (const seed of ranked.slice(0, 3)) {
			if (seed.cos < 0.35) continue;
			for (const ent of seed.doc.entities) {
				const deg = this.entToDocs.get(ent)?.length ?? 1;
				entScore.set(ent, (entScore.get(ent) ?? 0) + (seed.cos * 0.6) / Math.sqrt(deg));
			}
		}

		// 3) 2-iteration damped Personalized PageRank on bipartite Entity <-> Document graph
		const alpha = 0.5;
		const docPpr = new Map<string, number>();
		const seedDocMass = new Map<string, number>();
		for (const r of ranked.slice(0, 5)) {
			if (r.cos >= 0.3) seedDocMass.set(r.doc.id, r.cos);
		}

		let curEnt = new Map(entScore);
		for (let iter = 0; iter < 2; iter++) {
			// Entity -> Doc
			const nextDoc = new Map<string, number>();
			for (const [id, m] of seedDocMass) nextDoc.set(id, (nextDoc.get(id) ?? 0) + alpha * m);
			for (const [ent, mass] of curEnt) {
				const linked = this.entToDocs.get(ent);
				if (!linked || linked.length === 0) continue;
				const share = ((1 - alpha) * mass) / Math.sqrt(linked.length);
				for (const docId of linked) {
					nextDoc.set(docId, (nextDoc.get(docId) ?? 0) + share);
				}
			}
			for (const [id, v] of nextDoc) docPpr.set(id, v);

			// Doc -> Entity
			const nextEnt = new Map<string, number>();
			for (const [ent, m] of entScore) nextEnt.set(ent, alpha * m);
			for (const [docId, mass] of nextDoc) {
				const doc = this.docById.get(docId);
				if (!doc || doc.entities.length === 0) continue;
				const share = ((1 - alpha) * mass) / Math.sqrt(doc.entities.length);
				for (const ent of doc.entities) {
					nextEnt.set(ent, (nextEnt.get(ent) ?? 0) + share);
				}
			}
			curEnt = nextEnt;
		}

		const maxPpr = Math.max(...Array.from(docPpr.values()), 1e-9);
		const results: RankedHit[] = activeDocs.map((d) => {
			const cos = cosByDoc.get(d.id) ?? 0;
			const pprNorm = (docPpr.get(d.id) ?? 0) / maxPpr;
			// Gate PPR by whether there was any real seed signal
			const topCos = ranked[0]?.cos ?? 0;
			const score = topCos < 0.25 && qEntities.every((e) => !this.entToDocs.has(e))
				? cos
				: 0.55 * cos + 0.45 * pprNorm * Math.min(1, Math.max(topCos, 0.5));
			return { id: d.id, score, text: d.text };
		});
		results.sort((a, b) => b.score - a.score);
		return results.slice(0, topK);
	}
}
