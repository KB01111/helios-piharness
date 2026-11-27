import type {
	FileContentSearchOptions,
	FileFindOptions,
	FileIndexDatasetSummary,
	FileIndexMapResult,
	FileMatchHit,
	FileSearchHealth,
	FileSearchMode,
	FileSearchResult,
	FileSearchScope,
	FileSearchService,
	FileSymbolDefOptions,
	FileSymbolInfo,
} from "../contracts/app-capabilities.js";
import { readBoundedText } from "../mcp/mcp-client.js";

/**
 * Multi-tenant scoping strategy for mapping `FileSearchScope` to a XERJ index pattern.
 *
 * - `"tenant"` (default, safest for multi-tenant deployments): `<prefix>-<tenantSlug>-*`
 *   (or `<prefix>-<tenantSlug>-<workspaceSlug>-*` when `scope.workspaceId` is set).
 * - `"workspace"`: `<prefix>-<workspaceSlug || tenantSlug>-*`.
 * - `"shared"`: `<prefix>-*` (single-tenant local development or shared reference corpora).
 */
export type XerjScopeMode = "tenant" | "workspace" | "shared";

export interface XerjFileSearchOptions {
	/** Base URL of the XERJ ES-compatible REST surface (e.g. `http://127.0.0.1:9200`). */
	readonly baseUrl: string;
	/** Optional XERJ API key (from `<data_dir>/admin.key`). Sent as `Authorization: ApiKey <key>` and never logged. */
	readonly apiKey?: string;
	/** Authorization header scheme when `apiKey` is set (default `"ApiKey"`). */
	readonly authScheme?: "ApiKey" | "Bearer";
	/** Autoindex prefix (default `"ax"`, matching `xerj autoindex`). */
	readonly indexPrefix?: string;
	/** Tenant/workspace isolation mode (default `"tenant"`). */
	readonly scopeMode?: XerjScopeMode;
	/**
	 * Optional custom mapper from runtime `FileSearchScope` to a validated XERJ index pattern.
	 * Returned patterns are still strictly validated against path/wildcard injection.
	 */
	readonly resolveIndexPattern?: (scope: FileSearchScope) => string;
	/** Per-request HTTP timeout in milliseconds (default `10000`). */
	readonly timeoutMs?: number;
	/** Largest HTTP response body accepted from XERJ in bytes (default 4 MiB). */
	readonly maxResponseBytes?: number;
	/** TTL cache in ms for `getIndexMap` and `healthCheck` (default `10000`; `0` disables). */
	readonly cacheTtlMs?: number;
	/** Lightweight circuit breaker so an offline XERJ node fails fast instead of stalling turns. */
	readonly circuitBreaker?: {
		readonly failureThreshold?: number;
		readonly cooldownMs?: number;
	};
	readonly fetchImpl?: typeof fetch;
	readonly now?: () => number;
}

const DEFAULT_TIMEOUT_MS = 10_000;
const DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const DEFAULT_CACHE_TTL_MS = 10_000;
const DEFAULT_CB_FAILURES = 3;
const DEFAULT_CB_COOLDOWN_MS = 15_000;
const DEFAULT_SNIPPET_CHARS = 420;
const MAX_LIMIT = 50;

/** Source projection for file finding: intentionally omits `body` so payloads stay tiny. */
const FILE_FIND_SOURCE_FIELDS = [
	"title",
	"ax_path",
	"ax_paths",
	"path",
	"language",
	"ax_format",
	"ax_dataset",
	"ax_locator",
	"symbol_count",
] as const;

/**
 * Source projection for content & symbol search: omits full-file `body` and includes both
 * per-file metadata (`symbols`, `symbol_count`) and XERJ's per-symbol declaration fields
 * (`name`, `kind`, `line`, `start_line`, `code`).
 */
const CONTENT_SEARCH_SOURCE_FIELDS = [
	"title",
	"ax_path",
	"ax_paths",
	"path",
	"language",
	"ax_format",
	"ax_dataset",
	"ax_locator",
	"name",
	"kind",
	"line",
	"start_line",
	"code",
	"symbols",
	"symbol_count",
] as const;

/** Safe index pattern after resolution: lowercase alnum, `-`, `_`, optional single trailing `-*`. */
const SAFE_RESOLVED_INDEX_PATTERN = /^[a-z0-9][a-z0-9_-]{0,120}(?:-\*)?$/;

/**
 * Normalizes a prefix or scope segment using the same rule as `xerj autoindex --prefix`:
 * lowercase `[a-z0-9]` runs joined by `-`, trailing `-*` stripped, max 64 chars.
 */
export function sanitizeXerjIndexSlug(raw: string, label = "prefix"): string {
	const stripped = raw.trim().replace(/-\*$/, "");
	const runs = stripped.toLowerCase().match(/[a-z0-9]+/g);
	if (!runs || runs.length === 0) {
		throw new Error(`Invalid XERJ ${label} "${raw}": must contain at least one [a-z0-9] character`);
	}
	const slug = runs.join("-").slice(0, 64).replace(/-+$/g, "");
	if (!slug) {
		throw new Error(`Invalid XERJ ${label} "${raw}": empty after normalization`);
	}
	return slug;
}

export function validateXerjBaseUrl(raw: string): string {
	let parsed: URL;
	try {
		parsed = new URL(raw.trim());
	} catch {
		throw new Error(`Invalid XERJ URL "${raw}"`);
	}
	if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
		throw new Error(`Unsupported XERJ URL protocol "${parsed.protocol}"; expected http: or https:`);
	}
	if (parsed.username || parsed.password) {
		throw new Error("XERJ URL must not embed credentials; use apiKey / XERJ_API_KEY instead");
	}
	return parsed.toString().replace(/\/+$/, "");
}

/**
 * Normalizes a relative `pathPrefix` filter (forward slashes, no leading `./` or `/`) and
 * rejects path-traversal (`..`) or control characters.
 */
export function normalizePathPrefix(raw: string | undefined): string | undefined {
	if (raw === undefined) return undefined;
	const trimmed = raw.trim();
	if (!trimmed) return undefined;
	if (/[\0\r\n]/.test(trimmed)) {
		throw new Error("Invalid pathPrefix: control characters are not allowed");
	}
	const slashNorm = trimmed.replace(/\\+/g, "/").replace(/^(?:\.\/|\/)+/, "").replace(/\/{2,}/g, "/");
	if (!slashNorm) return undefined;
	const segments = slashNorm.split("/");
	if (segments.some((seg) => seg === "..")) {
		throw new Error('Invalid pathPrefix: parent directory traversal ("..") is not allowed');
	}
	return slashNorm;
}

function normalizeExtension(raw: string | undefined): string | undefined {
	if (!raw) return undefined;
	const cleaned = raw.trim().toLowerCase().replace(/^\.+/, "");
	if (!cleaned) return undefined;
	if (!/^[a-z0-9_-]{1,16}$/.test(cleaned)) {
		throw new Error(`Invalid file extension filter "${raw}"`);
	}
	return cleaned;
}

function normalizeLanguage(raw: string | undefined): string | undefined {
	if (!raw) return undefined;
	const cleaned = raw.trim().toLowerCase();
	if (!cleaned) return undefined;
	if (!/^[a-z0-9+#._-]{1,32}$/.test(cleaned)) {
		throw new Error(`Invalid language filter "${raw}"`);
	}
	return cleaned;
}

function clampLimit(limit: number | undefined, fallback: number): number {
	if (limit === undefined || !Number.isFinite(limit)) return fallback;
	return Math.max(1, Math.min(MAX_LIMIT, Math.trunc(limit)));
}

function escapeWildcardValue(raw: string): string {
	return raw.replace(/\\/g, "\\\\");
}

function toWildcardPattern(query: string): string {
	const cleaned = query.trim().replace(/\\+/g, "/");
	if (!cleaned) return "*";
	if (cleaned.includes("*") || cleaned.includes("?")) {
		return escapeWildcardValue(cleaned);
	}
	return `*${escapeWildcardValue(cleaned)}*`;
}

function truncateSnippet(text: string, maxChars: number): string {
	const trimmed = text.trim();
	if (trimmed.length <= maxChars) return trimmed;
	return `${trimmed.slice(0, Math.max(1, maxChars - 1)).trimEnd()}…`;
}

export class XerjRequestError extends Error {
	readonly status: number | undefined;
	readonly retryable: boolean;

	constructor(message: string, options?: { readonly status?: number; readonly retryable?: boolean; readonly cause?: unknown }) {
		super(message, options?.cause ? { cause: options.cause } : undefined);
		this.name = "XerjRequestError";
		this.status = options?.status;
		this.retryable = options?.retryable ?? false;
	}
}

interface EsHit {
	readonly _index?: string;
	readonly _id?: string;
	readonly _score?: number | null;
	readonly match_line?: number;
	readonly _source?: Record<string, unknown>;
	readonly fields?: Record<string, unknown>;
	readonly highlight?: Record<string, unknown>;
}

interface EsSearchResponse {
	readonly took?: number;
	readonly hits?: {
		readonly total?: number | { readonly value?: number };
		readonly hits?: readonly EsHit[];
	};
	readonly _xerj?: {
		readonly hints?: ReadonlyArray<{ readonly code?: string; readonly message?: string }>;
	};
}

function positiveInt(raw: unknown): number | undefined {
	return typeof raw === "number" && Number.isFinite(raw) && raw > 0 ? Math.trunc(raw) : undefined;
}

/**
 * Extracts symbol descriptors from either:
 *  - a XERJ per-symbol declaration doc (`_source.name`, `_source.kind`, `_source.line`/`start_line`), or
 *  - a per-file doc (`_source.symbols[]`).
 */
function parseSymbols(source: Record<string, unknown>): FileSymbolInfo[] {
	const out: FileSymbolInfo[] = [];

	if (typeof source.name === "string" && source.name.trim()) {
		const kind = typeof source.kind === "string" && source.kind.trim() ? source.kind.trim() : undefined;
		const line = positiveInt(source.line) ?? positiveInt(source.start_line);
		out.push({
			name: source.name.trim(),
			...(kind ? { kind } : {}),
			...(line !== undefined ? { line } : {}),
		});
	}

	if (Array.isArray(source.symbols)) {
		for (const item of source.symbols) {
			if (!item || typeof item !== "object") continue;
			const rec = item as Record<string, unknown>;
			if (typeof rec.name !== "string" || !rec.name.trim()) continue;
			const kind = typeof rec.kind === "string" && rec.kind.trim() ? rec.kind.trim() : undefined;
			const line = positiveInt(rec.line) ?? positiveInt(rec.start_line);
			out.push({
				name: rec.name.trim(),
				...(kind ? { kind } : {}),
				...(line !== undefined ? { line } : {}),
			});
		}
	}
	return out;
}

function extractPassage(hit: EsHit): { text?: string; line?: number } {
	const src = hit._source ?? {};
	const rawPassage = hit.fields?._passage;
	const first = Array.isArray(rawPassage) ? rawPassage[0] : rawPassage;
	if (typeof first === "string" && first.trim()) {
		return { text: first.trim() };
	}
	if (first && typeof first === "object") {
		const rec = first as Record<string, unknown>;
		const text = typeof rec.text === "string" && rec.text.trim() ? rec.text.trim() : undefined;
		const line = positiveInt(rec.match_line) ?? positiveInt(rec.start_line) ?? positiveInt(rec.line);
		if (text) return { text, ...(line !== undefined ? { line } : {}) };
	}
	// XERJ per-symbol declaration documents carry a compact declaration signature in `_source.code`.
	if (typeof src.code === "string" && src.code.trim()) {
		const line = positiveInt(src.line) ?? positiveInt(src.start_line);
		return { text: src.code.trim(), ...(line !== undefined ? { line } : {}) };
	}
	// Fallback to ES highlight fragments (used on `hybrid` queries where `_passage` is disallowed by XERJ).
	if (hit.highlight && typeof hit.highlight === "object") {
		for (const val of Object.values(hit.highlight)) {
			if (Array.isArray(val) && typeof val[0] === "string" && val[0].trim()) {
				return { text: val[0].trim() };
			}
		}
	}
	return {};
}

function extractPath(source: Record<string, unknown>, fallbackId?: string): string {
	const candidates = [source.ax_path, source.path, source.title, fallbackId];
	for (const candidate of candidates) {
		if (typeof candidate === "string" && candidate.trim()) {
			return candidate.trim().replace(/\\+/g, "/");
		}
	}
	return "(unknown)";
}

function extractDuplicatePaths(source: Record<string, unknown>, primaryPath: string): string[] | undefined {
	if (!Array.isArray(source.ax_paths)) return undefined;
	const extras = source.ax_paths
		.filter((p): p is string => typeof p === "string" && p.trim().length > 0)
		.map((p) => p.trim().replace(/\\+/g, "/"))
		.filter((p) => p !== primaryPath);
	return extras.length > 0 ? Array.from(new Set(extras)) : undefined;
}

function extractTotal(resp: EsSearchResponse, hitCount: number): number {
	const raw = resp.hits?.total;
	if (typeof raw === "number" && Number.isFinite(raw)) return raw;
	if (raw && typeof raw === "object" && typeof raw.value === "number" && Number.isFinite(raw.value)) {
		return raw.value;
	}
	return hitCount;
}

function extractHints(resp: EsSearchResponse): string[] {
	const hints = resp._xerj?.hints;
	if (!Array.isArray(hints)) return [];
	const out: string[] = [];
	for (const h of hints) {
		if (!h || typeof h !== "object") continue;
		const code = typeof h.code === "string" ? h.code.trim() : "";
		const msg = typeof h.message === "string" ? h.message.trim() : "";
		if (code && msg) out.push(`[${code}] ${msg}`);
		else if (msg) out.push(msg);
		else if (code) out.push(code);
	}
	return out;
}

function matchBestSymbolLine(symbols: readonly FileSymbolInfo[], query: string): number | undefined {
	if (symbols.length === 0) return undefined;
	const tokens = query
		.toLowerCase()
		.split(/[^a-z0-9_]+/)
		.filter((t) => t.length >= 2);
	if (tokens.length === 0) return undefined;
	const exactWhole = query.trim().toLowerCase();
	for (const sym of symbols) {
		if (sym.line !== undefined && sym.name.toLowerCase() === exactWhole) return sym.line;
	}
	for (const sym of symbols) {
		const lower = sym.name.toLowerCase();
		if (sym.line !== undefined && tokens.some((t) => lower === t)) return sym.line;
	}
	return undefined;
}

function isMappingOrSemanticError(status: number, bodyText: string): boolean {
	if (status !== 400 && status !== 422) return false;
	return /semantic|mapping|field|unsupported|query_shard_exception|parsing_exception|illegal_argument|search_phase_execution/i.test(
		bodyText,
	);
}

function isIndexNotFoundError(status: number, bodyText: string): boolean {
	if (status === 404) return true;
	return status === 400 && /index_not_found/i.test(bodyText);
}

/**
 * Zero-dependency XERJ client implementing `FileSearchService` over XERJ's Elasticsearch 8.x REST API.
 *
 * Verified live against XERJ `v1.0.0-rc.105`:
 * - Enforces tenant/workspace index scoping in code so LLM arguments can never cross tenant boundaries.
 * - Uses projected `_source` (omitting `body`) + `fields: ["_passage"]` for lexical/semantic queries,
 *   and `highlight` + `code` projection for `hybrid` queries (XERJ forbids `_passage` with `hybrid`).
 * - Handles both per-file documents (`defs`, `symbols`, `body`, `ax_locator: "code"|"file"`) and
 *   per-symbol declaration documents (`name`, `kind`, `line`, `start_line`, `code`).
 * - Matches file extensions via `wildcard` on `ax_path` (`*.ext`) because XERJ's `ax_format` holds
 *   the content family (`"code"`, `"json"`, `"txt-prose"`), not the filename extension.
 * - Refuses HTTP redirects, bounds response bytes, never logs credentials, and protects callers with a
 *   circuit breaker + single-flight health probe.
 */
export class XerjFileSearchService implements FileSearchService {
	readonly engine = "xerj";
	private readonly baseUrl: string;
	private readonly headers: Readonly<Record<string, string>>;
	private readonly indexPrefix: string;
	private readonly scopeMode: XerjScopeMode;
	private readonly customIndexResolver?: (scope: FileSearchScope) => string;
	private readonly timeoutMs: number;
	private readonly maxResponseBytes: number;
	private readonly cacheTtlMs: number;
	private readonly cbFailureThreshold: number;
	private readonly cbCooldownMs: number;
	private readonly fetchImpl: typeof fetch;
	private readonly now: () => number;

	private consecutiveFailures = 0;
	private circuitOpenUntil = 0;
	private healthCache?: { readonly at: number; readonly result: FileSearchHealth };
	private inFlightHealth?: Promise<FileSearchHealth>;
	private readonly indexMapCache = new Map<string, { readonly at: number; readonly result: FileIndexMapResult }>();

	constructor(options: XerjFileSearchOptions) {
		this.baseUrl = validateXerjBaseUrl(options.baseUrl);
		const scheme = options.authScheme ?? "ApiKey";
		this.headers = options.apiKey ? { authorization: `${scheme} ${options.apiKey}` } : {};
		this.indexPrefix = sanitizeXerjIndexSlug(options.indexPrefix ?? "ax", "indexPrefix");
		this.scopeMode = options.scopeMode ?? "tenant";
		this.customIndexResolver = options.resolveIndexPattern;
		this.timeoutMs = Math.max(500, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
		this.maxResponseBytes = Math.max(4096, options.maxResponseBytes ?? DEFAULT_MAX_RESPONSE_BYTES);
		this.cacheTtlMs = Math.max(0, options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS);
		this.cbFailureThreshold = Math.max(1, options.circuitBreaker?.failureThreshold ?? DEFAULT_CB_FAILURES);
		this.cbCooldownMs = Math.max(1_000, options.circuitBreaker?.cooldownMs ?? DEFAULT_CB_COOLDOWN_MS);
		this.fetchImpl = options.fetchImpl ?? fetch;
		this.now = options.now ?? (() => Date.now());
	}

	/** Resolves and strictly validates the XERJ index pattern for a runtime-bound `FileSearchScope`. */
	resolveIndex(scope: FileSearchScope): string {
		let pattern: string;
		if (this.customIndexResolver) {
			pattern = this.customIndexResolver(scope).trim().toLowerCase();
		} else if (this.scopeMode === "shared") {
			pattern = `${this.indexPrefix}-*`;
		} else if (this.scopeMode === "workspace") {
			const ws = sanitizeXerjIndexSlug(scope.workspaceId?.trim() || scope.tenantId, "workspaceId");
			pattern = `${this.indexPrefix}-${ws}-*`;
		} else {
			const tenantSlug = sanitizeXerjIndexSlug(scope.tenantId, "tenantId");
			const wsSlug = scope.workspaceId?.trim() ? sanitizeXerjIndexSlug(scope.workspaceId, "workspaceId") : undefined;
			pattern = wsSlug ? `${this.indexPrefix}-${tenantSlug}-${wsSlug}-*` : `${this.indexPrefix}-${tenantSlug}-*`;
		}

		if (!SAFE_RESOLVED_INDEX_PATTERN.test(pattern)) {
			throw new Error(`Unsafe resolved XERJ index pattern "${pattern}"`);
		}
		return pattern;
	}

	/** Returns the literal index prefix before `*` (e.g. `ax-tenant-a-` for `ax-tenant-a-*`). */
	private resolveLiteralPrefix(indexPattern: string): string {
		return indexPattern.endsWith("*") ? indexPattern.slice(0, -1) : indexPattern;
	}

	private buildFilters(options: {
		readonly pathPrefix?: string;
		readonly extension?: string;
		readonly language?: string;
	}): Array<Record<string, unknown>> {
		const filters: Array<Record<string, unknown>> = [];
		const pathPrefix = normalizePathPrefix(options.pathPrefix);
		const extension = normalizeExtension(options.extension);
		const language = normalizeLanguage(options.language);

		if (pathPrefix) {
			filters.push({ prefix: { ax_path: pathPrefix } });
		}
		if (extension) {
			// XERJ's `ax_format` stores the content family ("code", "json", "txt-prose"), not the file extension.
			// Match extensions reliably against the keyword `ax_path` field instead.
			filters.push({ wildcard: { ax_path: { value: `*.${extension}`, case_insensitive: true } } });
		}
		if (language) {
			filters.push({ term: { language } });
		}
		return filters;
	}

	async findFiles(scope: FileSearchScope, options: FileFindOptions): Promise<FileSearchResult> {
		const indexPattern = this.resolveIndex(scope);
		const limit = clampLimit(options.limit, 15);
		// Over-fetch slightly because XERJ autoindex can store both `ax_locator:"file"` and `"code"` records per file.
		const fetchSize = Math.min(MAX_LIMIT, Math.max(limit * 3, 20));
		const filters = this.buildFilters(options);
		const queryText = options.query?.trim() ?? "";

		// Exclude per-symbol declaration docs (`kind` exists) so file-finding returns file records only.
		const mustNot: Array<Record<string, unknown>> = [{ exists: { field: "kind" } }];

		let queryClause: Record<string, unknown>;
		if (!queryText) {
			queryClause = {
				bool: {
					...(filters.length > 0 ? { filter: filters } : {}),
					must_not: mustNot,
				},
			};
		} else {
			const wildcard = toWildcardPattern(queryText);
			const should: Array<Record<string, unknown>> = [
				{ wildcard: { ax_path: { value: wildcard, case_insensitive: true, boost: 4 } } },
				{ wildcard: { title: { value: wildcard, case_insensitive: true, boost: 3 } } },
				{ match: { title: { query: queryText, boost: 2 } } },
			];
			queryClause = {
				bool: {
					should,
					minimum_should_match: 1,
					...(filters.length > 0 ? { filter: filters } : {}),
					must_not: mustNot,
				},
			};
		}

		const body = {
			query: queryClause,
			size: fetchSize,
			_source: [...FILE_FIND_SOURCE_FIELDS],
		};

		const outcome = await this.searchIndex(indexPattern, body, options.signal);
		if (outcome.missingIndex) {
			return {
				hits: [],
				total: 0,
				engine: this.engine,
				mode: "lexical",
				notes: [`No indexed files found in scope "${indexPattern}". Index this workspace with \`xerj autoindex\`.`],
			};
		}

		// Deduplicate by path and merge metadata (e.g. `language` / `symbolCount` from `ax_locator:"code"` record).
		const byPath = new Map<string, FileMatchHit>();
		const order: string[] = [];
		for (const rawHit of outcome.response.hits?.hits ?? []) {
			const src = rawHit._source ?? {};
			const filePath = extractPath(src, rawHit._id);
			const score = typeof rawHit._score === "number" && Number.isFinite(rawHit._score) ? rawHit._score : 1;
			const dupes = extractDuplicatePaths(src, filePath);
			const language = typeof src.language === "string" && src.language ? src.language : undefined;
			const format = typeof src.ax_format === "string" && src.ax_format ? src.ax_format : undefined;
			const dataset = typeof src.ax_dataset === "string" && src.ax_dataset ? src.ax_dataset : undefined;
			const title = typeof src.title === "string" && src.title ? src.title : undefined;
			const symbolCount =
				typeof src.symbol_count === "number" && Number.isFinite(src.symbol_count) ? src.symbol_count : undefined;

			const existing = byPath.get(filePath);
			if (!existing) {
				order.push(filePath);
				byPath.set(filePath, {
					path: filePath,
					...(title ? { title } : {}),
					...(language ? { language } : {}),
					...(format ? { format } : {}),
					...(dataset ? { dataset } : {}),
					score,
					...(symbolCount !== undefined ? { symbolCount } : {}),
					...(dupes ? { duplicatePaths: dupes } : {}),
				});
			} else {
				byPath.set(filePath, {
					...existing,
					score: Math.max(existing.score, score),
					...(existing.language ? {} : language ? { language } : {}),
					...(existing.symbolCount !== undefined ? {} : symbolCount !== undefined ? { symbolCount } : {}),
					...(existing.duplicatePaths ? {} : dupes ? { duplicatePaths: dupes } : {}),
				});
			}
		}

		const hits = order.slice(0, limit).map((p) => byPath.get(p)!);
		const notes = extractHints(outcome.response);
		return {
			hits,
			total: Math.max(hits.length, extractTotal(outcome.response, hits.length)),
			...(typeof outcome.response.took === "number" ? { tookMs: outcome.response.took } : {}),
			engine: this.engine,
			mode: "lexical",
			...(notes.length > 0 ? { notes } : {}),
		};
	}

	async searchContent(scope: FileSearchScope, options: FileContentSearchOptions): Promise<FileSearchResult> {
		const queryText = options.query.trim();
		if (!queryText) {
			return this.findFiles(scope, {
				pathPrefix: options.pathPrefix,
				extension: options.extension,
				language: options.language,
				limit: options.limit,
				signal: options.signal,
			});
		}

		const indexPattern = this.resolveIndex(scope);
		const limit = clampLimit(options.limit, 8);
		const maxSnippetChars = Math.max(80, Math.min(2000, options.maxSnippetChars ?? DEFAULT_SNIPPET_CHARS));
		const requestedMode: FileSearchMode = options.mode ?? "lexical";
		const filters = this.buildFilters(options);

		// XERJ definition-first code query:
		// multi_match over [body, defs, code, title] + match_phrase on defs (boost 4, NOT defs^N)
		// + exact/wildcard symbol name boost for per-symbol docs.
		const lexicalQuery: Record<string, unknown> = {
			bool: {
				should: [
					{ multi_match: { query: queryText, fields: ["body", "defs", "code", "title"] } },
					{ match_phrase: { defs: { query: queryText, boost: 4 } } },
					{ term: { name: { value: queryText, boost: 6 } } },
					{ wildcard: { ax_path: { value: toWildcardPattern(queryText), case_insensitive: true, boost: 1.5 } } },
				],
				minimum_should_match: 1,
				...(filters.length > 0 ? { filter: filters } : {}),
			},
		};

		const buildBody = (mode: FileSearchMode): Record<string, unknown> => {
			if (mode === "semantic") {
				const semClause: Record<string, unknown> = {
					semantic: {
						field: "body",
						query: queryText,
						k: Math.max(limit, 10),
						...(filters.length > 0 ? { filter: { bool: { filter: filters } } } : {}),
					},
				};
				return {
					query: semClause,
					size: limit,
					_source: [...CONTENT_SEARCH_SOURCE_FIELDS],
					fields: ["_passage"],
				};
			}

			if (mode === "hybrid") {
				// XERJ forbids `fields: ["_passage"]` on hybrid queries ("no single unambiguous winning passage").
				// Use `highlight` + projected `code` instead so hybrid queries succeed without transferring `body`.
				const semClause: Record<string, unknown> = {
					semantic: {
						field: "body",
						query: queryText,
						k: Math.max(limit * 2, 16),
						...(filters.length > 0 ? { filter: { bool: { filter: filters } } } : {}),
					},
				};
				return {
					query: {
						hybrid: {
							queries: [
								{ query: lexicalQuery, weight: 1.0 },
								{ query: semClause, weight: 1.0 },
							],
							fusion: "rrf",
						},
					},
					size: limit,
					_source: [...CONTENT_SEARCH_SOURCE_FIELDS],
					highlight: { fields: { body: {}, code: {} } },
				};
			}

			return {
				query: lexicalQuery,
				size: limit,
				_source: [...CONTENT_SEARCH_SOURCE_FIELDS],
				fields: ["_passage"],
			};
		};

		let effectiveMode: FileSearchMode = requestedMode;
		const notes: string[] = [];
		let outcome: SearchOutcome;
		try {
			outcome = await this.searchIndex(indexPattern, buildBody(requestedMode), options.signal);
		} catch (error) {
			// If semantic/hybrid fails because `body` is not mapped as `semantic_text`, degrade cleanly to lexical.
			if (
				requestedMode !== "lexical" &&
				error instanceof XerjRequestError &&
				error.status !== undefined &&
				isMappingOrSemanticError(error.status, error.message)
			) {
				effectiveMode = "lexical";
				notes.push(`Index does not support ${requestedMode} search on "body"; fell back to definition-first lexical search.`);
				outcome = await this.searchIndex(indexPattern, buildBody("lexical"), options.signal);
			} else {
				throw error;
			}
		}

		if (outcome.missingIndex) {
			return {
				hits: [],
				total: 0,
				engine: this.engine,
				mode: effectiveMode,
				notes: [`No indexed files found in scope "${indexPattern}". Index this workspace with \`xerj autoindex\`.`],
			};
		}

		const hits: FileMatchHit[] = [];
		const seenLoc = new Set<string>();
		for (const rawHit of outcome.response.hits?.hits ?? []) {
			const src = rawHit._source ?? {};
			const filePath = extractPath(src, rawHit._id);
			const symbols = parseSymbols(src);
			const passage = extractPassage(rawHit);
			const bestLine =
				positiveInt(rawHit.match_line) ??
				positiveInt(src.line) ??
				positiveInt(src.start_line) ??
				passage.line ??
				matchBestSymbolLine(symbols, queryText);
			const locKey = `${filePath}:${bestLine ?? 0}`;
			if (seenLoc.has(locKey)) continue;
			seenLoc.add(locKey);
			const dupes = extractDuplicatePaths(src, filePath);

			hits.push({
				path: filePath,
				...(typeof src.title === "string" && src.title ? { title: src.title } : {}),
				...(typeof src.language === "string" && src.language ? { language: src.language } : {}),
				...(typeof src.ax_format === "string" && src.ax_format ? { format: src.ax_format } : {}),
				...(typeof src.ax_dataset === "string" && src.ax_dataset ? { dataset: src.ax_dataset } : {}),
				score: typeof rawHit._score === "number" && Number.isFinite(rawHit._score) ? rawHit._score : 0,
				...(bestLine !== undefined ? { line: bestLine } : {}),
				...(passage.text ? { snippet: truncateSnippet(passage.text, maxSnippetChars) } : {}),
				...(symbols.length > 0 ? { symbols: symbols.slice(0, 8) } : {}),
				...(typeof src.symbol_count === "number" && Number.isFinite(src.symbol_count)
					? { symbolCount: src.symbol_count }
					: {}),
				...(dupes ? { duplicatePaths: dupes } : {}),
			});
		}

		notes.push(...extractHints(outcome.response));
		return {
			hits,
			total: extractTotal(outcome.response, hits.length),
			...(typeof outcome.response.took === "number" ? { tookMs: outcome.response.took } : {}),
			engine: this.engine,
			mode: effectiveMode,
			...(notes.length > 0 ? { notes } : {}),
		};
	}

	async findSymbolDefinitions(scope: FileSearchScope, options: FileSymbolDefOptions): Promise<FileSearchResult> {
		const symbol = options.symbol.trim();
		if (!symbol) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical" };
		}

		const indexPattern = this.resolveIndex(scope);
		const limit = clampLimit(options.limit, 6);
		const fetchSize = Math.min(MAX_LIMIT, Math.max(limit * 4, 20));
		const filters = this.buildFilters({ pathPrefix: options.pathPrefix, language: options.language });
		const kindFilter = options.kind?.trim().toLowerCase();

		// Matches both XERJ per-symbol declaration documents (`name`, `kind`, `line`, `code`)
		// and per-file symbol index documents (`defs`, `symbols.name`).
		const body = {
			query: {
				bool: {
					should: [
						{ term: { name: { value: symbol, boost: 10 } } },
						{ wildcard: { name: { value: `*${escapeWildcardValue(symbol)}*`, case_insensitive: true, boost: 6 } } },
						{ match_phrase: { defs: { query: symbol, boost: 5 } } },
						{ match: { "symbols.name": { query: symbol, boost: 4 } } },
						{ match: { code: { query: symbol, boost: 3 } } },
						{ match: { defs: { query: symbol, boost: 2 } } },
					],
					minimum_should_match: 1,
					...(filters.length > 0 ? { filter: filters } : {}),
				},
			},
			size: fetchSize,
			_source: [...CONTENT_SEARCH_SOURCE_FIELDS],
			fields: ["_passage"],
		};

		const outcome = await this.searchIndex(indexPattern, body, options.signal);
		if (outcome.missingIndex) {
			return {
				hits: [],
				total: 0,
				engine: this.engine,
				mode: "lexical",
				notes: [`No indexed files found in scope "${indexPattern}".`],
			};
		}

		const symbolLower = symbol.toLowerCase();
		const candidates: Array<{ readonly hit: FileMatchHit; readonly rankKey: number }> = [];
		const seenLoc = new Set<string>();

		for (const rawHit of outcome.response.hits?.hits ?? []) {
			const src = rawHit._source ?? {};
			const filePath = extractPath(src, rawHit._id);
			const baseScore = typeof rawHit._score === "number" && Number.isFinite(rawHit._score) ? rawHit._score : 0;
			const symbols = parseSymbols(src);
			const passage = extractPassage(rawHit);
			const isDirectSymbolDoc = typeof src.name === "string" && src.name.trim().length > 0;

			const matchedSymbols = symbols.filter((sym) => {
				const nameLower = sym.name.toLowerCase();
				const nameMatches =
					nameLower === symbolLower || nameLower.includes(symbolLower) || symbolLower.includes(nameLower);
				if (!nameMatches) return false;
				if (kindFilter && sym.kind && sym.kind.toLowerCase() !== kindFilter) return false;
				if (kindFilter && !sym.kind) return false;
				return true;
			});

			if (matchedSymbols.length > 0) {
				for (const sym of matchedSymbols) {
					const locKey = `${filePath}:${sym.line ?? 0}:${sym.name}`;
					if (seenLoc.has(locKey)) continue;
					seenLoc.add(locKey);

					const nameLower = sym.name.toLowerCase();
					const matchBonus =
						sym.name === symbol
							? 100
							: nameLower === symbolLower
								? 80
								: nameLower.startsWith(symbolLower)
									? 40
									: 15;
					const directBonus = isDirectSymbolDoc ? 20 : 0;
					candidates.push({
						rankKey: matchBonus + directBonus + baseScore,
						hit: {
							path: filePath,
							...(typeof src.title === "string" && src.title ? { title: src.title } : {}),
							...(typeof src.language === "string" && src.language ? { language: src.language } : {}),
							...(typeof src.ax_format === "string" && src.ax_format ? { format: src.ax_format } : {}),
							...(typeof src.ax_dataset === "string" && src.ax_dataset ? { dataset: src.ax_dataset } : {}),
							score: baseScore,
							...(sym.line !== undefined ? { line: sym.line } : {}),
							symbols: [sym],
							...(passage.text ? { snippet: truncateSnippet(passage.text, DEFAULT_SNIPPET_CHARS) } : {}),
						},
					});
				}
			} else if (!kindFilter) {
				const line =
					positiveInt(rawHit.match_line) ?? positiveInt(src.line) ?? positiveInt(src.start_line) ?? passage.line;
				const locKey = `${filePath}:${line ?? 0}:_file`;
				if (seenLoc.has(locKey)) continue;
				seenLoc.add(locKey);
				candidates.push({
					rankKey: baseScore,
					hit: {
						path: filePath,
						...(typeof src.title === "string" && src.title ? { title: src.title } : {}),
						...(typeof src.language === "string" && src.language ? { language: src.language } : {}),
						...(typeof src.ax_format === "string" && src.ax_format ? { format: src.ax_format } : {}),
						...(typeof src.ax_dataset === "string" && src.ax_dataset ? { dataset: src.ax_dataset } : {}),
						score: baseScore,
						...(line !== undefined ? { line } : {}),
						...(passage.text ? { snippet: truncateSnippet(passage.text, DEFAULT_SNIPPET_CHARS) } : {}),
					},
				});
			}
		}

		candidates.sort((a, b) => b.rankKey - a.rankKey || a.hit.path.localeCompare(b.hit.path));
		const hits = candidates.slice(0, limit).map((c) => c.hit);
		const notes = extractHints(outcome.response);

		return {
			hits,
			total: extractTotal(outcome.response, hits.length),
			...(typeof outcome.response.took === "number" ? { tookMs: outcome.response.took } : {}),
			engine: this.engine,
			mode: "lexical",
			...(notes.length > 0 ? { notes } : {}),
		};
	}

	async getIndexMap(
		scope: FileSearchScope,
		options?: { readonly limit?: number; readonly signal?: AbortSignal },
	): Promise<FileIndexMapResult> {
		const indexPattern = this.resolveIndex(scope);
		const literalPrefix = this.resolveLiteralPrefix(indexPattern);
		const limit = clampLimit(options?.limit, 25);
		const cacheKey = `${indexPattern}:${limit}`;

		if (this.cacheTtlMs > 0) {
			const cached = this.indexMapCache.get(cacheKey);
			if (cached && this.now() - cached.at < this.cacheTtlMs) {
				return cached.result;
			}
		}

		const datasets: FileIndexDatasetSummary[] = [];
		const notes: string[] = [];

		// 1. Query XERJ's `autoindex-catalog` for dataset rows (`doc_kind: "dataset"` in rc.105+,
		//    or legacy catalog rows carrying `index`/`index_name`), skipping per-file catalog entries.
		const catalogOutcome = await this.searchIndex(
			"autoindex-catalog",
			{
				query: {
					bool: {
						should: [
							{ term: { doc_kind: "dataset" } },
							{ exists: { field: "index_name" } },
							{ exists: { field: "index" } },
						],
						minimum_should_match: 1,
					},
				},
				size: 100,
				_source: [
					"doc_kind",
					"index",
					"index_name",
					"slug",
					"dataset",
					"ax_dataset",
					"doc_count",
					"record_count",
					"file_count",
					"formats",
					"languages",
					"language",
					"sample_paths",
				],
			},
			options?.signal,
		).catch((err) => {
			if (err instanceof XerjRequestError && err.status === 404) {
				return { missingIndex: true as const, response: {} as EsSearchResponse };
			}
			throw err;
		});

		if (!catalogOutcome.missingIndex) {
			const seenIndices = new Set<string>();
			for (const hit of catalogOutcome.response.hits?.hits ?? []) {
				const src = hit._source ?? {};
				if (typeof src.doc_kind === "string" && src.doc_kind !== "dataset") continue;
				const idxName =
					(typeof src.index_name === "string" && src.index_name) ||
					(typeof src.index === "string" && src.index) ||
					"";
				if (!idxName || !idxName.startsWith(literalPrefix) || seenIndices.has(idxName)) continue;
				seenIndices.add(idxName);

				const docCount =
					typeof src.record_count === "number"
						? src.record_count
						: typeof src.doc_count === "number"
							? src.doc_count
							: typeof src.file_count === "number"
								? src.file_count
								: undefined;
				const languages = Array.isArray(src.languages)
					? src.languages.filter((l): l is string => typeof l === "string" && l.length > 0)
					: Array.isArray(src.formats)
						? src.formats.filter((l): l is string => typeof l === "string" && l.length > 0)
						: typeof src.language === "string" && src.language
							? [src.language]
							: undefined;
				const samplePaths = Array.isArray(src.sample_paths)
					? src.sample_paths.filter((p): p is string => typeof p === "string" && p.length > 0).slice(0, 5)
					: undefined;
				const dataset =
					(typeof src.slug === "string" && src.slug) ||
					(typeof src.dataset === "string" && src.dataset) ||
					(typeof src.ax_dataset === "string" && src.ax_dataset) ||
					undefined;

				datasets.push({
					index: idxName,
					...(dataset ? { dataset } : {}),
					...(docCount !== undefined ? { docCount } : {}),
					...(languages && languages.length > 0 ? { languages } : {}),
					...(samplePaths && samplePaths.length > 0 ? { samplePaths } : {}),
				});
				if (datasets.length >= limit) break;
			}
		}

		// 2. Fall back to `/_cat/indices/<pattern>?format=json` when catalog is absent or empty for this scope.
		if (datasets.length === 0) {
			const encoded = encodeURIComponent(indexPattern);
			try {
				const rows = (await this.requestJson<Array<Record<string, unknown>>>(
					"GET",
					`/_cat/indices/${encoded}?format=json`,
					undefined,
					options?.signal,
				)) as Array<Record<string, unknown>>;
				if (Array.isArray(rows)) {
					for (const row of rows) {
						const idx = typeof row.index === "string" ? row.index : "";
						if (!idx || !idx.startsWith(literalPrefix)) continue;
						const countRaw = row["docs.count"];
						const docCount =
							typeof countRaw === "number"
								? countRaw
								: typeof countRaw === "string" && /^\d+$/.test(countRaw)
									? Number.parseInt(countRaw, 10)
									: undefined;
						datasets.push({
							index: idx,
							...(docCount !== undefined ? { docCount } : {}),
						});
						if (datasets.length >= limit) break;
					}
				}
			} catch (err) {
				if (err instanceof XerjRequestError && err.status === 404) {
					notes.push(`No indices found matching "${indexPattern}".`);
				} else {
					throw err;
				}
			}
		}

		const totalDocs = datasets.reduce((sum, d) => sum + (d.docCount ?? 0), 0);
		const result: FileIndexMapResult = {
			engine: this.engine,
			indexScope: indexPattern,
			datasets,
			totalDocs,
			...(notes.length > 0 ? { notes } : {}),
		};

		if (this.cacheTtlMs > 0) {
			if (this.indexMapCache.size > 128) this.indexMapCache.clear();
			this.indexMapCache.set(cacheKey, { at: this.now(), result });
		}
		return result;
	}

	async healthCheck(signal?: AbortSignal): Promise<FileSearchHealth> {
		if (this.cacheTtlMs > 0 && this.healthCache && this.now() - this.healthCache.at < this.cacheTtlMs) {
			return this.healthCache.result;
		}
		if (this.inFlightHealth) return this.inFlightHealth;

		const run = (async (): Promise<FileSearchHealth> => {
			const started = this.now();
			try {
				const banner = await this.requestJson<{ readonly version?: { readonly number?: string } }>(
					"GET",
					"/",
					undefined,
					signal,
					{ bypassCircuitBreaker: true },
				);
				const result: FileSearchHealth = {
					ok: true,
					engine: this.engine,
					...(typeof banner?.version?.number === "string" ? { version: banner.version.number } : {}),
					latencyMs: Math.max(0, this.now() - started),
				};
				this.healthCache = { at: this.now(), result };
				return result;
			} catch (error) {
				const result: FileSearchHealth = {
					ok: false,
					engine: this.engine,
					latencyMs: Math.max(0, this.now() - started),
					error: error instanceof Error ? error.message : String(error),
				};
				return result;
			} finally {
				this.inFlightHealth = undefined;
			}
		})();

		this.inFlightHealth = run;
		return run;
	}

	private async searchIndex(
		indexPattern: string,
		body: Record<string, unknown>,
		signal?: AbortSignal,
	): Promise<SearchOutcome> {
		const encoded = encodeURIComponent(indexPattern);
		try {
			const response = await this.requestJson<EsSearchResponse>("POST", `/${encoded}/_search`, body, signal);
			return { missingIndex: false, response };
		} catch (error) {
			if (error instanceof XerjRequestError && error.status !== undefined && isIndexNotFoundError(error.status, error.message)) {
				return { missingIndex: true, response: {} };
			}
			throw error;
		}
	}

	private async requestJson<T>(
		method: "GET" | "POST",
		pathPart: string,
		body?: Record<string, unknown>,
		signal?: AbortSignal,
		opts?: { readonly bypassCircuitBreaker?: boolean },
	): Promise<T> {
		if (!opts?.bypassCircuitBreaker && this.circuitOpenUntil > this.now()) {
			throw new XerjRequestError("XERJ circuit breaker is open; search backend temporarily unavailable", {
				retryable: true,
			});
		}

		const controller = new AbortController();
		const timer = setTimeout(
			() => controller.abort(new XerjRequestError(`XERJ request timed out after ${this.timeoutMs}ms`, { retryable: true })),
			this.timeoutMs,
		);
		timer.unref?.();

		const onAbort = () => controller.abort(signal?.reason);
		if (signal) {
			if (signal.aborted) {
				clearTimeout(timer);
				throw signal.reason instanceof Error ? signal.reason : new Error("XERJ request aborted");
			}
			signal.addEventListener("abort", onAbort, { once: true });
		}

		try {
			const res = await this.fetchImpl(`${this.baseUrl}${pathPart}`, {
				method,
				headers: {
					accept: "application/json",
					...(body !== undefined ? { "content-type": "application/json" } : {}),
					...this.headers,
				},
				...(body !== undefined ? { body: JSON.stringify(body) } : {}),
				signal: controller.signal,
				// Refuse redirects so credentialed requests can never bounce to another origin.
				redirect: "error",
			});

			const text = await readBoundedText(res, this.maxResponseBytes);
			if (!res.ok) {
				if (res.status >= 500 || res.status === 429) {
					this.recordFailure();
					throw new XerjRequestError(`XERJ HTTP ${res.status}: ${text.slice(0, 240) || res.statusText}`, {
						status: res.status,
						retryable: true,
					});
				}
				if (res.status === 401 || res.status === 403) {
					throw new XerjRequestError(`XERJ authentication/authorization rejected (HTTP ${res.status})`, {
						status: res.status,
						retryable: false,
					});
				}
				throw new XerjRequestError(`XERJ HTTP ${res.status}: ${text.slice(0, 240) || res.statusText}`, {
					status: res.status,
					retryable: false,
				});
			}

			this.recordSuccess();
			if (!text.trim()) return {} as T;
			try {
				return JSON.parse(text) as T;
			} catch (error) {
				throw new XerjRequestError("XERJ returned invalid JSON", { retryable: false, cause: error });
			}
		} catch (error) {
			if (signal?.aborted) throw error;
			if (error instanceof XerjRequestError) {
				if (error.retryable && error.status === undefined) {
					this.recordFailure();
				}
				throw error;
			}
			this.recordFailure();
			throw new XerjRequestError(
				`XERJ request failed: ${error instanceof Error ? error.message : String(error)}`,
				{ retryable: true, cause: error },
			);
		} finally {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		}
	}

	private recordSuccess(): void {
		this.consecutiveFailures = 0;
		this.circuitOpenUntil = 0;
	}

	private recordFailure(): void {
		this.consecutiveFailures += 1;
		if (this.consecutiveFailures >= this.cbFailureThreshold) {
			this.circuitOpenUntil = this.now() + this.cbCooldownMs;
		}
	}
}

type SearchOutcome =
	| { readonly missingIndex: true; readonly response: EsSearchResponse }
	| { readonly missingIndex: false; readonly response: EsSearchResponse };
