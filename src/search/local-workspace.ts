import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import type {
	FileContentSearchOptions,
	FileFindOptions,
	FileIndexMapResult,
	FileMatchHit,
	FileSearchHealth,
	FileSearchResult,
	FileSearchScope,
	FileSearchService,
	FileSymbolDefOptions,
	FileSymbolInfo,
} from "../contracts/app-capabilities.js";
import { normalizePathPrefix } from "./xerj-client.js";

export interface LocalWorkspaceFileSearchOptions {
	/**
	 * Root directory for single-workspace deployments. For multi-tenant servers, prefer
	 * `resolveWorkspaceRoot` so each tenant/workspace resolves to its own directory.
	 */
	readonly workspaceRoot?: string;
	/** Optional multi-tenant mapper returning an explicit allowed root directory per scope (or undefined to deny). */
	readonly resolveWorkspaceRoot?: (scope: FileSearchScope) => string | undefined;
	/** Max matched files collected per query (default 4000). */
	readonly maxFiles?: number;
	/** Max total filesystem entries visited per walk regardless of filter matches (default `max(maxFiles * 4, 16000)`). */
	readonly maxVisitedEntries?: number;
	/** Max bytes read per text file (default 512 KiB). Enforced during the read itself. */
	readonly maxFileBytes?: number;
	/** Extra directory names to skip alongside defaults. */
	readonly ignoreDirs?: readonly string[];
}

const DEFAULT_IGNORE_DIRS = new Set([
	".git",
	"node_modules",
	"dist",
	"build",
	"target",
	".venv",
	"__pycache__",
	".vitest",
	".next",
	"coverage",
	".swarm",
]);

const EXT_TO_LANG: Readonly<Record<string, string>> = {
	ts: "typescript",
	tsx: "typescript",
	mts: "typescript",
	cts: "typescript",
	js: "javascript",
	jsx: "javascript",
	mjs: "javascript",
	cjs: "javascript",
	rs: "rust",
	py: "python",
	go: "go",
	java: "java",
	kt: "kotlin",
	cs: "csharp",
	cpp: "cpp",
	cc: "cpp",
	c: "c",
	h: "c",
	hpp: "cpp",
	rb: "ruby",
	php: "php",
	sql: "sql",
	md: "markdown",
	json: "json",
	yaml: "yaml",
	yml: "yaml",
	toml: "toml",
	sh: "shell",
	ps1: "powershell",
	html: "html",
	css: "css",
};

const SYMBOL_PATTERNS: ReadonlyArray<{ readonly kind: string; readonly regex: RegExp }> = [
	{ kind: "function", regex: /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)/ },
	{ kind: "class", regex: /^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)/ },
	{ kind: "interface", regex: /^\s*(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)/ },
	{ kind: "type", regex: /^\s*(?:export\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<|=)/ },
	{ kind: "const", regex: /^\s*(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?::|=)/ },
	{ kind: "function", regex: /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)/ },
	{ kind: "struct", regex: /^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)/ },
	{ kind: "enum", regex: /^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)/ },
	{ kind: "trait", regex: /^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)/ },
	{ kind: "function", regex: /^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/ },
	{ kind: "function", regex: /^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(/ },
];

const DEFAULT_SNIPPET_CHARS = 420;
const MAX_GLOB_PATTERN_CHARS = 256;
const YIELD_BATCH_SIZE = 64;
const O_NOFOLLOW = typeof fs.constants.O_NOFOLLOW === "number" ? fs.constants.O_NOFOLLOW : 0;

function yieldToEventLoop(): Promise<void> {
	return new Promise((resolve) => setImmediate(resolve));
}

function clampSnippetChars(raw: number | undefined): number {
	if (raw === undefined || !Number.isFinite(raw)) return DEFAULT_SNIPPET_CHARS;
	return Math.max(1, Math.min(2000, Math.trunc(raw)));
}

/**
 * Linear-time, ReDoS-free case-insensitive wildcard matcher supporting `*` and `?`.
 * Matches `pattern` anywhere inside `text` (wrapping with `*...*` when not already anchored with `*`).
 */
export function matchGlob(pattern: string, text: string): boolean {
	const bounded = pattern.slice(0, MAX_GLOB_PATTERN_CHARS).toLowerCase();
	const target = text.toLowerCase();
	if (!bounded) return true;
	const p = `*${bounded.replace(/\*+/g, "*")}*`;

	let pi = 0;
	let ti = 0;
	let starIdx = -1;
	let matchIdx = 0;

	while (ti < target.length) {
		if (pi < p.length && (p[pi] === "?" || p[pi] === target[ti])) {
			pi += 1;
			ti += 1;
		} else if (pi < p.length && p[pi] === "*") {
			starIdx = pi;
			matchIdx = ti;
			pi += 1;
		} else if (starIdx !== -1) {
			pi = starIdx + 1;
			matchIdx += 1;
			ti = matchIdx;
		} else {
			return false;
		}
	}
	while (pi < p.length && p[pi] === "*") {
		pi += 1;
	}
	return pi === p.length;
}

function detectFormatAndLanguage(relPath: string): { format: string; language?: string } {
	const ext = path.extname(relPath).toLowerCase().replace(/^\./, "");
	const format = ext || "unknown";
	const language = EXT_TO_LANG[format];
	return { format, ...(language ? { language } : {}) };
}

function extractSymbolsFromText(text: string): FileSymbolInfo[] {
	const lines = text.split(/\r?\n/);
	const out: FileSymbolInfo[] = [];
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i]!;
		if (line.length > 300) continue;
		for (const pat of SYMBOL_PATTERNS) {
			const m = pat.regex.exec(line);
			if (m && m[1]) {
				out.push({ name: m[1], kind: pat.kind, line: i + 1 });
				break;
			}
		}
	}
	return out;
}

function isInsideRoot(realPath: string, rootReal: string): boolean {
	return realPath === rootReal || realPath.startsWith(rootReal + path.sep);
}

/**
 * Safely reads a regular text file inside `rootReal` without following symlinks and enforcing
 * `maxFileBytes` *during* the read (reading at most `maxFileBytes + 1` bytes so a growing file
 * can never exhaust memory).
 */
async function readSafeTextFile(absPath: string, rootReal: string, maxFileBytes: number): Promise<string | undefined> {
	let real: string;
	try {
		const lst = await fsp.lstat(absPath);
		if (lst.isSymbolicLink() || !lst.isFile() || lst.size > maxFileBytes) return undefined;
		real = await fsp.realpath(absPath);
	} catch {
		return undefined;
	}
	if (!isInsideRoot(real, rootReal)) {
		return undefined;
	}

	let handle: fsp.FileHandle | undefined;
	try {
		handle = await fsp.open(real, fs.constants.O_RDONLY | O_NOFOLLOW);
		const st = await handle.stat();
		if (!st.isFile() || st.size > maxFileBytes) return undefined;

		const readCap = maxFileBytes + 1;
		const buf = Buffer.allocUnsafe(Math.min(readCap, Math.max(4096, st.size + 1)));
		let total = 0;
		const chunks: Buffer[] = [];
		for (;;) {
			const remaining = readCap - total;
			if (remaining <= 0) break;
			const target = total === 0 && remaining <= buf.length ? buf : Buffer.allocUnsafe(Math.min(65_536, remaining));
			const { bytesRead } = await handle.read(target, 0, Math.min(target.length, remaining), total);
			if (bytesRead === 0) break;
			total += bytesRead;
			if (total > maxFileBytes) return undefined;
			chunks.push(target.subarray(0, bytesRead));
		}

		const full = chunks.length === 1 ? chunks[0]! : Buffer.concat(chunks, total);
		const probeLen = Math.min(full.length, 4096);
		for (let i = 0; i < probeLen; i++) {
			if (full[i] === 0) return undefined;
		}
		return full.toString("utf8");
	} catch {
		return undefined;
	} finally {
		await handle?.close().catch(() => undefined);
	}
}

function tokenizeQuery(query: string): string[] {
	return Array.from(
		new Set(
			query
				.toLowerCase()
				.split(/[^a-z0-9_]+/)
				.filter((t) => t.length >= 2),
		),
	);
}

function truncateToCeiling(text: string, maxChars: number): string {
	const trimmed = text.trim();
	if (trimmed.length <= maxChars) return trimmed;
	if (maxChars <= 1) return trimmed.slice(0, maxChars);
	return `${trimmed.slice(0, maxChars - 1).trimEnd()}…`;
}

/**
 * Formats a snippet guaranteed to preserve the matched/definition anchor line inside `maxChars`,
 * adding surrounding context lines only when space remains.
 */
function buildMatchCenteredSnippet(
	lines: readonly string[],
	anchorIdx: number,
	queryLower: string,
	maxChars: number,
): string {
	const rawAnchor = (lines[anchorIdx] ?? "").trim();
	let anchor = rawAnchor;
	if (anchor.length > maxChars) {
		const hitPos = queryLower ? anchor.toLowerCase().indexOf(queryLower) : -1;
		if (hitPos > 0 && maxChars > 4) {
			const half = Math.floor((maxChars - 2) / 2);
			const start = Math.max(0, Math.min(hitPos - Math.floor(half / 2), anchor.length - (maxChars - 2)));
			const slice = anchor.slice(start, start + Math.max(1, maxChars - 2)).trim();
			const prefix = start > 0 ? "…" : "";
			const suffix = start + slice.length < anchor.length ? "…" : "";
			anchor = truncateToCeiling(`${prefix}${slice}${suffix}`, maxChars);
		} else {
			anchor = truncateToCeiling(anchor, maxChars);
		}
		return anchor;
	}

	const picked: string[] = [anchor];
	let used = anchor.length;

	// Prefer adding lines after the match/declaration first, then one preceding line if room remains.
	for (const offset of [1, 2, -1]) {
		const idx = anchorIdx + offset;
		if (idx < 0 || idx >= lines.length) continue;
		const candidate = lines[idx]!.trim();
		if (!candidate) continue;
		if (used + 1 + candidate.length <= maxChars) {
			if (offset < 0) picked.unshift(candidate);
			else picked.push(candidate);
			used += 1 + candidate.length;
		}
	}
	return picked.join("\n");
}

/**
 * Bounded, async, read-only local filesystem `FileSearchService` jailed strictly inside configured workspace root(s).
 *
 * Useful both as a graceful fallback when XERJ is offline and as a zero-daemon option for local workspaces.
 */
export class LocalWorkspaceFileSearchService implements FileSearchService {
	readonly engine = "local";
	private readonly workspaceRoot?: string;
	private readonly rootResolver?: (scope: FileSearchScope) => string | undefined;
	private readonly maxFiles: number;
	private readonly maxVisitedEntries: number;
	private readonly maxFileBytes: number;
	private readonly ignoreDirs: ReadonlySet<string>;

	constructor(options: LocalWorkspaceFileSearchOptions) {
		if (!options.workspaceRoot && !options.resolveWorkspaceRoot) {
			throw new Error("LocalWorkspaceFileSearchService requires workspaceRoot or resolveWorkspaceRoot");
		}
		this.workspaceRoot = options.workspaceRoot ? path.resolve(options.workspaceRoot) : undefined;
		this.rootResolver = options.resolveWorkspaceRoot;
		this.maxFiles = Math.max(10, options.maxFiles ?? 4000);
		this.maxVisitedEntries = Math.max(this.maxFiles, options.maxVisitedEntries ?? Math.max(this.maxFiles * 4, 16_000));
		this.maxFileBytes = Math.max(1024, options.maxFileBytes ?? 512 * 1024);
		this.ignoreDirs = new Set([...DEFAULT_IGNORE_DIRS, ...(options.ignoreDirs ?? [])]);
	}

	private async resolveRoot(scope: FileSearchScope): Promise<{ root: string; rootReal: string } | undefined> {
		const raw = this.rootResolver ? this.rootResolver(scope) : this.workspaceRoot;
		if (!raw) return undefined;
		const resolved = path.resolve(raw);
		try {
			const st = await fsp.stat(resolved);
			if (!st.isDirectory()) return undefined;
			const rootReal = await fsp.realpath(resolved);
			return { root: resolved, rootReal };
		} catch {
			return undefined;
		}
	}

	/**
	 * Safely walks files inside `rootReal` using async `fs.promises`:
	 * - Validates each queued directory with `lstat` (refusing symlinks) + `realpath` inside `rootReal`
	 *   immediately before opening it, preventing symlink replacement from enumerating outside the workspace.
	 * - Bounds both visited entries (`maxVisitedEntries`) and accepted files (`maxFiles`).
	 * - Yields periodically to the event loop and honors `AbortSignal`.
	 */
	private async walkFiles(
		root: string,
		rootReal: string,
		filters: { readonly pathPrefix?: string; readonly extension?: string; readonly language?: string },
		signal?: AbortSignal,
	): Promise<{
		readonly files: Array<{ readonly relPath: string; readonly absPath: string; readonly format: string; readonly language?: string }>;
		readonly hitVisitCap: boolean;
	}> {
		const pathPrefix = normalizePathPrefix(filters.pathPrefix);
		const extFilter = filters.extension?.trim().toLowerCase().replace(/^\.+/, "");
		const langFilter = filters.language?.trim().toLowerCase();

		const results: Array<{ relPath: string; absPath: string; format: string; language?: string }> = [];
		const stack: string[] = [rootReal];
		let visited = 0;
		let hitVisitCap = false;

		while (stack.length > 0 && results.length < this.maxFiles && visited < this.maxVisitedEntries) {
			if (signal?.aborted) break;
			const dir = stack.pop()!;

			// Re-verify the directory itself immediately before opening (blocks symlink swap TOCTOU).
			let dirReal: string;
			try {
				const lst = await fsp.lstat(dir);
				if (lst.isSymbolicLink() || !lst.isDirectory()) continue;
				dirReal = await fsp.realpath(dir);
			} catch {
				continue;
			}
			if (!isInsideRoot(dirReal, rootReal)) continue;

			let handle: fs.Dir | undefined;
			try {
				handle = await fsp.opendir(dirReal);
				const childDirs: string[] = [];
				for await (const entry of handle) {
					if (signal?.aborted) break;
					visited += 1;
					if (visited > this.maxVisitedEntries) {
						hitVisitCap = true;
						break;
					}
					if (visited % YIELD_BATCH_SIZE === 0) {
						await yieldToEventLoop();
					}
					if (results.length >= this.maxFiles) break;
					if (entry.isSymbolicLink()) continue;

					const abs = path.join(dirReal, entry.name);
					if (entry.isDirectory()) {
						if (this.ignoreDirs.has(entry.name) || entry.name.startsWith(".")) continue;
						// Prune subdirectories that cannot possibly match pathPrefix.
						if (pathPrefix) {
							const relDir = `${path.relative(rootReal, abs).replace(/\\+/g, "/")}/`;
							if (!pathPrefix.startsWith(relDir) && !relDir.startsWith(pathPrefix)) continue;
						}
						childDirs.push(abs);
						continue;
					}
					if (!entry.isFile()) continue;

					const relPath = path.relative(rootReal, abs).replace(/\\+/g, "/");
					if (pathPrefix && !relPath.startsWith(pathPrefix)) continue;
					const { format, language } = detectFormatAndLanguage(relPath);
					if (extFilter && format !== extFilter) continue;
					if (langFilter && language !== langFilter) continue;

					results.push({ relPath, absPath: abs, format, ...(language ? { language } : {}) });
				}
				childDirs.sort((a, b) => b.localeCompare(a));
				for (const sub of childDirs) stack.push(sub);
			} catch {
				continue;
			} finally {
				await handle?.close().catch(() => undefined);
			}
		}

		// Keep `root` reference meaningful if rootReal differs only by symlink/casing on Windows.
		void root;
		return { files: results, hitVisitCap };
	}

	async findFiles(scope: FileSearchScope, options: FileFindOptions): Promise<FileSearchResult> {
		const started = Date.now();
		if (options.signal?.aborted) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Search aborted."] };
		}
		const rootInfo = await this.resolveRoot(scope);
		if (!rootInfo) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Workspace root not configured for this scope."] };
		}

		const limit = Math.max(1, Math.min(50, Math.trunc(options.limit ?? 15)));
		const { files, hitVisitCap } = await this.walkFiles(rootInfo.root, rootInfo.rootReal, options, options.signal);
		const queryText = options.query?.trim() ?? "";

		const scored: FileMatchHit[] = [];
		const queryLower = queryText.toLowerCase();
		const tokens = tokenizeQuery(queryText);

		for (let i = 0; i < files.length; i++) {
			if (i % YIELD_BATCH_SIZE === 0 && options.signal?.aborted) break;
			const f = files[i]!;
			const base = path.basename(f.relPath);
			let score = 1;
			if (queryText) {
				const baseLower = base.toLowerCase();
				const pathLower = f.relPath.toLowerCase();
				if (baseLower === queryLower || pathLower === queryLower) {
					score = 10;
				} else if (matchGlob(queryText, base)) {
					score = 6;
				} else if (matchGlob(queryText, f.relPath)) {
					score = 4;
				} else if (tokens.length > 0 && tokens.every((t) => pathLower.includes(t))) {
					score = 3;
				} else {
					continue;
				}
			}
			scored.push({
				path: f.relPath,
				title: base,
				format: f.format,
				...(f.language ? { language: f.language } : {}),
				score,
			});
		}

		scored.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
		const notes = hitVisitCap ? [`Scan stopped at ${this.maxVisitedEntries} visited entries; narrow pathPrefix.`] : [];
		return {
			hits: scored.slice(0, limit),
			total: scored.length,
			tookMs: Date.now() - started,
			engine: this.engine,
			mode: "lexical",
			...(notes.length > 0 ? { notes } : {}),
		};
	}

	async searchContent(scope: FileSearchScope, options: FileContentSearchOptions): Promise<FileSearchResult> {
		const started = Date.now();
		if (options.signal?.aborted) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Search aborted."] };
		}
		const rootInfo = await this.resolveRoot(scope);
		if (!rootInfo) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Workspace root not configured for this scope."] };
		}

		const queryText = options.query.trim();
		if (!queryText) return this.findFiles(scope, options);

		const limit = Math.max(1, Math.min(50, Math.trunc(options.limit ?? 8)));
		const maxSnippetChars = clampSnippetChars(options.maxSnippetChars);
		const { files, hitVisitCap } = await this.walkFiles(rootInfo.root, rootInfo.rootReal, options, options.signal);
		const queryLower = queryText.toLowerCase();
		const tokens = tokenizeQuery(queryText);
		const notes: string[] = [];
		if (options.mode && options.mode !== "lexical") {
			notes.push(`Local workspace engine supports lexical search only (requested "${options.mode}").`);
		}
		if (hitVisitCap) {
			notes.push(`Scan stopped at ${this.maxVisitedEntries} visited entries; narrow pathPrefix.`);
		}

		const hits: FileMatchHit[] = [];
		for (let fi = 0; fi < files.length; fi++) {
			if (options.signal?.aborted) break;
			if (fi > 0 && fi % 16 === 0) await yieldToEventLoop();
			const f = files[fi]!;
			const content = await readSafeTextFile(f.absPath, rootInfo.rootReal, this.maxFileBytes);
			if (content === undefined) continue;

			const lines = content.split(/\r?\n/);
			const symbols = extractSymbolsFromText(content);
			let bestLineIdx = -1;
			let bestLineScore = 0;
			let matchedLines = 0;
			let phraseHits = 0;

			for (let i = 0; i < lines.length; i++) {
				const lower = lines[i]!.toLowerCase();
				let lineScore = 0;
				if (lower.includes(queryLower)) {
					lineScore += 6;
					phraseHits += 1;
				}
				let tokenHits = 0;
				for (const tok of tokens) {
					if (lower.includes(tok)) tokenHits += 1;
				}
				if (tokenHits > 0) {
					lineScore += (tokenHits / Math.max(1, tokens.length)) * 3;
					matchedLines += 1;
				}
				if (lineScore > bestLineScore) {
					bestLineScore = lineScore;
					bestLineIdx = i;
				}
			}

			let defBoost = 0;
			let defLine: number | undefined;
			for (const sym of symbols) {
				const symLower = sym.name.toLowerCase();
				if (symLower === queryLower) {
					defBoost = Math.max(defBoost, 10);
					defLine = sym.line;
				} else if (tokens.some((t) => symLower === t)) {
					defBoost = Math.max(defBoost, 4);
					if (defLine === undefined) defLine = sym.line;
				}
			}

			const pathBoost = f.relPath.toLowerCase().includes(queryLower) ? 2 : 0;
			const totalScore = bestLineScore + Math.min(3, matchedLines * 0.25) + Math.min(4, phraseHits) + defBoost + pathBoost;
			if (totalScore <= 0 || (bestLineIdx < 0 && defLine === undefined)) continue;

			const anchorIdx = defLine !== undefined && defBoost >= 4 ? defLine - 1 : Math.max(0, bestLineIdx);
			const snippet = buildMatchCenteredSnippet(lines, anchorIdx, queryLower, maxSnippetChars);

			hits.push({
				path: f.relPath,
				title: path.basename(f.relPath),
				format: f.format,
				...(f.language ? { language: f.language } : {}),
				score: Number(totalScore.toFixed(3)),
				line: anchorIdx + 1,
				snippet,
				...(symbols.length > 0 ? { symbols: symbols.slice(0, 8), symbolCount: symbols.length } : {}),
			});
		}

		hits.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
		return {
			hits: hits.slice(0, limit),
			total: hits.length,
			tookMs: Date.now() - started,
			engine: this.engine,
			mode: "lexical",
			...(notes.length > 0 ? { notes } : {}),
		};
	}

	async findSymbolDefinitions(scope: FileSearchScope, options: FileSymbolDefOptions): Promise<FileSearchResult> {
		const started = Date.now();
		if (options.signal?.aborted) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Search aborted."] };
		}
		const rootInfo = await this.resolveRoot(scope);
		if (!rootInfo) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Workspace root not configured for this scope."] };
		}

		const symbol = options.symbol.trim();
		if (!symbol) return { hits: [], total: 0, engine: this.engine, mode: "lexical" };
		const symbolLower = symbol.toLowerCase();
		const kindFilter = options.kind?.trim().toLowerCase();
		const limit = Math.max(1, Math.min(50, Math.trunc(options.limit ?? 6)));

		const { files, hitVisitCap } = await this.walkFiles(
			rootInfo.root,
			rootInfo.rootReal,
			{ pathPrefix: options.pathPrefix, language: options.language },
			options.signal,
		);

		const hits: FileMatchHit[] = [];
		for (let fi = 0; fi < files.length; fi++) {
			if (options.signal?.aborted) break;
			if (fi > 0 && fi % 16 === 0) await yieldToEventLoop();
			const f = files[fi]!;
			const content = await readSafeTextFile(f.absPath, rootInfo.rootReal, this.maxFileBytes);
			if (content === undefined) continue;
			const lines = content.split(/\r?\n/);
			const symbols = extractSymbolsFromText(content);
			for (const sym of symbols) {
				if (kindFilter && sym.kind?.toLowerCase() !== kindFilter) continue;
				const nameLower = sym.name.toLowerCase();
				let score = 0;
				if (sym.name === symbol) score = 100;
				else if (nameLower === symbolLower) score = 80;
				else if (nameLower.startsWith(symbolLower)) score = 40;
				else if (nameLower.includes(symbolLower)) score = 15;
				else continue;

				const lineIdx = Math.max(0, (sym.line ?? 1) - 1);
				const snippet = buildMatchCenteredSnippet(lines, lineIdx, symbolLower, 320);
				hits.push({
					path: f.relPath,
					title: path.basename(f.relPath),
					format: f.format,
					...(f.language ? { language: f.language } : {}),
					score,
					...(sym.line !== undefined ? { line: sym.line } : {}),
					symbols: [sym],
					snippet,
				});
			}
		}

		hits.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
		const notes = hitVisitCap ? [`Scan stopped at ${this.maxVisitedEntries} visited entries; narrow pathPrefix.`] : [];
		return {
			hits: hits.slice(0, limit),
			total: hits.length,
			tookMs: Date.now() - started,
			engine: this.engine,
			mode: "lexical",
			...(notes.length > 0 ? { notes } : {}),
		};
	}

	async getIndexMap(
		scope: FileSearchScope,
		options?: { readonly limit?: number; readonly signal?: AbortSignal },
	): Promise<FileIndexMapResult> {
		if (options?.signal?.aborted) {
			return { engine: this.engine, indexScope: "local-workspace", datasets: [], totalDocs: 0, notes: ["Aborted."] };
		}
		const rootInfo = await this.resolveRoot(scope);
		if (!rootInfo) {
			return { engine: this.engine, indexScope: "(unconfigured)", datasets: [], totalDocs: 0 };
		}
		const limit = Math.max(1, Math.min(50, Math.trunc(options?.limit ?? 25)));
		const { files, hitVisitCap } = await this.walkFiles(rootInfo.root, rootInfo.rootReal, {}, options?.signal);
		const byLang = new Map<string, string[]>();
		for (const f of files) {
			const key = f.language ?? f.format;
			const list = byLang.get(key) ?? [];
			list.push(f.relPath);
			byLang.set(key, list);
		}
		const datasets = Array.from(byLang.entries())
			.sort((a, b) => b[1].length - a[1].length)
			.slice(0, limit)
			.map(([lang, paths]) => ({
				index: `local:${lang}`,
				dataset: lang,
				docCount: paths.length,
				languages: [lang],
				samplePaths: paths.slice(0, 4),
			}));
		const notes = hitVisitCap ? [`Scan stopped at ${this.maxVisitedEntries} visited entries.`] : [];
		return {
			engine: this.engine,
			indexScope: "local-workspace",
			datasets,
			totalDocs: files.length,
			...(notes.length > 0 ? { notes } : {}),
		};
	}

	async healthCheck(): Promise<FileSearchHealth> {
		if (!this.workspaceRoot) return { ok: true, engine: this.engine, latencyMs: 0 };
		try {
			const st = await fsp.stat(this.workspaceRoot);
			return { ok: st.isDirectory(), engine: this.engine, latencyMs: 0 };
		} catch {
			return { ok: false, engine: this.engine, latencyMs: 0 };
		}
	}
}
