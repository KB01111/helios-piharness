import fs from "node:fs";
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
	/** Root directory for single-workspace deployments. */
	readonly workspaceRoot?: string;
	/** Optional multi-tenant mapper returning an explicit allowed root directory per scope (or undefined to deny). */
	readonly resolveWorkspaceRoot?: (scope: FileSearchScope) => string | undefined;
	/** Max files walked per query (default 4000). */
	readonly maxFiles?: number;
	/** Max bytes read per text file (default 512 KiB). */
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

function globToRegex(pattern: string): RegExp {
	const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".");
	return new RegExp(escaped, "i");
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

function readSafeTextFile(absPath: string, rootReal: string, maxFileBytes: number): string | undefined {
	let real: string;
	try {
		real = fs.realpathSync(absPath);
	} catch {
		return undefined;
	}
	if (real !== rootReal && !real.startsWith(rootReal + path.sep)) {
		return undefined;
	}
	let stat: fs.Stats;
	try {
		stat = fs.statSync(real);
	} catch {
		return undefined;
	}
	if (!stat.isFile() || stat.size > maxFileBytes) return undefined;
	let buf: Buffer;
	try {
		buf = fs.readFileSync(real);
	} catch {
		return undefined;
	}
	// Skip binary files (NUL byte in first 4 KiB)
	const probeLen = Math.min(buf.length, 4096);
	for (let i = 0; i < probeLen; i++) {
		if (buf[i] === 0) return undefined;
	}
	return buf.toString("utf8");
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

/**
 * Bounded, read-only local filesystem `FileSearchService` jailed strictly inside configured workspace root(s).
 *
 * Useful both as a graceful fallback when XERJ is offline and as a zero-daemon option for small local workspaces.
 */
export class LocalWorkspaceFileSearchService implements FileSearchService {
	readonly engine = "local";
	private readonly workspaceRoot?: string;
	private readonly rootResolver?: (scope: FileSearchScope) => string | undefined;
	private readonly maxFiles: number;
	private readonly maxFileBytes: number;
	private readonly ignoreDirs: ReadonlySet<string>;

	constructor(options: LocalWorkspaceFileSearchOptions) {
		if (!options.workspaceRoot && !options.resolveWorkspaceRoot) {
			throw new Error("LocalWorkspaceFileSearchService requires workspaceRoot or resolveWorkspaceRoot");
		}
		this.workspaceRoot = options.workspaceRoot ? path.resolve(options.workspaceRoot) : undefined;
		this.rootResolver = options.resolveWorkspaceRoot;
		this.maxFiles = Math.max(10, options.maxFiles ?? 4000);
		this.maxFileBytes = Math.max(1024, options.maxFileBytes ?? 512 * 1024);
		this.ignoreDirs = new Set([...DEFAULT_IGNORE_DIRS, ...(options.ignoreDirs ?? [])]);
	}

	private resolveRoot(scope: FileSearchScope): { root: string; rootReal: string } | undefined {
		const raw = this.rootResolver ? this.rootResolver(scope) : this.workspaceRoot;
		if (!raw) return undefined;
		const resolved = path.resolve(raw);
		if (!fs.existsSync(resolved)) return undefined;
		return { root: resolved, rootReal: fs.realpathSync(resolved) };
	}

	private walkFiles(
		root: string,
		rootReal: string,
		filters: { readonly pathPrefix?: string; readonly extension?: string; readonly language?: string },
		signal?: AbortSignal,
	): Array<{ readonly relPath: string; readonly absPath: string; readonly format: string; readonly language?: string }> {
		const pathPrefix = normalizePathPrefix(filters.pathPrefix);
		const extFilter = filters.extension?.trim().toLowerCase().replace(/^\.+/, "");
		const langFilter = filters.language?.trim().toLowerCase();

		const results: Array<{ relPath: string; absPath: string; format: string; language?: string }> = [];
		const stack: string[] = [root];

		while (stack.length > 0 && results.length < this.maxFiles) {
			if (signal?.aborted) break;
			const dir = stack.pop()!;
			let entries: fs.Dirent[];
			try {
				entries = fs.readdirSync(dir, { withFileTypes: true });
			} catch {
				continue;
			}
			entries.sort((a, b) => b.name.localeCompare(a.name));
			for (const entry of entries) {
				if (results.length >= this.maxFiles) break;
				if (entry.isSymbolicLink()) continue;
				const abs = path.join(dir, entry.name);
				if (entry.isDirectory()) {
					if (this.ignoreDirs.has(entry.name) || entry.name.startsWith(".")) continue;
					stack.push(abs);
					continue;
				}
				if (!entry.isFile()) continue;
				const relPath = path.relative(root, abs).replace(/\\+/g, "/");
				if (pathPrefix && !relPath.startsWith(pathPrefix)) continue;
				const { format, language } = detectFormatAndLanguage(relPath);
				if (extFilter && format !== extFilter) continue;
				if (langFilter && language !== langFilter) continue;

				// Double-check realpath boundary
				let real: string;
				try {
					real = fs.realpathSync(abs);
				} catch {
					continue;
				}
				if (real !== rootReal && !real.startsWith(rootReal + path.sep)) continue;

				results.push({ relPath, absPath: abs, format, ...(language ? { language } : {}) });
			}
		}

		return results;
	}

	async findFiles(scope: FileSearchScope, options: FileFindOptions): Promise<FileSearchResult> {
		const started = Date.now();
		const rootInfo = this.resolveRoot(scope);
		if (!rootInfo) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Workspace root not configured for this scope."] };
		}

		const limit = Math.max(1, Math.min(50, Math.trunc(options.limit ?? 15)));
		const files = this.walkFiles(rootInfo.root, rootInfo.rootReal, options, options.signal);
		const queryText = options.query?.trim() ?? "";

		const scored: FileMatchHit[] = [];
		const regex = queryText ? globToRegex(queryText) : undefined;
		const queryLower = queryText.toLowerCase();
		const tokens = tokenizeQuery(queryText);

		for (const f of files) {
			const base = path.basename(f.relPath);
			let score = 1;
			if (queryText) {
				const baseLower = base.toLowerCase();
				const pathLower = f.relPath.toLowerCase();
				if (baseLower === queryLower || pathLower === queryLower) {
					score = 10;
				} else if (regex?.test(base)) {
					score = 6;
				} else if (regex?.test(f.relPath)) {
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
		return {
			hits: scored.slice(0, limit),
			total: scored.length,
			tookMs: Date.now() - started,
			engine: this.engine,
			mode: "lexical",
		};
	}

	async searchContent(scope: FileSearchScope, options: FileContentSearchOptions): Promise<FileSearchResult> {
		const started = Date.now();
		const rootInfo = this.resolveRoot(scope);
		if (!rootInfo) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Workspace root not configured for this scope."] };
		}

		const queryText = options.query.trim();
		if (!queryText) return this.findFiles(scope, options);

		const limit = Math.max(1, Math.min(50, Math.trunc(options.limit ?? 8)));
		const maxSnippetChars = Math.max(80, Math.min(2000, options.maxSnippetChars ?? 420));
		const files = this.walkFiles(rootInfo.root, rootInfo.rootReal, options, options.signal);
		const queryLower = queryText.toLowerCase();
		const tokens = tokenizeQuery(queryText);
		const notes: string[] = [];
		if (options.mode && options.mode !== "lexical") {
			notes.push(`Local workspace engine supports lexical search only (requested "${options.mode}").`);
		}

		const hits: FileMatchHit[] = [];
		for (const f of files) {
			if (options.signal?.aborted) break;
			const content = readSafeTextFile(f.absPath, rootInfo.rootReal, this.maxFileBytes);
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

			// Definition boost matching XERJ's definition-first heuristic
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
			const startIdx = Math.max(0, anchorIdx - 1);
			const endIdx = Math.min(lines.length, anchorIdx + 3);
			const rawSnippet = lines.slice(startIdx, endIdx).join("\n").trim();
			const snippet = rawSnippet.length > maxSnippetChars ? `${rawSnippet.slice(0, maxSnippetChars - 1).trimEnd()}…` : rawSnippet;

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
		const rootInfo = this.resolveRoot(scope);
		if (!rootInfo) {
			return { hits: [], total: 0, engine: this.engine, mode: "lexical", notes: ["Workspace root not configured for this scope."] };
		}

		const symbol = options.symbol.trim();
		if (!symbol) return { hits: [], total: 0, engine: this.engine, mode: "lexical" };
		const symbolLower = symbol.toLowerCase();
		const kindFilter = options.kind?.trim().toLowerCase();
		const limit = Math.max(1, Math.min(50, Math.trunc(options.limit ?? 6)));

		const files = this.walkFiles(
			rootInfo.root,
			rootInfo.rootReal,
			{ pathPrefix: options.pathPrefix, language: options.language },
			options.signal,
		);

		const hits: FileMatchHit[] = [];
		for (const f of files) {
			if (options.signal?.aborted) break;
			const content = readSafeTextFile(f.absPath, rootInfo.rootReal, this.maxFileBytes);
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
				const snippet = lines.slice(lineIdx, Math.min(lines.length, lineIdx + 3)).join("\n").trim().slice(0, 320);
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
		return {
			hits: hits.slice(0, limit),
			total: hits.length,
			tookMs: Date.now() - started,
			engine: this.engine,
			mode: "lexical",
		};
	}

	async getIndexMap(scope: FileSearchScope): Promise<FileIndexMapResult> {
		const rootInfo = this.resolveRoot(scope);
		if (!rootInfo) {
			return { engine: this.engine, indexScope: "(unconfigured)", datasets: [], totalDocs: 0 };
		}
		const files = this.walkFiles(rootInfo.root, rootInfo.rootReal, {});
		const byLang = new Map<string, string[]>();
		for (const f of files) {
			const key = f.language ?? f.format;
			const list = byLang.get(key) ?? [];
			list.push(f.relPath);
			byLang.set(key, list);
		}
		const datasets = Array.from(byLang.entries())
			.sort((a, b) => b[1].length - a[1].length)
			.map(([lang, paths]) => ({
				index: `local:${lang}`,
				dataset: lang,
				docCount: paths.length,
				languages: [lang],
				samplePaths: paths.slice(0, 4),
			}));
		return {
			engine: this.engine,
			indexScope: "local-workspace",
			datasets,
			totalDocs: files.length,
		};
	}

	async healthCheck(): Promise<FileSearchHealth> {
		const ok = this.workspaceRoot ? fs.existsSync(this.workspaceRoot) : true;
		return { ok, engine: this.engine, latencyMs: 0 };
	}
}
