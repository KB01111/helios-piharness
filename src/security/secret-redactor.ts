/**
 * Deterministic, ReDoS-safe secret and credential redactor for tool outputs and artifacts.
 *
 * Inspired by `filter-output` and `pi-hermes-memory` secret scanning:
 * - Bounded regexes only (no nested/catastrophic quantifiers).
 * - Scans and redacts BEFORE oversized output is persisted to the artifact store
 *   and BEFORE tool text reaches the LLM context.
 * - Never returns raw matched secrets in diagnostics — only counts and category names.
 */

export interface SecretPattern {
	readonly category: string;
	readonly pattern: RegExp;
	readonly replacement?: string;
}

export interface SecretRedactionOptions {
	/** Additional caller-supplied patterns evaluated after the built-in set. */
	readonly extraPatterns?: readonly SecretPattern[];
	/** Maximum UTF-16 length scanned per pass (very large blobs are scanned in bounded chunks). Default 512 KiB. */
	readonly maxChunkChars?: number;
}

export interface SecretRedactionResult {
	readonly text: string;
	readonly redactedCount: number;
	readonly categories: readonly string[];
}

const DEFAULT_MAX_CHUNK_CHARS = 512 * 1024;

/**
 * Built-in credential patterns using bounded quantifiers only.
 * Ordered from most specific (PEM blocks, prefixed provider tokens) to generic assignments.
 */
export const DEFAULT_SECRET_PATTERNS: readonly SecretPattern[] = [
	{
		category: "private_key",
		pattern: /-----BEGIN [A-Z0-9 ]{0,32}PRIVATE KEY-----[\s\S]{0,8192}?-----END [A-Z0-9 ]{0,32}PRIVATE KEY-----/g,
		replacement: "[REDACTED:private_key]",
	},
	{
		category: "anthropic_key",
		pattern: /\bsk-ant-[A-Za-z0-9_-]{20,200}\b/g,
		replacement: "[REDACTED:anthropic_key]",
	},
	{
		category: "openai_key",
		pattern: /\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,200}\b/g,
		replacement: "[REDACTED:openai_key]",
	},
	{
		category: "github_token",
		pattern: /\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,100}|github_pat_[A-Za-z0-9_]{20,120})\b/g,
		replacement: "[REDACTED:github_token]",
	},
	{
		category: "aws_access_key",
		pattern: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g,
		replacement: "[REDACTED:aws_access_key]",
	},
	{
		category: "google_api_key",
		pattern: /\bAIza[0-9A-Za-z_-]{35}\b/g,
		replacement: "[REDACTED:google_api_key]",
	},
	{
		category: "slack_token",
		pattern: /\bxox[abprs]-[0-9A-Za-z-]{10,120}\b/g,
		replacement: "[REDACTED:slack_token]",
	},
	{
		category: "stripe_key",
		pattern: /\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,120}\b/g,
		replacement: "[REDACTED:stripe_key]",
	},
	{
		category: "gitlab_token",
		pattern: /\bglpat-[A-Za-z0-9_-]{20,100}\b/g,
		replacement: "[REDACTED:gitlab_token]",
	},
	{
		category: "npm_token",
		pattern: /\bnpm_[A-Za-z0-9]{36}\b/g,
		replacement: "[REDACTED:npm_token]",
	},
	{
		category: "google_oauth_token",
		pattern: /\bya29\.[0-9A-Za-z_-]{20,512}/g,
		replacement: "[REDACTED:google_oauth_token]",
	},
	{
		category: "sendgrid_key",
		pattern: /\bSG\.[A-Za-z0-9_-]{16,64}\.[A-Za-z0-9_-]{16,64}\b/g,
		replacement: "[REDACTED:sendgrid_key]",
	},
	{
		category: "jwt",
		pattern: /\beyJ[A-Za-z0-9_-]{10,512}\.[A-Za-z0-9_-]{10,1024}\.[A-Za-z0-9_-]{10,512}\b/g,
		replacement: "[REDACTED:jwt]",
	},
	{
		category: "auth_header",
		pattern: /\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{16,512}\b/gi,
		replacement: "$1 [REDACTED:auth_header]",
	},
	{
		category: "db_credentials",
		pattern: /\b(postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|rediss|amqp):\/\/([^/\s:@]{1,64}):([^/\s@]{1,128})@/gi,
		replacement: "$1://$2:[REDACTED:db_credentials]@",
	},
	{
		// Quoted values may contain spaces (passphrases): match up to the closing quote of the same kind.
		category: "secret_assignment",
		pattern:
			/\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|private[_-]?token|refresh[_-]?token|token|secret|password|passphrase|passwd)(["']?)\s*([:=])\s*(["'])((?:(?!\4)[^\n]){8,4096})\4/gi,
		replacement: "$1$2$3$4[REDACTED:secret_assignment]$4",
	},
	{
		category: "secret_assignment",
		pattern:
			// The key may itself be quoted (JSON: `"password": "..."`), hence the optional quote before the separator.
			/\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|private[_-]?token|refresh[_-]?token|token|secret|password|passwd)(["']?)\s*([:=])\s*(["']?)([^\s"'`,;}{]{8,4096})\4/gi,
		replacement: "$1$2$3$4[REDACTED:secret_assignment]$4",
	},
];

function ensureGlobal(pattern: RegExp): RegExp {
	const flags = pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`;
	return new RegExp(pattern.source, flags);
}

function redactChunk(
	chunk: string,
	patterns: readonly SecretPattern[],
	categories: Set<string>,
): { text: string; count: number } {
	let text = chunk;
	let count = 0;

	for (const rule of patterns) {
		const re = ensureGlobal(rule.pattern);
		const replacement = rule.replacement ?? `[REDACTED:${rule.category}]`;
		let matchedThisRule = false;
		text = text.replace(re, (...args) => {
			// Skip already-redacted placeholders
			const full = String(args[0] ?? "");
			if (full.includes("[REDACTED:")) return full;
			matchedThisRule = true;
			count += 1;
			// Support $1/$2/$3 group references in replacement templates
			const groups = args.slice(1, -2);
			return replacement.replace(/\$(\d)/g, (_, idx: string) => {
				const value = groups[Number.parseInt(idx, 10) - 1];
				return typeof value === "string" ? value : "";
			});
		});
		if (matchedThisRule) {
			categories.add(rule.category);
		}
	}

	return { text, count };
}

/**
 * Redacts credential-shaped values from `input` while keeping surrounding structure intact.
 * Deterministic and safe on arbitrary tool output sizes.
 */
export function redactSecrets(input: string, options?: SecretRedactionOptions): SecretRedactionResult {
	if (!input) {
		return { text: input, redactedCount: 0, categories: [] };
	}

	const patterns = options?.extraPatterns?.length
		? [...DEFAULT_SECRET_PATTERNS, ...options.extraPatterns]
		: DEFAULT_SECRET_PATTERNS;
	const maxChunk = Math.max(4096, options?.maxChunkChars ?? DEFAULT_MAX_CHUNK_CHARS);
	const categories = new Set<string>();

	if (input.length <= maxChunk) {
		const out = redactChunk(input, patterns, categories);
		return {
			text: out.text,
			redactedCount: out.count,
			categories: Array.from(categories).sort(),
		};
	}

	// Process oversized blobs in overlapping-safe line-aligned slices
	const parts: string[] = [];
	let totalCount = 0;
	let offset = 0;
	while (offset < input.length) {
		let end = Math.min(input.length, offset + maxChunk);
		if (end < input.length) {
			const lastNewline = input.lastIndexOf("\n", end);
			if (lastNewline > offset + Math.floor(maxChunk / 2)) {
				end = lastNewline + 1;
			}
		}
		const res = redactChunk(input.slice(offset, end), patterns, categories);
		parts.push(res.text);
		totalCount += res.count;
		offset = end;
	}

	return {
		text: parts.join(""),
		redactedCount: totalCount,
		categories: Array.from(categories).sort(),
	};
}
