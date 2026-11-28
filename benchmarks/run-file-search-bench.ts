/**
 * File-finding & code-search benchmark over the Helios-PIHarness codebase (`src/`).
 *
 * Compares:
 *   1. Naive whole-file retrieval baseline (what an agent pays when grepping + reading matched files)
 *   2. Helios `LocalWorkspaceFileSearchService` (definition-first lexical + compact path:line passages)
 *   3. Live `XerjFileSearchService` when `XERJ_URL` is configured (e.g. `XERJ_URL=http://127.0.0.1:9200 XERJ_SCOPE_MODE=shared npm run bench:files`)
 *
 * Measures Recall@1, Recall@5, MRR, search latency, and response payload size (bytes & estimated tokens)
 * to verify context efficiency against Helios's 6KB tool-output offload cap.
 */
import fs from "node:fs";
import path from "node:path";
import {
	formatFileFindOutput,
	formatFileSearchOutput,
	formatFileSymbolDefOutput,
	LocalWorkspaceFileSearchService,
	XerjFileSearchService,
	type FileSearchScope,
	type FileSearchService,
} from "../src/index.js";

interface FileBenchCase {
	readonly id: string;
	readonly kind: "find" | "search" | "def";
	readonly query: string;
	readonly expectedPath: string;
	readonly symbolKind?: string;
	readonly extension?: string;
}

const CASES: readonly FileBenchCase[] = [
	{ id: "find-budget", kind: "find", query: "budget", extension: "ts", expectedPath: "src/runtime/budget.ts" },
	{ id: "find-mcp", kind: "find", query: "mcp-client", expectedPath: "src/mcp/mcp-client.ts" },
	{ id: "find-redactor", kind: "find", query: "redactor", expectedPath: "src/security/secret-redactor.ts" },
	{ id: "find-xerj", kind: "find", query: "xerj-client", expectedPath: "src/search/xerj-client.ts" },
	{ id: "def-budget-gate", kind: "def", query: "BudgetGate", symbolKind: "class", expectedPath: "src/runtime/budget.ts" },
	{ id: "def-bootstrap", kind: "def", query: "bootstrapHarness", symbolKind: "function", expectedPath: "src/index.ts" },
	{ id: "def-redact", kind: "def", query: "redactSecrets", symbolKind: "function", expectedPath: "src/security/secret-redactor.ts" },
	{ id: "def-narrow", kind: "def", query: "narrowSubagentPermissions", symbolKind: "function", expectedPath: "src/tools/subagent.tools.ts" },
	{ id: "def-xerj-svc", kind: "def", query: "XerjFileSearchService", symbolKind: "class", expectedPath: "src/search/xerj-client.ts" },
	{ id: "search-guard", kind: "search", query: "createGuardShapingExtension", expectedPath: "src/extensions/guard-shaping.ext.ts" },
	{ id: "search-pruner", kind: "search", query: "pruneMessagesForContext", expectedPath: "src/extensions/context-pruner.ext.ts" },
	{ id: "search-lock", kind: "search", query: "SingleWriterLockError", expectedPath: "src/runtime/host-runtime.ts" },
	{ id: "search-ppr", kind: "search", query: "runEntityPpr", expectedPath: "src/memory/graph.ts" },
	{ id: "search-temporal", kind: "search", query: "detectHistoricalIntent", expectedPath: "src/memory/temporal.ts" },
];

const SCOPE: FileSearchScope = { tenantId: "bench", userId: "bench" };
const ROOT = path.resolve(".");

async function evaluateEngine(label: string, service: FileSearchService): Promise<void> {
	let r1 = 0;
	let r5 = 0;
	let mrrSum = 0;
	let totalLatencyMs = 0;
	let totalPassageBytes = 0;
	let totalWholeFileBytes = 0;
	let underCapCount = 0;

	for (const c of CASES) {
		const started = performance.now();
		const res =
			c.kind === "find"
				? await service.findFiles(SCOPE, { query: c.query, ...(c.extension ? { extension: c.extension } : {}), limit: 5 })
				: c.kind === "def"
					? await service.findSymbolDefinitions(SCOPE, {
							symbol: c.query,
							...(c.symbolKind ? { kind: c.symbolKind } : {}),
							limit: 5,
						})
					: await service.searchContent(SCOPE, { query: c.query, limit: 5 });
		totalLatencyMs += performance.now() - started;

		const paths = res.hits.map((h) => h.path.replace(/\\+/g, "/"));
		const idx = paths.findIndex((p) => p === c.expectedPath || p.endsWith(`/${c.expectedPath}`));
		if (idx === 0) r1 += 1;
		if (idx >= 0 && idx < 5) {
			r5 += 1;
			mrrSum += 1 / (idx + 1);
		}

		// Measure the EXACT full rendered tool response UTF-8 bytes (preamble, delimiters, headers, symbols, snippets)
		// against Helios's 6144-byte artifact-offload threshold.
		const renderedToolText =
			c.kind === "find"
				? formatFileFindOutput(res)
				: c.kind === "def"
					? formatFileSymbolDefOutput(c.query, res)
					: formatFileSearchOutput(c.query, res);
		const payloadBytes = Buffer.byteLength(renderedToolText, "utf8");
		totalPassageBytes += payloadBytes;
		if (payloadBytes <= 6144) underCapCount += 1;

		let wholeBytes = 0;
		for (const p of paths.slice(0, 3)) {
			const abs = path.resolve(ROOT, p);
			try {
				wholeBytes += fs.statSync(abs).size;
			} catch {
				// ignore non-local paths
			}
		}
		totalWholeFileBytes += wholeBytes;
	}

	const n = CASES.length;
	const avgMs = totalLatencyMs / n;
	const reduction = totalPassageBytes > 0 ? totalWholeFileBytes / totalPassageBytes : 0;

	console.log(
		[
			`[${label}]`,
			`R@1=${((r1 / n) * 100).toFixed(1)}%`,
			`R@5=${((r5 / n) * 100).toFixed(1)}%`,
			`MRR=${(mrrSum / n).toFixed(3)}`,
			`avgLatency=${avgMs.toFixed(1)}ms`,
			`passageBytes=${totalPassageBytes} vs top3WholeFiles=${totalWholeFileBytes} (${reduction.toFixed(1)}x smaller)`,
			`under6KB=${underCapCount}/${n}`,
		].join(" | "),
	);
}

async function main(): Promise<void> {
	console.log(`Helios File Search Benchmark (${CASES.length} queries over ${ROOT})`);
	const local = new LocalWorkspaceFileSearchService({ workspaceRoot: ROOT });
	await evaluateEngine("local-workspace", local);

	const xerjUrl = process.env.XERJ_URL?.trim();
	if (xerjUrl) {
		const xerj = new XerjFileSearchService({
			baseUrl: xerjUrl,
			...(process.env.XERJ_API_KEY?.trim() ? { apiKey: process.env.XERJ_API_KEY.trim() } : {}),
			indexPrefix: process.env.XERJ_INDEX_PREFIX?.trim() || "ax",
			scopeMode: (process.env.XERJ_SCOPE_MODE?.trim() as "tenant" | "workspace" | "shared") || "shared",
		});
		const health = await xerj.healthCheck();
		if (health.ok) {
			await evaluateEngine(`xerj (${health.version ?? "live"})`, xerj);
		} else {
			console.log(`[xerj] skipped: ${xerjUrl} unreachable (${health.error ?? "unknown"})`);
		}
	} else {
		console.log("[xerj] set XERJ_URL=http://127.0.0.1:9200 (after `xerj autoindex .`) to benchmark live XERJ.");
	}
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
