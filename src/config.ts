import path from "node:path";
import type { ModelSpec } from "./contracts/harness-port.js";
import { DEFAULT_RETENTION_INTERVAL_MS, DEFAULT_RETENTION_POLICY } from "./runtime/retention.js";
import { parseMcpServersEnv, type McpServerConfig } from "./mcp/mcp-client.js";

export interface AppConfig {
	readonly port: number;
	readonly host: string;
	readonly logLevel: "debug" | "info" | "warn" | "error";
	readonly dataDir: string;
	readonly piDbPath: string;
	readonly appDbPath: string;
	readonly artifactsDir: string;
	readonly skillsDir: string;
	readonly mainModel: ModelSpec;
	readonly fastModel: ModelSpec;
	readonly useMockModels: boolean;
	readonly maxToolOutputBytes: number;
	/** Interval between scheduled sleep-cycle sweeps. `0` disables the scheduler. */
	readonly sleepCycleIntervalMs: number;
	/** Master switch for the periodic sleep-cycle scheduler (SLEEP_CYCLE_SCHEDULER_ENABLED, default true). */
	readonly sleepCycleSchedulerEnabled: boolean;
	readonly staleSkillDays: number;
	// ── Budgets (0 = unlimited) ──
	/** MAX_CONVERSATION_TOKENS: total tokens one conversation may spend before the next generation is blocked. */
	readonly maxConversationTokens: number;
	/** MAX_CONVERSATION_COST_USD */
	readonly maxConversationCostUsd: number;
	/** MAX_TENANT_DAILY_TOKENS: per-tenant, per UTC day. */
	readonly maxTenantDailyTokens: number;
	/** MAX_TENANT_DAILY_COST_USD */
	readonly maxTenantDailyCostUsd: number;
	/** MAX_FAST_MODEL_DAILY_TOKENS: all background fast-model calls combined, per UTC day. */
	readonly maxFastModelDailyTokens: number;
	/** MAX_FAST_MODEL_DAILY_COST_USD */
	readonly maxFastModelDailyCostUsd: number;
	/** DEFAULT_MAX_TURNS: tool rounds per run when createConversation passes no maxTurns (default 24). */
	readonly defaultMaxTurns: number;
	// ── Model resilience ──
	/** MODEL_TIMEOUT_MS: per-request idle timeout of the main model (Pi Durable `stream.timeoutMs`). Default 120000. */
	readonly modelTimeoutMs: number;
	/** MODEL_MAX_RETRIES: durable generation retries of the main model (Pi Durable `retry.maxRetries`). Default 3. */
	readonly modelMaxRetries: number;
	/** MODEL_RETRY_BASE_DELAY_MS: base of the main model's exponential backoff (`retry.baseDelayMs`). Default 2000. */
	readonly modelRetryBaseDelayMs: number;
	/** MODEL_PROVIDER_MAX_RETRIES: in-attempt SDK retries inside one request (`stream.maxRetries`); `undefined` keeps the SDK default. */
	readonly modelProviderMaxRetries: number | undefined;
	/** FAST_MODEL_TIMEOUT_MS: per-attempt timeout of the fast model. Default 30000. */
	readonly fastModelTimeoutMs: number;
	/** FAST_MODEL_MAX_RETRIES: retries per fast-model candidate on transient errors. Default 2. */
	readonly fastModelMaxRetries: number;
	/** FAST_MODEL_FALLBACKS: ordered `provider:model` list tried after the fast model fails. */
	readonly fastModelFallbacks: readonly ModelSpec[];
	/** CIRCUIT_BREAKER_FAILURES: consecutive failures before a provider's circuit opens. Default 5. */
	readonly circuitBreakerFailureThreshold: number;
	/** CIRCUIT_BREAKER_COOLDOWN_MS: open time before a half-open probe. Default 30000. */
	readonly circuitBreakerCooldownMs: number;
	/** ALLOW_MOCK_MODELS: permit mock models when NODE_ENV=production (see `assertProductionModelSafety`). */
	readonly allowMockModels: boolean;
	// ── Codemode (model-written JS that calls tools inside a QuickJS sandbox) ──
	/** CODEMODE_ENABLED: install the durable `codemode` tool (default true). */
	readonly codemodeEnabled: boolean;
	/** CODEMODE_MEMORY_MB: heap cap per script VM (default 128). */
	readonly codemodeMemoryMb: number;
	/** CODEMODE_TIMEOUT_MS: wall-clock deadline per script, tool time included (default 120000). */
	readonly codemodeTimeoutMs: number;
	/** CODEMODE_MAX_WORKERS: process-wide cap on concurrently running script workers (default 4). */
	readonly codemodeMaxWorkers: number;
	/** CODEMODE_INLINE_BUDGET: estimated tokens of tool declarations in the codemode prompt section (default 3000). */
	readonly codemodeInlineBudget: number;
	// ── Self-improvement ──
	/** IMPROVEMENTS_DIR: records + active.json of the gated prompt-improvement loop (default `$DATA_DIR/improvements`). */
	readonly improvementsDir: string;
	// ── Answer verifier ──
	/** VERIFIER_ENABLED: review answers with the fast model before they stand (default: on with real models, off with mocks). */
	readonly verifierEnabled: boolean;
	/** VERIFIER_MAX_REVISIONS: times one run may be sent back for revision (default 1). */
	readonly verifierMaxRevisions: number;
	/** VERIFIER_MIN_ANSWER_CHARS: answers without tool use are only checked from this length (default 600). */
	readonly verifierMinAnswerChars: number;
	/** TRACE_EXPORT_PATH: append every settled Pi Durable span as one JSON line to this file (unset = off). */
	readonly traceExportPath: string | undefined;
	/** TRACE_EXPORT_MAX_BYTES: rotate the trace file (keeping one `.1`) at this size. Default 50 MiB. */
	readonly traceExportMaxBytes: number;
	/** IMPROVEMENT_PROBE_SAMPLES: eval-suite passes averaged per probe (default 1; use 3+ with real, non-deterministic models). */
	readonly improvementProbeSamples: number;
	/** IMPROVEMENT_TOOL_ERROR_TOLERANCE: allowed rise of the tool-error RATE in the acceptance gate (default 0.05). */
	readonly improvementToolErrorTolerance: number;
	/** EVALS_DIR: eval set root; cases live in `<dir>/cases/*.json`, the CI baseline in `<dir>/baseline.json` (default `./evals`). */
	readonly evalsDir: string;
	// ── Security & Permission Policy ──
	/** SECRET_REDACTION_ENABLED: scan and redact secrets/credentials in tool outputs before artifacts & LLM context (default true). */
	readonly secretRedactionEnabled: boolean;
	/** TOOL_APPROVAL_REQUIRED: comma-separated tool names/globs that require durable human approval (`ask` tier) before execution. */
	readonly toolApprovalRequired: readonly string[];
	/** TOOL_DENIED: comma-separated tool names/globs unconditionally blocked by policy (`deny` tier). */
	readonly toolDenied: readonly string[];
	/** TOOL_APPROVAL_TIMEOUT_MS: approval deadline for `ask`-tier tool gates (default 24h). */
	readonly toolApprovalTimeoutMs: number;
	// ── Subagents & Advisor ──
	/** SUBAGENTS_ENABLED: register the `subagents` tool pack (`subagent_delegate`, `subagent_parallel`, `advisor_consult`). Default true. */
	readonly subagentsEnabled: boolean;
	/** SUBAGENT_MAX_DEPTH: max recursive subagent nesting depth (default 2). */
	readonly subagentMaxDepth: number;
	/** SUBAGENT_MAX_TURNS: max tool rounds per child subagent (default 8). */
	readonly subagentMaxTurns: number;
	/** SUBAGENT_MAX_PARALLEL: max concurrent children per `subagent_parallel` call (default 4). */
	readonly subagentMaxParallel: number;
	/** SUBAGENT_TIMEOUT_MS: deadline per subagent delegation in ms (default 120000). */
	readonly subagentTimeoutMs: number;
	/** SUBAGENT_READONLY_MODEL: optional model override for read-only `researcher`/`reviewer` subagents (defaults to mainModel). */
	readonly subagentReadOnlyModel: ModelSpec | undefined;
	/** ADVISOR_MODEL: optional model for `advisor_consult` second opinions (defaults to fastModel). */
	readonly advisorModel: ModelSpec | undefined;
	// ── MCP & Deferred Tool Discovery ──
	/** TOOL_SEARCH_ENABLED: install Pi Durable's BM25 `tool_search` extension for deferred tools (default true). */
	readonly toolSearchEnabled: boolean;
	/** MCP_SERVERS: JSON array of lazily-connected MCP servers (`[{ name, url, headers?, readOnly?, timeoutMs? }]`). */
	readonly mcpServers: readonly McpServerConfig[];
	// ── Context Pruning & Structured Compaction ──
	/** CONTEXT_PRUNE_ENABLED: prune stale/superseded tool outputs before model requests when context exceeds threshold (default true). */
	readonly contextPruneEnabled: boolean;
	/** CONTEXT_PRUNE_THRESHOLD_CHARS: non-system character count above which pre-request pruning activates (default 32000). */
	readonly contextPruneThresholdChars: number;
	/** CONTEXT_PRUNE_KEEP_RECENT_TOOLS: number of most recent tool results always kept verbatim (default 6). */
	readonly contextPruneKeepRecentTools: number;
	/** STRUCTURED_COMPACTION_ENABLED: preserve active PlanDoc, mode, approvals & artifact IDs during compaction (default true). */
	readonly structuredCompactionEnabled: boolean;
	// ── Data retention (0 = keep forever) ──
	/** RETENTION_ENABLED: periodically prune append-only operational tables (default true). */
	readonly retentionEnabled: boolean;
	/** RETENTION_INTERVAL_MS: sweep interval (default 6h; 0 = sweep once at startup only). */
	readonly retentionIntervalMs: number;
	/** RETENTION_PERMISSION_AUDIT_DAYS (default 365). */
	readonly retentionPermissionAuditDays: number;
	/** RETENTION_TOOL_INVOCATION_DAYS (default 90). */
	readonly retentionToolInvocationDays: number;
	/** RETENTION_RUN_TELEMETRY_DAYS (default 90). */
	readonly retentionRunTelemetryDays: number;
	/** RETENTION_BUDGET_EVENT_DAYS (default 30). */
	readonly retentionBudgetEventDays: number;
	/** RETENTION_BUDGET_DAILY_DAYS: per-tenant daily spend ledger history (default 400). */
	readonly retentionBudgetDailyDays: number;
	/** RETENTION_ARTIFACT_DAYS: offloaded tool-output artifacts incl. files; opt-in because transcripts hold recall handles (default 0). */
	readonly retentionArtifactDays: number;
}

export function parseModelSpec(raw: string, fallback: ModelSpec): ModelSpec {
	const trimmed = raw.trim();
	if (!trimmed) return fallback;
	const separator = trimmed.includes(":") ? ":" : trimmed.includes("/") ? "/" : null;
	if (!separator) {
		return { provider: fallback.provider, modelId: trimmed };
	}
	const idx = trimmed.indexOf(separator);
	const provider = trimmed.slice(0, idx).trim();
	const modelId = trimmed.slice(idx + 1).trim();
	if (!provider || !modelId) {
		throw new Error(`Invalid model spec "${raw}". Expected "provider:modelId" or "provider/modelId".`);
	}
	return { provider, modelId };
}

function parsePositiveInt(raw: string | undefined, fallback: number): number {
	if (raw === undefined || raw.trim() === "") return fallback;
	const parsed = Number.parseInt(raw, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseNonNegativeInt(raw: string | undefined, fallback: number): number {
	if (raw === undefined || raw.trim() === "") return fallback;
	const parsed = Number.parseInt(raw, 10);
	return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function parseBoolean(raw: string | undefined, fallback: boolean): boolean {
	if (raw === undefined || raw.trim() === "") return fallback;
	const normalized = raw.trim().toLowerCase();
	if (normalized === "true" || normalized === "1" || normalized === "yes") return true;
	if (normalized === "false" || normalized === "0" || normalized === "no") return false;
	return fallback;
}

export function loadConfig(env: Readonly<Record<string, string | undefined>> = process.env): AppConfig {
	const dataDir = path.resolve(env.DATA_DIR ?? "./data");
	const useMockModels = parseBoolean(
		env.USE_MOCK_MODELS,
		!env.ANTHROPIC_API_KEY && !env.OPENAI_API_KEY && !env.GEMINI_API_KEY && !env.OPENROUTER_API_KEY,
	);

	const defaultMain: ModelSpec = useMockModels
		? { provider: "test", modelId: "main" }
		: { provider: "anthropic", modelId: "claude-sonnet-4-5" };
	const defaultFast: ModelSpec = useMockModels
		? { provider: "test", modelId: "fast" }
		: { provider: "anthropic", modelId: "claude-haiku-4-5" };

	const logLevelRaw = (env.LOG_LEVEL ?? "info").toLowerCase();
	const logLevel: AppConfig["logLevel"] =
		logLevelRaw === "debug" || logLevelRaw === "info" || logLevelRaw === "warn" || logLevelRaw === "error"
			? logLevelRaw
			: "info";

	return {
		port: parsePositiveInt(env.PORT, 4100),
		host: env.HOST ?? "127.0.0.1",
		logLevel,
		dataDir,
		piDbPath: env.PI_DB_PATH === ":memory:" ? ":memory:" : path.resolve(env.PI_DB_PATH ?? path.join(dataDir, "pi-durable.sqlite")),
		appDbPath: env.APP_DB_PATH === ":memory:" ? ":memory:" : path.resolve(env.APP_DB_PATH ?? path.join(dataDir, "app-memory.sqlite")),
		artifactsDir: path.resolve(env.ARTIFACTS_DIR ?? path.join(dataDir, "artifacts")),
		skillsDir: path.resolve(env.SKILLS_DIR ?? "./skills"),
		mainModel: useMockModels && !env.MAIN_MODEL ? defaultMain : parseModelSpec(env.MAIN_MODEL ?? "", defaultMain),
		fastModel: useMockModels && !env.FAST_MODEL ? defaultFast : parseModelSpec(env.FAST_MODEL ?? "", defaultFast),
		useMockModels,
		maxToolOutputBytes: parsePositiveInt(env.MAX_TOOL_OUTPUT_BYTES, 6144),
		sleepCycleIntervalMs: parseNonNegativeInt(env.SLEEP_CYCLE_CRON_INTERVAL_MS, 300_000),
		sleepCycleSchedulerEnabled: parseBoolean(env.SLEEP_CYCLE_SCHEDULER_ENABLED, true),
		staleSkillDays: parsePositiveInt(env.STALE_SKILL_DAYS, 30),
		maxConversationTokens: parseBudgetLimit("MAX_CONVERSATION_TOKENS", env.MAX_CONVERSATION_TOKENS, true),
		maxConversationCostUsd: parseBudgetLimit("MAX_CONVERSATION_COST_USD", env.MAX_CONVERSATION_COST_USD, false),
		maxTenantDailyTokens: parseBudgetLimit("MAX_TENANT_DAILY_TOKENS", env.MAX_TENANT_DAILY_TOKENS, true),
		maxTenantDailyCostUsd: parseBudgetLimit("MAX_TENANT_DAILY_COST_USD", env.MAX_TENANT_DAILY_COST_USD, false),
		maxFastModelDailyTokens: parseBudgetLimit("MAX_FAST_MODEL_DAILY_TOKENS", env.MAX_FAST_MODEL_DAILY_TOKENS, true),
		maxFastModelDailyCostUsd: parseBudgetLimit(
			"MAX_FAST_MODEL_DAILY_COST_USD",
			env.MAX_FAST_MODEL_DAILY_COST_USD,
			false,
		),
		defaultMaxTurns: parsePositiveInt(env.DEFAULT_MAX_TURNS, 24),
		modelTimeoutMs: parseNonNegativeInt(env.MODEL_TIMEOUT_MS, 120_000),
		modelMaxRetries: parseNonNegativeInt(env.MODEL_MAX_RETRIES, 3),
		modelRetryBaseDelayMs: parsePositiveInt(env.MODEL_RETRY_BASE_DELAY_MS, 2_000),
		modelProviderMaxRetries:
			env.MODEL_PROVIDER_MAX_RETRIES === undefined || env.MODEL_PROVIDER_MAX_RETRIES.trim() === ""
				? undefined
				: parseNonNegativeInt(env.MODEL_PROVIDER_MAX_RETRIES, 0),
		fastModelTimeoutMs: parseNonNegativeInt(env.FAST_MODEL_TIMEOUT_MS, 30_000),
		fastModelMaxRetries: parseNonNegativeInt(env.FAST_MODEL_MAX_RETRIES, 2),
		fastModelFallbacks: parseModelSpecList(env.FAST_MODEL_FALLBACKS),
		circuitBreakerFailureThreshold: parsePositiveInt(env.CIRCUIT_BREAKER_FAILURES, 5),
		circuitBreakerCooldownMs: parsePositiveInt(env.CIRCUIT_BREAKER_COOLDOWN_MS, 30_000),
		allowMockModels: parseBoolean(env.ALLOW_MOCK_MODELS, false),
		codemodeEnabled: parseBoolean(env.CODEMODE_ENABLED, true),
		codemodeMemoryMb: parsePositiveInt(env.CODEMODE_MEMORY_MB, 128),
		codemodeTimeoutMs: parsePositiveInt(env.CODEMODE_TIMEOUT_MS, 120_000),
		codemodeMaxWorkers: parsePositiveInt(env.CODEMODE_MAX_WORKERS, 4),
		codemodeInlineBudget: parseNonNegativeInt(env.CODEMODE_INLINE_BUDGET, 3_000),
		improvementsDir: path.resolve(env.IMPROVEMENTS_DIR ?? path.join(dataDir, "improvements")),
		verifierEnabled: parseBoolean(env.VERIFIER_ENABLED, !useMockModels),
		verifierMaxRevisions: parseNonNegativeInt(env.VERIFIER_MAX_REVISIONS, 1),
		verifierMinAnswerChars: parsePositiveInt(env.VERIFIER_MIN_ANSWER_CHARS, 600),
		traceExportPath: env.TRACE_EXPORT_PATH?.trim() ? path.resolve(env.TRACE_EXPORT_PATH) : undefined,
		traceExportMaxBytes: parsePositiveInt(env.TRACE_EXPORT_MAX_BYTES, 50 * 1024 * 1024),
		improvementProbeSamples: parsePositiveInt(env.IMPROVEMENT_PROBE_SAMPLES, 1),
		improvementToolErrorTolerance: parseNonNegativeNumber(env.IMPROVEMENT_TOOL_ERROR_TOLERANCE, 0.05),
		evalsDir: path.resolve(env.EVALS_DIR ?? "./evals"),
		secretRedactionEnabled: parseBoolean(env.SECRET_REDACTION_ENABLED, true),
		toolApprovalRequired: parseStringList(env.TOOL_APPROVAL_REQUIRED),
		toolDenied: parseStringList(env.TOOL_DENIED),
		toolApprovalTimeoutMs: parsePositiveInt(env.TOOL_APPROVAL_TIMEOUT_MS, 24 * 60 * 60 * 1000),
		subagentsEnabled: parseBoolean(env.SUBAGENTS_ENABLED, true),
		subagentMaxDepth: parsePositiveInt(env.SUBAGENT_MAX_DEPTH, 2),
		subagentMaxTurns: parsePositiveInt(env.SUBAGENT_MAX_TURNS, 8),
		subagentMaxParallel: parsePositiveInt(env.SUBAGENT_MAX_PARALLEL, 4),
		subagentTimeoutMs: parsePositiveInt(env.SUBAGENT_TIMEOUT_MS, 120_000),
		subagentReadOnlyModel: env.SUBAGENT_READONLY_MODEL?.trim()
			? parseModelSpec(env.SUBAGENT_READONLY_MODEL, defaultMain)
			: undefined,
		advisorModel: env.ADVISOR_MODEL?.trim() ? parseModelSpec(env.ADVISOR_MODEL, defaultFast) : undefined,
		toolSearchEnabled: parseBoolean(env.TOOL_SEARCH_ENABLED, true),
		mcpServers: parseMcpServersEnv(env.MCP_SERVERS),
		contextPruneEnabled: parseBoolean(env.CONTEXT_PRUNE_ENABLED, true),
		contextPruneThresholdChars: parsePositiveInt(env.CONTEXT_PRUNE_THRESHOLD_CHARS, 32_000),
		contextPruneKeepRecentTools: parsePositiveInt(env.CONTEXT_PRUNE_KEEP_RECENT_TOOLS, 6),
		structuredCompactionEnabled: parseBoolean(env.STRUCTURED_COMPACTION_ENABLED, true),
		retentionEnabled: parseBoolean(env.RETENTION_ENABLED, true),
		retentionIntervalMs: parseNonNegativeInt(env.RETENTION_INTERVAL_MS, DEFAULT_RETENTION_INTERVAL_MS),
		retentionPermissionAuditDays: parseNonNegativeInt(env.RETENTION_PERMISSION_AUDIT_DAYS, DEFAULT_RETENTION_POLICY.permissionAuditDays),
		retentionToolInvocationDays: parseNonNegativeInt(env.RETENTION_TOOL_INVOCATION_DAYS, DEFAULT_RETENTION_POLICY.toolInvocationDays),
		retentionRunTelemetryDays: parseNonNegativeInt(env.RETENTION_RUN_TELEMETRY_DAYS, DEFAULT_RETENTION_POLICY.runTelemetryDays),
		retentionBudgetEventDays: parseNonNegativeInt(env.RETENTION_BUDGET_EVENT_DAYS, DEFAULT_RETENTION_POLICY.budgetEventDays),
		retentionBudgetDailyDays: parseNonNegativeInt(env.RETENTION_BUDGET_DAILY_DAYS, DEFAULT_RETENTION_POLICY.budgetDailyDays),
		retentionArtifactDays: parseNonNegativeInt(env.RETENTION_ARTIFACT_DAYS, DEFAULT_RETENTION_POLICY.artifactDays),
	};
}

function parseStringList(raw: string | undefined): string[] {
	if (raw === undefined || raw.trim() === "") return [];
	return raw
		.split(",")
		.map((item) => item.trim())
		.filter((item) => item.length > 0);
}

/**
 * Budget limits are a safety control, so a typo must not silently mean "unlimited". Blank/unset stays 0
 * (unlimited) without noise; any other unparseable, negative, or (for token limits) non-integer value still
 * resolves to 0 but logs a loud warning that the limit is NOT being enforced.
 */
function parseBudgetLimit(name: string, raw: string | undefined, integer: boolean): number {
	if (raw === undefined || raw.trim() === "") return 0;
	const trimmed = raw.trim();
	const value = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed) ? Number(trimmed) : Number.NaN;
	if (Number.isFinite(value) && value >= 0 && (!integer || Number.isInteger(value))) return value;
	process.stderr.write(
		`${JSON.stringify({
			level: "warn",
			event: "invalid_budget_limit",
			message: `${name}=${JSON.stringify(raw.slice(0, 64))} is not a valid ${integer ? "non-negative integer" : "non-negative number"}; the limit is treated as 0 (UNLIMITED) and is NOT enforced. Fix the value to enable it.`,
			variable: name,
		})}\n`,
	);
	return 0;
}

function parseNonNegativeNumber(raw: string | undefined, fallback: number): number {
	if (raw === undefined || raw.trim() === "") return fallback;
	const parsed = Number.parseFloat(raw);
	return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

/** Comma-separated `provider:modelId` list (e.g. FAST_MODEL_FALLBACKS); blank entries are skipped. */
export function parseModelSpecList(raw: string | undefined): ModelSpec[] {
	if (raw === undefined || raw.trim() === "") return [];
	const placeholder: ModelSpec = { provider: "", modelId: "" };
	return raw
		.split(",")
		.map((part) => part.trim())
		.filter((part) => part.length > 0)
		.map((part) => {
			const spec = parseModelSpec(part, placeholder);
			if (!spec.provider) {
				throw new Error(`Invalid model spec "${part}" in list. Expected "provider:modelId".`);
			}
			return spec;
		});
}
