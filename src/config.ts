import path from "node:path";
import type { ModelSpec } from "./contracts/harness-port.js";

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
	/** EVALS_DIR: eval set root; cases live in `<dir>/cases/*.json`, the CI baseline in `<dir>/baseline.json` (default `./evals`). */
	readonly evalsDir: string;
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
		!env.ANTHROPIC_API_KEY && !env.OPENAI_API_KEY && !env.GEMINI_API_KEY,
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
		evalsDir: path.resolve(env.EVALS_DIR ?? "./evals"),
	};
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
