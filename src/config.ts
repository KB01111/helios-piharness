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
	readonly sleepCycleIntervalMs: number;
	readonly staleSkillDays: number;
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
		sleepCycleIntervalMs: parsePositiveInt(env.SLEEP_CYCLE_CRON_INTERVAL_MS, 300_000),
		staleSkillDays: parsePositiveInt(env.STALE_SKILL_DAYS, 30),
	};
}
