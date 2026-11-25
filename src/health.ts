import fs from "node:fs";
import { fileURLToPath } from "node:url";
import type { HarnessPort } from "./contracts/harness-port.js";
import type { AuthResolver } from "./transport/auth.js";

export interface HealthCheckerOptions {
	/** Only `getUsage` is used; a successful call proves the harness can serve requests. */
	readonly harness: Pick<HarnessPort, "getUsage">;
	/** Runs `SELECT 1` against the app database. Must throw or reject when the database does not answer. */
	readonly appDbPing: () => unknown | Promise<unknown>;
	/** Pi Durable SQLite path, or ":memory:". */
	readonly piDbPath: string;
	/** Whether the deterministic offline test models are active. Reported, not checked. */
	readonly useMockModels?: boolean;
	/** Per-check timeout. Default: 2000ms. */
	readonly timeoutMs?: number;
	/** Clock for latency measurement and timestamps. Default: Date.now. */
	readonly now?: () => number;
	/**
	 * How long a readiness result is reused (default 2000ms; 0 disables). Concurrent callers share one in-flight
	 * check, so an unauthenticated probe flood cannot multiply database/harness work.
	 */
	readonly cacheTtlMs?: number;
}

export interface HealthCheckResult {
	readonly ok: boolean;
	readonly latencyMs: number;
	/** Error class name only (no messages, paths or payloads), present when `ok` is false. */
	readonly error?: string;
}

export interface LivenessReport {
	readonly status: "alive";
	readonly version: string;
	readonly uptimeSec: number;
	readonly timestamp: number;
}

export interface ReadinessReport {
	readonly status: "ready" | "not_ready";
	readonly version: string;
	readonly uptimeSec: number;
	readonly mockModels: boolean;
	readonly checks: {
		readonly harness: HealthCheckResult;
		readonly appDb: HealthCheckResult;
		readonly piDb: HealthCheckResult;
	};
	readonly timestamp: number;
}

export interface HealthChecker {
	/** Process is up. Never touches dependencies, so it is safe for restart probes. */
	liveness(): LivenessReport;
	/** Harness and both SQLite databases answer within the timeout. Use for traffic routing. */
	readiness(): Promise<ReadinessReport>;
}

const DEFAULT_CHECK_TIMEOUT_MS = 2_000;
const DEFAULT_READINESS_CACHE_TTL_MS = 2_000;

class HealthCheckTimeoutError extends Error {
	constructor(timeoutMs: number) {
		super(`health check timed out after ${timeoutMs}ms`);
		this.name = "HealthCheckTimeoutError";
	}
}

let cachedVersion: string | undefined;

/** Reads the version from package.json next to src/ or dist/. Never throws. */
export function readPackageVersion(): string {
	if (cachedVersion !== undefined) return cachedVersion;
	try {
		const raw = fs.readFileSync(fileURLToPath(new URL("../package.json", import.meta.url)), "utf8");
		const parsed = JSON.parse(raw) as { version?: unknown };
		cachedVersion = typeof parsed.version === "string" ? parsed.version : "unknown";
	} catch {
		cachedVersion = "unknown";
	}
	return cachedVersion;
}

async function runCheck(
	check: () => unknown | Promise<unknown>,
	timeoutMs: number,
	now: () => number,
): Promise<HealthCheckResult> {
	const started = now();
	let timer: NodeJS.Timeout | undefined;
	try {
		await Promise.race([
			Promise.resolve().then(check),
			new Promise<never>((_resolve, reject) => {
				timer = setTimeout(() => reject(new HealthCheckTimeoutError(timeoutMs)), timeoutMs);
			}),
		]);
		return { ok: true, latencyMs: now() - started };
	} catch (error) {
		const name = error instanceof Error ? error.name : "Error";
		return { ok: false, latencyMs: now() - started, error: name };
	} finally {
		if (timer !== undefined) clearTimeout(timer);
	}
}

export function createHealthChecker(options: HealthCheckerOptions): HealthChecker {
	const timeoutMs = options.timeoutMs ?? DEFAULT_CHECK_TIMEOUT_MS;
	const now = options.now ?? (() => Date.now());
	const mockModels = options.useMockModels ?? false;
	const cacheTtlMs = options.cacheTtlMs ?? DEFAULT_READINESS_CACHE_TTL_MS;
	let cached: { readonly at: number; readonly report: ReadinessReport } | undefined;
	let inFlight: Promise<ReadinessReport> | undefined;

	const piDbCheck = (): HealthCheckResult => {
		const started = now();
		if (options.piDbPath === ":memory:" || fs.existsSync(options.piDbPath)) {
			return { ok: true, latencyMs: now() - started };
		}
		return { ok: false, latencyMs: now() - started, error: "FileNotFound" };
	};

	const computeReadiness = async (): Promise<ReadinessReport> => {
		const [harness, appDb] = await Promise.all([
			runCheck(() => options.harness.getUsage(), timeoutMs, now),
			runCheck(() => options.appDbPing(), timeoutMs, now),
		]);
		const piDb = piDbCheck();
		const ready = harness.ok && appDb.ok && piDb.ok;
		return {
			status: ready ? "ready" : "not_ready",
			version: readPackageVersion(),
			uptimeSec: Math.round(process.uptime()),
			mockModels,
			checks: { harness, appDb, piDb },
			timestamp: now(),
		};
	};

	return {
		liveness: () => ({
			status: "alive",
			version: readPackageVersion(),
			uptimeSec: Math.round(process.uptime()),
			timestamp: now(),
		}),
		readiness: () => {
			if (cacheTtlMs > 0 && cached && now() - cached.at < cacheTtlMs) return Promise.resolve(cached.report);
			if (inFlight) return inFlight;
			const run = computeReadiness().then((report) => {
				cached = { at: now(), report };
				return report;
			});
			inFlight = run;
			const clear = (): void => {
				if (inFlight === run) inFlight = undefined;
			};
			run.then(clear, clear);
			return run;
		},
	};
}

/** What an unauthenticated probe is told: the verdict only, nothing about versions, models or dependencies. */
export interface PublicReadinessReport {
	readonly status: ReadinessReport["status"];
}

export interface ReadinessResponse {
	readonly httpStatus: 200 | 503;
	readonly body: ReadinessReport | PublicReadinessReport;
}

/**
 * Builds the `/ready` handler. Everyone gets the verdict (200/503 + `{status}`); only requests that authenticate with
 * `auth` additionally receive version, mock-model flag and per-check details.
 */
export function createReadinessHandler(options: {
	readonly health: Pick<HealthChecker, "readiness">;
	readonly auth: AuthResolver;
}): (request: Request) => Promise<ReadinessResponse> {
	return async (request) => {
		const report = await options.health.readiness();
		const httpStatus = report.status === "ready" ? 200 : 503;
		let authenticated = false;
		try {
			authenticated = (await options.auth(request)) !== null;
		} catch {
			authenticated = false;
		}
		return { httpStatus, body: authenticated ? report : { status: report.status } };
	};
}
