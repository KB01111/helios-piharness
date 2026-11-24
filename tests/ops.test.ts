import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { createHealthChecker, createReadinessHandler, readPackageVersion } from "../src/health.js";
import { createLogger, resolveLogLevel, type LogLevel } from "../src/logger.js";
import {
	FileWriterLock,
	SingleWriterLockError,
	WRITER_LOCK_CORRUPT_STALE_MS,
	WRITER_LOCK_FOREIGN_STALE_MS,
} from "../src/runtime/host-runtime.js";

interface ParsedLine {
	readonly level: LogLevel;
	readonly event: string;
	readonly [key: string]: unknown;
}

function captureLogger(level: LogLevel = "debug") {
	const lines: string[] = [];
	const logger = createLogger({ level, sink: (line) => lines.push(line) });
	const parsed = (): ParsedLine[] => lines.map((line) => JSON.parse(line) as ParsedLine);
	const events = (): string[] => parsed().map((entry) => entry.event);
	return { logger, lines, parsed, events };
}

/** A pid that belonged to a process that has already exited. */
function deadPid(): number {
	const result = spawnSync(process.execPath, ["-e", "0"]);
	if (typeof result.pid !== "number") throw new Error("spawnSync did not report a pid");
	return result.pid;
}

let tmpRoot: string;
let dbPath: string;
let lockPath: string;
const openLocks: FileWriterLock[] = [];
let liveChild: ChildProcess | undefined;

beforeAll(() => {
	liveChild = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { stdio: "ignore" });
});

afterAll(() => {
	liveChild?.kill();
});

beforeEach(() => {
	tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "ops-test-"));
	dbPath = path.join(tmpRoot, "app.sqlite");
	lockPath = `${dbPath}.writer.lock`;
});

afterEach(() => {
	for (const lock of openLocks.splice(0)) lock.release();
	fs.rmSync(tmpRoot, { recursive: true, force: true });
});

function acquire(logger: ReturnType<typeof captureLogger>["logger"] = captureLogger().logger): FileWriterLock {
	const lock = FileWriterLock.acquire(dbPath, { logger });
	openLocks.push(lock);
	return lock;
}

function writeLock(content: string, mtime?: Date): void {
	fs.writeFileSync(lockPath, content, "utf8");
	if (mtime) fs.utimesSync(lockPath, mtime, mtime);
}

function ageMs(ms: number): Date {
	return new Date(Date.now() - ms);
}

describe("FileWriterLock recovery", () => {
	it("reclaims a corrupt lock file once it is older than the corrupt-lock threshold", () => {
		writeLock("{not json", ageMs(WRITER_LOCK_CORRUPT_STALE_MS + 30_000));
		const { logger, events } = captureLogger();

		const lock = acquire(logger);

		expect(events()).toContain("writer_lock_stale_removed");
		const owner = JSON.parse(fs.readFileSync(lockPath, "utf8")) as { pid: number };
		expect(owner.pid).toBe(process.pid);
		lock.release();
		expect(fs.existsSync(lockPath)).toBe(false);
	});

	it("reclaims an empty lock file left behind by a crash", () => {
		writeLock("", ageMs(WRITER_LOCK_CORRUPT_STALE_MS + 1_000));
		const { logger } = captureLogger();

		expect(() => acquire(logger)).not.toThrow();
	});

	it("treats a fresh empty lock file as contended, because its writer may still be starting", () => {
		writeLock("");

		expect(() => acquire()).toThrow(SingleWriterLockError);
		expect(fs.existsSync(lockPath)).toBe(true);
	});

	it("reclaims a lock whose owner pid is no longer running", () => {
		const pid = deadPid();
		writeLock(JSON.stringify({ pid, hostname: os.hostname(), startedAt: 1, acquiredAt: Date.now() }));
		const { logger, parsed } = captureLogger();

		const lock = acquire(logger);

		const warning = parsed().find((entry) => entry.event === "writer_lock_stale_removed");
		expect(warning?.level).toBe("warn");
		expect(warning?.previousPid).toBe(pid);
		lock.release();
	});

	it("reclaims a legacy lock without hostname when its pid is dead", () => {
		writeLock(JSON.stringify({ pid: deadPid(), acquiredAt: Date.now() }));

		expect(() => acquire()).not.toThrow();
	});
});

describe("FileWriterLock contention", () => {
	it("refuses a lock held by a live pid on this host", () => {
		const livePid = liveChild?.pid;
		if (typeof livePid !== "number") throw new Error("helper child process did not start");
		writeLock(JSON.stringify({ pid: livePid, hostname: os.hostname(), acquiredAt: Date.now() }));

		expect(() => acquire()).toThrow(/held by live pid/);
		expect(fs.existsSync(lockPath)).toBe(true);
	});

	it("refuses a lock recorded on another host, even when its pid is not running here", () => {
		writeLock(
			JSON.stringify({
				pid: deadPid(),
				hostname: `${os.hostname()}-elsewhere`,
				acquiredAt: Date.now(),
			}),
		);

		expect(() => acquire()).toThrow(/cannot verify/);
		expect(fs.existsSync(lockPath)).toBe(true);
	});

	it("refuses a second acquisition of the same path inside this process", () => {
		acquire();

		expect(() => FileWriterLock.acquire(dbPath)).toThrow(/already opened by an active writer/);
	});

	it("can be re-acquired after release", () => {
		acquire().release();

		expect(() => acquire()).not.toThrow();
	});
});

describe("FileWriterLock hardening", () => {
	const foreignRecord = (): string =>
		JSON.stringify({ pid: deadPid(), hostname: `${os.hostname()}-elsewhere`, acquiredAt: Date.now() });
	const staleFiles = (): string[] => fs.readdirSync(tmpRoot).filter((name) => name.includes(".stale."));

	it("reclaims a foreign-host lock whose heartbeat is older than the stale limit, loudly", () => {
		writeLock(foreignRecord(), ageMs(WRITER_LOCK_FOREIGN_STALE_MS + 60_000));
		const { logger, parsed } = captureLogger();

		const lock = FileWriterLock.acquire(dbPath, { logger });
		openLocks.push(lock);

		const warning = parsed().find((entry) => entry.event === "writer_lock_foreign_host_reclaimed");
		expect(warning?.level).toBe("warn");
		expect(String(warning?.previousHost)).toContain("-elsewhere");
		expect((JSON.parse(fs.readFileSync(lockPath, "utf8")) as { pid: number }).pid).toBe(process.pid);
		expect(staleFiles()).toEqual([]);
	});

	it("honours a custom foreign stale limit and keeps a fresh foreign lock contended", () => {
		writeLock(foreignRecord(), ageMs(5_000));

		expect(() => FileWriterLock.acquire(dbPath, { foreignStaleMs: 60_000 })).toThrow(/HELIOS_LOCK_STALE_MS/);
		expect(fs.existsSync(lockPath)).toBe(true);

		const lock = FileWriterLock.acquire(dbPath, { foreignStaleMs: 1_000 });
		openLocks.push(lock);
		expect((JSON.parse(fs.readFileSync(lockPath, "utf8")) as { pid: number }).pid).toBe(process.pid);
	});

	it("reclaims a fresh foreign-host lock when force-reclaim is set", () => {
		writeLock(foreignRecord());

		const lock = FileWriterLock.acquire(dbPath, { forceReclaim: true });
		openLocks.push(lock);

		expect((JSON.parse(fs.readFileSync(lockPath, "utf8")) as { pid: number }).pid).toBe(process.pid);
	});

	it("reads the force-reclaim and stale-limit overrides from the environment", () => {
		writeLock(foreignRecord());
		vi.stubEnv("HELIOS_LOCK_FORCE_RECLAIM", "true");
		try {
			const lock = FileWriterLock.acquire(dbPath);
			openLocks.push(lock);
		} finally {
			vi.unstubAllEnvs();
		}
		expect(fs.existsSync(lockPath)).toBe(true);
	});

	it("treats differently spelled paths of the same DB as one writer", () => {
		acquire();

		const dotted = path.join(tmpRoot, "nested", "..", "app.sqlite");
		expect(() => FileWriterLock.acquire(dotted)).toThrow(/already opened by an active writer/);
		if (process.platform === "win32") {
			expect(() => FileWriterLock.acquire(dbPath.toUpperCase())).toThrow(/already opened by an active writer/);
		}
		// The first writer's lock is untouched.
		expect((JSON.parse(fs.readFileSync(lockPath, "utf8")) as { pid: number }).pid).toBe(process.pid);
	});

	it("never reclaims a same-pid lock while its path is an active writer", () => {
		const lock = acquire();
		// Even if the lock file is rewritten (e.g. by a confused sibling) the active registry wins.
		expect(() => FileWriterLock.acquire(dbPath)).toThrow(SingleWriterLockError);
		expect(fs.existsSync(lock.path)).toBe(true);
	});

	it("leaves no .stale files behind after reclaiming a dead owner's lock", () => {
		writeLock(JSON.stringify({ pid: deadPid(), hostname: os.hostname(), acquiredAt: Date.now() }));

		acquire();

		expect(staleFiles()).toEqual([]);
	});
});

describe("FileWriterLock release safety", () => {
	it("removes its own lock file on release", () => {
		const lock = acquire();

		lock.release();

		expect(fs.existsSync(lockPath)).toBe(false);
	});

	it("leaves the lock file alone when it no longer records this process", () => {
		const { logger, events } = captureLogger();
		const lock = acquire(logger);
		const otherOwner = JSON.stringify({ pid: deadPid(), acquiredAt: 1 });
		fs.writeFileSync(lockPath, otherOwner, "utf8");

		lock.release();

		expect(fs.readFileSync(lockPath, "utf8")).toBe(otherOwner);
		expect(events()).toContain("writer_lock_not_owned_on_release");
	});

	it("is idempotent", () => {
		const lock = acquire();
		lock.release();

		expect(() => lock.release()).not.toThrow();
	});
});

describe("createLogger", () => {
	it("drops records below the configured level", () => {
		const { logger, events } = captureLogger("warn");

		logger.debug("d");
		logger.info("i");
		logger.warn("w");
		logger.error("e");

		expect(events()).toEqual(["w", "e"]);
		expect(logger.isEnabled("info")).toBe(false);
		expect(logger.isEnabled("error")).toBe(true);
	});

	it("emits one JSON object per line with time, level and event", () => {
		const { logger, parsed, lines } = captureLogger();

		logger.info("server_started", { port: 4100 });

		expect(lines).toHaveLength(1);
		const [entry] = parsed();
		expect(entry).toMatchObject({ level: "info", event: "server_started", port: 4100 });
		expect(typeof entry?.time).toBe("string");
	});

	it("merges base and child bindings into every record", () => {
		const { lines, parsed } = captureLogger();
		const base = createLogger({ base: { service: "svc" }, sink: (line) => lines.push(line) });

		base.child({ component: "writer-lock" }).warn("evt", { reason: "r" });

		expect(parsed()[0]).toMatchObject({ service: "svc", component: "writer-lock", reason: "r" });
	});

	it("serializes errors as name, message and stack without custom properties", () => {
		const { logger, lines } = captureLogger();
		const error = Object.assign(new Error("db unavailable"), { apiKey: "sk-live-123", code: "EIO" });

		logger.error("failed", { error });

		const raw = lines[0] ?? "";
		expect(raw).not.toContain("sk-live-123");
		const entry = JSON.parse(raw) as { error: { name: string; message: string; stack: string; code: string } };
		expect(entry.error.name).toBe("Error");
		expect(entry.error.message).toBe("db unavailable");
		expect(entry.error.stack).toContain("db unavailable");
		expect(entry.error.code).toBe("EIO");
	});

	it("redacts secret-looking field names", () => {
		const { logger, lines } = captureLogger();

		logger.info("evt", { apiKey: "abc", authToken: "xyz", maxTokens: 512 });

		const entry = JSON.parse(lines[0] ?? "") as Record<string, unknown>;
		expect(entry.apiKey).toBe("[REDACTED]");
		expect(entry.authToken).toBe("[REDACTED]");
		expect(entry.maxTokens).toBe(512);
	});

	it("does not throw on circular fields", () => {
		const { logger, parsed } = captureLogger();
		const circular: Record<string, unknown> = {};
		circular.self = circular;

		expect(() => logger.info("evt", { circular })).not.toThrow();
		expect(parsed()[0]?.event).toBe("evt");
	});

	it("resolveLogLevel falls back for unknown values", () => {
		expect(resolveLogLevel("WARN")).toBe("warn");
		expect(resolveLogLevel("verbose")).toBe("info");
		expect(resolveLogLevel(undefined, "error")).toBe("error");
	});
});

describe("createHealthChecker", () => {
	const okHarness = { getUsage: async () => ({}) as never };

	it("liveness reports the package version and never touches dependencies", () => {
		let pinged = false;
		const checker = createHealthChecker({
			harness: { getUsage: () => Promise.reject(new Error("should not be called")) },
			appDbPing: () => {
				pinged = true;
			},
			piDbPath: ":memory:",
		});

		const report = checker.liveness();

		expect(report.status).toBe("alive");
		expect(report.version).toBe(readPackageVersion());
		expect(pinged).toBe(false);
	});

	it("readiness is ready when harness, app DB and pi DB all answer", async () => {
		const checker = createHealthChecker({
			harness: okHarness,
			appDbPing: () => undefined,
			piDbPath: ":memory:",
			useMockModels: true,
		});

		const report = await checker.readiness();

		expect(report.status).toBe("ready");
		expect(report.mockModels).toBe(true);
		expect(report.checks.harness.ok).toBe(true);
		expect(report.checks.appDb.ok).toBe(true);
		expect(report.checks.piDb.ok).toBe(true);
		expect(report.version).toBe(readPackageVersion());
	});

	it("readiness is not_ready when the harness fails, reporting only the error name", async () => {
		const checker = createHealthChecker({
			harness: {
				getUsage: async () => {
					throw new TypeError("internal detail /secret/path");
				},
			},
			appDbPing: () => undefined,
			piDbPath: ":memory:",
		});

		const report = await checker.readiness();

		expect(report.status).toBe("not_ready");
		expect(report.checks.harness).toMatchObject({ ok: false, error: "TypeError" });
		expect(JSON.stringify(report)).not.toContain("/secret/path");
	});

	it("readiness times out a hung harness", async () => {
		const checker = createHealthChecker({
			harness: { getUsage: () => new Promise<never>(() => undefined) },
			appDbPing: () => undefined,
			piDbPath: ":memory:",
			timeoutMs: 20,
		});

		const report = await checker.readiness();

		expect(report.status).toBe("not_ready");
		expect(report.checks.harness).toMatchObject({ ok: false, error: "HealthCheckTimeoutError" });
	});

	it("readiness fails when the app DB ping throws", async () => {
		const checker = createHealthChecker({
			harness: okHarness,
			appDbPing: () => {
				throw new Error("database is locked");
			},
			piDbPath: ":memory:",
		});

		const report = await checker.readiness();

		expect(report.status).toBe("not_ready");
		expect(report.checks.appDb.ok).toBe(false);
	});

	it("readiness fails when the pi DB file is missing on disk", async () => {
		const checker = createHealthChecker({
			harness: okHarness,
			appDbPing: () => undefined,
			piDbPath: path.join(tmpRoot, "missing.sqlite"),
		});

		const report = await checker.readiness();

		expect(report.status).toBe("not_ready");
		expect(report.checks.piDb).toMatchObject({ ok: false, error: "FileNotFound" });
	});
});

describe("readiness caching and public view", () => {
	it("reuses a fresh result and shares one in-flight check between concurrent callers", async () => {
		let calls = 0;
		let release!: () => void;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		const checker = createHealthChecker({
			harness: {
				getUsage: async () => {
					calls++;
					await gate;
					return {} as never;
				},
			},
			appDbPing: () => undefined,
			piDbPath: ":memory:",
		});

		const concurrent = [checker.readiness(), checker.readiness(), checker.readiness()];
		release();
		const reports = await Promise.all(concurrent);
		expect(calls).toBe(1);
		expect(new Set(reports).size).toBe(1);

		await checker.readiness();
		expect(calls).toBe(1);
	});

	it("recomputes once the cache ttl has elapsed", async () => {
		let clock = 1_000;
		let calls = 0;
		const checker = createHealthChecker({
			harness: {
				getUsage: async () => {
					calls++;
					return {} as never;
				},
			},
			appDbPing: () => undefined,
			piDbPath: ":memory:",
			now: () => clock,
			cacheTtlMs: 2_000,
		});

		await checker.readiness();
		clock += 1_999;
		await checker.readiness();
		expect(calls).toBe(1);
		clock += 2;
		await checker.readiness();
		expect(calls).toBe(2);
	});

	it("gives unauthenticated callers only the verdict and authenticated callers the full report", async () => {
		const health = createHealthChecker({
			harness: { getUsage: async () => ({}) as never },
			appDbPing: () => undefined,
			piDbPath: ":memory:",
			useMockModels: true,
		});
		const auth = (request: Request) =>
			request.headers.get("authorization") === "Bearer good" ? { tenantId: "t", userId: "u", permissions: [] } : null;
		const handle = createReadinessHandler({ health, auth });

		const anonymous = await handle(new Request("http://h/ready"));
		expect(anonymous).toEqual({ httpStatus: 200, body: { status: "ready" } });

		const authed = await handle(new Request("http://h/ready", { headers: { authorization: "Bearer good" } }));
		expect(authed.httpStatus).toBe(200);
		expect(authed.body).toMatchObject({ status: "ready", mockModels: true, version: readPackageVersion() });
		expect((authed.body as { checks?: unknown }).checks).toBeDefined();

		const throwing = createReadinessHandler({
			health,
			auth: () => {
				throw new Error("resolver exploded");
			},
		});
		expect((await throwing(new Request("http://h/ready"))).body).toEqual({ status: "ready" });
	});

	it("keeps the 503 verdict public and hides the failing check details", async () => {
		const health = createHealthChecker({
			harness: {
				getUsage: async () => {
					throw new TypeError("internal /secret/path");
				},
			},
			appDbPing: () => undefined,
			piDbPath: ":memory:",
		});
		const handle = createReadinessHandler({ health, auth: () => null });

		const response = await handle(new Request("http://h/ready"));

		expect(response).toEqual({ httpStatus: 503, body: { status: "not_ready" } });
	});
});
