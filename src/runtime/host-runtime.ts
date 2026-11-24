import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { BACKGROUND_CONTEXT, type Context } from "@earendil-works/chord/context";
import type { Models } from "@earendil-works/pi-ai/models";
import {
	Harness,
	MemoryStorage,
	type HarnessOptions,
	type HarnessSettings,
	type RegistryReader,
	type Report,
	type SetTimer,
	type Storage,
	type Wake,
} from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";
import { createLogger, resolveLogLevel, serializeError, type Logger } from "../logger.js";
import { NodeWakeScheduler, type WakeScheduler } from "./wake-scheduler.js";

export class SingleWriterLockError extends Error {
	readonly dbPath: string;

	constructor(dbPath: string, detail: string) {
		super(`Single-writer lock failed for SQLite DB "${dbPath}": ${detail}`);
		this.name = "SingleWriterLockError";
		this.dbPath = dbPath;
	}
}

/**
 * A lock file that cannot be parsed (empty, truncated, or garbage) is only reclaimed
 * once its mtime is this old. Younger files may belong to a writer that is mid-create.
 */
export const WRITER_LOCK_CORRUPT_STALE_MS = 30_000;

const MAX_ACQUIRE_ATTEMPTS = 5;

/** Process-wide active DB path registry preventing two HostRuntime writers on one SQLite file. */
const ACTIVE_WRITER_PATHS = new Set<string>();

interface WriterLockRecord {
	readonly pid: number;
	/** Absent in lock files written by older versions. */
	readonly hostname?: string;
	/** Start time of the owning process (epoch ms). */
	readonly startedAt?: number;
	/** Time the lock was taken (epoch ms). Used with pid to recognise our own lock on release. */
	readonly acquiredAt: number;
}

type LockInspection =
	| { readonly kind: "missing" }
	| { readonly kind: "contended"; readonly reason: string }
	| {
			readonly kind: "stale";
			readonly reason: string;
			readonly raw: string;
			readonly previousPid?: number;
			/** Set when the lock was written on another host and reclaimed on age/force rather than by a pid check. */
			readonly previousHost?: string;
	  };

/**
 * A lock recorded by another host cannot be pid-checked, so it is only reclaimed once its heartbeat (file mtime,
 * refreshed by the holder) is this old. Override with `HELIOS_LOCK_STALE_MS`.
 */
export const WRITER_LOCK_FOREIGN_STALE_MS = 10 * 60_000;

const MAX_HEARTBEAT_INTERVAL_MS = 60_000;
const MIN_HEARTBEAT_INTERVAL_MS = 1_000;

function isProcessAlive(pid: number): boolean {
	if (!Number.isInteger(pid) || pid <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		const code = (error as NodeJS.ErrnoException | undefined)?.code;
		return code === "EPERM";
	}
}

function errorCode(error: unknown): string | undefined {
	const code = (error as NodeJS.ErrnoException | undefined)?.code;
	return typeof code === "string" ? code : undefined;
}

function parseWriterLockRecord(raw: string): WriterLockRecord | undefined {
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch {
		return undefined;
	}
	if (!parsed || typeof parsed !== "object") return undefined;
	const candidate = parsed as Partial<Record<keyof WriterLockRecord, unknown>>;
	if (typeof candidate.pid !== "number" || !Number.isInteger(candidate.pid) || candidate.pid <= 0) return undefined;
	if (typeof candidate.acquiredAt !== "number") return undefined;
	return {
		pid: candidate.pid,
		acquiredAt: candidate.acquiredAt,
		...(typeof candidate.hostname === "string" ? { hostname: candidate.hostname } : {}),
		...(typeof candidate.startedAt === "number" ? { startedAt: candidate.startedAt } : {}),
	};
}

function envFlag(raw: string | undefined): boolean {
	const normalized = raw?.trim().toLowerCase();
	return normalized === "true" || normalized === "1" || normalized === "yes";
}

function envPositiveMs(raw: string | undefined, fallback: number): number {
	if (raw === undefined || raw.trim() === "") return fallback;
	const parsed = Number.parseInt(raw, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

/**
 * Canonical identity of a DB path for the in-process registry: real path of the file (when it exists) or of its parent
 * directory, so symlinks, `..` segments and (on win32) drive/case differences map to one key. Creates the directory.
 */
function normalizeWriterPath(dbPath: string): { readonly resolved: string; readonly key: string } {
	const absolute = path.resolve(dbPath);
	const dir = path.dirname(absolute);
	fs.mkdirSync(dir, { recursive: true });
	let resolved: string;
	try {
		resolved = fs.realpathSync.native(absolute);
	} catch {
		let realDir = dir;
		try {
			realDir = fs.realpathSync.native(dir);
		} catch {
			// Keep the unresolved directory; the mkdir above makes this unlikely.
		}
		resolved = path.join(realDir, path.basename(absolute));
	}
	return { resolved, key: process.platform === "win32" ? resolved.toLowerCase() : resolved };
}

interface InspectionContext {
	/** Normalized in-process registry key for this DB. */
	readonly key: string;
	readonly foreignStaleMs: number;
	readonly forceReclaim: boolean;
}

/** Classifies the current lock file without modifying it. */
function inspectWriterLock(lockFilePath: string, context: InspectionContext): LockInspection {
	let raw: string;
	try {
		raw = fs.readFileSync(lockFilePath, "utf8");
	} catch (error) {
		if (errorCode(error) === "ENOENT") return { kind: "missing" };
		return { kind: "contended", reason: `lock file unreadable (${errorCode(error) ?? "unknown error"})` };
	}

	const record = parseWriterLockRecord(raw);
	if (!record) {
		let ageMs: number;
		try {
			ageMs = Date.now() - fs.statSync(lockFilePath).mtimeMs;
		} catch (error) {
			if (errorCode(error) === "ENOENT") return { kind: "missing" };
			return { kind: "contended", reason: "lock file is corrupt and could not be stat'ed" };
		}
		if (ageMs >= WRITER_LOCK_CORRUPT_STALE_MS) {
			return {
				kind: "stale",
				reason: `corrupt or empty lock file older than ${WRITER_LOCK_CORRUPT_STALE_MS}ms`,
				raw,
			};
		}
		return { kind: "contended", reason: "lock file is empty or corrupt and recent; another writer may be starting" };
	}

	const localHost = os.hostname();
	if (record.hostname !== undefined && record.hostname !== localHost) {
		// The pid belongs to another machine and cannot be checked from here. A live holder refreshes the lock
		// file's mtime (heartbeat), so an old mtime means the holder is gone; otherwise stay contended unless forced.
		if (context.forceReclaim) {
			return {
				kind: "stale",
				reason: `lock held by pid ${record.pid} on foreign host "${record.hostname}" force-reclaimed (HELIOS_LOCK_FORCE_RECLAIM)`,
				raw,
				previousPid: record.pid,
				previousHost: record.hostname,
			};
		}
		let ageMs: number;
		try {
			ageMs = Date.now() - fs.statSync(lockFilePath).mtimeMs;
		} catch (error) {
			if (errorCode(error) === "ENOENT") return { kind: "missing" };
			ageMs = Date.now() - record.acquiredAt;
		}
		if (ageMs >= context.foreignStaleMs) {
			return {
				kind: "stale",
				reason: `lock held by pid ${record.pid} on foreign host "${record.hostname}" was last refreshed ${Math.round(ageMs / 1000)}s ago (limit ${Math.round(context.foreignStaleMs / 1000)}s, HELIOS_LOCK_STALE_MS)`,
				raw,
				previousPid: record.pid,
				previousHost: record.hostname,
			};
		}
		return {
			kind: "contended",
			reason:
				`held by pid ${record.pid} on host "${record.hostname}" (cannot verify from "${localHost}"); ` +
				`reclaimed automatically after ${Math.round(context.foreignStaleMs / 1000)}s without a heartbeat ` +
				"(HELIOS_LOCK_STALE_MS), or set HELIOS_LOCK_FORCE_RECLAIM=true if that host is certainly down",
		};
	}

	if (record.pid === process.pid) {
		// A same-pid lock whose path is registered as active belongs to a live writer in this process, whatever its
		// age or contents: never reclaim it.
		if (ACTIVE_WRITER_PATHS.has(context.key)) {
			return { kind: "contended", reason: "held by an active writer in this process" };
		}
		// Not held by an active writer in this process, so this is a leftover from a release that failed earlier.
		return { kind: "stale", reason: "lock left behind by this process", raw, previousPid: record.pid };
	}
	if (isProcessAlive(record.pid)) {
		return { kind: "contended", reason: `held by live pid ${record.pid}` };
	}
	return { kind: "stale", reason: `owner pid ${record.pid} is not running`, raw, previousPid: record.pid };
}

/** Creates the lock file exclusively. Returns false when it already exists. */
function tryCreateExclusive(lockFilePath: string, payload: string): boolean {
	let fd: number;
	try {
		fd = fs.openSync(lockFilePath, "wx");
	} catch (error) {
		if (errorCode(error) === "EEXIST") return false;
		throw error;
	}
	try {
		fs.writeFileSync(fd, payload, "utf8");
	} catch (error) {
		try {
			fs.closeSync(fd);
		} catch {
			// Ignore close failure; the file is removed below either way.
		}
		fs.rmSync(lockFilePath, { force: true });
		throw error;
	}
	fs.closeSync(fd);
	return true;
}

/**
 * Atomically removes a stale lock. The lock is first renamed to a unique `.stale.<pid>.<ts>` name: rename is atomic,
 * so when several processes race to reclaim the same stale lock only one rename succeeds. The winner re-verifies the
 * renamed bytes against what was inspected (if a concurrent winner's fresh lock was moved by mistake it is linked
 * back, never overwritten), deletes the renamed file, and only then does the caller `wx`-create the new lock.
 */
function reclaimStaleLock(lockFilePath: string, inspectedRaw: string): boolean {
	const staleName = `${lockFilePath}.stale.${process.pid}.${Date.now()}`;
	try {
		fs.renameSync(lockFilePath, staleName);
	} catch {
		return false; // already gone or lost the rename race
	}
	let movedRaw: string;
	try {
		movedRaw = fs.readFileSync(staleName, "utf8");
	} catch {
		return false;
	}
	if (movedRaw !== inspectedRaw) {
		// We moved a lock that is not the stale one we inspected. Restore it without clobbering anything.
		try {
			fs.linkSync(staleName, lockFilePath);
		} catch {
			// A newer lock already exists; the displaced one is unrecoverable and re-inspection will sort it out.
		}
		fs.rmSync(staleName, { force: true });
		return false;
	}
	fs.rmSync(staleName, { force: true });
	return true;
}

export interface FileWriterLockOptions {
	readonly logger?: Logger;
	/** Age (ms since last heartbeat) after which a lock from another host is reclaimed. Default `HELIOS_LOCK_STALE_MS` or 10 min. */
	readonly foreignStaleMs?: number;
	/** Reclaim a lock from another host regardless of age. Default `HELIOS_LOCK_FORCE_RECLAIM`. */
	readonly forceReclaim?: boolean;
}

export class FileWriterLock {
	private readonly key: string;
	private readonly lockFilePath: string;
	private readonly record: WriterLockRecord;
	private readonly logger: Logger;
	private heartbeat: ReturnType<typeof setInterval> | undefined;
	private released = false;

	private constructor(key: string, lockFilePath: string, record: WriterLockRecord, logger: Logger) {
		this.key = key;
		this.lockFilePath = lockFilePath;
		this.record = record;
		this.logger = logger;
	}

	/** Path of the lock file guarding `dbPath`. */
	get path(): string {
		return this.lockFilePath;
	}

	static acquire(dbPath: string, options: FileWriterLockOptions = {}): FileWriterLock {
		const { resolved, key } = normalizeWriterPath(dbPath);
		if (ACTIVE_WRITER_PATHS.has(key)) {
			throw new SingleWriterLockError(resolved, "already opened by an active writer in this process");
		}

		const lockFilePath = `${resolved}.writer.lock`;
		const logger =
			options.logger?.child({ component: "writer-lock" }) ??
			createLogger({ level: resolveLogLevel(process.env.LOG_LEVEL), base: { component: "writer-lock" } });
		const context: InspectionContext = {
			key,
			foreignStaleMs:
				options.foreignStaleMs ?? envPositiveMs(process.env.HELIOS_LOCK_STALE_MS, WRITER_LOCK_FOREIGN_STALE_MS),
			forceReclaim: options.forceReclaim ?? envFlag(process.env.HELIOS_LOCK_FORCE_RECLAIM),
		};

		for (let attempt = 1; attempt <= MAX_ACQUIRE_ATTEMPTS; attempt++) {
			const record: WriterLockRecord = {
				pid: process.pid,
				hostname: os.hostname(),
				startedAt: Math.round(Date.now() - process.uptime() * 1000),
				acquiredAt: Date.now(),
			};

			if (tryCreateExclusive(lockFilePath, JSON.stringify(record))) {
				ACTIVE_WRITER_PATHS.add(key);
				const lock = new FileWriterLock(key, lockFilePath, record, logger);
				lock.startHeartbeat(context.foreignStaleMs);
				return lock;
			}

			const inspection = inspectWriterLock(lockFilePath, context);
			if (inspection.kind === "missing") {
				// Removed by its holder or a concurrent recovery; retry the exclusive create.
				continue;
			}
			if (inspection.kind === "contended") {
				throw new SingleWriterLockError(resolved, `${inspection.reason} (lock file "${lockFilePath}")`);
			}
			if (reclaimStaleLock(lockFilePath, inspection.raw)) {
				logger.warn("writer_lock_stale_removed", {
					lockFilePath,
					reason: inspection.reason,
					previousPid: inspection.previousPid,
				});
				if (inspection.previousHost !== undefined) {
					logger.warn("writer_lock_foreign_host_reclaimed", {
						lockFilePath,
						previousHost: inspection.previousHost,
						previousPid: inspection.previousPid,
						message:
							"Reclaimed a writer lock owned by another host. If that host is still writing to this database " +
							"the data can be corrupted; make sure only one host runs against it.",
					});
				}
			}
			// Loop: a concurrent winner may have created a fresh lock in the meantime, which
			// the next inspection will classify correctly (EEXIST again, or contended).
		}

		throw new SingleWriterLockError(
			resolved,
			`could not acquire lock file "${lockFilePath}" after ${MAX_ACQUIRE_ATTEMPTS} attempts`,
		);
	}

	/** Refreshes the lock file mtime so other hosts can tell a live holder from a dead one. */
	private startHeartbeat(foreignStaleMs: number): void {
		const intervalMs = Math.min(
			MAX_HEARTBEAT_INTERVAL_MS,
			Math.max(MIN_HEARTBEAT_INTERVAL_MS, Math.floor(foreignStaleMs / 5)),
		);
		const timer = setInterval(() => {
			try {
				const current = parseWriterLockRecord(fs.readFileSync(this.lockFilePath, "utf8"));
				if (current?.pid === this.record.pid && current.acquiredAt === this.record.acquiredAt) {
					const now = new Date();
					fs.utimesSync(this.lockFilePath, now, now);
				}
			} catch {
				// Best effort: a missing/unreadable lock is surfaced on release.
			}
		}, intervalMs);
		timer.unref?.();
		this.heartbeat = timer;
	}

	/** Deletes the lock file only if it still records this process's acquisition. */
	release(): void {
		if (this.released) return;
		this.released = true;
		if (this.heartbeat) clearInterval(this.heartbeat);
		this.heartbeat = undefined;
		ACTIVE_WRITER_PATHS.delete(this.key);
		try {
			const current = parseWriterLockRecord(fs.readFileSync(this.lockFilePath, "utf8"));
			if (current?.pid === this.record.pid && current.acquiredAt === this.record.acquiredAt) {
				fs.rmSync(this.lockFilePath, { force: true });
			} else {
				this.logger.warn("writer_lock_not_owned_on_release", {
					lockFilePath: this.lockFilePath,
					currentPid: current?.pid,
				});
			}
		} catch (error) {
			if (errorCode(error) !== "ENOENT") {
				this.logger.warn("writer_lock_release_failed", { lockFilePath: this.lockFilePath, error: serializeError(error) });
			}
		}
	}
}

export interface HostRuntimeOptions {
	/** SQLite file path or ":memory:" for ephemeral in-memory storage. */
	readonly piDbPath: string;
	/** Table prefix for Pi Durable tables when sharing SQLite. Default: "pi_" */
	readonly tablePrefix?: string;
	readonly models: Models | (() => Models);
	readonly registry: RegistryReader;
	readonly settings?: HarnessSettings | (() => HarnessSettings | undefined);
	readonly wakeScheduler?: WakeScheduler;
	readonly now?: () => number;
	readonly setTimer?: SetTimer;
	readonly autoResume?: boolean;
	readonly onReport?: (report: Report) => void;
	readonly conversationCreated?: HarnessOptions["conversationCreated"];
	/** Logger for lock diagnostics. Default: stdout logger at LOG_LEVEL. */
	readonly logger?: Logger;
}

/**
 * Owns the Pi Durable Harness lifecycle over SQLite with a single-writer guard
 * and a portable WakeScheduler.
 */
export class HostRuntime {
	readonly harness: Harness;
	readonly wakeScheduler: WakeScheduler;
	readonly reports: Report[] = [];
	readonly piDbPath: string;

	private readonly writerLock: FileWriterLock | undefined;
	private closed = false;

	private constructor(
		harness: Harness,
		wakeScheduler: WakeScheduler,
		piDbPath: string,
		writerLock: FileWriterLock | undefined,
	) {
		this.harness = harness;
		this.wakeScheduler = wakeScheduler;
		this.piDbPath = piDbPath;
		this.writerLock = writerLock;
	}

	static async open(options: HostRuntimeOptions, context: Context = BACKGROUND_CONTEXT): Promise<HostRuntime> {
		const isMemory = options.piDbPath === ":memory:";
		const writerLock = isMemory ? undefined : FileWriterLock.acquire(options.piDbPath, { logger: options.logger });
		const wakeScheduler = options.wakeScheduler ?? new NodeWakeScheduler({ now: options.now });
		const reports: Report[] = [];

		let storage: Storage;
		try {
			storage = isMemory
				? new MemoryStorage()
				: await openNodeSqliteStorage(options.piDbPath, {
						prefix: options.tablePrefix ?? "pi_",
						busyTimeoutMs: 5_000,
					});
		} catch (error) {
			writerLock?.release();
			wakeScheduler.dispose();
			throw error;
		}

		let harness: Harness;
		try {
			harness = await Harness.open(
				storage,
				{
					models: options.models,
					registry: options.registry,
					settings: options.settings,
					now: options.now,
					setTimer: options.setTimer,
					identity: { attribution: { name: "helios-piharness", url: "https://helios-piharness.local" } },
					conversationCreated: options.conversationCreated,
					onWake: (wake: Wake) => {
						wakeScheduler.schedule({ active: wake.active, ...(wake.at !== undefined ? { at: wake.at } : {}) });
					},
					onReport: (report: Report) => {
						reports.push(report);
						options.onReport?.(report);
					},
				},
				context,
			);
		} catch (error) {
			try {
				await storage.close(context);
			} catch {
				// Best-effort cleanup on failed open
			}
			writerLock?.release();
			wakeScheduler.dispose();
			throw error;
		}

		const runtime = new HostRuntime(harness, wakeScheduler, options.piDbPath, writerLock);
		runtime.reports.push(...reports);

		wakeScheduler.setOnFire(() => {
			if (!runtime.closed) {
				try {
					runtime.harness.resume();
				} catch {
					// Ignore if closing concurrently
				}
			}
		});

		const initialWake = harness.wake();
		wakeScheduler.schedule({
			active: initialWake.active,
			...(initialWake.at !== undefined ? { at: initialWake.at } : {}),
		});

		if (options.autoResume ?? true) {
			harness.resume();
		}

		return runtime;
	}

	resume(): void {
		if (!this.closed) {
			this.harness.resume();
		}
	}

	wake(): Wake {
		return this.harness.wake();
	}

	async close(context: Context = BACKGROUND_CONTEXT): Promise<void> {
		if (this.closed) return;
		this.closed = true;
		this.wakeScheduler.dispose();
		try {
			await this.harness.close(context);
		} finally {
			this.writerLock?.release();
		}
	}

	isClosed(): boolean {
		return this.closed;
	}
}
