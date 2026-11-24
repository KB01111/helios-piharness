import fs from "node:fs";
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
import { NodeWakeScheduler, type WakeScheduler } from "./wake-scheduler.js";

export class SingleWriterLockError extends Error {
	readonly dbPath: string;

	constructor(dbPath: string, detail: string) {
		super(`Single-writer lock failed for SQLite DB "${dbPath}": ${detail}`);
		this.name = "SingleWriterLockError";
		this.dbPath = dbPath;
	}
}

/** Process-wide active DB path registry preventing two HostRuntime writers on one SQLite file. */
const ACTIVE_WRITER_PATHS = new Set<string>();

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

class FileWriterLock {
	private readonly normalizedPath: string;
	private readonly lockFilePath: string;
	private released = false;

	private constructor(normalizedPath: string, lockFilePath: string) {
		this.normalizedPath = normalizedPath;
		this.lockFilePath = lockFilePath;
	}

	static acquire(dbPath: string): FileWriterLock {
		const normalizedPath = path.resolve(dbPath);
		if (ACTIVE_WRITER_PATHS.has(normalizedPath)) {
			throw new SingleWriterLockError(normalizedPath, "already opened by an active writer in this process");
		}

		fs.mkdirSync(path.dirname(normalizedPath), { recursive: true });
		const lockFilePath = `${normalizedPath}.writer.lock`;

		const payload = JSON.stringify({ pid: process.pid, acquiredAt: Date.now() });
		const tryCreate = (): void => {
			const fd = fs.openSync(lockFilePath, "wx");
			try {
				fs.writeFileSync(fd, payload, "utf8");
			} finally {
				fs.closeSync(fd);
			}
		};

		try {
			tryCreate();
		} catch (error) {
			const code = (error as NodeJS.ErrnoException | undefined)?.code;
			if (code !== "EEXIST") throw error;

			// Check for stale lock left behind by a crashed process
			let stale = false;
			try {
				const raw = fs.readFileSync(lockFilePath, "utf8");
				const parsed = JSON.parse(raw) as { pid?: number };
				if (typeof parsed.pid === "number" && parsed.pid !== process.pid && !isProcessAlive(parsed.pid)) {
					stale = true;
				}
			} catch {
				// Unreadable lock file treated as contended unless removed manually
			}

			if (!stale) {
				throw new SingleWriterLockError(normalizedPath, `lock file exists at "${lockFilePath}"`);
			}
			fs.rmSync(lockFilePath, { force: true });
			tryCreate();
		}

		ACTIVE_WRITER_PATHS.add(normalizedPath);
		return new FileWriterLock(normalizedPath, lockFilePath);
	}

	release(): void {
		if (this.released) return;
		this.released = true;
		ACTIVE_WRITER_PATHS.delete(this.normalizedPath);
		try {
			fs.rmSync(this.lockFilePath, { force: true });
		} catch {
			// Best-effort cleanup
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
		const writerLock = isMemory ? undefined : FileWriterLock.acquire(options.piDbPath);
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
