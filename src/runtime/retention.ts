import fs from "node:fs";
import type { AppDatabase } from "../memory/db.js";

/**
 * Data retention for append-only operational tables.
 *
 * Without it the app database and the artifact directory grow without bound: every tool call, run, permission
 * decision and budget event adds a row forever. Each window is in days; `0` keeps the data forever.
 *
 * Only tables whose readers look at a recent window are pruned (audit log, tool invocations, run telemetry, budget
 * events, daily spend ledger). Long-term memory, episodes and skills are never touched here: their lifecycle belongs
 * to the consolidator. Artifacts are opt-in because transcripts hold `art_*` recall handles that would dangle.
 */
export interface RetentionPolicy {
	/** `permission_audit_log` rows older than this many days are deleted (0 = keep). */
	readonly permissionAuditDays: number;
	/** `tool_invocations` (Skill Dojo metrics source). */
	readonly toolInvocationDays: number;
	/** `run_telemetry` (Skill Forge n-gram source). */
	readonly runTelemetryDays: number;
	/** `budget_events` (budget-block diagnostics). */
	readonly budgetEventDays: number;
	/** `budget_tenant_daily` (only the current UTC day enforces limits; older rows are reporting history). */
	readonly budgetDailyDays: number;
	/** Offloaded tool-output artifacts, including their files on disk. Opt-in. */
	readonly artifactDays: number;
}

export const DEFAULT_RETENTION_POLICY: RetentionPolicy = {
	permissionAuditDays: 365,
	toolInvocationDays: 90,
	runTelemetryDays: 90,
	budgetEventDays: 30,
	budgetDailyDays: 400,
	artifactDays: 0,
};

export interface RetentionReport {
	readonly startedAt: number;
	readonly deleted: Readonly<Record<string, number>>;
	readonly totalDeleted: number;
	/** Tables skipped because they do not exist in this database (e.g. budget tables when budgets are unused). */
	readonly skipped: readonly string[];
	readonly errors: readonly string[];
}

export interface RetentionServiceOptions {
	readonly appDb: AppDatabase;
	readonly policy?: Partial<RetentionPolicy>;
	/** Sweep interval. `<= 0` disables the periodic timer (`sweepOnce()` still works). Default 6 hours. */
	readonly intervalMs?: number;
	/** Rows deleted per statement; keeps each write lock short. Default 1000. */
	readonly batchSize?: number;
	readonly now?: () => number;
	readonly onReport?: (report: RetentionReport) => void;
	readonly onError?: (error: unknown) => void;
}

const DAY_MS = 86_400_000;
export const DEFAULT_RETENTION_INTERVAL_MS = 6 * 60 * 60 * 1000;
/** Upper bound on batches per table per sweep so one sweep cannot monopolise the event loop. */
const MAX_BATCHES_PER_TABLE = 200;

interface TimeTable {
	readonly table: string;
	readonly policyKey: keyof RetentionPolicy;
}

const TIME_TABLES: readonly TimeTable[] = [
	{ table: "permission_audit_log", policyKey: "permissionAuditDays" },
	{ table: "tool_invocations", policyKey: "toolInvocationDays" },
	{ table: "run_telemetry", policyKey: "runTelemetryDays" },
	{ table: "budget_events", policyKey: "budgetEventDays" },
];

function utcDay(atMs: number): string {
	return new Date(atMs).toISOString().slice(0, 10);
}

function yieldToEventLoop(): Promise<void> {
	return new Promise((resolve) => setImmediate(resolve));
}

export class RetentionService {
	private readonly appDb: AppDatabase;
	private readonly policy: RetentionPolicy;
	private readonly intervalMs: number;
	private readonly batchSize: number;
	private readonly now: () => number;
	private readonly onReport: ((report: RetentionReport) => void) | undefined;
	private readonly onError: ((error: unknown) => void) | undefined;
	private timer: ReturnType<typeof setInterval> | undefined;
	private running: Promise<RetentionReport> | undefined;
	private disposed = false;

	constructor(options: RetentionServiceOptions) {
		this.appDb = options.appDb;
		this.policy = { ...DEFAULT_RETENTION_POLICY, ...options.policy };
		this.intervalMs = options.intervalMs ?? DEFAULT_RETENTION_INTERVAL_MS;
		this.batchSize = Math.max(1, Math.floor(options.batchSize ?? 1000));
		this.now = options.now ?? (() => Date.now());
		this.onReport = options.onReport;
		this.onError = options.onError;
	}

	/** Runs one sweep now and arms the periodic timer. Safe to call once; a no-op after `dispose()`. */
	start(): void {
		if (this.disposed || this.timer !== undefined) return;
		void this.sweepOnce().catch((error: unknown) => this.onError?.(error));
		if (this.intervalMs > 0) {
			this.timer = setInterval(() => {
				void this.sweepOnce().catch((error: unknown) => this.onError?.(error));
			}, this.intervalMs);
			this.timer.unref?.();
		}
	}

	/** Single-flight: a sweep requested while one is running joins it instead of starting another. */
	sweepOnce(): Promise<RetentionReport> {
		if (this.running) return this.running;
		this.running = this.sweep().finally(() => {
			this.running = undefined;
		});
		return this.running;
	}

	dispose(): void {
		this.disposed = true;
		if (this.timer !== undefined) clearInterval(this.timer);
		this.timer = undefined;
	}

	/** Resolves when any in-flight sweep has finished (used to close the database safely). */
	async idle(): Promise<void> {
		await this.running?.catch(() => undefined);
	}

	private async sweep(): Promise<RetentionReport> {
		const startedAt = this.now();
		const deleted: Record<string, number> = {};
		const skipped: string[] = [];
		const errors: string[] = [];

		const run = async (label: string, fn: () => Promise<number | undefined>): Promise<void> => {
			if (this.disposed || this.appDb.isClosed()) return;
			try {
				const count = await fn();
				if (count === undefined) skipped.push(label);
				else if (count > 0) deleted[label] = count;
			} catch (error) {
				errors.push(`${label}: ${error instanceof Error ? error.message : String(error)}`);
				this.onError?.(error);
			}
		};

		for (const { table, policyKey } of TIME_TABLES) {
			const days = this.policy[policyKey];
			if (days <= 0) continue;
			await run(table, () => this.pruneByTime(table, startedAt - days * DAY_MS));
		}
		if (this.policy.budgetDailyDays > 0) {
			await run("budget_tenant_daily", () =>
				this.pruneBudgetDaily(utcDay(startedAt - this.policy.budgetDailyDays * DAY_MS)),
			);
		}
		if (this.policy.artifactDays > 0) {
			await run("artifacts", () => this.pruneArtifacts(startedAt - this.policy.artifactDays * DAY_MS));
		}

		const totalDeleted = Object.values(deleted).reduce((sum, n) => sum + n, 0);
		const report: RetentionReport = { startedAt, deleted, totalDeleted, skipped, errors };
		this.onReport?.(report);
		return report;
	}

	private tableExists(table: string): boolean {
		const row = this.appDb.db.prepare("SELECT 1 AS ok FROM sqlite_master WHERE type = 'table' AND name = ?").get(table);
		return row !== undefined;
	}

	/** Deletes rows with `created_at < cutoff` in bounded batches. Returns `undefined` when the table is absent. */
	private async pruneByTime(table: string, cutoffMs: number): Promise<number | undefined> {
		if (!this.tableExists(table)) return undefined;
		// `table` comes from the fixed TIME_TABLES allow-list above, never from input.
		const statement = this.appDb.db.prepare(
			`DELETE FROM ${table} WHERE id IN (SELECT id FROM ${table} WHERE created_at < ? ORDER BY id LIMIT ?)`,
		);
		return this.drain(() => Number(statement.run(cutoffMs, this.batchSize).changes));
	}

	private async pruneBudgetDaily(cutoffDay: string): Promise<number | undefined> {
		if (!this.tableExists("budget_tenant_daily")) return undefined;
		const statement = this.appDb.db.prepare(
			`DELETE FROM budget_tenant_daily WHERE rowid IN (SELECT rowid FROM budget_tenant_daily WHERE day < ? LIMIT ?)`,
		);
		return this.drain(() => Number(statement.run(cutoffDay, this.batchSize).changes));
	}

	private async pruneArtifacts(cutoffMs: number): Promise<number | undefined> {
		if (!this.tableExists("artifacts")) return undefined;
		const select = this.appDb.db.prepare("SELECT id, file_path FROM artifacts WHERE created_at < ? ORDER BY created_at LIMIT ?");
		const remove = this.appDb.db.prepare("DELETE FROM artifacts WHERE id = ?");
		return this.drain(() => {
			const rows = select.all(cutoffMs, this.batchSize) as Array<{ id: string; file_path: string | null }>;
			for (const row of rows) {
				// Row first: a failed file removal then leaves an orphan file, not a row pointing at nothing.
				remove.run(row.id);
				if (row.file_path) {
					try {
						fs.rmSync(row.file_path, { force: true });
					} catch (error) {
						this.onError?.(error);
					}
				}
			}
			return rows.length;
		});
	}

	private async drain(step: () => number): Promise<number> {
		let total = 0;
		for (let batch = 0; batch < MAX_BATCHES_PER_TABLE; batch++) {
			if (this.disposed || this.appDb.isClosed()) break;
			const changed = step();
			total += changed;
			if (changed < this.batchSize) break;
			await yieldToEventLoop();
		}
		return total;
	}
}
