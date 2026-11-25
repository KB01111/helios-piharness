import type { MemoryService, PendingEpisodeScope } from "../contracts/app-capabilities.js";
import type { HarnessPort } from "../contracts/harness-port.js";
import type { AppDatabase } from "../memory/db.js";

/** Injectable timer so tests can drive the scheduler deterministically. */
export interface SleepSchedulerTimer {
	setTimeout(callback: () => void, delayMs: number): unknown;
	clearTimeout(handle: unknown): void;
}

export const nodeSleepSchedulerTimer: SleepSchedulerTimer = {
	setTimeout: (callback, delayMs) => {
		const handle = setTimeout(callback, delayMs);
		if (typeof handle.unref === "function") handle.unref();
		return handle;
	},
	clearTimeout: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

export interface SleepCycleSchedulerOptions {
	readonly harness: Pick<HarnessPort, "triggerSleepCycle" | "waitTask">;
	readonly memory: MemoryService;
	/** Sweep interval. `<= 0` (or `enabled: false`) disables the scheduler entirely. */
	readonly intervalMs: number;
	readonly enabled?: boolean;
	readonly now?: () => number;
	readonly timer?: SleepSchedulerTimer;
	/** Returns [0,1). Injectable for deterministic jitter. */
	readonly random?: () => number;
	/** Optional app DB: enables the durable in-flight marker (survives restarts). */
	readonly appDb?: AppDatabase;
	/** Minimum unconsolidated episodes before a scope is considered. Default: 1 */
	readonly minPendingEpisodes?: number;
	/** Max scopes triggered per sweep. Default: 10 */
	readonly maxScopesPerTick?: number;
	/** +/- fraction of random jitter on the sweep interval and on backoff. Default: 0.1 */
	readonly jitterRatio?: number;
	/** First failure backoff. Default: intervalMs */
	readonly backoffBaseMs?: number;
	/** Backoff ceiling. Default: 1 hour */
	readonly backoffMaxMs?: number;
	/** A durable marker older than this is considered abandoned. Default: 30 minutes */
	readonly markerTtlMs?: number;
	readonly reason?: string;
	readonly onError?: (error: unknown, context: { readonly tenantId: string; readonly phase: string }) => void;
}

export interface SleepTickReport {
	readonly triggered: readonly { readonly tenantId: string; readonly userId: string; readonly taskId: string }[];
	readonly skippedInFlight: readonly string[];
	readonly skippedBackoff: readonly string[];
	readonly failed: readonly { readonly tenantId: string; readonly error: string }[];
}

type InFlight = { taskId?: string; startedAt: number; userId: string };

/** Backoff is tracked per (tenant, user) scope so one failing user cannot starve the rest of its tenant. */
function scopeKey(tenantId: string, userId: string): string {
	return `${tenantId}\u0000${userId}`;
}

/**
 * A settled cycle that "completed" but reports a consolidation problem (fast model failed / fallback failed)
 * did not do its job: it must back off like a failed task instead of resetting the failure counter.
 */
function settledWithConsolidationWarning(result: unknown): boolean {
	if (typeof result !== "object" || result === null) return false;
	const warnings = (result as { warnings?: unknown }).warnings;
	return Array.isArray(warnings) && warnings.some((w) => typeof w === "string" && w.startsWith("consolidation"));
}

const EMPTY_REPORT: SleepTickReport = { triggered: [], skippedInFlight: [], skippedBackoff: [], failed: [] };

/**
 * Durable per-tenant "a sleep cycle is running" marker. The table is created by the
 * scheduler itself (`CREATE TABLE IF NOT EXISTS`) so no shared migration is needed.
 */
class SleepCycleMarkerStore {
	constructor(private readonly appDb: AppDatabase) {
		appDb.db.exec(`
			CREATE TABLE IF NOT EXISTS sleep_cycle_markers (
				tenant_id TEXT PRIMARY KEY,
				user_id TEXT NOT NULL,
				conversation_id TEXT NOT NULL,
				task_id TEXT,
				started_at INTEGER NOT NULL
			);
		`);
	}

	/** Atomically claims the tenant unless a fresher marker exists. */
	claim(tenantId: string, userId: string, conversationId: string, nowMs: number, ttlMs: number): boolean {
		return this.appDb.transaction(() => {
			const row = this.appDb.db
				.prepare("SELECT started_at FROM sleep_cycle_markers WHERE tenant_id = ?")
				.get(tenantId) as { started_at: number } | undefined;
			if (row && nowMs - Number(row.started_at) < ttlMs) return false;
			this.appDb.db
				.prepare(
					`INSERT INTO sleep_cycle_markers (tenant_id, user_id, conversation_id, task_id, started_at)
					 VALUES (?, ?, ?, NULL, ?)
					 ON CONFLICT(tenant_id) DO UPDATE SET user_id = excluded.user_id,
					 conversation_id = excluded.conversation_id, task_id = NULL, started_at = excluded.started_at`,
				)
				.run(tenantId, userId, conversationId, nowMs);
			return true;
		});
	}

	setTask(tenantId: string, taskId: string): void {
		this.appDb.db.prepare("UPDATE sleep_cycle_markers SET task_id = ? WHERE tenant_id = ?").run(taskId, tenantId);
	}

	release(tenantId: string): void {
		this.appDb.db.prepare("DELETE FROM sleep_cycle_markers WHERE tenant_id = ?").run(tenantId);
	}

	listLive(
		nowMs: number,
		ttlMs: number,
	): Array<{ tenantId: string; userId: string; taskId: string; startedAt: number }> {
		const rows = this.appDb.db
			.prepare("SELECT tenant_id, user_id, task_id, started_at FROM sleep_cycle_markers WHERE task_id IS NOT NULL")
			.all() as Array<{ tenant_id: string; user_id: string; task_id: string; started_at: number }>;
		return rows
			.filter((row) => nowMs - Number(row.started_at) < ttlMs)
			.map((row) => ({
				tenantId: row.tenant_id,
				userId: row.user_id,
				taskId: row.task_id,
				startedAt: Number(row.started_at),
			}));
	}
}

/**
 * Periodically triggers the durable sleep-cycle task for tenants that have
 * unconsolidated episodes.
 *
 * - One sweep at a time (self-rescheduling timer, never overlapping).
 * - One cycle per tenant at a time: in-memory lock plus an optional durable marker
 *   (so a restart that resumes a running task does not start a second one).
 * - Failed tenants back off exponentially (with jitter); successes reset it.
 * - The timer is unref'd and `dispose()` stops it and cancels waiting (durable tasks keep running).
 */
export class SleepCycleScheduler {
	private readonly options: SleepCycleSchedulerOptions;
	private readonly now: () => number;
	private readonly timer: SleepSchedulerTimer;
	private readonly random: () => number;
	private readonly markers: SleepCycleMarkerStore | undefined;
	private readonly inFlight = new Map<string, InFlight>();
	private readonly failures = new Map<string, { count: number; retryAt: number }>();
	private readonly pending = new Set<Promise<void>>();
	private readonly abort = new AbortController();
	private handle: unknown;
	private started = false;
	private disposed = false;
	private sweeping: Promise<SleepTickReport> | undefined;

	constructor(options: SleepCycleSchedulerOptions) {
		this.options = options;
		this.now = options.now ?? (() => Date.now());
		this.timer = options.timer ?? nodeSleepSchedulerTimer;
		this.random = options.random ?? Math.random;
		this.markers = options.appDb ? new SleepCycleMarkerStore(options.appDb) : undefined;
	}

	get enabled(): boolean {
		return (this.options.enabled ?? true) && this.options.intervalMs > 0 && !this.disposed;
	}

	isRunning(tenantId: string): boolean {
		return this.inFlight.has(tenantId);
	}

	get inFlightCount(): number {
		return this.inFlight.size;
	}

	/** Starts the periodic sweep. No-op when disabled or already started. */
	start(): void {
		if (!this.enabled || this.started) return;
		this.started = true;
		this.recoverMarkers();
		this.scheduleNext();
	}

	/** Stops the timer and cancels waits on in-flight cycles. Idempotent. */
	dispose(): void {
		if (this.disposed) return;
		this.disposed = true;
		this.started = false;
		if (this.handle !== undefined) {
			this.timer.clearTimeout(this.handle);
			this.handle = undefined;
		}
		this.abort.abort();
	}

	/** Resolves when every currently tracked in-flight cycle has settled (test helper / graceful shutdown). */
	async idle(): Promise<void> {
		while (this.pending.size > 0) {
			await Promise.allSettled(Array.from(this.pending));
		}
	}

	/**
	 * One sweep: finds scopes with pending episodes and triggers a cycle for each tenant that is
	 * not already running and not in backoff. Returns once cycles are *triggered* (not finished).
	 */
	async tick(): Promise<SleepTickReport> {
		if (!this.enabled) return EMPTY_REPORT;
		if (this.sweeping) return this.sweeping;
		this.sweeping = this.sweep().finally(() => {
			this.sweeping = undefined;
		});
		return this.sweeping;
	}

	private async sweep(): Promise<SleepTickReport> {
		const memory = this.options.memory;
		if (typeof memory.listTenantsWithPendingEpisodes !== "function") return EMPTY_REPORT;

		let scopes: PendingEpisodeScope[];
		try {
			scopes = await memory.listTenantsWithPendingEpisodes({
				limit: Math.max(this.options.maxScopesPerTick ?? 10, 1) * 4,
				minEpisodes: this.options.minPendingEpisodes ?? 1,
			});
		} catch (error) {
			this.options.onError?.(error, { tenantId: "*", phase: "enumerate" });
			return EMPTY_REPORT;
		}

		const triggered: Array<{ tenantId: string; userId: string; taskId: string }> = [];
		const skippedInFlight: string[] = [];
		const skippedBackoff: string[] = [];
		const failed: Array<{ tenantId: string; error: string }> = [];
		const seenTenants = new Set<string>();
		const maxScopes = Math.max(this.options.maxScopesPerTick ?? 10, 1);

		for (const scope of scopes) {
			if (this.disposed) break;
			if (triggered.length >= maxScopes) break;
			// The learn phase of a sleep cycle is tenant-wide, so one cycle per tenant at a time: once a tenant
			// was handled (triggered, in flight, locked) its remaining users wait for a later sweep.
			if (seenTenants.has(scope.tenantId)) continue;

			if (this.inFlight.has(scope.tenantId)) {
				seenTenants.add(scope.tenantId);
				skippedInFlight.push(scope.tenantId);
				continue;
			}
			const failure = this.failures.get(scopeKey(scope.tenantId, scope.userId));
			if (failure && failure.retryAt > this.now()) {
				// Do NOT mark the tenant seen: move on to its next user instead of starving the tenant
				// behind a scope that keeps failing.
				if (!skippedBackoff.includes(scope.tenantId)) skippedBackoff.push(scope.tenantId);
				continue;
			}

			seenTenants.add(scope.tenantId);
			const outcome = await this.startCycle(scope);
			if (outcome === "locked") skippedInFlight.push(scope.tenantId);
			else if (typeof outcome === "string") triggered.push({ tenantId: scope.tenantId, userId: scope.userId, taskId: outcome });
			else {
				failed.push({ tenantId: scope.tenantId, error: outcome.error });
				// The failed start released the tenant; let its next (not backed-off) user try in this sweep.
				seenTenants.delete(scope.tenantId);
			}
		}

		return { triggered, skippedInFlight, skippedBackoff, failed };
	}

	private async startCycle(scope: PendingEpisodeScope): Promise<string | "locked" | { error: string }> {
		const { tenantId } = scope;
		const ttl = this.options.markerTtlMs ?? 30 * 60_000;
		const startedAt = this.now();
		const conversationIds = scope.conversationIds.length > 0 ? scope.conversationIds : [];
		if (conversationIds.length === 0) return { error: "no conversation available to host the sleep cycle" };

		this.inFlight.set(tenantId, { startedAt, userId: scope.userId });
		let markerHeld = false;
		try {
			if (this.markers) {
				markerHeld = this.markers.claim(tenantId, scope.userId, conversationIds[0]!, startedAt, ttl);
				if (!markerHeld) {
					this.inFlight.delete(tenantId);
					return "locked";
				}
			}

			let lastError: unknown;
			for (const conversationId of conversationIds) {
				try {
					const { taskId } = await this.options.harness.triggerSleepCycle(conversationId, {
						reason: this.options.reason ?? "scheduled",
					});
					const entry = this.inFlight.get(tenantId);
					if (entry) entry.taskId = taskId;
					// The durable task already exists: a marker write failure must neither trigger another cycle
					// nor release the lock, so it is isolated and the task is tracked regardless.
					try {
						this.markers?.setTask(tenantId, taskId);
					} catch (markerError) {
						this.options.onError?.(markerError, { tenantId, phase: "marker" });
					}
					this.track(tenantId, taskId);
					return taskId;
				} catch (error) {
					lastError = error;
				}
			}
			throw lastError ?? new Error("sleep cycle could not be triggered");
		} catch (error) {
			if (markerHeld) this.releaseMarker(tenantId);
			this.inFlight.delete(tenantId);
			this.recordFailure(tenantId, scope.userId);
			this.options.onError?.(error, { tenantId, phase: "trigger" });
			return { error: error instanceof Error ? error.message : String(error) };
		}
	}

	/** Waits for settlement in the background; releases the lock and records success/backoff. */
	private track(tenantId: string, taskId: string): void {
		const userId = this.inFlight.get(tenantId)?.userId ?? "";
		const run = (async () => {
			let settledOk = false;
			let cancelled = false;
			try {
				const settled = await this.options.harness.waitTask(taskId, this.abort.signal);
				settledOk = settled.status === "completed";
				if (settledOk && settledWithConsolidationWarning(settled.result)) {
					// Completed, but consolidation failed/degraded: not a success for backoff purposes.
					settledOk = false;
					this.options.onError?.(new Error("sleep cycle completed with consolidation warnings"), {
						tenantId,
						phase: "task",
					});
				}
				if (!settledOk && settled.status !== "completed") {
					this.options.onError?.(new Error(settled.error ?? `sleep cycle ${settled.status}`), {
						tenantId,
						phase: "task",
					});
				}
			} catch (error) {
				if (this.abort.signal.aborted) {
					cancelled = true;
				} else {
					this.options.onError?.(error, { tenantId, phase: "wait" });
				}
			}

			if (cancelled) {
				// Disposed while the durable task may still be running: keep the durable marker so a
				// restarted scheduler waits for it instead of starting a duplicate.
				this.inFlight.delete(tenantId);
				return;
			}
			if (settledOk) this.failures.delete(scopeKey(tenantId, userId));
			else this.recordFailure(tenantId, userId);
			// Always free the in-memory lock, even when the durable marker cannot be released right now.
			try {
				this.releaseMarker(tenantId);
			} finally {
				this.inFlight.delete(tenantId);
			}
		})();
		const tracked = run.catch(() => undefined).finally(() => this.pending.delete(tracked));
		this.pending.add(tracked);
	}

	/** After a restart, re-attach to cycles whose durable task may still be running. */
	private releaseMarker(tenantId: string): void {
		try {
			this.markers?.release(tenantId);
		} catch (error) {
			// An undeletable marker expires via its TTL; it must not wedge the in-memory lock.
			this.options.onError?.(error, { tenantId, phase: "marker" });
		}
	}

	private recoverMarkers(): void {
		if (!this.markers) return;
		let live: Array<{ tenantId: string; userId: string; taskId: string; startedAt: number }>;
		try {
			live = this.markers.listLive(this.now(), this.options.markerTtlMs ?? 30 * 60_000);
		} catch (error) {
			this.options.onError?.(error, { tenantId: "*", phase: "recover" });
			return;
		}
		for (const marker of live) {
			if (this.inFlight.has(marker.tenantId)) continue;
			this.inFlight.set(marker.tenantId, { taskId: marker.taskId, startedAt: marker.startedAt, userId: marker.userId });
			this.track(marker.tenantId, marker.taskId);
		}
	}

	private recordFailure(tenantId: string, userId: string): void {
		const key = scopeKey(tenantId, userId);
		const previous = this.failures.get(key)?.count ?? 0;
		const count = previous + 1;
		const base = this.options.backoffBaseMs ?? Math.max(this.options.intervalMs, 1);
		const max = this.options.backoffMaxMs ?? 60 * 60_000;
		const delay = Math.min(max, base * 2 ** (count - 1));
		this.failures.set(key, { count, retryAt: this.now() + this.jitter(delay) });
	}

	private jitter(delayMs: number): number {
		const ratio = Math.min(Math.max(this.options.jitterRatio ?? 0.1, 0), 1);
		const factor = 1 + (this.random() * 2 - 1) * ratio;
		return Math.max(0, Math.round(delayMs * factor));
	}

	private scheduleNext(): void {
		if (this.disposed || !this.started) return;
		this.handle = this.timer.setTimeout(() => {
			this.handle = undefined;
			void this.tick()
				.catch((error) => this.options.onError?.(error, { tenantId: "*", phase: "tick" }))
				.finally(() => this.scheduleNext());
		}, this.jitter(this.options.intervalMs));
	}
}
