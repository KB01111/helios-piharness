/**
 * Portable wake-scheduling abstraction for Pi Durable's `onWake` / `harness.wake()`.
 *
 * Implementations:
 * - `NodeWakeScheduler`: arms a Node.js timer for local/server deployments.
 * - `ManualWakeScheduler`: deterministic clock/alarm driver for tests and
 *   serverless/evictable hosts (Cloudflare Durable Objects, Rivet Actors)
 *   where the platform alarm invokes `trigger()`.
 */

export interface WakeState {
	readonly active: boolean;
	readonly at?: number;
}

export interface WakeScheduler {
	schedule(wake: WakeState): void;
	setOnFire(handler: () => void): void;
	currentWake(): WakeState;
	dispose(): void;
}

export interface NodeWakeSchedulerOptions {
	readonly now?: () => number;
	/** Keep Node event loop alive for scheduled sleeps (default: false so tests/servers exit cleanly on shutdown). */
	readonly refTimer?: boolean;
}

export class NodeWakeScheduler implements WakeScheduler {
	private readonly now: () => number;
	private readonly refTimer: boolean;
	private timer: ReturnType<typeof setTimeout> | undefined;
	private onFire: (() => void) | undefined;
	private wake: WakeState = { active: false };
	private disposed = false;

	constructor(options?: NodeWakeSchedulerOptions) {
		this.now = options?.now ?? (() => Date.now());
		this.refTimer = options?.refTimer ?? false;
	}

	setOnFire(handler: () => void): void {
		this.onFire = handler;
	}

	currentWake(): WakeState {
		return this.wake;
	}

	schedule(wake: WakeState): void {
		if (this.disposed) return;
		this.wake = wake;
		this.clearTimer();

		if (wake.at === undefined) return;
		const delayMs = Math.max(0, wake.at - this.now());
		this.timer = setTimeout(() => {
			this.timer = undefined;
			if (!this.disposed) {
				this.onFire?.();
			}
		}, delayMs);
		if (!this.refTimer && typeof this.timer.unref === "function") {
			this.timer.unref();
		}
	}

	dispose(): void {
		this.disposed = true;
		this.clearTimer();
	}

	private clearTimer(): void {
		if (this.timer !== undefined) {
			clearTimeout(this.timer);
			this.timer = undefined;
		}
	}
}

export class ManualWakeScheduler implements WakeScheduler {
	private onFire: (() => void) | undefined;
	private wake: WakeState = { active: false };
	private disposed = false;

	setOnFire(handler: () => void): void {
		this.onFire = handler;
	}

	currentWake(): WakeState {
		return this.wake;
	}

	schedule(wake: WakeState): void {
		if (this.disposed) return;
		this.wake = wake;
	}

	/** Simulate an external platform alarm (e.g. Durable Object `alarm()`). */
	trigger(): void {
		if (this.disposed) return;
		this.onFire?.();
	}

	dispose(): void {
		this.disposed = true;
	}
}
