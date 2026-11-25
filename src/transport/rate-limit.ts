export interface RateLimiterOptions {
	/** Max requests per key per window. */
	readonly limit: number;
	readonly windowMs: number;
	readonly now?: () => number;
	/** Max distinct keys tracked; the oldest are evicted beyond this (default 10_000). */
	readonly maxBuckets?: number;
}

export const DEFAULT_MAX_BUCKETS = 10_000;

export interface RateLimitDecision {
	readonly allowed: boolean;
	readonly remaining: number;
	readonly retryAfterSeconds: number;
}

/** Basic in-memory fixed-window limiter keyed by an arbitrary string (the tenant id). */
export class RateLimiter {
	private readonly buckets = new Map<string, { windowStart: number; count: number }>();
	private readonly limit: number;
	private readonly windowMs: number;
	private readonly now: () => number;
	private readonly maxBuckets: number;
	private lastSweep = 0;

	constructor(options: RateLimiterOptions) {
		this.limit = options.limit;
		this.windowMs = options.windowMs;
		this.now = options.now ?? (() => Date.now());
		this.maxBuckets = Math.max(1, options.maxBuckets ?? DEFAULT_MAX_BUCKETS);
	}

	/** Number of tracked keys (for tests/diagnostics). */
	get size(): number {
		return this.buckets.size;
	}

	check(key: string): RateLimitDecision {
		const now = this.now();
		this.sweep(now);
		let bucket = this.buckets.get(key);
		if (!bucket || now - bucket.windowStart >= this.windowMs) {
			bucket = { windowStart: now, count: 0 };
			// Delete first so a re-created key moves to the end of the insertion order (oldest-first eviction).
			this.buckets.delete(key);
			this.evictOldest();
			this.buckets.set(key, bucket);
		}
		bucket.count++;
		const retryAfterSeconds = Math.max(1, Math.ceil((bucket.windowStart + this.windowMs - now) / 1000));
		if (bucket.count > this.limit) return { allowed: false, remaining: 0, retryAfterSeconds };
		return { allowed: true, remaining: this.limit - bucket.count, retryAfterSeconds };
	}

	private evictOldest(): void {
		while (this.buckets.size >= this.maxBuckets) {
			const oldest = this.buckets.keys().next();
			if (oldest.done) return;
			this.buckets.delete(oldest.value);
		}
	}

	private sweep(now: number): void {
		if (now - this.lastSweep < this.windowMs) return;
		this.lastSweep = now;
		for (const [key, bucket] of this.buckets) {
			if (now - bucket.windowStart >= this.windowMs) this.buckets.delete(key);
		}
	}
}
