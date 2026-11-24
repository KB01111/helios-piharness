/**
 * Provider-call resilience primitives: error classification, per-call timeouts,
 * bounded retry with exponential backoff + jitter, a per-provider circuit breaker,
 * ordered model fallback, and the production model-safety startup check.
 *
 * Everything time- or randomness-dependent is injectable (`sleep`, `now`, `random`,
 * `setTimer`/`clearTimer`) so tests never wait on real clocks.
 *
 * NOTE: the *main* model is already retried durably by Pi Durable (`HarnessSettings.retry`,
 * `HarnessSettings.stream.timeoutMs`); these helpers are applied to the *fast* model client
 * (which has no SDK-level retry) and must not be layered on top of the main model.
 */

import type { HarnessSettings } from "@earendil-works/pi-durable";
import type { AppConfig } from "../config.js";

// ─── Errors ──────────────────────────────────────────────────────────────────

export class ModelTimeoutError extends Error {
	readonly timeoutMs: number;
	constructor(timeoutMs: number, label?: string) {
		super(`${label ?? "Model call"} timed out after ${timeoutMs}ms`);
		this.name = "ModelTimeoutError";
		this.timeoutMs = timeoutMs;
	}
}

export class CircuitOpenError extends Error {
	readonly provider: string;
	readonly retryAt: number;
	constructor(provider: string, retryAt: number) {
		super(`Circuit breaker for provider "${provider}" is open; not calling the provider until the cooldown elapses`);
		this.name = "CircuitOpenError";
		this.provider = provider;
		this.retryAt = retryAt;
	}
}

/** A model call that completed with `stopReason: "error"` instead of rejecting. */
export class ModelResponseError extends Error {
	readonly status?: number;
	readonly retryAfterMs?: number;
	constructor(message: string, details: { readonly status?: number; readonly retryAfterMs?: number } = {}) {
		super(message);
		this.name = "ModelResponseError";
		if (details.status !== undefined) this.status = details.status;
		if (details.retryAfterMs !== undefined) this.retryAfterMs = details.retryAfterMs;
	}
}

export interface ModelAttemptFailure {
	readonly model: string;
	readonly error: unknown;
	readonly kind: ErrorKind;
}

export class AllModelsFailedError extends Error {
	readonly failures: readonly ModelAttemptFailure[];
	constructor(failures: readonly ModelAttemptFailure[]) {
		const summary = failures
			.map((f) => `${f.model} [${f.kind}]: ${f.error instanceof Error ? f.error.message : String(f.error)}`)
			.join("; ");
		super(`All model candidates failed: ${summary || "none were attempted"}`);
		this.name = "AllModelsFailedError";
		this.failures = failures;
	}
}

// ─── Classification ──────────────────────────────────────────────────────────

export type ErrorKind =
	| "rate_limit"
	| "server"
	| "network"
	| "timeout"
	| "quota"
	| "aborted"
	| "circuit_open"
	| "permanent";

export interface ClassifiedError {
	readonly kind: ErrorKind;
	readonly retryable: boolean;
	readonly status?: number;
	/** Server-requested wait (Retry-After), when exposed on the error. */
	readonly retryAfterMs?: number;
	readonly message: string;
}

const NETWORK_CODES = new Set([
	"ECONNRESET",
	"ECONNREFUSED",
	"ECONNABORTED",
	"ETIMEDOUT",
	"EPIPE",
	"ENOTFOUND",
	"EAI_AGAIN",
	"ENETUNREACH",
	"EHOSTUNREACH",
	"UND_ERR_SOCKET",
	"UND_ERR_CONNECT_TIMEOUT",
	"UND_ERR_HEADERS_TIMEOUT",
	"UND_ERR_BODY_TIMEOUT",
]);

const QUOTA_PATTERN = /insufficient_quota|quota exceeded|out of budget|billing|BUDGET_EXCEEDED/i;
const RATE_LIMIT_PATTERN = /rate.?limit|too many requests|throttl/i;
const SERVER_PATTERN =
	/overloaded|server_busy|servers are currently busy|high demand|at capacity|service.?(?:is.?)?(?:temporarily.?)?unavailable|server.?error|internal.?error|bad gateway|gateway time-?out|upstream|provider returned (?:an )?error|you can retry your request|try your request again|please retry|ResourceExhausted/i;
const NETWORK_PATTERN =
	/network.?error|connection.?(?:error|refused|reset|closed|lost)|fetch failed|socket hang up|other side closed|stream (?:was )?(?:interrupted|ended|disconnected)|ended without|cut short|websocket.?(?:closed|error)|terminated|getaddrinfo|ENOTFOUND|EAI_AGAIN/i;
const TIMEOUT_PATTERN = /timed? ?out|timeout/i;
/** HTTP status mentioned like "status 429", "API error (503): ...", "HTTP 502" or a leading "429 ...". */
const STATUS_IN_MESSAGE = /(?:status(?: code)?|http|error|code)[^0-9a-z]{0,4}([45]\d\d)(?![0-9a-z])|^\s*([45]\d\d)(?![0-9a-z])/i;

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function numericField(source: Record<string, unknown>, ...keys: readonly string[]): number | undefined {
	for (const key of keys) {
		const value = source[key];
		if (typeof value === "number" && Number.isFinite(value)) return value;
	}
	return undefined;
}

function extractStatus(error: unknown, message: string): number | undefined {
	if (isRecord(error)) {
		const direct = numericField(error, "status", "statusCode");
		if (direct !== undefined) return direct;
		if (isRecord(error.response)) {
			const nested = numericField(error.response, "status", "statusCode");
			if (nested !== undefined) return nested;
		}
		if (isRecord(error.cause)) {
			const nested = numericField(error.cause, "status", "statusCode");
			if (nested !== undefined) return nested;
		}
	}
	const match = STATUS_IN_MESSAGE.exec(message);
	const raw = match?.[1] ?? match?.[2];
	return raw !== undefined ? Number.parseInt(raw, 10) : undefined;
}

function readHeader(headers: unknown, name: string): string | undefined {
	if (!isRecord(headers)) return undefined;
	const getter = (headers as { get?: unknown }).get;
	if (typeof getter === "function") {
		const value = (getter as (key: string) => unknown).call(headers, name);
		return typeof value === "string" ? value : undefined;
	}
	for (const [key, value] of Object.entries(headers)) {
		if (key.toLowerCase() === name && (typeof value === "string" || typeof value === "number")) return String(value);
	}
	return undefined;
}

/** Parses a Retry-After value (delta-seconds or HTTP date) into milliseconds from `nowMs`. */
export function parseRetryAfter(raw: string | undefined, nowMs: number = Date.now()): number | undefined {
	if (raw === undefined) return undefined;
	const trimmed = raw.trim();
	if (trimmed === "") return undefined;
	if (/^\d+(?:\.\d+)?$/.test(trimmed)) return Math.round(Number.parseFloat(trimmed) * 1000);
	const date = Date.parse(trimmed);
	return Number.isNaN(date) ? undefined : Math.max(0, date - nowMs);
}

function extractRetryAfterMs(error: unknown, message: string, nowMs: number): number | undefined {
	if (isRecord(error)) {
		const direct = numericField(error, "retryAfterMs");
		if (direct !== undefined && direct >= 0) return direct;
		const headerSources = [error.headers, isRecord(error.response) ? error.response.headers : undefined];
		for (const headers of headerSources) {
			const ms = readHeader(headers, "retry-after-ms");
			if (ms !== undefined && /^\d+(?:\.\d+)?$/.test(ms.trim())) return Math.round(Number.parseFloat(ms));
			const parsed = parseRetryAfter(readHeader(headers, "retry-after"), nowMs);
			if (parsed !== undefined) return parsed;
		}
	}
	const match = /retry[- ]?after[^0-9]{0,12}(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec|seconds?)?/i.exec(message);
	if (match) {
		const value = Number.parseFloat(match[1] as string);
		const unit = (match[2] ?? "s").toLowerCase();
		return Math.round(unit.startsWith("m") ? value : value * 1000);
	}
	return undefined;
}

/**
 * Classifies a thrown value (or an error-response message wrapped in an Error) into a retry decision.
 * Prefers structured fields (`status`, `code`, `headers`) and falls back to message patterns.
 */
export function classifyError(error: unknown, nowMs: number = Date.now()): ClassifiedError {
	if (error instanceof CircuitOpenError) {
		return { kind: "circuit_open", retryable: false, message: error.message };
	}

	const message = error instanceof Error ? error.message : typeof error === "string" ? error : String(error);
	const name = isRecord(error) && typeof error.name === "string" ? error.name : "";
	const code = isRecord(error) && typeof error.code === "string" ? error.code : undefined;

	if (error instanceof ModelTimeoutError) {
		return { kind: "timeout", retryable: true, message };
	}
	if (name === "AbortError" || code === "ABORT_ERR") {
		return { kind: "aborted", retryable: false, message };
	}

	const status = extractStatus(error, message);
	const retryAfterMs = extractRetryAfterMs(error, message, nowMs);
	const withExtras = <T extends ClassifiedError>(base: T): ClassifiedError => ({
		...base,
		...(status !== undefined ? { status } : {}),
		...(retryAfterMs !== undefined ? { retryAfterMs } : {}),
	});

	if (QUOTA_PATTERN.test(message)) return withExtras({ kind: "quota", retryable: false, message });

	if (status !== undefined) {
		if (status === 429) return withExtras({ kind: "rate_limit", retryable: true, message });
		if (status === 408 || status === 425) return withExtras({ kind: "timeout", retryable: true, message });
		if (status >= 500 && status <= 599) return withExtras({ kind: "server", retryable: true, message });
		if (status >= 400 && status < 500) return withExtras({ kind: "permanent", retryable: false, message });
	}

	if (code !== undefined && NETWORK_CODES.has(code)) {
		return withExtras({ kind: "network", retryable: true, message });
	}
	if (RATE_LIMIT_PATTERN.test(message)) return withExtras({ kind: "rate_limit", retryable: true, message });
	if (SERVER_PATTERN.test(message)) return withExtras({ kind: "server", retryable: true, message });
	if (NETWORK_PATTERN.test(message)) return withExtras({ kind: "network", retryable: true, message });
	if (TIMEOUT_PATTERN.test(message)) return withExtras({ kind: "timeout", retryable: true, message });

	return withExtras({ kind: "permanent", retryable: false, message });
}

// ─── Injectable dependencies ─────────────────────────────────────────────────

export interface ResilienceDeps {
	/** Abortable sleep. Default: setTimeout-based. */
	readonly sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
	/** Uniform [0,1) source for jitter. Default: Math.random. */
	readonly random?: () => number;
	/** Millisecond clock (circuit breaker cooldown, Retry-After dates). Default: Date.now. */
	readonly now?: () => number;
	readonly setTimer?: (callback: () => void, ms: number) => unknown;
	readonly clearTimer?: (handle: unknown) => void;
}

export function abortError(reason?: unknown): Error {
	if (reason instanceof Error && reason.name === "AbortError") return reason;
	const error = new Error(reason instanceof Error ? reason.message : "The operation was aborted");
	error.name = "AbortError";
	return error;
}

export function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		if (signal?.aborted) {
			reject(abortError(signal.reason));
			return;
		}
		const onAbort = (): void => {
			clearTimeout(timer);
			reject(abortError(signal?.reason));
		};
		const timer = setTimeout(() => {
			signal?.removeEventListener("abort", onAbort);
			resolve();
		}, ms);
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

// ─── Timeout ─────────────────────────────────────────────────────────────────

export interface TimeoutOptions extends Pick<ResilienceDeps, "setTimer" | "clearTimer"> {
	readonly signal?: AbortSignal;
	readonly label?: string;
}

/**
 * Runs `run` with a signal that aborts (with `ModelTimeoutError`) after `timeoutMs`. The returned
 * promise rejects at the deadline even if `run` ignores the signal. `timeoutMs <= 0` disables the deadline.
 */
export function withTimeout<T>(
	run: (signal: AbortSignal) => Promise<T>,
	timeoutMs: number,
	options: TimeoutOptions = {},
): Promise<T> {
	const parent = options.signal;
	if (parent?.aborted) return Promise.reject(abortError(parent.reason));
	if (!(timeoutMs > 0) || !Number.isFinite(timeoutMs)) {
		return run(parent ?? new AbortController().signal);
	}

	const setTimer = options.setTimer ?? ((cb: () => void, ms: number) => setTimeout(cb, ms));
	const clearTimer = options.clearTimer ?? ((handle: unknown) => clearTimeout(handle as ReturnType<typeof setTimeout>));
	const controller = new AbortController();

	return new Promise<T>((resolve, reject) => {
		let settled = false;
		const cleanup = (): void => {
			clearTimer(timer);
			parent?.removeEventListener("abort", onParentAbort);
		};
		const finish = (action: () => void): void => {
			if (settled) return;
			settled = true;
			cleanup();
			action();
		};
		const onParentAbort = (): void => {
			const error = abortError(parent?.reason);
			controller.abort(error);
			finish(() => reject(error));
		};
		const timer = setTimer(() => {
			const error = new ModelTimeoutError(timeoutMs, options.label);
			controller.abort(error);
			finish(() => reject(error));
		}, timeoutMs);
		parent?.addEventListener("abort", onParentAbort, { once: true });

		let promise: Promise<T>;
		try {
			promise = run(controller.signal);
		} catch (error) {
			finish(() => reject(error));
			return;
		}
		promise.then(
			(value) => finish(() => resolve(value)),
			(error: unknown) => finish(() => reject(error)),
		);
	});
}

// ─── Retry ───────────────────────────────────────────────────────────────────

export interface RetryEvent {
	/** 1-based retry number (the first retry is 1). */
	readonly retry: number;
	readonly maxRetries: number;
	readonly delayMs: number;
	readonly error: unknown;
	readonly classified: ClassifiedError;
}

export interface RetryOptions extends ResilienceDeps {
	/** Retries after the first attempt. 0 disables retrying. */
	readonly maxRetries: number;
	readonly baseDelayMs: number;
	/** Cap on a computed backoff delay. Default 30 000. */
	readonly maxDelayMs?: number;
	/** A server-requested wait above this is not honored; the call fails instead. Default 60 000. */
	readonly maxRetryAfterMs?: number;
	readonly signal?: AbortSignal;
	readonly classify?: (error: unknown) => ClassifiedError;
	readonly onRetry?: (event: RetryEvent) => void;
}

/** Exponential backoff with equal jitter: half of the delay is fixed, half is random. */
export function computeBackoffMs(
	retry: number,
	baseDelayMs: number,
	maxDelayMs: number,
	random: () => number = Math.random,
): number {
	const exponential = Math.min(maxDelayMs, baseDelayMs * 2 ** Math.max(0, retry - 1));
	return Math.round(exponential / 2 + (exponential / 2) * random());
}

export async function withRetry<T>(fn: (attempt: number) => Promise<T>, options: RetryOptions): Promise<T> {
	const sleep = options.sleep ?? defaultSleep;
	const random = options.random ?? Math.random;
	const now = options.now ?? Date.now;
	const maxDelayMs = options.maxDelayMs ?? 30_000;
	const maxRetryAfterMs = options.maxRetryAfterMs ?? 60_000;
	const classify = options.classify ?? ((error: unknown) => classifyError(error, now()));
	const maxRetries = Math.max(0, Math.floor(options.maxRetries));

	for (let attempt = 1; ; attempt++) {
		if (options.signal?.aborted) throw abortError(options.signal.reason);
		try {
			return await fn(attempt);
		} catch (error) {
			if (options.signal?.aborted) throw error;
			const classified = classify(error);
			const retriesUsed = attempt - 1;
			if (!classified.retryable || retriesUsed >= maxRetries) throw error;
			if (classified.retryAfterMs !== undefined && classified.retryAfterMs > maxRetryAfterMs) throw error;

			const backoff = computeBackoffMs(retriesUsed + 1, options.baseDelayMs, maxDelayMs, random);
			const delayMs = classified.retryAfterMs !== undefined ? Math.max(backoff, classified.retryAfterMs) : backoff;
			options.onRetry?.({ retry: retriesUsed + 1, maxRetries, delayMs, error, classified });
			await sleep(delayMs, options.signal);
		}
	}
}

// ─── Circuit breaker ─────────────────────────────────────────────────────────

export type CircuitState = "closed" | "open" | "half_open";

export interface CircuitBreakerOptions {
	/** Consecutive tripping failures before the circuit opens. Default 5. */
	readonly failureThreshold?: number;
	/** Time the circuit stays open before a single half-open probe is allowed. Default 30 000. */
	readonly cooldownMs?: number;
	readonly now?: () => number;
	/**
	 * How long a half-open probe may stay in flight before it is considered abandoned and another probe is
	 * allowed (so a hung call, e.g. with `timeoutMs: 0`, cannot wedge the breaker). Default: max(cooldownMs, 30 000).
	 */
	readonly probeTimeoutMs?: number;
	/** Which failures count against the provider. Default: retryable (transient) errors. */
	readonly shouldTrip?: (error: unknown) => boolean;
	readonly onStateChange?: (change: { readonly provider: string; readonly from: CircuitState; readonly to: CircuitState }) => void;
}

export class CircuitBreaker {
	readonly provider: string;
	private readonly failureThreshold: number;
	private readonly cooldownMs: number;
	private readonly now: () => number;
	private readonly shouldTrip: (error: unknown) => boolean;
	private readonly onStateChange: CircuitBreakerOptions["onStateChange"];
	private current: CircuitState = "closed";
	private consecutiveFailures = 0;
	private openedAt = 0;
	private probeInFlight = false;
	private probeStartedAt = 0;
	private readonly probeTimeoutMs: number;
	/** Bumped on every state transition / abandoned probe; results from calls begun in an older epoch are ignored. */
	private epoch = 0;

	constructor(provider: string, options: CircuitBreakerOptions = {}) {
		this.provider = provider;
		this.failureThreshold = Math.max(1, options.failureThreshold ?? 5);
		this.cooldownMs = Math.max(0, options.cooldownMs ?? 30_000);
		this.probeTimeoutMs = Math.max(1, options.probeTimeoutMs ?? Math.max(this.cooldownMs, 30_000));
		this.now = options.now ?? Date.now;
		this.shouldTrip = options.shouldTrip ?? ((error) => classifyError(error, this.now()).retryable);
		this.onStateChange = options.onStateChange;
	}

	/** Current state; an open circuit whose cooldown elapsed reports `half_open`. */
	get state(): CircuitState {
		if (this.current === "open" && this.now() - this.openedAt >= this.cooldownMs) {
			this.transition("half_open");
		}
		return this.current;
	}

	get failures(): number {
		return this.consecutiveFailures;
	}

	/** Reserves permission for one call: always when closed, one probe at a time when half-open. */
	allowRequest(): boolean {
		return this.acquire() !== undefined;
	}

	/** Returns a ticket (the current epoch) when a call may proceed, or undefined when it must be rejected. */
	private acquire(): number | undefined {
		const state = this.state;
		if (state === "closed") return this.epoch;
		if (state === "open") return undefined;
		if (this.probeInFlight) {
			if (this.now() - this.probeStartedAt < this.probeTimeoutMs) return undefined;
			// The previous probe never reported back: abandon it (its late result becomes stale) and probe again.
			this.epoch += 1;
		}
		this.probeInFlight = true;
		this.probeStartedAt = this.now();
		return this.epoch;
	}

	/** A result from a call that began before the current state (or an abandoned probe) says nothing about it. */
	private isStale(ticket: number | undefined): boolean {
		return ticket !== undefined && ticket !== this.epoch;
	}

	recordSuccess(ticket?: number): void {
		if (this.isStale(ticket)) return;
		this.probeInFlight = false;
		this.consecutiveFailures = 0;
		if (this.current !== "closed") this.transition("closed");
	}

	recordFailure(ticket?: number): void {
		if (this.isStale(ticket)) return;
		this.probeInFlight = false;
		if (this.current === "half_open" || this.state === "half_open") {
			this.open();
			return;
		}
		this.consecutiveFailures += 1;
		if (this.consecutiveFailures >= this.failureThreshold) this.open();
	}

	/**
	 * Frees a half-open probe slot WITHOUT recording success or failure: used for caller aborts and for
	 * non-retryable answers (4xx / quota) that neither prove the provider healthy nor count against it.
	 */
	private releaseProbe(ticket: number | undefined): void {
		if (this.isStale(ticket)) return;
		this.probeInFlight = false;
	}

	/** Runs `fn` through the breaker; throws `CircuitOpenError` while open. */
	async execute<T>(fn: () => Promise<T>): Promise<T> {
		const ticket = this.acquire();
		if (ticket === undefined) {
			throw new CircuitOpenError(this.provider, this.openedAt + this.cooldownMs);
		}
		let result: T;
		try {
			result = await fn();
		} catch (error) {
			const classified = classifyError(error, this.now());
			if (classified.kind === "aborted") {
				// A caller abort says nothing about provider health; free the probe slot only.
				this.releaseProbe(ticket);
			} else if (this.shouldTrip(error)) {
				this.recordFailure(ticket);
			} else {
				// Non-retryable answer (e.g. 400/quota): not a provider-health signal either way. Do not reset the
				// consecutive-failure count or close the circuit; just make sure a half-open probe slot is freed.
				this.releaseProbe(ticket);
			}
			throw error;
		}
		this.recordSuccess(ticket);
		return result;
	}

	private open(): void {
		this.openedAt = this.now();
		this.transition("open");
	}

	private transition(to: CircuitState): void {
		const from = this.current;
		if (from === to) return;
		this.current = to;
		this.epoch += 1;
		this.onStateChange?.({ provider: this.provider, from, to });
	}
}

/** Lazily creates one breaker per provider id. */
export class ProviderCircuitBreakers {
	private readonly breakers = new Map<string, CircuitBreaker>();
	constructor(private readonly options: CircuitBreakerOptions = {}) {}

	get(provider: string): CircuitBreaker {
		let breaker = this.breakers.get(provider);
		if (!breaker) {
			breaker = new CircuitBreaker(provider, this.options);
			this.breakers.set(provider, breaker);
		}
		return breaker;
	}

	snapshot(): Record<string, { readonly state: CircuitState; readonly failures: number }> {
		const out: Record<string, { readonly state: CircuitState; readonly failures: number }> = {};
		for (const [provider, breaker] of this.breakers) {
			out[provider] = { state: breaker.state, failures: breaker.failures };
		}
		return out;
	}
}

// ─── Fallback chain ──────────────────────────────────────────────────────────

export interface ModelCandidate<T> {
	/** Display key such as "anthropic:claude-haiku-4-5". */
	readonly key: string;
	/** Provider id used for circuit breaking. */
	readonly provider: string;
	readonly run: (signal: AbortSignal) => Promise<T>;
}

export type ResilienceEvent =
	| { readonly type: "retry"; readonly model: string; readonly retry: number; readonly maxRetries: number; readonly delayMs: number; readonly kind: ErrorKind; readonly message: string }
	| { readonly type: "fallback"; readonly from: string; readonly to: string; readonly kind: ErrorKind; readonly message: string }
	| { readonly type: "circuit_state"; readonly provider: string; readonly from: CircuitState; readonly to: CircuitState }
	| { readonly type: "failed"; readonly models: readonly string[] };

export interface FallbackChainOptions extends ResilienceDeps {
	readonly timeoutMs: number;
	readonly maxRetries: number;
	readonly baseDelayMs: number;
	readonly maxDelayMs?: number;
	readonly maxRetryAfterMs?: number;
	readonly breakers?: ProviderCircuitBreakers;
	readonly signal?: AbortSignal;
	readonly onEvent?: (event: ResilienceEvent) => void;
}

/**
 * Tries each candidate in order. Per candidate: circuit breaker gate, then a per-attempt timeout, then
 * bounded retry on transient errors. A candidate that is exhausted, circuit-open, or permanently failing
 * hands over to the next one; a caller abort stops the chain immediately.
 */
export async function runWithFallbacks<T>(
	candidates: readonly ModelCandidate<T>[],
	options: FallbackChainOptions,
): Promise<T> {
	const failures: ModelAttemptFailure[] = [];
	const now = options.now ?? Date.now;

	for (let index = 0; index < candidates.length; index++) {
		const candidate = candidates[index] as ModelCandidate<T>;
		const breaker = options.breakers?.get(candidate.provider);
		try {
			return await withRetry(
				(): Promise<T> => {
					const attempt = (): Promise<T> =>
						withTimeout(candidate.run, options.timeoutMs, {
							...(options.signal ? { signal: options.signal } : {}),
							label: candidate.key,
							...(options.setTimer ? { setTimer: options.setTimer } : {}),
							...(options.clearTimer ? { clearTimer: options.clearTimer } : {}),
						});
					return breaker ? breaker.execute(attempt) : attempt();
				},
				{
					maxRetries: options.maxRetries,
					baseDelayMs: options.baseDelayMs,
					...(options.maxDelayMs !== undefined ? { maxDelayMs: options.maxDelayMs } : {}),
					...(options.maxRetryAfterMs !== undefined ? { maxRetryAfterMs: options.maxRetryAfterMs } : {}),
					...(options.signal ? { signal: options.signal } : {}),
					...(options.sleep ? { sleep: options.sleep } : {}),
					...(options.random ? { random: options.random } : {}),
					now,
					onRetry: (event) =>
						options.onEvent?.({
							type: "retry",
							model: candidate.key,
							retry: event.retry,
							maxRetries: event.maxRetries,
							delayMs: event.delayMs,
							kind: event.classified.kind,
							message: event.classified.message,
						}),
				},
			);
		} catch (error) {
			const classified = classifyError(error, now());
			if (classified.kind === "aborted" || options.signal?.aborted) throw error;
			failures.push({ model: candidate.key, error, kind: classified.kind });
			const next = candidates[index + 1];
			if (next) {
				options.onEvent?.({
					type: "fallback",
					from: candidate.key,
					to: next.key,
					kind: classified.kind,
					message: classified.message,
				});
			}
		}
	}

	options.onEvent?.({ type: "failed", models: failures.map((f) => f.model) });
	throw new AllModelsFailedError(failures);
}

// ─── Production model safety ─────────────────────────────────────────────────

export interface ModelSafetyConfig {
	readonly useMockModels: boolean;
	/** Mirrors ALLOW_MOCK_MODELS; the env var is also consulted. */
	readonly allowMockModels?: boolean;
}

export class MockModelsInProductionError extends Error {
	constructor() {
		super(
			"Refusing to start: mock models are active while NODE_ENV=production. This usually means no provider API key " +
				"(ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY) is set, so USE_MOCK_MODELS silently defaulted to true. " +
				"Set a provider API key, or set ALLOW_MOCK_MODELS=true to knowingly run mock models in production.",
		);
		this.name = "MockModelsInProductionError";
	}
}

export interface ModelSafetyOptions {
	readonly env?: Readonly<Record<string, string | undefined>>;
	/** Warning sink. Default: one JSON line on stderr. */
	readonly warn?: (message: string, fields: Readonly<Record<string, unknown>>) => void;
}

function envFlag(raw: string | undefined): boolean {
	const normalized = raw?.trim().toLowerCase();
	return normalized === "true" || normalized === "1" || normalized === "yes";
}

/**
 * Startup guard against the silent mock-model fallback. Throws when mock models are active in production
 * unless explicitly allowed; otherwise logs a loud warning whenever mock models are active.
 */
export function assertProductionModelSafety(config: ModelSafetyConfig, options: ModelSafetyOptions = {}): void {
	if (!config.useMockModels) return;
	const env = options.env ?? process.env;
	const production = (env.NODE_ENV ?? "").trim().toLowerCase() === "production";
	const allowed = config.allowMockModels === true || envFlag(env.ALLOW_MOCK_MODELS);

	if (production && !allowed) {
		throw new MockModelsInProductionError();
	}

	const warn =
		options.warn ??
		((message: string, fields: Readonly<Record<string, unknown>>): void => {
			process.stderr.write(`${JSON.stringify({ level: "warn", event: "mock_models_active", message, ...fields })}\n`);
		});
	warn(
		production
			? "MOCK MODELS ARE ACTIVE IN PRODUCTION (ALLOW_MOCK_MODELS=true). Responses are canned echoes, not real model output."
			: "MOCK MODELS ARE ACTIVE: responses are canned echoes, not real model output. Set a provider API key to use real models.",
		{ nodeEnv: env.NODE_ENV ?? null, allowMockModels: allowed },
	);
}

// ─── Main-model settings (Pi Durable owns main-model retry/timeout) ─────────

/**
 * Maps config onto Pi Durable's `HarnessSettings` so the *main* model's per-request timeout and durable
 * retry/backoff (which survive restarts, unlike an in-process wrapper) are tunable without double-retrying.
 */
export function buildHarnessSettings(
	config: Pick<
		AppConfig,
		"modelTimeoutMs" | "modelMaxRetries" | "modelRetryBaseDelayMs" | "modelProviderMaxRetries"
	>,
): HarnessSettings {
	return {
		stream: {
			...(config.modelTimeoutMs > 0 ? { timeoutMs: config.modelTimeoutMs } : {}),
			...(config.modelProviderMaxRetries !== undefined ? { maxRetries: config.modelProviderMaxRetries } : {}),
		},
		retry: {
			enabled: config.modelMaxRetries > 0,
			maxRetries: config.modelMaxRetries,
			baseDelayMs: config.modelRetryBaseDelayMs,
		},
	};
}
