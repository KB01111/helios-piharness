import { DatabaseSync } from "node:sqlite";
import os from "node:os";
import path from "node:path";
import { isContextOverflow, isRetryableAssistantError, type AssistantMessage } from "@earendil-works/pi-ai";
import type { Models } from "@earendil-works/pi-ai/models";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
	AllModelsFailedError,
	assertProductionModelSafety,
	bootstrapHarness,
	BudgetStore,
	buildHarnessSettings,
	CircuitBreaker,
	CircuitOpenError,
	classifyError,
	createFastModelClient,
	DEFAULT_PERMISSIONS,
	evaluateBudget,
	loadConfig,
	MockModelsInProductionError,
	ModelTimeoutError,
	ProviderCircuitBreakers,
	withRetry,
	withTimeout,
	type BootstrappedHarness,
	type BudgetDiagnostic,
	type ResilienceEvent,
	type TenantContext,
} from "../src/index.js";
import { buildViolation, usageCostUsd } from "../src/runtime/budget.js";
import { parseRetryAfter, runWithFallbacks, type ModelCandidate } from "../src/runtime/resilience.js";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fastDeps() {
	const sleeps: number[] = [];
	return {
		sleeps,
		deps: {
			sleep: async (ms: number) => {
				sleeps.push(ms);
			},
			random: () => 0.5,
		},
	};
}

class HttpError extends Error {
	constructor(
		message: string,
		readonly status: number,
		readonly headers?: Record<string, string>,
	) {
		super(message);
	}
}

/** Minimal fake `Models`: scripted per-model completions. */
function fakeModels(
	script: Record<string, Array<Partial<AssistantMessage> | Error | (() => Promise<AssistantMessage>)>>,
	registered: readonly string[] = Object.keys(script),
): { models: Models; calls: string[] } {
	const calls: string[] = [];
	const models = {
		getModel(provider: string, id: string) {
			const key = `${provider}:${id}`;
			return registered.includes(key) ? { provider, id, api: "fake" } : undefined;
		},
		async completeSimple(model: { provider: string; id: string }) {
			const key = `${model.provider}:${model.id}`;
			calls.push(key);
			const next = script[key]?.shift();
			if (next === undefined) throw new Error(`no scripted response left for ${key}`);
			if (next instanceof Error) throw next;
			if (typeof next === "function") return next();
			return { role: "assistant", content: [], stopReason: "stop", ...next } as AssistantMessage;
		},
	} as unknown as Models;
	return { models, calls };
}

const text = (value: string) => [{ type: "text" as const, text: value }];
const okReply = (json: unknown): Partial<AssistantMessage> => ({ stopReason: "stop", content: text(JSON.stringify(json)) });
const errorReply = (errorMessage: string): Partial<AssistantMessage> => ({ stopReason: "error", errorMessage, content: [] });

// ─── Error classification ────────────────────────────────────────────────────

describe("classifyError", () => {
	it("classifies HTTP statuses from structured fields", () => {
		expect(classifyError(new HttpError("boom", 429))).toMatchObject({ kind: "rate_limit", retryable: true });
		expect(classifyError(new HttpError("boom", 503))).toMatchObject({ kind: "server", retryable: true });
		expect(classifyError(new HttpError("boom", 408))).toMatchObject({ kind: "timeout", retryable: true });
		expect(classifyError(new HttpError("bad request", 400))).toMatchObject({ kind: "permanent", retryable: false });
		expect(classifyError(new HttpError("nope", 401))).toMatchObject({ kind: "permanent", retryable: false });
	});

	it("classifies network codes, timeouts, messages and quota exhaustion", () => {
		const reset = Object.assign(new Error("read failed"), { code: "ECONNRESET" });
		expect(classifyError(reset)).toMatchObject({ kind: "network", retryable: true });
		expect(classifyError(new Error("fetch failed"))).toMatchObject({ kind: "network", retryable: true });
		expect(classifyError(new ModelTimeoutError(100))).toMatchObject({ kind: "timeout", retryable: true });
		expect(classifyError(new Error("Overloaded"))).toMatchObject({ kind: "server", retryable: true });
		expect(classifyError(new Error("Anthropic API error (529): overloaded"))).toMatchObject({ retryable: true });
		expect(classifyError(new Error("insufficient_quota: billing"))).toMatchObject({ kind: "quota", retryable: false });
		expect(classifyError(new Error("invalid api key"))).toMatchObject({ kind: "permanent", retryable: false });
		expect(classifyError(Object.assign(new Error("x"), { name: "AbortError" }))).toMatchObject({
			kind: "aborted",
			retryable: false,
		});
		expect(classifyError(new CircuitOpenError("p", 0))).toMatchObject({ kind: "circuit_open", retryable: false });
	});

	it("does not mistake incidental numbers for HTTP statuses", () => {
		expect(classifyError(new Error("max 500 tokens allowed in this field")).retryable).toBe(false);
	});

	it("extracts Retry-After from headers (plain, Headers, ms variant) and message text", () => {
		expect(classifyError(new HttpError("slow down", 429, { "Retry-After": "7" })).retryAfterMs).toBe(7000);
		const withHeaders = Object.assign(new Error("slow down"), {
			status: 429,
			response: { headers: new Headers({ "retry-after-ms": "1500" }) },
		});
		expect(classifyError(withHeaders).retryAfterMs).toBe(1500);
		expect(classifyError(new Error("429 rate limited, retry after 3 seconds")).retryAfterMs).toBe(3000);
		expect(parseRetryAfter("Wed, 21 Oct 2015 07:28:10 GMT", Date.parse("Wed, 21 Oct 2015 07:28:00 GMT"))).toBe(10_000);
		expect(parseRetryAfter("garbage")).toBeUndefined();
	});
});

// ─── Timeout ─────────────────────────────────────────────────────────────────

describe("withTimeout", () => {
	afterEach(() => {
		vi.useRealTimers();
	});

	it("rejects with ModelTimeoutError at the deadline and aborts the signal", async () => {
		vi.useFakeTimers();
		let received: AbortSignal | undefined;
		const pending = withTimeout(
			(signal) => {
				received = signal;
				return new Promise<string>(() => {
					/* never settles */
				});
			},
			1_000,
			{ label: "fast" },
		);
		const assertion = expect(pending).rejects.toBeInstanceOf(ModelTimeoutError);
		await vi.advanceTimersByTimeAsync(999);
		expect(received?.aborted).toBe(false);
		await vi.advanceTimersByTimeAsync(1);
		await assertion;
		expect(received?.aborted).toBe(true);
	});

	it("resolves normally and clears the timer", async () => {
		vi.useFakeTimers();
		await expect(withTimeout(async () => "ok", 1_000)).resolves.toBe("ok");
		expect(vi.getTimerCount()).toBe(0);
	});

	it("propagates a caller abort and a non-positive timeout disables the deadline", async () => {
		const controller = new AbortController();
		const pending = withTimeout(() => new Promise<string>(() => undefined), 60_000, { signal: controller.signal });
		controller.abort();
		await expect(pending).rejects.toMatchObject({ name: "AbortError" });
		await expect(withTimeout(async () => "x", 0)).resolves.toBe("x");
	});
});

// ─── Retry ───────────────────────────────────────────────────────────────────

describe("withRetry", () => {
	it("retries transient errors with exponential backoff + jitter and then succeeds", async () => {
		const { sleeps, deps } = fastDeps();
		let calls = 0;
		const result = await withRetry(
			async () => {
				calls += 1;
				if (calls < 4) throw new HttpError("unavailable", 503);
				return "done";
			},
			{ maxRetries: 3, baseDelayMs: 100, ...deps },
		);
		expect(result).toBe("done");
		expect(calls).toBe(4);
		// random()=0.5 → delay = exp/2 + exp/2*0.5 = 0.75*exp for exp = 100, 200, 400
		expect(sleeps).toEqual([75, 150, 300]);
	});

	it("caps the delay and never retries permanent errors", async () => {
		const { sleeps, deps } = fastDeps();
		await expect(
			withRetry(
				async () => {
					throw new HttpError("bad request", 400);
				},
				{ maxRetries: 5, baseDelayMs: 100, ...deps },
			),
		).rejects.toThrow("bad request");
		expect(sleeps).toEqual([]);

		let attempts = 0;
		await expect(
			withRetry(
				async () => {
					attempts += 1;
					throw new HttpError("down", 500);
				},
				{ maxRetries: 4, baseDelayMs: 1_000, maxDelayMs: 2_000, ...deps },
			),
		).rejects.toThrow("down");
		expect(attempts).toBe(5);
		expect(Math.max(...sleeps)).toBeLessThanOrEqual(2_000);
	});

	it("honors Retry-After (never sleeping less) and refuses absurd waits", async () => {
		const { sleeps, deps } = fastDeps();
		let calls = 0;
		await withRetry(
			async () => {
				calls += 1;
				if (calls === 1) throw new HttpError("slow down", 429, { "retry-after": "10" });
				return "ok";
			},
			{ maxRetries: 2, baseDelayMs: 100, ...deps },
		);
		expect(sleeps).toEqual([10_000]);

		await expect(
			withRetry(
				async () => {
					throw new HttpError("slow down", 429, { "retry-after": "3600" });
				},
				{ maxRetries: 2, baseDelayMs: 100, maxRetryAfterMs: 60_000, ...deps },
			),
		).rejects.toThrow("slow down");
	});

	it("emits onRetry events and stops when aborted", async () => {
		const { deps } = fastDeps();
		const events: number[] = [];
		const controller = new AbortController();
		await expect(
			withRetry(
				async () => {
					throw new Error("fetch failed");
				},
				{
					maxRetries: 3,
					baseDelayMs: 10,
					...deps,
					signal: controller.signal,
					onRetry: (e) => {
						events.push(e.retry);
						if (e.retry === 2) controller.abort();
					},
				},
			),
		).rejects.toThrow();
		expect(events).toEqual([1, 2]);
	});
});

// ─── Circuit breaker ─────────────────────────────────────────────────────────

describe("CircuitBreaker", () => {
	it("opens after N consecutive failures, half-opens after the cooldown, and closes on a successful probe", async () => {
		let now = 0;
		const changes: string[] = [];
		const breaker = new CircuitBreaker("anthropic", {
			failureThreshold: 3,
			cooldownMs: 1_000,
			now: () => now,
			onStateChange: (c) => changes.push(`${c.from}->${c.to}`),
		});
		const fail = () => breaker.execute(async () => Promise.reject(new HttpError("down", 503)));

		await expect(fail()).rejects.toThrow("down");
		await expect(fail()).rejects.toThrow("down");
		expect(breaker.state).toBe("closed");
		// a success resets the consecutive count
		await breaker.execute(async () => "ok");
		await expect(fail()).rejects.toThrow("down");
		await expect(fail()).rejects.toThrow("down");
		await expect(fail()).rejects.toThrow("down");
		expect(breaker.state).toBe("open");
		await expect(breaker.execute(async () => "never")).rejects.toBeInstanceOf(CircuitOpenError);

		now = 999;
		expect(breaker.state).toBe("open");
		now = 1_000;
		expect(breaker.state).toBe("half_open");

		// only one probe at a time in half-open
		let releaseProbe!: () => void;
		const probe = breaker.execute(
			() =>
				new Promise<string>((resolve) => {
					releaseProbe = () => resolve("probe-ok");
				}),
		);
		await expect(breaker.execute(async () => "second")).rejects.toBeInstanceOf(CircuitOpenError);
		releaseProbe();
		await expect(probe).resolves.toBe("probe-ok");
		expect(breaker.state).toBe("closed");
		expect(changes).toEqual(["closed->open", "open->half_open", "half_open->closed"]);
	});

	it("re-opens when the half-open probe fails, and ignores non-tripping (4xx) and aborted outcomes", async () => {
		let now = 0;
		const breaker = new CircuitBreaker("openai", { failureThreshold: 1, cooldownMs: 500, now: () => now });
		await expect(breaker.execute(async () => Promise.reject(new HttpError("down", 500)))).rejects.toThrow();
		expect(breaker.state).toBe("open");
		now = 500;
		expect(breaker.state).toBe("half_open");
		await expect(breaker.execute(async () => Promise.reject(new HttpError("down", 500)))).rejects.toThrow();
		expect(breaker.state).toBe("open");

		now = 1_000;
		// probe answered with a 400: neither success nor failure, so the circuit stays half-open but the
		// probe slot is freed for the next call, which then closes it.
		await expect(breaker.execute(async () => Promise.reject(new HttpError("bad request", 400)))).rejects.toThrow();
		expect(breaker.state).toBe("half_open");
		await expect(breaker.execute(async () => "ok")).resolves.toBe("ok");
		expect(breaker.state).toBe("closed");

		const abortable = new CircuitBreaker("p", { failureThreshold: 1, cooldownMs: 10, now: () => now });
		await expect(
			abortable.execute(async () => Promise.reject(Object.assign(new Error("aborted"), { name: "AbortError" }))),
		).rejects.toThrow();
		expect(abortable.state).toBe("closed");
	});

	it("ignores results of calls that began before the current state", async () => {
		let now = 0;
		const breaker = new CircuitBreaker("p", { failureThreshold: 1, cooldownMs: 100, now: () => now });
		let resolveSlow!: (value: string) => void;
		const slow = breaker.execute(
			() =>
				new Promise<string>((resolve) => {
					resolveSlow = resolve;
				}),
		);
		// meanwhile another call trips the breaker, the cooldown elapses and a probe is in flight
		await expect(breaker.execute(async () => Promise.reject(new HttpError("down", 500)))).rejects.toThrow();
		expect(breaker.state).toBe("open");
		now = 100;
		expect(breaker.state).toBe("half_open");
		let resolveProbe!: (value: string) => void;
		const probe = breaker.execute(
			() =>
				new Promise<string>((resolve) => {
					resolveProbe = resolve;
				}),
		);
		// the slow pre-trip call succeeding must NOT close the circuit or free the probe slot
		resolveSlow("late");
		await expect(slow).resolves.toBe("late");
		expect(breaker.state).toBe("half_open");
		await expect(breaker.execute(async () => "second")).rejects.toBeInstanceOf(CircuitOpenError);
		resolveProbe("probe-ok");
		await expect(probe).resolves.toBe("probe-ok");
		expect(breaker.state).toBe("closed");
	});

	it("non-retryable errors neither reset the failure count nor count as success", async () => {
		const breaker = new CircuitBreaker("p", { failureThreshold: 3, cooldownMs: 100, now: () => 0 });
		await expect(breaker.execute(async () => Promise.reject(new HttpError("down", 500)))).rejects.toThrow();
		await expect(breaker.execute(async () => Promise.reject(new HttpError("down", 500)))).rejects.toThrow();
		expect(breaker.failures).toBe(2);
		await expect(breaker.execute(async () => Promise.reject(new HttpError("bad request", 400)))).rejects.toThrow();
		expect(breaker.failures).toBe(2);
		await expect(breaker.execute(async () => Promise.reject(new HttpError("down", 500)))).rejects.toThrow();
		expect(breaker.state).toBe("open");
	});

	it("does not wedge on a hung half-open probe (e.g. timeoutMs=0): the slot is reclaimed after probeTimeoutMs", async () => {
		let now = 0;
		const breaker = new CircuitBreaker("p", { failureThreshold: 1, cooldownMs: 100, probeTimeoutMs: 1_000, now: () => now });
		await expect(breaker.execute(async () => Promise.reject(new HttpError("down", 500)))).rejects.toThrow();
		now = 100;
		void breaker.execute(() => new Promise<string>(() => undefined)); // hangs forever
		await expect(breaker.execute(async () => "x")).rejects.toBeInstanceOf(CircuitOpenError);
		now = 1_200;
		await expect(breaker.execute(async () => "recovered")).resolves.toBe("recovered");
		expect(breaker.state).toBe("closed");
	});

	it("keeps one breaker per provider", () => {
		const registry = new ProviderCircuitBreakers({ failureThreshold: 1 });
		expect(registry.get("a")).toBe(registry.get("a"));
		expect(registry.get("a")).not.toBe(registry.get("b"));
		expect(registry.snapshot()).toEqual({ a: { state: "closed", failures: 0 }, b: { state: "closed", failures: 0 } });
	});
});

// ─── Fallback chain ──────────────────────────────────────────────────────────

describe("runWithFallbacks", () => {
	it("falls back to the next candidate when the first is exhausted and reports events", async () => {
		const { sleeps, deps } = fastDeps();
		const events: ResilienceEvent[] = [];
		const candidates: ModelCandidate<string>[] = [
			{ key: "a:one", provider: "a", run: async () => Promise.reject(new HttpError("down", 503)) },
			{ key: "b:two", provider: "b", run: async () => "from-b" },
		];
		const result = await runWithFallbacks(candidates, {
			timeoutMs: 1_000,
			maxRetries: 1,
			baseDelayMs: 10,
			...deps,
			onEvent: (e) => events.push(e),
		});
		expect(result).toBe("from-b");
		expect(sleeps).toHaveLength(1);
		expect(events.map((e) => e.type)).toEqual(["retry", "fallback"]);
	});

	it("skips providers whose circuit is open and throws AllModelsFailedError when nothing works", async () => {
		const { deps } = fastDeps();
		const breakers = new ProviderCircuitBreakers({ failureThreshold: 1, cooldownMs: 60_000 });
		let aCalls = 0;
		const candidates: ModelCandidate<string>[] = [
			{
				key: "a:one",
				provider: "a",
				run: async () => {
					aCalls += 1;
					throw new HttpError("down", 500);
				},
			},
		];
		await expect(
			runWithFallbacks(candidates, { timeoutMs: 100, maxRetries: 3, baseDelayMs: 1, breakers, ...deps }),
		).rejects.toBeInstanceOf(AllModelsFailedError);
		// threshold 1: the first failure opens the circuit, so retries are short-circuited
		expect(aCalls).toBe(1);
		await expect(
			runWithFallbacks(candidates, { timeoutMs: 100, maxRetries: 3, baseDelayMs: 1, breakers, ...deps }),
		).rejects.toThrow(/circuit_open/);
		expect(aCalls).toBe(1);
	});

	it("times out a hung candidate (fake timers) and moves on", async () => {
		vi.useFakeTimers();
		try {
			const { deps } = fastDeps();
			const hung: ModelCandidate<string> = { key: "a:hung", provider: "a", run: () => new Promise(() => undefined) };
			const good: ModelCandidate<string> = { key: "b:ok", provider: "b", run: async () => "ok" };
			const promise = runWithFallbacks([hung, good], { timeoutMs: 5_000, maxRetries: 0, baseDelayMs: 1, ...deps });
			await vi.advanceTimersByTimeAsync(5_000);
			await expect(promise).resolves.toBe("ok");
		} finally {
			vi.useRealTimers();
		}
	});
});

// ─── Fast model client ───────────────────────────────────────────────────────

describe("createFastModelClient", () => {
	it("retries a failing-then-succeeding fast model without real waiting", async () => {
		const { models, calls } = fakeModels({
			"p:fast": [errorReply("Overloaded, please retry"), new HttpError("unavailable", 503), okReply({ ok: true })],
		});
		const { sleeps, deps } = fastDeps();
		const events: ResilienceEvent[] = [];
		const client = createFastModelClient(
			models,
			{ provider: "p", modelId: "fast" },
			{ maxRetries: 3, retryBaseDelayMs: 50, deps, onEvent: (e) => events.push(e) },
		);
		const out = await client.completeJson<{ ok: boolean }>("sys", "user");
		expect(out.parsed).toEqual({ ok: true });
		expect(calls).toEqual(["p:fast", "p:fast", "p:fast"]);
		expect(sleeps).toHaveLength(2);
		expect(events.filter((e) => e.type === "retry")).toHaveLength(2);
	});

	it("falls back along the ordered chain and trips the breaker of the failing provider", async () => {
		const failing = () => errorReply("Service unavailable (503)");
		const { models, calls } = fakeModels({
			"a:fast": [failing(), failing(), failing(), failing()],
			"b:backup": [errorReply("invalid request"), okReply({ from: "b" }), okReply({ from: "b2" })],
			"c:last": [okReply({ from: "c" })],
		});
		const { deps } = fastDeps();
		const events: ResilienceEvent[] = [];
		const client = createFastModelClient(
			models,
			{ provider: "a", modelId: "fast" },
			{
				fallbacks: [
					{ provider: "b", modelId: "backup" },
					{ provider: "c", modelId: "last" },
				],
				maxRetries: 1,
				retryBaseDelayMs: 1,
				breaker: { failureThreshold: 2, cooldownMs: 60_000 },
				deps,
				onEvent: (e) => events.push(e),
			},
		);

		// call 1: a fails twice (opens), b permanent 4xx-style, c succeeds
		const first = await client.completeJson("s", "u");
		expect(first.parsed).toEqual({ from: "c" });
		expect(calls).toEqual(["a:fast", "a:fast", "b:backup", "c:last"]);
		expect(events.filter((e) => e.type === "fallback").map((e) => (e as { to: string }).to)).toEqual(["b:backup", "c:last"]);

		// call 2: a's circuit is open → skipped without a provider call, b serves it
		const second = await client.completeJson("s", "u");
		expect(second.parsed).toEqual({ from: "b" });
		expect(calls.slice(4)).toEqual(["b:backup"]);
	});

	it("rejects with AllModelsFailedError (not a silent empty result) when every candidate fails", async () => {
		const { models } = fakeModels({ "a:fast": [errorReply("rate limit reached")] }, ["a:fast"]);
		const { deps } = fastDeps();
		const client = createFastModelClient(
			models,
			{ provider: "a", modelId: "fast" },
			{ maxRetries: 0, deps, fallbacks: [{ provider: "z", modelId: "missing" }] },
		);
		const failure = await client.completeJson("s", "u").catch((e: unknown) => e);
		expect(failure).toBeInstanceOf(AllModelsFailedError);
		expect((failure as Error).message).toContain("not registered");
	});

	it("passes a per-attempt abort signal and honors caller abort", async () => {
		const controller = new AbortController();
		const { models } = fakeModels({
			"a:fast": [
				async () => {
					controller.abort();
					throw Object.assign(new Error("aborted"), { name: "AbortError" });
				},
			],
		});
		const client = createFastModelClient(models, { provider: "a", modelId: "fast" }, { maxRetries: 3, deps: fastDeps().deps });
		await expect(client.completeJson("s", "u", { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
	});
});

// ─── Model safety / settings / config ────────────────────────────────────────

describe("assertProductionModelSafety", () => {
	it("throws in production with mock models unless ALLOW_MOCK_MODELS", () => {
		const warn = vi.fn();
		expect(() => assertProductionModelSafety({ useMockModels: true }, { env: { NODE_ENV: "production" }, warn })).toThrow(
			MockModelsInProductionError,
		);
		expect(warn).not.toHaveBeenCalled();

		expect(() =>
			assertProductionModelSafety({ useMockModels: true }, { env: { NODE_ENV: "production", ALLOW_MOCK_MODELS: "true" }, warn }),
		).not.toThrow();
		expect(warn).toHaveBeenCalledTimes(1);
		expect(warn.mock.calls[0]?.[0]).toMatch(/MOCK MODELS ARE ACTIVE IN PRODUCTION/);

		assertProductionModelSafety({ useMockModels: true, allowMockModels: true }, { env: { NODE_ENV: "production" }, warn });
		expect(warn).toHaveBeenCalledTimes(2);
	});

	it("warns loudly outside production and is silent with real models", () => {
		const warn = vi.fn();
		assertProductionModelSafety({ useMockModels: true }, { env: { NODE_ENV: "development" }, warn });
		expect(warn).toHaveBeenCalledTimes(1);
		assertProductionModelSafety({ useMockModels: false }, { env: { NODE_ENV: "production" }, warn });
		expect(warn).toHaveBeenCalledTimes(1);
	});
});

describe("config + harness settings", () => {
	it("parses budget / resilience env vars with safe defaults", () => {
		const defaults = loadConfig({});
		expect(defaults).toMatchObject({
			maxConversationTokens: 0,
			maxConversationCostUsd: 0,
			maxTenantDailyTokens: 0,
			maxTenantDailyCostUsd: 0,
			defaultMaxTurns: 24,
			modelTimeoutMs: 120_000,
			fastModelTimeoutMs: 30_000,
			allowMockModels: false,
			fastModelFallbacks: [],
		});

		const config = loadConfig({
			MAX_CONVERSATION_TOKENS: "5000",
			MAX_CONVERSATION_COST_USD: "1.5",
			MAX_TENANT_DAILY_TOKENS: "100000",
			MAX_TENANT_DAILY_COST_USD: "20",
			DEFAULT_MAX_TURNS: "8",
			MODEL_TIMEOUT_MS: "90000",
			MODEL_MAX_RETRIES: "5",
			FAST_MODEL_FALLBACKS: "openai:gpt-4.1-mini, google:gemini-2.5-flash",
			ALLOW_MOCK_MODELS: "true",
		});
		expect(config).toMatchObject({
			maxConversationTokens: 5000,
			maxConversationCostUsd: 1.5,
			maxTenantDailyTokens: 100_000,
			maxTenantDailyCostUsd: 20,
			defaultMaxTurns: 8,
			modelTimeoutMs: 90_000,
			modelMaxRetries: 5,
			allowMockModels: true,
		});
		expect(config.fastModelFallbacks).toEqual([
			{ provider: "openai", modelId: "gpt-4.1-mini" },
			{ provider: "google", modelId: "gemini-2.5-flash" },
		]);
		expect(() => loadConfig({ FAST_MODEL_FALLBACKS: "just-a-model" })).toThrow(/Invalid model spec/);
	});

	it("maps main-model settings onto Pi Durable HarnessSettings (no wrapper retry)", () => {
		const settings = buildHarnessSettings({
			modelTimeoutMs: 90_000,
			modelMaxRetries: 0,
			modelRetryBaseDelayMs: 1_000,
			modelProviderMaxRetries: 1,
		});
		expect(settings.stream).toEqual({ timeoutMs: 90_000, maxRetries: 1 });
		expect(settings.retry).toEqual({ enabled: false, maxRetries: 0, baseDelayMs: 1_000 });
	});
});

// ─── Budget primitives ───────────────────────────────────────────────────────

describe("budget primitives", () => {
	it("evaluateBudget treats 0 as unlimited and blocks once a limit is reached", () => {
		const none = { maxConversationTokens: 0, maxConversationCostUsd: 0, maxTenantDailyTokens: 0, maxTenantDailyCostUsd: 0 };
		expect(evaluateBudget({ limits: none, conversation: { tokens: 1e9, costUsd: 1e9 }, tenantDaily: { tokens: 1e9, costUsd: 1e9 } })).toBeUndefined();

		const limits = { ...none, maxConversationTokens: 100, maxTenantDailyCostUsd: 2 };
		expect(evaluateBudget({ limits, conversation: { tokens: 99, costUsd: 0 } })).toBeUndefined();
		expect(evaluateBudget({ limits, conversation: { tokens: 100, costUsd: 0 } })).toMatchObject({
			code: "conversation_tokens",
			scope: "conversation",
		});
		expect(evaluateBudget({ limits, conversation: { tokens: 1, costUsd: 0 }, tenantDaily: { tokens: 0, costUsd: 2 } })).toMatchObject({
			code: "tenant_daily_cost_usd",
		});
	});

	it("budget block messages are never retried and never look like a context overflow", () => {
		for (const [scope, metric, limit, used] of [
			["conversation", "tokens", 500, 500],
			["conversation", "cost_usd", 1, 1.2],
			["tenant_daily", "tokens", 429, 503],
			["tenant_daily", "cost_usd", 5, 5.5],
		] as const) {
			const violation = buildViolation(scope, metric, limit, used);
			const message = { role: "assistant", stopReason: "error", errorMessage: violation.message, content: [] } as unknown as AssistantMessage;
			expect(isRetryableAssistantError(message)).toBe(false);
			expect(isContextOverflow(message)).toBe(false);
			expect(classifyError(new Error(violation.message)).retryable).toBe(false);
		}
	});

	it("BudgetStore keeps a per-tenant, per-UTC-day ledger and an event log", () => {
		const db = new DatabaseSync(":memory:");
		let now = Date.parse("2026-03-01T23:00:00Z");
		const store = new BudgetStore(db, { now: () => now });
		store.recordUsage("t1", 100, 0.5);
		store.recordUsage("t1", 50, 0.25);
		store.recordUsage("t2", 7, 0.01);
		expect(store.getTenantDaily("t1")).toEqual({ tokens: 150, costUsd: 0.75 });
		expect(store.getTenantDaily("t2")).toEqual({ tokens: 7, costUsd: 0.01 });
		now = Date.parse("2026-03-02T00:00:01Z");
		expect(store.getTenantDaily("t1")).toEqual({ tokens: 0, costUsd: 0 });

		store.recordEvent({ tenantId: "t1", conversationId: "1", violation: buildViolation("conversation", "tokens", 10, 12) });
		expect(store.listEvents()).toMatchObject([{ tenantId: "t1", code: "conversation_tokens", limit: 10, used: 12 }]);
		// idempotent schema creation on a second store over the same db
		expect(() => new BudgetStore(db)).not.toThrow();
		expect(usageCostUsd({ cost: { total: 0.3 } })).toBe(0.3);
	});
});

// ─── Harness integration (mock models, in-memory DBs) ────────────────────────

const TENANT: TenantContext = { tenantId: "tenant-budget", userId: "user-1", permissions: [...DEFAULT_PERMISSIONS] };
const OTHER_TENANT: TenantContext = { tenantId: "tenant-other", userId: "user-2", permissions: [...DEFAULT_PERMISSIONS] };

describe("budget enforcement in the harness", () => {
	const runtimes: BootstrappedHarness[] = [];

	async function boot(
		config: Parameters<typeof bootstrapHarness>[0] extends infer O ? (O extends { config?: infer C } ? C : never) : never,
		extra: { onDiagnostic?: (d: BudgetDiagnostic) => void; estimateCostUsd?: (u: { totalTokens?: number }) => number } = {},
	): Promise<BootstrappedHarness> {
		const runtime = await bootstrapHarness({
			config: {
				piDbPath: ":memory:",
				appDbPath: ":memory:",
				useMockModels: true,
				skillsDir: path.join(os.tmpdir(), "helios-budget-test-no-skills"),
				...config,
			},
			sleepCycleScheduler: false,
			budget: { onDiagnostic: extra.onDiagnostic ?? (() => undefined), estimateCostUsd: extra.estimateCostUsd },
		});
		runtimes.push(runtime);
		return runtime;
	}

	afterEach(async () => {
		for (const runtime of runtimes.splice(0)) {
			await runtime.close();
		}
	});

	it("is a no-op with the default (unlimited) config", async () => {
		const runtime = await boot({});
		const conv = await runtime.harness.createConversation({ tenant: TENANT });
		for (const prompt of ["one", "two", "three"]) {
			const settled = await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: prompt });
			expect(settled.status).toBe("done");
		}
	});

	it("blocks the next generation once the per-conversation token budget is reached, without calling the provider", async () => {
		const diagnostics: BudgetDiagnostic[] = [];
		const runtime = await boot({ maxConversationTokens: 1 }, { onDiagnostic: (d) => diagnostics.push(d) });
		const conv = await runtime.harness.createConversation({ tenant: TENANT });

		const first = await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "first" });
		expect(first.status).toBe("done");
		expect(runtime.testModel.getRecordedRequests({ modelId: "main" })).toHaveLength(1);

		const second = await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "second" });
		expect(second.status).toBe("unanswered");
		expect(second).toMatchObject({ reason: "model_error" });
		expect(String((second as { detail?: unknown }).detail)).toContain("BUDGET_EXCEEDED conversation_tokens");
		// the veto happened before the provider: no second recorded request, and no retries
		expect(runtime.testModel.getRecordedRequests({ modelId: "main" })).toHaveLength(1);

		expect(diagnostics).toHaveLength(1);
		expect(diagnostics[0]).toMatchObject({ type: "budget_blocked", code: "conversation_tokens", tenantId: TENANT.tenantId });
		expect(runtime.budgetStore.listEvents()).toMatchObject([{ code: "conversation_tokens", conversationId: conv.conversationId }]);

		// a different conversation of the same tenant is unaffected by the per-conversation cap
		const other = await runtime.harness.createConversation({ tenant: TENANT });
		const fresh = await runtime.harness.submitAndWait({ conversationId: other.conversationId, content: "hello" });
		expect(fresh.status).toBe("done");
	});

	it("blocks on the per-conversation cost budget using an injected price table", async () => {
		const runtime = await boot(
			{ maxConversationCostUsd: 0.01 },
			{ estimateCostUsd: (usage) => (usage.totalTokens ?? 0) * 0.001 },
		);
		const conv = await runtime.harness.createConversation({ tenant: TENANT });
		expect((await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "first" })).status).toBe("done");
		const blocked = await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "second" });
		expect(blocked).toMatchObject({ status: "unanswered", reason: "model_error" });
		expect(String((blocked as { detail?: unknown }).detail)).toContain("conversation_cost_usd");
	});

	it("enforces the per-tenant daily cap across conversations and isolates tenants", async () => {
		const runtime = await boot({ maxTenantDailyTokens: 1 });
		const a1 = await runtime.harness.createConversation({ tenant: TENANT });
		expect((await runtime.harness.submitAndWait({ conversationId: a1.conversationId, content: "spend" })).status).toBe("done");
		expect(runtime.budgetStore.getTenantDaily(TENANT.tenantId).tokens).toBeGreaterThan(0);

		const a2 = await runtime.harness.createConversation({ tenant: TENANT });
		const blocked = await runtime.harness.submitAndWait({ conversationId: a2.conversationId, content: "more" });
		expect(blocked).toMatchObject({ status: "unanswered", reason: "model_error" });
		expect(String((blocked as { detail?: unknown }).detail)).toContain("tenant_daily_tokens");

		const b1 = await runtime.harness.createConversation({ tenant: OTHER_TENANT });
		expect((await runtime.harness.submitAndWait({ conversationId: b1.conversationId, content: "hi" })).status).toBe("done");
	});

	it("applies DEFAULT_MAX_TURNS when createConversation passes none, and lets an explicit maxTurns win", async () => {
		const runtime = await boot({ defaultMaxTurns: 2 });
		const toolRound = () => ({
			type: "tool_calls" as const,
			calls: [{ name: "memory_search", arguments: { query: "anything" } }],
		});

		const capped = await runtime.harness.createConversation({ tenant: TENANT });
		runtime.testModel.enqueueMainMany([toolRound(), toolRound(), toolRound(), { type: "text", text: "finished" }]);
		const hit = await runtime.harness.submitAndWait({ conversationId: capped.conversationId, content: "loop" });
		expect(hit).toMatchObject({ status: "unanswered", reason: "turn_limit" });

		runtime.testModel.reset();
		const explicit = await runtime.harness.createConversation({ tenant: TENANT, maxTurns: 6 });
		runtime.testModel.enqueueMainMany([toolRound(), toolRound(), toolRound(), { type: "text", text: "finished" }]);
		const ok = await runtime.harness.submitAndWait({ conversationId: explicit.conversationId, content: "loop" });
		expect(ok).toMatchObject({ status: "done", answerText: "finished" });
	});
});
