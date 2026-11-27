import { EmbeddingError } from "../types.js";

/** Combines a caller signal with a timeout. The returned `dispose` must be called to clear the timer. */
export function withTimeout(signal: AbortSignal | undefined, timeoutMs: number): { signal: AbortSignal; dispose: () => void } {
	const controller = new AbortController();
	const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(new EmbeddingError("TIMEOUT", `embedding request timed out after ${timeoutMs}ms`)), timeoutMs) : undefined;
	const onAbort = (): void => controller.abort(signal?.reason);
	if (signal) {
		if (signal.aborted) controller.abort(signal.reason);
		else signal.addEventListener("abort", onAbort, { once: true });
	}
	return {
		signal: controller.signal,
		dispose: () => {
			if (timer) clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		},
	};
}

/** Stops waiting on an operation that may ignore cancellation (e.g. a shared model load). */
export async function waitWithSignal<T>(operation: Promise<T>, signal?: AbortSignal): Promise<T> {
	if (!signal) return operation;
	let onAbort!: () => void;
	const aborted = new Promise<never>((_, reject) => {
		onAbort = () => reject(signal.reason);
		if (signal.aborted) onAbort();
		else signal.addEventListener("abort", onAbort, { once: true });
	});
	try {
		return await Promise.race([operation, aborted]);
	} finally {
		signal.removeEventListener("abort", onAbort);
	}
}

/** POSTs JSON and returns the parsed body, mapping network/HTTP failures onto `EmbeddingError`. Never logs secrets. */
export async function postJson(
	url: string,
	body: unknown,
	options: { readonly headers?: Record<string, string>; readonly timeoutMs: number; readonly signal?: AbortSignal; readonly fetchImpl?: typeof fetch },
): Promise<unknown> {
	const { signal, dispose } = withTimeout(options.signal, options.timeoutMs);
	const fetchImpl = options.fetchImpl ?? fetch;
	try {
		let res: Response;
		try {
			res = await fetchImpl(url, {
				method: "POST",
				headers: { "content-type": "application/json", accept: "application/json", ...options.headers },
				body: JSON.stringify(body),
				signal,
			});
		} catch (error) {
			if (signal.aborted && signal.reason instanceof EmbeddingError) throw signal.reason;
			throw new EmbeddingError("UNAVAILABLE", `embedding backend unreachable at ${redactUrl(url)}: ${errorText(error)}`, { cause: error });
		}
		const text = await res.text();
		if (!res.ok) {
			const code = res.status === 400 || res.status === 413 || res.status === 422 ? "BAD_REQUEST" : res.status >= 500 || res.status === 429 ? "UNAVAILABLE" : "UPSTREAM";
			throw new EmbeddingError(code, `embedding backend ${redactUrl(url)} returned HTTP ${res.status}: ${text.slice(0, 300)}`);
		}
		try {
			return JSON.parse(text) as unknown;
		} catch (error) {
			throw new EmbeddingError("UPSTREAM", `embedding backend ${redactUrl(url)} returned invalid JSON`, { cause: error });
		}
	} finally {
		dispose();
	}
}

/** GET returning status only (health probes). */
export async function probe(url: string, options: { readonly headers?: Record<string, string>; readonly timeoutMs: number; readonly signal?: AbortSignal; readonly fetchImpl?: typeof fetch }): Promise<{ ok: boolean; status: number; detail?: string }> {
	const { signal, dispose } = withTimeout(options.signal, options.timeoutMs);
	try {
		const res = await (options.fetchImpl ?? fetch)(url, { headers: options.headers ?? {}, signal });
		await res.body?.cancel().catch(() => undefined);
		return { ok: res.ok, status: res.status };
	} catch (error) {
		return { ok: false, status: 0, detail: errorText(error) };
	} finally {
		dispose();
	}
}

/** Strips query strings (API keys in `?key=`) and userinfo from URLs shown in errors. */
export function redactUrl(url: string): string {
	try {
		const u = new URL(url);
		u.search = "";
		u.username = "";
		u.password = "";
		return u.toString();
	} catch {
		return "<invalid url>";
	}
}

export function errorText(error: unknown): string {
	if (error instanceof Error) {
		const cause = (error as { cause?: unknown }).cause;
		return cause instanceof Error ? `${error.message} (${cause.message})` : error.message;
	}
	return String(error);
}

export function toFloat32(values: unknown, where: string): Float32Array {
	if (!Array.isArray(values) || values.length === 0) {
		throw new EmbeddingError("UPSTREAM", `${where}: missing or empty embedding vector`);
	}
	const out = new Float32Array(values.length);
	for (let i = 0; i < values.length; i++) {
		const v = values[i];
		if (typeof v !== "number" || !Number.isFinite(v)) throw new EmbeddingError("UPSTREAM", `${where}: non-finite value at index ${i}`);
		out[i] = v;
	}
	return out;
}

/** Decodes an OpenAI `encoding_format: "base64"` little-endian float32 payload. */
export function base64ToFloat32(b64: string, where: string): Float32Array {
	const buf = Buffer.from(b64, "base64");
	if (buf.byteLength === 0 || buf.byteLength % 4 !== 0) throw new EmbeddingError("UPSTREAM", `${where}: malformed base64 embedding`);
	const out = new Float32Array(buf.byteLength / 4);
	for (let i = 0; i < out.length; i++) out[i] = buf.readFloatLE(i * 4);
	return out;
}
