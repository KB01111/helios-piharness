/**
 * OpenAI-compatible embedding HTTP API over an `EmbeddingRouter`.
 *
 *   GET  /health          {status, space, model, dimensions, backends[]}   200 when any backend is healthy, else 503
 *   GET  /v1/models       OpenAI model list (+ `space`, `dimensions`, `calibration`, `matryoshka_dimensions`)
 *   POST /v1/embeddings   OpenAI request body plus optional `input_type: "query" | "document"`
 *                         (aliases: `search_query`, `search_document`, `passage`); default `document`.
 *
 * The `model` field of a request is informational: the server always answers with its configured space and
 * reports it back as `model` (`<space>@<dims>`), so clients can detect a mismatched embedding space.
 */

import { createHash, timingSafeEqual } from "node:crypto";
import { Hono } from "hono";
import { EmbeddingRouter } from "./router.js";
import { EmbeddingError, type EmbeddingInputType } from "./types.js";

export interface EmbedHttpServerOptions {
	readonly apiKeys?: readonly string[];
	readonly maxInputs?: number;
	readonly maxBodyBytes?: number;
	readonly onEvent?: (event: string, fields: Record<string, unknown>) => void;
}

function parseInputType(raw: unknown): EmbeddingInputType | undefined {
	if (raw === undefined || raw === null) return undefined;
	if (typeof raw !== "string") throw new EmbeddingError("BAD_REQUEST", "input_type must be a string");
	const v = raw.trim().toLowerCase();
	if (v === "query" || v === "search_query" || v === "retrieval_query") return "query";
	if (v === "document" || v === "search_document" || v === "passage" || v === "retrieval_document") return "document";
	throw new EmbeddingError("BAD_REQUEST", `input_type must be "query" or "document"`);
}

function digest(token: string): Buffer {
	return createHash("sha256").update(token).digest();
}

export function createEmbedHttpServer(router: EmbeddingRouter, options: EmbedHttpServerOptions = {}): Hono {
	const app = new Hono();
	const keyDigests = (options.apiKeys ?? []).map(digest);
	const maxInputs = options.maxInputs ?? 256;
	const maxBodyBytes = options.maxBodyBytes ?? 4 * 1024 * 1024;
	const onEvent = options.onEvent ?? (() => undefined);

	const errorBody = (message: string, type: string) => ({ error: { message, type } });

	app.onError((err, c) => {
		if (err instanceof EmbeddingError) {
			if (err.status >= 500) onEvent("embedding_request_failed", { code: err.code, error: err.message });
			return c.json(errorBody(err.message, err.code.toLowerCase()), err.status as 400 | 500 | 502 | 503 | 504);
		}
		onEvent("embedding_request_crashed", { error: err instanceof Error ? err.message : String(err) });
		return c.json(errorBody("internal error", "internal"), 500);
	});

	app.use("/v1/*", async (c, next) => {
		if (keyDigests.length === 0) return next();
		const header = c.req.header("authorization") ?? "";
		const match = /^Bearer\s+(.+)$/i.exec(header.trim());
		const presented = match ? digest(match[1]!.trim()) : undefined;
		let ok = false;
		if (presented) for (const d of keyDigests) if (timingSafeEqual(d, presented)) ok = true;
		if (!ok) return c.json(errorBody("missing or invalid bearer token", "unauthorized"), 401);
		return next();
	});

	app.get("/health", async (c) => {
		const health = await router.health();
		return c.json(
			{
				status: health.ok ? "ok" : "unavailable",
				space: router.space,
				model: router.modelId(),
				dimensions: router.defaultDimensions ?? router.info.nativeDimensions ?? null,
				backends: health.backends,
			},
			health.ok ? 200 : 503,
		);
	});

	app.get("/v1/models", (c) =>
		c.json({
			object: "list",
			data: [
				{
					id: router.modelId(),
					object: "model",
					owned_by: "helios-embed",
					space: router.space,
					dimensions: router.defaultDimensions ?? router.info.nativeDimensions ?? null,
					native_dimensions: router.info.nativeDimensions ?? null,
					matryoshka_dimensions: router.info.matryoshkaDimensions,
					calibration: router.calibration ?? null,
					backends: router.backendIds,
				},
			],
		}),
	);

	app.post("/v1/embeddings", async (c) => {
		const declared = Number(c.req.header("content-length") ?? "0");
		if (declared > maxBodyBytes) return c.json(errorBody(`request body exceeds ${maxBodyBytes} bytes`, "bad_request"), 413);
		const raw = await c.req.text();
		if (Buffer.byteLength(raw) > maxBodyBytes) return c.json(errorBody(`request body exceeds ${maxBodyBytes} bytes`, "bad_request"), 413);
		let body: Record<string, unknown>;
		try {
			const parsed = JSON.parse(raw) as unknown;
			if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("not an object");
			body = parsed as Record<string, unknown>;
		} catch {
			throw new EmbeddingError("BAD_REQUEST", "request body must be a JSON object");
		}

		const input = body.input;
		const texts = typeof input === "string" ? [input] : Array.isArray(input) ? input : undefined;
		if (!texts || texts.length === 0) throw new EmbeddingError("BAD_REQUEST", "input must be a non-empty string or array of strings");
		if (texts.length > maxInputs) throw new EmbeddingError("BAD_REQUEST", `at most ${maxInputs} inputs per request`);
		if (!texts.every((t): t is string => typeof t === "string")) {
			throw new EmbeddingError("BAD_REQUEST", "token-array inputs are not supported; send strings");
		}
		const dimensions = body.dimensions;
		if (dimensions !== undefined && (typeof dimensions !== "number" || !Number.isInteger(dimensions))) {
			throw new EmbeddingError("BAD_REQUEST", "dimensions must be an integer");
		}
		const encoding = body.encoding_format ?? "float";
		if (encoding !== "float" && encoding !== "base64") throw new EmbeddingError("BAD_REQUEST", `encoding_format must be "float" or "base64"`);
		const inputType = parseInputType(body.input_type ?? body.task_type);

		const started = performance.now();
		const result = await router.embed({
			texts,
			...(inputType ? { inputType } : {}),
			...(typeof dimensions === "number" ? { dimensions } : {}),
			signal: c.req.raw.signal,
		});
		onEvent("embedding_request", {
			inputs: texts.length,
			inputType: inputType ?? "document",
			dimensions: result.dimensions,
			cacheHits: result.cacheHits,
			servedBy: result.servedBy,
			ms: Math.round(performance.now() - started),
		});

		return c.json({
			object: "list",
			model: router.modelId(typeof dimensions === "number" ? dimensions : undefined),
			data: result.vectors.map((vec, index) => ({
				object: "embedding",
				index,
				embedding: encoding === "base64" ? Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength).toString("base64") : Array.from(vec),
			})),
			usage: { prompt_tokens: result.promptTokensEstimate, total_tokens: result.promptTokensEstimate },
		});
	});

	return app;
}
