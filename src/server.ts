import type { Server } from "node:http";
import { serve, type ServerType } from "@hono/node-server";
import { loadConfig } from "./config.js";
import type { EmbeddingProvider, FileSearchService, Reranker } from "./contracts/app-capabilities.js";
import { HttpEmbeddingProvider, HttpReranker } from "./embeddings/provider.js";
import { createHealthChecker, createReadinessHandler } from "./health.js";
import {
	bootstrapHarness,
	FallbackFileSearchService,
	LocalWorkspaceFileSearchService,
	XerjFileSearchService,
	type BootstrappedHarness,
} from "./index.js";
import { createLogger, serializeError } from "./logger.js";
import { assertProductionModelSafety } from "./runtime/resilience.js";
import { authFromEnv } from "./transport/auth.js";
import { createHttpServer } from "./transport/http-server.js";

const APP_BASE = { service: "helios-piharness" } as const;

/** Hard upper bound on graceful shutdown before the process exits anyway. */
const SHUTDOWN_TIMEOUT_MS = 15_000;

type DrainableServer = Partial<Pick<Server, "closeIdleConnections" | "closeAllConnections">>;

async function main(): Promise<void> {
	const config = loadConfig();
	const logger = createLogger({ level: config.logLevel, base: APP_BASE });

	// Throws (and aborts startup) when mock models would serve production traffic.
	assertProductionModelSafety(
		{ useMockModels: config.useMockModels },
		{ warn: (message, fields) => logger.warn("mock_models_active", { message, ...fields }) },
	);

	// Resolved before bootstrap so a bad auth configuration fails fast without ever taking the writer lock.
	const auth = authFromEnv(process.env, logger);

	let runtime: BootstrappedHarness | undefined;
	let bootstrapping: Promise<BootstrappedHarness> | undefined;
	let server: ServerType | undefined;
	let shutdownStarted = false;

	const shutdown = (reason: string, exitCode: number): void => {
		if (shutdownStarted) {
			logger.warn("shutdown_already_in_progress", { reason });
			return;
		}
		shutdownStarted = true;
		logger.info("shutdown_started", { reason });

		setTimeout(() => {
			logger.error("shutdown_timed_out", { reason, timeoutMs: SHUTDOWN_TIMEOUT_MS });
			process.exit(exitCode === 0 ? 1 : exitCode);
		}, SHUTDOWN_TIMEOUT_MS);

		const current = server;
		const drainable = current as unknown as DrainableServer | undefined;
		const stopped = new Promise<void>((resolve) => {
			if (!current) return resolve();
			current.close(() => resolve());
		});
		drainable?.closeIdleConnections?.();

		void (async () => {
			// A signal that lands mid-bootstrap must wait for it to settle, then release the writer lock it took.
			const booted = await bootstrapping?.catch(() => undefined);
			await (runtime ?? booted)?.close();
			// Long-lived SSE streams keep sockets open; drop them once the harness is closed.
			drainable?.closeAllConnections?.();
			await stopped;
		})().then(
			() => {
				logger.info("shutdown_complete", { reason, exitCode });
				process.exit(exitCode);
			},
			(error: unknown) => {
				logger.error("shutdown_failed", { reason, error: serializeError(error) });
				process.exit(1);
			},
		);
	};

	process.once("SIGINT", () => shutdown("SIGINT", 0));
	process.once("SIGTERM", () => shutdown("SIGTERM", 0));
	process.on("unhandledRejection", (reason: unknown) => {
		logger.error("unhandled_rejection", { error: serializeError(reason) });
		shutdown("unhandledRejection", 1);
	});
	process.on("uncaughtException", (error: Error) => {
		logger.error("uncaught_exception", { error: serializeError(error) });
		shutdown("uncaughtException", 1);
	});

	const embeddingProvider = await connectEmbeddings(config, logger);
	const reranker = await connectReranker(config, logger, embeddingProvider);
	const fileSearch = await connectFileSearch(config, logger);
	bootstrapping = bootstrapHarness({
		config,
		...(embeddingProvider ? { embeddingProvider } : {}),
		...(reranker ? { reranker } : {}),
		...(fileSearch ? { fileSearch } : {}),
	});
	runtime = await bootstrapping;
	if (shutdownStarted) {
		// Shutdown began while bootstrapping; it is closing the runtime, so do not start serving.
		return;
	}
	const activeRuntime = runtime;
	const health = createHealthChecker({
		harness: activeRuntime.harness,
		appDbPing: () => {
			activeRuntime.appDb.db.prepare("SELECT 1").get();
		},
		piDbPath: config.piDbPath,
		useMockModels: config.useMockModels,
	});
	const readiness = createReadinessHandler({ health, auth });
	const app = createHttpServer(activeRuntime.harness, {
		logger,
		auth,
		...(activeRuntime.improvements ? { improvements: activeRuntime.improvements } : {}),
		publicPaths: ["/health", "/ready"],
		registerPublicRoutes: (hono) => {
			// Public verdict only ({status}); version/check details require an authenticated request.
			hono.get("/ready", async (c) => {
				const { httpStatus, body } = await readiness(c.req.raw);
				return c.json(body, httpStatus);
			});
		},
	});

	server = serve(
		{
			fetch: app.fetch,
			hostname: config.host,
			port: config.port,
		},
		(info) => {
			logger.info("server_started", {
				url: `http://${info.address}:${info.port}`,
				mainModel: `${config.mainModel.provider}:${config.mainModel.modelId}`,
				fastModel: `${config.fastModel.provider}:${config.fastModel.modelId}`,
				useMockModels: config.useMockModels,
				piDbPath: config.piDbPath,
				appDbPath: config.appDbPath,
			});
		},
	);
}

/** Connects the optional embedding endpoint. Unreachable + not required => lexical-only memory with a warning. */
async function connectEmbeddings(
	config: ReturnType<typeof loadConfig>,
	logger: ReturnType<typeof createLogger>,
): Promise<EmbeddingProvider | undefined> {
	if (!config.embeddingsUrl) {
		if (config.embeddingsRequired) throw new Error("EMBEDDINGS_REQUIRED needs EMBEDDINGS_URL");
		return undefined;
	}
	try {
		const provider = await HttpEmbeddingProvider.connect({
			baseUrl: config.embeddingsUrl,
			...(config.embeddingsApiKey ? { apiKey: config.embeddingsApiKey } : {}),
			...(config.embeddingsModel ? { model: config.embeddingsModel } : {}),
			...(config.embeddingsDimensions ? { dimensions: config.embeddingsDimensions } : {}),
			timeoutMs: config.embeddingsTimeoutMs,
		});
		if (config.embeddingsRequired) await provider.embed(["startup probe"], { signal: AbortSignal.timeout(config.embeddingsTimeoutMs) });
		logger.info("embeddings_connected", {
			model: provider.model,
			heliosServer: provider.helios,
			...(provider.similarity ? { similarity: provider.similarity } : {}),
		});
		return provider;
	} catch (error) {
		if (config.embeddingsRequired) throw error;
		logger.warn("embeddings_unavailable", {
			message: "memory search runs lexical-only until restart",
			error: serializeError(error),
		});
		return undefined;
	}
}

/**
 * Connects the optional reranker: an explicit RERANK_URL (cloud: needs RERANK_MODEL), otherwise the Helios embedding
 * server's own reranker when it serves one. Failures only log: memory search works without a reranker.
 */
async function connectReranker(
	config: ReturnType<typeof loadConfig>,
	logger: ReturnType<typeof createLogger>,
	embeddings: EmbeddingProvider | undefined,
): Promise<Reranker | undefined> {
	if (!config.rerankEnabled) return undefined;
	const explicit = config.rerankUrl;
	const heliosServer = embeddings instanceof HttpEmbeddingProvider && embeddings.helios;
	const baseUrl = explicit ?? (heliosServer ? config.embeddingsUrl : undefined);
	if (!baseUrl) return undefined;
	try {
		const apiKey = config.rerankApiKey ?? (explicit ? undefined : config.embeddingsApiKey);
		const reranker = await HttpReranker.connect({
			baseUrl,
			...(apiKey ? { apiKey } : {}),
			...(config.rerankModel ? { model: config.rerankModel } : {}),
			timeoutMs: config.embeddingsTimeoutMs,
		});
		logger.info("reranker_connected", { model: reranker.model, answerThreshold: reranker.answerThreshold ?? null });
		return reranker;
	} catch (error) {
		// A Helios server running with EMBED_RERANK=off is a normal setup, not a warning.
		const log = explicit ? logger.warn.bind(logger) : logger.info.bind(logger);
		log("reranker_unavailable", { error: serializeError(error) });
		return undefined;
	}
}

/**
 * Connects optional indexed file search (`XERJ_URL` and/or `FILE_SEARCH_WORKSPACE_ROOT`).
 *
 * Security invariant: a single static `FILE_SEARCH_WORKSPACE_ROOT` has no per-tenant directory boundary,
 * so it is only attached when `XERJ_SCOPE_MODE=shared` (single-tenant / shared reference mode). In
 * multi-tenant modes (`tenant` / `workspace`), embedders should inject `LocalWorkspaceFileSearchService`
 * with `resolveWorkspaceRoot(scope)` programmatically rather than sharing one static directory across tenants.
 */
async function connectFileSearch(
	config: ReturnType<typeof loadConfig>,
	logger: ReturnType<typeof createLogger>,
): Promise<FileSearchService | undefined> {
	let local: LocalWorkspaceFileSearchService | undefined;
	if (config.fileSearchWorkspaceRoot) {
		if (config.xerjScopeMode === "shared") {
			local = new LocalWorkspaceFileSearchService({ workspaceRoot: config.fileSearchWorkspaceRoot });
		} else {
			logger.warn("file_search_local_root_skipped_multitenant", {
				message:
					"FILE_SEARCH_WORKSPACE_ROOT is ignored when XERJ_SCOPE_MODE is not 'shared' to prevent cross-tenant local file exposure. Set XERJ_SCOPE_MODE=shared for single-tenant use or supply a per-scope resolveWorkspaceRoot programmatically.",
				scopeMode: config.xerjScopeMode,
			});
		}
	}

	if (!config.xerjUrl) {
		if (config.xerjRequired) throw new Error("XERJ_REQUIRED needs XERJ_URL");
		if (local) {
			logger.info("file_search_connected", {
				engine: local.engine,
				deferred: config.fileSearchDeferred,
			});
		}
		return local;
	}

	const xerj = new XerjFileSearchService({
		baseUrl: config.xerjUrl,
		...(config.xerjApiKey ? { apiKey: config.xerjApiKey } : {}),
		indexPrefix: config.xerjIndexPrefix,
		scopeMode: config.xerjScopeMode,
		timeoutMs: config.xerjTimeoutMs,
	});

	const health = await xerj.healthCheck();
	if (health.ok) {
		if (config.xerjRequired) {
			// Verify search routes themselves work (not just `GET /`, which a reverse proxy could answer).
			await xerj.verifySearchCapability();
		}
		logger.info("file_search_connected", {
			engine: xerj.engine,
			scopeMode: config.xerjScopeMode,
			indexPrefix: config.xerjIndexPrefix,
			...(health.version ? { version: health.version } : {}),
			...(health.latencyMs !== undefined ? { latencyMs: health.latencyMs } : {}),
			fallback: local ? local.engine : "none",
			deferred: config.fileSearchDeferred,
		});
	} else {
		if (config.xerjRequired) {
			throw new Error(`XERJ_REQUIRED is set but XERJ node is unreachable: ${health.error ?? "unknown error"}`);
		}
		logger.warn("file_search_xerj_unavailable", {
			message: local
				? "XERJ unreachable at startup; falling back to local workspace search until XERJ recovers"
				: "XERJ unreachable at startup; file search tool calls will retry after circuit-breaker cooldown",
			scopeMode: config.xerjScopeMode,
			error: health.error,
		});
	}

	if (local) {
		return new FallbackFileSearchService({
			primary: xerj,
			fallback: local,
			onFallback: (operation, error) => {
				logger.warn("file_search_fallback", {
					operation,
					primary: xerj.engine,
					fallback: local.engine,
					error: serializeError(error),
				});
			},
		});
	}
	return xerj;
}

main().catch((error: unknown) => {
	createLogger({ base: APP_BASE }).error("startup_failed", { error: serializeError(error) });
	process.exit(1);
});
