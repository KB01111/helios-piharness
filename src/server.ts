import type { Server } from "node:http";
import { serve, type ServerType } from "@hono/node-server";
import { loadConfig } from "./config.js";
import { createHealthChecker, createReadinessHandler } from "./health.js";
import { bootstrapHarness, type BootstrappedHarness } from "./index.js";
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

	bootstrapping = bootstrapHarness({ config });
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

main().catch((error: unknown) => {
	createLogger({ base: APP_BASE }).error("startup_failed", { error: serializeError(error) });
	process.exit(1);
});
