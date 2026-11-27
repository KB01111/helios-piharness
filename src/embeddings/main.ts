/**
 * `npm run embed:server` — standalone OpenAI-compatible embedding server for Helios memory.
 *
 * Examples:
 *   EMBED_PROFILE=gemma2 npm run embed:server                          # EmbeddingGemma 2, in-process ONNX (CPU)
 *   EMBED_PROFILE=npu EMBED_NPU_AUTOSTART=true npm run embed:server    # EmbeddingGemma on the Ryzen AI NPU (FastFlowLM)
 *   EMBED_PROFILE=openai OPENAI_API_KEY=... npm run embed:server       # cloud
 *
 * Point the harness at it with EMBEDDINGS_URL=http://127.0.0.1:18765/v1.
 */

import { serve } from "@hono/node-server";
import { createLogger, serializeError } from "../logger.js";
import { isLoopbackHost, loadEmbedServerConfig } from "./config.js";
import { buildEmbeddingService } from "./factory.js";
import { createEmbedHttpServer } from "./server.js";

const logger = createLogger({ base: { service: "helios-embed" } });

/** `--profile <name>` / `--npu-autostart` / `--port <n>` override the matching EMBED_* variables. */
function envWithArgs(argv: readonly string[], env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
	const out: NodeJS.ProcessEnv = { ...env };
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === "--profile" && argv[i + 1]) out.EMBED_PROFILE = argv[++i];
		else if (arg === "--port" && argv[i + 1]) out.EMBED_PORT = argv[++i];
		else if (arg === "--npu-autostart") out.EMBED_NPU_AUTOSTART = "true";
	}
	return out;
}

async function main(): Promise<void> {
	const config = loadEmbedServerConfig(envWithArgs(process.argv.slice(2), process.env));
	if (!isLoopbackHost(config.host) && config.apiKeys.length === 0 && !config.allowUnauthenticated) {
		throw new Error(
			`Refusing to bind the embedding server to non-loopback host ${config.host} without EMBED_API_KEYS ` +
				`(set EMBED_ALLOW_UNAUTHENTICATED=true to override).`,
		);
	}
	const service = buildEmbeddingService(config, { onEvent: (event, fields) => logger.info(event, fields) });

	if (service.supervisor) {
		const up = await service.supervisor.ensureRunning();
		if (!up) logger.warn("npu_backend_unavailable", { message: "FastFlowLM did not start; requests fall back to the next backend" });
	}
	// Quality self-test before serving: measures calibration and quarantines backends whose vectors cannot
	// separate related from unrelated text (it also warms in-process models).
	if (config.selfTest) {
		for (const entry of await service.router.selfTest({ enforce: config.selfTestEnforce })) {
			if (entry.error) logger.warn("embedding_self_test_skipped", { backend: entry.id, error: entry.error });
			else if (entry.result?.passed) logger.info("embedding_self_test", { backend: entry.id, ...entry.result });
			else logger.error("embedding_self_test", { backend: entry.id, ...entry.result });
		}
		logger.info("embedding_calibration", { space: service.router.space, calibration: service.router.calibration ?? null });
		if (service.reranker) {
			try {
				const rr = await service.reranker.selfTest({ enforce: config.selfTestEnforce });
				logger[rr.passed ? "info" : "error"]("rerank_self_test", { model: service.reranker.model, ...rr });
			} catch (error) {
				logger.warn("rerank_self_test_skipped", { model: service.reranker.model, error: serializeError(error) });
			}
		}
	}
	const app = createEmbedHttpServer(service.router, {
		apiKeys: config.apiKeys,
		maxInputs: config.maxInputs,
		maxBodyBytes: config.maxBodyBytes,
		...(service.reranker ? { reranker: service.reranker } : {}),
		onEvent: (event, fields) => (event === "embedding_request" ? logger.debug(event, fields) : logger.warn(event, fields)),
	});

	const server = serve({ fetch: app.fetch, hostname: config.host, port: config.port }, (info) => {
		logger.info("embed_server_started", {
			url: `http://${info.address}:${info.port}/v1`,
			space: service.router.space,
			model: service.router.modelId(),
			backends: service.router.backendIds,
			reranker: service.reranker?.model ?? null,
			auth: config.apiKeys.length > 0 ? "bearer" : "none",
		});
	});

	let stopping = false;
	const stop = (signal: string): void => {
		if (stopping) return;
		stopping = true;
		logger.info("embed_server_stopping", { signal });
		const force = setTimeout(() => process.exit(1), 10_000);
		force.unref();
		server.close(() => {
			void service.close().finally(() => process.exit(0));
		});
	};
	process.once("SIGINT", () => stop("SIGINT"));
	process.once("SIGTERM", () => stop("SIGTERM"));
}

main().catch((error: unknown) => {
	logger.error("embed_server_failed", { error: serializeError(error) });
	process.exit(1);
});
