import { GeminiBackend } from "./backends/gemini.js";
import { OnnxBackend } from "./backends/onnx.js";
import { OpenAICompatibleBackend } from "./backends/openai-compatible.js";
import type { EmbedServerConfig } from "./config.js";
import { FlmSupervisor } from "./flm-supervisor.js";
import { EmbeddingRouter } from "./router.js";
import { EmbeddingError, type EmbeddingBackend } from "./types.js";

export interface BuiltEmbeddingService {
	readonly router: EmbeddingRouter;
	readonly supervisor: FlmSupervisor | undefined;
	close(): Promise<void>;
}

/** Builds backends (and the optional FastFlowLM supervisor) from config. Starts nothing over the network. */
export function buildEmbeddingService(
	config: EmbedServerConfig,
	options: { readonly onEvent?: (event: string, fields: Record<string, unknown>) => void } = {},
): BuiltEmbeddingService {
	const space = config.space;
	let supervisor: FlmSupervisor | undefined;
	const backends: EmbeddingBackend[] = config.backends.map((name) => {
		switch (name) {
			case "npu": {
				if (config.npu.autostart) {
					const url = new URL(config.npu.url);
					supervisor = new FlmSupervisor({
						command: config.npu.flmCommand,
						chatModel: config.npu.chatModel,
						host: url.hostname,
						port: Number(url.port || 52625),
						powerMode: config.npu.powerMode,
						log: (event, fields) => options.onEvent?.(event, fields ?? {}),
					});
				}
				return new OpenAICompatibleBackend({
					id: "npu",
					kind: "npu",
					baseUrl: config.npu.url,
					model: config.npu.model,
					timeoutMs: config.npu.timeoutMs,
					// FastFlowLM runs one NPU job at a time; moderate batches keep latency predictable.
					maxBatchSize: 32,
					...(space ? { space } : {}),
					...(config.promptFamily ? { promptFamily: config.promptFamily } : {}),
				});
			}
			case "onnx":
				return new OnnxBackend({
					id: "onnx",
					model: config.onnx.model,
					device: config.onnx.device,
					dtype: config.onnx.dtype,
					offline: config.onnx.offline,
					...(config.onnx.cacheDir ? { cacheDir: config.onnx.cacheDir } : {}),
					...(space ? { space } : {}),
				});
			case "openai":
				return new OpenAICompatibleBackend({
					id: "openai",
					kind: "openai",
					baseUrl: config.openai.url,
					model: config.openai.model,
					...(config.openai.apiKey ? { apiKey: config.openai.apiKey } : {}),
					sendDimensions: config.openai.sendDimensions,
					timeoutMs: config.openai.timeoutMs,
					...(space ? { space } : {}),
					...(config.promptFamily ? { promptFamily: config.promptFamily } : {}),
				});
			case "gemini":
				if (!config.gemini.apiKey) throw new EmbeddingError("MISCONFIGURED", "gemini backend needs EMBED_GEMINI_API_KEY or GEMINI_API_KEY");
				return new GeminiBackend({
					id: "gemini",
					apiKey: config.gemini.apiKey,
					model: config.gemini.model,
					timeoutMs: config.gemini.timeoutMs,
					...(space ? { space } : {}),
				});
		}
	});

	const router = new EmbeddingRouter({
		backends,
		...(config.dimensions !== undefined ? { dimensions: config.dimensions } : {}),
		cacheSize: config.cacheSize,
		maxInputChars: config.maxInputChars,
		...(config.calibration ? { calibration: config.calibration } : {}),
		...(config.promptFamily ? { promptFamily: config.promptFamily } : {}),
		...(options.onEvent ? { onEvent: options.onEvent } : {}),
	});

	return {
		router,
		supervisor,
		async close() {
			await router.close();
			await supervisor?.stop();
		},
	};
}
