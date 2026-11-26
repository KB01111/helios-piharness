/**
 * Environment configuration of the embedding server (`npm run embed:server`).
 *
 *   EMBED_BACKENDS        Ordered, comma-separated backends: `npu`, `onnx`, `openai`, `gemini` (default `onnx`).
 *                         All must produce the same embedding space; the first healthy one serves each request.
 *   EMBED_PROFILE         Shortcut that sets EMBED_BACKENDS + models when they are unset:
 *                           `gemma2`  -> onnx  onnx-community/embeddinggemma-2-ONNX (CPU)           [default]
 *                           `npu`     -> npu   FastFlowLM embed-gemma:300m on the Ryzen AI NPU (XDNA/XDNA2)
 *                           `openai`  -> openai text-embedding-3-small
 *                           `gemini`  -> gemini gemini-embedding-001
 *   EMBED_HOST / EMBED_PORT            Listen address (default 127.0.0.1:18765)
 *   EMBED_API_KEYS                     Comma-separated bearer tokens required by /v1/* (required off-loopback)
 *   EMBED_ALLOW_UNAUTHENTICATED        Permit a non-loopback bind without EMBED_API_KEYS (default false)
 *   EMBED_DIMENSIONS                   Default Matryoshka output size (e.g. 256); native when unset
 *   EMBED_CACHE_SIZE                   LRU vectors kept in memory (default 20000)
 *   EMBED_MAX_INPUTS / EMBED_MAX_INPUT_CHARS / EMBED_MAX_BODY_BYTES   Request limits (256 / 16000 / 4 MiB)
 *   EMBED_SPACE                        Assert the embedding-space id for all backends (enables custom fallbacks)
 *   EMBED_PROMPT_FAMILY                embeddinggemma | nomic | e5 | bge | none (derived from the model by default)
 *   EMBED_SIMILARITY_FLOOR / _CEILING  Cosine calibration published to clients (default: measured by the self-test)
 *   EMBED_SELF_TEST                    Run the startup quality self-test (default true)
 *   EMBED_SELF_TEST_ENFORCE            Quarantine backends that fail it (default true)
 *
 *   NPU (FastFlowLM):  EMBED_NPU_URL (http://127.0.0.1:52625/v1), EMBED_NPU_MODEL (embed-gemma:300m),
 *                      EMBED_NPU_AUTOSTART (false), EMBED_NPU_FLM_COMMAND (flm), EMBED_NPU_CHAT_MODEL (qwen3:0.6b),
 *                      EMBED_NPU_POWER_MODE (balanced), EMBED_NPU_TIMEOUT_MS (30000)
 *   ONNX:              EMBED_ONNX_MODEL, EMBED_ONNX_DEVICE (cpu|dml|cuda|webgpu|auto), EMBED_ONNX_DTYPE (q4),
 *                      EMBED_ONNX_CACHE_DIR, EMBED_ONNX_OFFLINE (false)
 *   OpenAI-compatible: EMBED_OPENAI_URL (https://api.openai.com/v1), EMBED_OPENAI_MODEL (text-embedding-3-small),
 *                      EMBED_OPENAI_API_KEY (falls back to OPENAI_API_KEY; OPENROUTER_API_KEY for openrouter.ai URLs),
 *                      EMBED_OPENAI_SEND_DIMENSIONS (true for api.openai.com), EMBED_OPENAI_TIMEOUT_MS
 *   Gemini:            EMBED_GEMINI_MODEL (gemini-embedding-001), EMBED_GEMINI_API_KEY (falls back to GEMINI_API_KEY)
 */

import { isPromptFamily, type PromptFamily } from "./models.js";
import { EmbeddingError, type SimilarityCalibration } from "./types.js";
import type { OnnxDevice, OnnxDtype } from "./backends/onnx.js";

export type BackendName = "npu" | "onnx" | "openai" | "gemini";

export interface NpuBackendConfig {
	readonly url: string;
	readonly model: string;
	readonly autostart: boolean;
	readonly flmCommand: string;
	readonly chatModel: string;
	readonly powerMode: string;
	readonly timeoutMs: number;
}

export interface OnnxBackendConfig {
	readonly model: string;
	readonly device: OnnxDevice;
	readonly dtype: OnnxDtype;
	readonly cacheDir: string | undefined;
	readonly offline: boolean;
}

export interface OpenAIBackendConfig {
	readonly url: string;
	readonly model: string;
	readonly apiKey: string | undefined;
	readonly sendDimensions: boolean;
	readonly timeoutMs: number;
}

export interface GeminiBackendConfig {
	readonly model: string;
	readonly apiKey: string | undefined;
	readonly timeoutMs: number;
}

export interface EmbedServerConfig {
	readonly host: string;
	readonly port: number;
	readonly apiKeys: readonly string[];
	readonly allowUnauthenticated: boolean;
	readonly backends: readonly BackendName[];
	readonly dimensions: number | undefined;
	readonly cacheSize: number;
	readonly maxInputs: number;
	readonly maxInputChars: number;
	readonly maxBodyBytes: number;
	readonly space: string | undefined;
	readonly promptFamily: PromptFamily | undefined;
	readonly calibration: SimilarityCalibration | undefined;
	readonly selfTest: boolean;
	readonly selfTestEnforce: boolean;
	readonly npu: NpuBackendConfig;
	readonly onnx: OnnxBackendConfig;
	readonly openai: OpenAIBackendConfig;
	readonly gemini: GeminiBackendConfig;
}

type Env = Readonly<Record<string, string | undefined>>;

const PROFILES: Record<string, { backends: BackendName[]; onnxModel?: string; npuModel?: string }> = {
	gemma2: { backends: ["onnx"], onnxModel: "onnx-community/embeddinggemma-2-ONNX" },
	npu: { backends: ["npu"], npuModel: "embed-gemma:300m" },
	openai: { backends: ["openai"] },
	gemini: { backends: ["gemini"] },
};

function str(env: Env, name: string): string | undefined {
	const v = env[name]?.trim();
	return v ? v : undefined;
}

function int(env: Env, name: string, fallback: number, min = 0): number {
	const raw = str(env, name);
	if (raw === undefined) return fallback;
	const n = Number(raw);
	if (!Number.isInteger(n) || n < min) throw new EmbeddingError("MISCONFIGURED", `${name}=${JSON.stringify(raw)} must be an integer >= ${min}`);
	return n;
}

function bool(env: Env, name: string, fallback: boolean): boolean {
	const raw = str(env, name)?.toLowerCase();
	if (raw === undefined) return fallback;
	if (["1", "true", "yes", "on"].includes(raw)) return true;
	if (["0", "false", "no", "off"].includes(raw)) return false;
	throw new EmbeddingError("MISCONFIGURED", `${name}=${JSON.stringify(raw)} must be true or false`);
}

function num(env: Env, name: string): number | undefined {
	const raw = str(env, name);
	if (raw === undefined) return undefined;
	const n = Number(raw);
	if (!Number.isFinite(n)) throw new EmbeddingError("MISCONFIGURED", `${name}=${JSON.stringify(raw)} must be a number`);
	return n;
}

export function loadEmbedServerConfig(env: Env = process.env): EmbedServerConfig {
	const profileName = (str(env, "EMBED_PROFILE") ?? "gemma2").toLowerCase();
	const profile = PROFILES[profileName];
	if (!profile) {
		throw new EmbeddingError("MISCONFIGURED", `EMBED_PROFILE=${profileName} is unknown (use ${Object.keys(PROFILES).join(", ")})`);
	}
	const backendsRaw = str(env, "EMBED_BACKENDS");
	const backends = (backendsRaw ? backendsRaw.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean) : profile.backends) as BackendName[];
	for (const b of backends) {
		if (!["npu", "onnx", "openai", "gemini"].includes(b)) {
			throw new EmbeddingError("MISCONFIGURED", `EMBED_BACKENDS contains unknown backend "${b}" (use npu, onnx, openai, gemini)`);
		}
	}
	if (backends.length === 0) throw new EmbeddingError("MISCONFIGURED", "EMBED_BACKENDS is empty");

	const dimsRaw = str(env, "EMBED_DIMENSIONS");
	const promptFamilyRaw = str(env, "EMBED_PROMPT_FAMILY")?.toLowerCase();
	if (promptFamilyRaw !== undefined && !isPromptFamily(promptFamilyRaw)) {
		throw new EmbeddingError("MISCONFIGURED", `EMBED_PROMPT_FAMILY=${promptFamilyRaw} is unknown`);
	}
	const floor = num(env, "EMBED_SIMILARITY_FLOOR");
	const ceiling = num(env, "EMBED_SIMILARITY_CEILING");
	if ((floor === undefined) !== (ceiling === undefined)) {
		throw new EmbeddingError("MISCONFIGURED", "set both EMBED_SIMILARITY_FLOOR and EMBED_SIMILARITY_CEILING, or neither");
	}
	if (floor !== undefined && ceiling !== undefined && !(ceiling > floor)) {
		throw new EmbeddingError("MISCONFIGURED", "EMBED_SIMILARITY_CEILING must be greater than EMBED_SIMILARITY_FLOOR");
	}

	const openaiUrl = str(env, "EMBED_OPENAI_URL") ?? "https://api.openai.com/v1";
	const isOpenRouter = /openrouter\.ai/i.test(openaiUrl);
	const device = (str(env, "EMBED_ONNX_DEVICE") ?? "cpu").toLowerCase();
	if (!["cpu", "dml", "cuda", "webgpu", "auto"].includes(device)) {
		throw new EmbeddingError("MISCONFIGURED", `EMBED_ONNX_DEVICE=${device} is unknown (cpu, dml, cuda, webgpu, auto)`);
	}
	const dtype = (str(env, "EMBED_ONNX_DTYPE") ?? "q4").toLowerCase();
	if (!["fp32", "fp16", "q8", "q4", "q4f16", "int8", "uint8"].includes(dtype)) {
		throw new EmbeddingError("MISCONFIGURED", `EMBED_ONNX_DTYPE=${dtype} is unknown`);
	}

	return {
		host: str(env, "EMBED_HOST") ?? "127.0.0.1",
		port: int(env, "EMBED_PORT", 18765, 1),
		apiKeys: (str(env, "EMBED_API_KEYS") ?? "").split(",").map((s) => s.trim()).filter(Boolean),
		allowUnauthenticated: bool(env, "EMBED_ALLOW_UNAUTHENTICATED", false),
		backends,
		dimensions: dimsRaw === undefined ? undefined : int(env, "EMBED_DIMENSIONS", 0, 8),
		cacheSize: int(env, "EMBED_CACHE_SIZE", 20_000),
		maxInputs: int(env, "EMBED_MAX_INPUTS", 256, 1),
		maxInputChars: int(env, "EMBED_MAX_INPUT_CHARS", 16_000, 64),
		maxBodyBytes: int(env, "EMBED_MAX_BODY_BYTES", 4 * 1024 * 1024, 1024),
		space: str(env, "EMBED_SPACE"),
		promptFamily: promptFamilyRaw as PromptFamily | undefined,
		calibration: floor !== undefined && ceiling !== undefined ? { floor, ceiling } : undefined,
		selfTest: bool(env, "EMBED_SELF_TEST", true),
		selfTestEnforce: bool(env, "EMBED_SELF_TEST_ENFORCE", true),
		npu: {
			url: str(env, "EMBED_NPU_URL") ?? "http://127.0.0.1:52625/v1",
			model: str(env, "EMBED_NPU_MODEL") ?? profile.npuModel ?? "embed-gemma:300m",
			autostart: bool(env, "EMBED_NPU_AUTOSTART", false),
			flmCommand: str(env, "EMBED_NPU_FLM_COMMAND") ?? "flm",
			chatModel: str(env, "EMBED_NPU_CHAT_MODEL") ?? "qwen3:0.6b",
			powerMode: str(env, "EMBED_NPU_POWER_MODE") ?? "balanced",
			timeoutMs: int(env, "EMBED_NPU_TIMEOUT_MS", 30_000, 1),
		},
		onnx: {
			model: str(env, "EMBED_ONNX_MODEL") ?? profile.onnxModel ?? "onnx-community/embeddinggemma-2-ONNX",
			device: device as OnnxDevice,
			dtype: dtype as OnnxDtype,
			cacheDir: str(env, "EMBED_ONNX_CACHE_DIR"),
			offline: bool(env, "EMBED_ONNX_OFFLINE", false),
		},
		openai: {
			url: openaiUrl,
			model: str(env, "EMBED_OPENAI_MODEL") ?? "text-embedding-3-small",
			apiKey: str(env, "EMBED_OPENAI_API_KEY") ?? (isOpenRouter ? str(env, "OPENROUTER_API_KEY") : str(env, "OPENAI_API_KEY")),
			sendDimensions: bool(env, "EMBED_OPENAI_SEND_DIMENSIONS", /api\.openai\.com/i.test(openaiUrl)),
			timeoutMs: int(env, "EMBED_OPENAI_TIMEOUT_MS", 30_000, 1),
		},
		gemini: {
			model: str(env, "EMBED_GEMINI_MODEL") ?? "gemini-embedding-001",
			apiKey: str(env, "EMBED_GEMINI_API_KEY") ?? str(env, "GEMINI_API_KEY"),
			timeoutMs: int(env, "EMBED_GEMINI_TIMEOUT_MS", 30_000, 1),
		},
	};
}

export function isLoopbackHost(host: string): boolean {
	const h = host.trim().toLowerCase().replace(/^\[|\]$/g, "");
	return h === "localhost" || h === "::1" || /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(h);
}
