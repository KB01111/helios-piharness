import { createModels, createProvider, type Models, type MutableModels } from "@earendil-works/pi-ai/models";
import { anthropicProvider } from "@earendil-works/pi-ai/providers/anthropic";
import {
	createFauxCore,
	fauxAssistantMessage,
	fauxText,
	fauxToolCall,
	type FauxContentBlock,
} from "@earendil-works/pi-ai/providers/faux";
import { googleProvider } from "@earendil-works/pi-ai/providers/google";
import { openaiProvider } from "@earendil-works/pi-ai/providers/openai";
import type { AssistantMessage, Message, SystemMessage, TranscriptContext } from "@earendil-works/pi-ai";
import { contentText, getSystemMessageText } from "@earendil-works/pi-ai";
import { getCurrentSystemPrompt, getCurrentTools } from "@earendil-works/pi-ai/utils/transcript";
import type { FastModelClient } from "../contracts/app-capabilities.js";
import type { ModelSpec } from "../contracts/harness-port.js";

/**
 * Pi-free snapshot of a single model invocation captured by TestModel.
 * Used by tests (e.g. KV-cache prefix stability) and trace-replay evals.
 */
export interface RecordedModelRequest {
	readonly index: number;
	readonly provider: string;
	readonly modelId: string;
	readonly sessionId?: string;
	/** Exact byte-for-byte text of the leading system message (message[0] if role === "system"). */
	readonly leadingSystemPrompt: string;
	/** Full replayed system prompt after applying all system messages in order. */
	readonly effectiveSystemPrompt: string;
	/** Number of system messages present in the request transcript. */
	readonly systemMessageCount: number;
	/** Exact sections object of the leading system message (if any). */
	readonly leadingSystemSections?: Readonly<Record<string, string | null>>;
	/** Names of currently active tools offered to the model. */
	readonly offeredToolNames: readonly string[];
	/** Total messages sent to the model. */
	readonly messageCount: number;
	/** Last user message text (if present). */
	readonly lastUserText?: string;
}

export type ScriptedTurnOutput =
	| { readonly type: "text"; readonly text: string }
	| {
			readonly type: "tool_calls";
			readonly calls: readonly {
				readonly id?: string;
				readonly name: string;
				readonly arguments: Record<string, unknown>;
			}[];
			readonly text?: string;
	  }
	| { readonly type: "error"; readonly errorMessage: string };

export type TestModelHandler = (
	request: RecordedModelRequest,
) => ScriptedTurnOutput | Promise<ScriptedTurnOutput>;

/**
 * Public, Pi-free controller for the deterministic offline TestModel.
 */
export interface TestModelController {
	readonly provider: string;
	readonly mainModel: ModelSpec;
	readonly fastModel: ModelSpec;
	enqueueMain(step: ScriptedTurnOutput | TestModelHandler): void;
	enqueueMainMany(steps: readonly (ScriptedTurnOutput | TestModelHandler)[]): void;
	enqueueFast(step: ScriptedTurnOutput | TestModelHandler): void;
	enqueueFastJson(payload: unknown): void;
	setDefaultMainHandler(handler: TestModelHandler | undefined): void;
	setDefaultFastHandler(handler: TestModelHandler | undefined): void;
	getRecordedRequests(filter?: { readonly modelId?: string }): readonly RecordedModelRequest[];
	clearRecordedRequests(): void;
	reset(): void;
}

function extractLastUserText(messages: readonly Message[]): string | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i];
		if (msg && msg.role === "user") {
			return contentText(msg.content);
		}
	}
	return undefined;
}

function toFauxMessage(output: ScriptedTurnOutput, callCounter: () => number): AssistantMessage {
	if (output.type === "error") {
		return fauxAssistantMessage("", { stopReason: "error", errorMessage: output.errorMessage });
	}
	if (output.type === "text") {
		return fauxAssistantMessage(output.text, { stopReason: "stop" });
	}
	const blocks: FauxContentBlock[] = [];
	if (output.text) {
		blocks.push(fauxText(output.text));
	}
	for (const call of output.calls) {
		blocks.push(
			fauxToolCall(call.name, call.arguments as Record<string, never>, {
				id: call.id ?? `call_det_${callCounter()}`,
			}),
		);
	}
	return fauxAssistantMessage(blocks, { stopReason: "toolUse" });
}

function defaultMainFallback(request: RecordedModelRequest): ScriptedTurnOutput {
	const userText = request.lastUserText ?? "";
	return {
		type: "text",
		text: `[mock-main] Echo: ${userText}`.trim(),
	};
}

function defaultFastFallback(): ScriptedTurnOutput {
	return {
		type: "text",
		text: JSON.stringify({ operations: [{ op: "NOOP", reason: "default mock fast model" }] }),
	};
}

function recordTranscriptRequest(
	index: number,
	provider: string,
	modelId: string,
	context: TranscriptContext,
	sessionId: string | undefined,
): RecordedModelRequest {
	const systemMessages = context.messages.filter((m): m is SystemMessage => m.role === "system");
	const leading = context.messages[0]?.role === "system" ? (context.messages[0] as SystemMessage) : undefined;
	const leadingSystemPrompt = leading ? getSystemMessageText(leading) : "";
	const effectiveSystemPrompt = getCurrentSystemPrompt(context.messages);
	const offeredToolNames = getCurrentTools(context.messages).map((t) => t.name);
	const lastUserText = extractLastUserText(context.messages);

	return {
		index,
		provider,
		modelId,
		...(sessionId ? { sessionId } : {}),
		leadingSystemPrompt,
		effectiveSystemPrompt,
		systemMessageCount: systemMessages.length,
		...(leading?.sections ? { leadingSystemSections: { ...leading.sections } } : {}),
		offeredToolNames,
		messageCount: context.messages.length,
		...(lastUserText !== undefined ? { lastUserText } : {}),
	};
}

export interface ModelCatalogBundle {
	readonly models: MutableModels;
	readonly mainModel: ModelSpec;
	readonly fastModel: ModelSpec;
	readonly testModel: TestModelController;
	readonly fastModelClient: FastModelClient;
}

export function createModelCatalog(options: {
	readonly mainModel: ModelSpec;
	readonly fastModel: ModelSpec;
	readonly useMockModels: boolean;
	readonly registerRealProviders?: boolean;
}): ModelCatalogBundle {
	const models = createModels();

	if (options.registerRealProviders ?? !options.useMockModels) {
		models.setProvider(anthropicProvider());
		models.setProvider(openaiProvider());
		models.setProvider(googleProvider());
	}

	let callSeq = 0;
	let requestSeq = 0;
	const mainQueue: Array<ScriptedTurnOutput | TestModelHandler> = [];
	const fastQueue: Array<ScriptedTurnOutput | TestModelHandler> = [];
	let defaultMainHandler: TestModelHandler | undefined;
	let defaultFastHandler: TestModelHandler | undefined;
	const recordedRequests: RecordedModelRequest[] = [];

	const core = createFauxCore({
		provider: "test",
		api: "test-faux",
		tokenSize: { min: 4, max: 4 },
		models: [
			{
				id: "main",
				name: "Deterministic Test Main Model",
				reasoning: false,
				input: ["text"],
				contextWindow: 128_000,
				maxTokens: 16_384,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			},
			{
				id: "fast",
				name: "Deterministic Test Fast Model",
				reasoning: false,
				input: ["text"],
				contextWindow: 64_000,
				maxTokens: 4_096,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			},
		],
	});

	const dispatchStream: typeof core.stream = (requestModel, context, streamOptions) => {
		const recorded = recordTranscriptRequest(
			++requestSeq,
			requestModel.provider,
			requestModel.id,
			context,
			streamOptions?.sessionId,
		);
		recordedRequests.push(recorded);

		const isFast = requestModel.id === "fast";
		const queue = isFast ? fastQueue : mainQueue;
		const next = queue.shift();

		core.setResponses([
			async () => {
				let output: ScriptedTurnOutput;
				if (next !== undefined) {
					output = typeof next === "function" ? await next(recorded) : next;
				} else if (isFast) {
					output = defaultFastHandler ? await defaultFastHandler(recorded) : defaultFastFallback();
				} else {
					output = defaultMainHandler ? await defaultMainHandler(recorded) : defaultMainFallback(recorded);
				}
				return toFauxMessage(output, () => ++callSeq);
			},
		]);

		return core.stream(requestModel, context, streamOptions);
	};

	const testProvider = createProvider({
		id: "test",
		name: "Deterministic Test Provider",
		auth: { apiKey: { name: "Test", resolve: async () => ({ auth: {} }) } },
		models: core.models,
		api: {
			stream: dispatchStream,
			streamSimple: dispatchStream,
			fetchDeferred: core.fetchDeferred,
			cancelDeferred: core.cancelDeferred,
		},
	});

	models.setProvider(testProvider);

	const testModel: TestModelController = {
		provider: "test",
		mainModel: { provider: "test", modelId: "main" },
		fastModel: { provider: "test", modelId: "fast" },
		enqueueMain(step) {
			mainQueue.push(step);
		},
		enqueueMainMany(steps) {
			mainQueue.push(...steps);
		},
		enqueueFast(step) {
			fastQueue.push(step);
		},
		enqueueFastJson(payload) {
			fastQueue.push({ type: "text", text: JSON.stringify(payload) });
		},
		setDefaultMainHandler(handler) {
			defaultMainHandler = handler;
		},
		setDefaultFastHandler(handler) {
			defaultFastHandler = handler;
		},
		getRecordedRequests(filter) {
			if (!filter?.modelId) return [...recordedRequests];
			return recordedRequests.filter((r) => r.modelId === filter.modelId);
		},
		clearRecordedRequests() {
			recordedRequests.length = 0;
		},
		reset() {
			mainQueue.length = 0;
			fastQueue.length = 0;
			recordedRequests.length = 0;
			defaultMainHandler = undefined;
			defaultFastHandler = undefined;
			callSeq = 0;
			requestSeq = 0;
		},
	};

	const mainModel = options.useMockModels ? testModel.mainModel : options.mainModel;
	const fastModel = options.useMockModels ? testModel.fastModel : options.fastModel;
	const fastModelClient = createFastModelClient(models, fastModel);

	return {
		models,
		mainModel,
		fastModel,
		testModel,
		fastModelClient,
	};
}

function tryParseJson<T>(raw: string): T | null {
	const trimmed = raw.trim();
	const unfenced = trimmed.startsWith("```")
		? trimmed.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "")
		: trimmed;
	try {
		return JSON.parse(unfenced) as T;
	} catch {
		const start = unfenced.indexOf("{");
		const end = unfenced.lastIndexOf("}");
		if (start >= 0 && end > start) {
			try {
				return JSON.parse(unfenced.slice(start, end + 1)) as T;
			} catch {
				return null;
			}
		}
		return null;
	}
}

export function createFastModelClient(models: Models, fastModelRef: ModelSpec): FastModelClient {
	return {
		async completeJson<T = unknown>(
			systemPrompt: string,
			userPrompt: string,
			options?: { readonly signal?: AbortSignal },
		): Promise<{ readonly rawText: string; readonly parsed: T | null }> {
			const model = models.getModel(fastModelRef.provider, fastModelRef.modelId);
			if (!model) {
				throw new Error(
					`Fast model "${fastModelRef.provider}:${fastModelRef.modelId}" is not registered in model catalog`,
				);
			}
			const response = await models.completeSimple(
				model,
				{
					systemPrompt,
					messages: [{ role: "user", content: userPrompt, timestamp: Date.now() }],
				},
				{
					signal: options?.signal,
				},
			);
			const rawText = contentText(response.content);
			return {
				rawText,
				parsed: tryParseJson<T>(rawText),
			};
		},
	};
}
