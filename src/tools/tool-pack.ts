import type { Context } from "@earendil-works/chord/context";
import type { JsonValue } from "@earendil-works/chord/json";
import {
	defineTool,
	type ToolExecutionApi,
	type ToolExecutionMode,
	type ToolExecutionResult,
	type ToolPark,
	type ToolRegistration,
} from "@earendil-works/pi-durable";
import type { Static } from "@earendil-works/pi-schema";
import type { TenantContext } from "../contracts/tenant-context.js";
import { ConversationMetaDoc, deserializeTenantFromDoc } from "../harness/app-documents.js";
import type { ToolPackDefinition } from "../harness/registry.js";

const FORBIDDEN_TENANT_PARAM_KEYS = new Set(["tenantId", "tenant_id", "userId", "user_id"]);

function assertNoTenantParamsInSchema(toolName: string, schema: object): void {
	const props = (schema as { properties?: Record<string, unknown> }).properties;
	if (!props || typeof props !== "object") return;
	for (const key of Object.keys(props)) {
		if (FORBIDDEN_TENANT_PARAM_KEYS.has(key)) {
			throw new Error(
				`Security violation in tool "${toolName}": parameter "${key}" must NOT be exposed to the LLM; bind tenant scope via closure/ConversationMetaDoc.`,
			);
		}
	}
}

export interface TenantScopedToolConfig<
	TParameters extends object,
	TDetails extends JsonValue = JsonValue,
	C extends JsonValue = JsonValue,
> {
	readonly name: string;
	readonly description: string;
	readonly parameters: TParameters;
	readonly replay?: "safe" | "unsafe";
	/** Who may call the tool. `["model"]` keeps it out of reach of codemode scripts (e.g. human-approval parks). */
	readonly callers?: readonly ("model" | "tools")[];
	readonly executionMode?: ToolExecutionMode;
	readonly structuredOutputSchema?: object;
	readonly outputLimits?: {
		readonly maxBytes?: number;
		readonly maxLines?: number;
		readonly retain?: "head" | "tail";
	};
	execute(
		args: Static<TParameters>,
		tenant: TenantContext,
		api: ToolExecutionApi<TDetails, C>,
		context: Context,
	): Promise<ToolExecutionResult<TDetails> | ToolPark<C>>;
	cancel?(
		args: Static<TParameters>,
		tenant: TenantContext | undefined,
		api: ToolExecutionApi<TDetails, C>,
		context: Context,
	): Promise<void>;
}

/**
 * Builds a Pi Durable tool whose handler always receives the conversation's
 * verified `TenantContext` resolved from committed `ConversationMetaDoc`.
 *
 * Refuses to register any tool whose schema exposes `tenantId` or `userId`
 * as model-supplied parameters.
 */
export function createTenantScopedTool<
	TParameters extends object,
	TDetails extends JsonValue = JsonValue,
	C extends JsonValue = JsonValue,
>(config: TenantScopedToolConfig<TParameters, TDetails, C>): ToolRegistration<TParameters, TDetails, C> {
	assertNoTenantParamsInSchema(config.name, config.parameters);

	return defineTool<TParameters, TDetails, C>({
		name: config.name,
		description: config.description,
		parameters: config.parameters,
		...(config.replay !== undefined ? { replay: config.replay } : {}),
		...(config.callers !== undefined ? { callers: config.callers } : {}),
		...(config.executionMode !== undefined ? { executionMode: config.executionMode } : {}),
		...(config.structuredOutputSchema !== undefined ? { structuredOutputSchema: config.structuredOutputSchema } : {}),
		...(config.outputLimits !== undefined ? { outputLimits: config.outputLimits } : {}),
		execute: async (args, api, context) => {
			const meta = await api.snapshot(ConversationMetaDoc, api.conversationId, context);
			if (!meta) {
				return {
					isError: true,
					output: [{ type: "text", text: `Missing conversation tenant metadata for tool "${config.name}"` }],
				};
			}
			const tenant = deserializeTenantFromDoc(meta);
			return config.execute(args as Static<TParameters>, tenant, api, context);
		},
		...(config.cancel
			? {
					cancel: async (args, api, context) => {
						const meta = await api.snapshot(ConversationMetaDoc, api.conversationId, context);
						const tenant = meta ? deserializeTenantFromDoc(meta) : undefined;
						await config.cancel!(args as Static<TParameters>, tenant, api, context);
					},
				}
			: {}),
	});
}

export function defineToolPack(pack: ToolPackDefinition): ToolPackDefinition {
	return pack;
}
