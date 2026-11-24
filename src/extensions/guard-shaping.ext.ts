import { contentText } from "@earendil-works/pi-ai";
import { defineExtension, hook, ToolTask, type Extension, type ToolExecutionResult } from "@earendil-works/pi-durable";
import type { AppCapabilities } from "../contracts/app-capabilities.js";
import { hasPermission } from "../contracts/tenant-context.js";
import { ConversationMetaDoc, deserializeTenantFromDoc } from "../harness/app-documents.js";

export const DEFAULT_TOOL_PERMISSIONS: Readonly<Record<string, string>> = {
	memory_search: "memory:read",
	memory_pin: "memory:write",
	memory_forget: "memory:write",
	artifact_read: "artifacts:read",
	skill_read: "skills:read",
	domain_lookup_record: "domain:read",
	domain_mutate_record: "domain:write",
	domain_request_approval: "domain:approve_request",
};

const SPOOFED_TENANT_KEYS = new Set(["tenantId", "tenant_id", "userId", "user_id"]);

export interface GuardShapingExtensionOptions {
	readonly capabilities: AppCapabilities;
	/** Additional or overridden tool -> required permission mappings. */
	readonly toolPermissions?: Readonly<Record<string, string>>;
	/** Max UTF-8 bytes allowed in model-visible tool text output before offloading. Default: 6144 (6KB). */
	readonly maxToolOutputBytes?: number;
}

function truncateUtf8HeadTail(text: string, maxBytes: number): { head: string; tail: string } {
	const headBudget = Math.max(256, Math.floor(maxBytes * 0.65));
	const tailBudget = Math.max(128, Math.floor(maxBytes * 0.25));
	const buf = Buffer.from(text, "utf8");
	const head = buf.subarray(0, headBudget).toString("utf8");
	const tail = buf.subarray(Math.max(headBudget, buf.length - tailBudget)).toString("utf8");
	return { head, tail };
}

/**
 * Isolated Extension: Authorization guardrails + 6KB tool output capping & artifact offloading.
 */
export function createGuardShapingExtension(options: GuardShapingExtensionOptions): Extension {
	const { capabilities } = options;
	const permissionMap: Record<string, string> = {
		...DEFAULT_TOOL_PERMISSIONS,
		...options.toolPermissions,
	};
	const maxBytes = options.maxToolOutputBytes ?? capabilities.maxToolOutputBytes ?? 6144;

	return defineExtension({
		name: "guard-shaping",
		hooks: [
			hook(ToolTask, {
				beforeTool: async (call, api, context) => {
					// Defense-in-depth: block any attempt to inject tenant/user identity via args
					for (const key of Object.keys(call.arguments ?? {})) {
						if (SPOOFED_TENANT_KEYS.has(key)) {
							return {
								block: `Security violation: argument "${key}" is forbidden; tenant scope is bound by runtime.`,
							};
						}
					}

					const requiredPermission = permissionMap[call.name];
					if (!requiredPermission) {
						return undefined;
					}

					const meta = await api.snapshot(ConversationMetaDoc, api.conversationId, context);
					if (!meta) {
						return { block: `Missing conversation tenant metadata for authorization check on "${call.name}"` };
					}

					const tenant = deserializeTenantFromDoc(meta);
					if (!hasPermission(tenant, requiredPermission)) {
						return {
							block: `Authorization denied: missing permission "${requiredPermission}" for tool "${call.name}"`,
						};
					}

					return undefined;
				},

				afterTool: async (call, result, api, context) => {
					if (!result.output || result.output.length === 0) {
						return undefined;
					}

					const textBlocks = result.output.filter((b) => b.type === "text");
					if (textBlocks.length === 0) return undefined;

					const fullText = contentText(textBlocks);
					const totalBytes = Buffer.byteLength(fullText, "utf8");
					if (totalBytes <= maxBytes) {
						return undefined;
					}

					const meta = await api.snapshot(ConversationMetaDoc, api.conversationId, context);
					const tenantId = meta?.tenant.tenantId ?? "default";
					const userId = meta?.tenant.userId ?? "anonymous";

					const artifact = await capabilities.artifacts.put(tenantId, userId, fullText, {
						contentType: "text/plain; charset=utf-8",
						sourceTool: call.name,
						conversationId: String(api.conversationId),
					});

					const notice = [
						`\n\n[Output truncated: ${totalBytes} bytes exceeded ${maxBytes}-byte cap.]`,
						`Full output saved as artifact "${artifact.id}".`,
						`Use artifact_read({ "artifactId": "${artifact.id}", "offset": 0, "limit": 4000 }) to inspect further.`,
					].join("\n");

					const noticeBytes = Buffer.byteLength(notice, "utf8");
					const previewBudget = Math.max(512, maxBytes - noticeBytes - 64);
					const { head, tail } = truncateUtf8HeadTail(fullText, previewBudget);
					const shapedText = `${head}\n\n... [truncated ${totalBytes} bytes total] ...\n\n${tail}${notice}`;

					const nonTextBlocks = result.output.filter((b) => b.type !== "text");
					const existingDetails =
						result.details !== undefined &&
						result.details !== null &&
						typeof result.details === "object" &&
						!Array.isArray(result.details)
							? (result.details as Record<string, unknown>)
							: {};

					const shaped: ToolExecutionResult = {
						...result,
						output: [{ type: "text", text: shapedText }, ...nonTextBlocks],
						details: {
							...existingDetails,
							offloadedArtifactId: artifact.id,
							originalBytes: totalBytes,
							cappedBytes: maxBytes,
						} as ToolExecutionResult["details"],
						diagnostics: [
							...(result.diagnostics ?? []),
							{
								severity: "info",
								code: "artifact_offloaded",
								message: `Tool "${call.name}" output (${totalBytes} bytes) offloaded to artifact "${artifact.id}".`,
							},
						],
					};

					return shaped;
				},
			}),
		],
	});
}
