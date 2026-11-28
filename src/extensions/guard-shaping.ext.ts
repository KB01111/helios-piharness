import { contentText } from "@earendil-works/pi-ai";
import {
	defineExtension,
	defineSignal,
	hook,
	ToolTask,
	type Extension,
	type ToolExecutionResult,
} from "@earendil-works/pi-durable";
import { bool, obj, opt, str } from "@earendil-works/pi-schema";
import type { AppCapabilities } from "../contracts/app-capabilities.js";
import { hasPermission } from "../contracts/tenant-context.js";
import { ConversationMetaDoc, ConversationModeDoc, deserializeTenantFromDoc } from "../harness/app-documents.js";
import {
	evaluatePermissionPolicy,
	type PermissionAuditStore,
	type PermissionPolicyConfig,
} from "../security/permission-policy.js";
import { redactSecrets, type SecretPattern } from "../security/secret-redactor.js";

export const DEFAULT_TOOL_PERMISSIONS: Readonly<Record<string, string>> = {
	memory_search: "memory:read",
	memory_pin: "memory:write",
	memory_forget: "memory:write",
	artifact_read: "artifacts:read",
	skill_read: "skills:read",
	file_find: "files:read",
	file_search: "files:read",
	file_symbol_def: "files:read",
	file_index_map: "files:read",
	domain_lookup_record: "domain:read",
	domain_mutate_record: "domain:write",
	domain_request_approval: "domain:approve_request",
	subagent_delegate: "subagent:spawn",
	subagent_parallel: "subagent:spawn",
	advisor_consult: "advisor:consult",
	mcp_list_servers: "mcp:read",
	mcp_discover: "mcp:read",
};

/** Built-in tools guaranteed not to mutate external or memory state (permitted in read-only Plan Mode). */
export const DEFAULT_READ_ONLY_TOOLS: ReadonlySet<string> = new Set([
	"memory_search",
	"artifact_read",
	"skill_read",
	"file_find",
	"file_search",
	"file_symbol_def",
	"file_index_map",
	"plan_set",
	"plan_update",
	"plan_view",
	"domain_lookup_record",
	"tool_search",
	"advisor_consult",
	"mcp_list_servers",
	"mcp_discover",
]);

/**
 * Typed Pi Durable signal delivered externally to approve or deny a tool call
 * parked at a three-tier `ask` permission gate in `beforeTool`.
 */
export const ToolGateApprovalSignal = defineSignal({
	name: "app.tool_gate.decision",
	version: 1,
	schema: obj({
		approved: bool(),
		reviewer: opt(str()),
		note: opt(str()),
	}),
});

const SPOOFED_TENANT_KEYS = new Set(["tenantId", "tenant_id", "userId", "user_id"]);

export interface GuardShapingExtensionOptions {
	readonly capabilities: AppCapabilities;
	/** Additional or overridden tool -> required permission mappings. */
	readonly toolPermissions?: Readonly<Record<string, string>>;
	/** Max UTF-8 bytes allowed in model-visible tool text output before offloading. Default: 6144 (6KB). */
	readonly maxToolOutputBytes?: number;
	/** Three-tier permission policy (`allow` | `ask` | `deny`). Default: allow (subject to RBAC). */
	readonly permissionPolicy?: PermissionPolicyConfig;
	/** SQLite audit & pending-gate-approval store (optional; enables durable audit log & `/approvals` surfacing). */
	readonly auditStore?: PermissionAuditStore;
	/** Scan and redact secrets/credentials in tool output before artifacts and LLM context (default true). */
	readonly secretRedactionEnabled?: boolean;
	/** Additional secret redaction patterns evaluated after the built-in set. */
	readonly extraRedactionPatterns?: readonly SecretPattern[];
	/** Extra read-only tool names permitted in read-only Plan Mode. */
	readonly readOnlyTools?: readonly string[];
	/** Dynamic read-only predicate (e.g. for read-only MCP tools or read-only subagent profiles). */
	readonly isReadOnlyTool?: (toolName: string, args: Readonly<Record<string, unknown>>) => boolean;
}

function truncateUtf8HeadTail(text: string, maxBytes: number): { head: string; tail: string } {
	const headBudget = Math.max(256, Math.floor(maxBytes * 0.65));
	const tailBudget = Math.max(128, Math.floor(maxBytes * 0.25));
	const buf = Buffer.from(text, "utf8");
	const head = buf.subarray(0, headBudget).toString("utf8");
	const tail = buf.subarray(Math.max(headBudget, buf.length - tailBudget)).toString("utf8");
	return { head, tail };
}

function summarizeArgs(
	args: Readonly<Record<string, unknown>> | undefined,
	extraPatterns?: readonly SecretPattern[],
): string {
	if (!args || Object.keys(args).length === 0) return "{}";
	try {
		// Approval summaries are persisted (pending approvals, audit log) and served over HTTP, so redact BEFORE truncating:
		// truncating first could cut a secret in half and defeat the pattern.
		const raw = redactSecrets(JSON.stringify(args), extraPatterns ? { extraPatterns } : undefined).text;
		return raw.length > 240 ? `${raw.slice(0, 237)}...` : raw;
	} catch {
		return "{...}";
	}
}

/**
 * Isolated Extension:
 * - Tenant-spoof defense (`tenantId`/`userId` args forbidden)
 * - Runtime-enforced Read-Only Plan Mode (`ConversationModeDoc.mode === "plan"`)
 * - RBAC permission enforcement (`DEFAULT_TOOL_PERMISSIONS` + dynamic MCP read/write)
 * - Three-tier Permission Policy (`allow` | `ask` | `deny`) with durable `beforeTool` approval parking & audit log
 * - Tool output secret redaction (before artifacts and before LLM context, including nested calls)
 * - 6KB tool output capping & artifact offloading
 */
export function createGuardShapingExtension(options: GuardShapingExtensionOptions): Extension {
	const { capabilities, permissionPolicy, auditStore } = options;
	const permissionMap: Record<string, string> = {
		...DEFAULT_TOOL_PERMISSIONS,
		...options.toolPermissions,
	};
	const readOnlySet = new Set<string>([...DEFAULT_READ_ONLY_TOOLS, ...(options.readOnlyTools ?? [])]);
	const maxBytes = options.maxToolOutputBytes ?? capabilities.maxToolOutputBytes ?? 6144;
	const secretRedactionEnabled = options.secretRedactionEnabled !== false;

	const isReadOnlyCall = (toolName: string, args: Readonly<Record<string, unknown>>): boolean => {
		if (readOnlySet.has(toolName)) return true;
		if (toolName === "subagent_delegate") {
			return args.profile === "researcher" || args.profile === "reviewer";
		}
		if (toolName === "subagent_parallel") {
			const tasks = args.tasks;
			if (!Array.isArray(tasks) || tasks.length === 0) return false;
			return tasks.every(
				(item) =>
					item &&
					typeof item === "object" &&
					((item as Record<string, unknown>).profile === "researcher" ||
						(item as Record<string, unknown>).profile === "reviewer"),
			);
		}
		return options.isReadOnlyTool?.(toolName, args) === true;
	};

	const requiredPermissionFor = (toolName: string, args: Readonly<Record<string, unknown>>): string | undefined => {
		if (permissionMap[toolName]) return permissionMap[toolName];
		if (toolName.startsWith("mcp_")) {
			return isReadOnlyCall(toolName, args) ? "mcp:read" : "mcp:write";
		}
		return undefined;
	};

	return defineExtension({
		name: "guard-shaping",
		hooks: [
			hook(ToolTask, {
				beforeTool: async (call, api, context) => {
					const callArgs = (call.arguments ?? {}) as Record<string, unknown>;

					// 1. Defense-in-depth: block any attempt to inject tenant/user identity via args
					for (const key of Object.keys(callArgs)) {
						if (SPOOFED_TENANT_KEYS.has(key)) {
							return {
								block: `Security violation: argument "${key}" is forbidden; tenant scope is bound by runtime.`,
							};
						}
					}

					const meta = await api.snapshot(ConversationMetaDoc, api.conversationId, context);
					const tenant = meta ? deserializeTenantFromDoc(meta) : undefined;
					const tenantId = tenant?.tenantId ?? "default";
					const userId = tenant?.userId ?? "anonymous";

					const redactAuditText = (text: string): string =>
						redactSecrets(text, options.extraRedactionPatterns ? { extraPatterns: options.extraRedactionPatterns } : undefined).text;

					const recordAudit = (
						tier: "allow" | "ask" | "deny",
						decision: "allowed" | "blocked" | "approval_requested" | "approved" | "denied" | "expired",
						reason?: string,
						reviewer?: string,
					): void => {
						try {
							auditStore?.recordAudit({
								tenantId,
								userId,
								conversationId: String(api.conversationId),
								taskId: String(api.taskId),
								callId: call.id,
								toolName: call.name,
								tier,
								decision,
								// Reasons can carry approver notes or policy text: persist them redacted, like approval summaries.
								...(reason ? { reason: redactAuditText(reason) } : {}),
								...(reviewer ? { reviewer } : {}),
								createdAt: api.now(),
							});
						} catch {
							// Audit storage failure must never crash tool dispatch
						}
					};

					// 2. Read-only Plan Mode check (fail-closed: mutating tools deterministically blocked)
					const modeDoc = await api.snapshot(ConversationModeDoc, api.conversationId, context);
					if (modeDoc?.mode === "plan" && !isReadOnlyCall(call.name, callArgs)) {
						const reason = `Plan mode active: mutating tool "${call.name}" is blocked. Finalize your plan with plan_set/plan_view and wait for execution mode.`;
						recordAudit("deny", "blocked", reason);
						return { block: reason };
					}

					// 3. RBAC permission check
					const requiredPermission = requiredPermissionFor(call.name, callArgs);
					if (requiredPermission) {
						if (!tenant) {
							const reason = `Missing conversation tenant metadata for authorization check on "${call.name}"`;
							recordAudit("deny", "blocked", reason);
							return { block: reason };
						}
						if (!hasPermission(tenant, requiredPermission)) {
							const reason = `Authorization denied: missing permission "${requiredPermission}" for tool "${call.name}"`;
							recordAudit("deny", "blocked", reason);
							return { block: reason };
						}
					}

					// 4. Three-tier permission policy (`allow` | `ask` | `deny`)
					const policyDecision = evaluatePermissionPolicy(call.name, permissionPolicy);
					if (policyDecision.tier === "deny") {
						const reason = policyDecision.reason ?? `Tool "${call.name}" is denied by permission policy.`;
						recordAudit("deny", "blocked", reason);
						return { block: reason };
					}

					if (policyDecision.tier === "ask") {
						const signalKey = await api.memo("tool_gate.key", `gate_${call.id}`, context);
						const signalState = await api.readSignal(ToolGateApprovalSignal, signalKey, context);

						if (signalState !== undefined) {
							try {
								auditStore?.deletePendingGateApproval(String(api.taskId));
							} catch {
								// best effort cleanup
							}

							if ("expired" in signalState) {
								const reason = `Approval gate for tool "${call.name}" expired without a decision.`;
								recordAudit("ask", "expired", reason);
								return { block: reason };
							}

							const decision = signalState.value;
							const reviewer = decision.reviewer ?? "reviewer";
							const notePart = decision.note ? ` Note: ${decision.note}` : "";
							if (!decision.approved) {
								const reason = `Permission denied by ${reviewer} for tool "${call.name}".${notePart}`;
								recordAudit("ask", "denied", reason, reviewer);
								return { block: reason };
							}

							recordAudit("ask", "approved", `Approved by ${reviewer}.${notePart}`, reviewer);
							return undefined;
						}

						// No decision yet: record pending gate approval and durably park before tool executes
						const createdAt = api.now();
						const summary = `${policyDecision.reason ?? `Approve execution of "${call.name}"`} args=${summarizeArgs(callArgs, options.extraRedactionPatterns)}`;
						try {
							auditStore?.upsertPendingGateApproval({
								taskId: String(api.taskId),
								tenantId,
								userId,
								conversationId: String(api.conversationId),
								callId: call.id,
								toolName: call.name,
								action: `tool_gate:${call.name}`,
								summary,
								signalName: ToolGateApprovalSignal.name,
								signalKey,
								createdAt,
							});
						} catch {
							// best effort
						}
						recordAudit("ask", "approval_requested", summary);

						return {
							sleep: {
								signals: [ToolGateApprovalSignal.key(signalKey)],
								until: createdAt + policyDecision.approvalTimeoutMs,
							},
						};
					}

					if (policyDecision.matchedRule) {
						recordAudit("allow", "allowed", policyDecision.reason);
					}
					return undefined;
				},

				afterTool: async (call, result, api, context) => {
					// Clean up any pending gate record if the task settled/aborted/errored
					try {
						auditStore?.deletePendingGateApproval(String(api.taskId));
					} catch {
						// best effort
					}

					if (!result.output || result.output.length === 0) {
						return undefined;
					}

					const textBlocks = result.output.filter((b) => b.type === "text");
					if (textBlocks.length === 0) return undefined;

					let fullText = contentText(textBlocks);
					let mutated = false;
					const extraDiagnostics: NonNullable<ToolExecutionResult["diagnostics"]>[number][] = [];
					const extraDetails: Record<string, unknown> = {};

					// 1. Secret & credential redaction (runs BEFORE artifact persistence AND for nested codemode calls)
					if (secretRedactionEnabled) {
						const redaction = redactSecrets(fullText, {
							...(options.extraRedactionPatterns ? { extraPatterns: options.extraRedactionPatterns } : {}),
						});
						if (redaction.redactedCount > 0) {
							fullText = redaction.text;
							mutated = true;
							extraDetails.secretsRedacted = redaction.redactedCount;
							extraDetails.secretCategories = [...redaction.categories];
							extraDiagnostics.push({
								severity: "warn",
								code: "secret_redacted",
								message: `Redacted ${redaction.redactedCount} sensitive value(s) (${redaction.categories.join(", ")}) from "${call.name}" output.`,
							});
						}
					}

					const nonTextBlocks = result.output.filter((b) => b.type !== "text");
					const existingDetails =
						result.details !== undefined &&
						result.details !== null &&
						typeof result.details === "object" &&
						!Array.isArray(result.details)
							? (result.details as Record<string, unknown>)
							: {};

					// Nested calls (codemode scripts) hand their value to the script, not the transcript,
					// so offloading their text would only litter the artifact store (secret redaction still applies!).
					if (call.parent) {
						if (!mutated) return undefined;
						return {
							...result,
							output: [{ type: "text", text: fullText }, ...nonTextBlocks],
							details: { ...existingDetails, ...extraDetails } as ToolExecutionResult["details"],
							diagnostics: [...(result.diagnostics ?? []), ...extraDiagnostics],
						};
					}

					// 2. 6KB Tool output cap + artifact offload
					const totalBytes = Buffer.byteLength(fullText, "utf8");
					if (totalBytes <= maxBytes) {
						if (!mutated) return undefined;
						return {
							...result,
							output: [{ type: "text", text: fullText }, ...nonTextBlocks],
							details: { ...existingDetails, ...extraDetails } as ToolExecutionResult["details"],
							diagnostics: [...(result.diagnostics ?? []), ...extraDiagnostics],
						};
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

					const shaped: ToolExecutionResult = {
						...result,
						output: [{ type: "text", text: shapedText }, ...nonTextBlocks],
						details: {
							...existingDetails,
							...extraDetails,
							offloadedArtifactId: artifact.id,
							originalBytes: totalBytes,
							cappedBytes: maxBytes,
						} as ToolExecutionResult["details"],
						diagnostics: [
							...(result.diagnostics ?? []),
							...extraDiagnostics,
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
