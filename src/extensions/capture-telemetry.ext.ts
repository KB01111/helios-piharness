import { contentText, type AssistantMessage, type ToolCall, type ToolResultMessage } from "@earendil-works/pi-ai";
import {
	AssistantEntry,
	defineExtension,
	defineTask,
	GenerationTask,
	hook,
	hookTask,
	ToolResultEntry,
	ToolTask,
	UserEntry,
	type EntryId,
	type EntryRecord,
	type Extension,
} from "@earendil-works/pi-durable";
import type { AppCapabilities, ToolTraceItem } from "../contracts/app-capabilities.js";
import { ConversationMetaDoc, RunTelemetryDoc } from "../harness/app-documents.js";
import { detectFrictionSignals } from "../memory/friction-signals.js";

export type CaptureEpisodeInput = {
	readonly first: EntryId;
	readonly answer: EntryId;
	readonly last: EntryId;
	readonly reason: "answer" | "terminate";
};

type CaptureEpisodeCheckpoint = {
	phase: "capture";
};

export type CaptureEpisodeResult = {
	readonly episodeId?: string;
	readonly frictionScore: number;
	readonly toolCount: number;
};

function buildToolTrace(entries: readonly EntryRecord[]): ToolTraceItem[] {
	const callById = new Map<string, ToolCall>();

	for (const entry of entries) {
		if (!AssistantEntry.is(entry)) continue;
		const msg = entry.model?.[0] as AssistantMessage | undefined;
		if (!msg || msg.role !== "assistant") continue;
		for (const block of msg.content) {
			if (block.type === "toolCall") {
				callById.set(block.id, block);
			}
		}
	}

	const trace: ToolTraceItem[] = [];
	for (const entry of entries) {
		if (!ToolResultEntry.is(entry)) continue;
		const msg = entry.model?.[0] as ToolResultMessage | undefined;
		if (!msg || msg.role !== "toolResult") continue;
		const call = callById.get(msg.toolCallId);
		const diagnostics = entry.data?.diagnostics ?? [];
		const diagnosticCodes = diagnostics
			.map((d) => d.code)
			.filter((code): code is string => typeof code === "string" && code.length > 0);

		trace.push({
			callId: msg.toolCallId,
			toolName: msg.toolName,
			argumentsJson: JSON.stringify(call?.arguments ?? {}),
			isError: msg.isError,
			diagnosticCodes,
			...(msg.durationMs !== undefined ? { durationMs: msg.durationMs } : {}),
		});
	}

	return trace;
}

function extractUserAndAssistantText(entries: readonly EntryRecord[], answerEntryId: EntryId): {
	userMessage: string;
	assistantSummary: string;
} {
	const userParts: string[] = [];
	let assistantSummary = "";

	for (const entry of entries) {
		const msg = entry.model?.[0];
		if (!msg) continue;
		if (UserEntry.is(entry) || msg.role === "user") {
			const text = contentText(msg.content as never).trim();
			if (text) userParts.push(text);
		}
		if (entry.id === answerEntryId && msg.role === "assistant") {
			assistantSummary = contentText(msg.content).trim().slice(0, 1200);
		}
	}

	return {
		userMessage: userParts.join("\n").slice(0, 2000),
		assistantSummary,
	};
}

/**
 * Isolated Extension (Loop 1): Passive mid-term capture, deterministic friction
 * tagging, and Skill Dojo per-tool/skill health stats.
 *
 * Zero LLM calls occur inside Loop 1.
 */
export function createCaptureTelemetryExtension(capabilities: AppCapabilities): Extension {
	const CaptureEpisodeTask = defineTask<CaptureEpisodeInput, CaptureEpisodeCheckpoint, CaptureEpisodeResult>({
		name: "app.capture-episode",
		version: 1,
		initial: () => ({ phase: "capture" }),
		phases: {
			capture: async (task, runtime, context) => {
				const meta = await runtime.snapshot(ConversationMetaDoc, runtime.conversationId, context);
				if (!meta) {
					await runtime.commit(
						() => ({
							status: "terminal",
							outcome: { status: "completed", result: { frictionScore: 0, toolCount: 0 } },
						}),
						context,
					);
					return;
				}

				const allEntries: EntryRecord[] = [];
				let cursor: Awaited<ReturnType<typeof runtime.entries>>["next"];
				do {
					const page = await runtime.entries(
						{
							minEntryId: task.input.first,
							maxEntryId: task.input.last,
							order: "ascending",
						},
						256,
						cursor,
						context,
					);
					allEntries.push(...page.items);
					cursor = page.next;
				} while (cursor !== undefined);

				const toolTrace = buildToolTrace(allEntries);
				const { userMessage, assistantSummary } = extractUserAndAssistantText(allEntries, task.input.answer);
				const friction = detectFrictionSignals({
					userMessage,
					assistantSummary,
					toolTrace,
				});

				const runId = `run:${task.input.first}:${task.input.last}`;
				const timestamp = runtime.now();

				const episode = await capabilities.memory.recordEpisode({
					tenantId: meta.tenant.tenantId,
					userId: meta.tenant.userId,
					conversationId: String(runtime.conversationId),
					runId,
					userMessage,
					assistantSummary,
					toolTrace,
					frictionSignals: friction.signals,
					frictionScore: friction.score,
					createdAt: timestamp,
				});

				await capabilities.telemetry.recordRunTelemetry({
					tenantId: meta.tenant.tenantId,
					userId: meta.tenant.userId,
					conversationId: String(runtime.conversationId),
					runId,
					toolNames: toolTrace.map((t) => t.toolName),
					frictionScore: friction.score,
					frictionKinds: Array.from(new Set(friction.signals.map((s) => s.kind))),
					createdAt: timestamp,
				});

				await runtime.commit(async (tx) => {
					const doc = await tx.doc(RunTelemetryDoc, runtime.conversationId);
					doc.totalRunsCaptured += 1;
					doc.lastCapturedEntryId = Math.max(doc.lastCapturedEntryId, Number(task.input.last));
					return {
						status: "terminal",
						outcome: {
							status: "completed",
							result: {
								episodeId: episode.id,
								frictionScore: friction.score,
								toolCount: toolTrace.length,
							},
						},
					};
				}, context);
			},
		},
		abort: async (_task, runtime, context) => {
			await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), context);
		},
	});

	return defineExtension({
		name: "capture-telemetry",
		tasks: [CaptureEpisodeTask],
		hooks: [
			hook(ToolTask, {
				afterTool: async (call, result, api, context) => {
					const meta = await api.snapshot(ConversationMetaDoc, api.conversationId, context);
					if (!meta) return undefined;

					const firstDiagCode = result.diagnostics?.find((d) => d.code !== undefined)?.code;
					const skillName =
						call.name === "skill_read" && typeof call.arguments?.name === "string"
							? call.arguments.name
							: undefined;

					await capabilities.telemetry.recordToolInvocation({
						tenantId: meta.tenant.tenantId,
						userId: meta.tenant.userId,
						conversationId: String(api.conversationId),
						callId: call.id,
						toolName: call.name,
						...(skillName ? { skillName } : {}),
						isError: result.isError === true,
						...(firstDiagCode ? { diagnosticCode: firstDiagCode } : {}),
						createdAt: api.now(),
					});

					return undefined;
				},
			}),
			hook(GenerationTask, {
				afterRun: async (run) => {
					return {
						tasks: [
							hookTask(CaptureEpisodeTask, {
								first: run.first,
								answer: run.answer,
								last: run.last,
								reason: run.reason,
							}),
						],
					};
				},
			}),
		],
	});
}
