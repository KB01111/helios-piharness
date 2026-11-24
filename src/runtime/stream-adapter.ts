import { BACKGROUND_CONTEXT, withAbortSignal } from "@earendil-works/chord/context";
import { isBase, type Op } from "@earendil-works/chord/delta";
import { contentText, type Message } from "@earendil-works/pi-ai";
import {
	type Conversation,
	type ConversationView,
	type EntryRecord,
	type LiveState,
	type ToolSlot,
	type UsageState,
} from "@earendil-works/pi-durable";
import type {
	HarnessEvent,
	LiveStatusView,
	PendingApproval,
	ToolSlotView,
	TranscriptEntryView,
	UsageBucketView,
	UsageSummaryView,
} from "../contracts/harness-port.js";

export function entryToView(entry: EntryRecord): TranscriptEntryView {
	const firstMsg = entry.model?.[0] as Message | undefined;
	if (!firstMsg) {
		return {
			id: String(entry.id),
			kind: entry.kind,
			...(entry.data !== undefined ? { data: entry.data } : {}),
		};
	}

	if (firstMsg.role === "user") {
		return {
			id: String(entry.id),
			kind: entry.kind,
			role: "user",
			text: contentText(firstMsg.content),
			...(entry.data !== undefined ? { data: entry.data } : {}),
		};
	}

	if (firstMsg.role === "assistant") {
		return {
			id: String(entry.id),
			kind: entry.kind,
			role: "assistant",
			text: contentText(firstMsg.content),
			...(entry.data !== undefined ? { data: entry.data } : {}),
		};
	}

	if (firstMsg.role === "toolResult") {
		return {
			id: String(entry.id),
			kind: entry.kind,
			role: "toolResult",
			text: contentText(firstMsg.content),
			toolName: firstMsg.toolName,
			toolCallId: firstMsg.toolCallId,
			isError: firstMsg.isError,
			...(entry.data !== undefined ? { data: entry.data } : {}),
		};
	}

	return {
		id: String(entry.id),
		kind: entry.kind,
		role: "system",
		...(entry.data !== undefined ? { data: entry.data } : {}),
	};
}

export function usageToView(usage: UsageState | undefined): UsageSummaryView {
	const mapBuckets = (
		raw: Record<string, { input: number; output: number; cacheRead: number; cacheWrite: number; totalTokens: number; cost: { total: number } }> | undefined,
	): Record<string, UsageBucketView> => {
		const out: Record<string, UsageBucketView> = {};
		if (!raw) return out;
		for (const [key, u] of Object.entries(raw)) {
			out[key] = {
				input: u.input,
				output: u.output,
				cacheRead: u.cacheRead,
				cacheWrite: u.cacheWrite,
				totalTokens: u.totalTokens,
				totalCost: u.cost?.total ?? 0,
			};
		}
		return out;
	};

	return {
		models: mapBuckets(usage?.models as never),
		tools: mapBuckets(usage?.tools as never),
	};
}

function slotToView(slot: ToolSlot): ToolSlotView {
	return {
		callId: slot.callId,
		...(slot.taskId !== undefined ? { taskId: String(slot.taskId) } : {}),
		toolName: slot.name,
		status: slot.status,
		...(slot.output !== undefined ? { output: slot.output } : {}),
		...(slot.entry !== undefined ? { entryId: String(slot.entry) } : {}),
	};
}

function extractLiveView(view: ConversationView): LiveStatusView {
	const live = view.docs["pi.live"] as LiveState | undefined;
	const partialMsg = live?.generation?.message;
	const streamingText = partialMsg ? contentText(partialMsg.content as never) : undefined;
	return {
		busy: live?.run !== undefined,
		turns: live?.run?.turns ?? 0,
		...(streamingText ? { streamingText } : {}),
		tools: (live?.tools ?? []).map(slotToView),
		...(live?.runFailure
			? {
					runFailure: {
						reason: live.runFailure.reason,
						...(live.runFailure.detail !== undefined ? { detail: live.runFailure.detail } : {}),
					},
				}
			: {}),
	};
}

function extractApprovalFromSlot(conversationId: string, slot: ToolSlot): PendingApproval | undefined {
	if (slot.status !== "sleeping" || slot.taskId === undefined) return undefined;
	const details = slot.details as Record<string, unknown> | undefined;
	if (!details || details.kind !== "approval_request") return undefined;
	return {
		taskId: String(slot.taskId),
		conversationId,
		callId: slot.callId,
		toolName: slot.name,
		action: typeof details.action === "string" ? details.action : "approve",
		summary: typeof details.summary === "string" ? details.summary : "",
		signalName: typeof details.signalName === "string" ? details.signalName : "app.approval.decision",
		signalKey: typeof details.signalKey === "string" ? details.signalKey : "decision",
		createdAt: typeof details.createdAt === "number" ? details.createdAt : Date.now(),
	};
}

/**
 * Adapts Pi Durable's `Conversation.watch()` structural view + Chord `Op[]`
 * deltas into a transport-agnostic `AsyncIterable<HarnessEvent>`.
 */
export function watchConversationEvents(
	conversation: Conversation,
	signal?: AbortSignal,
): AsyncIterable<HarnessEvent> {
	const conversationId = String(conversation.id);

	return {
		async *[Symbol.asyncIterator](): AsyncGenerator<HarnessEvent> {
			const ctx = signal ? withAbortSignal(signal, BACKGROUND_CONTEXT) : BACKGROUND_CONTEXT;
			const watch = await conversation.watch(ctx);

			const queue: HarnessEvent[] = [];
			let waitingResolve: (() => void) | null = null;
			let ended = false;

			const push = (event: HarnessEvent): void => {
				queue.push(event);
				if (waitingResolve) {
					const resolve = waitingResolve;
					waitingResolve = null;
					resolve();
				}
			};

			const initial = watch.value;
			const initialLive = extractLiveView(initial);
			const initialUsage = usageToView(initial.docs["pi.usage"] as UsageState | undefined);

			let lastEntryId = initial.entries.length > 0 ? initial.entries[initial.entries.length - 1]!.id : 0;
			let lastStreamText = initialLive.streamingText ?? "";
			let lastBusy = initialLive.busy;
			let lastTurns = initialLive.turns;
			let lastFailureKey = initialLive.runFailure ? JSON.stringify(initialLive.runFailure) : "";
			const lastToolSnapshots = new Map<string, string>();
			const emittedApprovals = new Set<string>();

			for (const slot of initialLive.tools) {
				lastToolSnapshots.set(slot.callId, JSON.stringify(slot));
			}

			push({
				type: "snapshot",
				conversationId,
				entries: initial.entries.map(entryToView),
				live: initialLive,
				usage: initialUsage,
			});

			const processFrame = (view: ConversationView, ops: readonly Op[]): void => {
				// 1. Chord ops summary event
				if (ops.length > 0) {
					push({
						type: "delta_ops",
						conversationId,
						opCount: ops.length,
						replaced: isBase(ops),
					});
				}

				// 2. Newly appended entries
				for (const entry of view.entries) {
					if (entry.id > lastEntryId) {
						lastEntryId = entry.id;
						push({
							type: "entry_appended",
							conversationId,
							entry: entryToView(entry),
						});
					}
				}

				// 3. Streaming text delta
				const nextLive = extractLiveView(view);
				const nextStreamText = nextLive.streamingText ?? "";
				if (nextStreamText.length > lastStreamText.length && nextStreamText.startsWith(lastStreamText)) {
					const delta = nextStreamText.slice(lastStreamText.length);
					push({
						type: "text_delta",
						conversationId,
						delta,
						fullText: nextStreamText,
					});
				} else if (nextStreamText.length > 0 && nextStreamText !== lastStreamText) {
					push({
						type: "text_delta",
						conversationId,
						delta: nextStreamText,
						fullText: nextStreamText,
					});
				}
				lastStreamText = nextStreamText;

				// 4. Tool slot updates + approval requests
				const rawSlots = ((view.docs["pi.live"] as LiveState | undefined)?.tools ?? []) as readonly ToolSlot[];
				for (const rawSlot of rawSlots) {
					const slotView = slotToView(rawSlot);
					const serialized = JSON.stringify(slotView);
					if (lastToolSnapshots.get(rawSlot.callId) !== serialized) {
						lastToolSnapshots.set(rawSlot.callId, serialized);
						push({
							type: "tool_update",
							conversationId,
							slot: slotView,
						});
					}
					const approval = extractApprovalFromSlot(conversationId, rawSlot);
					if (approval && !emittedApprovals.has(approval.taskId)) {
						emittedApprovals.add(approval.taskId);
						push({
							type: "approval_requested",
							conversationId,
							approval,
						});
					}
				}

				// 5. Run busy / failure transitions
				const nextFailureKey = nextLive.runFailure ? JSON.stringify(nextLive.runFailure) : "";
				if (nextLive.busy !== lastBusy || nextLive.turns !== lastTurns || nextFailureKey !== lastFailureKey) {
					lastBusy = nextLive.busy;
					lastTurns = nextLive.turns;
					lastFailureKey = nextFailureKey;
					push({
						type: "run_status",
						conversationId,
						busy: nextLive.busy,
						turns: nextLive.turns,
						...(nextLive.runFailure ? { failure: nextLive.runFailure } : {}),
					});
				}
			};

			watch.start(async (call) => {
				if (call.kind === "value") {
					processFrame(call.value, call.ops);
				}
			});

			void watch.closed.then((end) => {
				ended = true;
				push({
					type: "stream_closed",
					conversationId,
					reason: end.kind,
				});
			});

			try {
				while (true) {
					while (queue.length > 0) {
						const event = queue.shift()!;
						yield event;
						if (event.type === "stream_closed") {
							return;
						}
					}
					if (ended) return;
					await new Promise<void>((resolve) => {
						waitingResolve = resolve;
					});
				}
			} finally {
				await watch.stop();
			}
		},
	};
}
