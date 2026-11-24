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
	StreamCloseReason,
	ToolSlotView,
	TranscriptEntryView,
	UsageBucketView,
	UsageSummaryView,
	WatchConversationOptions,
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

/**
 * Approvals without an explicit `details.createdAt` get a first-seen timestamp that is remembered
 * process-wide (bounded) so reconnecting streams report the same, stable `createdAt`.
 */
const APPROVAL_FIRST_SEEN_MAX = 10_000;
const approvalFirstSeen = new Map<string, number>();

function stableApprovalCreatedAt(conversationId: string, taskId: string): number {
	const key = `${conversationId}:${taskId}`;
	const existing = approvalFirstSeen.get(key);
	if (existing !== undefined) return existing;
	const now = Date.now();
	if (approvalFirstSeen.size >= APPROVAL_FIRST_SEEN_MAX) {
		const oldest = approvalFirstSeen.keys().next();
		if (!oldest.done) approvalFirstSeen.delete(oldest.value);
	}
	approvalFirstSeen.set(key, now);
	return now;
}

function extractApprovalFromSlot(conversationId: string, slot: ToolSlot): PendingApproval | undefined {
	if (slot.status !== "sleeping" || slot.taskId === undefined) return undefined;
	const details = slot.details as Record<string, unknown> | undefined;
	if (!details || details.kind !== "approval_request") return undefined;
	const taskId = String(slot.taskId);
	return {
		taskId,
		conversationId,
		callId: slot.callId,
		toolName: slot.name,
		action: typeof details.action === "string" ? details.action : "approve",
		summary: typeof details.summary === "string" ? details.summary : "",
		signalName: typeof details.signalName === "string" ? details.signalName : "app.approval.decision",
		signalKey: typeof details.signalKey === "string" ? details.signalKey : "decision",
		createdAt:
			typeof details.createdAt === "number" && Number.isFinite(details.createdAt)
				? details.createdAt
				: stableApprovalCreatedAt(conversationId, taskId),
	};
}

/** Default max buffered events per watch stream. Override per call or with `HARNESS_STREAM_MAX_QUEUE`. */
export const DEFAULT_STREAM_MAX_QUEUE = 1000;

function resolveMaxQueue(option: number | undefined): number {
	if (option !== undefined && Number.isFinite(option) && option >= 1) return Math.floor(option);
	const fromEnv = Number(process.env.HARNESS_STREAM_MAX_QUEUE);
	if (Number.isFinite(fromEnv) && fromEnv >= 1) return Math.floor(fromEnv);
	return DEFAULT_STREAM_MAX_QUEUE;
}

/** Parse a resume cursor; returns undefined (=> full snapshot) when unparseable or beyond the known entries. */
function resolveResumeCursor(afterEntryId: string | undefined, lastKnownEntryId: number): number | undefined {
	if (afterEntryId === undefined) return undefined;
	const trimmed = afterEntryId.trim();
	if (!/^\d+$/.test(trimmed)) return undefined;
	const n = Number(trimmed);
	if (!Number.isSafeInteger(n) || n > lastKnownEntryId) return undefined;
	return n;
}

type CoalescibleEvent = Extract<HarnessEvent, { type: "text_delta" | "tool_update" | "delta_ops" }>;

function isCoalescible(event: HarnessEvent): event is CoalescibleEvent {
	return event.type === "text_delta" || event.type === "tool_update" || event.type === "delta_ops";
}

/** Merge `next` into `prev` when they describe the same stream of changes; undefined when not mergeable. */
function mergeEvents(prev: CoalescibleEvent, next: CoalescibleEvent): HarnessEvent | undefined {
	if (prev.type !== next.type) return undefined;
	if (prev.type === "delta_ops" && next.type === "delta_ops") {
		return { ...next, opCount: prev.opCount + next.opCount, replaced: prev.replaced || next.replaced };
	}
	if (prev.type === "tool_update" && next.type === "tool_update") {
		return prev.slot.callId === next.slot.callId ? next : undefined;
	}
	if (prev.type === "text_delta" && next.type === "text_delta") {
		// Continuation: keep the latest fullText, concatenate the deltas. Otherwise (reset) the newest wins.
		if (prev.fullText + next.delta === next.fullText) {
			return { ...next, delta: prev.delta + next.delta };
		}
		return next;
	}
	return undefined;
}

/**
 * Bounded, coalescing event buffer. Only the trailing run of coalescible events (text_delta / tool_update /
 * delta_ops) is merged, so ordering relative to snapshot / entry_appended / approval / run_status events is
 * preserved. A hard overflow closes the stream with `stream_closed` reason "overflow".
 */
class BoundedEventQueue {
	private items: HarnessEvent[] = [];
	private head = 0;
	private wake: (() => void) | null = null;
	private closedFlag = false;

	constructor(
		private readonly conversationId: string,
		private readonly limit: number,
		private readonly onClose: (reason: StreamCloseReason) => void,
	) {}

	get closed(): boolean {
		return this.closedFlag;
	}

	get size(): number {
		return this.items.length - this.head;
	}

	push(event: HarnessEvent): void {
		if (this.closedFlag) return;
		if (event.type === "stream_closed") {
			this.items.push(event);
			this.finish(event.reason);
			return;
		}
		if (isCoalescible(event)) {
			for (let i = this.items.length - 1; i >= this.head; i--) {
				const candidate = this.items[i]!;
				if (!isCoalescible(candidate)) break;
				const merged = mergeEvents(candidate, event);
				if (merged) {
					this.items[i] = merged;
					this.signal();
					return;
				}
			}
		}
		if (this.size >= this.limit) {
			// Hard overflow: stop buffering, let the client reconnect and resume from its last entry id.
			this.close("overflow");
			return;
		}
		this.items.push(event);
		this.signal();
	}

	/** Terminate the stream with a final `stream_closed`; optionally discard anything still buffered. */
	close(reason: StreamCloseReason, discardPending = false): void {
		if (this.closedFlag) return;
		if (discardPending) {
			this.items = [];
			this.head = 0;
		}
		this.items.push({ type: "stream_closed", conversationId: this.conversationId, reason });
		this.finish(reason);
	}

	shift(): HarnessEvent | undefined {
		if (this.head >= this.items.length) return undefined;
		const event = this.items[this.head]!;
		this.head++;
		if (this.head > 64 && this.head * 2 >= this.items.length) {
			this.items = this.items.slice(this.head);
			this.head = 0;
		}
		return event;
	}

	waitForEvent(): Promise<void> {
		if (this.size > 0) return Promise.resolve();
		return new Promise<void>((resolve) => {
			this.wake = resolve;
		});
	}

	private finish(reason: StreamCloseReason): void {
		this.closedFlag = true;
		this.signal();
		this.onClose(reason);
	}

	private signal(): void {
		const wake = this.wake;
		if (wake) {
			this.wake = null;
			wake();
		}
	}
}

interface ToolSlotFingerprint {
	taskId: unknown;
	name: string;
	status: string;
	output: string | undefined;
	entry: unknown;
}

/**
 * Adapts Pi Durable's `Conversation.watch()` structural view + Chord `Op[]`
 * deltas into a transport-agnostic `AsyncIterable<HarnessEvent>`.
 *
 * Resume contract: `entry_appended` events carry `entry.id`, which transports should use as the SSE
 * event id. A reconnecting client sends it back (`Last-Event-ID`) as `options.afterEntryId`, and the
 * initial `snapshot` then only includes entries with a greater id (live status + usage always included).
 *
 * Backpressure: events are buffered in a bounded queue (`options.maxQueueSize`, env
 * `HARNESS_STREAM_MAX_QUEUE`, default 1000). Consecutive text_delta / tool_update / delta_ops events are
 * coalesced; on hard overflow the stream ends with `stream_closed` reason "overflow".
 */
export function watchConversationEvents(
	conversation: Conversation,
	signal?: AbortSignal,
	options?: WatchConversationOptions,
): AsyncIterable<HarnessEvent> {
	const conversationId = String(conversation.id);

	return {
		async *[Symbol.asyncIterator](): AsyncGenerator<HarnessEvent> {
			if (signal?.aborted) {
				yield { type: "stream_closed", conversationId, reason: "aborted" };
				return;
			}
			const ctx = signal ? withAbortSignal(signal, BACKGROUND_CONTEXT) : BACKGROUND_CONTEXT;
			const watch = await conversation.watch(ctx);

			let stopRequested = false;
			const stopWatch = (): void => {
				if (stopRequested) return;
				stopRequested = true;
				void watch.stop().catch(() => undefined);
			};
			const queue = new BoundedEventQueue(conversationId, resolveMaxQueue(options?.maxQueueSize), (reason) => {
				// Overflow / abort: stop producing right away; the consumer drains what is left.
				if (reason === "overflow" || reason === "aborted") stopWatch();
			});
			const onAbort = (): void => queue.close("aborted", true);

			try {
				if (signal) {
					signal.addEventListener("abort", onAbort, { once: true });
					if (signal.aborted) onAbort();
				}

				const initial = watch.value;
				const initialLive = extractLiveView(initial);
				const initialUsage = usageToView(initial.docs["pi.usage"] as UsageState | undefined);
				const initialLiveDoc = initial.docs["pi.live"] as LiveState | undefined;

				let lastEntryId = initial.entries.length > 0 ? Number(initial.entries[initial.entries.length - 1]!.id) : 0;
				const resumeAfter = resolveResumeCursor(options?.afterEntryId, lastEntryId);
				const snapshotEntries =
					resumeAfter === undefined ? initial.entries : initial.entries.filter((e) => Number(e.id) > resumeAfter);

				let lastLiveDoc = initialLiveDoc;
				let lastPartial = initialLiveDoc?.generation?.message;
				let lastPartialText = initialLive.streamingText ?? "";
				let lastStreamText = lastPartialText;
				let lastAttempt = initialLiveDoc?.generation?.attempt;
				let lastBusy = initialLive.busy;
				let lastTurns = initialLive.turns;
				let lastFailureReason = initialLive.runFailure?.reason;
				let lastFailureDetail = initialLive.runFailure?.detail;
				const lastToolSlots = new Map<string, ToolSlotFingerprint>();
				const emittedApprovals = new Set<string>();

				const fingerprint = (slot: ToolSlot): ToolSlotFingerprint => ({
					taskId: slot.taskId,
					name: slot.name,
					status: slot.status,
					output: slot.output,
					entry: slot.entry,
				});
				for (const slot of initialLiveDoc?.tools ?? []) {
					lastToolSlots.set(slot.callId, fingerprint(slot));
				}

				queue.push({
					type: "snapshot",
					conversationId,
					entries: snapshotEntries.map(entryToView),
					live: initialLive,
					usage: initialUsage,
				});

				const processFrame = (view: ConversationView, ops: readonly Op[]): void => {
					if (queue.closed) return;

					// 1. Chord ops summary event
					if (ops.length > 0) {
						queue.push({
							type: "delta_ops",
							conversationId,
							opCount: ops.length,
							replaced: isBase(ops),
						});
					}

					// 2. Newly appended entries (entries are ascending; walk back to the first new one).
					// The entry id doubles as the SSE event id / resume cursor for reconnecting clients.
					const entries = view.entries;
					let firstNew = entries.length;
					while (firstNew > 0 && Number(entries[firstNew - 1]!.id) > lastEntryId) firstNew--;
					const entryAppended = firstNew < entries.length;
					for (let i = firstNew; i < entries.length; i++) {
						const entry = entries[i]!;
						lastEntryId = Number(entry.id);
						queue.push({
							type: "entry_appended",
							conversationId,
							entry: entryToView(entry),
						});
					}

					const live = view.docs["pi.live"] as LiveState | undefined;
					if (live === lastLiveDoc && !entryAppended) return; // nothing live changed
					lastLiveDoc = live;

					// 3. Streaming text delta
					const partial = live?.generation?.message;
					if (partial !== lastPartial) {
						lastPartial = partial;
						lastPartialText = partial ? contentText(partial.content as never) : "";
					}
					const nextStreamText = lastPartialText;
					const attempt = live?.generation?.attempt;
					// A new generation (retry attempt) or a freshly committed entry starts a new streamed message:
					// never diff it against the previous message's text.
					if (entryAppended || attempt !== lastAttempt) lastStreamText = "";
					lastAttempt = attempt;
					if (nextStreamText.length > lastStreamText.length && nextStreamText.startsWith(lastStreamText)) {
						queue.push({
							type: "text_delta",
							conversationId,
							delta: nextStreamText.slice(lastStreamText.length),
							fullText: nextStreamText,
						});
					} else if (nextStreamText.length > 0 && nextStreamText !== lastStreamText) {
						queue.push({
							type: "text_delta",
							conversationId,
							delta: nextStreamText,
							fullText: nextStreamText,
						});
					}
					lastStreamText = nextStreamText;

					// 4. Tool slot updates + approval requests (field-wise change detection, no serialization)
					const rawSlots = live?.tools ?? [];
					for (const rawSlot of rawSlots) {
						const prev = lastToolSlots.get(rawSlot.callId);
						if (
							!prev ||
							prev.status !== rawSlot.status ||
							prev.output !== rawSlot.output ||
							prev.entry !== rawSlot.entry ||
							prev.taskId !== rawSlot.taskId ||
							prev.name !== rawSlot.name
						) {
							lastToolSlots.set(rawSlot.callId, fingerprint(rawSlot));
							queue.push({
								type: "tool_update",
								conversationId,
								slot: slotToView(rawSlot),
							});
						}
						if (rawSlot.status === "sleeping") {
							const approval = extractApprovalFromSlot(conversationId, rawSlot);
							if (approval && !emittedApprovals.has(approval.taskId)) {
								emittedApprovals.add(approval.taskId);
								queue.push({
									type: "approval_requested",
									conversationId,
									approval,
								});
							}
						}
					}
					if (lastToolSlots.size > rawSlots.length) {
						const present = new Set<string>();
						for (const s of rawSlots) present.add(s.callId);
						for (const callId of lastToolSlots.keys()) {
							if (!present.has(callId)) lastToolSlots.delete(callId);
						}
					}

					// 5. Run busy / failure transitions
					const busy = live?.run !== undefined;
					const turns = live?.run?.turns ?? 0;
					const failure = live?.runFailure;
					if (
						busy !== lastBusy ||
						turns !== lastTurns ||
						failure?.reason !== lastFailureReason ||
						failure?.detail !== lastFailureDetail
					) {
						lastBusy = busy;
						lastTurns = turns;
						lastFailureReason = failure?.reason;
						lastFailureDetail = failure?.detail;
						queue.push({
							type: "run_status",
							conversationId,
							busy,
							turns,
							...(failure
								? {
										failure: {
											reason: failure.reason,
											...(failure.detail !== undefined ? { detail: failure.detail } : {}),
										},
									}
								: {}),
						});
					}
				};

				watch.start(async (call) => {
					if (call.kind === "value") {
						processFrame(call.value, call.ops);
					}
				});

				void watch.closed.then(
					(end) => queue.close(end.kind),
					() => queue.close("failed"),
				);

				while (true) {
					const event = queue.shift();
					if (event) {
						yield event;
						if (event.type === "stream_closed") return;
						continue;
					}
					await queue.waitForEvent();
				}
			} finally {
				signal?.removeEventListener("abort", onAbort);
				stopWatch();
				try {
					await watch.stop();
				} catch {
					// already stopped / closed
				}
			}
		},
	};
}
