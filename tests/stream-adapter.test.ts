import path from "node:path";
import { describe, expect, it } from "vitest";
import { watchConversationEvents } from "../src/runtime/stream-adapter.js";
import {
	bootstrapHarness,
	DEFAULT_PERMISSIONS,
	type BootstrappedHarness,
	type HarnessEvent,
	type TenantContext,
} from "../src/index.js";

const TENANT: TenantContext = {
	tenantId: "tenant-stream",
	userId: "user-stream",
	permissions: [...DEFAULT_PERMISSIONS],
};

async function boot(): Promise<BootstrappedHarness> {
	return bootstrapHarness({
		config: {
			piDbPath: ":memory:",
			appDbPath: ":memory:",
			useMockModels: true,
			skillsDir: path.resolve("skills"),
		},
	});
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
	return Promise.race([
		promise,
		new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`timeout: ${label}`)), ms)),
	]);
}

// ---------------------------------------------------------------------------
// Deterministic fake Conversation / watch for unit-level behaviour
// ---------------------------------------------------------------------------

type FakeView = { entries: Array<{ id: number }>; docs: Record<string, unknown> };

function liveDoc(opts: {
	text?: string;
	attempt?: number;
	tools?: unknown[];
	busy?: boolean;
}): Record<string, unknown> {
	return {
		...(opts.busy ? { run: { turns: 1 } } : {}),
		generation: {
			attempt: opts.attempt ?? 1,
			...(opts.text !== undefined ? { message: { role: "assistant", content: [{ type: "text", text: opts.text }] } } : {}),
		},
		tools: opts.tools ?? [],
	};
}

function fakeConversation(initial: FakeView) {
	let listener: ((call: unknown, ctx: unknown) => Promise<void>) | undefined;
	let resolveClosed: (end: { kind: string }) => void = () => undefined;
	const closed = new Promise<{ kind: string }>((resolve) => {
		resolveClosed = resolve;
	});
	const state = { stops: 0, starts: 0 };
	const handle = {
		value: initial,
		start(l: (call: unknown, ctx: unknown) => Promise<void>) {
			state.starts++;
			listener = l;
		},
		async stop() {
			state.stops++;
			resolveClosed({ kind: "stopped" });
			return { kind: "stopped" };
		},
		closed,
	};
	const conversation = {
		id: 7,
		watch: async () => handle,
	};
	return {
		conversation: conversation as never,
		state,
		emit(view: FakeView, ops: unknown[] = [["r", {}]]) {
			handle.value = view;
			return listener?.({ kind: "value", value: view, ops, retired: false }, {});
		},
		end(kind = "ended") {
			resolveClosed({ kind });
		},
	};
}

async function collect(iter: AsyncIterator<HarnessEvent>, n: number): Promise<HarnessEvent[]> {
	const out: HarnessEvent[] = [];
	for (let i = 0; i < n; i++) {
		const r = await withTimeout(iter.next(), 2000, `event ${i}`);
		if (r.done) break;
		out.push(r.value);
	}
	return out;
}

describe("stream adapter (unit, fake watch)", () => {
	it("coalesces text_delta / delta_ops / tool_update instead of growing the queue", async () => {
		const fake = fakeConversation({ entries: [], docs: { "pi.live": liveDoc({ busy: true }) } });
		const iter = watchConversationEvents(fake.conversation, undefined, { maxQueueSize: 4 })[Symbol.asyncIterator]();
		const [snapshot] = await collect(iter, 1);
		expect(snapshot?.type).toBe("snapshot");

		let text = "";
		for (let i = 0; i < 200; i++) {
			text += `w${i} `;
			await fake.emit({
				entries: [],
				docs: {
					"pi.live": liveDoc({
						busy: true,
						text,
						tools: [{ callId: "c1", name: "t", status: "running", output: `out${i}` }],
					}),
				},
			});
		}
		fake.end("ended");

		const events: HarnessEvent[] = [];
		for (;;) {
			const r = await withTimeout(iter.next(), 2000, "drain");
			if (r.done) break;
			events.push(r.value);
		}
		const types = events.map((e) => e.type);
		expect(types.filter((t) => t === "text_delta")).toHaveLength(1);
		expect(types.filter((t) => t === "tool_update")).toHaveLength(1);
		expect(types.filter((t) => t === "delta_ops")).toHaveLength(1);
		const td = events.find((e) => e.type === "text_delta");
		if (td?.type !== "text_delta") throw new Error("no text_delta");
		expect(td.fullText).toBe(text);
		expect(td.delta).toBe(text); // concatenation of all intermediate deltas
		const ops = events.find((e) => e.type === "delta_ops");
		if (ops?.type !== "delta_ops") throw new Error("no delta_ops");
		expect(ops.opCount).toBe(200);
		const tu = events.find((e) => e.type === "tool_update");
		if (tu?.type !== "tool_update") throw new Error("no tool_update");
		expect(tu.slot.output).toBe("out199");
		expect(events.at(-1)).toMatchObject({ type: "stream_closed", reason: "ended" });
	});

	it("closes with reason overflow when non-coalescible events exceed the queue limit", async () => {
		const fake = fakeConversation({ entries: [], docs: { "pi.live": liveDoc({}) } });
		const iter = watchConversationEvents(fake.conversation, undefined, { maxQueueSize: 3 })[Symbol.asyncIterator]();
		await collect(iter, 1); // snapshot; consumer now stalls

		const entries: Array<{ id: number }> = [];
		for (let i = 1; i <= 10; i++) {
			entries.push({ id: i });
			await fake.emit({ entries: [...entries], docs: { "pi.live": liveDoc({}) } }, []);
		}

		const rest: HarnessEvent[] = [];
		for (;;) {
			const r = await withTimeout(iter.next(), 2000, "drain overflow");
			if (r.done) break;
			rest.push(r.value);
		}
		expect(rest.at(-1)).toMatchObject({ type: "stream_closed", reason: "overflow" });
		expect(rest.length).toBeLessThanOrEqual(4);
		// the buffered prefix is contiguous so a client can resume from its last entry id
		const ids = rest.filter((e) => e.type === "entry_appended").map((e) => (e as { entry: { id: string } }).entry.id);
		expect(ids).toEqual(["1", "2", "3"].slice(0, ids.length));
		expect(fake.state.stops).toBeGreaterThanOrEqual(1);
	});

	it("does not duplicate or drop text when a new generation restarts with the same prefix", async () => {
		const fake = fakeConversation({ entries: [], docs: { "pi.live": liveDoc({ busy: true }) } });
		const deltas: Array<{ delta: string; fullText: string }> = [];
		const consumer = (async () => {
			for await (const ev of watchConversationEvents(fake.conversation)) {
				if (ev.type === "text_delta") deltas.push({ delta: ev.delta, fullText: ev.fullText });
			}
		})();
		const tick = () => new Promise((r) => setTimeout(r, 5));
		await tick();

		await fake.emit({ entries: [], docs: { "pi.live": liveDoc({ busy: true, text: "Hi" }) } });
		await tick();
		await fake.emit({ entries: [], docs: { "pi.live": liveDoc({ busy: true, text: "Hi there" }) } });
		await tick();
		// entry committed, then a brand new message starting with the same prefix
		await fake.emit({ entries: [{ id: 1 }], docs: { "pi.live": liveDoc({ busy: true, text: "Hi there" }) } });
		await tick();
		// retry attempt restarts the stream with a shorter text
		await fake.emit({ entries: [{ id: 1 }], docs: { "pi.live": liveDoc({ busy: true, attempt: 2, text: "Hi" }) } });
		await tick();
		fake.end();
		await withTimeout(consumer, 2000, "consumer");

		expect(deltas).toEqual([
			{ delta: "Hi", fullText: "Hi" },
			{ delta: " there", fullText: "Hi there" },
			{ delta: "Hi there", fullText: "Hi there" },
			{ delta: "Hi", fullText: "Hi" },
		]);
	});

	it("gives approvals a stable createdAt (details value, else first-seen, never re-stamped)", async () => {
		const slot = (extra: Record<string, unknown>, taskId: number) => ({
			callId: `c${taskId}`,
			name: "danger",
			status: "sleeping",
			taskId,
			details: { kind: "approval_request", action: "x", ...extra },
		});
		const fake = fakeConversation({ entries: [], docs: { "pi.live": liveDoc({}) } });
		const iter = watchConversationEvents(fake.conversation)[Symbol.asyncIterator]();
		await collect(iter, 1);
		await fake.emit({
			entries: [],
			docs: { "pi.live": liveDoc({ tools: [slot({ createdAt: 1234 }, 501), slot({}, 502)] }) },
		});
		fake.end();
		const approvals: Array<{ taskId: string; createdAt: number }> = [];
		for (;;) {
			const r = await withTimeout(iter.next(), 2000, "drain approvals");
			if (r.done) break;
			if (r.value.type === "approval_requested") approvals.push(r.value.approval);
		}
		expect(approvals.find((x) => x.taskId === "501")?.createdAt).toBe(1234);
		const first = approvals.find((x) => x.taskId === "502")?.createdAt;
		expect(typeof first).toBe("number");

		// second stream (reconnect) observes the same first-seen value
		await new Promise((r) => setTimeout(r, 15));
		const fake2 = fakeConversation({ entries: [], docs: { "pi.live": liveDoc({}) } });
		const iter2 = watchConversationEvents(fake2.conversation)[Symbol.asyncIterator]();
		await collect(iter2, 1);
		await fake2.emit({ entries: [], docs: { "pi.live": liveDoc({ tools: [slot({}, 502)] }) } });
		fake2.end();
		for (;;) {
			const r = await withTimeout(iter2.next(), 2000, "drain approvals 2");
			if (r.done) break;
			if (r.value.type === "approval_requested") expect(r.value.approval.createdAt).toBe(first);
		}
	});

	it("abort ends the iterator promptly while waiting and removes the watch", async () => {
		const fake = fakeConversation({ entries: [], docs: { "pi.live": liveDoc({}) } });
		const ac = new AbortController();
		const iter = watchConversationEvents(fake.conversation, ac.signal)[Symbol.asyncIterator]();
		await collect(iter, 1);
		const pending = iter.next(); // now parked waiting for events
		const started = Date.now();
		ac.abort();
		const r = await withTimeout(pending, 500, "abort next");
		expect(r.done).toBe(false);
		expect(r.value).toMatchObject({ type: "stream_closed", reason: "aborted" });
		const end = await withTimeout(iter.next(), 500, "abort end");
		expect(end.done).toBe(true);
		expect(Date.now() - started).toBeLessThan(500);
		expect(fake.state.stops).toBeGreaterThanOrEqual(1);
	});

	it("an already-aborted signal yields a single aborted close without opening a watch", async () => {
		const fake = fakeConversation({ entries: [], docs: { "pi.live": liveDoc({}) } });
		const ac = new AbortController();
		ac.abort();
		const events: HarnessEvent[] = [];
		for await (const e of watchConversationEvents(fake.conversation, ac.signal)) events.push(e);
		expect(events).toEqual([{ type: "stream_closed", conversationId: "7", reason: "aborted" }]);
		expect(fake.state.starts).toBe(0);
	});

	it("stops the watch when the consumer breaks early", async () => {
		const fake = fakeConversation({ entries: [], docs: { "pi.live": liveDoc({}) } });
		for await (const e of watchConversationEvents(fake.conversation)) {
			expect(e.type).toBe("snapshot");
			break;
		}
		expect(fake.state.stops).toBeGreaterThanOrEqual(1);
	});
});

// ---------------------------------------------------------------------------
// Integration with the real (mock-model) harness
// ---------------------------------------------------------------------------

describe("watchConversation (mock harness)", () => {
	it("resumes with afterEntryId: snapshot only carries newer entries but full live + usage", async () => {
		const runtime = await boot();
		try {
			const conv = await runtime.harness.createConversation({ tenant: TENANT });
			runtime.testModel.enqueueMain({ type: "text", text: "first answer" });
			await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "one" });
			runtime.testModel.enqueueMain({ type: "text", text: "second answer" });
			await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: "two" });

			const transcript = await runtime.harness.getTranscript(conv.conversationId);
			expect(transcript.length).toBeGreaterThanOrEqual(4);

			const snapshotOf = async (afterEntryId?: string) => {
				const ac = new AbortController();
				try {
					for await (const ev of runtime.harness.watchConversation(
						conv.conversationId,
						ac.signal,
						afterEntryId === undefined ? undefined : { afterEntryId },
					)) {
						if (ev.type === "snapshot") return ev;
					}
				} finally {
					ac.abort();
				}
				throw new Error("no snapshot");
			};

			const full = await snapshotOf();
			expect(full.entries.map((e) => e.id)).toEqual(transcript.map((e) => e.id));

			const cursor = transcript[1]!;
			const resumed = await snapshotOf(cursor.id);
			expect(resumed.entries.map((e) => e.id)).toEqual(transcript.slice(2).map((e) => e.id));
			expect(resumed.live).toBeDefined();
			expect(resumed.usage).toBeDefined();
			expect(resumed.live.tools).toEqual(full.live.tools);

			const last = transcript.at(-1)!;
			expect((await snapshotOf(last.id)).entries).toEqual([]);

			// unparseable / unknown cursors fall back to the full transcript
			expect((await snapshotOf("not-a-number")).entries).toHaveLength(transcript.length);
			expect((await snapshotOf("99999999")).entries).toHaveLength(transcript.length);
		} finally {
			await runtime.close();
		}
	});

	it("closes a stalled stream with reason overflow and the client can resume from its last entry id", async () => {
		const runtime = await boot();
		try {
			const conv = await runtime.harness.createConversation({ tenant: TENANT });
			const iter = runtime.harness
				.watchConversation(conv.conversationId, undefined, { maxQueueSize: 3 })
				[Symbol.asyncIterator]();
			const first = await withTimeout(iter.next(), 2000, "snapshot");
			expect(first.value).toMatchObject({ type: "snapshot" });

			// consumer stalls while many entries are produced
			for (let i = 0; i < 4; i++) {
				runtime.testModel.enqueueMain({ type: "text", text: `answer ${i}` });
				await runtime.harness.submitAndWait({ conversationId: conv.conversationId, content: `msg ${i}` });
			}

			const rest: HarnessEvent[] = [];
			for (;;) {
				const r = await withTimeout(iter.next(), 3000, "drain");
				if (r.done) break;
				rest.push(r.value);
			}
			expect(rest.at(-1)).toMatchObject({ type: "stream_closed", reason: "overflow" });
			expect(rest.length).toBeLessThanOrEqual(4);

			const lastSeen = rest
				.filter((e) => e.type === "entry_appended")
				.map((e) => (e as { entry: { id: string } }).entry.id)
				.at(-1);
			const transcript = await runtime.harness.getTranscript(conv.conversationId);
			const ac = new AbortController();
			let resumed: Extract<HarnessEvent, { type: "snapshot" }> | undefined;
			for await (const ev of runtime.harness.watchConversation(conv.conversationId, ac.signal, {
				...(lastSeen ? { afterEntryId: lastSeen } : {}),
			})) {
				if (ev.type === "snapshot") {
					resumed = ev;
					break;
				}
			}
			ac.abort();
			const seenIdx = lastSeen ? transcript.findIndex((e) => e.id === lastSeen) : -1;
			expect(resumed?.entries.map((e) => e.id)).toEqual(transcript.slice(seenIdx + 1).map((e) => e.id));
		} finally {
			await runtime.close();
		}
	});

	it("abort ends a waiting stream promptly and early break does not leak the watch", async () => {
		const runtime = await boot();
		try {
			const conv = await runtime.harness.createConversation({ tenant: TENANT });

			const ac = new AbortController();
			const iter = runtime.harness.watchConversation(conv.conversationId, ac.signal)[Symbol.asyncIterator]();
			await withTimeout(iter.next(), 2000, "snapshot");
			const pending = iter.next();
			ac.abort();
			const r = await withTimeout(pending, 1000, "abort");
			expect(r.value).toMatchObject({ type: "stream_closed", reason: "aborted" });
			expect((await withTimeout(iter.next(), 1000, "done")).done).toBe(true);

			// early break: wrap the real conversation to observe watch.stop()
			const real = await (
				runtime.harness as unknown as { requireConversation(id: string): Promise<{ id: unknown; watch(ctx: unknown): Promise<any> }> }
			).requireConversation(conv.conversationId);
			let stops = 0;
			const spy = {
				id: real.id,
				watch: async (ctx: unknown) => {
					const h = await real.watch(ctx);
					return {
						get value() {
							return h.value;
						},
						start: (l: unknown) => h.start(l),
						stop: () => {
							stops++;
							return h.stop();
						},
						closed: h.closed,
					};
				},
			};
			for await (const ev of watchConversationEvents(spy as never)) {
				expect(ev.type).toBe("snapshot");
				break;
			}
			expect(stops).toBeGreaterThanOrEqual(1);
		} finally {
			await runtime.close();
		}
	});
});
