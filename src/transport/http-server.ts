import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import type { CreateConversationInput, HarnessPort, SubmitMessageInput } from "../contracts/harness-port.js";
import { DEFAULT_PERMISSIONS, type TenantContext } from "../contracts/tenant-context.js";

function extractTenant(
	headers: { get(name: string): string | undefined },
	bodyTenant?: Partial<TenantContext>,
): TenantContext {
	const tenantId = bodyTenant?.tenantId ?? headers.get("x-tenant-id") ?? "default";
	const userId = bodyTenant?.userId ?? headers.get("x-user-id") ?? "anonymous";
	const headerPerms = headers.get("x-permissions");
	const permissions =
		bodyTenant?.permissions ??
		(headerPerms
			? headerPerms
					.split(",")
					.map((p) => p.trim())
					.filter(Boolean)
			: [...DEFAULT_PERMISSIONS]);

	return {
		tenantId,
		userId,
		permissions,
		...(bodyTenant?.workspaceId ? { workspaceId: bodyTenant.workspaceId } : {}),
		...(bodyTenant?.metadata ? { metadata: bodyTenant.metadata } : {}),
	};
}

/**
 * Thin Hono HTTP + native SSE transport wrapping ONLY `HarnessPort`.
 * Zero Pi Durable / Pi AI / Chord imports or leakage.
 */
export function createHttpServer(harness: HarnessPort): Hono {
	const app = new Hono();

	app.onError((err, c) => {
		const message = err instanceof Error ? err.message : String(err);
		const status = message.includes("not found") ? 404 : message.includes("Invalid") ? 400 : 500;
		return c.json({ error: message }, status);
	});

	app.get("/health", (c) => {
		return c.json({ status: "ok", timestamp: Date.now() });
	});

	app.post("/conversations", async (c) => {
		const body = (await c.req.json().catch(() => ({}))) as Partial<CreateConversationInput>;
		const tenant = extractTenant({ get: (name) => c.req.header(name) }, body.tenant);
		const created = await harness.createConversation({
			tenant,
			...(body.model ? { model: body.model } : {}),
			...(body.instructions !== undefined ? { instructions: body.instructions } : {}),
			...(body.enabledToolPacks ? { enabledToolPacks: body.enabledToolPacks } : {}),
			...(body.maxTurns !== undefined ? { maxTurns: body.maxTurns } : {}),
		});
		return c.json(created, 201);
	});

	app.get("/conversations/:id", async (c) => {
		const id = c.req.param("id");
		const info = await harness.getConversation(id);
		if (!info) return c.json({ error: `Conversation "${id}" not found` }, 404);
		const transcript = await harness.getTranscript(id);
		return c.json({ conversation: info, transcript });
	});

	app.post("/conversations/:id/fork", async (c) => {
		const id = c.req.param("id");
		const body = (await c.req.json()) as {
			atEntryId: string;
			model?: CreateConversationInput["model"];
			instructions?: string;
		};
		if (!body.atEntryId) {
			return c.json({ error: "atEntryId is required" }, 400);
		}
		const forked = await harness.forkConversation(id, body.atEntryId, {
			...(body.model ? { model: body.model } : {}),
			...(body.instructions !== undefined ? { instructions: body.instructions } : {}),
		});
		return c.json(forked, 201);
	});

	app.post("/conversations/:id/messages", async (c) => {
		const conversationId = c.req.param("id");
		const body = (await c.req.json()) as {
			content: string;
			requestId?: string;
			whenBusy?: SubmitMessageInput["whenBusy"];
			wait?: boolean;
		};
		if (typeof body.content !== "string" || body.content.length === 0) {
			return c.json({ error: "content must be a non-empty string" }, 400);
		}

		const submitInput: SubmitMessageInput = {
			conversationId,
			content: body.content,
			...(body.requestId ? { requestId: body.requestId } : {}),
			...(body.whenBusy ? { whenBusy: body.whenBusy } : {}),
		};

		if (body.wait === true) {
			const settled = await harness.submitAndWait(submitInput, c.req.raw.signal);
			return c.json(settled);
		}

		const handle = await harness.submit(submitInput);
		return c.json(handle, 202);
	});

	app.get("/conversations/:id/stream", (c) => {
		const conversationId = c.req.param("id");
		return streamSSE(c, async (stream) => {
			const abortController = new AbortController();
			stream.onAbort(() => {
				abortController.abort();
			});

			let eventSeq = 0;
			for await (const event of harness.watchConversation(conversationId, abortController.signal)) {
				await stream.writeSSE({
					id: String(++eventSeq),
					event: event.type,
					data: JSON.stringify(event),
				});
			}
		});
	});

	app.post("/conversations/:id/abort", async (c) => {
		await harness.abortConversation(c.req.param("id"));
		return c.json({ aborted: true });
	});

	app.post("/conversations/:id/reset", async (c) => {
		const body = (await c.req.json().catch(() => ({}))) as { handoff?: string };
		await harness.resetConversation(c.req.param("id"), body.handoff);
		return c.json({ reset: true });
	});

	app.post("/conversations/:id/compact", async (c) => {
		const body = (await c.req.json().catch(() => ({}))) as { instructions?: string };
		const result = await harness.compactConversation(c.req.param("id"), body.instructions);
		return c.json(result);
	});

	app.post("/conversations/:id/sleep-cycle", async (c) => {
		const body = (await c.req.json().catch(() => ({}))) as { reason?: string; wait?: boolean };
		const { taskId } = await harness.triggerSleepCycle(c.req.param("id"), { reason: body.reason });
		if (body.wait) {
			const settled = await harness.waitTask(taskId, c.req.raw.signal);
			return c.json(settled);
		}
		return c.json({ taskId }, 202);
	});

	app.get("/approvals", async (c) => {
		const conversationId = c.req.query("conversationId");
		const approvals = await harness.listPendingApprovals(conversationId);
		return c.json({ approvals });
	});

	app.post("/signals", async (c) => {
		const body = (await c.req.json()) as {
			taskId: string;
			signalName: string;
			key: string;
			payload: unknown;
		};
		if (!body.taskId || !body.signalName || !body.key) {
			return c.json({ error: "taskId, signalName, and key are required" }, 400);
		}
		const result = await harness.deliverSignal(body);
		return c.json(result);
	});

	app.get("/tool-packs", (c) => {
		return c.json({ toolPacks: harness.listToolPacks() });
	});

	app.post("/tool-packs/:name", async (c) => {
		const name = c.req.param("name");
		const body = (await c.req.json()) as { enabled: boolean };
		if (typeof body.enabled !== "boolean") {
			return c.json({ error: "enabled boolean is required" }, 400);
		}
		harness.setToolPackEnabled(name, body.enabled);
		return c.json({ toolPacks: harness.listToolPacks() });
	});

	app.get("/usage", async (c) => {
		return c.json(await harness.getUsage());
	});

	return app;
}
