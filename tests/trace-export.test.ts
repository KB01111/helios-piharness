import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { bootstrapHarness, DEFAULT_PERMISSIONS, type TenantContext } from "../src/index.js";
import { JsonlTraceExporter } from "../src/runtime/trace-export.js";

const TENANT: TenantContext = { tenantId: "t-trace", userId: "u-trace", permissions: [...DEFAULT_PERMISSIONS] };

describe("trace export", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
	});
	const tmp = () => {
		const d = fs.mkdtempSync(path.join(os.tmpdir(), "helios-trace-"));
		dirs.push(d);
		return d;
	};

	it("writes parent/child spans, drops oversized attributes, marks errors, and rotates", async () => {
		const file = path.join(tmp(), "t.jsonl");
		const exporter = new JsonlTraceExporter({ filePath: file, maxAttributeChars: 20 });
		await exporter.startSpan({ name: "root", attributes: { ok: "short", leaked: "x".repeat(50) } }, async (root) => {
			await root.startSpan({ name: "child" }, (child) => child.addEvent("e", { k: 1 }));
			await expect(
				root.startSpan({ name: "boom" }, () => {
					throw new Error("bad");
				}),
			).rejects.toThrow("bad");
		});
		exporter.close();

		const spans = fs
			.readFileSync(file, "utf8")
			.split("\n")
			.filter(Boolean)
			.map((line) => JSON.parse(line) as { name: string; id: string; traceId: string; parentId?: string; attributes: Record<string, unknown>; status: { status: string } });
		const root = spans.find((s) => s.name === "root")!;
		const child = spans.find((s) => s.name === "child")!;
		expect(child.parentId).toBe(root.id);
		expect(child.traceId).toBe(root.traceId);
		expect(root.attributes).toEqual({ ok: "short" });
		expect(spans.find((s) => s.name === "boom")?.status.status).toBe("error");
	});

	it("rotates at maxBytes keeping one previous file", async () => {
		const file = path.join(tmp(), "rot.jsonl");
		const exporter = new JsonlTraceExporter({ filePath: file, maxBytes: 600 });
		for (let i = 0; i < 10; i++) await exporter.startSpan({ name: `filler-${i}` }, () => undefined);
		exporter.close();
		expect(fs.existsSync(`${file}.1`)).toBe(true);
		expect(fs.statSync(file).size).toBeLessThanOrEqual(600);
		const last = fs.readFileSync(file, "utf8").trim().split("\n").at(-1)!;
		expect((JSON.parse(last) as { name: string }).name).toBe("filler-9");
	});

	it("exports Pi Durable spans for a real run without message content", async () => {
		const file = path.join(tmp(), "run.jsonl");
		const rt = await bootstrapHarness({
			config: { piDbPath: ":memory:", appDbPath: ":memory:", useMockModels: true, traceExportPath: file },
			sleepCycleScheduler: false,
			improvements: false,
		});
		try {
			const conv = await rt.harness.createConversation({ tenant: TENANT });
			rt.testModel.enqueueMain({ type: "text", text: "the answer is PRIVATE_ANSWER_TEXT" });
			await rt.harness.submitAndWait({ conversationId: conv.conversationId, content: "PRIVATE_QUESTION_TEXT" });
			expect(rt.traceExporter?.written()).toBeGreaterThan(0);
		} finally {
			await rt.close();
		}
		const text = fs.readFileSync(file, "utf8");
		const names = new Set(text.split("\n").filter(Boolean).map((l) => (JSON.parse(l) as { name: string }).name));
		expect(names.has("pi.durable.phase")).toBe(true);
		expect(text).not.toContain("PRIVATE_QUESTION_TEXT");
		expect(text).not.toContain("PRIVATE_ANSWER_TEXT");
	});
});
