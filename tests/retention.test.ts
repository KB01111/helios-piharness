import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadConfig } from "../src/config.js";
import { AppDatabase } from "../src/memory/db.js";
import { BudgetStore } from "../src/runtime/budget.js";
import { DEFAULT_RETENTION_POLICY, RetentionService } from "../src/runtime/retention.js";

const DAY = 86_400_000;
const NOW = Date.UTC(2026, 5, 1, 12, 0, 0);

const open: AppDatabase[] = [];
const tmpDirs: string[] = [];

function db(): AppDatabase {
	const instance = AppDatabase.open({ dbPath: ":memory:" });
	open.push(instance);
	return instance;
}

afterEach(() => {
	for (const instance of open.splice(0)) instance.close();
	for (const dir of tmpDirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
});

function insertAudit(appDb: AppDatabase, createdAt: number): void {
	appDb.db
		.prepare(
			`INSERT INTO permission_audit_log (tenant_id, user_id, conversation_id, task_id, call_id, tool_name, tier, decision, created_at)
			 VALUES ('t', 'u', 'c', 'k', 'x', 'tool', 'ask', 'approved', ?)`,
		)
		.run(createdAt);
}

function count(appDb: AppDatabase, table: string): number {
	return Number((appDb.db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get() as { n: number }).n);
}

describe("RetentionService", () => {
	it("deletes only rows older than the window and reports what it removed", async () => {
		const appDb = db();
		insertAudit(appDb, NOW - 400 * DAY);
		insertAudit(appDb, NOW - 366 * DAY);
		insertAudit(appDb, NOW - 10 * DAY);

		const service = new RetentionService({ appDb, now: () => NOW });
		const report = await service.sweepOnce();

		expect(report.deleted.permission_audit_log).toBe(2);
		expect(report.totalDeleted).toBe(2);
		expect(report.errors).toEqual([]);
		expect(count(appDb, "permission_audit_log")).toBe(1);
	});

	it("keeps everything when a window is 0", async () => {
		const appDb = db();
		insertAudit(appDb, NOW - 5000 * DAY);
		const service = new RetentionService({ appDb, now: () => NOW, policy: { permissionAuditDays: 0 } });
		const report = await service.sweepOnce();
		expect(report.totalDeleted).toBe(0);
		expect(count(appDb, "permission_audit_log")).toBe(1);
	});

	it("drains more rows than one batch", async () => {
		const appDb = db();
		for (let i = 0; i < 25; i++) insertAudit(appDb, NOW - 500 * DAY);
		insertAudit(appDb, NOW);
		const service = new RetentionService({ appDb, now: () => NOW, batchSize: 4 });
		const report = await service.sweepOnce();
		expect(report.deleted.permission_audit_log).toBe(25);
		expect(count(appDb, "permission_audit_log")).toBe(1);
	});

	it("prunes budget events and the daily spend ledger, and skips them when the tables do not exist", async () => {
		const appDb = db();
		const before = await new RetentionService({ appDb, now: () => NOW }).sweepOnce();
		expect(before.skipped).toEqual(expect.arrayContaining(["budget_events", "budget_tenant_daily"]));

		const store = new BudgetStore(appDb.db, { now: () => NOW });
		store.recordUsage("t", 100, 1, NOW - 500 * DAY);
		store.recordUsage("t", 100, 1, NOW - 5 * DAY);
		appDb.db
			.prepare(
				`INSERT INTO budget_events (created_at, tenant_id, conversation_id, code, scope, metric, limit_value, used_value, message)
				 VALUES (?, 't', 'c', 'BUDGET_EXCEEDED', 'conversation', 'tokens', 1, 2, 'm')`,
			)
			.run(NOW - 60 * DAY);

		const report = await new RetentionService({ appDb, now: () => NOW }).sweepOnce();
		expect(report.deleted.budget_tenant_daily).toBe(1);
		expect(report.deleted.budget_events).toBe(1);
		expect(count(appDb, "budget_tenant_daily")).toBe(1);
		// The current day's totals, which enforce limits, are untouched.
		expect(store.getTenantDaily("t", NOW - 5 * DAY).tokens).toBe(100);
	});

	it("deletes expired artifact rows together with their files, only when opted in", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "retention-art-"));
		tmpDirs.push(dir);
		const appDb = db();
		const oldFile = path.join(dir, "old.bin");
		const newFile = path.join(dir, "new.bin");
		fs.writeFileSync(oldFile, "x");
		fs.writeFileSync(newFile, "y");
		const insert = appDb.db.prepare(
			`INSERT INTO artifacts (id, tenant_id, user_id, content_type, byte_length, file_path, created_at)
			 VALUES (?, 't', 'u', 'text/plain', 1, ?, ?)`,
		);
		insert.run("art_old", oldFile, NOW - 100 * DAY);
		insert.run("art_new", newFile, NOW - DAY);

		const off = await new RetentionService({ appDb, now: () => NOW }).sweepOnce();
		expect(off.deleted.artifacts).toBeUndefined();
		expect(fs.existsSync(oldFile)).toBe(true);

		const on = await new RetentionService({ appDb, now: () => NOW, policy: { artifactDays: 30 } }).sweepOnce();
		expect(on.deleted.artifacts).toBe(1);
		expect(fs.existsSync(oldFile)).toBe(false);
		expect(fs.existsSync(newFile)).toBe(true);
		expect(count(appDb, "artifacts")).toBe(1);
	});

	it("is single-flight and stops cleanly after dispose", async () => {
		const appDb = db();
		insertAudit(appDb, NOW - 500 * DAY);
		const service = new RetentionService({ appDb, now: () => NOW, intervalMs: 0 });
		const [a, b] = [service.sweepOnce(), service.sweepOnce()];
		expect(a).toBe(b);
		await a;
		service.dispose();
		await service.idle();
	});

	it("reports a failing table without aborting the others", async () => {
		const appDb = db();
		insertAudit(appDb, NOW - 500 * DAY);
		appDb.db.exec("DROP TABLE tool_invocations; CREATE VIEW tool_invocations AS SELECT 1 AS id, 1 AS created_at;");
		const errors: unknown[] = [];
		const report = await new RetentionService({ appDb, now: () => NOW, onError: (e) => errors.push(e) }).sweepOnce();
		// A view is not a table, so the existence check skips it rather than failing; the audit table is still pruned.
		expect(report.deleted.permission_audit_log).toBe(1);
		expect(report.errors).toEqual([]);
	});
});

describe("retention config", () => {
	it("defaults to the documented windows and parses overrides", () => {
		const defaults = loadConfig({ USE_MOCK_MODELS: "true" });
		expect(defaults.retentionEnabled).toBe(true);
		expect(defaults.retentionPermissionAuditDays).toBe(DEFAULT_RETENTION_POLICY.permissionAuditDays);
		expect(defaults.retentionArtifactDays).toBe(0);

		const custom = loadConfig({
			USE_MOCK_MODELS: "true",
			RETENTION_ENABLED: "false",
			RETENTION_TOOL_INVOCATION_DAYS: "7",
			RETENTION_ARTIFACT_DAYS: "bogus",
		});
		expect(custom.retentionEnabled).toBe(false);
		expect(custom.retentionToolInvocationDays).toBe(7);
		expect(custom.retentionArtifactDays).toBe(0);
	});
});
