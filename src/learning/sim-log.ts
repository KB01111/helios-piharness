import crypto from "node:crypto";
import type { AppDatabase } from "../memory/db.js";

export type SimLogOutcome = "win" | "failed" | "reverted" | "inconclusive";

export interface SimLogEntryInput {
	readonly tenantId: string;
	readonly targetModule: string;
	readonly hypothesis: string;
	readonly patchSummary: string;
	readonly outcome: SimLogOutcome;
	readonly evolveDelta: Readonly<Record<string, number>>;
	readonly heldOutDelta: Readonly<Record<string, number>>;
	readonly regressionDetected: boolean;
	readonly notes?: string;
	readonly createdAt?: number;
}

export interface SimLogEntry extends SimLogEntryInput {
	readonly id: string;
	readonly createdAt: number;
}

export interface SimLogSummary {
	readonly total: number;
	readonly wins: number;
	readonly failed: number;
	readonly reverted: number;
	readonly inconclusive: number;
	readonly regressions: number;
}

type SimLogRow = {
	id: string;
	tenant_id: string;
	target_module: string;
	hypothesis: string;
	patch_summary: string;
	outcome: string;
	evolve_delta_json: string;
	held_out_delta_json: string;
	regression_detected: number;
	notes: string | null;
	created_at: number;
};

function rowToEntry(row: SimLogRow): SimLogEntry {
	return {
		id: row.id,
		tenantId: row.tenant_id,
		targetModule: row.target_module,
		hypothesis: row.hypothesis,
		patchSummary: row.patch_summary,
		outcome: row.outcome as SimLogOutcome,
		evolveDelta: JSON.parse(row.evolve_delta_json) as Record<string, number>,
		heldOutDelta: JSON.parse(row.held_out_delta_json) as Record<string, number>,
		regressionDetected: Number(row.regression_detected) === 1,
		...(row.notes ? { notes: row.notes } : {}),
		createdAt: Number(row.created_at),
	};
}

/**
 * Loop 3: Append-only Self-Improvement Memory (SIM log).
 *
 * Persists BOTH winning patches AND failed/reverted edits along with their
 * evolve vs held-out metric deltas so offline optimizers never repeat
 * previously disproven hypotheses.
 */
export class SimLog {
	private readonly appDb: AppDatabase;
	private readonly now: () => number;

	constructor(appDb: AppDatabase, options?: { readonly now?: () => number }) {
		this.appDb = appDb;
		this.now = options?.now ?? (() => Date.now());
	}

	append(input: SimLogEntryInput): SimLogEntry {
		const id = `sim_${crypto.randomUUID()}`;
		const createdAt = input.createdAt ?? this.now();

		this.appDb.db
			.prepare(
				`INSERT INTO sim_log (
					id, tenant_id, target_module, hypothesis, patch_summary,
					outcome, evolve_delta_json, held_out_delta_json,
					regression_detected, notes, created_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			)
			.run(
				id,
				input.tenantId,
				input.targetModule,
				input.hypothesis,
				input.patchSummary,
				input.outcome,
				JSON.stringify(input.evolveDelta),
				JSON.stringify(input.heldOutDelta),
				input.regressionDetected ? 1 : 0,
				input.notes ?? null,
				createdAt,
			);

		return {
			...input,
			id,
			createdAt,
		};
	}

	/**
	 * Idempotent append: no-op (returns `undefined`) when this tenant already has an
	 * entry for the same module + hypothesis. Lets replayed durable phases record once.
	 */
	appendIfAbsent(input: SimLogEntryInput): SimLogEntry | undefined {
		const existing = this.appDb.db
			.prepare("SELECT 1 AS present FROM sim_log WHERE tenant_id = ? AND target_module = ? AND hypothesis = ? LIMIT 1")
			.get(input.tenantId, input.targetModule, input.hypothesis);
		if (existing) return undefined;
		return this.append(input);
	}

	list(filter?: {
		readonly tenantId?: string;
		readonly targetModule?: string;
		readonly outcome?: SimLogOutcome;
		readonly limit?: number;
	}): SimLogEntry[] {
		const clauses: string[] = [];
		const params: (string | number)[] = [];

		if (filter?.tenantId) {
			clauses.push("tenant_id = ?");
			params.push(filter.tenantId);
		}
		if (filter?.targetModule) {
			clauses.push("target_module = ?");
			params.push(filter.targetModule);
		}
		if (filter?.outcome) {
			clauses.push("outcome = ?");
			params.push(filter.outcome);
		}

		const where = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
		params.push(filter?.limit ?? 200);

		const rows = this.appDb.db
			.prepare(`SELECT * FROM sim_log ${where} ORDER BY created_at ASC, id ASC LIMIT ?`)
			.all(...params) as SimLogRow[];
		return rows.map(rowToEntry);
	}

	getFailedHypothesesForModule(tenantId: string, targetModule: string): SimLogEntry[] {
		const rows = this.appDb.db
			.prepare(
				`SELECT * FROM sim_log
				 WHERE tenant_id = ? AND target_module = ? AND outcome IN ('failed', 'reverted')
				 ORDER BY created_at DESC, id DESC`,
			)
			.all(tenantId, targetModule) as SimLogRow[];
		return rows.map(rowToEntry);
	}

	summarize(tenantId?: string): SimLogSummary {
		const entries = this.list({ tenantId, limit: 10_000 });
		let wins = 0;
		let failed = 0;
		let reverted = 0;
		let inconclusive = 0;
		let regressions = 0;

		for (const entry of entries) {
			if (entry.outcome === "win") wins += 1;
			else if (entry.outcome === "failed") failed += 1;
			else if (entry.outcome === "reverted") reverted += 1;
			else inconclusive += 1;
			if (entry.regressionDetected) regressions += 1;
		}

		return {
			total: entries.length,
			wins,
			failed,
			reverted,
			inconclusive,
			regressions,
		};
	}
}
