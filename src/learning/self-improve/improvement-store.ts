import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { HarnessError } from "../../contracts/errors.js";
import type { ParetoMetrics } from "../../evals/types.js";
import { ADDENDUM_LIMITS, joinAddenda, lintAddendum, type CandidateScope } from "./candidate.js";

export type ImprovementStatus = "pending_approval" | "active" | "rejected" | "rolled_back";

export interface ImprovementRecord {
	readonly id: string;
	readonly createdAt: number;
	readonly requestedBy: string;
	readonly scope: CandidateScope | "merged";
	readonly hypothesis: string;
	readonly addendum: string;
	/** Ids of the active improvements this candidate was measured on top of. Approval is refused if that set changed. */
	readonly baselineActiveIds: readonly string[];
	readonly baseline: { readonly evolve: ParetoMetrics; readonly heldOut: ParetoMetrics };
	readonly candidate: { readonly evolve: ParetoMetrics; readonly heldOut: ParetoMetrics };
	readonly gateReasons: readonly string[];
	readonly status: ImprovementStatus;
	readonly decidedBy?: string;
	readonly decidedAt?: number;
	readonly note?: string;
}

export type ImprovementErrorCode = "not_found" | "bad_state" | "stale" | "limit" | "lint";

/** Typed so the HTTP transport maps it by class: 404 for a missing record, 409 for every refused transition. */
export class ImprovementError extends HarnessError {
	constructor(message: string, code: ImprovementErrorCode) {
		super(message, code === "not_found" ? 404 : 409, `improvement_${code}`);
	}
}

const ID_PATTERN = /^imp_[a-f0-9-]{36}$/;

/**
 * Filesystem store: `<dir>/records/<id>.json` plus `<dir>/active.json` (the ordered ids whose addenda are live).
 * Writes are atomic (tmp + rename). Only a human decision (`approve`) can make a record active.
 */
export class ImprovementStore {
	private readonly recordsDir: string;
	private readonly activeFile: string;

	constructor(
		private readonly dir: string,
		private readonly now: () => number = () => Date.now(),
	) {
		this.recordsDir = path.join(dir, "records");
		this.activeFile = path.join(dir, "active.json");
	}

	private writeAtomic(file: string, data: unknown): void {
		fs.mkdirSync(path.dirname(file), { recursive: true });
		const tmp = `${file}.${process.pid}.${crypto.randomUUID()}.tmp`;
		fs.writeFileSync(tmp, `${JSON.stringify(data, null, 2)}\n`, "utf8");
		fs.renameSync(tmp, file);
	}

	private recordPath(id: string): string {
		if (!ID_PATTERN.test(id)) throw new ImprovementError(`Invalid improvement id`, "not_found");
		return path.join(this.recordsDir, `${id}.json`);
	}

	activeIds(): string[] {
		if (!fs.existsSync(this.activeFile)) return [];
		const raw: unknown = JSON.parse(fs.readFileSync(this.activeFile, "utf8"));
		const ids = (raw as { ids?: unknown }).ids;
		return Array.isArray(ids) ? ids.filter((v): v is string => typeof v === "string" && ID_PATTERN.test(v)) : [];
	}

	private saveActive(ids: readonly string[]): void {
		this.writeAtomic(this.activeFile, { version: 1, ids });
	}

	/** Addenda currently live, in approval order. Records that vanished or are no longer `active` are skipped. */
	activeAddenda(): string[] {
		const out: string[] = [];
		for (const id of this.activeIds()) {
			const rec = this.get(id);
			if (rec && rec.status === "active") out.push(rec.addendum);
		}
		return out;
	}

	get(id: string): ImprovementRecord | undefined {
		const file = this.recordPath(id);
		if (!fs.existsSync(file)) return undefined;
		return JSON.parse(fs.readFileSync(file, "utf8")) as ImprovementRecord;
	}

	list(): ImprovementRecord[] {
		if (!fs.existsSync(this.recordsDir)) return [];
		return fs
			.readdirSync(this.recordsDir)
			.filter((name) => name.endsWith(".json"))
			.map((name) => JSON.parse(fs.readFileSync(path.join(this.recordsDir, name), "utf8")) as ImprovementRecord)
			.sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id));
	}

	/** Stores a gate-accepted candidate as `pending_approval`. It changes nothing until a human approves it. */
	propose(input: Omit<ImprovementRecord, "id" | "createdAt" | "status" | "baselineActiveIds">): ImprovementRecord {
		const record: ImprovementRecord = {
			...input,
			id: `imp_${crypto.randomUUID()}`,
			createdAt: this.now(),
			status: "pending_approval",
			baselineActiveIds: this.activeIds(),
		};
		this.writeAtomic(this.recordPath(record.id), record);
		return record;
	}

	private update(id: string, patch: Partial<ImprovementRecord>): ImprovementRecord {
		const current = this.get(id);
		if (!current) throw new ImprovementError(`Improvement "${id}" not found`, "not_found");
		const next = { ...current, ...patch };
		this.writeAtomic(this.recordPath(id), next);
		return next;
	}

	approve(id: string, approver: string, note?: string): ImprovementRecord {
		const record = this.get(id);
		if (!record) throw new ImprovementError(`Improvement "${id}" not found`, "not_found");
		if (record.status !== "pending_approval") {
			throw new ImprovementError(`Improvement is ${record.status}, not pending_approval`, "bad_state");
		}
		const active = this.activeIds();
		if (JSON.stringify(active) !== JSON.stringify(record.baselineActiveIds)) {
			throw new ImprovementError(
				"The set of active improvements changed after this one was measured; re-run the improvement loop so it is evaluated on top of the current prompt.",
				"stale",
			);
		}
		const lint = lintAddendum(record.addendum);
		if (!lint.ok) throw new ImprovementError(`Addendum failed lint: ${lint.problems.join("; ")}`, "lint");
		const total = joinAddenda([...this.activeAddenda(), record.addendum]).length;
		if (total > ADDENDUM_LIMITS.maxTotalChars) {
			throw new ImprovementError(
				`Active guidance would be ${total} chars, over the ${ADDENDUM_LIMITS.maxTotalChars} cap; roll back an older improvement first.`,
				"limit",
			);
		}
		const next = this.update(id, {
			status: "active",
			decidedBy: approver,
			decidedAt: this.now(),
			...(note ? { note } : {}),
		});
		this.saveActive([...active, id]);
		return next;
	}

	reject(id: string, reviewer: string, note?: string): ImprovementRecord {
		const record = this.get(id);
		if (!record) throw new ImprovementError(`Improvement "${id}" not found`, "not_found");
		if (record.status !== "pending_approval") {
			throw new ImprovementError(`Improvement is ${record.status}, not pending_approval`, "bad_state");
		}
		return this.update(id, {
			status: "rejected",
			decidedBy: reviewer,
			decidedAt: this.now(),
			...(note ? { note } : {}),
		});
	}

	rollback(id: string, reviewer: string, note?: string): ImprovementRecord {
		const record = this.get(id);
		if (!record) throw new ImprovementError(`Improvement "${id}" not found`, "not_found");
		if (record.status !== "active") throw new ImprovementError(`Improvement is ${record.status}, not active`, "bad_state");
		const next = this.update(id, {
			status: "rolled_back",
			decidedBy: reviewer,
			decidedAt: this.now(),
			...(note ? { note } : {}),
		});
		this.saveActive(this.activeIds().filter((activeId) => activeId !== id));
		return next;
	}
}
