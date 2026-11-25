import path from "node:path";
import { ForbiddenError } from "../../contracts/errors.js";
import { ADMIN_PERMISSIONS, hasPermission, type TenantContext } from "../../contracts/tenant-context.js";
import { loadEvalSet } from "../../evals/eval-set.js";
import type { SimLog } from "../sim-log.js";
import { composePrompt } from "./candidate.js";
import { ImprovementError, type ImprovementRecord, type ImprovementStore } from "./improvement-store.js";
import { SELF_IMPROVE_TARGET, type ImprovementRunReport, type SelfImprover } from "./self-improver.js";

export interface ImprovementServiceOptions {
	readonly store: ImprovementStore;
	readonly improver: SelfImprover;
	readonly simLog: SimLog;
	/** Root of the eval set: cases are read from `<evalsDir>/cases/*.json`. */
	readonly evalsDir: string;
	/** Static prompt without any learned addendum. */
	readonly basePrompt: string;
	/** Installs the effective prompt into the live harness (hot-replaces the core-prompt extension). */
	readonly applyPrompt: (effectivePrompt: string) => void;
}

/**
 * Operator-facing surface of the self-improvement loop. Every method requires `admin:improvements`; the loop
 * itself can only create `pending_approval` records, and only `approve` (a different, explicit call) goes live.
 */
export class ImprovementService {
	constructor(private readonly options: ImprovementServiceOptions) {}

	private require(tenant: TenantContext): void {
		if (!hasPermission(tenant, ADMIN_PERMISSIONS.improvements)) {
			throw new ForbiddenError(`Missing permission "${ADMIN_PERMISSIONS.improvements}"`);
		}
	}

	effectivePrompt(): string {
		return composePrompt(this.options.basePrompt, this.options.store.activeAddenda());
	}

	/** Re-applies the persisted active addenda to the live harness (used at startup and after every decision). */
	apply(): void {
		this.options.applyPrompt(this.effectivePrompt());
	}

	async run(tenant: TenantContext): Promise<ImprovementRunReport> {
		this.require(tenant);
		const { cases } = loadEvalSet(path.join(this.options.evalsDir, "cases"));
		return this.options.improver.run({
			requestedBy: `${tenant.tenantId}/${tenant.userId}`,
			tenantId: tenant.tenantId,
			cases,
		});
	}

	isRunning(): boolean {
		return this.options.improver.isRunning();
	}

	list(tenant: TenantContext): { records: ImprovementRecord[]; activeIds: string[] } {
		this.require(tenant);
		return { records: this.options.store.list(), activeIds: this.options.store.activeIds() };
	}

	get(tenant: TenantContext, id: string): ImprovementRecord {
		this.require(tenant);
		const record = this.options.store.get(id);
		if (!record) throw new ImprovementError(`Improvement "${id}" not found`, "not_found");
		return record;
	}

	approve(tenant: TenantContext, id: string, note?: string): ImprovementRecord {
		this.require(tenant);
		const record = this.options.store.approve(id, `${tenant.tenantId}/${tenant.userId}`, note);
		this.apply();
		return record;
	}

	reject(tenant: TenantContext, id: string, note?: string): ImprovementRecord {
		this.require(tenant);
		return this.options.store.reject(id, `${tenant.tenantId}/${tenant.userId}`, note);
	}

	rollback(tenant: TenantContext, id: string, note?: string): ImprovementRecord {
		this.require(tenant);
		const record = this.options.store.rollback(id, `${tenant.tenantId}/${tenant.userId}`, note);
		this.apply();
		// A rolled-back idea must not be proposed again: record it next to its original "win".
		this.options.simLog.append({
			tenantId: tenant.tenantId,
			targetModule: SELF_IMPROVE_TARGET,
			hypothesis: record.hypothesis,
			patchSummary: `rolled back prompt addendum ${record.id}`,
			outcome: "reverted",
			evolveDelta: {},
			heldOutDelta: {},
			regressionDetected: true,
			notes: note ?? "rolled back by operator",
		});
		return record;
	}
}
