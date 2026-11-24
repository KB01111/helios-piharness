import type { SkillHealthMetric, TelemetryStore } from "../contracts/app-capabilities.js";

export interface SkillDojoOptions {
	/** Days of inactivity before a skill is auto-disabled as stale. Default: 30 */
	readonly staleSkillDays?: number;
	/** Maximum error rate (0..1) before a skill is auto-disabled as degraded. Default: 0.6 */
	readonly maxErrorRate?: number;
	/** Minimum invocations before error-rate disabling activates. Default: 5 */
	readonly minErrorSamples?: number;
}

export interface SkillDojoEvaluation {
	readonly evaluatedAt: number;
	readonly metrics: readonly SkillHealthMetric[];
	readonly newlyDisabled: readonly { readonly name: string; readonly reason: string }[];
	readonly allDisabled: readonly string[];
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

function positiveOr(value: number | undefined, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

function clampRate(value: number | undefined, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return fallback;
	return Math.min(1, value);
}

/**
 * Skill Dojo (Loop 1/2): tracks per-tool and per-skill health metrics and
 * automatically disables stale (> 30d unused) or severely degraded skills.
 */
export class SkillDojo {
	private readonly telemetry: TelemetryStore;
	private readonly staleSkillDays: number;
	private readonly maxErrorRate: number;
	private readonly minErrorSamples: number;

	constructor(telemetry: TelemetryStore, options?: SkillDojoOptions) {
		this.telemetry = telemetry;
		// Guard against NaN / zero / negative thresholds, which would otherwise disable every skill
		// immediately (staleSkillDays <= 0) or never (NaN comparisons are always false).
		this.staleSkillDays = positiveOr(options?.staleSkillDays, 30);
		this.maxErrorRate = clampRate(options?.maxErrorRate, 0.6);
		this.minErrorSamples = Math.max(1, Math.floor(positiveOr(options?.minErrorSamples, 5)));
	}

	async getHealthReport(tenantId: string, kind?: "tool" | "skill"): Promise<SkillHealthMetric[]> {
		return this.telemetry.getSkillMetrics(tenantId, kind);
	}

	async evaluateHealth(
		tenantId: string,
		nowMs: number = Date.now(),
		registeredSkills?: readonly { readonly name: string; readonly registeredAt?: number }[],
	): Promise<SkillDojoEvaluation> {
		const skillMetrics = await this.telemetry.getSkillMetrics(tenantId, "skill");
		const existingDisabled = new Set(await this.telemetry.listDisabledSkills(tenantId));
		const staleCutoffMs = nowMs - this.staleSkillDays * MS_PER_DAY;
		const newlyDisabled: Array<{ name: string; reason: string }> = [];

		const seenSkills = new Set<string>();

		for (const metric of skillMetrics) {
			seenSkills.add(metric.name);
			if (existingDisabled.has(metric.name)) continue;

			if (metric.lastUsedAt < staleCutoffMs) {
				const ageDays = Math.floor((nowMs - metric.lastUsedAt) / MS_PER_DAY);
				const reason = `stale: unused for ${ageDays} days (threshold ${this.staleSkillDays}d)`;
				await this.telemetry.setSkillDisabled(tenantId, metric.name, true, reason, nowMs);
				existingDisabled.add(metric.name);
				newlyDisabled.push({ name: metric.name, reason });
				continue;
			}

			if (metric.calls >= this.minErrorSamples && metric.errorRate >= this.maxErrorRate) {
				const pct = Math.round(metric.errorRate * 100);
				const reason = `degraded: ${pct}% error rate across ${metric.calls} calls`;
				await this.telemetry.setSkillDisabled(tenantId, metric.name, true, reason, nowMs);
				existingDisabled.add(metric.name);
				newlyDisabled.push({ name: metric.name, reason });
			}
		}

		// Also check registered skills that have never been used since registration past staleSkillDays
		if (registeredSkills) {
			for (const skill of registeredSkills) {
				if (seenSkills.has(skill.name) || existingDisabled.has(skill.name)) continue;
				if (skill.registeredAt !== undefined && skill.registeredAt < staleCutoffMs) {
					const ageDays = Math.floor((nowMs - skill.registeredAt) / MS_PER_DAY);
					const reason = `stale: never used in ${ageDays} days since registration (threshold ${this.staleSkillDays}d)`;
					await this.telemetry.setSkillDisabled(tenantId, skill.name, true, reason, nowMs);
					existingDisabled.add(skill.name);
					newlyDisabled.push({ name: skill.name, reason });
				}
			}
		}

		const updatedMetrics = await this.telemetry.getSkillMetrics(tenantId);
		return {
			evaluatedAt: nowMs,
			metrics: updatedMetrics,
			newlyDisabled,
			allDisabled: Array.from(existingDisabled).sort(),
		};
	}

	async reenableSkill(tenantId: string, skillName: string): Promise<void> {
		await this.telemetry.setSkillDisabled(tenantId, skillName, false);
	}
}
