import { defineExtension, defineTask, type Extension, type Task } from "@earendil-works/pi-durable";
import type { AppCapabilities } from "../contracts/app-capabilities.js";
import { ConversationMetaDoc, RunTelemetryDoc } from "../harness/app-documents.js";
import { SkillDojo } from "../learning/skill-dojo.js";
import { SkillForge } from "../learning/skill-forge.js";
import { MemoryConsolidator, type ConsolidationGateConfig } from "../memory/consolidator.js";
import { loadSkillCatalog } from "./skills-loader.ext.js";

export type SleepCycleInput = {
	reason: string;
	staleSkillDays: number | null;
};

export type ConsolidationSummaryCheckpoint = {
	processedEpisodes: number;
	addedCount: number;
	updatedCount: number;
	supersededCount: number;
	noopCount: number;
	rejectedCount: number;
};

export type SleepCycleCheckpoint =
	| { phase: "consolidate" }
	| {
			phase: "learn";
			consolidation: ConsolidationSummaryCheckpoint;
	  };

export type SleepCycleResult = {
	consolidation: ConsolidationSummaryCheckpoint;
	minedWorkflows: {
		id: string;
		suggestedSkillName: string;
		sequence: string[];
		support: number;
	}[];
	newlyDisabledSkills: {
		name: string;
		reason: string;
	}[];
	allDisabledSkills: string[];
};

export interface SleepCycleExtensionOptions {
	readonly capabilities: AppCapabilities;
	readonly consolidationGates?: Partial<ConsolidationGateConfig>;
	readonly skillForge?: SkillForge;
	readonly skillDojo?: SkillDojo;
}

export interface SleepCycleBundle {
	readonly extension: Extension;
	readonly SleepCycleTask: Task<SleepCycleInput, SleepCycleCheckpoint, SleepCycleResult, object>;
}

/**
 * Isolated Extension (Loop 2): Durable Pi Task running offline sleep cycles
 * for ACE memory diffs + Skill Forge n-gram mining + Skill Dojo health pruning.
 *
 * Checkpoints cleanly between memory consolidation and skill learning phases so
 * a crash between phases resumes without re-running completed work.
 */
export function createSleepCycleExtension(options: SleepCycleExtensionOptions): SleepCycleBundle {
	const { capabilities } = options;
	const consolidator = new MemoryConsolidator({
		memory: capabilities.memory,
		fastModel: capabilities.fastModel,
		gates: options.consolidationGates,
	});
	const skillForge = options.skillForge ?? new SkillForge();
	const skillDojo =
		options.skillDojo ??
		new SkillDojo(capabilities.telemetry, {
			staleSkillDays: capabilities.staleSkillDays,
		});

	const SleepCycleTask = defineTask<SleepCycleInput, SleepCycleCheckpoint, SleepCycleResult>({
		name: "app.sleep-cycle",
		version: 1,
		initial: () => ({ phase: "consolidate" }),
		phases: {
			consolidate: async (_task, runtime, context) => {
				const meta = await runtime.snapshot(ConversationMetaDoc, runtime.conversationId, context);
				if (!meta) {
					const emptyConsolidation: ConsolidationSummaryCheckpoint = {
						processedEpisodes: 0,
						addedCount: 0,
						updatedCount: 0,
						supersededCount: 0,
						noopCount: 0,
						rejectedCount: 0,
					};
					await runtime.commit(
						() => ({
							status: "terminal",
							outcome: {
								status: "completed",
								result: {
									consolidation: emptyConsolidation,
									minedWorkflows: [],
									newlyDisabledSkills: [],
									allDisabledSkills: [],
								},
							},
						}),
						context,
					);
					return;
				}

				const report = await consolidator.consolidate(meta.tenant.tenantId, meta.tenant.userId, {
					signal: context.abortSignal,
				});

				const summary: ConsolidationSummaryCheckpoint = {
					processedEpisodes: report.processedEpisodeIds.length,
					addedCount: report.addedIds.length,
					updatedCount: report.updatedIds.length,
					supersededCount: report.supersededOldIds.length,
					noopCount: report.noopCount,
					rejectedCount: report.rejectedByGate.length,
				};

				await runtime.commit(
					() => ({
						status: "running",
						checkpoint: {
							phase: "learn",
							consolidation: summary,
						},
					}),
					context,
				);
			},

			learn: async (task, runtime, context) => {
				const meta = await runtime.snapshot(ConversationMetaDoc, runtime.conversationId, context);
				const tenantId = meta?.tenant.tenantId ?? "default";
				const userId = meta?.tenant.userId;

				const sequences = await capabilities.telemetry.getRecentToolSequences(tenantId, userId, 200);
				const mined = skillForge.mine(sequences);

				const catalog = loadSkillCatalog(capabilities.skillsDir);
				const dojo =
					task.input.staleSkillDays !== null
						? new SkillDojo(capabilities.telemetry, { staleSkillDays: task.input.staleSkillDays })
						: skillDojo;

				const evaluation = await dojo.evaluateHealth(
					tenantId,
					runtime.now(),
					catalog.map((s) => ({ name: s.name })),
				);

				const result: SleepCycleResult = {
					consolidation: task.state.checkpoint.consolidation,
					minedWorkflows: mined.map((w) => ({
						id: w.id,
						suggestedSkillName: w.suggestedSkillName,
						sequence: [...w.sequence],
						support: w.support,
					})),
					newlyDisabledSkills: evaluation.newlyDisabled.map((d) => ({
						name: d.name,
						reason: d.reason,
					})),
					allDisabledSkills: [...evaluation.allDisabled],
				};

				await runtime.commit(async (tx) => {
					const doc = await tx.doc(RunTelemetryDoc, runtime.conversationId);
					doc.sleepCyclesCompleted += 1;
					doc.lastSleepCycleAt = runtime.now();
					return {
						status: "terminal",
						outcome: {
							status: "completed",
							result,
						},
					};
				}, context);
			},
		},
		abort: async (_task, runtime, context) => {
			await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), context);
		},
	});

	const extension = defineExtension({
		name: "sleep-cycle",
		tasks: [SleepCycleTask],
	});

	return { extension, SleepCycleTask };
}
