import fs from "node:fs";
import { defineExtension, defineTask, type Extension, type Task } from "@earendil-works/pi-durable";
import type { AppCapabilities } from "../contracts/app-capabilities.js";
import { ConversationMetaDoc, RunTelemetryDoc } from "../harness/app-documents.js";
import { SimLog } from "../learning/sim-log.js";
import { SkillDojo } from "../learning/skill-dojo.js";
import { SkillForge } from "../learning/skill-forge.js";
import { proposeSkillDrafts } from "../learning/skill-promotion.js";
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
			/** Non-fatal problems from the consolidate phase (e.g. fast-model fallback). */
			warnings?: string[];
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
	/** Non-fatal problems (consolidation fallback, proposal/SimLog write failures). Absent when clean. */
	warnings?: string[];
	/** Draft skills newly written to `<skillsDir>/_proposed/` this cycle (require approval to activate). */
	proposedSkills?: {
		name: string;
		workflowId?: string;
	}[];
};

export interface SleepCycleRetryOptions {
	/** Total consolidate attempts with the fast model before degrading. Default: 3 */
	readonly maxAttempts?: number;
	/** Base delay; doubles each retry. Default: 250ms */
	readonly baseDelayMs?: number;
	/** Injectable delay (tests). */
	readonly sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

export interface SleepCycleExtensionOptions {
	readonly capabilities: AppCapabilities;
	readonly consolidationGates?: Partial<ConsolidationGateConfig>;
	readonly skillForge?: SkillForge;
	readonly skillDojo?: SkillDojo;
	/** Append-only self-improvement log; sleep-cycle outcomes are recorded here (tenant-scoped) when provided. */
	readonly simLog?: SimLog;
	/** Retry/backoff for the fast-model consolidate phase. */
	readonly consolidationRetry?: SleepCycleRetryOptions;
	/** Write mined workflows as drafts under `<skillsDir>/_proposed/`. Default: true */
	readonly proposeSkills?: boolean;
	/** Maximum NEW skill drafts written per cycle. Default: 5 */
	readonly maxProposedSkillsPerCycle?: number;
}

function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve) => {
		if (signal?.aborted) return resolve();
		const timer = setTimeout(done, ms);
		function done(): void {
			clearTimeout(timer);
			signal?.removeEventListener("abort", done);
			resolve();
		}
		signal?.addEventListener("abort", done, { once: true });
	});
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

export interface SleepCycleBundle {
	readonly extension: Extension;
	readonly SleepCycleTask: Task<SleepCycleInput, SleepCycleCheckpoint, SleepCycleResult, object>;
}

function skillRegisteredAt(filePath: string): number | undefined {
	try {
		return fs.statSync(filePath).mtimeMs;
	} catch {
		return undefined;
	}
}

function recordSimLog(
	simLog: SimLog,
	input: {
		readonly tenantId: string;
		readonly taskKey: string;
		readonly now: number;
		readonly consolidation: ConsolidationSummaryCheckpoint;
		readonly warnings: readonly string[];
		readonly newlyDisabledSkills: readonly { readonly name: string; readonly reason: string }[];
		readonly proposedSkills: readonly { readonly name: string }[];
	},
): void {
	const { tenantId, taskKey, now, consolidation: c } = input;
	const changed = c.addedCount + c.updatedCount + c.supersededCount;
	const consolidationFailed = input.warnings.some((w) => w.startsWith("consolidation failed") || w.includes("fallback failed"));

	if (c.processedEpisodes > 0 || consolidationFailed || input.warnings.length > 0) {
		simLog.appendIfAbsent({
			tenantId,
			targetModule: "memory-consolidation",
			hypothesis: `sleep-cycle:${taskKey}:consolidation`,
			patchSummary: `processed ${c.processedEpisodes} episode(s): +${c.addedCount} added, ${c.updatedCount} updated, ${c.supersededCount} superseded, ${c.noopCount} noop, ${c.rejectedCount} rejected by gate`,
			outcome: consolidationFailed ? "failed" : changed > 0 && input.warnings.length === 0 ? "win" : "inconclusive",
			evolveDelta: {
				processedEpisodes: c.processedEpisodes,
				added: c.addedCount,
				updated: c.updatedCount,
				superseded: c.supersededCount,
				rejected: c.rejectedCount,
			},
			heldOutDelta: {},
			regressionDetected: false,
			...(input.warnings.length > 0 ? { notes: input.warnings.join("; ") } : {}),
			createdAt: now,
		});
	}

	for (const skill of input.newlyDisabledSkills) {
		simLog.appendIfAbsent({
			tenantId,
			targetModule: "skill-dojo",
			hypothesis: `sleep-cycle:${taskKey}:disable:${skill.name}`,
			patchSummary: `disabled skill "${skill.name}": ${skill.reason}`,
			outcome: "inconclusive",
			evolveDelta: { disabledSkills: 1 },
			heldOutDelta: {},
			regressionDetected: false,
			createdAt: now,
		});
	}

	for (const skill of input.proposedSkills) {
		simLog.appendIfAbsent({
			tenantId,
			targetModule: "skill-forge",
			hypothesis: `sleep-cycle:${taskKey}:propose:${skill.name}`,
			patchSummary: `proposed draft skill "${skill.name}" (awaiting approval)`,
			outcome: "inconclusive",
			evolveDelta: { proposedSkills: 1 },
			heldOutDelta: {},
			regressionDetected: false,
			createdAt: now,
		});
	}
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
	// Deterministic (no fast model) consolidator used when the fast model keeps failing.
	const fallbackConsolidator = new MemoryConsolidator({
		memory: capabilities.memory,
		gates: options.consolidationGates,
	});
	const retryAttempts = Math.max(1, options.consolidationRetry?.maxAttempts ?? 3);
	const retryBaseDelayMs = Math.max(0, options.consolidationRetry?.baseDelayMs ?? 250);
	const sleep = options.consolidationRetry?.sleep ?? defaultSleep;
	const skillForge = options.skillForge ?? new SkillForge();
	const skillDojo =
		options.skillDojo ??
		new SkillDojo(capabilities.telemetry, {
			staleSkillDays: capabilities.staleSkillDays,
		});

	const emptySummary = (): ConsolidationSummaryCheckpoint => ({
		processedEpisodes: 0,
		addedCount: 0,
		updatedCount: 0,
		supersededCount: 0,
		noopCount: 0,
		rejectedCount: 0,
	});

	const summarize = (report: Awaited<ReturnType<MemoryConsolidator["consolidate"]>>): ConsolidationSummaryCheckpoint => ({
		processedEpisodes: report.processedEpisodeIds.length,
		addedCount: report.addedIds.length,
		updatedCount: report.updatedIds.length,
		supersededCount: report.supersededOldIds.length,
		noopCount: report.noopCount,
		rejectedCount: report.rejectedByGate.length,
	});

	/**
	 * Fast-model consolidation with bounded retry + exponential backoff, then a deterministic
	 * (model-free) fallback. Never throws except on abort so the learn phase always runs.
	 */
	async function consolidateWithResilience(
		tenantId: string,
		userId: string,
		signal: AbortSignal | undefined,
		warnings: string[],
	): Promise<ConsolidationSummaryCheckpoint> {
		let lastError: unknown;
		const attempts = capabilities.fastModel ? retryAttempts : 1;
		for (let attempt = 1; attempt <= attempts; attempt++) {
			try {
				return summarize(await consolidator.consolidate(tenantId, userId, { signal }));
			} catch (error) {
				if (signal?.aborted) throw error;
				lastError = error;
				if (attempt < attempts) await sleep(retryBaseDelayMs * 2 ** (attempt - 1), signal);
			}
		}

		if (!capabilities.fastModel) {
			// No model to blame: the deterministic path itself failed, so degrade instead of retrying it.
			warnings.push(`consolidation failed: ${errorMessage(lastError)}`);
			return emptySummary();
		}

		warnings.push(
			`consolidation: fast model failed after ${attempts} attempt(s) (${errorMessage(lastError)}); used deterministic fallback`,
		);
		try {
			return summarize(await fallbackConsolidator.consolidate(tenantId, userId, { signal }));
		} catch (fallbackError) {
			if (signal?.aborted) throw fallbackError;
			warnings.push(`consolidation fallback failed: ${errorMessage(fallbackError)}`);
			return emptySummary();
		}
	}

	const SleepCycleTask = defineTask<SleepCycleInput, SleepCycleCheckpoint, SleepCycleResult>({
		name: "app.sleep-cycle",
		version: 1,
		initial: () => ({ phase: "consolidate" }),
		phases: {
			consolidate: async (_task, runtime, context) => {
				const meta = await runtime.snapshot(ConversationMetaDoc, runtime.conversationId, context);
				if (!meta) {
					await runtime.commit(
						() => ({
							status: "terminal",
							outcome: {
								status: "completed",
								result: {
									consolidation: emptySummary(),
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

				const warnings: string[] = [];
				const summary = await consolidateWithResilience(
					meta.tenant.tenantId,
					meta.tenant.userId,
					context.abortSignal,
					warnings,
				);

				await runtime.commit(
					() => ({
						status: "running",
						checkpoint: {
							phase: "learn",
							consolidation: summary,
							...(warnings.length > 0 ? { warnings } : {}),
						},
					}),
					context,
				);
			},

			learn: async (task, runtime, context) => {
				const meta = await runtime.snapshot(ConversationMetaDoc, runtime.conversationId, context);
				const tenantId = meta?.tenant.tenantId ?? "default";
				const userId = meta?.tenant.userId;
				const warnings: string[] = [...(task.state.checkpoint.warnings ?? [])];
				const taskKey = String(runtime.taskId);

				const sequences = await capabilities.telemetry.getRecentToolSequences(tenantId, userId, 200);
				const mined = skillForge.mine(sequences);

				const catalog = loadSkillCatalog(capabilities.skillsDir);
				const dojo =
					task.input.staleSkillDays !== null
						? new SkillDojo(capabilities.telemetry, { staleSkillDays: task.input.staleSkillDays })
						: skillDojo;

				// Skills that were never used are judged against their file age (mtime), otherwise
				// "never used since registration" could never trigger.
				const evaluation = await dojo.evaluateHealth(
					tenantId,
					runtime.now(),
					catalog.map((s) => ({ name: s.name, registeredAt: skillRegisteredAt(s.filePath) })),
				);

				// SkillPromotion: persist mined workflows as DRAFTS (approval required to activate).
				let proposedSkills: { name: string; workflowId?: string }[] = [];
				if (options.proposeSkills !== false && mined.length > 0) {
					try {
						const proposal = proposeSkillDrafts(capabilities.skillsDir, mined, {
							provenance: {
								tenantId,
								source: `sleep-cycle skill-forge (${mined.length} mined workflow(s))`,
								proposedAt: runtime.now(),
								taskId: taskKey,
							},
							maxNew: options.maxProposedSkillsPerCycle ?? 5,
							excludeNames: evaluation.allDisabled,
						});
						proposedSkills = proposal.created.map((c) => ({
							name: c.name,
							...(c.workflowId ? { workflowId: c.workflowId } : {}),
						}));
					} catch (error) {
						if (context.abortSignal?.aborted) throw error;
						warnings.push(`skill proposal failed: ${errorMessage(error)}`);
					}
				}

				const consolidation = task.state.checkpoint.consolidation;
				const newlyDisabledSkills = evaluation.newlyDisabled.map((d) => ({ name: d.name, reason: d.reason }));

				// SimLog (append-only, tenant-scoped, idempotent per task so a replayed phase records once).
				if (options.simLog) {
					try {
						recordSimLog(options.simLog, {
							tenantId,
							taskKey,
							now: runtime.now(),
							consolidation,
							warnings,
							newlyDisabledSkills,
							proposedSkills,
						});
					} catch (error) {
						warnings.push(`sim-log write failed: ${errorMessage(error)}`);
					}
				}

				const result: SleepCycleResult = {
					consolidation,
					minedWorkflows: mined.map((w) => ({
						id: w.id,
						suggestedSkillName: w.suggestedSkillName,
						sequence: [...w.sequence],
						support: w.support,
					})),
					newlyDisabledSkills,
					allDisabledSkills: [...evaluation.allDisabled],
					...(warnings.length > 0 ? { warnings } : {}),
					...(proposedSkills.length > 0 ? { proposedSkills } : {}),
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
