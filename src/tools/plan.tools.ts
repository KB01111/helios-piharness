import { arr, obj, oneOf, opt, str } from "@earendil-works/pi-schema";
import { defineTool, type ToolExecutionResult } from "@earendil-works/pi-durable";
import { PlanDoc, type PlanDocState, type PlanStep, type PlanStepStatus } from "../harness/app-documents.js";
import type { ToolPackDefinition } from "../harness/registry.js";
import { defineToolPack } from "./tool-pack.js";

export const PLAN_LIMITS = { maxSteps: 12, maxTitleChars: 160, maxNoteChars: 300 } as const;

const PlanSetParams = obj({
	steps: arr(str({ description: `One concrete step (max ${PLAN_LIMITS.maxTitleChars} chars).` }), {
		minItems: 1,
		maxItems: PLAN_LIMITS.maxSteps,
		description: "Ordered steps of the plan. Replaces any existing plan.",
	}),
});

const PlanUpdateParams = obj({
	stepId: str({ description: 'Id of the step, e.g. "s2".' }),
	status: oneOf(["pending", "in_progress", "done", "blocked"]),
	note: opt(str({ description: `Result summary, or WHY it is blocked (required when blocked). Max ${PLAN_LIMITS.maxNoteChars} chars.` })),
});

const PlanViewParams = obj({});

const oneLine = (value: string, max: number): string => value.replace(/\s+/g, " ").trim().slice(0, max);

/** The text the model sees: stable, compact, and says what to do next. */
export function renderPlan(plan: PlanDocState): string {
	if (plan.steps.length === 0) return "No plan set. For tasks with 3+ steps call plan_set first.";
	const lines = plan.steps.map((step) => {
		const mark = { pending: "[ ]", in_progress: "[~]", done: "[x]", blocked: "[!]" }[step.status];
		return `${mark} ${step.id}. ${step.title}${step.note ? ` — ${step.note}` : ""}`;
	});
	const open = plan.steps.filter((s) => s.status === "pending" || s.status === "in_progress").length;
	const blocked = plan.steps.filter((s) => s.status === "blocked").length;
	const next = plan.steps.find((s) => s.status === "in_progress") ?? plan.steps.find((s) => s.status === "pending");
	return [
		`Plan (revision ${plan.revision}):`,
		...lines,
		open === 0 && blocked === 0
			? "All steps are done: give the final answer."
			: blocked > 0
				? `${blocked} blocked step(s): re-plan with plan_set, or explain the blocker in your answer.`
				: `Next: ${next?.id}. ${open} step(s) left.`,
	].join("\n");
}

/** Steps that should have been finished or explained before a final answer. Used by the answer verifier. */
export function unfinishedSteps(plan: PlanDocState | undefined): PlanStep[] {
	return (plan?.steps ?? []).filter((s) => s.status === "pending" || s.status === "in_progress");
}

function result(plan: PlanDocState, isError = false): ToolExecutionResult {
	return { ...(isError ? { isError: true } : {}), output: [{ type: "text", text: renderPlan(plan) }], details: { revision: plan.revision } };
}

/**
 * Planning tool pack: a durable, bounded plan the agent maintains on long tasks. Idempotent by construction (`plan_set`
 * replaces, `plan_update` assigns), so all three tools are replay-safe across crashes.
 */
export function createPlanToolPack(): ToolPackDefinition {
	const planSet = defineTool({
		name: "plan_set",
		description:
			"Create or replace your working plan for a task with 3+ steps (or when a step fails and you must re-plan). Keep steps concrete and ordered.",
		parameters: PlanSetParams,
		replay: "safe",
		execute: async (args, api, context) => {
			const titles = args.steps.map((s) => oneLine(s, PLAN_LIMITS.maxTitleChars)).filter(Boolean);
			if (titles.length === 0) return result({ revision: 0, steps: [] }, true);
			let next: PlanDocState | undefined;
			await api.commit(async (tx) => {
				const doc = await tx.doc(PlanDoc, api.conversationId);
				doc.revision += 1;
				doc.steps = titles.slice(0, PLAN_LIMITS.maxSteps).map((title, i) => ({
					id: `s${i + 1}`,
					title,
					status: i === 0 ? "in_progress" : "pending",
					note: "",
				}));
				next = JSON.parse(JSON.stringify(doc)) as PlanDocState;
			}, context);
			return result(next!);
		},
	});

	const planUpdate = defineTool({
		name: "plan_update",
		description:
			"Mark a plan step in_progress, done, or blocked. A blocked step needs a note saying why; then re-plan with plan_set or explain the blocker.",
		parameters: PlanUpdateParams,
		replay: "safe",
		execute: async (args, api, context) => {
			const status: PlanStepStatus = args.status;
			const note = oneLine(args.note ?? "", PLAN_LIMITS.maxNoteChars);
			if (status === "blocked" && !note) {
				const current = await api.snapshot(PlanDoc, api.conversationId, context);
				return { ...result(current ?? { revision: 0, steps: [] }, true), output: [{ type: "text", text: "A blocked step needs a note explaining why." }] };
			}
			let outcome: { plan: PlanDocState; found: boolean } | undefined;
			await api.commit(async (tx) => {
				const doc = await tx.doc(PlanDoc, api.conversationId);
				const step = doc.steps.find((s) => s.id === args.stepId);
				if (step) {
					step.status = status;
					step.note = note;
					doc.revision += 1;
				}
				outcome = { plan: JSON.parse(JSON.stringify(doc)) as PlanDocState, found: step !== undefined };
			}, context);
			if (!outcome!.found) {
				return {
					isError: true,
					output: [{ type: "text", text: `No step "${args.stepId}". ${renderPlan(outcome!.plan)}` }],
				};
			}
			return result(outcome!.plan);
		},
	});

	const planView = defineTool({
		name: "plan_view",
		description: "Show the current plan. Use it after a long tool sequence or a context compaction to re-orient.",
		parameters: PlanViewParams,
		replay: "safe",
		execute: async (_args, api, context) => {
			const plan = await api.snapshot(PlanDoc, api.conversationId, context);
			return result(plan ?? { revision: 0, steps: [] });
		},
	});

	return defineToolPack({
		name: "planning",
		description: "Durable working plan for multi-step tasks: plan_set, plan_update, plan_view.",
		tools: [planSet, planUpdate, planView],
	});
}

