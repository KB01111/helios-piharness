import type { HarnessPort } from "../contracts/harness-port.js";
import type { TestModelController } from "../runtime/model-catalog.js";
import {
	aggregateParetoMetrics,
	type EvalCase,
	type EvalCaseResult,
	type ParetoMetrics,
} from "./types.js";

export interface EvalSuiteResult {
	readonly results: readonly EvalCaseResult[];
	readonly evolveMetrics: ParetoMetrics;
	readonly heldOutMetrics: ParetoMetrics;
	readonly overallMetrics: ParetoMetrics;
}

/**
 * Trace-replay evaluation runner built over `HarnessPort.forkConversation()`
 * and the deterministic `TestModelController`.
 *
 * Every eval case forks from its seeded conversation point so cases run in
 * isolated branches without mutating shared transcripts.
 */
export class ReplayEvalRunner {
	private readonly harness: HarnessPort;
	private readonly testModel?: TestModelController;

	constructor(options: { readonly harness: HarnessPort; readonly testModel?: TestModelController }) {
		this.harness = options.harness;
		this.testModel = options.testModel;
	}

	async runCase(evalCase: EvalCase): Promise<EvalCaseResult> {
		const baseConv = await this.harness.createConversation({
			tenant: evalCase.tenant,
			...(evalCase.maxTurns !== undefined ? { maxTurns: evalCase.maxTurns } : {}),
		});

		let forkConversationId = baseConv.conversationId;

		// Seed prefix turns if provided, then fork at the last seeded entry
		if (evalCase.seedTurns && evalCase.seedTurns.length > 0) {
			for (const seed of evalCase.seedTurns) {
				if (this.testModel && seed.scriptedResponses) {
					this.testModel.enqueueMainMany(seed.scriptedResponses);
				}
				await this.harness.submitAndWait({
					conversationId: baseConv.conversationId,
					content: seed.userPrompt,
				});
			}

			const transcript = await this.harness.getTranscript(baseConv.conversationId);
			const lastEntry = transcript[transcript.length - 1];
			if (lastEntry) {
				const forked = await this.harness.forkConversation(baseConv.conversationId, lastEntry.id);
				forkConversationId = forked.conversationId;
			}
		}

		const baselineEntries = await this.harness.getTranscript(forkConversationId);
		const baselineCount = baselineEntries.length;

		if (this.testModel && evalCase.scriptedResponses) {
			this.testModel.enqueueMainMany(evalCase.scriptedResponses);
		}

		const startedAt = Date.now();
		const settled = await this.harness.submitAndWait({
			conversationId: forkConversationId,
			content: evalCase.evalPrompt,
		});
		const latencyMs = Date.now() - startedAt;

		const afterEntries = await this.harness.getTranscript(forkConversationId);
		const deltaEntries = afterEntries.slice(baselineCount);

		const toolCalls: string[] = [];
		let toolErrors = 0;
		for (const entry of deltaEntries) {
			if (entry.role === "toolResult" && entry.toolName) {
				toolCalls.push(entry.toolName);
				if (entry.isError) toolErrors += 1;
			}
		}

		const answerText = settled.answerText ?? "";
		const failures: string[] = [];
		let safetyViolations = 0;

		if (settled.status !== "done") {
			failures.push(`Expected submission status "done", got "${settled.status}" (${settled.reason ?? "unknown"})`);
		}

		for (const expected of evalCase.expectedSubstrings ?? []) {
			if (!answerText.toLowerCase().includes(expected.toLowerCase())) {
				failures.push(`Answer missing expected substring "${expected}"`);
			}
		}

		for (const forbidden of evalCase.forbiddenSubstrings ?? []) {
			if (answerText.toLowerCase().includes(forbidden.toLowerCase())) {
				safetyViolations += 1;
				failures.push(`Answer contained forbidden substring "${forbidden}"`);
			}
		}

		for (const expectedTool of evalCase.expectedToolNames ?? []) {
			if (!toolCalls.includes(expectedTool)) {
				failures.push(`Expected tool "${expectedTool}" to be called (actual: [${toolCalls.join(", ")}])`);
			}
		}

		if (evalCase.maxLatencyMs !== undefined && latencyMs > evalCase.maxLatencyMs) {
			failures.push(`Latency ${latencyMs}ms exceeded cap ${evalCase.maxLatencyMs}ms`);
		}

		return {
			caseId: evalCase.id,
			name: evalCase.name,
			split: evalCase.split,
			passed: failures.length === 0,
			answerText,
			latencyMs,
			toolCalls,
			toolErrors,
			safetyViolations,
			failures,
		};
	}

	async runSuite(cases: readonly EvalCase[]): Promise<EvalSuiteResult> {
		const results: EvalCaseResult[] = [];
		for (const evalCase of cases) {
			results.push(await this.runCase(evalCase));
		}

		const evolveResults = results.filter((r) => r.split === "evolve");
		const heldOutResults = results.filter((r) => r.split === "held_out");

		return {
			results,
			evolveMetrics: aggregateParetoMetrics(evolveResults),
			heldOutMetrics: aggregateParetoMetrics(heldOutResults),
			overallMetrics: aggregateParetoMetrics(results),
		};
	}
}
