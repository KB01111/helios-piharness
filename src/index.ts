import { loadConfig, parseModelSpec, type AppConfig } from "./config.js";
import type {
	AppCapabilities,
	ArtifactMetadata,
	ArtifactReadResult,
	ArtifactStore,
	EmbeddingProvider,
	FastModelClient,
	FrictionSignal,
	FrictionSignalKind,
	LongTermMemory,
	MemoryKind,
	MemoryMutationOp,
	MemorySearchResult,
	MemoryService,
	MemoryStatus,
	MidTermEpisode,
	PendingEpisodeScope,
	RunTelemetryRecord,
	SkillHealthMetric,
	TelemetryStore,
	ToolInvocationMetric,
	ToolTraceItem,
} from "./contracts/app-capabilities.js";
import type {
	ConversationInfo,
	CreateConversationInput,
	HarnessEvent,
	HarnessPort,
	LiveStatusView,
	ModelSpec,
	PendingApproval,
	SettledSubmission,
	SignalDeliveryInput,
	SignalDeliveryResult,
	SubmissionHandle,
	SubmitMessageInput,
	TaskSettlementView,
	ToolPackSummary,
	ToolSlotView,
	TranscriptEntryView,
	UsageBucketView,
	UsageSummaryView,
} from "./contracts/harness-port.js";
import { DEFAULT_PERMISSIONS, hasPermission, validateTenantContext, type TenantContext } from "./contracts/tenant-context.js";
import { ReplayEvalRunner, type EvalSuiteResult } from "./evals/replay-runner.js";
import {
	aggregateParetoMetrics,
	comparePareto,
	type EvalCase,
	type EvalCaseResult,
	type EvalSeedTurn,
	type EvalSplit,
	type ParetoComparison,
	type ParetoMetrics,
} from "./evals/types.js";
import { createCaptureTelemetryExtension } from "./extensions/capture-telemetry.ext.js";
import {
	buildEvalCaseFromTranscript,
	checkAgainstBaseline,
	EvalSetError,
	loadEvalSet,
	parseEvalCases,
	readBaseline,
	redactForEval,
	writeBaseline,
	type EvalBaseline,
	type RegressionReport,
} from "./evals/eval-set.js";
import {
	ADDENDUM_LIMITS,
	composePrompt,
	evaluateGate,
	lintAddendum,
	type CandidateDraft,
	type GateOptions,
	type GateVerdict,
} from "./learning/self-improve/candidate.js";
import { ImprovementError, ImprovementStore, type ImprovementRecord } from "./learning/self-improve/improvement-store.js";
import { FastModelProposer, StaticProposer } from "./learning/self-improve/llm-proposer.js";
import { ImprovementService } from "./learning/self-improve/service.js";
import { SelfImprover, type CandidateProposer, type ImprovementRunReport } from "./learning/self-improve/self-improver.js";
import { createHeliosCodemode, type CodemodeHandle } from "./extensions/codemode.ext.js";
import { createCorePromptExtension, DEFAULT_STATIC_SYSTEM_PROMPT } from "./extensions/core-prompt.ext.js";
import { createGuardShapingExtension, DEFAULT_TOOL_PERMISSIONS } from "./extensions/guard-shaping.ext.js";
import {
	createBudgetExtension,
	createDefaultMaxTurnsInitializer,
	type BudgetDiagnostic,
} from "./extensions/budget.ext.js";
import {
	BudgetGate,
	BudgetStore,
	budgetLimitsFromConfig,
	createBudgetGatedModels,
	createFastModelBudget,
	type BudgetLimits,
	type UsageRecord,
} from "./runtime/budget.js";
import { buildHarnessSettings, type ProviderCircuitBreakers } from "./runtime/resilience.js";
import {
	createSkillsLoaderExtension,
	formatFrozenSkillsIndex,
	loadSkillCatalog,
	readSkillBody,
	type SkillFrontmatter,
} from "./extensions/skills-loader.ext.js";
import { createSleepCycleExtension, type SleepCycleRetryOptions } from "./extensions/sleep-cycle.ext.js";
import { ConversationService } from "./harness/conversation-svc.js";
import { HarnessRegistryManager } from "./harness/registry.js";
import { SimLog, type SimLogEntry, type SimLogEntryInput, type SimLogOutcome, type SimLogSummary } from "./learning/sim-log.js";
import { SkillDojo, type SkillDojoEvaluation, type SkillDojoOptions } from "./learning/skill-dojo.js";
import { SkillForge, type MinedWorkflow, type SkillForgeOptions } from "./learning/skill-forge.js";
import {
	DEFAULT_CONSOLIDATION_GATES,
	MemoryConsolidator,
	type ConsolidationGateConfig,
	type ConsolidationPlan,
	type ConsolidationReport,
	type ProposedConsolidationOp,
	type RejectedProposal,
} from "./memory/consolidator.js";
import { AppDatabase, type AppDatabaseOptions } from "./memory/db.js";
import { detectFrictionSignals, type FrictionAnalysisInput, type FrictionAnalysisResult } from "./memory/friction-signals.js";
import { SqliteArtifactStore, SqliteMemoryService, SqliteTelemetryStore } from "./memory/sqlite-memory.js";
export type { EmbeddingProvider } from "./contracts/app-capabilities.js";
export {
	DEFAULT_SEVERITY_WEIGHTS,
	type FrictionOptions,
	type SeverityWeights,
} from "./memory/friction-signals.js";
export { DEFAULT_RANKING_WEIGHTS, type RankingOptions, type RankingWeights } from "./memory/retrieval.js";
export {
	MEMORY_LIMITS,
	renderUntrustedMemoryBlock,
	sanitizeMemoryKey,
	sanitizeMemoryText,
	type SanitizeResult,
} from "./memory/sanitize.js";
import { HostRuntime, SingleWriterLockError } from "./runtime/host-runtime.js";
import {
	createModelCatalog,
	type FastModelClientOptions,
	type RecordedModelRequest,
	type ScriptedTurnOutput,
	type TestModelController,
	type TestModelHandler,
} from "./runtime/model-catalog.js";
import {
	ManualWakeScheduler,
	NodeWakeScheduler,
	type WakeScheduler,
	type WakeState,
} from "./runtime/wake-scheduler.js";
import {
	SleepCycleScheduler,
	type SleepCycleSchedulerOptions,
	type SleepSchedulerTimer,
	type SleepTickReport,
} from "./runtime/sleep-scheduler.js";
import {
	listProposedSkills,
	promoteProposedSkill,
	PROPOSED_SKILLS_DIRNAME,
	proposeSkillDrafts,
	rejectProposedSkill,
	sanitizeSkillName,
} from "./learning/skill-promotion.js";
import { createArtifactToolPack } from "./tools/artifact.tools.js";
import {
	ApprovalDecisionSignal,
	createExampleDomainToolPack,
	InMemoryDomainStore,
	type DomainRecord,
	type DomainStore,
	type InMemoryDomainStoreHooks,
} from "./tools/example-domain.tools.js";
import { createMemoryToolPack } from "./tools/memory.tools.js";
import { createHttpServer } from "./transport/http-server.js";

export interface BootstrapHarnessOptions {
	/**
	 * Gated self-improvement loop. `false` disables it (used by the isolated probe harnesses it spawns itself).
	 * `proposer` overrides the default fast-model proposer; `gate` tunes the acceptance gate.
	 */
	readonly improvements?: false | { readonly proposer?: CandidateProposer; readonly gate?: GateOptions; readonly maxCandidates?: number };
	/**
	 * Default behaviour of the offline TestModel when no scripted turn is queued. Lets deterministic tests make the
	 * "model" react to its system prompt (also inherited by self-improvement probe harnesses).
	 */
	readonly mockMainHandler?: TestModelHandler;
	/** Apply operator-approved prompt addenda from `config.improvementsDir` on startup (default true). */
	readonly applyActiveImprovements?: boolean;
	readonly config?: Partial<AppConfig>;
	readonly capabilities?: Partial<AppCapabilities>;
	/**
	 * Optional embeddings for hybrid memory retrieval (lexical FTS re-ranked with cosine similarity).
	 * Ignored when `capabilities.memory` is supplied (configure that service directly).
	 */
	readonly embeddingProvider?: EmbeddingProvider;
	readonly domainStore?: DomainStore;
	readonly wakeScheduler?: WakeScheduler;
	readonly now?: () => number;
	readonly staticSystemPrompt?: string;
	readonly toolPermissions?: Readonly<Record<string, string>>;
	readonly consolidationGates?: Partial<ConsolidationGateConfig>;
	readonly autoResume?: boolean;
	readonly registerRealProviders?: boolean;
	/** Budget enforcement extras; the limits themselves come from `config` (MAX_CONVERSATION_TOKENS etc.). */
	readonly budget?: {
		/** Dollar cost of one usage record; default is Pi AI's catalog-derived `usage.cost.total`. */
		readonly estimateCostUsd?: (usage: UsageRecord) => number;
		/** Called whenever a generation is blocked by a budget; default writes a JSON warning line to stderr. */
		readonly onDiagnostic?: (diagnostic: BudgetDiagnostic) => void;
	};
	/** Fast-model resilience overrides (injectable sleep/clock/random for tests, event sink). */
	readonly fastModelResilience?: Pick<FastModelClientOptions, "deps" | "onEvent" | "breaker">;
	/** Retry/backoff for the sleep-cycle fast-model consolidate phase (defaults: 3 attempts, 250ms base). */
	readonly consolidationRetry?: SleepCycleRetryOptions;
	/**
	 * Periodic sleep-cycle scheduler overrides. `false` disables it. Defaults come from
	 * `config.sleepCycleIntervalMs` / `config.sleepCycleSchedulerEnabled`.
	 */
	readonly sleepCycleScheduler?:
		| false
		| Partial<Omit<SleepCycleSchedulerOptions, "harness" | "memory" | "appDb">>;
}

export interface BootstrappedHarness {
	readonly config: AppConfig;
	readonly harness: HarnessPort;
	readonly capabilities: AppCapabilities;
	readonly testModel: TestModelController;
	readonly domainStore: DomainStore;
	readonly appDb: AppDatabase;
	readonly simLog: SimLog;
	readonly skillForge: SkillForge;
	readonly skillDojo: SkillDojo;
	readonly wakeScheduler: WakeScheduler;
	/** Per-tenant daily spend ledger and budget-block event log (tables created lazily in the app DB). */
	readonly budgetStore: BudgetStore;
	/** Effective budget limits (0 = unlimited). */
	readonly budgetLimits: BudgetLimits;
	/** Fast-model per-provider circuit breakers (`.snapshot()` for health reporting). */
	readonly circuitBreakers: ProviderCircuitBreakers;
	/** Periodic sleep-cycle scheduler (inactive when disabled via config or `sleepCycleScheduler: false`). */
	readonly sleepCycleScheduler: SleepCycleScheduler;
	/** Codemode sandbox handle; undefined when `CODEMODE_ENABLED=false`. */
	readonly codemode?: CodemodeHandle;
	/** Gated prompt self-improvement (propose/probe/compose, human approval). Undefined for isolated probe harnesses. */
	readonly improvements?: ImprovementService;
	close(): Promise<void>;
}

/**
 * Composes and opens the complete durable agent runtime behind the Pi-free
 * `HarnessPort` facade. Safe for both standalone HTTP execution and direct
 * TypeScript embedding inside a larger LLM application.
 */
export async function bootstrapHarness(options: BootstrapHarnessOptions = {}): Promise<BootstrappedHarness> {
	const baseConfig = loadConfig();
	const config: AppConfig = {
		...baseConfig,
		...options.config,
	};

	const now = options.now ?? (() => Date.now());
	const appDb = AppDatabase.open({ dbPath: config.appDbPath });

	// Spend ledger shared by the main-conversation budget extension and the background fast-model budget.
	const budgetLimits = budgetLimitsFromConfig(config);
	const budgetStore = new BudgetStore(appDb.db, { now });
	const fastModelBudget =
		config.maxFastModelDailyTokens > 0 || config.maxFastModelDailyCostUsd > 0
			? createFastModelBudget({
					store: budgetStore,
					maxDailyTokens: config.maxFastModelDailyTokens,
					maxDailyCostUsd: config.maxFastModelDailyCostUsd,
					now,
					estimateCostUsd: options.budget?.estimateCostUsd,
				})
			: undefined;

	const catalog = createModelCatalog({
		mainModel: config.mainModel,
		fastModel: config.fastModel,
		useMockModels: config.useMockModels,
		registerRealProviders: options.registerRealProviders,
		fastModelResilience: {
			fallbacks: config.fastModelFallbacks,
			timeoutMs: config.fastModelTimeoutMs,
			maxRetries: config.fastModelMaxRetries,
			...(fastModelBudget ? { guard: fastModelBudget.guard, onUsage: fastModelBudget.onUsage } : {}),
			...options.fastModelResilience,
			breaker: {
				failureThreshold: config.circuitBreakerFailureThreshold,
				cooldownMs: config.circuitBreakerCooldownMs,
				...options.fastModelResilience?.breaker,
			},
		},
	});

	if (options.mockMainHandler) catalog.testModel.setDefaultMainHandler(options.mockMainHandler);

	const memory =
		options.capabilities?.memory ??
		new SqliteMemoryService(appDb, {
			now,
			...(options.embeddingProvider ? { embeddingProvider: options.embeddingProvider } : {}),
		});
	const artifacts =
		options.capabilities?.artifacts ??
		new SqliteArtifactStore(appDb, {
			artifactsDir: config.appDbPath === ":memory:" ? undefined : config.artifactsDir,
			now,
		});
	const telemetry = options.capabilities?.telemetry ?? new SqliteTelemetryStore(appDb, { now });

	const capabilities: AppCapabilities = {
		memory,
		artifacts,
		telemetry,
		fastModel: options.capabilities?.fastModel ?? catalog.fastModelClient,
		skillsDir: options.capabilities?.skillsDir ?? config.skillsDir,
		maxToolOutputBytes: options.capabilities?.maxToolOutputBytes ?? config.maxToolOutputBytes,
		staleSkillDays: options.capabilities?.staleSkillDays ?? config.staleSkillDays,
	};

	const skillForge = new SkillForge();
	const skillDojo = new SkillDojo(capabilities.telemetry, { staleSkillDays: capabilities.staleSkillDays });
	const simLog = new SimLog(appDb, { now });

	const registryManager = new HarnessRegistryManager();

	// Install isolated core extensions in deterministic order:
	// guard-shaping runs before capture-telemetry so telemetry observes blocked & offloaded diagnostics.
	const basePrompt = options.staticSystemPrompt ?? DEFAULT_STATIC_SYSTEM_PROMPT;
	const improvementStore = new ImprovementStore(config.improvementsDir, now);
	const effectivePrompt =
		options.applyActiveImprovements === false ? basePrompt : composePrompt(basePrompt, improvementStore.activeAddenda());
	registryManager.installCoreExtension(createCorePromptExtension({ staticSystemPrompt: effectivePrompt }));
	registryManager.installCoreExtension(
		createGuardShapingExtension({
			capabilities,
			toolPermissions: options.toolPermissions,
			maxToolOutputBytes: capabilities.maxToolOutputBytes,
		}),
	);
	// Budget enforcement runs right after the guard: it vetoes a generation (via the gated Models below)
	// before any provider call, and meters per-tenant daily spend.
	const budgetGate = new BudgetGate();
	registryManager.installCoreExtension(
		createBudgetExtension({
			store: budgetStore,
			gate: budgetGate,
			limits: budgetLimits,
			now,
			estimateCostUsd: options.budget?.estimateCostUsd,
			onDiagnostic:
				options.budget?.onDiagnostic ??
				((diagnostic) => {
					process.stderr.write(`${JSON.stringify({ level: "warn", event: "budget_blocked", ...diagnostic })}\n`);
				}),
			onError: (error) => {
				process.stderr.write(
					`${JSON.stringify({ level: "error", event: "budget_check_failed", message: error instanceof Error ? error.message : String(error) })}\n`,
				);
			},
		}),
	);
	registryManager.installCoreExtension(createCaptureTelemetryExtension(capabilities));
	const sleepCycle = createSleepCycleExtension({
		capabilities,
		consolidationGates: options.consolidationGates,
		skillForge,
		skillDojo,
		simLog,
		consolidationRetry: options.consolidationRetry,
	});
	registryManager.installCoreExtension(sleepCycle.extension);
	registryManager.installCoreExtension(createSkillsLoaderExtension(capabilities));
	// Codemode last: its nested tool calls pass through every hook installed above (tenant guard, budget, capture).
	const codemode: CodemodeHandle | undefined = config.codemodeEnabled
		? createHeliosCodemode({
				memoryLimitBytes: config.codemodeMemoryMb * 1024 * 1024,
				inlineBudget: config.codemodeInlineBudget,
				maxWorkers: config.codemodeMaxWorkers,
				timeoutMs: config.codemodeTimeoutMs,
			})
		: undefined;
	if (codemode) registryManager.installCoreExtension(codemode.extension);

	// Register modular toggleable ToolPacks
	const memoryPack = createMemoryToolPack(capabilities.memory);
	const artifactPack = createArtifactToolPack(capabilities.artifacts);
	const domainBundle = createExampleDomainToolPack({ store: options.domainStore });

	registryManager.registerToolPack(memoryPack, { enabled: true });
	registryManager.registerToolPack(artifactPack, { enabled: true });
	registryManager.registerToolPack(domainBundle.pack, { enabled: true });

	const wakeScheduler = options.wakeScheduler ?? new NodeWakeScheduler({ now });

	let hostRuntime: HostRuntime;
	try {
		hostRuntime = await HostRuntime.open({
			piDbPath: config.piDbPath,
			tablePrefix: "pi_",
			models: createBudgetGatedModels(catalog.models, budgetGate),
			registry: registryManager.getRegistry(),
			// Main-model timeout + durable retry/backoff live in Pi Durable's settings (no extra wrapper = no double retry).
			settings: buildHarnessSettings(config),
			// Default maxTurns for conversations/forks that don't set one.
			conversationCreated: createDefaultMaxTurnsInitializer(config.defaultMaxTurns),
			wakeScheduler,
			now,
			autoResume: options.autoResume ?? true,
		});
	} catch (error) {
		appDb.close();
		throw error;
	}

	const conversationSvc = new ConversationService({
		hostRuntime,
		registryManager,
		capabilities,
		defaultModel: catalog.mainModel,
		sleepCycleTask: sleepCycle.SleepCycleTask,
		signals: [ApprovalDecisionSignal],
		now,
		onClose: () => {
			appDb.close();
		},
	});

	const schedulerOverrides = options.sleepCycleScheduler === false ? undefined : options.sleepCycleScheduler;
	const sleepCycleScheduler = new SleepCycleScheduler({
		intervalMs: config.sleepCycleIntervalMs,
		enabled: config.sleepCycleSchedulerEnabled && options.sleepCycleScheduler !== false,
		now,
		...schedulerOverrides,
		harness: conversationSvc,
		memory: capabilities.memory,
		appDb,
	});
	sleepCycleScheduler.start();

	let improvements: ImprovementService | undefined;
	if (options.improvements !== false) {
		const improvementOptions = options.improvements ?? {};
		const proposer =
			improvementOptions.proposer ??
			(capabilities.fastModel
				? new FastModelProposer(capabilities.fastModel)
				: { propose: async () => [] });
		// Probes are fully isolated harnesses (in-memory databases, no scheduler, no nested loop) over the same models.
		const probeFactory = async (systemPrompt: string) => {
			const probe = await bootstrapHarness({
				...options,
				improvements: false,
				applyActiveImprovements: false,
				staticSystemPrompt: systemPrompt,
				sleepCycleScheduler: false,
				autoResume: false,
				wakeScheduler: undefined,
				capabilities: { skillsDir: capabilities.skillsDir, maxToolOutputBytes: capabilities.maxToolOutputBytes },
				config: {
					...config,
					piDbPath: ":memory:",
					appDbPath: ":memory:",
					sleepCycleSchedulerEnabled: false,
					improvementsDir: config.improvementsDir,
				},
			});
			return {
				runner: new ReplayEvalRunner({ harness: probe.harness, testModel: probe.testModel }),
				close: () => probe.close(),
			};
		};
		improvements = new ImprovementService({
			store: improvementStore,
			simLog,
			evalsDir: config.evalsDir,
			basePrompt,
			applyPrompt: (prompt) => registryManager.replaceExtension(createCorePromptExtension({ staticSystemPrompt: prompt })),
			improver: new SelfImprover({
				store: improvementStore,
				simLog,
				probeFactory,
				proposer,
				basePrompt,
				...(improvementOptions.gate ? { gate: improvementOptions.gate } : {}),
				...(improvementOptions.maxCandidates ? { maxCandidates: improvementOptions.maxCandidates } : {}),
			}),
		});
	}

	return {
		config,
		harness: conversationSvc,
		capabilities,
		testModel: catalog.testModel,
		domainStore: domainBundle.store,
		appDb,
		simLog,
		skillForge,
		skillDojo,
		wakeScheduler,
		budgetStore,
		budgetLimits,
		circuitBreakers: catalog.circuitBreakers,
		sleepCycleScheduler,
		...(codemode ? { codemode } : {}),
		...(improvements ? { improvements } : {}),
		close: async () => {
			sleepCycleScheduler.dispose();
			try {
				await conversationSvc.close();
			} finally {
				await codemode?.close();
			}
		},
	};
}

// Operations: health probes and structured logging
export {
	createHealthChecker,
	readPackageVersion,
	type HealthChecker,
	type HealthCheckerOptions,
	type HealthCheckResult,
	type LivenessReport,
	type ReadinessReport,
} from "./health.js";
export { createLogger, resolveLogLevel, type LogLevel, type LogFields, type Logger, type LoggerOptions } from "./logger.js";

// Budgets & model resilience
export {
	BudgetGate,
	BudgetStore,
	budgetLimitsFromConfig,
	createFastModelBudget,
	evaluateBudget,
	FastModelBudgetExceededError,
	UNLIMITED_BUDGET,
	type FastModelBudget,
	type FastModelBudgetOptions,
	type BudgetEventRecord,
	type BudgetLimits,
	type BudgetViolation,
	type SpendTotals,
	type UsageRecord,
} from "./runtime/budget.js";
export { type BudgetDiagnostic } from "./extensions/budget.ext.js";
export {
	AllModelsFailedError,
	assertProductionModelSafety,
	buildHarnessSettings,
	CircuitBreaker,
	CircuitOpenError,
	classifyError,
	MockModelsInProductionError,
	ModelTimeoutError,
	ProviderCircuitBreakers,
	withRetry,
	withTimeout,
	type CircuitState,
	type ClassifiedError,
	type ErrorKind,
	type ModelSafetyConfig,
	type ResilienceEvent,
} from "./runtime/resilience.js";
export { createFastModelClient, type FastModelClientOptions } from "./runtime/model-catalog.js";

// Public Pi-free exports
export {
	// Config
	loadConfig,
	parseModelSpec,
	type AppConfig,
	// Contracts
	DEFAULT_PERMISSIONS,
	hasPermission,
	validateTenantContext,
	type TenantContext,
	type AppCapabilities,
	type ArtifactMetadata,
	type ArtifactReadResult,
	type ArtifactStore,
	type FastModelClient,
	type FrictionSignal,
	type FrictionSignalKind,
	type LongTermMemory,
	type MemoryKind,
	type MemoryMutationOp,
	type MemorySearchResult,
	type MemoryService,
	type MemoryStatus,
	type MidTermEpisode,
	type RunTelemetryRecord,
	type SkillHealthMetric,
	type TelemetryStore,
	type ToolInvocationMetric,
	type ToolTraceItem,
	type ConversationInfo,
	type CreateConversationInput,
	type HarnessEvent,
	type HarnessPort,
	type LiveStatusView,
	type ModelSpec,
	type PendingApproval,
	type SettledSubmission,
	type SignalDeliveryInput,
	type SignalDeliveryResult,
	type SubmissionHandle,
	type SubmitMessageInput,
	type TaskSettlementView,
	type ToolPackSummary,
	type ToolSlotView,
	type TranscriptEntryView,
	type UsageBucketView,
	type UsageSummaryView,
	// Transport
	createHttpServer,
	// Runtime utilities (Pi-free interfaces only)
	SingleWriterLockError,
	NodeWakeScheduler,
	ManualWakeScheduler,
	type WakeScheduler,
	type WakeState,
	type RecordedModelRequest,
	type ScriptedTurnOutput,
	type TestModelController,
	type TestModelHandler,
	// Memory & Learning
	AppDatabase,
	type AppDatabaseOptions,
	SqliteMemoryService,
	SqliteArtifactStore,
	SqliteTelemetryStore,
	detectFrictionSignals,
	type FrictionAnalysisInput,
	type FrictionAnalysisResult,
	DEFAULT_CONSOLIDATION_GATES,
	MemoryConsolidator,
	type ConsolidationGateConfig,
	type ConsolidationPlan,
	type ConsolidationReport,
	type ProposedConsolidationOp,
	type RejectedProposal,
	SkillForge,
	type MinedWorkflow,
	type SkillForgeOptions,
	SkillDojo,
	type SkillDojoEvaluation,
	type SkillDojoOptions,
	SleepCycleScheduler,
	type SleepCycleSchedulerOptions,
	type SleepTickReport,
	type SleepSchedulerTimer,
	type PendingEpisodeScope,
	promoteProposedSkill,
	rejectProposedSkill,
	listProposedSkills,
	proposeSkillDrafts,
	sanitizeSkillName,
	PROPOSED_SKILLS_DIRNAME,
	SimLog,
	type SimLogEntry,
	type SimLogEntryInput,
	type SimLogOutcome,
	type SimLogSummary,
	// Skills catalog helpers
	loadSkillCatalog,
	formatFrozenSkillsIndex,
	readSkillBody,
	type SkillFrontmatter,
	// Domain test double
	InMemoryDomainStore,
	type DomainRecord,
	type DomainStore,
	type InMemoryDomainStoreHooks,
	// Defaults
	DEFAULT_STATIC_SYSTEM_PROMPT,
	DEFAULT_TOOL_PERMISSIONS,
	// Evals & gated self-improvement
	ReplayEvalRunner,
	loadEvalSet,
	parseEvalCases,
	buildEvalCaseFromTranscript,
	redactForEval,
	readBaseline,
	writeBaseline,
	checkAgainstBaseline,
	EvalSetError,
	ImprovementStore,
	ImprovementService,
	ImprovementError,
	SelfImprover,
	FastModelProposer,
	StaticProposer,
	evaluateGate,
	lintAddendum,
	composePrompt,
	ADDENDUM_LIMITS,
	type EvalBaseline,
	type RegressionReport,
	type ImprovementRecord,
	type ImprovementRunReport,
	type CandidateDraft,
	type CandidateProposer,
	type GateVerdict,
	aggregateParetoMetrics,
	comparePareto,
	type EvalCase,
	type EvalCaseResult,
	type EvalSeedTurn,
	type EvalSplit,
	type EvalSuiteResult,
	type ParetoComparison,
	type ParetoMetrics,
};
