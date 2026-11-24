import { loadConfig, parseModelSpec, type AppConfig } from "./config.js";
import type {
	AppCapabilities,
	ArtifactMetadata,
	ArtifactReadResult,
	ArtifactStore,
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
import { createCorePromptExtension, DEFAULT_STATIC_SYSTEM_PROMPT } from "./extensions/core-prompt.ext.js";
import { createGuardShapingExtension, DEFAULT_TOOL_PERMISSIONS } from "./extensions/guard-shaping.ext.js";
import {
	createSkillsLoaderExtension,
	formatFrozenSkillsIndex,
	loadSkillCatalog,
	readSkillBody,
	type SkillFrontmatter,
} from "./extensions/skills-loader.ext.js";
import { createSleepCycleExtension } from "./extensions/sleep-cycle.ext.js";
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
import { HostRuntime, SingleWriterLockError } from "./runtime/host-runtime.js";
import {
	createModelCatalog,
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
	readonly config?: Partial<AppConfig>;
	readonly capabilities?: Partial<AppCapabilities>;
	readonly domainStore?: DomainStore;
	readonly wakeScheduler?: WakeScheduler;
	readonly now?: () => number;
	readonly staticSystemPrompt?: string;
	readonly toolPermissions?: Readonly<Record<string, string>>;
	readonly consolidationGates?: Partial<ConsolidationGateConfig>;
	readonly autoResume?: boolean;
	readonly registerRealProviders?: boolean;
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

	const catalog = createModelCatalog({
		mainModel: config.mainModel,
		fastModel: config.fastModel,
		useMockModels: config.useMockModels,
		registerRealProviders: options.registerRealProviders,
	});

	const memory = options.capabilities?.memory ?? new SqliteMemoryService(appDb, { now });
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
	registryManager.installCoreExtension(
		createCorePromptExtension({
			staticSystemPrompt: options.staticSystemPrompt ?? DEFAULT_STATIC_SYSTEM_PROMPT,
		}),
	);
	registryManager.installCoreExtension(
		createGuardShapingExtension({
			capabilities,
			toolPermissions: options.toolPermissions,
			maxToolOutputBytes: capabilities.maxToolOutputBytes,
		}),
	);
	registryManager.installCoreExtension(createCaptureTelemetryExtension(capabilities));
	const sleepCycle = createSleepCycleExtension({
		capabilities,
		consolidationGates: options.consolidationGates,
		skillForge,
		skillDojo,
	});
	registryManager.installCoreExtension(sleepCycle.extension);
	registryManager.installCoreExtension(createSkillsLoaderExtension(capabilities));

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
			models: catalog.models,
			registry: registryManager.getRegistry(),
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
		close: () => conversationSvc.close(),
	};
}

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
	// Evals
	ReplayEvalRunner,
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
