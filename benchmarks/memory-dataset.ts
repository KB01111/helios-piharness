/**
 * Comprehensive Agent Memory Benchmark Dataset (LoCoMo + LongMemEval + Multi-Hop + Cognee-inspired).
 *
 * Covers 8 retrieval & reasoning dimensions:
 *  1. single_hop       — exact facts, preferences, technical identifiers (snake_case, camelCase, dotted keys)
 *  2. semantic_gap     — zero lexical overlap paraphrase queries requiring semantic understanding
 *  3. multi_hop        — 2-hop relational chains across separate memories bridged by shared entities
 *  4. knowledge_update — evolving facts with SUPERSEDE chains (latest truth must win; stale must stay out)
 *  5. temporal_window  — date/month/year/range-anchored recall (Cognee TEMPORAL / LongMemEval temporal)
 *  6. historical_audit — "before / previously / originally" queries targeting superseded history
 *  7. abstention       — unanswerable negative queries (measures false-positive rejection)
 *  8. session_recent   — unconsolidated mid-term episodes captured before sleep-cycle runs
 */

import type { MemoryKind } from "../src/contracts/app-capabilities.js";

export type QueryCategory =
	| "single_hop"
	| "semantic_gap"
	| "multi_hop"
	| "knowledge_update"
	| "temporal_window"
	| "historical_audit"
	| "abstention"
	| "session_recent";

export interface BenchmarkMemory {
	/** Stable short tag embedded cleanly so all engines (including chunk-only ones) can be scored identically. */
	readonly id: string;
	readonly key: string;
	readonly kind: MemoryKind;
	readonly content: string;
	readonly confidence?: number;
	readonly pinned?: boolean;
	readonly evidenceCount?: number;
	/** Relative offset in days before benchmark `nowMs` (negative = older). */
	readonly ageDays?: number;
	/** If present, this memory supersedes the memory with `supersedesId` (same or drifted key). */
	readonly supersedesId?: string;
}

export interface BenchmarkEpisode {
	readonly id: string;
	readonly conversationId: string;
	readonly runId: string;
	readonly userMessage: string;
	readonly assistantSummary: string;
	readonly ageMinutes?: number;
}

export interface BenchmarkQuery {
	readonly id: string;
	readonly category: QueryCategory;
	readonly query: string;
	/** Primary gold memory IDs (hitting any counts as recall; hitting all counts for multi-hop full recall). */
	readonly expectedIds: readonly string[];
	/** For multi-hop: BOTH bridge and answer IDs needed for full reasoning. */
	readonly requiredAllIds?: readonly string[];
	/** For knowledge_update: stale/superseded IDs that must NOT outrank or pollute top results. */
	readonly forbiddenIds?: readonly string[];
	/** For abstention: query has NO answer in memory. */
	readonly expectAbstain?: boolean;
	/** For session_recent: expected episode IDs. */
	readonly expectedEpisodeIds?: readonly string[];
}

export const BENCHMARK_NOW_MS = Date.UTC(2026, 10, 25, 12, 0, 0); // 2026-11-25T12:00:00Z

export const BENCHMARK_MEMORIES: readonly BenchmarkMemory[] = [
	// ── 1. Single-hop technical & user profile memories ────────────────────────────────────────
	{
		id: "M01",
		key: "user.editor.primary",
		kind: "preference",
		content: "Daily driver code editor is Neovim with lazy.nvim and Lua configuration.",
		confidence: 0.95,
		pinned: true,
		evidenceCount: 5,
		ageDays: 45,
	},
	{
		id: "M02",
		key: "infra.k8s.prod_region",
		kind: "fact",
		content: "Production Kubernetes cluster runs in AWS us-east-1 behind CloudFront WAF.",
		confidence: 0.92,
		evidenceCount: 3,
		ageDays: 30,
	},
	{
		id: "M03",
		key: "repo.retry.config",
		kind: "procedure",
		content: "Set maxRetryCount to 4 and backoffBaseMs to 250 in gateway_client_config.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 20,
	},
	{
		id: "M04",
		key: "db.connection_pooler",
		kind: "fact",
		content: "Transaction pooling uses pg_bouncer in transaction mode with max_client_conn 2000.",
		confidence: 0.91,
		evidenceCount: 3,
		ageDays: 60,
	},
	{
		id: "M05",
		key: "release.sdk.version",
		kind: "fact",
		content: "Helios runtime pins @earendil-works/pi-durable at v1.6.0 with SQLite WAL mode.",
		confidence: 0.94,
		evidenceCount: 4,
		ageDays: 10,
	},
	{
		id: "M06",
		key: "user.deploy.constraint",
		kind: "constraint",
		content: "Never push production schema migrations on Fridays after 15:00 CET.",
		confidence: 0.98,
		pinned: true,
		evidenceCount: 6,
		ageDays: 90,
	},
	{
		id: "M07",
		key: "user.testing.runner",
		kind: "procedure",
		content: "Run TypeScript unit tests with vitest run and strict tsc --noEmit before committing.",
		confidence: 0.92,
		evidenceCount: 4,
		ageDays: 25,
	},
	{
		id: "M08",
		key: "user.coffee.preference",
		kind: "preference",
		content: "Discovered specialty pour-over coffee with Ethiopian Yirgacheffe washed beans.",
		confidence: 0.85,
		evidenceCount: 2,
		ageDays: 120,
	},
	{
		id: "M09",
		key: "cooking.pasta.technique",
		kind: "procedure",
		content: "Salt pasta water generously like the sea, never break spaghetti, and reserve starchy pasta water to emulsify the sauce.",
		confidence: 0.88,
		evidenceCount: 2,
		ageDays: 150,
	},
	{
		id: "M10",
		key: "devops.docker.repro",
		kind: "fact",
		content: "Adopted Docker in 2015 with a 15-line Dockerfile to eliminate works-on-my-machine drift in Python services.",
		confidence: 0.86,
		evidenceCount: 2,
		ageDays: 200,
	},
	{
		id: "M11",
		key: "homelab.network.setup",
		kind: "fact",
		content: "Apartment mesh Wi-Fi uses VLAN isolation for IoT devices and Pi-hole for DNS ad blocking.",
		confidence: 0.87,
		evidenceCount: 2,
		ageDays: 110,
	},
	{
		id: "M12",
		key: "backup.policy.rule",
		kind: "constraint",
		content: "After a 2016 laptop hard drive crash, follow the 3-2-1 backup rule using an external drive and encrypted cloud storage.",
		confidence: 0.93,
		evidenceCount: 3,
		ageDays: 180,
	},
	{
		id: "M13",
		key: "user.communication.style",
		kind: "preference",
		content: "Keep technical answers direct, concise, and backed by reproducible command output.",
		confidence: 0.96,
		pinned: true,
		evidenceCount: 7,
		ageDays: 15,
	},
	{
		id: "M14",
		key: "cooking.grandma.stew",
		kind: "fact",
		content: "Grandmother's cast-iron beef stew simmers for four hours and uses red wine vinegar at the end as the secret ingredient.",
		confidence: 0.84,
		evidenceCount: 2,
		ageDays: 240,
	},
	{
		id: "M15",
		key: "git.recovery.workflow",
		kind: "procedure",
		content: "Use git reflog to recover commits lost after an accidental git reset --hard.",
		confidence: 0.9,
		evidenceCount: 3,
		ageDays: 140,
	},
	{
		id: "M16",
		key: "user.timezone.location",
		kind: "fact",
		content: "User is based in Berlin Kreuzberg in the Europe/Berlin CET timezone.",
		confidence: 0.95,
		evidenceCount: 4,
		ageDays: 80,
	},
	{
		id: "M17",
		key: "security.auth.token",
		kind: "constraint",
		content: "Rotate OAuth2 signing keys every 90 days and store credentials exclusively in HashiCorp Vault.",
		confidence: 0.94,
		evidenceCount: 3,
		ageDays: 35,
	},
	{
		id: "M18",
		key: "ci.canary.pattern",
		kind: "procedure",
		content: "After GitHub Actions silently skipped tests in 2021, keep an intentional canary assertion check in CI setup.",
		confidence: 0.89,
		evidenceCount: 2,
		ageDays: 130,
	},

	// ── 2. Multi-Hop Relational Chains (2-hop entity bridges across distinct memories!) ────────
	// Chain 1: Project Atlas -> Marcus Chen -> ClickHouse
	{
		id: "MH01A",
		key: "org.project.atlas.lead",
		kind: "fact",
		content: "Project Atlas is led by staff engineer Marcus Chen.",
		confidence: 0.92,
		evidenceCount: 2,
		ageDays: 40,
	},
	{
		id: "MH01B",
		key: "team.marcus_chen.telemetry_db",
		kind: "fact",
		content: "Marcus Chen's team stores high-volume telemetry analytics in ClickHouse.",
		confidence: 0.91,
		evidenceCount: 2,
		ageDays: 38,
	},

	// Chain 2: Kestrel API -> Horizon Cluster -> Aurora PostgreSQL
	{
		id: "MH02A",
		key: "service.kestrel.cluster",
		kind: "fact",
		content: "Kestrel API is deployed exclusively on the Horizon Cluster.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 42,
	},
	{
		id: "MH02B",
		key: "infra.horizon_cluster.db",
		kind: "fact",
		content: "Horizon Cluster connects to Aurora PostgreSQL with read replicas in Frankfurt.",
		confidence: 0.91,
		evidenceCount: 2,
		ageDays: 39,
	},

	// Chain 3: Billing Service -> Elena Rostova -> Go 1.23 & gRPC
	{
		id: "MH03A",
		key: "org.billing.owner",
		kind: "fact",
		content: "Billing Service architecture is owned by Elena Rostova.",
		confidence: 0.92,
		evidenceCount: 2,
		ageDays: 50,
	},
	{
		id: "MH03B",
		key: "person.elena_rostova.stack",
		kind: "fact",
		content: "Elena Rostova standardizes all her backend services on Go 1.23 and gRPC protobufs.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 48,
	},

	// Chain 4: Cat Luna -> Dr. Priya Patel -> Westside Vet Clinic
	{
		id: "MH04A",
		key: "pet.luna.vet",
		kind: "fact",
		content: "Our shelter cat Luna has an orange left ear and sees veterinarian Dr. Priya Patel.",
		confidence: 0.93,
		evidenceCount: 3,
		ageDays: 70,
	},
	{
		id: "MH04B",
		key: "contact.priya_patel.clinic",
		kind: "fact",
		content: "Dr. Priya Patel practices at Westside Vet Clinic on Schillerstrasse.",
		confidence: 0.91,
		evidenceCount: 2,
		ageDays: 68,
	},

	// Chain 5: Sourdough starter -> Clint Yeastwood -> organic rye flour
	{
		id: "MH05A",
		key: "baking.sourdough.name",
		kind: "fact",
		content: "My lockdown sourdough starter is named Clint Yeastwood.",
		confidence: 0.88,
		evidenceCount: 2,
		ageDays: 160,
	},
	{
		id: "MH05B",
		key: "baking.clint_yeastwood.feed",
		kind: "procedure",
		content: "Feed Clint Yeastwood a 1:2:2 ratio of warm water and organic dark rye flour every Sunday.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 155,
	},

	// Chain 6: Solaris Mobile -> Vega Design System -> Tailwind CSS v4
	{
		id: "MH06A",
		key: "project.solaris.ui",
		kind: "fact",
		content: "Solaris Mobile web portal is built on the Vega Design System.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 28,
	},
	{
		id: "MH06B",
		key: "design.vega.styling",
		kind: "fact",
		content: "Vega Design System uses Tailwind CSS v4 tokens and Radix primitives.",
		confidence: 0.91,
		evidenceCount: 2,
		ageDays: 26,
	},

	// Chain 7: Compliance Audit -> Henrik Lindqvist -> SOC2 Type II
	{
		id: "MH07A",
		key: "security.audit.lead",
		kind: "fact",
		content: "Annual compliance audit sign-off goes through Henrik Lindqvist.",
		confidence: 0.92,
		evidenceCount: 2,
		ageDays: 33,
	},
	{
		id: "MH07B",
		key: "person.henrik_lindqvist.scope",
		kind: "fact",
		content: "Henrik Lindqvist manages our SOC2 Type II and ISO 27001 evidence binders.",
		confidence: 0.91,
		evidenceCount: 2,
		ageDays: 31,
	},

	// Chain 8: Meridian Pipeline -> Redpanda Broker -> Avro Schema Registry
	{
		id: "MH08A",
		key: "data.meridian.transport",
		kind: "fact",
		content: "Meridian Pipeline streams ingestion events through the Redpanda Broker.",
		confidence: 0.91,
		evidenceCount: 2,
		ageDays: 22,
	},
	{
		id: "MH08B",
		key: "infra.redpanda.serialization",
		kind: "fact",
		content: "Redpanda Broker enforces Avro Schema Registry validation on every topic.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 21,
	},

	// Chain 9: Mentor David -> Socratic Review -> Characterization Tests
	{
		id: "MH09A",
		key: "career.mentor.david",
		kind: "fact",
		content: "Senior mentor David Nakamura taught me Socratic code reviews during my first job.",
		confidence: 0.88,
		evidenceCount: 2,
		ageDays: 210,
	},
	{
		id: "MH09B",
		key: "engineering.david_nakamura.rule",
		kind: "procedure",
		content: "David Nakamura insists on writing characterization tests before refactoring legacy monolith modules.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 205,
	},

	// Chain 10: Nordic Gateway -> Cerberus Proxy -> mTLS Client Certs
	{
		id: "MH10A",
		key: "infra.nordic_gateway.ingress",
		kind: "fact",
		content: "Nordic Gateway routes all external traffic through Cerberus Proxy.",
		confidence: 0.91,
		evidenceCount: 2,
		ageDays: 19,
	},
	{
		id: "MH10B",
		key: "security.cerberus_proxy.auth",
		kind: "constraint",
		content: "Cerberus Proxy requires mutual TLS mTLS client certificates issued by our internal CA.",
		confidence: 0.94,
		evidenceCount: 3,
		ageDays: 18,
	},

	// Chain 11: Zephyr Search -> BM25F Index -> Porter Stemmer
	{
		id: "MH11A",
		key: "search.zephyr.engine",
		kind: "fact",
		content: "Zephyr Search delegates lexical scoring to the Cobalt Tokenizer module.",
		confidence: 0.89,
		evidenceCount: 2,
		ageDays: 27,
	},
	{
		id: "MH11B",
		key: "search.cobalt_tokenizer.algo",
		kind: "fact",
		content: "Cobalt Tokenizer uses Snowball English stemming and Unicode NFKC normalization.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 26,
	},

	// Chain 12: Hackathon Team -> SignBridge App -> MobileNetV3 CNN
	{
		id: "MH12A",
		key: "hackathon.2017.project",
		kind: "fact",
		content: "Our 2017 winning hackathon project was called SignBridge for real-time ASL translation.",
		confidence: 0.87,
		evidenceCount: 2,
		ageDays: 250,
	},
	{
		id: "MH12B",
		key: "ml.signbridge.model",
		kind: "fact",
		content: "SignBridge achieved 85 percent accuracy using a fine-tuned MobileNetV3 convolutional neural network.",
		confidence: 0.88,
		evidenceCount: 2,
		ageDays: 249,
	},

	// Chain 13: Vertex Warehouse -> Snowflake Role -> FinOps Alerting
	{
		id: "MH13A",
		key: "data.vertex.warehouse",
		kind: "fact",
		content: "Vertex Analytics queries run under the Quarry Compute Warehouse.",
		confidence: 0.9,
		evidenceCount: 2,
		ageDays: 14,
	},
	{
		id: "MH13B",
		key: "finops.quarry.budget",
		kind: "constraint",
		content: "Quarry Compute Warehouse auto-suspends after 60 seconds and caps spend at 400 credits per week.",
		confidence: 0.93,
		evidenceCount: 2,
		ageDays: 13,
	},

	// Chain 14: Tokyo Trip -> Ramen Shop -> Ichiran Shibuya
	{
		id: "MH14A",
		key: "travel.japan.guide",
		kind: "fact",
		content: "My friend Kenji Sato planned our culinary itinerary for Tokyo and Kyoto.",
		confidence: 0.86,
		evidenceCount: 2,
		ageDays: 170,
	},
	{
		id: "MH14B",
		key: "person.kenji_sato.favorite_spot",
		kind: "fact",
		content: "Kenji Sato booked us counter seats at Afuri Yuzu Ramen in Ebisu.",
		confidence: 0.87,
		evidenceCount: 2,
		ageDays: 169,
	},

	// Chain 15: Incident Runbook -> PagerDuty Escalation -> On-Call Rotation
	{
		id: "MH15A",
		key: "ops.sev1.channel",
		kind: "procedure",
		content: "Severity-1 checkout outages trigger the Valkyrie Escalation Policy.",
		confidence: 0.94,
		evidenceCount: 3,
		ageDays: 12,
	},
	{
		id: "MH15B",
		key: "ops.valkyrie.paging",
		kind: "procedure",
		content: "Valkyrie Escalation Policy pages the primary SRE on Slack #war-room and falls back to SMS after 5 minutes.",
		confidence: 0.93,
		evidenceCount: 3,
		ageDays: 11,
	},

	// ── 3. Knowledge Updates & Contradiction Chains (SUPERSEDE) ────────────────────────────────
	// Update 1: Package manager: npm -> pnpm -> bun
	{
		id: "KU01_V1",
		key: "user.package_manager",
		kind: "preference",
		content: "Uses npm as the default JavaScript package manager.",
		confidence: 0.85,
		ageDays: 300,
	},
	{
		id: "KU01_V2",
		key: "user.package_manager",
		kind: "preference",
		content: "Switched from npm to pnpm for all TypeScript monorepos.",
		confidence: 0.92,
		ageDays: 120,
		supersedesId: "KU01_V1",
	},
	{
		id: "KU01_V3",
		key: "user.package_manager",
		kind: "preference",
		content: "Migrated from pnpm to bun for fast installs and script execution.",
		confidence: 0.96,
		evidenceCount: 3,
		ageDays: 5,
		supersedesId: "KU01_V2",
	},

	// Update 2: Primary database for side projects: MySQL -> SQLite
	{
		id: "KU02_V1",
		key: "user.sideproject.db",
		kind: "preference",
		content: "Uses MySQL on Debian for personal web projects.",
		confidence: 0.85,
		ageDays: 400,
	},
	{
		id: "KU02_V2",
		key: "user.sideproject.db",
		kind: "preference",
		content: "Uses embedded SQLite in WAL mode via node:sqlite for all new side projects.",
		confidence: 0.95,
		evidenceCount: 4,
		ageDays: 8,
		supersedesId: "KU02_V1",
	},

	// Update 3: Cloud region for EU compliance: us-west-2 -> eu-central-1
	{
		id: "KU03_V1",
		key: "infra.eu_service.region",
		kind: "fact",
		content: "EU customer staging environment originally ran in us-west-2 Oregon.",
		confidence: 0.86,
		ageDays: 250,
	},
	{
		id: "KU03_V2",
		key: "infra.eu_service.region",
		kind: "fact",
		content: "EU customer staging environment moved to eu-central-1 Frankfurt for GDPR compliance.",
		confidence: 0.96,
		evidenceCount: 3,
		ageDays: 10,
		supersedesId: "KU03_V1",
	},

	// Update 4: Code formatter: Prettier -> Biome
	{
		id: "KU04_V1",
		key: "repo.formatter",
		kind: "preference",
		content: "Formats TypeScript repositories with Prettier and ESLint.",
		confidence: 0.88,
		ageDays: 200,
	},
	{
		id: "KU04_V2",
		key: "repo.formatter",
		kind: "preference",
		content: "Replaced Prettier and ESLint with Biome for unified linting and formatting.",
		confidence: 0.95,
		evidenceCount: 3,
		ageDays: 7,
		supersedesId: "KU04_V1",
	},

	// Update 5: Primary LLM model: Claude 3.5 Sonnet -> Claude Sonnet 4.5
	{
		id: "KU05_V1",
		key: "agent.default_model",
		kind: "fact",
		content: "Default main reasoning model is anthropic:claude-3-5-sonnet.",
		confidence: 0.88,
		ageDays: 180,
	},
	{
		id: "KU05_V2",
		key: "agent.default_model",
		kind: "fact",
		content: "Upgraded default main reasoning model to anthropic:claude-sonnet-4-5.",
		confidence: 0.95,
		evidenceCount: 4,
		ageDays: 4,
		supersedesId: "KU05_V1",
	},

	// Update 6: Team standup time: 09:00 -> 10:30
	{
		id: "KU06_V1",
		key: "team.standup.time",
		kind: "fact",
		content: "Daily engineering standup is scheduled at 09:00 CET.",
		confidence: 0.87,
		ageDays: 150,
	},
	{
		id: "KU06_V2",
		key: "team.standup.time",
		kind: "fact",
		content: "Daily engineering standup moved to 10:30 CET to accommodate London engineers.",
		confidence: 0.94,
		evidenceCount: 3,
		ageDays: 6,
		supersedesId: "KU06_V1",
	},

	// Update 7: Note taking app: Notion -> Obsidian
	{
		id: "KU07_V1",
		key: "user.notes.app",
		kind: "preference",
		content: "Stores personal architecture notes in Notion workspaces.",
		confidence: 0.85,
		ageDays: 260,
	},
	{
		id: "KU07_V2",
		key: "user.notes.app",
		kind: "preference",
		content: "Migrated all personal architecture notes from Notion to local Markdown files in Obsidian.",
		confidence: 0.95,
		evidenceCount: 3,
		ageDays: 9,
		supersedesId: "KU07_V1",
	},

	// Update 8: Alerting backend: PagerDuty -> Incident.io
	{
		id: "KU08_V1",
		key: "ops.incident.tool",
		kind: "fact",
		content: "Incident response coordination runs in Opsgenie.",
		confidence: 0.86,
		ageDays: 220,
	},
	{
		id: "KU08_V2",
		key: "ops.incident.tool",
		kind: "fact",
		content: "Migrated incident response coordination from Opsgenie to Incident.io.",
		confidence: 0.95,
		evidenceCount: 3,
		ageDays: 12,
		supersedesId: "KU08_V1",
	},

	// Update 9: API gateway: NGINX -> Envoy
	{
		id: "KU09_V1",
		key: "infra.edge.proxy",
		kind: "fact",
		content: "Edge HTTP routing uses NGINX ingress controller.",
		confidence: 0.88,
		ageDays: 190,
	},
	{
		id: "KU09_V2",
		key: "infra.edge.proxy",
		kind: "fact",
		content: "Replaced NGINX ingress controller with Envoy Gateway for gRPC load balancing.",
		confidence: 0.95,
		evidenceCount: 3,
		ageDays: 11,
		supersedesId: "KU09_V1",
	},

	// Update 10: Shell: Bash -> Zsh -> Fish
	{
		id: "KU10_V1",
		key: "user.shell",
		kind: "preference",
		content: "Uses Bash with custom PS1 prompt on Linux workstations.",
		confidence: 0.84,
		ageDays: 320,
	},
	{
		id: "KU10_V2",
		key: "user.shell",
		kind: "preference",
		content: "Switched terminal shell from Bash to Zsh with Oh My Zsh.",
		confidence: 0.9,
		ageDays: 140,
		supersedesId: "KU10_V1",
	},
	{
		id: "KU10_V3",
		key: "user.shell",
		kind: "preference",
		content: "Now uses Fish shell with Starship prompt for interactive terminal sessions.",
		confidence: 0.96,
		evidenceCount: 3,
		ageDays: 3,
		supersedesId: "KU10_V2",
	},

	// ── 4. Temporal-Anchored Memories (Events across 1998–2025) ────────────────────────────────
	{
		id: "TM01",
		key: "history.eclipse.1999",
		kind: "episodic",
		content: "On August 11, 1999, watched a total solar eclipse from the rooftop in Europe as the sky darkened and birds fell silent.",
		confidence: 0.9,
		ageDays: 300,
	},
	{
		id: "TM02",
		key: "history.outage.2019",
		kind: "episodic",
		content: "In October 2019, our production database crashed at 3 AM because a deployment reduced max pool connections from 100 to 10.",
		confidence: 0.93,
		evidenceCount: 3,
		ageDays: 160,
	},
	{
		id: "TM03",
		key: "history.microservices.2020",
		kind: "episodic",
		content: "In 2020, spent eight months migrating a Python monolith to microservices using the strangler fig pattern, cutting latency by 40 percent.",
		confidence: 0.91,
		evidenceCount: 2,
		ageDays: 140,
	},
	{
		id: "TM04",
		key: "history.alps.2015",
		kind: "episodic",
		content: "In July 2015, hiked the 170-kilometer Tour du Mont Blanc through France, Italy, and Switzerland over eleven days.",
		confidence: 0.89,
		ageDays: 220,
	},
	{
		id: "TM05",
		key: "history.marathon.2016",
		kind: "episodic",
		content: "In April 2016, completed my first marathon along the Ringstrasse in Vienna in 4 hours and 12 minutes.",
		confidence: 0.89,
		ageDays: 210,
	},
	{
		id: "TM06",
		key: "history.berlin.2018",
		kind: "episodic",
		content: "In March 2018, relocated to Berlin and joined a Kreuzberg co-working community.",
		confidence: 0.9,
		ageDays: 190,
	},
	{
		id: "TM07",
		key: "history.opensource.2014",
		kind: "episodic",
		content: "In 2014, merged my first open-source pull request fixing a log file rotation race condition in a Python logging library.",
		confidence: 0.88,
		ageDays: 240,
	},
	{
		id: "TM08",
		key: "history.launch.2024_03",
		kind: "episodic",
		content: "On 2024-03-15, shipped Helios Durable Execution Engine v1.0 with zero-downtime SQLite checkpointing.",
		confidence: 0.95,
		evidenceCount: 3,
		ageDays: 60,
	},
	{
		id: "TM09",
		key: "history.blackout.2003",
		kind: "episodic",
		content: "During the August 2003 Northeast blackout, power stayed out for two days while neighbors sang by candlelight.",
		confidence: 0.87,
		ageDays: 280,
	},
	{
		id: "TM10",
		key: "history.garden.2021",
		kind: "episodic",
		content: "In Spring 2021, planted tomatoes, basil, and peppers on the apartment balcony and harvested weekly pesto by August.",
		confidence: 0.86,
		ageDays: 150,
	},
	{
		id: "TM11",
		key: "history.audit.2025_01",
		kind: "episodic",
		content: "On January 20, 2025, completed the zero-trust tenant isolation security penetration test with zero high findings.",
		confidence: 0.94,
		evidenceCount: 3,
		ageDays: 35,
	},
	{
		id: "TM12",
		key: "history.snowday.1998",
		kind: "episodic",
		content: "In January 1998, heavy snowfall canceled school for three days and we built a snow fort with tunnels in the park.",
		confidence: 0.85,
		ageDays: 310,
	},

	// ── 5. Distractor / Background Noise Memories (realistic corpus density ~100 memories) ────
	{
		id: "D01",
		key: "memory.concert.radiohead",
		kind: "episodic",
		content: "Attended a Radiohead festival concert in 2006 where they opened with There There and Karma Police.",
		confidence: 0.8,
		ageDays: 260,
	},
	{
		id: "D02",
		key: "memory.florence.david",
		kind: "episodic",
		content: "Visited Galleria dell'Accademia in Florence to see Michelangelo's marble statue of David.",
		confidence: 0.82,
		ageDays: 230,
	},
	{
		id: "D03",
		key: "memory.books.tolkien",
		kind: "fact",
		content: "Read the entire Lord of the Rings trilogy in one week during summer break at age fourteen.",
		confidence: 0.82,
		ageDays: 270,
	},
	{
		id: "D04",
		key: "memory.chess.uncle",
		kind: "fact",
		content: "Learned chess at age seven from an uncle and placed third in a regional school tournament.",
		confidence: 0.84,
		ageDays: 250,
	},
	{
		id: "D05",
		key: "memory.dog.rex",
		kind: "fact",
		content: "Childhood German Shepherd dog named Rex lived with the family for thirteen years.",
		confidence: 0.85,
		ageDays: 290,
	},
	{
		id: "D06",
		key: "memory.kyoto.bamboo",
		kind: "episodic",
		content: "Walked alone through Kyoto's Arashiyama bamboo grove during a rainstorm.",
		confidence: 0.83,
		ageDays: 175,
	},
	{
		id: "D07",
		key: "memory.podcasts.commute",
		kind: "preference",
		content: "Listens to roughly ten hours of technology history podcasts per week.",
		confidence: 0.84,
		ageDays: 100,
	},
	{
		id: "D08",
		key: "memory.stargazing.milkyway",
		kind: "episodic",
		content: "Spotted Orion, Cassiopeia, and the Milky Way on a remote countryside camping trip.",
		confidence: 0.8,
		ageDays: 215,
	},
	{
		id: "D09",
		key: "memory.foodbank.volunteer",
		kind: "episodic",
		content: "Volunteered sorting perishable food boxes at a community food bank every Saturday for six months.",
		confidence: 0.83,
		ageDays: 185,
	},
	{
		id: "D10",
		key: "memory.php.refactor",
		kind: "episodic",
		content: "Refactored a 10,000-line legacy PHP monolith into twenty modular tested classes over three months.",
		confidence: 0.86,
		ageDays: 195,
	},
	{
		id: "D11",
		key: "memory.basic.c64",
		kind: "fact",
		content: "Wrote first computer program in BASIC on a Commodore 64 at age twelve: a number guessing game.",
		confidence: 0.85,
		ageDays: 320,
	},
	{
		id: "D12",
		key: "memory.recursion.hanoi",
		kind: "fact",
		content: "Understood recursion intuitively during a Tower of Hanoi university lecture and built a maze solver.",
		confidence: 0.84,
		ageDays: 275,
	},
	{
		id: "D13",
		key: "memory.prague.train",
		kind: "episodic",
		content: "Spoke with a retired particle accelerator physicist about quantum tunneling on a delayed train from Prague to Vienna.",
		confidence: 0.82,
		ageDays: 225,
	},
	{
		id: "D14",
		key: "memory.dolomites.sunrise",
		kind: "episodic",
		content: "Hiked at 4 AM with a headlamp to watch an orange sunrise over the Dolomites peaks.",
		confidence: 0.82,
		ageDays: 165,
	},
	{
		id: "D15",
		key: "memory.worldcup.2014",
		kind: "episodic",
		content: "Watched Mario Goetze score in extra time during the Germany vs Argentina World Cup final.",
		confidence: 0.81,
		ageDays: 245,
	},
	{
		id: "D16",
		key: "memory.treehouse.oak",
		kind: "episodic",
		content: "Built a backyard oak treehouse with a rope ladder and west-facing window over three weekends.",
		confidence: 0.8,
		ageDays: 310,
	},
	{
		id: "D17",
		key: "memory.swimming.petrovic",
		kind: "episodic",
		content: "Learned back-floating and swimming from instructor Mrs. Petrovic at the municipal pool.",
		confidence: 0.79,
		ageDays: 330,
	},
	{
		id: "D18",
		key: "memory.japanese.study",
		kind: "fact",
		content: "Studied Hiragana, Katakana, and Kanji to hold basic Japanese conversations when visiting Tokyo.",
		confidence: 0.86,
		ageDays: 145,
	},
	{
		id: "D19",
		key: "memory.demo.hotspot",
		kind: "episodic",
		content: "Recovered a live client dashboard demo during a building power cut by tethering to a phone hotspot.",
		confidence: 0.84,
		ageDays: 190,
	},
	{
		id: "D20",
		key: "memory.belgrade.cs",
		kind: "fact",
		content: "Enrolled in the Computer Science program at the University of Belgrade.",
		confidence: 0.88,
		ageDays: 350,
	},
];

/** Recent unconsolidated mid-term episodes (before sleep-cycle runs). */
export const BENCHMARK_EPISODES: readonly BenchmarkEpisode[] = [
	{
		id: "EP01",
		conversationId: "conv-live-1",
		runId: "run-live-1",
		userMessage: "Heads up: the temporary staging database password rotation ticket is SEC-4819 and the maintenance window starts Thursday 02:00 UTC.",
		assistantSummary: "Noted ticket SEC-4819 and the Thursday 02:00 UTC maintenance window for staging rotation.",
		ageMinutes: 12,
	},
	{
		id: "EP02",
		conversationId: "conv-live-1",
		runId: "run-live-2",
		userMessage: "Please use port 9443 for the local Webhook Inspector proxy during today's integration test.",
		assistantSummary: "Will configure the local Webhook Inspector proxy on port 9443.",
		ageMinutes: 8,
	},
	{
		id: "EP03",
		conversationId: "conv-live-2",
		runId: "run-live-3",
		userMessage: "We just decided to rename the internal codename for the search rewrite from Project Lantern to Project Beacon.",
		assistantSummary: "Recorded the rename from Project Lantern to Project Beacon.",
		ageMinutes: 5,
	},
	{
		id: "EP04",
		conversationId: "conv-live-2",
		runId: "run-live-4",
		userMessage: "Remember that Sarah Kim is covering on-call triage while Marcus Chen is at KubeCon.",
		assistantSummary: "Noted Sarah Kim as temporary on-call triage lead while Marcus Chen is at KubeCon.",
		ageMinutes: 2,
	},
];

export const BENCHMARK_QUERIES: readonly BenchmarkQuery[] = [
	// ── Category 1: Single-hop exact & technical identifiers (12 queries) ──────────────────────
	{ id: "SH01", category: "single_hop", query: "What is my primary code editor?", expectedIds: ["M01"] },
	{ id: "SH02", category: "single_hop", query: "Which AWS region hosts our production Kubernetes cluster?", expectedIds: ["M02"] },
	{ id: "SH03", category: "single_hop", query: "maxRetryCount gateway client config", expectedIds: ["M03"] },
	{ id: "SH04", category: "single_hop", query: "pg_bouncer connection pooler mode", expectedIds: ["M04"] },
	{ id: "SH05", category: "single_hop", query: "What version of pi-durable does Helios pin?", expectedIds: ["M05"] },
	{ id: "SH06", category: "single_hop", query: "Friday production schema migration rule", expectedIds: ["M06"] },
	{ id: "SH07", category: "single_hop", query: "How do I run TypeScript unit tests before committing?", expectedIds: ["M07"] },
	{ id: "SH08", category: "single_hop", query: "What specialty coffee beans do I like?", expectedIds: ["M08"] },
	{ id: "SH09", category: "single_hop", query: "Pi-hole VLAN home network", expectedIds: ["M11"] },
	{ id: "SH10", category: "single_hop", query: "Where are OAuth2 signing credentials stored?", expectedIds: ["M17"] },
	{ id: "SH11", category: "single_hop", query: "How to recover commits after git reset --hard?", expectedIds: ["M15"] },
	{ id: "SH12", category: "single_hop", query: "What timezone am I in?", expectedIds: ["M16"] },

	// ── Category 2: Semantic / Paraphrase Gap (zero or minimal lexical overlap) (12 queries) ───
	{ id: "SG01", category: "semantic_gap", query: "text editing software setup", expectedIds: ["M01"] },
	{ id: "SG02", category: "semantic_gap", query: "containerization tool that solved environment drift", expectedIds: ["M10"] },
	{ id: "SG03", category: "semantic_gap", query: "culinary trick for making noodle sauce cling better", expectedIds: ["M09"] },
	{ id: "SG04", category: "semantic_gap", query: "preventing permanent data loss from hardware failure", expectedIds: ["M12"] },
	{ id: "SG05", category: "semantic_gap", query: "secret acidic addition in family slow-cooked meat dish", expectedIds: ["M14"] },
	{ id: "SG06", category: "semantic_gap", query: "blocking advertisements across household wireless devices", expectedIds: ["M11"] },
	{ id: "SG07", category: "semantic_gap", query: "end-of-week database change freeze policy", expectedIds: ["M06"] },
	{ id: "SG08", category: "semantic_gap", query: "ensuring automated build pipelines are actually executing checks", expectedIds: ["M18"] },
	{ id: "SG09", category: "semantic_gap", query: "how should responses be formatted and written for me", expectedIds: ["M13"] },
	{ id: "SG10", category: "semantic_gap", query: "morning caffeinated beverage flavor profile", expectedIds: ["M08"] },
	{ id: "SG11", category: "semantic_gap", query: "credential rotation schedule and secret management vault", expectedIds: ["M17"] },
	{ id: "SG12", category: "semantic_gap", query: "undoing destructive branch history wipe in version control", expectedIds: ["M15"] },

	// ── Category 3: Multi-Hop Relational Reasoning (2-hop entity bridges) (15 queries) ─────────
	{
		id: "MHQ01",
		category: "multi_hop",
		query: "What database does Project Atlas use for telemetry analytics?",
		expectedIds: ["MH01B"],
		requiredAllIds: ["MH01A", "MH01B"],
	},
	{
		id: "MHQ02",
		category: "multi_hop",
		query: "Which database does Kestrel API connect to?",
		expectedIds: ["MH02B"],
		requiredAllIds: ["MH02A", "MH02B"],
	},
	{
		id: "MHQ03",
		category: "multi_hop",
		query: "What programming language and RPC framework does the Billing Service use?",
		expectedIds: ["MH03B"],
		requiredAllIds: ["MH03A", "MH03B"],
	},
	{
		id: "MHQ04",
		category: "multi_hop",
		query: "Which clinic does our cat Luna go to?",
		expectedIds: ["MH04B"],
		requiredAllIds: ["MH04A", "MH04B"],
	},
	{
		id: "MHQ05",
		category: "multi_hop",
		query: "What flour and feeding schedule do I use for my sourdough starter?",
		expectedIds: ["MH05B"],
		requiredAllIds: ["MH05A", "MH05B"],
	},
	{
		id: "MHQ06",
		category: "multi_hop",
		query: "What CSS framework styles the Solaris Mobile web portal?",
		expectedIds: ["MH06B"],
		requiredAllIds: ["MH06A", "MH06B"],
	},
	{
		id: "MHQ07",
		category: "multi_hop",
		query: "Which compliance frameworks are covered by our annual audit sign-off?",
		expectedIds: ["MH07B"],
		requiredAllIds: ["MH07A", "MH07B"],
	},
	{
		id: "MHQ08",
		category: "multi_hop",
		query: "What serialization schema validation does Meridian Pipeline enforce?",
		expectedIds: ["MH08B"],
		requiredAllIds: ["MH08A", "MH08B"],
	},
	{
		id: "MHQ09",
		category: "multi_hop",
		query: "What testing step does my first job mentor insist on before refactoring monoliths?",
		expectedIds: ["MH09B"],
		requiredAllIds: ["MH09A", "MH09B"],
	},
	{
		id: "MHQ10",
		category: "multi_hop",
		query: "What client authentication does Nordic Gateway require for external traffic?",
		expectedIds: ["MH10B"],
		requiredAllIds: ["MH10A", "MH10B"],
	},
	{
		id: "MHQ11",
		category: "multi_hop",
		query: "What stemming algorithm does Zephyr Search use?",
		expectedIds: ["MH11B"],
		requiredAllIds: ["MH11A", "MH11B"],
	},
	{
		id: "MHQ12",
		category: "multi_hop",
		query: "What neural network model powered our 2017 hackathon ASL translator?",
		expectedIds: ["MH12B"],
		requiredAllIds: ["MH12A", "MH12B"],
	},
	{
		id: "MHQ13",
		category: "multi_hop",
		query: "What is the weekly credit budget cap for Vertex Analytics queries?",
		expectedIds: ["MH13B"],
		requiredAllIds: ["MH13A", "MH13B"],
	},
	{
		id: "MHQ14",
		category: "multi_hop",
		query: "Where did the friend who planned our Japan itinerary book ramen seats?",
		expectedIds: ["MH14B"],
		requiredAllIds: ["MH14A", "MH14B"],
	},
	{
		id: "MHQ15",
		category: "multi_hop",
		query: "How are Severity-1 checkout outages paged and escalated?",
		expectedIds: ["MH15B"],
		requiredAllIds: ["MH15A", "MH15B"],
	},

	// ── Category 4: Knowledge Updates & Contradictions (10 queries) ────────────────────────────
	{ id: "KUQ01", category: "knowledge_update", query: "What JavaScript package manager do I currently use?", expectedIds: ["KU01_V3"], forbiddenIds: ["KU01_V1", "KU01_V2"] },
	{ id: "KUQ02", category: "knowledge_update", query: "What database do I use for new side projects?", expectedIds: ["KU02_V2"], forbiddenIds: ["KU02_V1"] },
	{ id: "KUQ03", category: "knowledge_update", query: "Which cloud region hosts our EU customer staging environment?", expectedIds: ["KU03_V2"], forbiddenIds: ["KU03_V1"] },
	{ id: "KUQ04", category: "knowledge_update", query: "What tool formats and lints our TypeScript repositories?", expectedIds: ["KU04_V2"], forbiddenIds: ["KU04_V1"] },
	{ id: "KUQ05", category: "knowledge_update", query: "What is our default main reasoning LLM model?", expectedIds: ["KU05_V2"], forbiddenIds: ["KU05_V1"] },
	{ id: "KUQ06", category: "knowledge_update", query: "What time is the daily engineering standup?", expectedIds: ["KU06_V2"], forbiddenIds: ["KU06_V1"] },
	{ id: "KUQ07", category: "knowledge_update", query: "Where do I store personal architecture notes?", expectedIds: ["KU07_V2"], forbiddenIds: ["KU07_V1"] },
	{ id: "KUQ08", category: "knowledge_update", query: "What tool handles our incident response coordination?", expectedIds: ["KU08_V2"], forbiddenIds: ["KU08_V1"] },
	{ id: "KUQ09", category: "knowledge_update", query: "What ingress proxy handles edge HTTP and gRPC routing?", expectedIds: ["KU09_V2"], forbiddenIds: ["KU09_V1"] },
	{ id: "KUQ10", category: "knowledge_update", query: "What interactive terminal shell do I use now?", expectedIds: ["KU10_V3"], forbiddenIds: ["KU10_V1", "KU10_V2"] },

	// ── Category 5: Temporal Window Reasoning (10 queries) ─────────────────────────────────────
	{ id: "TQ01", category: "temporal_window", query: "What astronomical event did I watch in August 1999?", expectedIds: ["TM01"] },
	{ id: "TQ02", category: "temporal_window", query: "What outage happened in October 2019?", expectedIds: ["TM02"] },
	{ id: "TQ03", category: "temporal_window", query: "What major architecture migration took place in 2020?", expectedIds: ["TM03"] },
	{ id: "TQ04", category: "temporal_window", query: "What trail did I hike in July 2015?", expectedIds: ["TM04"] },
	{ id: "TQ05", category: "temporal_window", query: "What race did I run in April 2016?", expectedIds: ["TM05"] },
	{ id: "TQ06", category: "temporal_window", query: "Where did I move in March 2018?", expectedIds: ["TM06"] },
	{ id: "TQ07", category: "temporal_window", query: "What shipped on 2024-03-15?", expectedIds: ["TM08"] },
	{ id: "TQ08", category: "temporal_window", query: "What happened in August 2003?", expectedIds: ["TM09"] },
	{ id: "TQ09", category: "temporal_window", query: "What security test finished in January 2025?", expectedIds: ["TM11"] },
	{ id: "TQ10", category: "temporal_window", query: "What outdoor events happened between 2015 and 2016?", expectedIds: ["TM04", "TM05"] },

	// ── Category 6: Historical / Superseded Audit (5 queries) ──────────────────────────────────
	{ id: "HQ01", category: "historical_audit", query: "What JavaScript package manager did I use before bun?", expectedIds: ["KU01_V2", "KU01_V1"] },
	{ id: "HQ02", category: "historical_audit", query: "What cloud region did EU staging originally run in before Frankfurt?", expectedIds: ["KU03_V1"] },
	{ id: "HQ03", category: "historical_audit", query: "What note app did I previously use before migrating to Obsidian?", expectedIds: ["KU07_V1"] },
	{ id: "HQ04", category: "historical_audit", query: "What incident coordination tool did we use prior to Incident.io?", expectedIds: ["KU08_V1"] },
	{ id: "HQ05", category: "historical_audit", query: "What shell did I use before switching to Fish?", expectedIds: ["KU10_V2", "KU10_V1"] },

	// ── Category 7: Abstention / Unanswerable Negatives (10 queries) ───────────────────────────
	{ id: "AB01", category: "abstention", query: "What brand of luxury yacht do I own?", expectedIds: [], expectAbstain: true },
	{ id: "AB02", category: "abstention", query: "Which airline loyalty program do I have diamond status with?", expectedIds: [], expectAbstain: true },
	{ id: "AB03", category: "abstention", query: "What is my favorite scuba diving reef in Australia?", expectedIds: [], expectAbstain: true },
	{ id: "AB04", category: "abstention", query: "Which electric motorcycle model is parked in my garage?", expectedIds: [], expectAbstain: true },
	{ id: "AB05", category: "abstention", query: "What piano concerto am I rehearsing for Carnegie Hall?", expectedIds: [], expectAbstain: true },
	{ id: "AB06", category: "abstention", query: "What is the serial number of my espresso grinder?", expectedIds: [], expectAbstain: true },
	{ id: "AB07", category: "abstention", query: "Which cryptocurrency hardware wallet seed phrase do I use?", expectedIds: [], expectAbstain: true },
	{ id: "AB08", category: "abstention", query: "What is my blood type and allergy medication dosage?", expectedIds: [], expectAbstain: true },
	{ id: "AB09", category: "abstention", query: "Which university did my sister graduate from in Brazil?", expectedIds: [], expectAbstain: true },
	{ id: "AB10", category: "abstention", query: "What golf handicap did I record at St Andrews?", expectedIds: [], expectAbstain: true },

	// ── Category 8: Recent Unconsolidated Session Episodes (4 queries) ─────────────────────────
	{ id: "SR01", category: "session_recent", query: "What is the ticket ID for the staging database rotation?", expectedIds: [], expectedEpisodeIds: ["EP01"] },
	{ id: "SR02", category: "session_recent", query: "Which port should the local Webhook Inspector proxy use?", expectedIds: [], expectedEpisodeIds: ["EP02"] },
	{ id: "SR03", category: "session_recent", query: "What is the new codename for Project Lantern?", expectedIds: [], expectedEpisodeIds: ["EP03"] },
	{ id: "SR04", category: "session_recent", query: "Who is covering on-call triage while Marcus Chen is at KubeCon?", expectedIds: [], expectedEpisodeIds: ["EP04"] },
];
