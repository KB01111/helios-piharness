/**
 * HELD-OUT memory benchmark: separate memories and queries that were NOT used while tuning retrieval.
 * Report-only: do not iterate on individual failures here (use the core dataset for development), otherwise it
 * stops measuring generalisation.
 *
 * Emphasis on the two hardest categories:
 *  - abstention: mostly *topical near-misses*, where the topic is in memory but the asked-for detail is not
 *    (the failure mode bi-encoders are worst at)
 *  - multi_hop: 2-hop entity bridges where the answer lives in the second memory
 */

import type { BenchmarkEpisode, BenchmarkMemory, BenchmarkQuery } from "./memory-dataset.js";

export const HELDOUT_NOW_MS = Date.UTC(2026, 10, 20, 9, 0, 0);

export const HELDOUT_MEMORIES: readonly BenchmarkMemory[] = [
	// ── Profile / preferences ──
	{ id: "H01", key: "user.keyboard", kind: "preference", content: "Types on a split ergonomic keyboard with Colemak-DH layout and brown switches.", ageDays: 60, evidenceCount: 3 },
	{ id: "H02", key: "user.diet", kind: "preference", content: "Follows a mostly plant-based diet but eats fish twice a week.", ageDays: 90, evidenceCount: 2 },
	{ id: "H03", key: "user.reading", kind: "preference", content: "Reads hard science fiction on an e-ink tablet before sleeping, currently the Expanse series.", ageDays: 20 },
	{ id: "H04", key: "user.commute", kind: "fact", content: "Commutes by folding bicycle and regional train, about 40 minutes each way.", ageDays: 75 },
	{ id: "H05", key: "user.language_learning", kind: "fact", content: "Practising Japanese daily with spaced-repetition flashcards; passed JLPT N4 last spring.", ageDays: 30 },
	{ id: "H06", key: "user.music", kind: "preference", content: "Listens to lo-fi jazz playlists while doing deep work and cannot focus with lyrics.", ageDays: 15 },
	{ id: "H07", key: "user.allergy", kind: "fact", content: "Has a mild pollen allergy that flares up every May.", ageDays: 200 },
	{ id: "H08", key: "user.travel.next", kind: "fact", content: "Booked a two-week rail trip through Norway for early September.", ageDays: 10 },

	// ── Engineering facts ──
	{ id: "H10", key: "infra.ci.runner", kind: "fact", content: "Self-hosted CI runners are ARM64 machines in the Dublin data centre.", ageDays: 50 },
	{ id: "H11", key: "infra.secrets", kind: "fact", content: "Application secrets are injected at deploy time from HashiCorp Vault via short-lived tokens.", ageDays: 70 },
	{ id: "H12", key: "infra.observability", kind: "fact", content: "Traces are exported with OpenTelemetry to a self-hosted Tempo cluster; dashboards live in Grafana.", ageDays: 40 },
	{ id: "H13", key: "code.style.python", kind: "constraint", content: "Python services must pass ruff and mypy --strict before merge.", ageDays: 100 },
	{ id: "H14", key: "code.release", kind: "fact", content: "Mobile apps ship every second Thursday through a staged rollout reaching 100% after 72 hours.", ageDays: 35 },
	{ id: "H15", key: "incident.2025.dns", kind: "episodic", content: "In March 2025 an expired DNS delegation took the marketing site offline for three hours.", ageDays: 600 },

	// ── Multi-hop chains (A = bridge, B = answer) ──
	{ id: "HM01A", key: "service.orion.owner", kind: "fact", content: "The Orion search service is maintained by the Discovery squad.", ageDays: 45 },
	{ id: "HM01B", key: "team.discovery.oncall", kind: "fact", content: "The Discovery squad handles on-call through a weekly rotation led by Amara Okafor.", ageDays: 44 },
	{ id: "HM02A", key: "person.lukas.role", kind: "fact", content: "Lukas Brandt is my running coach.", ageDays: 120 },
	{ id: "HM02B", key: "person.lukas.plan", kind: "fact", content: "Lukas Brandt prescribes three easy runs and one interval session every week.", ageDays: 118 },
	{ id: "HM03A", key: "project.helix.datastore", kind: "fact", content: "Project Helix stores its event log in the Meridian cluster.", ageDays: 55 },
	{ id: "HM03B", key: "infra.meridian.engine", kind: "fact", content: "The Meridian cluster runs ScyllaDB 6 on NVMe-backed bare metal.", ageDays: 53 },
	{ id: "HM04A", key: "home.plants.monstera", kind: "fact", content: "The big monstera in the living room was a gift from my neighbour Ingrid.", ageDays: 300 },
	{ id: "HM04B", key: "person.ingrid.job", kind: "fact", content: "Ingrid works as a landscape architect at a firm in Aarhus.", ageDays: 290 },
	{ id: "HM05A", key: "service.ledger.queue", kind: "fact", content: "The Ledger service publishes settlement events to the Saffron topic.", ageDays: 65 },
	{ id: "HM05B", key: "infra.saffron.retention", kind: "fact", content: "The Saffron topic keeps messages for 14 days with compaction enabled.", ageDays: 64 },
	{ id: "HM06A", key: "family.sister.city", kind: "fact", content: "My sister Noor lives in Lisbon.", ageDays: 400 },
	{ id: "HM06B", key: "place.lisbon.cafe", kind: "preference", content: "Whenever visiting Lisbon I go to a tiny bakery in Alfama for pastéis de nata.", ageDays: 380 },
	{ id: "HM07A", key: "tool.design.owner", kind: "fact", content: "Our design tokens are owned by Priyanka Rao's platform team.", ageDays: 80 },
	{ id: "HM07B", key: "team.priyanka.tooling", kind: "fact", content: "Priyanka Rao's platform team generates tokens with Style Dictionary and publishes them as an npm package.", ageDays: 78 },

	// ── Knowledge updates ──
	{ id: "HK01_V1", key: "user.phone", kind: "fact", content: "Daily phone is a Pixel 7.", ageDays: 500 },
	{ id: "HK01_V2", key: "user.phone", kind: "fact", content: "Replaced the Pixel 7 with a Fairphone 5 for repairability.", ageDays: 25, supersedesId: "HK01_V1" },
	{ id: "HK02_V1", key: "team.standup.tool", kind: "fact", content: "Async standups are posted in Slack threads.", ageDays: 300 },
	{ id: "HK02_V2", key: "team.standup.tool", kind: "fact", content: "Async standups moved from Slack threads to Linear project updates.", ageDays: 12, supersedesId: "HK02_V1" },
	{ id: "HK03_V1", key: "infra.k8s.version", kind: "fact", content: "Production Kubernetes clusters run version 1.29.", ageDays: 250 },
	{ id: "HK03_V2", key: "infra.k8s.version", kind: "fact", content: "Production Kubernetes clusters were upgraded to version 1.33.", ageDays: 8, supersedesId: "HK03_V1" },
];

export const HELDOUT_EPISODES: readonly BenchmarkEpisode[] = [
	{
		id: "HEP01",
		conversationId: "conv_h1",
		runId: "run_h1",
		userMessage: "Can you remind me to renew the TLS certificate for the status page before Friday?",
		assistantSummary: "Noted a reminder to renew the status page TLS certificate before Friday.",
		ageMinutes: 30,
	},
	{
		id: "HEP02",
		conversationId: "conv_h2",
		runId: "run_h2",
		userMessage: "I just decided to switch my personal blog from Hugo to Astro.",
		assistantSummary: "User is migrating the personal blog from Hugo to Astro.",
		ageMinutes: 90,
	},
];

export const HELDOUT_QUERIES: readonly BenchmarkQuery[] = [
	// single-hop (8)
	{ id: "HSH01", category: "single_hop", query: "What keyboard layout do I type with?", expectedIds: ["H01"] },
	{ id: "HSH02", category: "single_hop", query: "Where are the CI runners hosted?", expectedIds: ["H10"] },
	{ id: "HSH03", category: "single_hop", query: "Which tool injects application secrets?", expectedIds: ["H11"] },
	{ id: "HSH04", category: "single_hop", query: "What do Python services need to pass before merge?", expectedIds: ["H13"] },
	{ id: "HSH05", category: "single_hop", query: "How often do the mobile apps ship?", expectedIds: ["H14"] },
	{ id: "HSH06", category: "single_hop", query: "What JLPT level did I pass?", expectedIds: ["H05"] },
	{ id: "HSH07", category: "single_hop", query: "Where am I travelling in September?", expectedIds: ["H08"] },
	{ id: "HSH08", category: "single_hop", query: "Where do traces get exported?", expectedIds: ["H12"] },

	// semantic gap (10)
	{ id: "HSG01", category: "semantic_gap", query: "input device and finger arrangement preferences", expectedIds: ["H01"] },
	{ id: "HSG02", category: "semantic_gap", query: "what I eat and avoid at meals", expectedIds: ["H02"] },
	{ id: "HSG03", category: "semantic_gap", query: "bedtime novels and how I read them", expectedIds: ["H03"] },
	{ id: "HSG04", category: "semantic_gap", query: "how I get to the office", expectedIds: ["H04"] },
	{ id: "HSG05", category: "semantic_gap", query: "background sound that helps me concentrate", expectedIds: ["H06"] },
	{ id: "HSG06", category: "semantic_gap", query: "seasonal health issue in late spring", expectedIds: ["H07"] },
	{ id: "HSG07", category: "semantic_gap", query: "how credentials reach running applications", expectedIds: ["H11"] },
	{ id: "HSG08", category: "semantic_gap", query: "monitoring stack for distributed request timing", expectedIds: ["H12"] },
	{ id: "HSG09", category: "semantic_gap", query: "gradual percentage-based app deployment schedule", expectedIds: ["H14"] },
	{ id: "HSG10", category: "semantic_gap", query: "website outage caused by a lapsed domain record", expectedIds: ["H15"] },

	// multi-hop (7)
	{ id: "HMQ01", category: "multi_hop", query: "Who leads on-call for the Orion search service?", expectedIds: ["HM01B"], requiredAllIds: ["HM01A", "HM01B"] },
	{ id: "HMQ02", category: "multi_hop", query: "What weekly training does my running coach prescribe?", expectedIds: ["HM02B"], requiredAllIds: ["HM02A", "HM02B"] },
	{ id: "HMQ03", category: "multi_hop", query: "Which database engine backs Project Helix's event log?", expectedIds: ["HM03B"], requiredAllIds: ["HM03A", "HM03B"] },
	{ id: "HMQ04", category: "multi_hop", query: "What does the person who gave me the monstera do for a living?", expectedIds: ["HM04B"], requiredAllIds: ["HM04A", "HM04B"] },
	{ id: "HMQ05", category: "multi_hop", query: "How long are Ledger settlement events retained?", expectedIds: ["HM05B"], requiredAllIds: ["HM05A", "HM05B"] },
	{ id: "HMQ06", category: "multi_hop", query: "Where do I get pastries in the city my sister lives in?", expectedIds: ["HM06B"], requiredAllIds: ["HM06A", "HM06B"] },
	{ id: "HMQ07", category: "multi_hop", query: "How are our design tokens generated and published?", expectedIds: ["HM07B"], requiredAllIds: ["HM07A", "HM07B"] },

	// knowledge update (3)
	{ id: "HKU01", category: "knowledge_update", query: "What phone do I use?", expectedIds: ["HK01_V2"], forbiddenIds: ["HK01_V1"] },
	{ id: "HKU02", category: "knowledge_update", query: "Where are async standups posted?", expectedIds: ["HK02_V2"], forbiddenIds: ["HK02_V1"] },
	{ id: "HKU03", category: "knowledge_update", query: "Which Kubernetes version runs in production?", expectedIds: ["HK03_V2"], forbiddenIds: ["HK03_V1"] },

	// historical (2)
	{ id: "HHQ01", category: "historical_audit", query: "What phone did I have before the Fairphone?", expectedIds: ["HK01_V1"] },
	{ id: "HHQ02", category: "historical_audit", query: "Which Kubernetes version did production run before the upgrade?", expectedIds: ["HK03_V1"] },

	// session recent (2)
	{ id: "HSR01", category: "session_recent", query: "status page TLS certificate renewal", expectedIds: [], expectedEpisodeIds: ["HEP01"] },
	{ id: "HSR02", category: "session_recent", query: "blog migration from Hugo to Astro", expectedIds: [], expectedEpisodeIds: ["HEP02"] },

	// abstention (16): 12 topical near-misses + 4 fully unrelated
	{ id: "HAB01", category: "abstention", query: "What colour are the keycaps on my keyboard?", expectedIds: [], expectAbstain: true },
	{ id: "HAB02", category: "abstention", query: "Which fish recipe do I cook on Fridays?", expectedIds: [], expectAbstain: true },
	{ id: "HAB03", category: "abstention", query: "What brand is my e-ink tablet?", expectedIds: [], expectAbstain: true },
	{ id: "HAB04", category: "abstention", query: "What is the frame size of my folding bicycle?", expectedIds: [], expectAbstain: true },
	{ id: "HAB05", category: "abstention", query: "What score did I get on the JLPT N4 listening section?", expectedIds: [], expectAbstain: true },
	{ id: "HAB06", category: "abstention", query: "Which antihistamine do I take for hay fever?", expectedIds: [], expectAbstain: true },
	{ id: "HAB07", category: "abstention", query: "What hotel am I staying at in Bergen?", expectedIds: [], expectAbstain: true },
	{ id: "HAB08", category: "abstention", query: "How many CPU cores do the Dublin CI runners have?", expectedIds: [], expectAbstain: true },
	{ id: "HAB09", category: "abstention", query: "What is the Vault token TTL in minutes?", expectedIds: [], expectAbstain: true },
	{ id: "HAB10", category: "abstention", query: "Who was paged during the March 2025 DNS outage?", expectedIds: [], expectAbstain: true },
	{ id: "HAB11", category: "abstention", query: "What is Amara Okafor's phone number?", expectedIds: [], expectAbstain: true },
	{ id: "HAB12", category: "abstention", query: "What is the name of Ingrid's landscape firm?", expectedIds: [], expectAbstain: true },
	{ id: "HAB13", category: "abstention", query: "Which violin do I play in the orchestra?", expectedIds: [], expectAbstain: true },
	{ id: "HAB14", category: "abstention", query: "What is my favourite ski resort in Colorado?", expectedIds: [], expectAbstain: true },
	{ id: "HAB15", category: "abstention", query: "Which horse did I bet on at the derby?", expectedIds: [], expectAbstain: true },
	{ id: "HAB16", category: "abstention", query: "What tattoo design did I get last year?", expectedIds: [], expectAbstain: true },
];
