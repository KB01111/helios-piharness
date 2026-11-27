<img src="assets/icon.svg" width="64" height="64" alt="">

# Helios-PIHarness

Standalone TypeScript backend agent harness and durable runtime powered by **`@earendil-works/pi-durable`**.

Designed to run independently over HTTP/SSE today or embed directly into a larger LLM application via `HarnessPort` with **zero leakage of Pi internals outside `src/runtime/`, `src/harness/`, `src/extensions/`, and `src/tools/`**.

## Key Features

- **Durable Execution & Crash Recovery:** SQLite-backed Pi Durable `Harness` (`node:sqlite`, WAL mode, `pi_*` table prefix) with a single-writer lock guard, idempotent `requestId` submissions, `replay: "safe"` vs `replay: "unsafe"` tool semantics, and parked `{ sleep: { signals, until } }` human-approval waits that survive restarts.
- **KV-Cache Prefix Stability:** Static system instructions + session-start frozen Core Memory and progressive-disclosure `SKILL.md` index stored in rewindable `ConversationMetaDoc`, keeping the leading system prompt byte-identical across turns.
- **Guardrails & Output Shaping:** Closure/document-bound `TenantContext` isolation (forbidding `tenantId`/`userId` LLM tool parameters), three-tier tool permission policy (`allow` / `ask` / `deny`) with durable `beforeTool` approval parking and SQLite audit log, runtime-enforced read-only **Plan Mode** (`POST /conversations/:id/mode`), automatic tool-output secret/credential redaction (before artifacts and LLM context), and 6KB tool output capping with artifact offloading (`artifact_read`).
- **Durable Subagents & Advisor:** Task-owned child conversations (`subagent_delegate`, `subagent_parallel`) with `researcher` / `reviewer` (read-only) and `worker` profiles, strict nesting depth caps, permission narrowing (never widening, never inheriting admin privileges), crash recovery via `SubmissionSettled` sleep parks, plus `advisor_consult` for fast second opinions.
- **Lazy MCP Adapter & Deferred Tool Discovery:** Zero-dependency JSON-RPC 2.0 Streamable HTTP + pluggable transport MCP client (`mcp_list_servers`, `mcp_discover`) integrated with Pi Durable's native BM25 `tool_search` so external MCP tools load on demand as `deferred: true` without bloating initial prompts.
- **Context Pruning, Structured Compaction & Cached `/btw`:** KV-cache-safe pre-request stale/duplicate tool output pruning (`beforeRequest`), Plan/Artifact-aware compaction summaries (`beforeCompact`), ephemeral `/btw` side questions (`POST /conversations/:id/btw`) reusing the conversation prompt-cache prefix without mutating the transcript, and `cacheHitRatio` usage telemetry.
- **Self-Improving Memory & Learning Loops:**
  - **Loop 1 (`capture-telemetry.ext.ts`):** Passive durable mid-term episode capture, deterministic pre-LLM friction signals, and Skill Dojo metrics.
  - **Loop 2 (`sleep-cycle.ext.ts`):** Durable multi-phase `SleepCycleTask` running ACE memory consolidation (`ADD`/`UPDATE`/`SUPERSEDE` tombstones + promotion gates), Skill Forge deterministic 3–7 tool n-gram mining, and Skill Dojo stale (`>30d`) / degraded skill disabling.
  - **Loop 3 (`sim-log.ts`, `evals/`):** Append-only Self-Improvement Memory (`sim_log`) storing wins and failed edits, plus trace-replay evaluations with conversation forking (`evolve` vs `held_out` splits & Pareto metrics).
- **Transport-Agnostic API:** Thin Hono HTTP + native SSE server (`src/transport/http-server.ts`) wrapping `HarnessPort`.

## Quick Start

```bash
cp .env.example .env
npm install --ignore-scripts
npm run typecheck
npm test
npm run dev
```

By default, when no provider API keys are configured (or `USE_MOCK_MODELS=true`), the server runs with the deterministic offline `TestModel` on `http://127.0.0.1:4100`.

## Scripts

- `npm run dev` — Start HTTP/SSE server in watch mode (`tsx`)
- `npm run start` — Start HTTP/SSE server
- `npm run build` — Compile ESM library and declarations to `dist/`
- `npm run typecheck` — Strict TypeScript type-check (`src/`, `tests/`, `benchmarks/`)
- `npm test` — Run all Vitest integration suites
- `npm run embed:server` — OpenAI-compatible embedding server (EmbeddingGemma 2 by default; `embed:server:npu`, `embed:server:cloud`)
- `npm run bench:memory` — Memory quality + scalability benchmark (uses the embedding server when it is running)
- `npm run bench:memory:gate` / `bench:memory:gate:heldout` — Same, failing on regressions against `benchmarks/thresholds*.json` (CI runs both lexical-only)
- `npm run bench:public -- --dataset locomo|longmemeval` — Public long-term-memory benchmarks (downloaded on first use, never committed)

## Embeddings & Memory Benchmark

Hybrid memory retrieval (lexical FTS5 + entity-graph PPR + temporal + **semantic vectors**) needs an embedding model. Helios ships its own embedding server with four backends behind one OpenAI-compatible API (details: [`docs/embeddings.md`](docs/embeddings.md)):

| Profile | Backend | Model | Hardware |
| --- | --- | --- | --- |
| `gemma2` *(default)* | in-process ONNX (Transformers.js) | EmbeddingGemma 2 (`onnx-community/embeddinggemma-2-ONNX`, q4) | CPU, or `EMBED_ONNX_DEVICE=dml` for the Radeon iGPU |
| `npu` | FastFlowLM (`flm serve … --embed 1`) | `embed-gemma:300m` | AMD Ryzen AI NPU (XDNA/XDNA2) |
| `openai` | any OpenAI-compatible API | `text-embedding-3-small` (OpenAI, OpenRouter, Lemonade, Ollama, TEI, vLLM…) | cloud / remote |
| `gemini` | Google Gemini `batchEmbedContents` | `gemini-embedding-001` | cloud |

```bash
npm run embed:server                       # EmbeddingGemma 2 on 127.0.0.1:18765
EMBEDDINGS_URL=http://127.0.0.1:18765/v1 npm run dev
```

The server applies EmbeddingGemma's asymmetric query/document prompts, Matryoshka truncation (`EMBED_DIMENSIONS`), caching, circuit breaking and same-space fallback, and runs a **quality self-test at startup** that measures the model's cosine calibration (published to the harness) and quarantines any backend whose vectors cannot separate related from unrelated text.

It also serves a **cross-encoder reranker** (`POST /v1/rerank`, bge-reranker-v2-m3 in-process, or Jina/Cohere/Voyage), which the harness discovers automatically. Memory search uses it as *positive evidence only*: confident "this answers the query" scores lift candidates, low scores never demote. Measured CPU cross-encoders score keyword-free paraphrases as low as non-answers, so gating on them would destroy recall ([details](docs/embeddings.md#reranker-post-v1rerank)).

**Benchmarks** (deterministic; tuning happens only on the core set):

*Core dev set* (`npm run bench:memory`, 78 queries / 8 categories):

| Engine | R@1 | R@5 | MRR | Semantic R@5 | Multi-hop both@5 | Update fresh@1 | Historical R@5 | Abstain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helios (lexical, no model) | 66.2% | 88.2% | 0.770 | 33.3% | 100% | 90% | 100% | 100% |
| **helios (hybrid, EmbeddingGemma 2)** | **85.3%** | **100%** | **0.926** | **100%** | 100% | 100% | 100% | 60% |
| helios (hybrid + rerank) | 85.3% | 100% | 0.926 | 100% | 100% | 100% | 100% | 60% |
| graphiti-ref (same embeddings) | 69.1% | 94.1% | 0.806 | 100% | 100% | 90% | 100% | 0% |
| hipporag-ref (same embeddings) | 47.1% | 86.8% | 0.657 | 100% | 100% | 50% | 0% | 0% |

*Held-out set* (`-- --dataset heldout`, 48 new queries; report-only, abstention = topical near-misses):

| Engine | R@1 | R@5 | MRR | Abstain |
| --- | ---: | ---: | ---: | ---: |
| helios (lexical) | 50.0% | 65.6% | 0.578 | 43.8% |
| helios (hybrid) | 78.1% | 96.9% | 0.875 | 25.0% |
| **helios (hybrid + rerank)** | **81.3%** | **96.9%** | **0.891** | 25.0% |
| graphiti-ref | 71.9% | 93.8% | 0.818 | 0% |
| hipporag-ref | 62.5% | 87.5% | 0.732 | 0% |

*LoCoMo* (`npm run bench:public -- --dataset locomo`, 1,535 scored questions, turn-level evidence recall):

| Engine | R@1 | R@5 | R@10 | MRR@10 | Multi-hop R@1 | Search p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helios (lexical) | 37.6% | 62.7% | 70.6% | 0.482 | 21.6% | 3 ms |
| helios (hybrid) | 34.7% | 70.9% | 80.8% | 0.499 | 22.3% | 36 ms |
| **helios (hybrid + rerank)** | **48.2%** | **76.9%** | **83.5%** | **0.604** | **35.5%** | 750 ms |

*LongMemEval_S* (`-- --dataset longmemeval --limit 20`, stratified subset): answerable session recall R@5 100% for lexical and hybrid (MRR 0.95 / 0.97); abstention 0/4.

**Known limits:** abstention generalises poorly (60% dev vs 25% held-out vs 0/4 LongMemEval). Rejecting "on topic, but the asked detail isn't stored" needs a reader (an LLM answerability check), not a better retriever. The reranker adds ~0.7 s per search on CPU; lower `RERANK_CANDIDATES` or set `RERANK_ENABLED=false` for latency-critical paths.

## Configuration Reference

All settings are read from environment variables (see `.env.example`). Unknown or invalid numeric values fall back to the default.

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `4100` | HTTP listen port |
| `HOST` | `127.0.0.1` | HTTP bind address |
| `LOG_LEVEL` | `info` | Log threshold: `debug`, `info`, `warn`, `error`. Unknown values fall back to `info` |
| `DATA_DIR` | `./data` | Base directory for SQLite files and artifacts |
| `PI_DB_PATH` | `$DATA_DIR/pi-durable.sqlite` | Pi Durable SQLite file, or `:memory:`. Guarded by a single-writer lock |
| `APP_DB_PATH` | `$DATA_DIR/app-memory.sqlite` | App memory and telemetry SQLite file, or `:memory:` |
| `ARTIFACTS_DIR` | `$DATA_DIR/artifacts` | Storage for offloaded oversized tool output |
| `SKILLS_DIR` | `./skills` | Root of the `SKILL.md` catalog |
| `MAIN_MODEL` | `anthropic:claude-sonnet-4-5` (mock: `test:main`) | Main model as `provider:modelId` or `provider/modelId` |
| `FAST_MODEL` | `anthropic:claude-haiku-4-5` (mock: `test:fast`) | Fast model for lightweight tasks |
| `USE_MOCK_MODELS` | `true` when no provider key is set, else `false` | Deterministic offline `TestModel` |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | empty | Provider credentials. Never written to logs. OpenRouter models are addressed as `MAIN_MODEL=openrouter:<id>` (free ones end in `:free`) |
| `MAX_TOOL_OUTPUT_BYTES` | `6144` | Tool output size cap before artifact offload |
| `SLEEP_CYCLE_CRON_INTERVAL_MS` | `300000` | Sleep-cycle sweep interval in milliseconds |
| `SLEEP_CYCLE_SCHEDULER_ENABLED` | `true` | Enables or disables the sleep-cycle scheduler |
| `STALE_SKILL_DAYS` | `30` | Age after which a skill is considered stale by Skill Dojo |

### Security / Budgets / Resilience

| Variable | Default | Purpose |
| --- | --- | --- |
| `HELIOS_API_KEYS` | _(unset)_ | JSON map `token -> {tenantId,userId,permissions?}` for bearer auth. Without auth configured every route except `/health` and `/ready` returns 401 |
| `HELIOS_AUTH_TRUSTED_HEADERS` | `false` | **Dev only.** Trust `x-tenant-id`/`x-user-id`/`x-permissions` headers |
| `HELIOS_MAX_BODY_BYTES` | `1048576` | Request body limit (413 beyond) |
| `HELIOS_RATE_LIMIT_PER_MINUTE` | `600` | Per-tenant rate limit, `0` disables |
| `HELIOS_SSE_KEEPALIVE_MS` | `15000` | SSE ping interval, `0` disables |
| `HELIOS_MAX_STREAMS_PER_TENANT` / `HELIOS_MAX_STREAMS` | `20` / `1000` | Concurrent SSE stream caps (429 beyond) |
| `HELIOS_AUTH_FAIL_LIMIT_PER_MINUTE` | `60` | Failed-auth attempts per remote address before 429 |
| `HELIOS_LOCK_STALE_MS` / `HELIOS_LOCK_FORCE_RECLAIM` | `600000` / `false` | When a writer lock held by another host may be reclaimed |
| `HARNESS_STREAM_MAX_QUEUE` | `1000` | Per-stream event buffer; overflow closes the stream with reason `overflow` |
| `MAX_CONVERSATION_TOKENS` / `MAX_CONVERSATION_COST_USD` | `0` | Per-conversation budget (0 = unlimited) |
| `MAX_TENANT_DAILY_TOKENS` / `MAX_TENANT_DAILY_COST_USD` | `0` | Per-tenant daily budget (0 = unlimited) |
| `MAX_FAST_MODEL_DAILY_TOKENS` / `MAX_FAST_MODEL_DAILY_COST_USD` | `0` | Process-wide UTC-day cap on background fast-model calls (consolidation, skill mining). Once reached, calls fail fast with `BUDGET_EXCEEDED` instead of reaching the provider (0 = unlimited) |
| `DEFAULT_MAX_TURNS` | `24` | Turn cap for conversations that do not set one |
| `MODEL_TIMEOUT_MS`, `MODEL_MAX_RETRIES`, `MODEL_RETRY_BASE_DELAY_MS` | `120000`, `3`, `2000` | Main-model timeout / durable retry |
| `FAST_MODEL_TIMEOUT_MS`, `FAST_MODEL_MAX_RETRIES`, `FAST_MODEL_FALLBACKS` | `30000`, `2`, _(unset)_ | Fast-model timeout, retry and fallback chain |
| `CIRCUIT_BREAKER_FAILURES`, `CIRCUIT_BREAKER_COOLDOWN_MS` | `5`, `30000` | Fast-model circuit breaker |
| `ALLOW_MOCK_MODELS` | `false` | Allow mock models when `NODE_ENV=production` (otherwise startup fails) |
| `SLEEP_CYCLE_CRON_INTERVAL_MS` | `300000` | Periodic sleep-cycle sweep; `0` disables |
| `SLEEP_CYCLE_SCHEDULER_ENABLED` | `true` | Toggle the sleep-cycle scheduler |

### Permissions, Subagents, MCP & Context

| Variable | Default | Purpose |
| --- | --- | --- |
| `SECRET_REDACTION_ENABLED` | `true` | Redact credential-shaped values in tool output before artifacts and model context. Approval summaries and the audit log are always redacted |
| `TOOL_APPROVAL_REQUIRED` | _(empty)_ | Comma-separated tool names or globs (`domain_mutate_record,mcp_prod__*`) that need durable human approval |
| `TOOL_DENIED` | _(empty)_ | Comma-separated tool names or globs blocked unconditionally |
| `TOOL_APPROVAL_TIMEOUT_MS` | `86400000` | How long an `ask`-tier call stays parked before it expires as blocked |
| `SUBAGENTS_ENABLED` | `true` | Register `subagent_delegate`, `subagent_parallel` and `advisor_consult` |
| `SUBAGENT_MAX_DEPTH` / `SUBAGENT_MAX_TURNS` / `SUBAGENT_MAX_PARALLEL` / `SUBAGENT_TIMEOUT_MS` | `2` / `8` / `4` / `120000` | Nesting depth, tool rounds per child, children per `subagent_parallel` call, deadline per delegation |
| `SUBAGENT_READONLY_MODEL` / `ADVISOR_MODEL` | _(main / fast model)_ | Optional model overrides for read-only subagents and `advisor_consult` |
| `TOOL_SEARCH_ENABLED` | `true` | BM25 `tool_search` over `deferred: true` tools |
| `MCP_SERVERS` | _(empty)_ | JSON array `[{ "name", "url", "headers"?, "readOnly"?, "timeoutMs"? }]`. Servers connect lazily; responses are capped at 4 MiB and redirects are refused |
| `CONTEXT_PRUNE_ENABLED` / `CONTEXT_PRUNE_THRESHOLD_CHARS` / `CONTEXT_PRUNE_KEEP_RECENT_TOOLS` | `true` / `32000` / `6` | KV-cache-safe pruning of stale tool output |
| `STRUCTURED_COMPACTION_ENABLED` | `true` | Keep plan, mode, approvals and artifact handles through compaction |
| `MODEL_PROVIDER_MAX_RETRIES` | _(SDK default)_ | In-request SDK retries inside one model call (separate from the durable `MODEL_MAX_RETRIES`) |
| `EVAL_CASE_TIMEOUT_MS` / `EVAL_ONLY` | _(unset)_ | Bound a live eval run / run only the listed case ids |

### Data retention

Append-only operational tables would otherwise grow forever. A sweeper (`src/runtime/retention.ts`) runs at startup and then every `RETENTION_INTERVAL_MS`, deleting in bounded batches so write locks stay short. A window of `0` keeps that data forever. Long-term memory, episodes, skills and the Pi Durable transcript are never touched by it.

| Variable | Default | Prunes |
| --- | --- | --- |
| `RETENTION_ENABLED` | `true` | Master switch (never runs against `:memory:` databases) |
| `RETENTION_INTERVAL_MS` | `21600000` | Sweep interval; `0` sweeps once at startup only |
| `RETENTION_PERMISSION_AUDIT_DAYS` | `365` | `permission_audit_log` |
| `RETENTION_TOOL_INVOCATION_DAYS` | `90` | `tool_invocations` (Skill Dojo metrics; keep this above `STALE_SKILL_DAYS`) |
| `RETENTION_RUN_TELEMETRY_DAYS` | `90` | `run_telemetry` (Skill Forge input) |
| `RETENTION_BUDGET_EVENT_DAYS` | `30` | `budget_events` |
| `RETENTION_BUDGET_DAILY_DAYS` | `400` | `budget_tenant_daily` history (the current day, which enforces limits, is always kept) |
| `RETENTION_ARTIFACT_DAYS` | `0` | Offloaded tool-output artifacts and their files. Opt-in: transcripts hold `art_*` recall handles that would dangle |

Forgetting a memory (`forgetMemory`) also scrubs the raw episodes that evidenced it (verbatim user message, summary and tool arguments), not just the memory row.

### Embeddings (hybrid memory retrieval)

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDINGS_URL` | _(unset: lexical-only)_ | Helios embedding server (`http://127.0.0.1:18765/v1`) or any OpenAI-compatible endpoint |
| `EMBEDDINGS_API_KEY` | _(unset)_ | Bearer token (falls back to `OPENAI_API_KEY` for `api.openai.com`) |
| `EMBEDDINGS_MODEL` | _(from server)_ | Upstream model; required for non-Helios endpoints |
| `EMBEDDINGS_DIMENSIONS` | _(native)_ | Matryoshka output size requested from the endpoint |
| `EMBEDDINGS_TIMEOUT_MS` / `EMBEDDINGS_REQUIRED` | `15000` / `false` | Request timeout; abort startup when unreachable instead of running lexical-only |
| `RERANK_ENABLED` | `true` | Use a cross-encoder reranker for memory search when one is available (auto-discovered from a Helios `EMBEDDINGS_URL`) |
| `RERANK_URL` / `RERANK_MODEL` / `RERANK_API_KEY` | _(unset)_ | Cloud rerank API (Jina, Cohere, Voyage) instead of the embedding server's reranker |
| `RERANK_CANDIDATES` | `20` | Fused candidates reranked per search (latency ~20 ms per candidate on CPU) |

Embedding-server variables (`EMBED_*`) are documented in [`docs/embeddings.md`](docs/embeddings.md) and `.env.example`.

### HTTP API

Every route except `/health` and `/ready` needs a bearer token. Other tenants' resources answer 404. Responses carry `x-content-type-options: nosniff`, `cache-control: no-store` and `referrer-policy: no-referrer`.

| Route | Purpose |
| --- | --- |
| `POST /conversations` | Create (`model`, `instructions`, `enabledToolPacks`, `maxTurns`, `mode`) |
| `GET /conversations/:id` | Conversation info and full transcript |
| `POST /conversations/:id/messages` | Submit (`content`, `requestId`, `whenBusy`, `wait`) |
| `GET /conversations/:id/stream` | SSE; resume with `Last-Event-ID` or `?afterEntryId=` |
| `POST /conversations/:id/{fork,abort,reset,compact,sleep-cycle,mode,btw}` | Lifecycle, memory (`memory:write`), plan/execute mode, side question |
| `GET /approvals` | Pending tool-gate approvals |
| `POST /signals` | Deliver a decision to a parked task. The recorded `reviewer` is always the authenticated user |
| `GET /tool-packs`, `POST /tool-packs/:name` | List; toggle (`admin:tool-packs`) |
| `GET /mcp/servers` | MCP connection state. `lastError` is shown only with `admin:usage` |
| `GET /usage` | Process-wide usage (`admin:usage`) |
| `GET /audit/permissions` | Tenant permission audit (`admin:audit`) |
| `/improvements/*` | Gated self-improvement (`admin:improvements`), see below |

### Codemode

`@earendil-works/pi-durable/codemode` is installed by default (`src/extensions/codemode.ext.ts`). The model writes JavaScript that calls tools as `await tools.<name>(args)` in a QuickJS/WASM sandbox and only the script's output reaches it. Scripts survive restarts by deterministic replay. Every `tools.*` call is a nested tool call, so tenant guards and permissions still apply. `domain_request_approval` is `callers: ["model"]`, so a script cannot bypass human approval, and `models.*` is disabled.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEMODE_ENABLED` | `true` | Install the `codemode` tool |
| `CODEMODE_MEMORY_MB` | `128` | Heap cap per script VM |
| `CODEMODE_TIMEOUT_MS` | `120000` | Wall-clock deadline per script, tool time included |
| `CODEMODE_MAX_WORKERS` | `4` | Process-wide cap on concurrent script workers |
| `CODEMODE_INLINE_BUDGET` | `3000` | Estimated tokens of tool declarations in the prompt section |

### Answer verifier

A quality gate before an answer stands (`src/extensions/verifier.ext.ts`). It is a durable `onYield` task, so the caller only receives an answer after the check and a crash mid-check resumes it. Short, tool-free answers are not checked. Otherwise one fast-model call reviews the answer against the request, the tool results and any unfinished plan steps. It can send the agent back at most `VERIFIER_MAX_REVISIONS` times per run, and only for concrete problems: unsupported claims, ignored tool errors, an unanswered request, or unfinished plan steps. It fails open (missing fast model, budget block, provider error or bad output lets the answer through). The reviewer's issues are screened for injection before being fed back, because it read untrusted tool output. Default: on with real models, off with mock models (`VERIFIER_ENABLED`, `VERIFIER_MAX_REVISIONS`, `VERIFIER_MIN_ANSWER_CHARS`).

### Planning

The `planning` tool pack (`plan_set`, `plan_update`, `plan_view`) gives the agent a durable, bounded working plan (12 steps) stored in a rewindable document, so restarts and forks see the plan as of their entry. A blocked step requires a reason, which pushes the agent to re-plan or explain the blocker. The plan is never injected into the system prompt (that would break the KV-cache prefix); each tool call returns the whole plan instead. The verifier treats unfinished steps as grounds for revision.

### Trace export

Pi Durable emits typed spans (`pi.durable.phase`, `.submit`, `.entry`, ...) into a host-provided telemetry adapter. Set `TRACE_EXPORT_PATH` to append every settled span as one JSON line (`StoredSpan` shape; one previous file kept at `<path>.1` after `TRACE_EXPORT_MAX_BYTES`). Spans carry IDs and control metadata only, never messages, tool arguments or results, and oversized attribute values are dropped. Pass `bootstrapHarness({ telemetry })` to use your own adapter (for example OpenTelemetry) instead.

### Gated self-improvement (Harness²-style propose / probe / compose)

The harness can improve its own operating guidance **without ever changing guardrails, permissions, tools or budgets**. The only editable surface is a *prompt addendum* appended to the static system prompt.

1. **Baseline** – the eval set (`evals/cases/*.json`, `evolve` and `held_out` splits) runs on the current prompt.
2. **Propose** – a proposer (default: the fast model, so fast-model budgets and the circuit breaker apply) reads the failing `evolve` cases, with answers redacted and treated as untrusted data, and drafts up to `maxCandidates` addenda: general ("invariant") and failure-specific ("domain"). Hypotheses already disproven in the SIM log are never retried.
3. **Probe** – each candidate runs the whole eval set in its own isolated harness (in-memory databases, no scheduler) against the same models.
4. **Gate** (`evaluateGate`) – accept only with held-out non-regression, zero safety violations, a big-enough eval set (default 3 held-out cases) and a strict evolve gain in pass rate or tool-error rate. Latency never justifies a change. A lint rejects text that weakens controls, mentions identity or secrets, or contains URLs, markup or code.
5. **Compose** – if several survive, their merge is probed and gated too; the best survivor wins. Every outcome is written to the SIM log.
6. **Human promotion** – the winner is stored as `pending_approval` and changes nothing until an operator approves it. Approval is refused if the set of active improvements changed since the candidate was measured (re-run instead). Active guidance is capped (4000 chars), persisted in `IMPROVEMENTS_DIR`, re-applied at startup, hot-applied to new conversations, and can be rolled back.

Admin routes (permission `admin:improvements`): `GET /improvements`, `POST /improvements/run` (single-flight), `GET /improvements/:id`, `POST /improvements/:id/{approve,reject,rollback}` with an optional `{ "note" }`. Programmatic access: `bootstrapHarness().improvements`.

**Eval set tooling.** `npm run eval` runs the committed eval set and fails on any regression versus `evals/baseline.json` (pass rate, tool errors, safety violations, or a shrunken set; latency is ignored). It runs in CI and in `npm test`. After an intentional change run `npm run eval:baseline` and review the diff. Cases can be built from real conversations with `buildEvalCaseFromTranscript` (emails, keys and phone-like numbers are redacted, and a human must review the file). `npm run eval:live` runs the same cases against real models (needs a provider key and spends tokens; it refuses to run on the mock model, ignores the scripted responses, skips the mock-based baseline, and exits 1 only on safety violations). Verified against real free OpenRouter models (`MAIN_MODEL=openrouter:nvidia/nemotron-3.5-lightning:free`, `OPENROUTER_API_KEY`): free models are slow and rate-limited (429s are retried), so set `EVAL_CASE_TIMEOUT_MS` and `EVAL_ONLY=id1,id2` to bound a run. Use a real model for meaningful probes (`USE_MOCK_MODELS=false`); the scripted cases are deterministic smoke tests.

| Variable | Default | Purpose |
| --- | --- | --- |
| `EVALS_DIR` | `./evals` | Eval set root (`cases/*.json`, `baseline.json`) |
| `IMPROVEMENT_PROBE_SAMPLES` | `1` | Eval-suite passes averaged per probe. Real models are non-deterministic and a single pass over a few cases is mostly noise: use `3` or more with them |
| `IMPROVEMENT_TOOL_ERROR_TOLERANCE` | `0.05` | How far the tool-error *rate* may rise in the gate. It is a ratio, so a prompt that makes the agent call more tools moves it without making anything worse. Pass rate and safety get no slack |
| `IMPROVEMENTS_DIR` | `$DATA_DIR/improvements` | Improvement records and the active list |

Cost limits (`*_COST_USD`) only bite with a priced model (Pi AI's catalog `usage.cost.total`) or a custom `bootstrapHarness({ budget: { estimateCostUsd } })`; the mock provider always reports zero cost, so use the token limits there.

Embeddings are optional: pass `bootstrapHarness({ embeddingProvider })` to re-rank memory search results by cosine similarity on top of FTS. Core memory frozen into a conversation is refreshed on `resetConversation` and compaction (the prompt prefix is rebuilt then anyway) and stays byte-identical otherwise. Manual `POST /conversations/:id/sleep-cycle` is durably single-flight per conversation, even across restarts.

Admin-only routes: `POST /tool-packs/:name` needs `admin:tool-packs`, `GET /usage` needs `admin:usage` (`*` and `admin:*` also work). Mined skill drafts are written to `skills/_proposed/` and only become active after `promoteProposedSkill()`.

## Logging

The server writes one JSON object per line. Each record has `time`, `level`, and `event`, plus context fields such as `url` or `reason`.

- `debug` and `info` go to stdout. `warn` and `error` go to stderr.
- `LOG_LEVEL` filters records below the threshold.
- Errors are serialized as `{ name, message, stack, code? }`. Custom error properties are dropped, and keys that look like secrets (`apiKey`, `*token`, `password`, `authorization`, and similar) are replaced with `[REDACTED]`.
- Lock diagnostics use the same logger with `component: "writer-lock"`.

Shutdown on `SIGINT` or `SIGTERM` stops accepting connections, closes the harness, drops open SSE streams, and exits 0. If shutdown takes longer than 15 seconds, the process logs `shutdown_timed_out` and exits 1. An `unhandledRejection` or `uncaughtException` is logged and triggers the same shutdown with exit code 1.

## Health Endpoints

`createHealthChecker` in `src/health.ts` provides two probes. It is exported from the package index.

- `liveness()` reports that the process is up, along with `version` (from `package.json`) and `uptimeSec`. It does not touch any dependency.
- `readiness()` runs three checks: `harness.getUsage()` must succeed within 2 seconds, the app DB must answer `SELECT 1`, and the Pi DB file must exist (or be `:memory:`). It returns `status: "ready"` or `"not_ready"`, per-check latency, `mockModels`, and `version`. Failed checks report only the error class name.

The HTTP transport should map these to `/health` (liveness) and `/ready` (readiness, HTTP 503 when `not_ready`). Wiring into `src/transport/http-server.ts` is owned by the transport layer.

## Operations: Single-Writer Lock

Each durable SQLite file (`PI_DB_PATH`) is guarded by a lock file at `<PI_DB_PATH>.writer.lock`. It holds JSON `{ pid, hostname, startedAt, acquiredAt }`.

| Lock file state | Behavior |
|-----------------|----------|
| Owner pid is not running on this host | Reclaimed, with a `writer_lock_stale_removed` warning |
| Owner pid is alive on this host | Startup refused (`SingleWriterLockError`) |
| Recorded on a different host | Startup refused, because the pid cannot be verified from here |
| Empty or corrupt, older than 30 seconds | Reclaimed, with a warning |
| Empty or corrupt, younger than 30 seconds | Startup refused, because another writer may be starting |

Stale-lock recovery re-checks the file contents before deleting, so a lock freshly taken by a concurrent process is not removed. `release()` deletes the lock only while it still records this process's pid and acquisition time.

If a crashed harness left a lock whose PID was reused by an unrelated live process, startup is refused. Confirm that no harness is running for that database, then delete `<PI_DB_PATH>.writer.lock`.

See [docs/architecture.md](docs/architecture.md) for the harness architecture, extension ordering, and `awesome-pi` feature integration matrix, and [docs/sdk-notes.md](docs/sdk-notes.md) for verified `@earendil-works/pi-durable` `1.6.0` API notes.
