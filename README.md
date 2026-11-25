<img src="assets/icon.svg" width="64" height="64" alt="">

# Helios-PIHarness

Standalone TypeScript backend agent harness and durable runtime powered by **`@earendil-works/pi-durable`**.

Designed to run independently over HTTP/SSE today or embed directly into a larger LLM application via `HarnessPort` with **zero leakage of Pi internals outside `src/runtime/`, `src/harness/`, `src/extensions/`, and `src/tools/`**.

## Key Features

- **Durable Execution & Crash Recovery:** SQLite-backed Pi Durable `Harness` (`node:sqlite`, WAL mode, `pi_*` table prefix) with a single-writer lock guard, idempotent `requestId` submissions, `replay: "safe"` vs `replay: "unsafe"` tool semantics, and parked `{ sleep: { signals, until } }` human-approval waits that survive restarts.
- **KV-Cache Prefix Stability:** Static system instructions + session-start frozen Core Memory and progressive-disclosure `SKILL.md` index stored in rewindable `ConversationMetaDoc`, keeping the leading system prompt byte-identical across turns.
- **Guardrails & Output Shaping:** Closure/document-bound `TenantContext` isolation (forbidding `tenantId`/`userId` LLM tool parameters), per-tool permission checks, and automatic 6KB tool output capping with artifact offloading (`artifact_read`).
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
- `npm run typecheck` — Strict TypeScript type-check (`src/` and `tests/`)
- `npm test` — Run all Vitest integration suites

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
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` | empty | Provider credentials. Never written to logs |
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

See [docs/sdk-notes.md](docs/sdk-notes.md) for verified `@earendil-works/pi-durable` `1.6.0` API notes.
