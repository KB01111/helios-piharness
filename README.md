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

See [docs/sdk-notes.md](docs/sdk-notes.md) for verified `@earendil-works/pi-durable` `1.6.0` API notes.
