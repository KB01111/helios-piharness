# Helios-PIHarness Architecture & `awesome-pi` Integration Guide

## Architectural Philosophy: `Agent = Model + Durable Harness`

Helios-PIHarness is a **standalone, multi-tenant, durable backend agent harness** built on `@earendil-works/pi-durable` (`1.6.0`). Following modern agent harness architecture best practices:

1. **Minimalist Core + Composable In-Process Extensions:** Keep the core prompt compact and deterministic; layer capabilities (guardrails, budgets, context pruning, telemetry, memory learning loops, deferred tool search, MCP, codemode, answer verification) as isolated Pi Durable extensions in explicit precedence order.
2. **Strict KV-Cache Prefix Stability:** Leading system instructions, frozen Core Memory, and the frozen progressive-disclosure `SKILL.md` index are committed at conversation start into rewindable `ConversationMetaDoc` and never mutated mid-session. Dynamic state (`PlanDoc`, `ConversationModeDoc`, `PendingApprovalsDoc`) stays out of the system prompt.
3. **Deterministic Lifecycle Hooks Over Prompt Engineering:** Safety, tenant isolation, read-only Plan Mode, three-tier `allow`/`ask`/`deny` approval gates, secret redaction, and budget caps are enforced deterministically in `beforeTool`, `afterTool`, `beforeRequest`, and `onYield` hooks — never left to model willpower.
4. **Bounded Delegation (Builder-Verifier Split):** Subagents run as durable **task-owned child conversations** with isolated context windows, strict depth caps, permission narrowing (never widening, never inheriting `admin:*` or `*`), read-only `researcher`/`reviewer` profiles, and automatic abort cascades.
5. **Zero Pi Leakage Across the Port Boundary:** External callers and the Hono HTTP/SSE server talk strictly to `HarnessPort` using plain JSON and opaque string IDs.

---

## Extension & Hook Execution Order

Extensions are installed in `bootstrapHarness()` (`src/index.ts`) in strict precedence order:

| Order | Extension | Hooks / Tools | Responsibility |
| --- | --- | --- | --- |
| 1 | `core-prompt` | `sections: core_instructions, core_memory` | Byte-identical static prompt + session-start frozen Core Memory |
| 2 | `guard-shaping` | `ToolTask: beforeTool, afterTool` | Tenant-spoof defense → Read-only Plan Mode → RBAC → Three-tier `allow`/`ask`/`deny` durable approval gate + audit log → Secret redaction → 6KB artifact offload |
| 3 | `budget` | `GenerationTask: beforeRequest, afterResponse` | Per-conversation & per-tenant daily token/USD caps + spend ledger |
| 4 | `context-pruner` | `GenerationTask: beforeRequest`, `CompactionTask: beforeCompact` | KV-cache-safe stale/duplicate tool output pruning above threshold + Plan/Artifact/Approval-aware compaction summaries |
| 5 | `capture-telemetry` | `ToolTask: afterTool`, `GenerationTask: afterRun` | Loop 1 passive episode capture, deterministic friction signals, Skill Dojo metrics |
| 6 | `sleep-cycle` | `SleepCycleTask` | Loop 2 ACE memory consolidation, Skill Forge n-gram mining, Skill Dojo disabling |
| 7 | `skills-loader` | `section: skills_index`, `skill_read` | Progressive-disclosure `SKILL.md` catalog |
| 8 | `tool-search` | `tool_search` | Native BM25 discovery & on-demand loading of `deferred: true` tools |
| 9 | `mcp-adapter` *(when configured)* | `mcp_list_servers`, `mcp_discover`, dynamic `mcp:<server>` | Lazy MCP JSON-RPC 2.0 discovery registering external tools as `deferred: true` |
| 10 | `codemode` | `codemode` | QuickJS/WASM sandbox whose nested `tools.*` calls traverse every hook above |
| 11 | `answer-verifier` | `GenerationTask: onYield` | Durable fast-model answer verification against tool results & unfinished plan steps |

Toggleable ToolPacks: `memory`, `artifact`, `example-domain`, `planning`, `subagents`, and `file-search` *(registered when `XERJ_URL`, `FILE_SEARCH_WORKSPACE_ROOT`, or `bootstrapHarness({ fileSearch })` is configured; see [`docs/file-search.md`](file-search.md))*.

---

## `awesome-pi` Feature Evaluation Matrix

Evaluated against [`bubbleptr/awesome-pi`](https://github.com/bubbleptr/awesome-pi/blob/main/README.en.md) for a headless, multi-tenant, durable backend harness:

| `awesome-pi` Category | Reference Packages | Status in Helios-PIHarness | Implementation / Rationale |
| --- | --- | --- | --- |
| **Security & Permission** | `@gotgenes/pi-permission-system`, `pi-verdict`, `filter-output`, `cc-safety-net` | **Integrated** | Three-tier `allow`/`ask`/`deny` policy with durable `beforeTool` `ToolGateApprovalSignal` parking + SQLite `permission_audit_log` (`src/security/permission-policy.ts`), plus deterministic tool-output secret redaction (`src/security/secret-redactor.ts`) running before artifact storage and LLM visibility. |
| **Plan Mode** | `@narumitw/pi-plan-mode` | **Integrated** | Runtime-enforced `ConversationModeDoc` (`"plan"` vs `"execute"`, rewindable & fork-inherited) blocking mutating tools in `beforeTool`, exposed via `setConversationMode()` and `POST /conversations/:id/mode`. |
| **Subagents & Code Review** | `pi-subagents`, `pi-durable-subagents`, `@quintinshaw/pi-dynamic-workflows`, `@juicesharp/rpiv-advisor` | **Integrated** | `subagents` ToolPack (`src/tools/subagent.tools.ts`): `subagent_delegate`, `subagent_parallel` (task-owned durable child conversations, `SubmissionSettled` sleep parks, depth cap, permission narrowing, read-only `researcher`/`reviewer` profiles) and `advisor_consult` for fast second opinions. |
| **MCP Adapter** | `pi-mcp-adapter`, `antiproton` | **Integrated** | Zero-dep JSON-RPC 2.0 Streamable HTTP + pluggable transport `McpRegistry` (`src/mcp/mcp-client.ts`) + `mcp-adapter.ext.ts` registering external tools lazily as `deferred: true` for BM25 `tool_search`, keeping credentials strictly out of model context and routing calls through `guard-shaping`. |
| **Context Management** | `pi-context-prune`, `pi-context-kit`, `pi-protokoll`, `pi-cache-graph`, `context-mode` | **Integrated** | `context-pruner.ext.ts`: KV-cache-safe pre-request pruning of stale/duplicate tool outputs (preserving `art_*` recall handles), structured `beforeCompact` preservation of `PlanDoc`, mode, approvals & artifacts, plus `cacheHitRatio` on `UsageBucketView`. |
| **Side Questions (`/btw`)** | `pi-btw`, `@spinejit/pi-btw-cached` | **Integrated** | `askSideQuestion()` & `POST /conversations/:id/btw`: read-only, tool-free side queries reusing the conversation's `sessionId` and leading system prompt prefix without writing to the durable transcript. |
| **Persistent Memory** | `pi-hermes-memory`, `pi-memory`, `pi-mate-companion` | **Already Built** | Entity-graph PPR, bi-temporal retrieval, calibrated hybrid fusion, SQLite FTS5 + optional embeddings, ACE consolidation, Skill Forge & Skill Dojo (`src/memory/`, `src/learning/`). Secret scanning adopted into `guard-shaping`. |
| **Programmable Runtime** | `pi-fabric` | **Already Built** | Native `@earendil-works/pi-durable/codemode` QuickJS/WASM sandbox (`src/extensions/codemode.ext.ts`). |
| **Task Management** | `rpiv-todo`, `pi-tasks` | **Already Built** | Durable `PlanDoc` + `planning` ToolPack (`plan_set`, `plan_update`, `plan_view`) integrated with `answer-verifier`. |
| **Retry & Observability** | `pi-retry`, `pi-langfuse` | **Already Built** | Durable generation retry + fast-model retry/fallback/circuit-breaker (`src/runtime/resilience.ts`) and typed `pi-telemetry` spans + JSONL exporter (`src/runtime/trace-export.ts`). |
| **UI Enhancement / Themes / Editor Integrations / Local Browser & Worktrees** | `pi-statusline`, `pi-catppuccin`, `pi-acp`, `pi-chrome`, `pi-worktree`, `pi-sandbox` | **Excluded by Design** | Helios-PIHarness is a headless multi-tenant backend runtime. Terminal UI, IDE plugins, and host-machine browser/worktree execution belong to client frontends or host-provided `ExecutionEnv` / domain tool packs. |
| **Vendor-Locked SaaS Wrappers** | `pi-jev-*`, `pi-venice`, `pi-memory-honcho`, `pi-telegram` | **Excluded by Design** | Helios stays model-agnostic (`@earendil-works/pi-ai`), storage-agnostic, and transport-agnostic (`HarnessPort`). |
