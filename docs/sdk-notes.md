# Installed Pi Durable SDK Notes (Verified)

Generated in Step 0 from the installed `node_modules/@earendil-works/*` declarations and documentation.

## Installed Exact Versions

| Package | Exact Version | Purpose |
| --- | --- | --- |
| `@earendil-works/pi-durable` | `1.6.0` | Durable agent harness, conversations, documents, tasks, signals, SQLite storage |
| `@earendil-works/pi-ai` | `1.6.0` | Multi-provider LLM streaming & `fauxProvider` deterministic test provider |
| `@earendil-works/chord` | `1.6.0` | `Context` (`BACKGROUND_CONTEXT`, `withAbortSignal`), `WatchHandle`, `Op` deltas, `NodeSqliteDatabase` |
| `@earendil-works/pi-schema` | `1.6.0` | Typed JSON Schema builders (`obj`, `str`, `num`, `int`, `bool`, `arr`, `opt`, `oneOf`, `lit`, `Static`) |
| `hono` | `4.13.21` | Lightweight HTTP + native SSE (`hono/streaming` `streamSSE`) |
| `@hono/node-server` | `2.1.11` | Node HTTP adapter for Hono |
| `typescript` | `7.0.2` | Strict ESM TypeScript compiler |
| `tsx` | `4.23.15` | TypeScript runtime for dev/start |
| `vitest` | `5.1.1` | Test runner |
| `@types/node` | `26.6.7` | Node.js type definitions (incl. `node:sqlite`) |

Node.js built-in `node:sqlite` (`DatabaseSync`) is used directly (verified FTS5 + WAL support, zero native build scripts required with `--ignore-scripts`).

---

## Verified API Signatures

### 1. Harness & Node SQLite Storage

```ts
import { BACKGROUND_CONTEXT, withAbortSignal, type Context } from "@earendil-works/chord/context";
import { Harness, MemoryStorage, createRegistry } from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";

// File-backed SQLite storage with table prefix so app tables stay separate:
const storage = await openNodeSqliteStorage(dbPath, {
  prefix: "pi_",
  busyTimeoutMs: 5000,
});

const harness = await Harness.open(
  storage,
  {
    models,
    registry,
    settings,
    now,
    setTimer,
    onWake,        // (wake: Wake) => void — portable to DO/Rivet alarms
    onReport,      // (report: Report) => void
    conversationCreated,
  },
  context,
);

harness.resume();
const wake = harness.wake(); // { active: boolean; at?: number }
await harness.close(context);
```

### 2. Registry & Extensions

```ts
import {
  createRegistry,
  defineExtension,
  defineTool,
  hook,
  hookTask,
  section,
  wrapSection,
  wrapTool,
  GenerationTask,
  ToolTask,
  CompactionTask,
} from "@earendil-works/pi-durable";

const registry = createRegistry();
registry.install(ext);    // installs or replaces by ext.name in place
registry.uninstall(ext);  // removes by ext.name
```

Extension fields:
- `name: string`
- `tools?: readonly ToolRegistration[]`
- `sections?: readonly PromptSection[]`
- `hooks?: readonly HookRegistration[]`
- `wraps?: readonly Wrap[]`
- `tasks?: readonly AnyTask[]`

Prompt sections:
- `section(key, (input: PromptInput, context: Context) => string | undefined | Promise<string | undefined>, { tag?: boolean })`
- `PromptInput`: `{ conversationId, agent, env, shown, read: DocumentReader }`
- Sections are diffed by `planSystemEntries()` against `replaySections(view.messages)`. When every section renders the identical text across turns, zero new `pi.system` entries are emitted and the leading system prompt prefix remains byte-identical (preserving provider KV caches).

### 3. Tools, `@earendil-works/pi-schema`, `replay: "safe"`, and Parked Waits

```ts
import { obj, str, num, int, bool, arr, opt, oneOf, lit, type Static } from "@earendil-works/pi-schema";

const tool = defineTool<typeof Params, DetailsType, CheckpointType>({
  name: "my_tool",
  description: "...",
  parameters: Params,
  replay: "safe", // "safe" reruns after mid-execution crash; default "unsafe" settles with "interrupted" error
  structuredOutputSchema: OutputSchema, // optional
  outputLimits: { maxBytes, maxLines, retain: "head" | "tail" },
  execute: async (args, api, context) => {
    // Normal return: { output: [{ type: "text", text }], isError?, details?, diagnostics?, control? }
    // OR park without holding a process:
    //   { sleep: { checkpoint?, signals?: [MySignal.key(key)], until?: api.now() + ms } }
    //   { wait: { checkpoint?, on: [taskId], policy: "allSettled" | "failFast" | "any" } }
  },
  cancel: async (args, api, context) => {
    // Runs if aborted while parked
  },
});
```

Key `ToolExecutionApi` members:
- `taskId`, `conversationId`, `callId`, `wake`, `checkpoint`, `registry`, `models`, `env`
- `now()`, `output(chunk)`, `retainedOutput()`, `diagnostic(diag)`, `details(val, ctx)`
- `snapshot(token, ...args)`, `commit(tx => ..., ctx, options?)`
- `memo(name, ctx)` / `memo(name, candidate, ctx)`
- `readSignal(token, key, ctx)`
- `createTask(...)`, `getTask(...)`, `waitForTask(...)`, `conversation(id, ctx)`, `executeTool(...)`

Tool recovery semantics (verified in `dist/harness/tool.js`):
1. `call` phase runs `beforeTool` hooks, then commits `{ phase: "execute", arguments, replay }` before running `execute()`.
2. If the process stops/crashes mid-`execute()`, recovery resumes at `{ phase: "execute" }`:
   - `replay === "safe"` -> reruns `execute()` cleanly.
   - `replay === "unsafe"` -> does NOT rerun; settles with `isError: true` and diagnostic code `"interrupted"`.
3. A parked tool (`{ sleep: { signals, until, checkpoint } }`) persists `{ phase: "parked" }` in SQLite, survives close/reopen without running tool code, and re-invokes `execute()` with `api.wake` and `api.checkpoint` once a matching signal or deadline arrives.

### 4. Hooks

- `hook(ToolTask, { beforeTool, afterTool })`
  - `beforeTool(call, api, context)` -> `{ arguments?, block?, sleep?: { until?, signals? } } | undefined`
  - `afterTool(call, result, api, context)` -> replacement `ToolExecutionResult | undefined`
- `hook(GenerationTask, { beforeRequest, beforeProviderRequest, afterResponse, onYield, afterTools, afterRun })`
  - `afterRun(run: RunSummary, api, context)` -> `{ tasks: [hookTask(MyTask, input)] } | undefined`
- `hook(CompactionTask, { beforeCompact })`

### 5. Documents & Atomic Commits

```ts
import { defineDoc } from "@earendil-works/pi-durable";

const MyDoc = defineDoc<MyState>({
  kind: "app.my-doc",
  version: 1,
  scope: "conversation", // "session" | "conversation" | "task"
  history: "rewindable", // "latest" | "rewindable" (conversation scope only)
  fork: "asOf",          // "initial" | "current" | "asOf" (asOf requires rewindable)
  initial: () => ({ ... }),
  checkpointWhen: (_value, _ops, info) => info.deltasSinceBase >= 32,
});

await conversation.commit(async (tx) => {
  const draft = await tx.doc(MyDoc, conversation.id);
  draft.field = "updated";
}, context);

const snap = await harness.snapshot(MyDoc, conversation.id, context);
```

### 6. Custom Durable Tasks & Signals

```ts
import { defineSignal, defineTask } from "@earendil-works/pi-durable";

const ApprovalSignal = defineSignal({
  name: "app.approval.decision",
  version: 1,
  schema: obj({ approved: bool(), reviewer: opt(str()), note: opt(str()) }),
});

const MyTask = defineTask<Input, Checkpoint, Result>({
  name: "app.my-task",
  version: 1,
  initial: (input) => ({ phase: "step1" }),
  phases: {
    step1: async (task, runtime, context) => {
      await runtime.commit(() => ({ status: "terminal", outcome: { status: "completed", result } }), context);
    },
  },
  abort: async (_task, runtime, context) => {
    await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), context);
  },
});

// Deliver signal externally:
const res = await harness.signal(taskId, ApprovalSignal, "key", { approved: true }, context);
// res: "delivered" | "duplicate" | "invalid" | "closed" | "not_found"
```

### 7. Conversations, Submissions, Forks, Reset, Compaction, Watch

- `harness.createConversation({ ownership: { kind: "ownerless" }, agent, init }, context)`
- `conversation.fork(atEntryId, { ownership: { kind: "ownerless" }, agent, init }, context)`
- `conversation.submit({ type: "input", content, requestId, whenBusy }, context)` -> `Submission` (`id`, `status(ctx)`, `wait(ctx)`, `abort(ctx)`)
  - Retrying with the same `requestId` on the same conversation returns the existing `Submission`.
- `harness.submission(id, context)` reacquires a submission across restarts.
- `conversation.reset(handoff, context)` appends a `pi.reset` entry (`head: "self"`).
- `conversation.compact(instructions, context)` starts a durable `CompactionTask` returning `TaskId<CompactionResult>`.
- `conversation.watch(context, window?)` -> `WatchHandle<ConversationView>`
  - `watch.value`: `{ conversation, entries, before?, docs }` (`docs["pi.live"]`, `docs["pi.usage"]`, `docs["pi.agent"]`, `docs["pi.inbox"]`, `docs["pi.provider"]`)
  - `watch.start(async (call, ctx) => { if (call.kind === "value") { const { value, ops, retired } = call; ... } })`
  - Chord `Op` union: `["r", val] | ["s", path, val] | ["d", path] | ["a", path, str] | ["t", path, n] | ["p", path, idx, del, ins] | ["m", path, perm]`

### 8. `@earendil-works/pi-ai` Models & Faux Provider

```ts
import { createModels } from "@earendil-works/pi-ai/models";
import { anthropicProvider } from "@earendil-works/pi-ai/providers/anthropic";
import { openaiProvider } from "@earendil-works/pi-ai/providers/openai";
import { googleProvider } from "@earendil-works/pi-ai/providers/google";
import { fauxProvider, fauxAssistantMessage, fauxToolCall, fauxText } from "@earendil-works/pi-ai/providers/faux";
```

- `fauxProvider({ provider, models, tokenSize: { min: 4, max: 4 } })` provides deterministic streaming, simulated prompt caching (`cacheRead`/`cacheWrite` keyed by `sessionId`), and response factories `(context: TranscriptContext, options, state, model) => AssistantMessage`.
- `models.completeSimple(model, context, options)` supports one-shot fast-model JSON completions (used behind our Pi-free `FastModelClient` adapter for memory consolidation).
