# Indexed File & Code Search (XERJ)

Helios-PIHarness provides an optional, engine-agnostic **`FileSearchService`** capability (`src/contracts/app-capabilities.ts`) paired with a tenant-scoped, read-only **`file-search`** ToolPack (`src/tools/file-search.tools.ts`) and a first-class **[XERJ](https://github.com/xerj-org/xerj)** adapter (`src/search/xerj-client.ts`).

## Why XERJ (and Why Native Adapter Instead of Raw MCP)?

1. **AST-aware tree-sitter indexing:** `xerj autoindex <folder>` parses source files with tree-sitter (`symbols`, `defs`, `language`, `ax_path`) alongside Markdown, PDF, DOCX, CSV, JSONL, SQLite, and logs in a single zero-JVM Rust binary, respecting `.gitignore` and `.xerjignore` with incremental content-hashed re-indexing.
2. **Definition-first passages fit under Helios's 6KB output cap:** Every `file_search` and `file_symbol_def` call projects `_source` *without* `body` and requests `fields: ["_passage"]` with XERJ's documented `match_phrase` on `defs` (`boost: 4`). Agents get exact `path:line` hits and compact code snippets in one turn instead of reading whole files and triggering artifact offloads.
3. **Strict multi-tenant isolation:** Exposing a raw `xerj mcp` server lets a model pass arbitrary `index` or `*` wildcards across tenants. Helios's `file-search` ToolPack binds `FileSearchScope` (`tenantId`, `userId`, `workspaceId`) strictly from committed `ConversationMetaDoc` and maps it to a sanitized tenant prefix (`<prefix>-<tenantSlug>-*` by default) in code — zero `tenantId`/`userId` tool arguments, zero cross-tenant escape.
4. **Works everywhere in Helios:** All 4 tools (`file_find`, `file_search`, `file_symbol_def`, `file_index_map`) are `replay: "safe"`, gated by `files:read`, permitted in read-only **Plan Mode** and **`researcher` / `reviewer` subagents**, callable from **`codemode`**, and routed through **`guard-shaping`** (secret redaction before artifacts and LLM context).
5. **Memory stays on `SqliteMemoryService`:** Helios keeps its calibrated hybrid memory engine (EmbeddingGemma 2 + cross-encoder reranker + entity-graph PPR + bi-temporal tombstones + ACE consolidation). XERJ handles indexed file/code/document search.

## Quick Start with XERJ

1. Install and start XERJ (keep `--data-dir` **outside** any folder you autoindex):
   ```sh
   # Linux / macOS: curl -fsSL https://xerj.org/get | sh
   # Windows PowerShell: irm https://xerj.org/get.ps1 | iex
   xerj --insecure --data-dir ~/xerj-data &
   ```
2. Index a tenant workspace (matching `XERJ_INDEX_PREFIX=ax` and `XERJ_SCOPE_MODE=tenant`):
   ```sh
   # For tenantId="acme" (default XERJ_SCOPE_MODE=tenant -> queries `ax-acme-*`)
   xerj autoindex --prefix ax-acme /path/to/acme-repo

   # Or for single-tenant local dev (`XERJ_SCOPE_MODE=shared` -> queries `ax-*`)
   xerj autoindex /path/to/repo
   ```
3. Start Helios pointing at XERJ:
   ```sh
   XERJ_URL=http://127.0.0.1:9200 XERJ_SCOPE_MODE=shared npm run dev
   ```

### Sharing Helios's Local Embedding Server with XERJ

By default, XERJ's `semantic_text` embedder is **lexical feature-hashing**, not neural. For real neural semantic/hybrid file search without running a second model, point XERJ's `--embed-mode proxy` at Helios's own `embed:server` (EmbeddingGemma 2 ONNX/NPU):

```sh
npm run embed:server
xerj --insecure --data-dir ~/xerj-data \
  --embed-mode proxy --embed-url http://127.0.0.1:18765/v1/embeddings &
xerj autoindex /path/to/repo
```

> Note: `--embed-mode` applies at ingest time; re-index after changing embed mode. If an index does not map `body` as `semantic_text`, `XerjFileSearchService` automatically falls back from `semantic`/`hybrid` to definition-first `lexical` search with a clear note.

## Tools (`file-search` ToolPack)

| Tool | Permission | Purpose |
| --- | --- | --- |
| `file_find` | `files:read` | Find files by filename, wildcard pattern, extension, language, or `pathPrefix` (projects metadata only, never `body`) |
| `file_search` | `files:read` | Definition-first code & content search returning compact `path:line` passages (`lexical`, `semantic`, `hybrid`) |
| `file_symbol_def` | `files:read` | Jump directly to function/class/struct/interface/type definitions (`path:line`, deduplicated by location) |
| `file_index_map` | `files:read` | Inspect indexed datasets, doc counts, and languages scoped to the caller's tenant |

## Resilience & Local Fallback

- Set `FILE_SEARCH_WORKSPACE_ROOT=/path/to/repo` alongside `XERJ_URL` to wrap XERJ in `FallbackFileSearchService`: if XERJ is down or its circuit breaker opens (3 failures / 15s cooldown), queries degrade seamlessly to the bounded, symlink-safe `LocalWorkspaceFileSearchService`.
- Set `FILE_SEARCH_DEFERRED=true` to register all 4 tools as `deferred: true` so they add zero initial tool-schema tokens and load on demand via `tool_search`.
- When neither `XERJ_URL` nor `FILE_SEARCH_WORKSPACE_ROOT` (nor programmatic `bootstrapHarness({ fileSearch })`) is supplied, the `file-search` pack is omitted with zero overhead.

## Benchmark

```sh
npm run bench:files
# With live XERJ (after `xerj autoindex .`):
XERJ_URL=http://127.0.0.1:9200 XERJ_SCOPE_MODE=shared npm run bench:files
```

Measured across 14 file-find, symbol-definition, and content-search queries over `helios-piharness` (139 indexed files, XERJ `v1.0.0-rc.105`):

| Engine | R@1 | R@5 | MRR | Avg Latency | Payload vs Top-3 Whole Files | Under 6 KB Cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `local-workspace` | 92.9% | 100.0% | 0.964 | 122.0 ms | 5.3 KB vs 687.5 KB (129.3× smaller) | 14 / 14 |
| **`xerj` (`v1.0.0-rc.105`)** | **92.9%** | **100.0%** | **0.964** | **15.3 ms** | **10.3 KB vs 515.2 KB (50.2× smaller)** | **14 / 14** |
