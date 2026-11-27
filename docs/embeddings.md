# Embeddings: server, backends and memory calibration

Helios memory search fuses five lanes: calibrated BM25 lexical, **semantic vectors**, entity-graph PPR,
temporal anchors and recent episodes. Without an embedding model the vector lane is off, and paraphrased
recall ("text editing software setup" vs "Neovim with lazy.nvim") drops to whatever the words share.
The embedding server in `src/embeddings/` provides that model.

```
 harness (SqliteMemoryService)
   └─ HttpEmbeddingProvider ── POST /v1/embeddings {input, input_type: query|document, dimensions}
                                  │
                    Helios embedding server (npm run embed:server, 127.0.0.1:18765)
                       EmbeddingRouter: task prompts · Matryoshka · LRU cache · circuit breaker · self-test
                          ├─ onnx    EmbeddingGemma 2 in-process (Transformers.js / ONNX Runtime; CPU, DirectML)
                          ├─ npu     FastFlowLM on the AMD Ryzen AI NPU (OpenAI-compatible, port 52625)
                          ├─ openai  any OpenAI-compatible API (OpenAI, OpenRouter, Lemonade, Ollama, TEI, vLLM)
                          └─ gemini  Google Gemini batchEmbedContents (taskType RETRIEVAL_QUERY/DOCUMENT)
                       RerankService (POST /v1/rerank): bge-reranker-v2-m3 in-process, or Jina/Cohere/Voyage
```

## Quick start

```bash
# 1. Local EmbeddingGemma 2 (downloads ~200 MB on first start, then runs offline)
npm run embed:server

# 2. Point the harness at it
EMBEDDINGS_URL=http://127.0.0.1:18765/v1 npm run dev
```

Other profiles:

```bash
npm run embed:server:npu                            # FastFlowLM on the NPU, auto-starts `flm serve`
npm run embed:server:cloud                          # OpenAI (OPENAI_API_KEY)
EMBED_PROFILE=gemini GEMINI_API_KEY=... npm run embed:server
EMBED_PROFILE=openai EMBED_OPENAI_URL=https://openrouter.ai/api/v1 \
  EMBED_OPENAI_MODEL=<model> OPENROUTER_API_KEY=... npm run embed:server
EMBED_ONNX_DEVICE=dml npm run embed:server          # EmbeddingGemma 2 on the Radeon iGPU via DirectML
EMBED_DIMENSIONS=256 npm run embed:server           # Matryoshka: 3x smaller vectors, ~1% MTEB loss
```

The harness can also skip the server and call a cloud endpoint directly:
`EMBEDDINGS_URL=https://api.openai.com/v1 EMBEDDINGS_MODEL=text-embedding-3-small` (key from `OPENAI_API_KEY`).
The server is still recommended, because it adds prompts, caching, fallback and the self-test.

## API

| Route | Description |
| --- | --- |
| `GET /health` | `{status, space, model, dimensions, backends[]}`; 200 when any backend is usable, else 503. Public. |
| `GET /v1/models` | One entry: `id` = `<space>@<dims>`, `space`, `dimensions`, `matryoshka_dimensions`, `calibration`, `backends`. |
| `POST /v1/embeddings` | OpenAI body (`input`, `dimensions`, `encoding_format` float/base64) plus `input_type`: `query` \| `document` (aliases `search_query`, `search_document`, `passage`). Default `document`. |

`/v1/*` requires `Authorization: Bearer <token>` when `EMBED_API_KEYS` is set. The server refuses to bind a
non-loopback host without keys unless `EMBED_ALLOW_UNAUTHENTICATED=true`. Limits: 256 inputs, 16,000 chars
per input (truncated) and a 4 MiB body (`EMBED_MAX_*`).

## Embedding spaces: why fallback is restricted

Vectors are only comparable when they come from the same model. Each backend therefore has an **embedding space**,
and the router only falls back between backends of one space (configuration fails fast otherwise). The harness
stores vectors under `<space>@<dims>` and rejects responses from a server that has switched space, falling back to
lexical search instead of mixing incomparable vectors. Changing model or dimensions re-embeds lazily, in the sleep
cycle and on search. `EMBED_SPACE` lets you declare that two model ids share a space, for example the same weights
served by two runtimes.

## Calibration and the quality self-test

Embedding models differ wildly in their *absolute* cosine ranges. Unrelated sentences score ~0.1 with MiniLM but
~0.54 with EmbeddingGemma 2, where true matches score ~0.77. A fixed similarity threshold is therefore wrong for at
least one of them.

At startup the server embeds a neutral corpus: 20 general-knowledge paraphrase pairs and 10 distractors, sharing no
content with the benchmark or user data (`src/embeddings/self-test.ts`). From it, it measures:

- `floor`: the 95th percentile of unrelated query→document cosines
- `ceiling`: the median cosine of matching pairs
- `separation` (matches above `floor`) and `top1` (a query's own document ranks first)

The measured `{floor, ceiling}` is published in `/v1/models` and the harness maps cosine onto relevance over that
band. A backend with `separation` or `top1` below 0.8 is **quarantined**: it serves no requests and `/health` reports
it. Run the same measurement any time with `npx tsx benchmarks/calibrate-embeddings.ts`.

On top of absolute calibration, memory search scores each vector hit by its **query-relative salience**: the z-score
of its cosine against that query's own similarity distribution over the store, with a smoothstep from 1.5σ to 3σ.
It applies only to stores of 20+ memories. A genuine match stands out from everything else the query touches; an
unanswerable query produces no outlier. This stops near-noise vector hits from seeding the entity graph and
flooding results.

## AMD Ryzen AI NPU status (measured on Ryzen AI 9 HX 370, XDNA2, FLM v1.0.11)

- The `npu` backend works end-to-end. FastFlowLM serves `embed-gemma:300m` on the NPU through
  `flm serve <llm> --embed 1`, and `EMBED_NPU_AUTOSTART=true` spawns it (with the smallest chat model, `qwen3:0.6b`).
- **EmbeddingGemma 2 is not yet compiled for XDNA by FastFlowLM.** The NPU currently runs EmbeddingGemma **1** (300M).
- **The current FLM `embed-gemma:300m` vectors fail the quality self-test:** separation 0.15, top-1 0.15, and the
  median cosine is 0.81 for unrelated pairs vs 0.82 for matches. They are also unrelated to Google's reference
  export (cosine ≈ 0 for identical input), which is why that model has its own space
  (`fastflowlm/embed-gemma-300m`). The server quarantines it automatically, so it cannot degrade memory quality.
  Throughput was also lower than the CPU path (32 texts: ~9.4 s on the NPU vs ~0.4 s with EmbeddingGemma 2 q4 on CPU).
- When FastFlowLM ships a working EmbeddingGemma 2 NPU build, set `EMBED_NPU_MODEL` to its tag (add `EMBED_SPACE`
  only if its vectors match the ONNX export). The self-test validates it before any request is served.
- ONNX Runtime's VitisAI execution provider (the other route onto the NPU) is not shipped in `onnxruntime-node`,
  so in-process ONNX runs on CPU or, with DirectML, the Radeon iGPU.

## Reranker (`POST /v1/rerank`)

The server also hosts a cross-encoder reranker (`EMBED_RERANK=onnx`, default for the local profiles:
`onnx-community/bge-reranker-v2-m3-ONNX`, q8, ~20 ms per pair on CPU). The API is Cohere/Jina-compatible:
`{query, documents, top_n}` returns `{results: [{index, relevance_score}]}`. The `http` backend proxies cloud rerankers
(Jina `https://api.jina.ai/v1`, Cohere `https://api.cohere.com/v2`, Voyage `https://api.voyageai.com/v1`). The
harness can also call those directly with `RERANK_URL` + `RERANK_MODEL`. When `EMBEDDINGS_URL` points to a Helios
server that serves a reranker, the harness picks it up automatically (`RERANK_ENABLED=false` opts out).

**How memory search uses it: as positive evidence only.** We measured several CPU cross-encoders (bge-reranker-v2-m3
at q8 and fp16, Qwen3-Reranker-0.6B, gte-reranker-modernbert, jina-turbo) on neutral pairs and on the dev set:

| | reranker score |
| --- | --- |
| keyword-overlapping answers (single-hop, updates, dates) | mostly 0.6–0.999 |
| keyword-free paraphrased answers (semantic gap) | 0.000–0.045 |
| second hop of a 2-hop chain (does not mention the query entity) | ~0.001 |
| same-topic non-answers ("serial number of my espresso grinder") | ~0.000 |

A high score reliably means "this answers the query"; a low score means nothing, because true paraphrases score as low
as non-answers. Using the reranker as a gate would destroy semantic and multi-hop recall without fixing abstention.
So the top 20 fused candidates (`RERANK_CANDIDATES`) are reranked. A confident score lifts a candidate by noisy-OR,
ramping from the model's measured non-answer level (the self-test `answer_threshold`, 0.078 for bge-v2-m3) to 0.5. A
low score never demotes or removes anything. Recall and abstention stay exactly as fused retrieval had them; only the
order among plausible candidates improves.

The reranker self-test checks what that role needs: non-answers (near-misses and unrelated text) must stay low and
typical answers must clear them. A reranker that scores near-misses like answers (e.g. gte-modernbert at q8) is
quarantined.

**Abstention is not solved by retrieval.** Neither bi-encoders nor CPU cross-encoders can tell "on topic but doesn't
contain the asked detail" from "a paraphrased answer". Deciding that needs a reader: an LLM answerability check over
the retrieved context.

## Benchmark

`npm run bench:memory` uses the embedding server when one is reachable on `127.0.0.1:18765`. Alternatively use
`--in-process <profile>` (no server) or `--lexical-only`; add `--rerank` for the reranked engine.

| Dataset | Command | Purpose |
| --- | --- | --- |
| core (78 queries) | `npm run bench:memory` | development set: tuning happens here |
| held-out (48 queries) | `npm run bench:memory -- --dataset heldout` | report-only generalisation check (new memories; abstention = topical near-misses) |
| LoCoMo (1,535 scored QA, 10 conversations) | `npm run bench:public -- --dataset locomo` | public long-conversation benchmark; turn-level evidence recall |
| LongMemEval_S (500 Q) | `npm run bench:public -- --dataset longmemeval --limit N` | public chat-history benchmark; session-level recall + real abstention questions |

The public datasets are downloaded on first use into `data/benchmarks/` (gitignored) and verified by SHA-256; they
are never committed. `--gate` exits non-zero on regressions against `benchmarks/thresholds*.json`; CI runs the
lexical gates for core and held-out on every push. Measured results are in the README.
