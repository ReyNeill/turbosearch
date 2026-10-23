# TurboSearch

Local semantic search for agent-accessible client corpora.

The project is intentionally split into ingestion, storage, embedding, and vector backend layers. Exact vector search works as the baseline. A TurboQuant-inspired compressed backend is included behind the same API so it can be benchmarked and replaced with a faster Rust/Zig backend later if needed.

## Quick Start

```bash
bun install
bun run src/cli.ts ingest docs
bun run src/cli.ts search "TurboQuant semantic search" --backend exact
bun run src/cli.ts search "TurboQuant semantic search" --backend turbo
bun run src/cli.ts search "TurboQuant semantic search" --backend hybrid
```

By default this uses local hash embeddings, which are private and deterministic but not high-quality semantic embeddings. For real semantic search, copy `.env.local.example` to `.env.local`, set `OPENAI_API_KEY`, and ingest with:

```bash
bun run src/cli.ts ingest ./client-data --embedding openai
bun run src/cli.ts search "contract renewal terms" --embedding openai
```

For Silixon's private deployment, run the Apache-2.0
`unsloth/Qwen3-Embedding-4B` checkpoint behind vLLM's OpenAI-compatible API. The
recommended 1,024-dimensional Matryoshka output fits pgvector's regular HNSW
`vector` index while retaining instruction-aware Qwen3 query embeddings:

```bash
TURBOSEARCH_EMBEDDING_PROVIDER=openai-compatible
TURBOSEARCH_EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1
TURBOSEARCH_EMBEDDING_MODEL=unsloth/Qwen3-Embedding-4B
TURBOSEARCH_EMBEDDING_DIMENSIONS=1024
TURBOSEARCH_EMBEDDING_AUTH=none
bun run src/cli.ts ingest ./client-data --embedding openai-compatible
```

Each model family expects its own query and document prompts; choose them with
`TURBOSEARCH_EMBEDDING_PROMPTS` (`--embedding-prompts`). The default, `qwen3`,
embeds documents without a prompt and wraps queries in Qwen's required
`Instruct: ...\nQuery:...` format. Override the task sentence with
`TURBOSEARCH_EMBEDDING_QUERY_INSTRUCTION` for a different corpus. `none` sends
text as is.

The smaller Apache-2.0 `google/embeddinggemma-2` (270M parameters for text,
768 dimensions, Matryoshka 512/256/128, 8K context) runs on llama.cpp's
`llama-server` (b11836 or newer) on a laptop CPU or GPU. Its `embeddinggemma`
prompts are `task: search result | query: ...` for queries and
`title: ... | text: ...` for documents; org-brain entries pass their title.

```bash
llama-server --model embeddinggemma-2-Q8_0.gguf --embedding \
  --ctx-size 8192 --batch-size 8192 --ubatch-size 8192 --port 8091
TURBOSEARCH_EMBEDDING_PROVIDER=openai-compatible
TURBOSEARCH_EMBEDDING_BASE_URL=http://127.0.0.1:8091/v1
TURBOSEARCH_EMBEDDING_MODEL=google/embeddinggemma-2
TURBOSEARCH_EMBEDDING_PROMPTS=embeddinggemma
TURBOSEARCH_EMBEDDING_DIMENSIONS=768
```

The GGUF files are in `ggml-org/embeddinggemma-2-GGUF` on Hugging Face. Add
`--mmproj mmproj-embeddinggemma-2-Q8_0.gguf` to embed images too (see image
embeddings below).

For SageMaker, use a base URL ending in `/openai/v1` and set
`TURBOSEARCH_EMBEDDING_AUTH=sagemaker`. TurboSearch resolves the configured AWS
profile or workload role and creates a five-minute bearer token in memory for
each request; it never stores or logs the token.

## What Works Now

- Text and code file ingestion.
- Page-aware PDF extraction through LiteParse first, then `pdftotext`, then GLM-OCR for scanned/sparse pages.
- Image OCR through GLM-OCR, with sidecar fallback.
- Audio/video transcription through Cohere Transcribe or a vLLM-compatible endpoint.
- Video frame OCR with fixed-interval frame sampling and GLM-OCR.
- Optional direct image and video-frame embeddings on a multimodal embedding server.
- SQLite chunk store with source metadata.
- SQLite FTS5 BM25 lexical search.
- Exact cosine search.
- Compressed TurboQuant-style candidate search with exact reranking, 1 to 8 bits per dimension.
- Hybrid vector + BM25 search with reciprocal-rank fusion.
- Optional Postgres storage mode for production corpora, with pgvector-backed vector search, binary-quantized candidate search, and psql_bm25s-backed BM25/hybrid fusion.
- CLI JSON mode for agent calls.
- JSON Lines ingestion events for streaming progress.

## JSON Mode

One-shot JSON requests are meant for agent tools:

```bash
bun run src/cli.ts ingest --json '{"paths":["docs"],"db":".turbosearch/index.sqlite"}'
bun run src/cli.ts search --json '{"query":"TurboQuant semantic search","backend":"turbo","k":5}'
bun run src/cli.ts extract --json '{"paths":["docs/turboquant.pdf"],"pdfParser":"liteparse"}'
```

Use JSON Lines when a human or agent wants progress while ingestion runs:

```bash
bun run src/cli.ts ingest docs --jsonl
```

Each line is an event such as `document_started`, `extraction_pending`, `document_indexed`, or `ingest_done`.

## Org-brain corpus mode

The `org-brain` adapter parses the repository's restricted frontmatter, indexes
Markdown sections without treating frontmatter as evidence, and attaches stable
entry ids, lifecycle state, headings, and line ranges to every result. Active
entries are searched by default; project, type, state, author, tag, and date
filters are available.

```bash
bun run src/cli.ts ingest /path/to/org-brain \
  --corpus org-brain --embedding openai-compatible --db .turbosearch/org-brain.sqlite
bun run src/cli.ts search "How do production releases work?" \
  --corpus org-brain --project silixon-core --backend hybrid \
  --embedding openai-compatible --db .turbosearch/org-brain.sqlite
```

Ingestion is incremental by content hash and removes indexed documents that no
longer exist in the corpus. Run the versioned retrieval evaluation dataset with:

```bash
bun run src/cli.ts eval --dataset /path/to/org-brain/evals/search.json \
  --backend hybrid --embedding openai-compatible --db .turbosearch/org-brain.sqlite
```

## Extraction Support

TurboSearch indexes only real extracted evidence. It does not index filenames as semantic evidence.

| Input | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Text/code/JSON/CSV/YAML/HTML | Good | Raw text chunks | Format-aware table chunking is still future work. |
| PDF with embedded text | Good | Page text | LiteParse first, then `pdftotext`. |
| Scanned/sparse PDF | Good | Per-page OCR text | Uses GLM-OCR after text parsers are sparse; capped by `TURBOSEARCH_PDF_OCR_MAX_PAGES`. |
| Images | Good | OCR text; optionally the image itself | Uses GLM-OCR, or sidecar fallback. Image embeddings are opt-in. |
| Audio | Good | Transcript text; timestamped segment evidence when provider returns real segments | Cohere Transcribe currently returns text only, without timestamps. |
| Video audio | Good | Transcript text; timestamped segment evidence when provider returns real segments | Uses `ffmpeg` to extract WAV before transcription. |
| Video frames | Good | OCR text and optionally the frame itself, with sampled frame path and timestamp | Fixed interval sampling, capped by max frames. |
| DOCX/PPTX/XLSX | Missing | None | Next practical format family to add. |

Sidecars are still supported as fallback evidence:

- image sidecars: `image.png.ocr.txt`, `image.png.txt`, or `image.png.md`
- video sidecars: `video.mp4.transcript.txt`, `video.mp4.ocr.txt`, `video.mp4.txt`, or `video.mp4.md`

PDFs use this chain:

1. LiteParse for layout-aware page text/table-shaped extraction.
2. `pdftotext` for pages where LiteParse is unavailable or sparse.
3. GLM-OCR for scanned/sparse pages after both text parsers fail to produce enough evidence.

LiteParse OCR is intentionally disabled so OCR stays on the selected GLM-OCR provider.

Selected model hooks:

- images: Z.ai GLM-OCR through Ollama for local development
- video/audio: Cohere Transcribe 03-2026 through Cohere API for testing, vLLM-compatible provider for private/prod
- video frames: Z.ai GLM-OCR through Ollama or vLLM after fixed-interval frame sampling
- production OCR target: vLLM GLM-OCR through an OpenAI-compatible vision endpoint

GLM-OCR notes live in [docs/glm-ocr.md](docs/glm-ocr.md). Ollama is wired for local OCR; vLLM is wired for production OCR through `/v1/chat/completions`.

OCR prompt modes are available through `ocrMode`/`--ocr-mode`: `text`, `table`, `formula`, or `figure`. This is a manual routing knob for now, not auto-detection.

Transcription is available through `transcriptionProvider`/`--transcription`: `cohere-api`, `vllm-cohere-transcribe`, or `none`. Video files require `ffmpeg`, which is used to extract WAV audio before transcription.

Video frame OCR controls:

```bash
TURBOSEARCH_VIDEO_FRAME_OCR=true
TURBOSEARCH_VIDEO_FRAME_OCR_INTERVAL_SECONDS=30
TURBOSEARCH_VIDEO_FRAME_OCR_MAX_FRAMES=60
```

Image embeddings put pictures in the same vector space as text queries, for
what OCR cannot read: schematics, board photos, diagrams, slides. Turn them on
with `TURBOSEARCH_IMAGE_EMBEDDINGS=true` (`--image-embeddings true`) and an
openai-compatible multimodal server, such as `llama-server` with EmbeddingGemma 2
and its `--mmproj` file. Each image, and each sampled video frame, becomes one
textless chunk with modality `image_visual` or `video_frame_visual` next to its
OCR text. PNG and JPEG are sent as is; other formats are converted to PNG with
`ffmpeg`. Frames use the video frame sampling settings below.

TurboQuant belongs in the vector backend, not in ingestion. The retrieval quality will mostly come from chunking, multimodal extraction, metadata filters, and reranking.

## Storage And Retrieval Backends

SQLite is the default storage backend because it keeps TurboSearch plug-and-play:

```bash
bun run src/cli.ts ingest ./client-data --storage sqlite
bun run src/cli.ts search "contract renewal terms" --backend hybrid
```

Postgres is available for production/shared corpora. It expects PostgreSQL 18 with `pgvector` and `psql_bm25s` installed or available to `CREATE EXTENSION`:

```bash
TURBOSEARCH_STORAGE=postgres
TURBOSEARCH_POSTGRES_URL=postgres://postgres:postgres@127.0.0.1:5432/postgres
TURBOSEARCH_POSTGRES_PGVECTOR=true
TURBOSEARCH_POSTGRES_PSQL_BM25S=true
bun run src/cli.ts ingest ./client-data --storage postgres
bun run src/cli.ts search "contract renewal terms" --storage postgres --backend hybrid
```

Postgres `exact` uses pgvector cosine ordering. Postgres `turbo` takes
candidates from an HNSW index over `binary_quantize(embedding)` (1 bit per
dimension, Hamming distance) and rescores them against the full vectors; it needs
pgvector 0.7 or newer. Postgres `hybrid` fuses pgvector candidates with `psql_bm25s` candidates in SQL, so production search does not need to pull every chunk into the app process. SQLite keeps the same backend names but uses local FTS5 plus in-process vector scoring.

For local production-mode testing, Docker can start a disposable PostgreSQL 18 database with both extensions:

```bash
bun run postgres:up
bun run smoke:postgres
bun run postgres:down
```

`bun run postgres:up` prints a `postgres://...` URL on port `55432`. `bun run smoke:postgres` uses its own temporary container on port `55433`, ingests a small corpus, and exercises Postgres `bm25`, `exact`, and `hybrid` search.

Retrieval backends:

- `exact`: exact cosine over stored embeddings.
- `turbo`: TurboQuant candidate search with exact reranking. `--turbo-bits`
  (default 4) sets the code width; 1, 2, 4 and 8 bits score through one table
  lookup per code byte. Codes are written at ingest, so changing the width means
  re-ingesting. `--turbo-rerank` (default 80) sets how many candidates are
  rescored exactly.
- `bm25`: lexical BM25 over chunk text.
- `hybrid`: vector + BM25 reciprocal-rank fusion.
