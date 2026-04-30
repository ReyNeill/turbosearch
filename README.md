# TurboSearch

Local semantic search for agent-accessible client corpora.

The project is intentionally split into ingestion, storage, embedding, and vector backend layers. Exact vector search works as the baseline. A TurboQuant-inspired compressed backend is included behind the same API so it can be benchmarked and replaced with a faster Rust/Zig backend later if needed.

## Quick Start

```bash
bun install
bun run src/cli.ts ingest docs
bun run src/cli.ts search "TurboQuant semantic search" --backend exact
bun run src/cli.ts search "TurboQuant semantic search" --backend turbo
```

By default this uses local hash embeddings, which are private and deterministic but not high-quality semantic embeddings. For real semantic search, copy `.env.local.example` to `.env.local`, set `OPENAI_API_KEY`, and ingest with:

```bash
bun run src/cli.ts ingest ./client-data --embedding openai
bun run src/cli.ts search "contract renewal terms" --embedding openai
```

## What Works Now

- Text and code file ingestion.
- Page-aware PDF text extraction through `pdfinfo`/`pdftotext` when available.
- SQLite chunk store with source metadata.
- Exact cosine search.
- Compressed TurboQuant-style candidate search with exact reranking.
- CLI JSON mode for agent calls.
- JSON Lines ingestion events for streaming progress.

## JSON Mode

One-shot JSON requests are meant for agent tools:

```bash
bun run src/cli.ts ingest --json '{"paths":["docs"],"db":".turbosearch/index.sqlite"}'
bun run src/cli.ts search --json '{"query":"TurboQuant semantic search","backend":"turbo","k":5}'
bun run src/cli.ts extract --json '{"paths":["docs/turboquant.pdf"]}'
```

Use JSON Lines when a human or agent wants progress while ingestion runs:

```bash
bun run src/cli.ts ingest docs --jsonl
```

Each line is an event such as `document_started`, `extraction_pending`, `document_indexed`, or `ingest_done`.

## Multimodal Plan

Images and videos are currently indexed only when real text evidence exists as sidecar files:

- image sidecars: `image.png.ocr.txt`, `image.png.txt`, or `image.png.md`
- video sidecars: `video.mp4.transcript.txt`, `video.mp4.ocr.txt`, `video.mp4.txt`, or `video.mp4.md`

When sidecars are missing, extraction reports a pending local-model integration instead of indexing filenames as fake semantic evidence. Scanned PDF pages fall back to GLM-OCR when `pdftotext` finds little or no embedded text.

Selected/pending local model hooks:

- images: Z.ai GLM-OCR through Ollama for local development
- video/audio: Cohere Transcribe 03-2026 through Cohere API for testing, vLLM-compatible provider for private/prod
- video frames: Z.ai GLM-OCR through Ollama after frame sampling
- production OCR target: vLLM GLM-OCR provider

GLM-OCR notes live in [docs/glm-ocr.md](docs/glm-ocr.md). Ollama is wired for local image OCR; vLLM is the production target and still needs the exact service contract.

OCR prompt modes are available through `ocrMode`/`--ocr-mode`: `text`, `table`, `formula`, or `figure`. This is a manual routing knob for now, not auto-detection.

Transcription is available through `transcriptionProvider`/`--transcription`: `cohere-api`, `vllm-cohere-transcribe`, or `none`. Video files require `ffmpeg`, which is used to extract WAV audio before transcription.

TurboQuant belongs in the vector backend, not in ingestion. The retrieval quality will mostly come from chunking, multimodal extraction, metadata filters, and reranking.
