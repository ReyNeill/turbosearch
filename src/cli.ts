#!/usr/bin/env bun
import { createConfig, loadEnvLocal, type AppConfig } from "./config.ts";
import { createEmbeddingProvider } from "./embeddings/provider.ts";
import { extractEvidence } from "./ingest/extractors.ts";
import { createOcrProvider } from "./ocr/provider.ts";
import { SearchEngine } from "./search/search-engine.ts";
import { createTranscriptionProvider } from "./transcription/provider.ts";
import type { EmbeddingProviderName, PdfParserName, SearchBackendName, StorageProviderName } from "./types.ts";

type ParsedArgs = {
  command: string | undefined;
  positionals: string[];
  flags: Record<string, string | boolean>;
};

type JsonPayload = Record<string, unknown>;

async function main(): Promise<void> {
  loadEnvLocal();
  const args = parseArgs(process.argv.slice(2));
  const payload = await readJsonPayload(args);
  const command = args.command ?? getString(payload, "command");

  if (!command || args.flags.help === true || args.flags.h === true) {
    printHelp();
    return;
  }

  const config = configFromArgs(args, payload);
  const provider = createEmbeddingProvider(config);
  const engine = new SearchEngine(config, provider);

  try {
    if (command === "ingest") {
      await runIngest(engine, args, payload);
      return;
    }

    if (command === "search") {
      await runSearch(engine, args, payload);
      return;
    }

    if (command === "extract") {
      await runExtract(args, payload, config);
      return;
    }

    if (command === "stats") {
      const stats = await engine.stats();
      writeJson({ ok: true, command: "stats", stats });
      return;
    }

    throw new Error(`Unknown command: ${command}`);
  } finally {
    await engine.close();
  }
}

async function runIngest(engine: SearchEngine, args: ParsedArgs, payload: JsonPayload): Promise<void> {
  const paths = getStringArray(payload, "paths") ?? args.positionals;
  if (paths.length === 0) {
    throw new Error("ingest requires at least one file or directory path.");
  }

  const jsonl = args.flags.jsonl === true || payload.jsonl === true;
  const summary = await engine.ingestPaths(paths, jsonl ? (event) => writeJsonLine(event) : undefined);
  if (!jsonl) writeJson({ ok: true, command: "ingest", summary });
}

async function runSearch(engine: SearchEngine, args: ParsedArgs, payload: JsonPayload): Promise<void> {
  const query = getString(payload, "query") ?? args.positionals.join(" ").trim();
  if (!query) throw new Error("search requires a query.");

  const backend = parseBackend(getValue(args.flags.backend, payload.backend));
  const k = getNumber(payload, "k") ?? Number(args.flags.k ?? 10);
  const results = await engine.search(query, { backend, k });

  writeJson({ ok: true, command: "search", query, backend, k, results });
}

async function runExtract(args: ParsedArgs, payload: JsonPayload, config: AppConfig): Promise<void> {
  const paths = getStringArray(payload, "paths") ?? maybeSinglePath(args, payload);
  if (paths.length === 0) throw new Error("extract requires a path or paths.");

  const ocrProvider = createOcrProvider(config);
  const transcriptionProvider = createTranscriptionProvider(config);
  const results = [];
  for (const path of paths) {
    results.push(await extractEvidence(path, ocrProvider, transcriptionProvider, {
      pdfParser: config.pdfParser,
      pdfOcrMaxPages: config.pdfOcrMaxPages,
      videoFrameOcr: config.videoFrameOcr,
      videoFrameOcrIntervalSeconds: config.videoFrameOcrIntervalSeconds,
      videoFrameOcrMaxFrames: config.videoFrameOcrMaxFrames
    }));
  }

  writeJson({ ok: true, command: "extract", results });
}

function configFromArgs(args: ParsedArgs, payload: JsonPayload): AppConfig {
  return createConfig({
    storageProvider: parseStorageProvider(getValue(args.flags.storage, payload.storage) ?? process.env.TURBOSEARCH_STORAGE),
    dbPath: getString(payload, "db") ?? String(args.flags.db ?? ".turbosearch/index.sqlite"),
    postgresUrl: getString(payload, "postgresUrl") ?? String(args.flags["postgres-url"] ?? process.env.TURBOSEARCH_POSTGRES_URL ?? ""),
    postgresUsePsqlBm25s: parseBoolean(getValue(args.flags["postgres-psql-bm25s"], payload.postgresPsqlBm25s) ?? process.env.TURBOSEARCH_POSTGRES_PSQL_BM25S, true),
    postgresUsePgvector: parseBoolean(getValue(args.flags["postgres-pgvector"], payload.postgresPgvector) ?? process.env.TURBOSEARCH_POSTGRES_PGVECTOR, true),
    postgresVectorIndex: parseBoolean(getValue(args.flags["postgres-vector-index"], payload.postgresVectorIndex) ?? process.env.TURBOSEARCH_POSTGRES_VECTOR_INDEX, true),
    embeddingProvider: parseEmbeddingProvider(getValue(args.flags.embedding, payload.embedding)),
    pdfParser: parsePdfParser(
      getValue(args.flags["pdf-parser"], payload.pdfParser) ?? process.env.TURBOSEARCH_PDF_PARSER
    ),
    pdfOcrMaxPages: getNumber(payload, "pdfOcrMaxPages") ?? Number(args.flags["pdf-ocr-max-pages"] ?? process.env.TURBOSEARCH_PDF_OCR_MAX_PAGES ?? 200),
    ocrProvider: parseOcrProvider(getValue(args.flags.ocr, payload.ocrProvider)),
    ollamaBaseUrl: getString(payload, "ollamaBaseUrl") ?? String(args.flags["ollama-base-url"] ?? process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434"),
    ollamaGlmOcrModel: getString(payload, "ollamaGlmOcrModel") ?? String(args.flags["ollama-glm-ocr-model"] ?? process.env.OLLAMA_GLM_OCR_MODEL ?? "glm-ocr:latest"),
    vllmGlmOcrBaseUrl: getString(payload, "vllmGlmOcrBaseUrl") ?? String(args.flags["vllm-glm-ocr-base-url"] ?? process.env.VLLM_GLM_OCR_BASE_URL ?? "http://127.0.0.1:8000/v1"),
    vllmGlmOcrApiKey: getString(payload, "vllmGlmOcrApiKey") ?? String(args.flags["vllm-glm-ocr-api-key"] ?? process.env.VLLM_GLM_OCR_API_KEY ?? ""),
    vllmGlmOcrModel: getString(payload, "vllmGlmOcrModel") ?? String(args.flags["vllm-glm-ocr-model"] ?? process.env.VLLM_GLM_OCR_MODEL ?? "zai-org/GLM-OCR"),
    ocrMode: parseOcrMode(getValue(args.flags["ocr-mode"], payload.ocrMode)),
    ocrPrompt: getString(payload, "ocrPrompt") ?? String(args.flags["ocr-prompt"] ?? process.env.TURBOSEARCH_OCR_PROMPT ?? "Text Recognition:"),
    ocrContextTokens: getNumber(payload, "ocrContextTokens") ?? Number(args.flags["ocr-context-tokens"] ?? process.env.TURBOSEARCH_OCR_CONTEXT_TOKENS ?? 8192),
    ocrMaxOutputTokens: getNumber(payload, "ocrMaxOutputTokens") ?? Number(args.flags["ocr-max-output-tokens"] ?? process.env.TURBOSEARCH_OCR_MAX_OUTPUT_TOKENS ?? 2048),
    transcriptionProvider: parseTranscriptionProvider(getValue(args.flags.transcription, payload.transcriptionProvider) ?? process.env.TURBOSEARCH_TRANSCRIPTION_PROVIDER),
    cohereApiKey: getString(payload, "cohereApiKey") ?? process.env.COHERE_API_KEY ?? "",
    cohereTranscribeEndpoint: getString(payload, "cohereTranscribeEndpoint") ?? String(args.flags["cohere-transcribe-endpoint"] ?? process.env.COHERE_TRANSCRIBE_ENDPOINT ?? "https://api.cohere.com/v2/audio/transcriptions"),
    cohereTranscribeModel: getString(payload, "cohereTranscribeModel") ?? String(args.flags["cohere-transcribe-model"] ?? process.env.COHERE_TRANSCRIBE_MODEL ?? "cohere-transcribe-03-2026"),
    cohereTranscribeLanguage: getString(payload, "cohereTranscribeLanguage") ?? String(args.flags["cohere-transcribe-language"] ?? process.env.COHERE_TRANSCRIBE_LANGUAGE ?? "en"),
    videoFrameOcr: parseBoolean(getValue(args.flags["video-frame-ocr"], payload.videoFrameOcr) ?? process.env.TURBOSEARCH_VIDEO_FRAME_OCR, true),
    videoFrameOcrIntervalSeconds: getNumber(payload, "videoFrameOcrIntervalSeconds") ?? Number(args.flags["video-frame-ocr-interval"] ?? process.env.TURBOSEARCH_VIDEO_FRAME_OCR_INTERVAL_SECONDS ?? 30),
    videoFrameOcrMaxFrames: getNumber(payload, "videoFrameOcrMaxFrames") ?? Number(args.flags["video-frame-ocr-max-frames"] ?? process.env.TURBOSEARCH_VIDEO_FRAME_OCR_MAX_FRAMES ?? 60),
    turboBits: getNumber(payload, "turboBits") ?? Number(args.flags["turbo-bits"] ?? process.env.TURBOSEARCH_TURBO_BITS ?? 4),
    turboRerankLimit: getNumber(payload, "turboRerank") ?? Number(args.flags["turbo-rerank"] ?? process.env.TURBOSEARCH_TURBO_RERANK_LIMIT ?? 80)
  });
}

function parseStorageProvider(value: string | boolean | undefined): StorageProviderName {
  if (value === "postgres") return "postgres";
  if (value === "sqlite" || value === undefined || value === false) return "sqlite";
  throw new Error(`Unsupported storage provider: ${String(value)}`);
}

function parseTranscriptionProvider(value: string | boolean | undefined): AppConfig["transcriptionProvider"] {
  if (value === "cohere-api") return "cohere-api";
  if (value === "vllm-cohere-transcribe") return "vllm-cohere-transcribe";
  if (value === "none" || value === undefined || value === false) return "none";
  throw new Error(`Unsupported transcription provider: ${String(value)}`);
}

function parseOcrMode(value: string | boolean | undefined): AppConfig["ocrMode"] {
  if (value === "table") return "table";
  if (value === "formula") return "formula";
  if (value === "figure") return "figure";
  if (value === "text" || value === undefined || value === false) return "text";
  throw new Error(`Unsupported OCR mode: ${String(value)}`);
}

function parseOcrProvider(value: string | boolean | undefined): AppConfig["ocrProvider"] {
  if (value === "none") return "none";
  if (value === "vllm-glm-ocr") return "vllm-glm-ocr";
  if (value === "ollama-glm-ocr" || value === undefined || value === false) return "ollama-glm-ocr";
  throw new Error(`Unsupported OCR provider: ${String(value)}`);
}

function parseEmbeddingProvider(value: string | boolean | undefined): EmbeddingProviderName {
  if (value === "openai") return "openai";
  if (value === "hash" || value === undefined || value === false) return "hash";
  throw new Error(`Unsupported embedding provider: ${String(value)}`);
}

function parsePdfParser(value: string | boolean | undefined): PdfParserName {
  if (value === "pdftotext") return "pdftotext";
  if (value === "liteparse" || value === undefined || value === false) return "liteparse";
  throw new Error(`Unsupported PDF parser: ${String(value)}`);
}

function parseBackend(value: string | boolean | undefined): SearchBackendName {
  if (value === "hybrid") return "hybrid";
  if (value === "bm25") return "bm25";
  if (value === "turbo") return "turbo";
  if (value === "exact" || value === undefined || value === false) return "exact";
  throw new Error(`Unsupported backend: ${String(value)}`);
}

function parseBoolean(value: string | boolean | undefined, defaultValue: boolean): boolean {
  if (typeof value === "boolean") return value;
  if (value === undefined) return defaultValue;
  if (value === "true" || value === "1" || value === "yes") return true;
  if (value === "false" || value === "0" || value === "no") return false;
  throw new Error(`Unsupported boolean value: ${String(value)}`);
}

function parseArgs(argv: string[]): ParsedArgs {
  const [command, ...rest] = argv;
  const positionals: string[] = [];
  const flags: Record<string, string | boolean> = {};

  for (let i = 0; i < rest.length; i += 1) {
    const arg = rest[i];
    if (!arg) continue;

    if (arg.startsWith("--")) {
      const [key, inlineValue] = arg.slice(2).split("=", 2);
      if (!key) continue;
      if (inlineValue !== undefined) {
        flags[key] = inlineValue;
        continue;
      }

      const next = rest[i + 1];
      if (next && !next.startsWith("-")) {
        flags[key] = next;
        i += 1;
      } else {
        flags[key] = true;
      }
      continue;
    }

    if (arg.startsWith("-") && arg.length === 2) {
      flags[arg.slice(1)] = true;
      continue;
    }

    positionals.push(arg);
  }

  return { command, positionals, flags };
}

function printHelp(): void {
  console.log(`TurboSearch

Usage:
  bun run src/cli.ts ingest <file-or-directory...> [--storage sqlite|postgres] [--db path] [--embedding hash|openai]
  bun run src/cli.ts search <query> [--backend exact|turbo|bm25|hybrid] [--k 10] [--storage sqlite|postgres]
  bun run src/cli.ts extract <file> [--json]
  bun run src/cli.ts stats [--db path]

JSON mode:
  bun run src/cli.ts ingest --json '{"paths":["docs"],"db":".turbosearch/index.sqlite"}'
  bun run src/cli.ts ingest --jsonl docs
  bun run src/cli.ts search --json '{"query":"TurboQuant search","backend":"turbo","k":5}'
  bun run src/cli.ts extract --json '{"paths":["image.png","video.mp4"]}'

Options:
  --embedding      hash by default; openai requires OPENAI_API_KEY in .env.local
  --storage        sqlite by default; postgres requires TURBOSEARCH_POSTGRES_URL
  --postgres-url   Postgres connection URL for --storage postgres
  --postgres-psql-bm25s
                   true by default; creates/uses psql_bm25s for Postgres BM25
  --postgres-pgvector
                   true by default; creates/uses pgvector for Postgres vector search
  --postgres-vector-index
                   true by default; creates an HNSW cosine index for pgvector
  --backend        exact, turbo, bm25, or hybrid
  --pdf-parser     liteparse by default; use pdftotext to skip LiteParse
  --ocr            ollama-glm-ocr by default; use none to skip OCR or vllm-glm-ocr for prod placeholder
  --ocr-mode       text, table, formula, or figure
  --transcription  none by default; cohere-api or vllm-cohere-transcribe
  --pdf-ocr-max-pages
                   Max scanned/sparse PDF pages to OCR, default 200
  --video-frame-ocr
                   true by default; set false to skip visual OCR on videos
  --video-frame-ocr-interval
                   Seconds between sampled video frames, default 30
  --video-frame-ocr-max-frames
                   Max video frames to OCR per file, default 60
  --ollama-base-url
                   Ollama base URL, default http://127.0.0.1:11434
  --vllm-glm-ocr-base-url
                   OpenAI-compatible vLLM base URL, default http://127.0.0.1:8000/v1
  --json           read a JSON request from the next argument, or stdin when no value is provided
  --jsonl          stream ingestion events as JSON Lines
  --turbo-bits     compressed backend bit width, default 4
  --turbo-rerank   exact rerank candidate count, default 80
`);
}

async function readJsonPayload(args: ParsedArgs): Promise<JsonPayload> {
  const jsonFlag = args.flags.json;
  if (jsonFlag === undefined || jsonFlag === false) return {};

  const body = typeof jsonFlag === "string" ? jsonFlag : await Bun.stdin.text();
  const trimmed = body.trim();
  if (!trimmed) return {};

  const parsed = JSON.parse(trimmed) as unknown;
  if (!isRecord(parsed)) throw new Error("--json payload must be an object.");
  return parsed;
}

function maybeSinglePath(args: ParsedArgs, payload: JsonPayload): string[] {
  const payloadPath = getString(payload, "path");
  if (payloadPath) return [payloadPath];
  return args.positionals;
}

function getString(payload: JsonPayload, key: string): string | undefined {
  const value = payload[key];
  return typeof value === "string" ? value : undefined;
}

function getNumber(payload: JsonPayload, key: string): number | undefined {
  const value = payload[key];
  return typeof value === "number" ? value : undefined;
}

function getStringArray(payload: JsonPayload, key: string): string[] | undefined {
  const value = payload[key];
  if (!Array.isArray(value)) return undefined;
  if (!value.every((item) => typeof item === "string")) {
    throw new Error(`${key} must be an array of strings.`);
  }
  return value;
}

function getValue(primary: unknown, fallback: unknown): string | boolean | undefined {
  if (typeof primary === "string" || typeof primary === "boolean") return primary;
  if (typeof fallback === "string" || typeof fallback === "boolean") return fallback;
  return undefined;
}

function isRecord(value: unknown): value is JsonPayload {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function writeJson(value: unknown): void {
  console.log(JSON.stringify(value, null, 2));
}

function writeJsonLine(value: unknown): void {
  console.log(JSON.stringify(value));
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});
