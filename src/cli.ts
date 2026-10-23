#!/usr/bin/env bun
import { createConfig, loadEnvLocal, parseEmbeddingPromptFormat, type AppConfig } from "./config.ts";
import { createEmbeddingProvider } from "./embeddings/provider.ts";
import { evaluateRetrieval, loadRetrievalEvalDataset } from "./eval/retrieval.ts";
import { extractEvidence, extractionOptions } from "./ingest/extractors.ts";
import { createOcrProvider } from "./ocr/provider.ts";
import { SearchEngine } from "./search/search-engine.ts";
import { distinctDocumentResults } from "./search/result-utils.ts";
import { createTranscriptionProvider } from "./transcription/provider.ts";
import type {
  EmbeddingProviderName,
  PdfParserName,
  SearchBackendName,
  SearchFilters,
  StorageProviderName
} from "./types.ts";

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

    if (command === "eval") {
      await runEval(engine, args, payload);
      return;
    }

    throw new Error(`Unknown command: ${command}`);
  } finally {
    await engine.close();
  }
}

async function runEval(engine: SearchEngine, args: ParsedArgs, payload: JsonPayload): Promise<void> {
  const datasetPath = getString(payload, "dataset") ?? stringFlag(args.flags.dataset) ?? args.positionals[0];
  if (!datasetPath) throw new Error("eval requires --dataset <path>.");
  const dataset = await loadRetrievalEvalDataset(datasetPath);
  const backend = parseBackend(getValue(args.flags.backend, payload.backend));
  const k = getNumber(payload, "k") ?? Number(args.flags.k ?? 5);
  const report = await evaluateRetrieval(engine, dataset, { backend, k });
  writeJson({ ok: true, command: "eval", backend, ...report });
}

async function runIngest(engine: SearchEngine, args: ParsedArgs, payload: JsonPayload): Promise<void> {
  const paths = getStringArray(payload, "paths") ?? args.positionals;
  if (paths.length === 0) {
    throw new Error("ingest requires at least one file or directory path.");
  }

  const jsonl = args.flags.jsonl === true || payload.jsonl === true;
  const corpus = parseCorpus(getValue(args.flags.corpus, payload.corpus));
  const onEvent = jsonl ? (event: Parameters<NonNullable<Parameters<SearchEngine["ingestPaths"]>[1]>>[0]) => writeJsonLine(event) : undefined;
  if (corpus === "org-brain" && paths.length !== 1) {
    throw new Error("org-brain ingestion requires exactly one repository root path.");
  }
  const summary = corpus === "org-brain"
    ? await engine.ingestOrgBrain(paths[0] as string, onEvent)
    : await engine.ingestPaths(paths, onEvent);
  if (!jsonl) writeJson({ ok: true, command: "ingest", summary });
}

async function runSearch(engine: SearchEngine, args: ParsedArgs, payload: JsonPayload): Promise<void> {
  const query = getString(payload, "query") ?? args.positionals.join(" ").trim();
  if (!query) throw new Error("search requires a query.");

  const backend = parseBackend(getValue(args.flags.backend, payload.backend));
  const k = getNumber(payload, "k") ?? Number(args.flags.k ?? 10);
  const corpus = parseCorpus(getValue(args.flags.corpus, payload.corpus));
  const includeInactive = parseBoolean(getValue(args.flags["include-inactive"], payload.includeInactive), false);
  const filters = searchFilters(args, payload, corpus, includeInactive);
  const candidates = await engine.search(query, { backend, k: corpus === "org-brain" ? k * 10 : k, filters });
  const results = corpus === "org-brain" ? distinctDocumentResults(candidates, k) : candidates;

  writeJson({ ok: true, command: "search", query, backend, k, filters, results });
}

function searchFilters(
  args: ParsedArgs,
  payload: JsonPayload,
  corpus: "org-brain" | undefined,
  includeInactive: boolean
): SearchFilters | undefined {
  const tags = getStringArray(payload, "tags") ?? commaSeparated(args.flags.tags);
  const filters: SearchFilters = {
    corpus,
    project: getString(payload, "project") ?? stringFlag(args.flags.project),
    type: getString(payload, "type") ?? stringFlag(args.flags.type),
    state: getString(payload, "state") ?? stringFlag(args.flags.state) ?? (corpus === "org-brain" && !includeInactive ? "active" : undefined),
    author: getString(payload, "author") ?? stringFlag(args.flags.author),
    tags: tags.length > 0 ? tags : undefined,
    dateFrom: getString(payload, "dateFrom") ?? stringFlag(args.flags["date-from"]),
    dateTo: getString(payload, "dateTo") ?? stringFlag(args.flags["date-to"])
  };
  return Object.values(filters).some((value) => value !== undefined) ? filters : undefined;
}

async function runExtract(args: ParsedArgs, payload: JsonPayload, config: AppConfig): Promise<void> {
  const paths = getStringArray(payload, "paths") ?? maybeSinglePath(args, payload);
  if (paths.length === 0) throw new Error("extract requires a path or paths.");

  const ocrProvider = createOcrProvider(config);
  const transcriptionProvider = createTranscriptionProvider(config);
  const results = [];
  for (const path of paths) {
    results.push(await extractEvidence(path, ocrProvider, transcriptionProvider, extractionOptions(config)));
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
    compatibleEmbeddingBaseUrl: getString(payload, "embeddingBaseUrl") ?? String(args.flags["embedding-base-url"] ?? process.env.TURBOSEARCH_EMBEDDING_BASE_URL ?? "http://127.0.0.1:8000/v1"),
    compatibleEmbeddingApiKey: getString(payload, "embeddingApiKey") ?? process.env.TURBOSEARCH_EMBEDDING_API_KEY ?? "",
    compatibleEmbeddingModel: getString(payload, "embeddingModel") ?? String(args.flags["embedding-model"] ?? process.env.TURBOSEARCH_EMBEDDING_MODEL ?? "unsloth/Qwen3-Embedding-4B"),
    compatibleEmbeddingPromptFormat: parseEmbeddingPromptFormat(getValue(args.flags["embedding-prompts"], payload.embeddingPrompts) ?? process.env.TURBOSEARCH_EMBEDDING_PROMPTS),
    compatibleEmbeddingQueryInstruction: getString(payload, "embeddingQueryInstruction") ?? String(args.flags["embedding-query-instruction"] ?? process.env.TURBOSEARCH_EMBEDDING_QUERY_INSTRUCTION ?? "Given a question about Silixon's organizational knowledge, retrieve the most relevant passages that answer it"),
    compatibleEmbeddingAuth: parseEmbeddingAuth(getValue(args.flags["embedding-auth"], payload.embeddingAuth) ?? process.env.TURBOSEARCH_EMBEDDING_AUTH),
    awsRegion: getString(payload, "awsRegion") ?? String(args.flags["aws-region"] ?? process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? "eu-west-2"),
    awsProfile: getString(payload, "awsProfile") ?? String(args.flags["aws-profile"] ?? process.env.AWS_PROFILE ?? ""),
    embeddingDimensions: getNumber(payload, "embeddingDimensions") ?? Number(args.flags["embedding-dimensions"] ?? process.env.TURBOSEARCH_EMBEDDING_DIMENSIONS ?? 1024),
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
    imageEmbeddings: parseBoolean(getValue(args.flags["image-embeddings"], payload.imageEmbeddings) ?? process.env.TURBOSEARCH_IMAGE_EMBEDDINGS, false),
    turboBits: getNumber(payload, "turboBits") ?? Number(args.flags["turbo-bits"] ?? process.env.TURBOSEARCH_TURBO_BITS ?? 4),
    turboRerankLimit: getNumber(payload, "turboRerank") ?? Number(args.flags["turbo-rerank"] ?? process.env.TURBOSEARCH_TURBO_RERANK_LIMIT ?? 80)
  });
}

function parseStorageProvider(value: string | boolean | undefined): StorageProviderName {
  if (value === "postgres") return "postgres";
  if (value === "sqlite" || value === undefined || value === false) return "sqlite";
  throw new Error(`Unsupported storage provider: ${String(value)}`);
}

function parseCorpus(value: string | boolean | undefined): "org-brain" | undefined {
  if (value === "org-brain") return "org-brain";
  if (value === undefined || value === false) return undefined;
  throw new Error(`Unsupported corpus adapter: ${String(value)}`);
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
  if (value === "openai-compatible") return "openai-compatible";
  if (value === "openai") return "openai";
  if (value === "hash" || value === undefined || value === false) return "hash";
  throw new Error(`Unsupported embedding provider: ${String(value)}`);
}

function parseEmbeddingAuth(value: string | boolean | undefined): AppConfig["compatibleEmbeddingAuth"] {
  if (value === "bearer") return "bearer";
  if (value === "sagemaker") return "sagemaker";
  if (value === "none" || value === undefined || value === false) return "none";
  throw new Error(`Unsupported embedding auth: ${String(value)}`);
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
  bun run src/cli.ts ingest <file-or-directory...> [--corpus org-brain] [--storage sqlite|postgres] [--db path]
  bun run src/cli.ts search <query> [--corpus org-brain] [--backend exact|turbo|bm25|hybrid] [--k 10]
  bun run src/cli.ts extract <file> [--json]
  bun run src/cli.ts stats [--db path]
  bun run src/cli.ts eval --dataset evals/search.json [--backend hybrid] [--k 5]

JSON mode:
  bun run src/cli.ts ingest --json '{"paths":["docs"],"db":".turbosearch/index.sqlite"}'
  bun run src/cli.ts ingest --jsonl docs
  bun run src/cli.ts search --json '{"query":"TurboQuant search","backend":"turbo","k":5}'
  bun run src/cli.ts extract --json '{"paths":["image.png","video.mp4"]}'

Options:
  --embedding      hash, openai, or openai-compatible
  --embedding-base-url
                   OpenAI-compatible base URL, default http://127.0.0.1:8000/v1
  --embedding-model
                   Model name, default unsloth/Qwen3-Embedding-4B
  --embedding-dimensions
                   Matryoshka output size, default 1024
  --embedding-prompts
                   qwen3 (default), embeddinggemma, or none: the query/document prompts the model expects
  --image-embeddings
                   false by default; embed images and sampled video frames directly (multimodal server)
  --embedding-auth none, bearer, or sagemaker; sagemaker uses short-lived IAM tokens
  --storage        sqlite by default; postgres requires TURBOSEARCH_POSTGRES_URL
  --postgres-url   Postgres connection URL for --storage postgres
  --postgres-psql-bm25s
                   true by default; creates/uses psql_bm25s for Postgres BM25
  --postgres-pgvector
                   true by default; creates/uses pgvector for Postgres vector search
  --postgres-vector-index
                   true by default; creates an HNSW cosine index for pgvector
  --backend        exact, turbo, bm25, or hybrid
  --corpus         org-brain enables schema-aware ingestion, lifecycle defaults, and citations
  --project        filter org-brain results by project
  --type           filter org-brain results by entry type
  --state          filter org-brain results by lifecycle state; active is the default
  --author         filter org-brain results by author
  --tags           comma-separated tags; every tag must match
  --date-from      inclusive YYYY-MM-DD lower bound
  --date-to        inclusive YYYY-MM-DD upper bound
  --include-inactive
                   do not apply the default active lifecycle filter
  --pdf-parser     liteparse by default; use pdftotext to skip LiteParse
  --ocr            ollama-glm-ocr by default; use none to skip OCR or vllm-glm-ocr for prod placeholder
  --ocr-mode       text, table, formula, or figure
  --transcription  none by default; cohere-api or vllm-cohere-transcribe
  --pdf-ocr-max-pages
                   Max scanned/sparse PDF pages to OCR, default 200
  --video-frame-ocr
                   true by default; set false to skip visual OCR on videos
  --video-frame-ocr-interval
                   Seconds between sampled video frames (OCR and image embeddings), default 30
  --video-frame-ocr-max-frames
                   Max video frames sampled per file, default 60
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

function stringFlag(value: string | boolean | undefined): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function commaSeparated(value: string | boolean | undefined): string[] {
  return typeof value === "string"
    ? value.split(",").map((item) => item.trim()).filter(Boolean)
    : [];
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
