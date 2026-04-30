import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type {
  EmbeddingProviderName,
  OcrMode,
  OcrProviderName,
  PdfParserName,
  SearchBackendName,
  TranscriptionProviderName
} from "./types.ts";

export type AppConfig = {
  dbPath: string;
  embeddingProvider: EmbeddingProviderName;
  openAiEmbeddingModel: string;
  embeddingDimensions: number;
  chunkTokens: number;
  chunkOverlapTokens: number;
  turboBits: number;
  turboRerankLimit: number;
  defaultBackend: SearchBackendName;
  pdfParser: PdfParserName;
  pdfOcrMaxPages: number;
  ocrProvider: OcrProviderName;
  ollamaBaseUrl: string;
  ollamaGlmOcrModel: string;
  vllmGlmOcrBaseUrl: string;
  vllmGlmOcrApiKey: string;
  vllmGlmOcrModel: string;
  ocrMode: OcrMode;
  ocrPrompt: string;
  ocrContextTokens: number;
  ocrMaxOutputTokens: number;
  transcriptionProvider: TranscriptionProviderName;
  cohereApiKey: string;
  cohereTranscribeEndpoint: string;
  cohereTranscribeModel: string;
  cohereTranscribeLanguage: string;
  videoFrameOcr: boolean;
  videoFrameOcrIntervalSeconds: number;
  videoFrameOcrMaxFrames: number;
};

export function loadEnvLocal(cwd = process.cwd()): void {
  const envPath = join(cwd, ".env.local");
  if (!existsSync(envPath)) return;

  const body = readFileSync(envPath, "utf8");
  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const equals = line.indexOf("=");
    if (equals === -1) continue;

    const key = line.slice(0, equals).trim();
    const value = line.slice(equals + 1).trim().replace(/^['"]|['"]$/g, "");
    if (!(key in process.env)) process.env[key] = value;
  }
}

export function createConfig(overrides: Partial<AppConfig> = {}): AppConfig {
  loadEnvLocal();

  return {
    dbPath: ".turbosearch/index.sqlite",
    embeddingProvider: parseEmbeddingProvider(process.env.TURBOSEARCH_EMBEDDING_PROVIDER),
    openAiEmbeddingModel: process.env.TURBOSEARCH_OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small",
    embeddingDimensions: Number(process.env.TURBOSEARCH_EMBEDDING_DIMENSIONS ?? 1536),
    chunkTokens: Number(process.env.TURBOSEARCH_CHUNK_TOKENS ?? 700),
    chunkOverlapTokens: Number(process.env.TURBOSEARCH_CHUNK_OVERLAP_TOKENS ?? 100),
    turboBits: Number(process.env.TURBOSEARCH_TURBO_BITS ?? 4),
    turboRerankLimit: Number(process.env.TURBOSEARCH_TURBO_RERANK_LIMIT ?? 80),
    defaultBackend: "exact",
    pdfParser: parsePdfParser(process.env.TURBOSEARCH_PDF_PARSER),
    pdfOcrMaxPages: Number(process.env.TURBOSEARCH_PDF_OCR_MAX_PAGES ?? 200),
    ocrProvider: parseOcrProvider(process.env.TURBOSEARCH_OCR_PROVIDER),
    ollamaBaseUrl: process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434",
    ollamaGlmOcrModel: process.env.OLLAMA_GLM_OCR_MODEL ?? "glm-ocr:latest",
    vllmGlmOcrBaseUrl: process.env.VLLM_GLM_OCR_BASE_URL ?? "http://127.0.0.1:8000/v1",
    vllmGlmOcrApiKey: process.env.VLLM_GLM_OCR_API_KEY ?? "",
    vllmGlmOcrModel: process.env.VLLM_GLM_OCR_MODEL ?? "zai-org/GLM-OCR",
    ocrMode: parseOcrMode(process.env.TURBOSEARCH_OCR_MODE),
    ocrPrompt: process.env.TURBOSEARCH_OCR_PROMPT ?? "",
    ocrContextTokens: Number(process.env.TURBOSEARCH_OCR_CONTEXT_TOKENS ?? 8192),
    ocrMaxOutputTokens: Number(process.env.TURBOSEARCH_OCR_MAX_OUTPUT_TOKENS ?? 2048),
    transcriptionProvider: parseTranscriptionProvider(process.env.TURBOSEARCH_TRANSCRIPTION_PROVIDER),
    cohereApiKey: process.env.COHERE_API_KEY ?? "",
    cohereTranscribeEndpoint: process.env.COHERE_TRANSCRIBE_ENDPOINT ?? "https://api.cohere.com/v2/audio/transcriptions",
    cohereTranscribeModel: process.env.COHERE_TRANSCRIBE_MODEL ?? "cohere-transcribe-03-2026",
    cohereTranscribeLanguage: process.env.COHERE_TRANSCRIBE_LANGUAGE ?? "en",
    videoFrameOcr: parseBoolean(process.env.TURBOSEARCH_VIDEO_FRAME_OCR, true),
    videoFrameOcrIntervalSeconds: Number(process.env.TURBOSEARCH_VIDEO_FRAME_OCR_INTERVAL_SECONDS ?? 30),
    videoFrameOcrMaxFrames: Number(process.env.TURBOSEARCH_VIDEO_FRAME_OCR_MAX_FRAMES ?? 60),
    ...overrides
  };
}

function parseOcrMode(value: string | undefined): OcrMode {
  if (value === "table") return "table";
  if (value === "formula") return "formula";
  if (value === "figure") return "figure";
  return "text";
}

function parseEmbeddingProvider(value: string | undefined): EmbeddingProviderName {
  if (value === "openai") return "openai";
  return "hash";
}

function parsePdfParser(value: string | undefined): PdfParserName {
  if (value === "pdftotext") return "pdftotext";
  return "liteparse";
}

function parseOcrProvider(value: string | undefined): OcrProviderName {
  if (value === "none") return "none";
  if (value === "vllm-glm-ocr") return "vllm-glm-ocr";
  return "ollama-glm-ocr";
}

function parseTranscriptionProvider(value: string | undefined): TranscriptionProviderName {
  if (value === "none") return "none";
  if (value === "vllm-cohere-transcribe") return "vllm-cohere-transcribe";
  if (value === "cohere-api") return "cohere-api";
  return "none";
}

function parseBoolean(value: string | undefined, defaultValue: boolean): boolean {
  if (value === undefined) return defaultValue;
  if (value === "true" || value === "1" || value === "yes") return true;
  if (value === "false" || value === "0" || value === "no") return false;
  return defaultValue;
}
