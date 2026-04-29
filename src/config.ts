import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { EmbeddingProviderName, OcrProviderName, SearchBackendName } from "./types.ts";

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
  ocrProvider: OcrProviderName;
  ollamaBaseUrl: string;
  ollamaGlmOcrModel: string;
  ocrPrompt: string;
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
    ocrProvider: parseOcrProvider(process.env.TURBOSEARCH_OCR_PROVIDER),
    ollamaBaseUrl: process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434",
    ollamaGlmOcrModel: process.env.OLLAMA_GLM_OCR_MODEL ?? "glm-ocr:latest",
    ocrPrompt: process.env.TURBOSEARCH_OCR_PROMPT ?? "Text Recognition:",
    ...overrides
  };
}

function parseEmbeddingProvider(value: string | undefined): EmbeddingProviderName {
  if (value === "openai") return "openai";
  return "hash";
}

function parseOcrProvider(value: string | undefined): OcrProviderName {
  if (value === "none") return "none";
  if (value === "vllm-glm-ocr") return "vllm-glm-ocr";
  return "ollama-glm-ocr";
}
