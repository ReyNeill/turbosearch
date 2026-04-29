import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { EmbeddingProviderName, SearchBackendName } from "./types.ts";

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
    ...overrides
  };
}

function parseEmbeddingProvider(value: string | undefined): EmbeddingProviderName {
  if (value === "openai") return "openai";
  return "hash";
}
