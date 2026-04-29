import type { AppConfig } from "../config.ts";
import type { EmbeddingProviderName } from "../types.ts";
import { HashEmbeddingProvider } from "./hash.ts";
import { OpenAiEmbeddingProvider } from "./openai.ts";

export interface EmbeddingProvider {
  readonly name: EmbeddingProviderName;
  readonly dimensions: number;
  embedTexts(texts: string[]): Promise<Float32Array[]>;
  embedText(text: string): Promise<Float32Array>;
}

export function createEmbeddingProvider(config: AppConfig): EmbeddingProvider {
  if (config.embeddingProvider === "openai") {
    return new OpenAiEmbeddingProvider({
      model: config.openAiEmbeddingModel,
      dimensions: config.embeddingDimensions
    });
  }

  return new HashEmbeddingProvider(config.embeddingDimensions);
}
