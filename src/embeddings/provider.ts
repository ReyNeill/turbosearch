import type { AppConfig } from "../config.ts";
import type { EmbeddingProviderName } from "../types.ts";
import { HashEmbeddingProvider } from "./hash.ts";
import { OpenAiEmbeddingProvider } from "./openai.ts";
import { OpenAiCompatibleEmbeddingProvider } from "./openai-compatible.ts";

export interface EmbeddingProvider {
  readonly name: EmbeddingProviderName;
  readonly dimensions: number;
  embedTexts(texts: string[]): Promise<Float32Array[]>;
  embedText(text: string): Promise<Float32Array>;
}

export function createEmbeddingProvider(config: AppConfig): EmbeddingProvider {
  if (config.embeddingProvider === "openai-compatible") {
    return new OpenAiCompatibleEmbeddingProvider({
      baseUrl: config.compatibleEmbeddingBaseUrl,
      apiKey: config.compatibleEmbeddingApiKey,
      auth: config.compatibleEmbeddingAuth,
      awsRegion: config.awsRegion,
      awsProfile: config.awsProfile,
      model: config.compatibleEmbeddingModel,
      queryInstruction: config.compatibleEmbeddingQueryInstruction,
      dimensions: config.embeddingDimensions
    });
  }

  if (config.embeddingProvider === "openai") {
    return new OpenAiEmbeddingProvider({
      model: config.openAiEmbeddingModel,
      dimensions: config.embeddingDimensions
    });
  }

  return new HashEmbeddingProvider(config.embeddingDimensions);
}
