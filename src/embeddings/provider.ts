import type { AppConfig } from "../config.ts";
import type { EmbeddingProviderName } from "../types.ts";
import { HashEmbeddingProvider } from "./hash.ts";
import { OpenAiEmbeddingProvider } from "./openai.ts";
import { OpenAiCompatibleEmbeddingProvider } from "./openai-compatible.ts";

export type TextEmbeddingDocument = { kind: "text"; text: string; title?: string };
// An image file to embed directly; only providers with supportsImages accept it.
export type ImageEmbeddingDocument = { kind: "image"; path: string };
export type EmbeddingDocument = TextEmbeddingDocument | ImageEmbeddingDocument;

export interface EmbeddingProvider {
  readonly name: EmbeddingProviderName;
  readonly dimensions: number;
  readonly supportsImages: boolean;
  embedDocuments(documents: EmbeddingDocument[]): Promise<Float32Array[]>;
  embedQuery(query: string): Promise<Float32Array>;
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
      promptFormat: config.compatibleEmbeddingPromptFormat,
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

/** Text-only providers call this once per batch; it names the provider that cannot embed images. */
export function textDocuments(provider: EmbeddingProviderName, documents: EmbeddingDocument[]): TextEmbeddingDocument[] {
  return documents.map((document) => {
    if (document.kind === "text") return document;
    throw new Error(`The ${provider} embedding provider cannot embed images (${document.path}).`);
  });
}
