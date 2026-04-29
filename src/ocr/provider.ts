import type { AppConfig } from "../config.ts";
import type { OcrProviderName } from "../types.ts";
import { OllamaGlmOcrProvider } from "./providers/ollama.ts";
import { VllmGlmOcrProvider } from "./providers/vllm.ts";

export type OcrRequest = {
  imagePath: string;
  prompt?: string;
};

export type OcrResponse = {
  text: string;
  model: string;
  provider: OcrProviderName;
};

export interface OcrProvider {
  readonly name: OcrProviderName;
  extractText(request: OcrRequest): Promise<OcrResponse>;
}

export function createOcrProvider(config: AppConfig): OcrProvider | null {
  if (config.ocrProvider === "none") return null;

  if (config.ocrProvider === "vllm-glm-ocr") {
    return new VllmGlmOcrProvider();
  }

  return new OllamaGlmOcrProvider({
    baseUrl: config.ollamaBaseUrl,
    model: config.ollamaGlmOcrModel,
    defaultPrompt: config.ocrPrompt,
    contextTokens: config.ocrContextTokens,
    maxOutputTokens: config.ocrMaxOutputTokens
  });
}
