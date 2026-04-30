import type { AppConfig } from "../config.ts";
import type { OcrMode, OcrProviderName } from "../types.ts";
import { OllamaGlmOcrProvider } from "./providers/ollama.ts";
import { VllmGlmOcrProvider } from "./providers/vllm.ts";

export type OcrRequest = {
  imagePath: string;
  mode?: OcrMode;
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
    defaultMode: config.ocrMode,
    defaultPrompt: config.ocrPrompt,
    contextTokens: config.ocrContextTokens,
    maxOutputTokens: config.ocrMaxOutputTokens
  });
}

export function promptForOcrMode(mode: OcrMode): string {
  if (mode === "table") return "Table Recognition:";
  if (mode === "formula") return "Formula Recognition:";
  if (mode === "figure") return "Figure Recognition:";
  return "Text Recognition:";
}
