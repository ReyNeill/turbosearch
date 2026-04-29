import type { OcrProvider, OcrRequest, OcrResponse } from "../provider.ts";

export class VllmGlmOcrProvider implements OcrProvider {
  readonly name = "vllm-glm-ocr" as const;

  async extractText(_request: OcrRequest): Promise<OcrResponse> {
    throw new Error(
      "vLLM GLM-OCR provider is reserved for production but is not wired yet. Use TURBOSEARCH_OCR_PROVIDER=ollama-glm-ocr for local development."
    );
  }
}
