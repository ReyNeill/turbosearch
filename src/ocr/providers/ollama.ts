import { readFile } from "node:fs/promises";
import type { OcrProvider, OcrRequest, OcrResponse } from "../provider.ts";

type OllamaGlmOcrOptions = {
  baseUrl: string;
  model: string;
  defaultPrompt: string;
};

type OllamaGenerateResponse = {
  response?: string;
  error?: string;
};

export class OllamaGlmOcrProvider implements OcrProvider {
  readonly name = "ollama-glm-ocr" as const;
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly defaultPrompt: string;

  constructor(options: OllamaGlmOcrOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.model = options.model;
    this.defaultPrompt = options.defaultPrompt;
  }

  async extractText(request: OcrRequest): Promise<OcrResponse> {
    const image = await readFile(request.imagePath);
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        prompt: request.prompt ?? this.defaultPrompt,
        images: [image.toString("base64")],
        stream: false,
        options: {
          temperature: 0
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama GLM-OCR request failed: ${response.status} ${await response.text()}`);
    }

    const json = await response.json() as OllamaGenerateResponse;
    if (json.error) throw new Error(`Ollama GLM-OCR error: ${json.error}`);

    const text = (json.response ?? "").trim();
    if (!text) throw new Error("Ollama GLM-OCR returned empty text.");

    return {
      text,
      model: this.model,
      provider: this.name
    };
  }
}
