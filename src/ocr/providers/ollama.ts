import { readFile } from "node:fs/promises";
import type { OcrMode } from "../../types.ts";
import { promptForOcrMode, type OcrProvider, type OcrRequest, type OcrResponse } from "../provider.ts";

type OllamaGlmOcrOptions = {
  baseUrl: string;
  model: string;
  defaultMode: OcrMode;
  defaultPrompt: string;
  contextTokens: number;
  maxOutputTokens: number;
};

type OllamaGenerateResponse = {
  response?: string;
  error?: string;
};

export class OllamaGlmOcrProvider implements OcrProvider {
  readonly name = "ollama-glm-ocr" as const;
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly defaultMode: OcrMode;
  private readonly defaultPrompt: string;
  private readonly contextTokens: number;
  private readonly maxOutputTokens: number;

  constructor(options: OllamaGlmOcrOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.model = options.model;
    this.defaultMode = options.defaultMode;
    this.defaultPrompt = options.defaultPrompt;
    this.contextTokens = options.contextTokens;
    this.maxOutputTokens = options.maxOutputTokens;
  }

  async extractText(request: OcrRequest): Promise<OcrResponse> {
    const image = await readFile(request.imagePath);
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        prompt: request.prompt || this.defaultPrompt || promptForOcrMode(request.mode ?? this.defaultMode),
        images: [image.toString("base64")],
        stream: false,
        options: {
          temperature: 0,
          num_ctx: this.contextTokens,
          num_predict: this.maxOutputTokens
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
