import { readFile } from "node:fs/promises";
import { extensionOf } from "../../utils/files.ts";
import type { OcrMode } from "../../types.ts";
import { promptForOcrMode, type OcrProvider, type OcrRequest, type OcrResponse } from "../provider.ts";

type VllmGlmOcrOptions = {
  baseUrl: string;
  apiKey: string;
  model: string;
  defaultMode: OcrMode;
  defaultPrompt: string;
  maxOutputTokens: number;
};

type ChatCompletionResponse = {
  choices?: Array<{
    message?: {
      content?: string | Array<{ type?: string; text?: string }>;
    };
  }>;
  error?: { message?: string };
};

export class VllmGlmOcrProvider implements OcrProvider {
  readonly name = "vllm-glm-ocr" as const;
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly defaultMode: OcrMode;
  private readonly defaultPrompt: string;
  private readonly maxOutputTokens: number;

  constructor(options: VllmGlmOcrOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.model = options.model;
    this.defaultMode = options.defaultMode;
    this.defaultPrompt = options.defaultPrompt;
    this.maxOutputTokens = options.maxOutputTokens;
  }

  async extractText(request: OcrRequest): Promise<OcrResponse> {
    const image = await readFile(request.imagePath);
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {})
      },
      body: JSON.stringify({
        model: this.model,
        temperature: 0,
        max_tokens: this.maxOutputTokens,
        messages: [{
          role: "user",
          content: [
            { type: "text", text: request.prompt || this.defaultPrompt || promptForOcrMode(request.mode ?? this.defaultMode) },
            {
              type: "image_url",
              image_url: {
                url: `data:${mimeTypeForImage(request.imagePath)};base64,${image.toString("base64")}`
              }
            }
          ]
        }]
      })
    });

    if (!response.ok) {
      throw new Error(`vLLM GLM-OCR request failed: ${response.status} ${await response.text()}`);
    }

    const json = await response.json() as ChatCompletionResponse;
    if (json.error?.message) throw new Error(`vLLM GLM-OCR error: ${json.error.message}`);

    const text = extractTextContent(json).trim();
    if (!text) throw new Error("vLLM GLM-OCR returned empty text.");

    return {
      text,
      model: this.model,
      provider: this.name
    };
  }
}

function extractTextContent(response: ChatCompletionResponse): string {
  const content = response.choices?.[0]?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => part.text ?? "")
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function mimeTypeForImage(path: string): string {
  const extension = extensionOf(path);
  if (extension === ".jpg" || extension === ".jpeg") return "image/jpeg";
  if (extension === ".webp") return "image/webp";
  if (extension === ".gif") return "image/gif";
  if (extension === ".bmp") return "image/bmp";
  if (extension === ".tiff") return "image/tiff";
  return "image/png";
}
