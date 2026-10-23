import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import type { EmbeddingPromptFormat } from "../types.ts";
import { l2Normalize } from "../vector/math.ts";
import type { EmbeddingDocument, EmbeddingProvider } from "./provider.ts";
import { formatDocument, formatQuery } from "./prompts.ts";
import { createSageMakerBearerToken } from "./sagemaker-token.ts";

type CompatibleEmbeddingOptions = {
  baseUrl: string;
  apiKey?: string;
  auth?: "none" | "bearer" | "sagemaker";
  awsRegion?: string;
  awsProfile?: string;
  model: string;
  promptFormat: EmbeddingPromptFormat;
  queryInstruction: string;
  dimensions: number;
  fetch?: typeof fetch;
};

type CompatibleEmbeddingResponse = {
  data?: Array<{ index?: number; embedding?: number[] }>;
};

// One /embeddings input: a plain string, or the OpenAI-style content array that multimodal
// servers (llama-server with --mmproj) read images from.
type CompatibleEmbeddingInput =
  | string
  | { content: Array<{ type: "image_url"; image_url: { url: string } }> };

const imageMimeTypes: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg"
};

export class OpenAiCompatibleEmbeddingProvider implements EmbeddingProvider {
  readonly name = "openai-compatible" as const;
  // The server decides; a text-only server rejects the request with its own error.
  readonly supportsImages = true;
  readonly dimensions: number;
  private readonly options: CompatibleEmbeddingOptions;
  private readonly request: typeof fetch;

  constructor(options: CompatibleEmbeddingOptions) {
    if (!Number.isInteger(options.dimensions) || options.dimensions < 1 || options.dimensions > 2560) {
      throw new Error("Embedding dimensions must be an integer from 1 to 2560.");
    }
    this.options = options;
    this.dimensions = options.dimensions;
    this.request = options.fetch ?? fetch;
  }

  async embedQuery(query: string): Promise<Float32Array> {
    const [embedding] = await this.embed([
      formatQuery(this.options.promptFormat, query, this.options.queryInstruction)
    ]);
    if (!embedding) throw new Error("The embedding service returned no query embedding.");
    return embedding;
  }

  async embedDocuments(documents: EmbeddingDocument[]): Promise<Float32Array[]> {
    return this.embed(await Promise.all(documents.map((document) => this.input(document))));
  }

  private async input(document: EmbeddingDocument): Promise<CompatibleEmbeddingInput> {
    if (document.kind === "text") return formatDocument(this.options.promptFormat, document);
    const mimeType = imageMimeTypes[extname(document.path).toLowerCase()];
    if (!mimeType) throw new Error(`Image embeddings need a PNG or JPEG file: ${document.path}`);
    const data = (await readFile(document.path)).toString("base64");
    // Images carry no prompt: EmbeddingGemma documents its task prefixes for text only.
    return { content: [{ type: "image_url", image_url: { url: `data:${mimeType};base64,${data}` } }] };
  }

  private async embed(inputs: CompatibleEmbeddingInput[]): Promise<Float32Array[]> {
    if (inputs.length === 0) return [];
    const token = await this.bearerToken();
    const response = await this.request(`${this.options.baseUrl.replace(/\/$/, "")}/embeddings`, {
      method: "POST",
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: this.options.model,
        input: inputs,
        dimensions: this.dimensions,
        encoding_format: "float"
      })
    });

    if (!response.ok) {
      const detail = (await response.text()).slice(0, 1000);
      throw new Error(`OpenAI-compatible embeddings request failed: ${response.status} ${detail}`);
    }

    const json = await response.json() as CompatibleEmbeddingResponse;
    if (!Array.isArray(json.data) || json.data.length !== inputs.length) {
      throw new Error(`Embedding service returned ${json.data?.length ?? 0} vectors for ${inputs.length} inputs.`);
    }
    return json.data
      .map((item, responseIndex) => ({ item, responseIndex }))
      .sort((left, right) => (left.item.index ?? left.responseIndex) - (right.item.index ?? right.responseIndex))
      .map(({ item }) => this.normalize(item.embedding));
  }

  private normalize(values: number[] | undefined): Float32Array {
    if (!values || values.length < this.dimensions) {
      throw new Error(`Embedding service returned dimension ${values?.length ?? 0}; expected at least ${this.dimensions}.`);
    }
    const selected = values.slice(0, this.dimensions);
    if (!selected.every(Number.isFinite)) throw new Error("Embedding service returned a non-finite value.");
    return l2Normalize(Float32Array.from(selected));
  }

  private async bearerToken(): Promise<string> {
    if (this.options.auth === "sagemaker") {
      return createSageMakerBearerToken({
        region: this.options.awsRegion ?? "eu-west-2",
        profile: this.options.awsProfile || undefined,
        expiresIn: 300
      });
    }
    if (this.options.auth === "bearer") {
      if (!this.options.apiKey) throw new Error("TURBOSEARCH_EMBEDDING_API_KEY is required for bearer auth.");
      return this.options.apiKey;
    }
    return "";
  }
}
