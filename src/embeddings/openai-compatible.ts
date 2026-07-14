import { l2Normalize } from "../vector/math.ts";
import type { EmbeddingProvider } from "./provider.ts";
import { createSageMakerBearerToken } from "./sagemaker-token.ts";

type CompatibleEmbeddingOptions = {
  baseUrl: string;
  apiKey?: string;
  auth?: "none" | "bearer" | "sagemaker";
  awsRegion?: string;
  awsProfile?: string;
  model: string;
  queryInstruction: string;
  dimensions: number;
  fetch?: typeof fetch;
};

type CompatibleEmbeddingResponse = {
  data?: Array<{ index?: number; embedding?: number[] }>;
};

export class OpenAiCompatibleEmbeddingProvider implements EmbeddingProvider {
  readonly name = "openai-compatible" as const;
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

  async embedText(text: string): Promise<Float32Array> {
    const query = this.options.queryInstruction.trim()
      ? `Instruct: ${this.options.queryInstruction.trim()}\nQuery:${text}`
      : text;
    const [embedding] = await this.embed([query]);
    if (!embedding) throw new Error("The embedding service returned no query embedding.");
    return embedding;
  }

  async embedTexts(texts: string[]): Promise<Float32Array[]> {
    return this.embed(texts);
  }

  private async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const token = await this.bearerToken();
    const response = await this.request(`${this.options.baseUrl.replace(/\/$/, "")}/embeddings`, {
      method: "POST",
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: this.options.model,
        input: texts,
        dimensions: this.dimensions,
        encoding_format: "float"
      })
    });

    if (!response.ok) {
      const detail = (await response.text()).slice(0, 1000);
      throw new Error(`OpenAI-compatible embeddings request failed: ${response.status} ${detail}`);
    }

    const json = await response.json() as CompatibleEmbeddingResponse;
    if (!Array.isArray(json.data) || json.data.length !== texts.length) {
      throw new Error(`Embedding service returned ${json.data?.length ?? 0} vectors for ${texts.length} inputs.`);
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
