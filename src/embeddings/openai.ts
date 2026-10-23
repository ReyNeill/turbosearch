import { textDocuments, type EmbeddingDocument, type EmbeddingProvider } from "./provider.ts";
import { formatDocument } from "./prompts.ts";
import { l2Normalize } from "../vector/math.ts";

type OpenAiEmbeddingOptions = {
  model: string;
  dimensions: number;
};

type OpenAiEmbeddingResponse = {
  data: Array<{ embedding: number[] }>;
};

export class OpenAiEmbeddingProvider implements EmbeddingProvider {
  readonly name = "openai" as const;
  readonly supportsImages = false;
  readonly dimensions: number;
  private readonly model: string;

  constructor(options: OpenAiEmbeddingOptions) {
    this.model = options.model;
    this.dimensions = options.dimensions;
  }

  async embedQuery(query: string): Promise<Float32Array> {
    const [embedding] = await this.embedTexts([query]);
    if (!embedding) throw new Error("OpenAI returned no embedding.");
    return embedding;
  }

  async embedDocuments(documents: EmbeddingDocument[]): Promise<Float32Array[]> {
    return this.embedTexts(textDocuments(this.name, documents).map((document) => formatDocument("none", document)));
  }

  private async embedTexts(texts: string[]): Promise<Float32Array[]> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OPENAI_API_KEY is required when using --embedding openai.");
    }

    const response = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: this.model,
        input: texts,
        dimensions: this.dimensions
      })
    });

    if (!response.ok) {
      throw new Error(`OpenAI embeddings request failed: ${response.status} ${await response.text()}`);
    }

    const json = await response.json() as OpenAiEmbeddingResponse;
    return json.data.map((item) => l2Normalize(Float32Array.from(item.embedding)));
  }
}
