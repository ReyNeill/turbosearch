import { textDocuments, type EmbeddingDocument, type EmbeddingProvider } from "./provider.ts";
import { formatDocument } from "./prompts.ts";
import { hashToUint32 } from "../utils/hash.ts";
import { l2Normalize } from "../vector/math.ts";

export class HashEmbeddingProvider implements EmbeddingProvider {
  readonly name = "hash" as const;
  readonly supportsImages = false;

  constructor(readonly dimensions: number) {}

  async embedQuery(query: string): Promise<Float32Array> {
    return this.embed(query);
  }

  async embedDocuments(documents: EmbeddingDocument[]): Promise<Float32Array[]> {
    return textDocuments(this.name, documents).map((document) => this.embed(formatDocument("none", document)));
  }

  private embed(text: string): Float32Array {
    const vector = new Float32Array(this.dimensions);
    const tokens = text.toLowerCase().match(/[a-z0-9_'-]+/g) ?? [];

    for (const token of tokens) {
      this.addFeature(vector, token, 1);

      for (let i = 0; i < token.length - 2; i += 1) {
        this.addFeature(vector, token.slice(i, i + 3), 0.35);
      }
    }

    return l2Normalize(vector);
  }

  private addFeature(vector: Float32Array, feature: string, weight: number): void {
    const hash = hashToUint32(feature);
    const index = hash % this.dimensions;
    const sign = (hash & 1) === 0 ? 1 : -1;
    vector[index] += sign * weight;
  }
}
