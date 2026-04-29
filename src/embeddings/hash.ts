import type { EmbeddingProvider } from "./provider.ts";
import { hashToUint32 } from "../utils/hash.ts";
import { l2Normalize } from "../vector/math.ts";

export class HashEmbeddingProvider implements EmbeddingProvider {
  readonly name = "hash" as const;

  constructor(readonly dimensions: number) {}

  async embedText(text: string): Promise<Float32Array> {
    return this.embed(text);
  }

  async embedTexts(texts: string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.embed(text));
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
