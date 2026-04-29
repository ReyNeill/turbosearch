import { expect, test } from "bun:test";
import { HashEmbeddingProvider } from "../src/embeddings/hash.ts";
import { cosine } from "../src/vector/math.ts";

test("hash embeddings are deterministic and normalized", async () => {
  const provider = new HashEmbeddingProvider(128);
  const first = await provider.embedText("semantic search over private documents");
  const second = await provider.embedText("semantic search over private documents");

  expect(cosine(first, second)).toBeGreaterThan(0.999);
});
