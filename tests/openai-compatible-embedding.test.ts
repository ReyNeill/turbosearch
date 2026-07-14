import { expect, test } from "bun:test";
import { OpenAiCompatibleEmbeddingProvider } from "../src/embeddings/openai-compatible.ts";

test("adds the Qwen query instruction only to queries and requests Matryoshka dimensions", async () => {
  const requests: Array<Record<string, unknown>> = [];
  const provider = new OpenAiCompatibleEmbeddingProvider({
    baseUrl: "http://embedding.internal/v1/",
    model: "unsloth/Qwen3-Embedding-4B",
    queryInstruction: "Retrieve passages that answer the org question",
    dimensions: 3,
    fetch: mockFetch(requests, [
      { index: 0, embedding: [3, 4, 0, 99] },
      { index: 1, embedding: [0, 0, 5, 99] }
    ])
  });

  const query = await provider.embedText("How do releases work?");
  const documents = await provider.embedTexts(["Document A", "Document B"]);

  expect(requests[0]?.input).toEqual([
    "Instruct: Retrieve passages that answer the org question\nQuery:How do releases work?"
  ]);
  expect(requests[1]?.input).toEqual(["Document A", "Document B"]);
  expect(requests[0]?.dimensions).toBe(3);
  expect(Array.from(query)).toEqual([0.6000000238418579, 0.800000011920929, 0]);
  expect(Array.from(documents[1] ?? [])).toEqual([0, 0, 1]);
});

test("orders indexed responses and rejects incomplete vectors", async () => {
  const provider = new OpenAiCompatibleEmbeddingProvider({
    baseUrl: "http://embedding.internal/v1",
    model: "unsloth/Qwen3-Embedding-4B",
    queryInstruction: "",
    dimensions: 32,
    fetch: mockFetch([], [
      { index: 1, embedding: unitVector(32, 1) },
      { index: 0, embedding: unitVector(32, 0) }
    ])
  });
  const vectors = await provider.embedTexts(["first", "second"]);
  expect(vectors[0]?.[0]).toBe(1);
  expect(vectors[1]?.[1]).toBe(1);
});

function mockFetch(
  requests: Array<Record<string, unknown>>,
  data: Array<{ index: number; embedding: number[] }>
): typeof fetch {
  return (async (_input: string | URL | Request, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    requests.push(body);
    const inputCount = Array.isArray(body.input) ? body.input.length : 0;
    return new Response(JSON.stringify({ data: data.slice(0, inputCount) }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  }) as typeof fetch;
}

function unitVector(dimensions: number, active: number): number[] {
  return Array.from({ length: dimensions }, (_, index) => index === active ? 1 : 0);
}
