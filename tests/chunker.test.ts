import { expect, test } from "bun:test";
import { chunkDocument } from "../src/chunking/chunker.ts";

test("chunkDocument creates overlapping chunks", () => {
  const chunks = chunkDocument(
    {
      path: "sample.txt",
      kind: "text",
      text: "one two three four five six seven eight nine ten",
      metadata: { source: "test" }
    },
    { targetTokens: 4, overlapTokens: 1 }
  );

  expect(chunks).toHaveLength(3);
  expect(chunks[0]?.text).toContain("one");
  expect(chunks[1]?.text.startsWith("four")).toBe(true);
});
