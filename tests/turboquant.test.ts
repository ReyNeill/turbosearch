import { expect, test } from "bun:test";
import { TurboQuantCodec } from "../src/vector/turboquant/index.ts";

test("TurboQuant codec produces finite self scores", () => {
  const codec = new TurboQuantCodec({ dimension: 8, bits: 4 });
  const vector = new Float32Array([0.5, 0.2, -0.1, 0.3, 0.4, -0.2, 0.7, 0.1]);
  const encoded = codec.encode(vector);
  const query = codec.prepareQuery(vector);
  const score = codec.scorePreparedQuery(query, encoded);

  expect(Number.isFinite(score)).toBe(true);
  expect(encoded.code.length).toBe(4);
});
