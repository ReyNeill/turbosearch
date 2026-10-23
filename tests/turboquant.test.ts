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

test("byte-table scoring matches per-dimension scoring at every byte-aligned width", () => {
  let seed = 7;
  const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5;
  const vector = (dimension: number) => Float32Array.from({ length: dimension }, random);

  for (const bits of [1, 2, 4, 8]) {
    const codec = new TurboQuantCodec({ dimension: 768, bits });
    const query = codec.prepareQuery(vector(768));
    expect(query.byteTable).not.toBeNull();
    for (let i = 0; i < 5; i += 1) {
      const encoded = codec.encode(vector(768));
      const reference = codec.scorePreparedQuery({ rotated: query.rotated, byteTable: null }, encoded);
      expect(codec.scorePreparedQuery(query, encoded)).toBeCloseTo(reference, 4);
    }
  }
});
