import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { expect, test } from "bun:test";
import { SqliteSearchDatabase } from "../src/storage/sqlite-database.ts";

test("SQLite storage keeps FTS BM25 index in sync with replaced documents", () => {
  const dir = mkdtempSync(join(tmpdir(), "turbosearch-sqlite-"));
  const db = new SqliteSearchDatabase(join(dir, "index.sqlite"));

  try {
    db.replaceDocument(
      { path: "sample.txt", kind: "text", text: "alpha keyword", metadata: {} },
      [
        {
          documentPath: "sample.txt",
          ordinal: 0,
          text: "alpha keyword",
          startOffset: 0,
          endOffset: 13,
          metadata: {}
        }
      ],
      [new Float32Array([1, 0])],
      [null]
    );

    expect(db.searchLexical("keyword", 5)[0]?.chunk.text).toBe("alpha keyword");

    db.replaceDocument(
      { path: "sample.txt", kind: "text", text: "beta replacement", metadata: {} },
      [
        {
          documentPath: "sample.txt",
          ordinal: 0,
          text: "beta replacement",
          startOffset: 0,
          endOffset: 16,
          metadata: {}
        }
      ],
      [new Float32Array([0, 1])],
      [null]
    );

    expect(db.searchLexical("keyword", 5)).toHaveLength(0);
    expect(db.searchLexical("replacement", 5)[0]?.chunk.text).toBe("beta replacement");
  } finally {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
