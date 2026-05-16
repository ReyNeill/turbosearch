import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { Database as BunDatabase } from "bun:sqlite";
import type { ExtractedDocument, StoredChunk, TextChunk } from "../types.ts";
import { sha256Text } from "../utils/hash.ts";
import { bytesToFloat32, float32ToBytes } from "../vector/math.ts";
import type { TurboQuantEncoded } from "../vector/turboquant/index.ts";
import type { LexicalSearchResult, SearchStorage } from "./types.ts";

type ChunkRow = {
  id: number;
  document_id: number;
  document_path: string;
  ordinal: number;
  text: string;
  start_offset: number;
  end_offset: number;
  metadata_json: string;
  embedding: Uint8Array;
  turbo_code: Uint8Array | null;
  turbo_norm: number | null;
};

type StatsRow = {
  documents: number;
  chunks: number;
};

type LexicalChunkRow = ChunkRow & {
  lexical_score: number;
};

export class SqliteSearchDatabase implements SearchStorage {
  private readonly db: BunDatabase;

  constructor(readonly path: string) {
    mkdirSync(dirname(path), { recursive: true });
    this.db = new BunDatabase(path);
    this.db.exec("PRAGMA journal_mode = WAL;");
    this.db.exec("PRAGMA busy_timeout = 5000;");
    this.migrate();
  }

  close(): void {
    this.db.close();
  }

  replaceDocument(
    document: ExtractedDocument,
    chunks: TextChunk[],
    embeddings: Float32Array[],
    turboVectors: Array<TurboQuantEncoded | null>
  ): void {
    if (chunks.length !== embeddings.length || chunks.length !== turboVectors.length) {
      throw new Error("Chunk, embedding, and turbo vector counts must match.");
    }

    const metadataJson = JSON.stringify(document.metadata);
    const hash = sha256Text(document.text);

    const transaction = this.db.transaction(() => {
      const existing = this.db.query<{ id: number }, [string]>("SELECT id FROM documents WHERE path = ?").get(document.path);
      if (existing) {
        this.deleteChunkFtsRows(existing.id);
        this.db.query("DELETE FROM chunks WHERE document_id = ?").run(existing.id);
        this.db.query("UPDATE documents SET kind = ?, sha256 = ?, metadata_json = ?, updated_at = unixepoch() WHERE id = ?")
          .run(document.kind, hash, metadataJson, existing.id);
        this.insertChunks(existing.id, chunks, embeddings, turboVectors);
        return;
      }

      const result = this.db.query("INSERT INTO documents (path, kind, sha256, metadata_json, created_at, updated_at) VALUES (?, ?, ?, ?, unixepoch(), unixepoch())")
        .run(document.path, document.kind, hash, metadataJson);
      this.insertChunks(Number(result.lastInsertRowid), chunks, embeddings, turboVectors);
    });

    transaction();
  }

  getAllChunks(): StoredChunk[] {
    const rows = this.db.query<ChunkRow, []>(`
      SELECT
        chunks.id,
        chunks.document_id,
        documents.path AS document_path,
        chunks.ordinal,
        chunks.text,
        chunks.start_offset,
        chunks.end_offset,
        chunks.metadata_json,
        chunks.embedding,
        chunks.turbo_code,
        chunks.turbo_norm
      FROM chunks
      INNER JOIN documents ON documents.id = chunks.document_id
      ORDER BY chunks.id ASC
    `).all();

    return rows.map(rowToStoredChunk);
  }

  searchLexical(query: string, k: number): LexicalSearchResult[] {
    const ftsQuery = toFtsQuery(query);
    if (!ftsQuery) return [];

    const rows = this.db.query<LexicalChunkRow, [string, number]>(`
      SELECT
        chunks.id,
        chunks.document_id,
        documents.path AS document_path,
        chunks.ordinal,
        chunks.text,
        chunks.start_offset,
        chunks.end_offset,
        chunks.metadata_json,
        chunks.embedding,
        chunks.turbo_code,
        chunks.turbo_norm,
        -bm25(chunks_fts) AS lexical_score
      FROM chunks_fts
      INNER JOIN chunks ON chunks.id = chunks_fts.rowid
      INNER JOIN documents ON documents.id = chunks.document_id
      WHERE chunks_fts MATCH ?
      ORDER BY bm25(chunks_fts) ASC
      LIMIT ?
    `).all(ftsQuery, k);

    return rows.map((row) => ({
      chunk: rowToStoredChunk(row),
      score: row.lexical_score
    }));
  }

  getStats(): StatsRow {
    const documents = this.db.query<{ count: number }, []>("SELECT count(*) AS count FROM documents").get()?.count ?? 0;
    const chunks = this.db.query<{ count: number }, []>("SELECT count(*) AS count FROM chunks").get()?.count ?? 0;
    return { documents, chunks };
  }

  private insertChunks(
    documentId: number,
    chunks: TextChunk[],
    embeddings: Float32Array[],
    turboVectors: Array<TurboQuantEncoded | null>
  ): void {
    const insert = this.db.query(`
      INSERT INTO chunks (
        document_id,
        ordinal,
        text,
        start_offset,
        end_offset,
        metadata_json,
        embedding,
        turbo_code,
        turbo_norm
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (let i = 0; i < chunks.length; i += 1) {
      const chunk = chunks[i];
      const embedding = embeddings[i];
      const turboVector = turboVectors[i] ?? null;
      if (!chunk || !embedding) continue;

      const chunkId = Number(insert.run(
        documentId,
        chunk.ordinal,
        chunk.text,
        chunk.startOffset,
        chunk.endOffset,
        JSON.stringify(chunk.metadata),
        float32ToBytes(embedding),
        turboVector?.code ?? null,
        turboVector?.norm ?? null
      ).lastInsertRowid);
      this.db.query("INSERT INTO chunks_fts (rowid, text, chunk_id) VALUES (?, ?, ?)").run(chunkId, chunk.text, chunkId);
    }
  }

  private deleteChunkFtsRows(documentId: number): void {
    this.db.query(`
      DELETE FROM chunks_fts
      WHERE rowid IN (SELECT id FROM chunks WHERE document_id = ?)
    `).run(documentId);
  }

  private migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        text TEXT NOT NULL,
        start_offset INTEGER NOT NULL,
        end_offset INTEGER NOT NULL,
        metadata_json TEXT NOT NULL,
        embedding BLOB NOT NULL,
        turbo_code BLOB,
        turbo_norm REAL,
        UNIQUE(document_id, ordinal)
      );

      CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);

      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text,
        chunk_id UNINDEXED
      );
    `);
    this.backfillFts();
  }

  private backfillFts(): void {
    this.db.exec(`
      INSERT INTO chunks_fts (rowid, text, chunk_id)
      SELECT chunks.id, chunks.text, chunks.id
      FROM chunks
      WHERE NOT EXISTS (
        SELECT 1
        FROM chunks_fts
        WHERE chunks_fts.rowid = chunks.id
      );
    `);
  }
}

function rowToStoredChunk(row: ChunkRow): StoredChunk {
  return {
    id: row.id,
    documentId: row.document_id,
    documentPath: row.document_path,
    ordinal: row.ordinal,
    text: row.text,
    startOffset: row.start_offset,
    endOffset: row.end_offset,
    metadata: JSON.parse(row.metadata_json) as Record<string, unknown>,
    embedding: bytesToFloat32(row.embedding),
    turboCode: row.turbo_code,
    turboNorm: row.turbo_norm
  };
}

function toFtsQuery(query: string): string {
  const terms = query.toLowerCase().match(/[\p{L}\p{N}_'-]+/gu) ?? [];
  return terms.map((term) => `"${term.replaceAll("\"", "\"\"")}"`).join(" OR ");
}
