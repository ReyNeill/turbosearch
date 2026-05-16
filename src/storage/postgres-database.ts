import type { ExtractedDocument, StoredChunk, TextChunk } from "../types.ts";
import { sha256Text } from "../utils/hash.ts";
import type { TurboQuantEncoded } from "../vector/turboquant/index.ts";
import type { LexicalSearchResult, SearchStorage, StorageStats } from "./types.ts";

type PostgresSearchDatabaseOptions = {
  url: string;
  usePsqlBm25s: boolean;
};

type PostgresChunkRow = {
  id: number | string;
  document_id: number | string;
  document_path: string;
  ordinal: number;
  text: string;
  start_offset: number;
  end_offset: number;
  metadata_json: Record<string, unknown> | string;
  embedding: number[] | string;
  turbo_code: Uint8Array | ArrayBuffer | null;
  turbo_norm: number | string | null;
};

type PostgresLexicalChunkRow = PostgresChunkRow & {
  lexical_score: number | string;
};

export class PostgresSearchDatabase implements SearchStorage {
  private readonly sql: Bun.SQL;
  private readonly ready: Promise<void>;

  constructor(private readonly options: PostgresSearchDatabaseOptions) {
    if (!options.url) {
      throw new Error("TURBOSEARCH_POSTGRES_URL is required when --storage postgres is used.");
    }

    this.sql = new Bun.SQL(options.url);
    this.ready = this.migrate();
  }

  async close(): Promise<void> {
    await this.sql.close();
  }

  async replaceDocument(
    document: ExtractedDocument,
    chunks: TextChunk[],
    embeddings: Float32Array[],
    turboVectors: Array<TurboQuantEncoded | null>
  ): Promise<void> {
    await this.ready;
    if (chunks.length !== embeddings.length || chunks.length !== turboVectors.length) {
      throw new Error("Chunk, embedding, and turbo vector counts must match.");
    }

    const metadataJson = JSON.stringify(document.metadata);
    const hash = sha256Text(document.text);

    await this.sql.begin(async (sql) => {
      const [storedDocument] = await sql<{ id: number | string }[]>`
        INSERT INTO turbosearch_documents (path, kind, sha256, metadata_json, created_at, updated_at)
        VALUES (${document.path}, ${document.kind}, ${hash}, ${metadataJson}::jsonb, now(), now())
        ON CONFLICT (path) DO UPDATE SET
          kind = EXCLUDED.kind,
          sha256 = EXCLUDED.sha256,
          metadata_json = EXCLUDED.metadata_json,
          updated_at = now()
        RETURNING id
      `;
      if (!storedDocument) throw new Error("Postgres did not return an indexed document id.");

      await sql`
        DELETE FROM turbosearch_chunks
        WHERE document_id = ${storedDocument.id}
      `;

      for (let i = 0; i < chunks.length; i += 1) {
        const chunk = chunks[i];
        const embedding = embeddings[i];
        const turboVector = turboVectors[i] ?? null;
        if (!chunk || !embedding) continue;

        await sql`
          INSERT INTO turbosearch_chunks (
            document_id,
            ordinal,
            content,
            start_offset,
            end_offset,
            metadata_json,
            embedding,
            turbo_code,
            turbo_norm
          ) VALUES (
            ${storedDocument.id},
            ${chunk.ordinal},
            ${chunk.text},
            ${chunk.startOffset},
            ${chunk.endOffset},
            ${JSON.stringify(chunk.metadata)}::jsonb,
            ${sql.array(Array.from(embedding), "real")},
            ${turboVector?.code ?? null},
            ${turboVector?.norm ?? null}
          )
        `;
      }
    });
  }

  async getAllChunks(): Promise<StoredChunk[]> {
    await this.ready;
    const rows = await this.sql<PostgresChunkRow[]>`
      SELECT
        turbosearch_chunks.id,
        turbosearch_chunks.document_id,
        turbosearch_documents.path AS document_path,
        turbosearch_chunks.ordinal,
        turbosearch_chunks.content AS text,
        turbosearch_chunks.start_offset,
        turbosearch_chunks.end_offset,
        turbosearch_chunks.metadata_json,
        turbosearch_chunks.embedding,
        turbosearch_chunks.turbo_code,
        turbosearch_chunks.turbo_norm
      FROM turbosearch_chunks
      INNER JOIN turbosearch_documents ON turbosearch_documents.id = turbosearch_chunks.document_id
      ORDER BY turbosearch_chunks.id ASC
    `;

    return rows.map(rowToStoredChunk);
  }

  async searchLexical(query: string, k: number): Promise<LexicalSearchResult[]> {
    await this.ready;
    if (!this.options.usePsqlBm25s) return [];

    const rows = await this.sql<PostgresLexicalChunkRow[]>`
      SELECT
        turbosearch_chunks.id,
        turbosearch_chunks.document_id,
        turbosearch_documents.path AS document_path,
        turbosearch_chunks.ordinal,
        turbosearch_chunks.content AS text,
        turbosearch_chunks.start_offset,
        turbosearch_chunks.end_offset,
        turbosearch_chunks.metadata_json,
        turbosearch_chunks.embedding,
        turbosearch_chunks.turbo_code,
        turbosearch_chunks.turbo_norm,
        hit.score AS lexical_score
      FROM psql_bm25s_query('turbosearch_chunks_bm25_idx'::regclass, ${query}, ${k}) AS hit
      INNER JOIN turbosearch_chunks ON turbosearch_chunks.ctid = hit.ctid
      INNER JOIN turbosearch_documents ON turbosearch_documents.id = turbosearch_chunks.document_id
      ORDER BY hit.score DESC, turbosearch_chunks.id ASC
    `;

    return rows.map((row) => ({
      chunk: rowToStoredChunk(row),
      score: Number(row.lexical_score)
    }));
  }

  async getStats(): Promise<StorageStats> {
    await this.ready;
    const [documents] = await this.sql<{ count: string | number }[]>`
      SELECT count(*) AS count FROM turbosearch_documents
    `;
    const [chunks] = await this.sql<{ count: string | number }[]>`
      SELECT count(*) AS count FROM turbosearch_chunks
    `;

    return {
      documents: Number(documents?.count ?? 0),
      chunks: Number(chunks?.count ?? 0)
    };
  }

  private async migrate(): Promise<void> {
    await this.sql`
      CREATE TABLE IF NOT EXISTS turbosearch_documents (
        id BIGSERIAL PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        metadata_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
      )
    `;

    // Keep vectors as REAL[] until a pgvector/VectorChord adapter exists; psql_bm25s owns lexical retrieval in this backend.
    await this.sql`
      CREATE TABLE IF NOT EXISTS turbosearch_chunks (
        id BIGSERIAL PRIMARY KEY,
        document_id BIGINT NOT NULL REFERENCES turbosearch_documents(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        content TEXT NOT NULL,
        start_offset INTEGER NOT NULL,
        end_offset INTEGER NOT NULL,
        metadata_json JSONB NOT NULL,
        embedding REAL[] NOT NULL,
        turbo_code BYTEA,
        turbo_norm DOUBLE PRECISION,
        UNIQUE(document_id, ordinal)
      )
    `;

    await this.sql`
      CREATE INDEX IF NOT EXISTS turbosearch_chunks_document_id_idx
      ON turbosearch_chunks(document_id)
    `;

    if (!this.options.usePsqlBm25s) return;

    try {
      await this.sql`CREATE EXTENSION IF NOT EXISTS psql_bm25s`;
      await this.sql`
        CREATE INDEX IF NOT EXISTS turbosearch_chunks_bm25_idx
        ON turbosearch_chunks USING psql_bm25s (content)
      `;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Postgres storage requires the psql_bm25s extension for BM25 search: ${message}`);
    }
  }
}

function rowToStoredChunk(row: PostgresChunkRow): StoredChunk {
  return {
    id: Number(row.id),
    documentId: Number(row.document_id),
    documentPath: row.document_path,
    ordinal: row.ordinal,
    text: row.text,
    startOffset: row.start_offset,
    endOffset: row.end_offset,
    metadata: parseJsonRecord(row.metadata_json),
    embedding: Float32Array.from(parseRealArray(row.embedding)),
    turboCode: toUint8Array(row.turbo_code),
    turboNorm: row.turbo_norm === null ? null : Number(row.turbo_norm)
  };
}

function parseJsonRecord(value: Record<string, unknown> | string): Record<string, unknown> {
  if (typeof value === "string") return JSON.parse(value) as Record<string, unknown>;
  return value;
}

function parseRealArray(value: number[] | string): number[] {
  if (Array.isArray(value)) return value.map(Number);
  const trimmed = value.trim();
  if (!trimmed || trimmed === "{}") return [];
  return trimmed
    .slice(1, -1)
    .split(",")
    .map((item) => Number(item));
}

function toUint8Array(value: Uint8Array | ArrayBuffer | null): Uint8Array | null {
  if (value === null) return null;
  if (value instanceof Uint8Array) return value;
  return new Uint8Array(value);
}
