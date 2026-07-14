import type { ExtractedDocument, StoredChunk, TextChunk } from "../types.ts";
import { sha256Text } from "../utils/hash.ts";
import type { TurboQuantEncoded } from "../vector/turboquant/index.ts";
import type { LexicalSearchResult, SearchStorage, StorageStats, VectorSearchResult } from "./types.ts";

type PostgresSearchDatabaseOptions = {
  url: string;
  usePsqlBm25s: boolean;
  usePgvector: boolean;
  vectorIndex: boolean;
  embeddingDimensions: number;
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
  embedding: number[] | string | null;
  turbo_code: Uint8Array | ArrayBuffer | null;
  turbo_norm: number | string | null;
};

type PostgresLexicalChunkRow = PostgresChunkRow & {
  lexical_score: number | string;
};

type PostgresVectorChunkRow = PostgresChunkRow & {
  vector_score: number | string;
};

type PostgresHybridChunkRow = PostgresChunkRow & {
  hybrid_score: number | string;
};

export class PostgresSearchDatabase implements SearchStorage {
  private readonly sql: Bun.SQL;
  private readonly ready: Promise<void>;

  constructor(private readonly options: PostgresSearchDatabaseOptions) {
    if (!options.url) {
      throw new Error("TURBOSEARCH_POSTGRES_URL is required when --storage postgres is used.");
    }
    if (!Number.isInteger(options.embeddingDimensions) || options.embeddingDimensions <= 0) {
      throw new Error("Postgres pgvector storage requires a positive embedding dimension.");
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
        if (embedding.length !== this.options.embeddingDimensions) {
          throw new Error(
            `Embedding dimension ${embedding.length} does not match Postgres vector dimension ${this.options.embeddingDimensions}.`
          );
        }

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
            ${toVectorLiteral(embedding)}::vector,
            ${turboVector?.code ?? null},
            ${turboVector?.norm ?? null}
          )
        `;
      }
    });

    await this.ensureBm25Index();
  }

  async getDocumentHash(path: string): Promise<string | null> {
    await this.ready;
    const [row] = await this.sql<{ sha256: string }[]>`
      SELECT sha256 FROM turbosearch_documents WHERE path = ${path}
    `;
    return row?.sha256 ?? null;
  }

  async listDocumentPaths(): Promise<string[]> {
    await this.ready;
    const rows = await this.sql<{ path: string }[]>`
      SELECT path FROM turbosearch_documents ORDER BY path
    `;
    return rows.map((row) => row.path);
  }

  async deleteDocuments(paths: string[]): Promise<number> {
    await this.ready;
    if (paths.length === 0) return 0;
    let removed = 0;
    for (const path of paths) {
      const rows = await this.sql<{ id: number | string }[]>`
        DELETE FROM turbosearch_documents WHERE path = ${path} RETURNING id
      `;
      removed += rows.length;
    }
    return removed;
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
        turbosearch_chunks.embedding::text AS embedding,
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
    await this.ensureBm25Index();

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
        turbosearch_chunks.embedding::text AS embedding,
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

  async searchVector(queryVector: Float32Array, k: number): Promise<VectorSearchResult[]> {
    await this.ready;
    if (!this.options.usePgvector) return [];

    const rows = await this.sql<PostgresVectorChunkRow[]>`
      SELECT
        turbosearch_chunks.id,
        turbosearch_chunks.document_id,
        turbosearch_documents.path AS document_path,
        turbosearch_chunks.ordinal,
        turbosearch_chunks.content AS text,
        turbosearch_chunks.start_offset,
        turbosearch_chunks.end_offset,
        turbosearch_chunks.metadata_json,
        turbosearch_chunks.embedding::text AS embedding,
        turbosearch_chunks.turbo_code,
        turbosearch_chunks.turbo_norm,
        1 - (turbosearch_chunks.embedding <=> ${toVectorLiteral(queryVector)}::vector) AS vector_score
      FROM turbosearch_chunks
      INNER JOIN turbosearch_documents ON turbosearch_documents.id = turbosearch_chunks.document_id
      ORDER BY turbosearch_chunks.embedding <=> ${toVectorLiteral(queryVector)}::vector
      LIMIT ${k}
    `;

    return rows.map((row) => ({
      chunk: rowToStoredChunk(row),
      score: Number(row.vector_score)
    }));
  }

  async searchHybrid(
    query: string,
    queryVector: Float32Array,
    k: number,
    candidateLimit: number
  ): Promise<VectorSearchResult[]> {
    await this.ready;
    if (!this.options.usePsqlBm25s || !this.options.usePgvector) return [];
    await this.ensureBm25Index();

    const vectorLiteral = toVectorLiteral(queryVector);
    const rows = await this.sql<PostgresHybridChunkRow[]>`
      WITH vector_ranked AS (
        SELECT
          id,
          row_number() OVER (ORDER BY embedding <=> ${vectorLiteral}::vector) AS vector_rank
        FROM turbosearch_chunks
        ORDER BY embedding <=> ${vectorLiteral}::vector
        LIMIT ${candidateLimit}
      ),
      lexical_ranked AS (
        SELECT
          turbosearch_chunks.id,
          row_number() OVER (ORDER BY hit.score DESC, turbosearch_chunks.id ASC) AS lexical_rank
        FROM psql_bm25s_query('turbosearch_chunks_bm25_idx'::regclass, ${query}, ${candidateLimit}) AS hit
        INNER JOIN turbosearch_chunks ON turbosearch_chunks.ctid = hit.ctid
      ),
      fused AS (
        SELECT
          COALESCE(vector_ranked.id, lexical_ranked.id) AS id,
          COALESCE(0.55 / (60 + vector_ranked.vector_rank), 0) +
            COALESCE(0.45 / (60 + lexical_ranked.lexical_rank), 0) AS hybrid_score
        FROM vector_ranked
        FULL OUTER JOIN lexical_ranked ON lexical_ranked.id = vector_ranked.id
      )
      SELECT
        turbosearch_chunks.id,
        turbosearch_chunks.document_id,
        turbosearch_documents.path AS document_path,
        turbosearch_chunks.ordinal,
        turbosearch_chunks.content AS text,
        turbosearch_chunks.start_offset,
        turbosearch_chunks.end_offset,
        turbosearch_chunks.metadata_json,
        turbosearch_chunks.embedding::text AS embedding,
        turbosearch_chunks.turbo_code,
        turbosearch_chunks.turbo_norm,
        fused.hybrid_score
      FROM fused
      INNER JOIN turbosearch_chunks ON turbosearch_chunks.id = fused.id
      INNER JOIN turbosearch_documents ON turbosearch_documents.id = turbosearch_chunks.document_id
      ORDER BY fused.hybrid_score DESC, turbosearch_chunks.id ASC
      LIMIT ${k}
    `;

    return rows.map((row) => ({
      chunk: rowToStoredChunk(row),
      score: Number(row.hybrid_score)
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
    if (!this.options.usePgvector) {
      throw new Error("Postgres storage requires pgvector. Set TURBOSEARCH_POSTGRES_PGVECTOR=true and install the vector extension.");
    }
    const dimension = this.options.embeddingDimensions;

    try {
      await this.sql`CREATE EXTENSION IF NOT EXISTS vector`;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Postgres storage requires the pgvector extension for vector search: ${message}`);
    }

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

    await this.sql.unsafe(`
      CREATE TABLE IF NOT EXISTS turbosearch_chunks (
        id BIGSERIAL PRIMARY KEY,
        document_id BIGINT NOT NULL REFERENCES turbosearch_documents(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        content TEXT NOT NULL,
        start_offset INTEGER NOT NULL,
        end_offset INTEGER NOT NULL,
        metadata_json JSONB NOT NULL,
        embedding vector(${dimension}) NOT NULL,
        turbo_code BYTEA,
        turbo_norm DOUBLE PRECISION,
        UNIQUE(document_id, ordinal)
      )
    `);

    await this.ensureVectorColumn(dimension);

    await this.sql`
      CREATE INDEX IF NOT EXISTS turbosearch_chunks_document_id_idx
      ON turbosearch_chunks(document_id)
    `;

    if (this.options.vectorIndex) {
      await this.sql.unsafe(`
        CREATE INDEX IF NOT EXISTS turbosearch_chunks_embedding_hnsw_idx
        ON turbosearch_chunks USING hnsw (embedding vector_cosine_ops)
      `);
    }

    try {
      await this.sql`CREATE EXTENSION IF NOT EXISTS psql_bm25s`;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Postgres storage requires the psql_bm25s extension for BM25 search: ${message}`);
    }
  }

  private async ensureVectorColumn(dimension: number): Promise<void> {
    const [column] = await this.sql<{ column_type: string }[]>`
      SELECT format_type(pg_attribute.atttypid, pg_attribute.atttypmod) AS column_type
      FROM pg_attribute
      INNER JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
      INNER JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
      WHERE pg_namespace.nspname = current_schema()
        AND pg_class.relname = 'turbosearch_chunks'
        AND pg_attribute.attname = 'embedding'
        AND NOT pg_attribute.attisdropped
    `;
    const columnType = column?.column_type;
    const expected = `vector(${dimension})`;
    if (!columnType || columnType === expected) return;

    if (columnType === "real[]" || columnType === "double precision[]") {
      await this.sql.unsafe(`
        ALTER TABLE turbosearch_chunks
        ALTER COLUMN embedding TYPE vector(${dimension})
        USING ('[' || array_to_string(embedding, ',') || ']')::vector(${dimension})
      `);
      return;
    }

    // A populated vector column with a different dimension likely belongs to a different embedding model.
    // Re-indexing is safer than silently reshaping vectors and corrupting retrieval scores.
    throw new Error(
      `Existing Postgres embedding column is ${columnType}, expected ${expected}. Recreate the index or use the original embedding dimension.`
    );
  }

  private async ensureBm25Index(): Promise<void> {
    if (!this.options.usePsqlBm25s) return;

    const [existing] = await this.sql<{ exists: boolean }[]>`
      SELECT EXISTS (
        SELECT 1
        FROM pg_class
        WHERE oid = to_regclass('turbosearch_chunks_bm25_idx')
      ) AS exists
    `;
    if (existing?.exists) return;

    const [stats] = await this.sql<{ count: string | number }[]>`
      SELECT count(*) AS count FROM turbosearch_chunks
    `;
    if (Number(stats?.count ?? 0) === 0) return;

    try {
      // psql_bm25s v0.4.14 rejects scalar-text index builds on empty tables, so this is intentionally lazy.
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
    embedding: Float32Array.from(parseVector(row.embedding)),
    turboCode: toUint8Array(row.turbo_code),
    turboNorm: row.turbo_norm === null ? null : Number(row.turbo_norm)
  };
}

function parseJsonRecord(value: Record<string, unknown> | string): Record<string, unknown> {
  if (typeof value === "string") return JSON.parse(value) as Record<string, unknown>;
  return value;
}

function parseVector(value: number[] | string | null): number[] {
  if (value === null) return [];
  if (Array.isArray(value)) return value.map(Number);
  const trimmed = value.trim();
  if (!trimmed || trimmed === "{}" || trimmed === "[]") return [];
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    return trimmed
      .slice(1, -1)
      .split(",")
      .filter(Boolean)
      .map((item) => Number(item));
  }
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

function toVectorLiteral(vector: Float32Array): string {
  return `[${Array.from(vector, (value) => String(Number(value))).join(",")}]`;
}
