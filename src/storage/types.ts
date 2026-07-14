import type { ExtractedDocument, StoredChunk, TextChunk } from "../types.ts";
import type { TurboQuantEncoded } from "../vector/turboquant/index.ts";

export type StorageStats = {
  documents: number;
  chunks: number;
};

export type LexicalSearchResult = {
  chunk: StoredChunk;
  score: number;
};

export type VectorSearchResult = {
  chunk: StoredChunk;
  score: number;
};

export interface SearchStorage {
  close(): void | Promise<void>;
  replaceDocument(
    document: ExtractedDocument,
    chunks: TextChunk[],
    embeddings: Float32Array[],
    turboVectors: Array<TurboQuantEncoded | null>
  ): void | Promise<void>;
  getDocumentHash(path: string): string | null | Promise<string | null>;
  listDocumentPaths(): string[] | Promise<string[]>;
  deleteDocuments(paths: string[]): number | Promise<number>;
  getAllChunks(): StoredChunk[] | Promise<StoredChunk[]>;
  searchLexical(query: string, k: number): LexicalSearchResult[] | Promise<LexicalSearchResult[]>;
  searchVector?(queryVector: Float32Array, k: number): VectorSearchResult[] | Promise<VectorSearchResult[]>;
  searchHybrid?(
    query: string,
    queryVector: Float32Array,
    k: number,
    candidateLimit: number
  ): VectorSearchResult[] | Promise<VectorSearchResult[]>;
  getStats(): StorageStats | Promise<StorageStats>;
}
