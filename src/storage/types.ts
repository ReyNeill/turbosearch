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

export interface SearchStorage {
  close(): void | Promise<void>;
  replaceDocument(
    document: ExtractedDocument,
    chunks: TextChunk[],
    embeddings: Float32Array[],
    turboVectors: Array<TurboQuantEncoded | null>
  ): void | Promise<void>;
  getAllChunks(): StoredChunk[] | Promise<StoredChunk[]>;
  searchLexical(query: string, k: number): LexicalSearchResult[] | Promise<LexicalSearchResult[]>;
  getStats(): StorageStats | Promise<StorageStats>;
}
