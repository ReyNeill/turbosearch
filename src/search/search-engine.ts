import type { AppConfig } from "../config.ts";
import { chunkDocument } from "../chunking/chunker.ts";
import type { EmbeddingProvider } from "../embeddings/provider.ts";
import { extractEvidence } from "../ingest/extractors.ts";
import { createOcrProvider, type OcrProvider } from "../ocr/provider.ts";
import { SearchDatabase } from "../storage/database.ts";
import { createTranscriptionProvider, type TranscriptionProvider } from "../transcription/provider.ts";
import type {
  ExtractedDocument,
  ExtractedEvidence,
  IngestEvent,
  IngestSummary,
  SearchBackendName,
  SearchResult,
  StoredChunk,
  TextChunk
} from "../types.ts";
import { collectInputFiles } from "../utils/files.ts";
import { cosine } from "../vector/math.ts";
import { TurboQuantCodec } from "../vector/turboquant/index.ts";

export class SearchEngine {
  private readonly database: SearchDatabase;
  private readonly turbo: TurboQuantCodec;
  private readonly ocrProvider: OcrProvider | null;
  private readonly transcriptionProvider: TranscriptionProvider | null;

  constructor(
    private readonly config: AppConfig,
    private readonly embeddings: EmbeddingProvider
  ) {
    this.database = new SearchDatabase(config.dbPath);
    this.turbo = new TurboQuantCodec({
      dimension: embeddings.dimensions,
      bits: config.turboBits
    });
    this.ocrProvider = createOcrProvider(config);
    this.transcriptionProvider = createTranscriptionProvider(config);
  }

  close(): void {
    this.database.close();
  }

  async ingestPaths(paths: string[], onEvent?: (event: IngestEvent) => void): Promise<IngestSummary> {
    const files = await collectInputFiles(paths);
    const summary: IngestSummary = { documents: 0, chunks: 0, skipped: 0, pendingExtractions: 0 };

    for (const file of files) {
      onEvent?.({ type: "document_started", path: file });
      const extraction = await extractEvidence(
        file,
        this.ocrProvider,
        this.transcriptionProvider,
        this.config.pdfParser
      );
      summary.pendingExtractions += extraction.pending.length;
      for (const pending of extraction.pending) {
        onEvent?.({
          type: "extraction_pending",
          path: pending.sourcePath,
          reason: pending.reason,
          requiredExtractor: pending.requiredExtractor
        });
      }

      const document = toDocument(extraction);
      if (!document) {
        summary.skipped += 1;
        onEvent?.({
          type: "document_skipped",
          path: file,
          reason: extraction.skipped ? "unsupported file type" : "no searchable evidence"
        });
        continue;
      }

      const chunks = chunkExtraction(extraction.evidence, {
        targetTokens: this.config.chunkTokens,
        overlapTokens: this.config.chunkOverlapTokens
      });

      if (chunks.length === 0) {
        summary.skipped += 1;
        onEvent?.({ type: "document_skipped", path: file, reason: "no chunks produced" });
        continue;
      }

      const vectors = await this.embeddings.embedTexts(chunks.map((chunk) => chunk.text));
      const turboVectors = vectors.map((vector) => this.turbo.encode(vector));
      this.database.replaceDocument(document, chunks, vectors, turboVectors);

      summary.documents += 1;
      summary.chunks += chunks.length;
      onEvent?.({ type: "document_indexed", path: file, chunks: chunks.length });
    }

    onEvent?.({ type: "ingest_done", summary });
    return summary;
  }

  async search(query: string, options: { backend?: SearchBackendName; k?: number } = {}): Promise<SearchResult[]> {
    const backend = options.backend ?? this.config.defaultBackend;
    const k = options.k ?? 10;
    const queryVector = await this.embeddings.embedText(query);
    const chunks = this.database.getAllChunks();

    if (backend === "turbo") {
      return this.searchTurbo(queryVector, chunks, k);
    }

    return this.searchExact(queryVector, chunks, k);
  }

  stats(): { documents: number; chunks: number } {
    return this.database.getStats();
  }

  private searchExact(queryVector: Float32Array, chunks: StoredChunk[], k: number): SearchResult[] {
    return chunks
      .map((chunk) => ({ chunk, score: cosine(queryVector, chunk.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(({ chunk, score }) => toSearchResult(chunk, score));
  }

  private searchTurbo(queryVector: Float32Array, chunks: StoredChunk[], k: number): SearchResult[] {
    const rotatedQuery = this.turbo.prepareQuery(queryVector);
    const candidates = chunks
      .filter((chunk) => chunk.turboCode && chunk.turboNorm !== null)
      .map((chunk) => ({
        chunk,
        approximateScore: this.turbo.scorePreparedQuery(rotatedQuery, {
          code: chunk.turboCode as Uint8Array,
          norm: chunk.turboNorm ?? 1
        })
      }))
      .sort((a, b) => b.approximateScore - a.approximateScore)
      .slice(0, Math.max(k, this.config.turboRerankLimit));

    return candidates
      .map(({ chunk }) => ({ chunk, score: cosine(queryVector, chunk.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(({ chunk, score }) => toSearchResult(chunk, score));
  }
}

type ChunkSettings = {
  targetTokens: number;
  overlapTokens: number;
};

function chunkExtraction(evidenceItems: ExtractedEvidence[], settings: ChunkSettings): TextChunk[] {
  const chunks: TextChunk[] = [];

  for (const evidence of evidenceItems) {
    const evidenceDocument: ExtractedDocument = {
      path: evidence.sourcePath,
      kind: evidence.sourceKind,
      text: evidence.text,
      metadata: {
        evidence: {
          modality: evidence.modality,
          locator: evidence.locator,
          metadata: evidence.metadata
        }
      }
    };

    for (const chunk of chunkDocument(evidenceDocument, settings)) {
      chunks.push({
        ...chunk,
        ordinal: chunks.length
      });
    }
  }

  return chunks;
}

function toDocument(extraction: Awaited<ReturnType<typeof extractEvidence>>): ExtractedDocument | null {
  if (extraction.skipped || extraction.evidence.length === 0) return null;

  return {
    path: extraction.path,
    kind: extraction.kind,
    text: extraction.evidence.map((evidence) => evidence.text).join("\n\n"),
    metadata: {
      evidenceCount: extraction.evidence.length,
      pendingExtractions: extraction.pending.length
    }
  };
}

function toSearchResult(chunk: StoredChunk, score: number): SearchResult {
  return {
    chunkId: chunk.id,
    documentPath: chunk.documentPath,
    score,
    text: chunk.text,
    metadata: {
      ...chunk.metadata,
      ordinal: chunk.ordinal,
      startOffset: chunk.startOffset,
      endOffset: chunk.endOffset
    }
  };
}
