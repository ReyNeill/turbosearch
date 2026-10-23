import type { AppConfig } from "../config.ts";
import { chunkDocument } from "../chunking/chunker.ts";
import { collectOrgBrainEntryPaths, loadOrgBrainDocument } from "../corpora/org-brain.ts";
import type { EmbeddingDocument, EmbeddingProvider } from "../embeddings/provider.ts";
import { extractEvidence, extractionOptions } from "../ingest/extractors.ts";
import { createOcrProvider, type OcrProvider } from "../ocr/provider.ts";
import { createSearchStorage, type SearchStorage } from "../storage/database.ts";
import type { LexicalSearchResult } from "../storage/types.ts";
import { createTranscriptionProvider, type TranscriptionProvider } from "../transcription/provider.ts";
import type {
  ExtractedDocument,
  ExtractedEvidence,
  IngestEvent,
  IngestSummary,
  SearchBackendName,
  SearchFilters,
  SearchResult,
  StoredChunk,
  TextChunk
} from "../types.ts";
import { collectInputFiles } from "../utils/files.ts";
import { sha256Text } from "../utils/hash.ts";
import { cosine } from "../vector/math.ts";
import { TurboQuantCodec } from "../vector/turboquant/index.ts";

export class SearchEngine {
  private readonly database: SearchStorage;
  private readonly turbo: TurboQuantCodec;
  private readonly ocrProvider: OcrProvider | null;
  private readonly transcriptionProvider: TranscriptionProvider | null;

  constructor(
    private readonly config: AppConfig,
    private readonly embeddings: EmbeddingProvider
  ) {
    if (config.imageEmbeddings && !embeddings.supportsImages) {
      throw new Error(
        "TURBOSEARCH_IMAGE_EMBEDDINGS needs --embedding openai-compatible on a multimodal server (llama-server with --mmproj)."
      );
    }
    this.database = createSearchStorage(config);
    this.turbo = new TurboQuantCodec({
      dimension: embeddings.dimensions,
      bits: config.turboBits
    });
    this.ocrProvider = createOcrProvider(config);
    this.transcriptionProvider = createTranscriptionProvider(config);
  }

  async close(): Promise<void> {
    await this.database.close();
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
        extractionOptions(this.config)
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

      const vectors = await this.embeddings.embedDocuments(chunks.map(chunkEmbeddingDocument));
      const turboVectors = vectors.map((vector) => this.turbo.encode(vector));
      await this.database.replaceDocument(document, chunks, vectors, turboVectors);

      summary.documents += 1;
      summary.chunks += chunks.length;
      onEvent?.({ type: "document_indexed", path: file, chunks: chunks.length });
    }

    onEvent?.({ type: "ingest_done", summary });
    return summary;
  }

  async ingestOrgBrain(root: string, onEvent?: (event: IngestEvent) => void): Promise<IngestSummary> {
    const files = await collectOrgBrainEntryPaths(root);
    const summary: IngestSummary = {
      documents: 0,
      chunks: 0,
      skipped: 0,
      pendingExtractions: 0,
      unchanged: 0,
      removed: 0
    };
    const indexedPaths = new Set<string>();

    for (const file of files) {
      onEvent?.({ type: "document_started", path: file });
      const parsed = await loadOrgBrainDocument(root, file, {
        targetTokens: this.config.chunkTokens,
        overlapTokens: this.config.chunkOverlapTokens
      });
      indexedPaths.add(parsed.document.path);
      const existingHash = await this.database.getDocumentHash(parsed.document.path);
      if (existingHash === sha256Text(parsed.document.text)) {
        summary.unchanged = (summary.unchanged ?? 0) + 1;
        onEvent?.({ type: "document_unchanged", path: parsed.document.path });
        continue;
      }

      const vectors = await this.embeddings.embedDocuments(parsed.chunks.map(chunkEmbeddingDocument));
      const turboVectors = vectors.map((vector) => this.turbo.encode(vector));
      await this.database.replaceDocument(parsed.document, parsed.chunks, vectors, turboVectors);
      summary.documents += 1;
      summary.chunks += parsed.chunks.length;
      onEvent?.({ type: "document_indexed", path: parsed.document.path, chunks: parsed.chunks.length });
    }

    const stalePaths = (await this.database.listDocumentPaths()).filter((path) => !indexedPaths.has(path));
    summary.removed = await this.database.deleteDocuments(stalePaths);
    onEvent?.({ type: "ingest_done", summary });
    return summary;
  }

  async search(
    query: string,
    options: { backend?: SearchBackendName; k?: number; filters?: SearchFilters } = {}
  ): Promise<SearchResult[]> {
    const backend = options.backend ?? this.config.defaultBackend;
    const k = options.k ?? 10;
    const filters = options.filters;

    if (filters && Object.keys(filters).length > 0) {
      const chunks = (await this.database.getAllChunks()).filter((chunk) => chunkMatchesFilters(chunk, filters));
      if (backend === "bm25") {
        return (await this.database.searchLexical(query, Math.max(k, chunks.length)))
          .filter(({ chunk }) => chunkMatchesFilters(chunk, filters))
          .slice(0, k)
          .map(({ chunk, score }) => toSearchResult(chunk, score));
      }

      const queryVector = await this.embeddings.embedQuery(query);
      if (backend === "hybrid") {
        const lexical = (await this.database.searchLexical(query, Math.max(k, chunks.length)))
          .filter(({ chunk }) => chunkMatchesFilters(chunk, filters));
        return this.searchHybrid(queryVector, chunks, lexical, k);
      }
      if (backend === "turbo") return this.searchTurbo(queryVector, chunks, k);
      return this.searchExact(queryVector, chunks, k);
    }

    if (backend === "bm25") {
      return (await this.database.searchLexical(query, k)).map(({ chunk, score }) => toSearchResult(chunk, score));
    }

    const queryVector = await this.embeddings.embedQuery(query);

    if (backend === "hybrid") {
      if (this.database.searchHybrid) {
        return (await this.database.searchHybrid(
          query,
          queryVector,
          k,
          Math.max(k, this.config.turboRerankLimit)
        )).map(({ chunk, score }) => toSearchResult(chunk, score));
      }

      const [chunks, lexicalResults] = await Promise.all([
        this.database.getAllChunks(),
        this.database.searchLexical(query, Math.max(k, this.config.turboRerankLimit))
      ]);
      return this.searchHybrid(queryVector, chunks, lexicalResults, k);
    }

    if (backend === "exact" && this.database.searchVector) {
      return (await this.database.searchVector(queryVector, k)).map(({ chunk, score }) => toSearchResult(chunk, score));
    }

    if (backend === "turbo" && this.database.searchCompressed) {
      return (await this.database.searchCompressed(queryVector, k, this.config.turboRerankLimit))
        .map(({ chunk, score }) => toSearchResult(chunk, score));
    }

    const chunks = await this.database.getAllChunks();
    if (backend === "turbo") {
      return this.searchTurbo(queryVector, chunks, k);
    }

    return this.searchExact(queryVector, chunks, k);
  }

  async stats(): Promise<{ documents: number; chunks: number }> {
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

  private searchHybrid(
    queryVector: Float32Array,
    chunks: StoredChunk[],
    lexicalResults: LexicalSearchResult[],
    k: number
  ): SearchResult[] {
    const vectorRanked = this.searchExactCandidates(queryVector, chunks, Math.max(k, this.config.turboRerankLimit));
    const scores = new Map<number, { chunk: StoredChunk; score: number }>();

    addReciprocalRankScores(scores, vectorRanked.map(({ chunk }) => chunk), 0.55);
    addReciprocalRankScores(scores, lexicalResults.map(({ chunk }) => chunk), 0.45);

    return Array.from(scores.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(({ chunk, score }) => toSearchResult(chunk, score));
  }

  private searchExactCandidates(
    queryVector: Float32Array,
    chunks: StoredChunk[],
    limit: number
  ): Array<{ chunk: StoredChunk; score: number }> {
    return chunks
      .map((chunk) => ({ chunk, score: cosine(queryVector, chunk.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }
}

function addReciprocalRankScores(
  scores: Map<number, { chunk: StoredChunk; score: number }>,
  rankedChunks: StoredChunk[],
  weight: number
): void {
  const rankConstant = 60;
  for (let index = 0; index < rankedChunks.length; index += 1) {
    const chunk = rankedChunks[index];
    if (!chunk) continue;

    const existing = scores.get(chunk.id);
    const score = weight / (rankConstant + index + 1);
    scores.set(chunk.id, {
      chunk,
      score: (existing?.score ?? 0) + score
    });
  }
}

type ChunkSettings = {
  targetTokens: number;
  overlapTokens: number;
};

function chunkExtraction(evidenceItems: ExtractedEvidence[], settings: ChunkSettings): TextChunk[] {
  const chunks: TextChunk[] = [];

  for (const evidence of evidenceItems) {
    const metadata = {
      evidence: {
        modality: evidence.modality,
        locator: evidence.locator,
        metadata: evidence.metadata
      }
    };

    // Visual evidence is one textless chunk whose vector comes from the image itself.
    if (evidence.imagePath) {
      chunks.push({
        documentPath: evidence.sourcePath,
        ordinal: chunks.length,
        text: "",
        startOffset: 0,
        endOffset: 0,
        metadata: { ...metadata, sourceKind: evidence.sourceKind, embeddingImage: evidence.imagePath }
      });
      continue;
    }

    const evidenceDocument: ExtractedDocument = {
      path: evidence.sourcePath,
      kind: evidence.sourceKind,
      text: evidence.text,
      metadata
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
    text: extraction.evidence.map((evidence) => evidence.text).filter(Boolean).join("\n\n"),
    metadata: {
      evidenceCount: extraction.evidence.length,
      pendingExtractions: extraction.pending.length
    }
  };
}

function toSearchResult(chunk: StoredChunk, score: number): SearchResult {
  const { embeddingText: _text, embeddingTitle: _title, embeddingImage: _image, ...metadata } = chunk.metadata;
  return {
    chunkId: chunk.id,
    documentPath: chunk.documentPath,
    score,
    text: chunk.text,
    metadata: {
      ...metadata,
      ordinal: chunk.ordinal,
      startOffset: chunk.startOffset,
      endOffset: chunk.endOffset
    }
  };
}

function chunkEmbeddingDocument(chunk: TextChunk): EmbeddingDocument {
  const { embeddingImage, embeddingText, embeddingTitle } = chunk.metadata;
  if (typeof embeddingImage === "string") return { kind: "image", path: embeddingImage };
  return {
    kind: "text",
    text: typeof embeddingText === "string" ? embeddingText : chunk.text,
    title: typeof embeddingTitle === "string" ? embeddingTitle : undefined
  };
}

function chunkMatchesFilters(chunk: StoredChunk, filters: SearchFilters): boolean {
  if (filters.corpus && chunk.metadata.corpus !== filters.corpus) return false;
  const orgBrain = asRecord(chunk.metadata.orgBrain);
  if (!orgBrain) return !requiresOrgBrainMetadata(filters);
  if (filters.project && !sameText(orgBrain.project, filters.project)) return false;
  if (filters.type && !sameText(orgBrain.type, filters.type)) return false;
  if (filters.state && !sameText(orgBrain.state, filters.state)) return false;
  if (filters.author && !sameText(orgBrain.author, filters.author)) return false;
  if (filters.dateFrom && (typeof orgBrain.date !== "string" || orgBrain.date < filters.dateFrom)) return false;
  if (filters.dateTo && (typeof orgBrain.date !== "string" || orgBrain.date > filters.dateTo)) return false;
  if (filters.tags && filters.tags.length > 0) {
    const tags = Array.isArray(orgBrain.tags)
      ? orgBrain.tags.filter((tag): tag is string => typeof tag === "string").map((tag) => tag.toLowerCase())
      : [];
    if (!filters.tags.every((tag) => tags.includes(tag.toLowerCase()))) return false;
  }
  return true;
}

function requiresOrgBrainMetadata(filters: SearchFilters): boolean {
  return Boolean(
    filters.project || filters.type || filters.state || filters.author || filters.dateFrom || filters.dateTo || filters.tags?.length
  );
}

function sameText(value: unknown, expected: string): boolean {
  return typeof value === "string" && value.toLowerCase() === expected.toLowerCase();
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}
