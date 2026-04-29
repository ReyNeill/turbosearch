export type EmbeddingProviderName = "hash" | "openai";
export type SearchBackendName = "exact" | "turbo";

export type SourceKind = "text" | "pdf" | "image" | "video" | "binary";

export type EvidenceModality =
  | "text"
  | "pdf_text"
  | "image_ocr"
  | "video_transcript"
  | "video_frame_ocr";

export type EvidenceLocator = {
  page?: number;
  timestampMs?: number;
  framePath?: string;
};

export type ExtractedEvidence = {
  sourcePath: string;
  sourceKind: SourceKind;
  modality: EvidenceModality;
  text: string;
  locator: EvidenceLocator;
  metadata: Record<string, unknown>;
};

export type ExtractionPending = {
  sourcePath: string;
  sourceKind: SourceKind;
  pending: true;
  reason: string;
  requiredExtractor: "local-ocr-model" | "local-transcription-model";
};

export type ExtractionResult = {
  path: string;
  kind: SourceKind;
  evidence: ExtractedEvidence[];
  pending: ExtractionPending[];
  skipped: boolean;
};

export type ExtractedDocument = {
  path: string;
  kind: SourceKind;
  text: string;
  metadata: Record<string, unknown>;
};

export type TextChunk = {
  documentPath: string;
  ordinal: number;
  text: string;
  startOffset: number;
  endOffset: number;
  metadata: Record<string, unknown>;
};

export type StoredChunk = TextChunk & {
  id: number;
  documentId: number;
  embedding: Float32Array;
  turboCode: Uint8Array | null;
  turboNorm: number | null;
};

export type SearchResult = {
  chunkId: number;
  documentPath: string;
  score: number;
  text: string;
  metadata: Record<string, unknown>;
};

export type IngestSummary = {
  documents: number;
  chunks: number;
  skipped: number;
  pendingExtractions: number;
};

export type IngestEvent =
  | { type: "document_started"; path: string }
  | { type: "document_skipped"; path: string; reason: string }
  | { type: "extraction_pending"; path: string; reason: string; requiredExtractor: string }
  | { type: "document_indexed"; path: string; chunks: number }
  | { type: "ingest_done"; summary: IngestSummary };
