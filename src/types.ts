export type EmbeddingProviderName = "hash" | "openai";
export type SearchBackendName = "exact" | "turbo";
export type OcrProviderName = "none" | "ollama-glm-ocr" | "vllm-glm-ocr";
export type OcrMode = "text" | "table" | "formula" | "figure";
export type TranscriptionProviderName = "none" | "cohere-api" | "vllm-cohere-transcribe";
export type PdfParserName = "liteparse" | "pdftotext";

export type SourceKind = "text" | "pdf" | "image" | "audio" | "video" | "binary";

export type EvidenceModality =
  | "text"
  | "pdf_text"
  | "pdf_ocr"
  | "image_ocr"
  | "audio_transcript"
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
  requiredExtractor: OcrProviderName | TranscriptionProviderName;
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
