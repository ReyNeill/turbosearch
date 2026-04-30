import { mkdir, readFile, stat } from "node:fs/promises";
import { spawn } from "node:child_process";
import { basename, join } from "node:path";
import { LiteParse, type ParseResult, type ParsedPage } from "@llamaindex/liteparse";
import type {
  ExtractedDocument,
  ExtractedEvidence,
  ExtractionPending,
  ExtractionResult,
  PdfParserName
} from "../types.ts";
import type { OcrProvider } from "../ocr/provider.ts";
import type { TranscriptionProvider } from "../transcription/provider.ts";
import { extensionOf } from "../utils/files.ts";

const textExtensions = new Set([
  ".txt",
  ".md",
  ".mdx",
  ".json",
  ".jsonl",
  ".csv",
  ".tsv",
  ".yaml",
  ".yml",
  ".xml",
  ".html",
  ".css",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".py",
  ".rb",
  ".go",
  ".rs",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
  ".sql",
  ".toml"
]);

const imageExtensions = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif", ".tiff", ".bmp"]);
const audioExtensions = new Set([".wav", ".mp3", ".mpeg", ".mpga", ".m4a", ".ogg", ".flac"]);
const videoExtensions = new Set([".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"]);

export async function extractDocument(
  path: string,
  ocrProvider?: OcrProvider | null,
  transcriptionProvider?: TranscriptionProvider | null,
  pdfParser: PdfParserName = "liteparse"
): Promise<ExtractedDocument | null> {
  const result = await extractEvidence(path, ocrProvider, transcriptionProvider, pdfParser);
  if (result.skipped || result.evidence.length === 0) return null;

  return {
    path,
    kind: result.kind,
    text: result.evidence.map((evidence) => evidence.text).join("\n\n"),
    metadata: {
      evidenceCount: result.evidence.length,
      pendingExtractions: result.pending.length,
      evidence: result.evidence.map((evidence) => ({
        modality: evidence.modality,
        locator: evidence.locator,
        metadata: evidence.metadata
      }))
    }
  };
}

export async function extractEvidence(
  path: string,
  ocrProvider?: OcrProvider | null,
  transcriptionProvider?: TranscriptionProvider | null,
  pdfParser: PdfParserName = "liteparse"
): Promise<ExtractionResult> {
  const fileStat = await stat(path);
  if (!fileStat.isFile()) {
    throw new Error(`Extraction path is not a file: ${path}`);
  }

  const extension = extensionOf(path);

  if (textExtensions.has(extension)) {
    return {
      path,
      kind: "text",
      evidence: [{
        sourcePath: path,
        sourceKind: "text",
        modality: "text",
        text: await readFile(path, "utf8"),
        locator: {},
        metadata: { extension }
      }],
      pending: [],
      skipped: false
    };
  }

  if (extension === ".pdf") {
    const pdf = await extractPdfEvidence(path, ocrProvider ?? null, pdfParser);
    return {
      path,
      kind: "pdf",
      evidence: pdf.evidence,
      pending: pdf.pending,
      skipped: false
    };
  }

  if (imageExtensions.has(extension)) {
    return extractMediaEvidence(path, "image", ocrProvider ?? null, transcriptionProvider ?? null);
  }

  if (audioExtensions.has(extension)) {
    return extractMediaEvidence(path, "audio", ocrProvider ?? null, transcriptionProvider ?? null);
  }

  if (videoExtensions.has(extension)) {
    return extractMediaEvidence(path, "video", ocrProvider ?? null, transcriptionProvider ?? null);
  }

  return { path, kind: "binary", evidence: [], pending: [], skipped: true };
}

async function extractPdfEvidence(
  path: string,
  ocrProvider: OcrProvider | null,
  pdfParser: PdfParserName
): Promise<{ evidence: ExtractedEvidence[]; pending: ExtractionPending[] }> {
  if (pdfParser === "liteparse") {
    return extractPdfEvidenceWithLiteParse(path, ocrProvider);
  }

  return extractPdfEvidenceWithPdftotext(path, ocrProvider);
}

async function extractPdfEvidenceWithLiteParse(
  path: string,
  ocrProvider: OcrProvider | null
): Promise<{ evidence: ExtractedEvidence[]; pending: ExtractionPending[] }> {
  let liteparse: ParseResult | null = null;
  try {
    liteparse = await parsePdfWithLiteParse(path);
  } catch {
    return extractPdfEvidenceWithPdftotext(path, ocrProvider);
  }

  const pageCount = await getPdfPageCount(path);
  const pages = liteparse.pages.length > 0 ? liteparse.pages : pagesFromFullText(liteparse.text);
  const pagesByNumber = new Map(pages.map((page) => [page.pageNum, page]));
  const maxPage = pageCount ?? Math.max(...pages.map((page) => page.pageNum), 0);

  if (maxPage === 0) {
    return extractPdfEvidenceWithPdftotext(path, ocrProvider);
  }

  const evidence: ExtractedEvidence[] = [];
  const pending: ExtractionPending[] = [];

  for (let page = 1; page <= maxPage; page += 1) {
    const liteparsePage = pagesByNumber.get(page);
    const liteparseText = liteparsePage?.text.trim() ?? "";
    if (!isSparsePdfText(liteparseText)) {
      evidence.push(pdfTextEvidence(path, page, liteparseText, "liteparse", liteparseMetadata(liteparsePage)));
      continue;
    }

    const fallback = await extractPdfPageWithPdftotextOrOcr(path, page, ocrProvider);
    if (fallback.evidence) evidence.push(fallback.evidence);
    if (fallback.pending) pending.push(fallback.pending);
  }

  return { evidence, pending };
}

async function extractPdfEvidenceWithPdftotext(
  path: string,
  ocrProvider: OcrProvider | null
): Promise<{ evidence: ExtractedEvidence[]; pending: ExtractionPending[] }> {
  const pageCount = await getPdfPageCount(path);
  if (pageCount === null) {
    const result = await runCommand("pdftotext", ["-layout", path, "-"]);
    if (result.exitCode !== 0 || isSparsePdfText(result.stdout)) {
      if (!ocrProvider) {
        return {
          evidence: [],
          pending: [createPdfOcrPending(path, undefined, "PDF has no extractable text and OCR is disabled.")]
        };
      }

      try {
        const imagePath = await renderPdfPage(path, 1);
        const ocr = await ocrProvider.extractText({ imagePath, mode: "text" });
        return {
          evidence: [pdfOcrEvidence(path, 1, ocr.text, ocr.provider, ocr.model)],
          pending: []
        };
      } catch (error) {
        return {
          evidence: [],
          pending: [createPdfOcrPending(path, undefined, error)]
        };
      }
    }

    if (result.exitCode !== 0) {
      throw new Error(`Failed to extract PDF text from ${path}. Install poppler/pdftotext.`);
    }

    return {
      evidence: [pdfTextEvidence(path, undefined, result.stdout)],
      pending: []
    };
  }

  const evidence: ExtractedEvidence[] = [];
  const pending: ExtractionPending[] = [];

  for (let page = 1; page <= pageCount; page += 1) {
    const fallback = await extractPdfPageWithPdftotextOrOcr(path, page, ocrProvider);
    if (fallback.evidence) evidence.push(fallback.evidence);
    if (fallback.pending) pending.push(fallback.pending);
  }

  return { evidence, pending };
}

async function parsePdfWithLiteParse(path: string): Promise<ParseResult> {
  const parser = new LiteParse({
    ocrEnabled: false,
    outputFormat: "json",
    preciseBoundingBox: false,
    preserveLayoutAlignmentAcrossPages: true
  });
  return parser.parse(path, true);
}

function pagesFromFullText(text: string): ParsedPage[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  return [{
    pageNum: 1,
    width: 0,
    height: 0,
    text: trimmed,
    textItems: []
  }];
}

async function extractPdfPageWithPdftotextOrOcr(
  path: string,
  page: number,
  ocrProvider: OcrProvider | null
): Promise<{ evidence?: ExtractedEvidence; pending?: ExtractionPending }> {
  let pdftotextError: unknown = null;
  try {
    const result = await runCommand("pdftotext", ["-f", String(page), "-l", String(page), "-layout", path, "-"]);
    if (result.exitCode === 0) {
      const text = result.stdout.trim();
      if (!isSparsePdfText(text)) {
        return { evidence: pdfTextEvidence(path, page, text, "pdftotext") };
      }
    } else {
      pdftotextError = result.stderr || result.stdout;
    }
  } catch (error) {
    pdftotextError = error;
  }

  if (!ocrProvider) {
    const detail = pdftotextError
      ? `pdftotext fallback failed or was sparse: ${errorMessage(pdftotextError)}`
      : "PDF page has little/no extractable text and OCR is disabled.";
    return {
      pending: createPdfOcrPending(path, page, detail)
    };
  }

  try {
    const imagePath = await renderPdfPage(path, page);
    const ocr = await ocrProvider.extractText({ imagePath, mode: "text" });
    return { evidence: pdfOcrEvidence(path, page, ocr.text, ocr.provider, ocr.model) };
  } catch (error) {
    return { pending: createPdfOcrPending(path, page, error) };
  }
}

async function getPdfPageCount(path: string): Promise<number | null> {
  const result = await runCommand("pdfinfo", [path]);
  if (result.exitCode !== 0) return null;

  const match = result.stdout.match(/^Pages:\s+(\d+)$/m);
  return match ? Number(match[1]) : null;
}

async function extractMediaEvidence(
  path: string,
  kind: "image" | "audio" | "video",
  ocrProvider: OcrProvider | null,
  transcriptionProvider: TranscriptionProvider | null
): Promise<ExtractionResult> {
  const sidecar = await readFirstExisting(kind === "image"
    ? [`${path}.ocr.txt`, `${path}.txt`, `${path}.md`]
    : [`${path}.transcript.txt`, `${path}.txt`, `${path}.md`]);

  const pending: ExtractionPending[] = [];
  if (kind === "image" && ocrProvider) {
    try {
      const ocr = await ocrProvider.extractText({ imagePath: path });
      return {
        path,
        kind,
        evidence: [{
          sourcePath: path,
          sourceKind: kind,
          modality: "image_ocr",
          text: ocr.text,
          locator: {},
          metadata: {
            extension: extensionOf(path),
            extractor: ocr.provider,
            model: ocr.model
          }
        }],
        pending,
        skipped: false
      };
    } catch (error) {
      pending.push(createPendingExtraction(path, kind, error));
    }
  } else if (kind === "audio" || kind === "video") {
    if (transcriptionProvider) {
      try {
        const audioPath = kind === "audio" ? path : await extractAudioForTranscription(path);
        const transcript = await transcriptionProvider.transcribe({ audioPath });
        return {
          path,
          kind,
          evidence: [{
            sourcePath: path,
            sourceKind: kind,
            modality: kind === "audio" ? "audio_transcript" : "video_transcript",
            text: transcript.text,
            locator: {},
            metadata: {
              extension: extensionOf(path),
              extractor: transcript.provider,
              model: transcript.model,
              language: transcript.language
            }
          }],
          pending,
          skipped: false
        };
      } catch (error) {
        pending.push(createTranscriptionPending(path, kind, error));
      }
    } else {
      pending.push(createTranscriptionPending(path, kind));
    }
  }

  if (!sidecar) {
    return {
      path,
      kind,
      evidence: [],
      pending,
      skipped: false
    };
  }

  return {
    path,
    kind,
    evidence: [{
      sourcePath: path,
      sourceKind: kind,
      modality: sidecarModality(kind),
      text: sidecar,
      locator: {},
      metadata: {
        extension: extensionOf(path),
        extractor: "sidecar",
        localModelExtractor: kind === "image" ? "glm-ocr-fallback" : "transcription-fallback"
      }
    }],
    pending,
    skipped: false
  };
}

function sidecarModality(kind: "image" | "audio" | "video"): ExtractedEvidence["modality"] {
  if (kind === "image") return "image_ocr";
  if (kind === "audio") return "audio_transcript";
  return "video_transcript";
}

function createPendingExtraction(path: string, kind: "image" | "video", error?: unknown): ExtractionPending {
  if (kind === "image") {
    const detail = error instanceof Error ? ` ${error.message}` : "";
    return {
      sourcePath: path,
      sourceKind: kind,
      pending: true,
      reason: `GLM-OCR evidence is pending.${detail}`,
      requiredExtractor: "ollama-glm-ocr"
    };
  }

  return createTranscriptionPending(path, kind);
}

function createTranscriptionPending(path: string, kind: "audio" | "video", error?: unknown): ExtractionPending {
  const detail = error instanceof Error ? ` ${error.message}` : "";
  return {
    sourcePath: path,
    sourceKind: kind,
    pending: true,
    reason: `Video/audio transcription with Cohere Transcribe is pending.${detail}`,
    requiredExtractor: "cohere-api"
  };
}

function pdfTextEvidence(
  path: string,
  page: number | undefined,
  text: string,
  extractor: "liteparse" | "pdftotext" = "pdftotext",
  metadata: Record<string, unknown> = {}
): ExtractedEvidence {
  return {
    sourcePath: path,
    sourceKind: "pdf",
    modality: "pdf_text",
    text,
    locator: page ? { page } : {},
    metadata: { extension: ".pdf", extractor, ...metadata }
  };
}

function liteparseMetadata(page: ParsedPage | undefined): Record<string, unknown> {
  if (!page) return {};
  return {
    parser: "layout-aware",
    pageWidth: page.width,
    pageHeight: page.height,
    textItems: page.textItems.length
  };
}

function pdfOcrEvidence(path: string, page: number, text: string, extractor: string, model: string): ExtractedEvidence {
  return {
    sourcePath: path,
    sourceKind: "pdf",
    modality: "pdf_ocr",
    text,
    locator: { page },
    metadata: { extension: ".pdf", extractor, model }
  };
}

function createPdfOcrPending(path: string, page: number | undefined, error: unknown): ExtractionPending {
  const detail = errorMessage(error);
  return {
    sourcePath: path,
    sourceKind: "pdf",
    pending: true,
    reason: `PDF OCR is pending${page ? ` for page ${page}` : ""}. ${detail}`,
    requiredExtractor: "ollama-glm-ocr"
  };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isSparsePdfText(text: string): boolean {
  const normalized = text.replace(/\s+/g, "");
  return normalized.length < 40;
}

async function renderPdfPage(path: string, page: number): Promise<string> {
  const directory = ".turbosearch/pdf-ocr-pages";
  await mkdir(directory, { recursive: true });

  const safeName = basename(path).replace(/[^a-zA-Z0-9._-]/g, "_");
  const prefix = join(directory, `${safeName}-page-${page}`);
  const result = await runCommand("pdftoppm", [
    "-r",
    "150",
    "-f",
    String(page),
    "-l",
    String(page),
    "-singlefile",
    "-png",
    path,
    prefix
  ]);

  if (result.exitCode !== 0) {
    throw new Error(`Failed to render PDF page ${page}: ${result.stderr || result.stdout}`);
  }

  return `${prefix}.png`;
}

async function extractAudioForTranscription(path: string): Promise<string> {
  const directory = ".turbosearch/audio";
  await mkdir(directory, { recursive: true });

  const safeName = basename(path).replace(/[^a-zA-Z0-9._-]/g, "_");
  const outputPath = join(directory, `${safeName}.wav`);
  const result = await runCommand("ffmpeg", [
    "-y",
    "-i",
    path,
    "-vn",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-c:a",
    "pcm_s16le",
    outputPath
  ]);

  if (result.exitCode !== 0) {
    throw new Error(`Failed to extract audio with ffmpeg: ${result.stderr || result.stdout}`);
  }

  return outputPath;
}

export async function extractPdfText(path: string): Promise<string> {
  const result = await runCommand("pdftotext", ["-layout", path, "-"]);
  if (result.exitCode === 0) return result.stdout;

  throw new Error(
    `Failed to extract PDF text from ${path}. Install poppler/pdftotext or add a sidecar text file.`
  );
}

async function readFirstExisting(paths: string[]): Promise<string | null> {
  for (const path of paths) {
    try {
      return await readFile(path, "utf8");
    } catch {
      continue;
    }
  }
  return null;
}

function runCommand(command: string, args: string[]): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];

    child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
    child.on("error", reject);
    child.on("close", (exitCode) => {
      resolve({
        exitCode: exitCode ?? 1,
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8")
      });
    });
  });
}
