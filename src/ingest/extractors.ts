import { readFile, stat } from "node:fs/promises";
import { spawn } from "node:child_process";
import type {
  ExtractedDocument,
  ExtractedEvidence,
  ExtractionPending,
  ExtractionResult
} from "../types.ts";
import type { OcrProvider } from "../ocr/provider.ts";
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
const videoExtensions = new Set([".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"]);

export async function extractDocument(path: string, ocrProvider?: OcrProvider | null): Promise<ExtractedDocument | null> {
  const result = await extractEvidence(path, ocrProvider);
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

export async function extractEvidence(path: string, ocrProvider?: OcrProvider | null): Promise<ExtractionResult> {
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
    return {
      path,
      kind: "pdf",
      evidence: await extractPdfEvidence(path),
      pending: [],
      skipped: false
    };
  }

  if (imageExtensions.has(extension) || videoExtensions.has(extension)) {
    return extractMediaEvidence(path, imageExtensions.has(extension) ? "image" : "video", ocrProvider ?? null);
  }

  return { path, kind: "binary", evidence: [], pending: [], skipped: true };
}

async function extractPdfEvidence(path: string): Promise<ExtractedEvidence[]> {
  const pageCount = await getPdfPageCount(path);
  if (pageCount === null) {
    const result = await runCommand("pdftotext", ["-layout", path, "-"]);
    if (result.exitCode !== 0) {
      throw new Error(`Failed to extract PDF text from ${path}. Install poppler/pdftotext.`);
    }

    return [{
      sourcePath: path,
      sourceKind: "pdf",
      modality: "pdf_text",
      text: result.stdout,
      locator: {},
      metadata: { extension: ".pdf", extractor: "pdftotext" }
    }];
  }

  const evidence: ExtractedEvidence[] = [];

  for (let page = 1; page <= pageCount; page += 1) {
    const result = await runCommand("pdftotext", ["-f", String(page), "-l", String(page), "-layout", path, "-"]);
    if (result.exitCode !== 0) {
      throw new Error(`Failed to extract PDF page ${page} from ${path}. Install poppler/pdftotext.`);
    }

    const text = result.stdout.trim();
    if (!text) continue;

    evidence.push({
      sourcePath: path,
      sourceKind: "pdf",
      modality: "pdf_text",
      text,
      locator: { page },
      metadata: { extension: ".pdf", extractor: "pdftotext" }
    });
  }

  return evidence;
}

async function getPdfPageCount(path: string): Promise<number | null> {
  const result = await runCommand("pdfinfo", [path]);
  if (result.exitCode !== 0) return null;

  const match = result.stdout.match(/^Pages:\s+(\d+)$/m);
  return match ? Number(match[1]) : null;
}

async function extractMediaEvidence(path: string, kind: "image" | "video", ocrProvider: OcrProvider | null): Promise<ExtractionResult> {
  const sidecar = await readFirstExisting(kind === "image"
    ? [`${path}.ocr.txt`, `${path}.txt`, `${path}.md`]
    : [`${path}.transcript.txt`, `${path}.ocr.txt`, `${path}.txt`, `${path}.md`]);

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
  } else if (kind === "video") {
    pending.push(createPendingExtraction(path, kind));
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
      modality: kind === "image" ? "image_ocr" : "video_transcript",
      text: sidecar,
      locator: {},
      metadata: {
        extension: extensionOf(path),
        extractor: "sidecar",
        localModelExtractor: kind === "image" ? "glm-ocr-fallback" : "pending"
      }
    }],
    pending,
    skipped: false
  };
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

  return {
    sourcePath: path,
    sourceKind: kind,
    pending: true,
    reason: "Video/audio transcription local model is not configured yet.",
    requiredExtractor: "local-transcription-model"
  };
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
