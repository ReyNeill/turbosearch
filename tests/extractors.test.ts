import { expect, test } from "bun:test";
import { extractEvidence } from "../src/ingest/extractors.ts";
import type { OcrProvider } from "../src/ocr/provider.ts";
import type { TranscriptionProvider } from "../src/transcription/provider.ts";

const fixture = (name: string) => new URL(`./fixtures/${name}`, import.meta.url).pathname;

const mockOcrProvider: OcrProvider = {
  name: "ollama-glm-ocr",
  async extractText() {
    return {
      text: "Invoice total is 42 dollars.",
      model: "glm-ocr:latest",
      provider: "ollama-glm-ocr"
    };
  }
};

const mockTranscriptionProvider: TranscriptionProvider = {
  name: "cohere-api",
  async transcribe() {
    return {
      text: "Welcome. The renewal date is May first.",
      model: "cohere-transcribe-03-2026",
      provider: "cohere-api",
      language: "en",
      segments: [
        { text: "Welcome.", startMs: 0, endMs: 800 },
        { text: "The renewal date is May first.", startMs: 900, endMs: 2600 }
      ]
    };
  }
};

test("extractEvidence indexes image OCR evidence with provider metadata", async () => {
  const result = await extractEvidence(fixture("sample.png"), mockOcrProvider, null);

  expect(result.kind).toBe("image");
  expect(result.evidence).toHaveLength(1);
  expect(result.evidence[0]?.modality).toBe("image_ocr");
  expect(result.evidence[0]?.metadata.extractor).toBe("ollama-glm-ocr");
  expect(result.evidence[0]?.metadata.ocrMode).toBe("text");
});

test("extractEvidence preserves real transcription segment timestamps", async () => {
  const result = await extractEvidence(fixture("sample.wav"), null, mockTranscriptionProvider);

  expect(result.kind).toBe("audio");
  expect(result.evidence).toHaveLength(2);
  expect(result.evidence[0]?.locator.timestampMs).toBe(0);
  expect(result.evidence[1]?.locator.timestampMs).toBe(900);
  expect(result.evidence[1]?.metadata.endMs).toBe(2600);
});
