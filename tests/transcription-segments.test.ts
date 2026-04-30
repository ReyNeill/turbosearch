import { expect, test } from "bun:test";
import { parseTranscriptionSegments } from "../src/transcription/segments.ts";

test("parseTranscriptionSegments only keeps segments with real timestamps", () => {
  const segments = parseTranscriptionSegments([
    { text: "first", start: 0, end: 1.25 },
    { text: "second", start_ms: 1300, end_ms: 2500 },
    { text: "missing timestamps" }
  ]);

  expect(segments).toEqual([
    { text: "first", startMs: 0, endMs: 1250 },
    { text: "second", startMs: 1300, endMs: 2500 }
  ]);
});
