import type { TranscriptionSegment } from "./provider.ts";

type SegmentLike = {
  text?: unknown;
  start?: unknown;
  end?: unknown;
  start_ms?: unknown;
  end_ms?: unknown;
  startMs?: unknown;
  endMs?: unknown;
};

export function parseTranscriptionSegments(value: unknown): TranscriptionSegment[] | undefined {
  if (!Array.isArray(value)) return undefined;

  const segments = value.flatMap((item) => {
    if (!isSegmentLike(item)) return [];
    const text = typeof item.text === "string" ? item.text.trim() : "";
    const startMs = timestampToMs(item.startMs ?? item.start_ms ?? item.start);
    const endMs = timestampToMs(item.endMs ?? item.end_ms ?? item.end);
    // Do not infer timestamps from text position; agents should only receive provider-grounded locators.
    if (!text || startMs === null || endMs === null || endMs < startMs) return [];
    return [{ text, startMs, endMs }];
  });

  return segments.length > 0 ? segments : undefined;
}

function timestampToMs(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value > 1000 ? Math.round(value) : Math.round(value * 1000);
}

function isSegmentLike(value: unknown): value is SegmentLike {
  return typeof value === "object" && value !== null;
}
