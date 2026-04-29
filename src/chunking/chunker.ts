import type { ExtractedDocument, TextChunk } from "../types.ts";

export type ChunkOptions = {
  targetTokens: number;
  overlapTokens: number;
};

const tokenPattern = /\S+/g;

export function chunkDocument(document: ExtractedDocument, options: ChunkOptions): TextChunk[] {
  const tokens = Array.from(document.text.matchAll(tokenPattern), (match) => ({
    text: match[0],
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length
  }));

  if (tokens.length === 0) return [];

  const chunks: TextChunk[] = [];
  const target = Math.max(1, options.targetTokens);
  const overlap = Math.max(0, Math.min(options.overlapTokens, target - 1));
  let startToken = 0;

  while (startToken < tokens.length) {
    const endToken = Math.min(tokens.length, startToken + target);
    const startOffset = tokens[startToken]?.start ?? 0;
    const endOffset = tokens[endToken - 1]?.end ?? startOffset;
    const text = document.text.slice(startOffset, endOffset).trim();

    if (text) {
      chunks.push({
        documentPath: document.path,
        ordinal: chunks.length,
        text,
        startOffset,
        endOffset,
        metadata: {
          ...document.metadata,
          sourceKind: document.kind
        }
      });
    }

    if (endToken === tokens.length) break;
    startToken = endToken - overlap;
  }

  return chunks;
}
