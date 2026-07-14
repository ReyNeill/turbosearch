import type { SearchResult } from "../types.ts";

export function distinctDocumentResults(results: SearchResult[], limit: number): SearchResult[] {
  const seen = new Set<string>();
  const distinct: SearchResult[] = [];
  for (const result of results) {
    if (seen.has(result.documentPath)) continue;
    seen.add(result.documentPath);
    distinct.push(result);
    if (distinct.length >= limit) break;
  }
  return distinct;
}
