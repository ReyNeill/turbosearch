import { readFile } from "node:fs/promises";
import type { SearchEngine } from "../search/search-engine.ts";
import { distinctDocumentResults } from "../search/result-utils.ts";
import type { SearchBackendName, SearchFilters } from "../types.ts";

export type RetrievalEvalCase = {
  query: string;
  expectedEntryIds: string[];
  filters?: SearchFilters;
};

export type RetrievalEvalDataset = {
  version: 1;
  cases: RetrievalEvalCase[];
};

export type RetrievalEvalReport = {
  cases: number;
  k: number;
  recallAtK: number;
  meanReciprocalRank: number;
  results: Array<{
    query: string;
    expectedEntryIds: string[];
    retrievedEntryIds: string[];
    firstRelevantRank: number | null;
  }>;
};

export async function loadRetrievalEvalDataset(path: string): Promise<RetrievalEvalDataset> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as unknown;
  if (!isRecord(parsed) || parsed.version !== 1 || !Array.isArray(parsed.cases)) {
    throw new Error("evaluation dataset must be an object with version 1 and a cases array");
  }
  const cases = parsed.cases.map((value, index) => parseCase(value, index));
  if (cases.length === 0) throw new Error("evaluation dataset must contain at least one case");
  return { version: 1, cases };
}

export async function evaluateRetrieval(
  engine: SearchEngine,
  dataset: RetrievalEvalDataset,
  options: { k: number; backend: SearchBackendName }
): Promise<RetrievalEvalReport> {
  const results: RetrievalEvalReport["results"] = [];

  for (const testCase of dataset.cases) {
    const candidates = await engine.search(testCase.query, {
      backend: options.backend,
      k: options.k * 10,
      filters: {
        corpus: "org-brain",
        state: "active",
        ...testCase.filters
      }
    });
    const matches = distinctDocumentResults(candidates, options.k);
    const retrievedEntryIds = matches
      .map((match) => orgBrainEntryId(match.metadata))
      .filter((id): id is string => Boolean(id));
    const firstRelevant = retrievedEntryIds.findIndex((id) => testCase.expectedEntryIds.includes(id));
    results.push({
      query: testCase.query,
      expectedEntryIds: testCase.expectedEntryIds,
      retrievedEntryIds,
      firstRelevantRank: firstRelevant === -1 ? null : firstRelevant + 1
    });
  }

  const relevant = results.filter((result) => result.firstRelevantRank !== null).length;
  const reciprocalRank = results.reduce((total, result) => (
    total + (result.firstRelevantRank ? 1 / result.firstRelevantRank : 0)
  ), 0);
  return {
    cases: results.length,
    k: options.k,
    recallAtK: relevant / results.length,
    meanReciprocalRank: reciprocalRank / results.length,
    results
  };
}

function parseCase(value: unknown, index: number): RetrievalEvalCase {
  if (!isRecord(value) || typeof value.query !== "string" || !value.query.trim()) {
    throw new Error(`evaluation case ${index + 1} must contain a non-empty query`);
  }
  if (!Array.isArray(value.expectedEntryIds) || value.expectedEntryIds.length === 0 ||
      !value.expectedEntryIds.every((id) => typeof id === "string" && id.length > 0)) {
    throw new Error(`evaluation case ${index + 1} must contain expectedEntryIds`);
  }
  const filters = value.filters === undefined ? undefined : parseFilters(value.filters, index);
  return { query: value.query, expectedEntryIds: value.expectedEntryIds as string[], filters };
}

function parseFilters(value: unknown, index: number): SearchFilters {
  if (!isRecord(value)) throw new Error(`evaluation case ${index + 1} filters must be an object`);
  const filters: SearchFilters = {};
  for (const key of ["corpus", "project", "type", "state", "author", "dateFrom", "dateTo"] as const) {
    const item = value[key];
    if (item !== undefined && typeof item !== "string") {
      throw new Error(`evaluation case ${index + 1} filter ${key} must be a string`);
    }
    if (typeof item === "string") filters[key] = item;
  }
  if (value.tags !== undefined) {
    if (!Array.isArray(value.tags) || !value.tags.every((tag) => typeof tag === "string")) {
      throw new Error(`evaluation case ${index + 1} filter tags must be an array of strings`);
    }
    filters.tags = value.tags as string[];
  }
  return filters;
}

function orgBrainEntryId(metadata: Record<string, unknown>): string | null {
  const orgBrain = metadata.orgBrain;
  if (!isRecord(orgBrain) || typeof orgBrain.id !== "string") return null;
  return orgBrain.id;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
