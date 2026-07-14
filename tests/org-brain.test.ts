import { mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test } from "bun:test";
import { createConfig } from "../src/config.ts";
import { parseOrgBrainDocument } from "../src/corpora/org-brain.ts";
import { HashEmbeddingProvider } from "../src/embeddings/hash.ts";
import { evaluateRetrieval } from "../src/eval/retrieval.ts";
import { SearchEngine } from "../src/search/search-engine.ts";
import { distinctDocumentResults } from "../src/search/result-utils.ts";

const activeId = "2026-07-14-release-policy--reyneill--11111111";
const inactiveId = "2026-07-13-old-policy--reyneill--22222222";

test("org-brain adapter creates lifecycle metadata and line citations", () => {
  const path = `knowledge/silixon-core/decisions/${activeId}.md`;
  const parsed = parseOrgBrainDocument("/repo", `/repo/${path}`, entry({
    id: activeId,
    name: "release-policy",
    description: "Release policy",
    state: "active",
    body: "## Decision\n\nProduction releases merge dev into prod."
  }), { targetTokens: 50, overlapTokens: 5 });

  expect(parsed.entry.project).toBe("silixon-core");
  expect(parsed.entry.state).toBe("active");
  expect(parsed.document.path).toBe(path);
  const metadata = parsed.chunks[0]?.metadata.orgBrain as Record<string, unknown>;
  expect(metadata.heading).toBe("Decision");
  expect(metadata.lineStart).toBeGreaterThan(1);
  expect(metadata.lineEnd).toBeGreaterThanOrEqual(metadata.lineStart as number);
  expect(parsed.chunks[0]?.metadata.embeddingText).toContain("Release policy");
});

test("org-brain result shaping keeps the best chunk per entry", () => {
  const base = { chunkId: 1, documentPath: "one.md", score: 1, text: "one", metadata: {} };
  expect(distinctDocumentResults([
    base,
    { ...base, chunkId: 2, score: 0.9 },
    { ...base, chunkId: 3, documentPath: "two.md", score: 0.8 }
  ], 5).map((result) => result.documentPath)).toEqual(["one.md", "two.md"]);
});

test("org-brain ingestion filters lifecycle state, evaluates retrieval, and removes deleted entries", async () => {
  const root = mkdtempSync(join(tmpdir(), "turbosearch-org-brain-"));
  const decisions = join(root, "knowledge", "silixon-core", "decisions");
  mkdirSync(decisions, { recursive: true });
  const activePath = join(decisions, `${activeId}.md`);
  const inactivePath = join(decisions, `${inactiveId}.md`);
  writeFileSync(activePath, entry({
    id: activeId,
    name: "release-policy",
    description: "Production release branch policy",
    state: "active",
    body: "## Decision\n\nDeployments release by merging the dev integration branch into prod."
  }));
  writeFileSync(inactivePath, entry({
    id: inactiveId,
    name: "old-policy",
    description: "Old release policy",
    state: "superseded",
    body: "## Decision\n\nDeployments previously released from main."
  }));

  const dimensions = 64;
  const engine = new SearchEngine(createConfig({
    dbPath: join(root, "index.sqlite"),
    embeddingDimensions: dimensions,
    embeddingProvider: "hash",
    chunkTokens: 50,
    chunkOverlapTokens: 5,
    ocrProvider: "none",
    transcriptionProvider: "none"
  }), new HashEmbeddingProvider(dimensions));

  try {
    const first = await engine.ingestOrgBrain(root);
    expect(first.documents).toBe(2);
    expect((await engine.stats()).documents).toBe(2);

    const results = await engine.search("production deployments release dev prod", {
      backend: "hybrid",
      k: 5,
      filters: { corpus: "org-brain", project: "silixon-core", state: "active" }
    });
    expect(results).toHaveLength(1);
    const metadata = results[0]?.metadata.orgBrain as Record<string, unknown>;
    expect(metadata.id).toBe(activeId);
    expect(results[0]?.metadata.embeddingText).toBeUndefined();

    const report = await evaluateRetrieval(engine, {
      version: 1,
      cases: [{ query: "production deployment release branch", expectedEntryIds: [activeId] }]
    }, { backend: "hybrid", k: 3 });
    expect(report.recallAtK).toBe(1);
    expect(report.meanReciprocalRank).toBe(1);

    const unchanged = await engine.ingestOrgBrain(root);
    expect(unchanged.unchanged).toBe(2);
    unlinkSync(inactivePath);
    const synced = await engine.ingestOrgBrain(root);
    expect(synced.removed).toBe(1);
    expect((await engine.stats()).documents).toBe(1);
  } finally {
    await engine.close();
    rmSync(root, { recursive: true, force: true });
  }
});

function entry(options: {
  id: string;
  name: string;
  description: string;
  state: "active" | "superseded";
  body: string;
}): string {
  return `---
schema_version: 1
id: "${options.id}"
name: "${options.name}"
project: "silixon-core"
type: "decision"
author: "reyneill"
date: "2026-07-14"
verified_at: "2026-07-14T12:00:00Z"
state: "${options.state}"
supersedes: []
disputes: []
expires_at: null
description: "${options.description}"
tags: ["release", "git"]
---

${options.body}
`;
}
