import { readFile, readdir } from "node:fs/promises";
import { join, relative, resolve, sep } from "node:path";
import { chunkDocument, type ChunkOptions } from "../chunking/chunker.ts";
import type { ExtractedDocument, TextChunk } from "../types.ts";

export const ORG_BRAIN_CORPUS = "org-brain";

const entryTypes = ["finding", "learning", "reference", "decision", "status"] as const;
const entryStates = ["active", "superseded", "disputed", "retracted"] as const;
const directoryForType: Record<OrgBrainEntryType, string> = {
  finding: "findings",
  learning: "learnings",
  reference: "references",
  decision: "decisions",
  status: "status"
};

export type OrgBrainEntryType = typeof entryTypes[number];
export type OrgBrainEntryState = typeof entryStates[number] | "expired";

export type OrgBrainEntry = {
  id: string;
  name: string;
  project: string;
  type: OrgBrainEntryType;
  author: string;
  date: string;
  verifiedAt: string;
  state: OrgBrainEntryState;
  declaredState: typeof entryStates[number];
  supersedes: string[];
  disputes: string[];
  expiresAt: string | null;
  description: string;
  tags: string[];
  sourcePath: string;
  bodyStartOffset: number;
  body: string;
};

export type OrgBrainDocument = {
  document: ExtractedDocument;
  chunks: TextChunk[];
  entry: OrgBrainEntry;
};

export async function collectOrgBrainEntryPaths(root: string): Promise<string[]> {
  const knowledge = join(root, "knowledge");
  const paths: string[] = [];
  await walkMarkdown(knowledge, paths);
  return paths.sort();
}

export async function loadOrgBrainDocument(
  root: string,
  path: string,
  options: ChunkOptions
): Promise<OrgBrainDocument> {
  const content = await readFile(path, "utf8");
  return parseOrgBrainDocument(root, path, content, options);
}

export function parseOrgBrainDocument(
  root: string,
  path: string,
  content: string,
  options: ChunkOptions
): OrgBrainDocument {
  const sourcePath = normalizeRelativePath(relative(resolve(root), resolve(path)));
  const entry = parseOrgBrainEntry(sourcePath, content);
  validateSourcePath(entry);
  const chunks = chunkOrgBrainEntry(entry, content, options);
  if (chunks.length === 0) throw new Error(`${sourcePath}: entry body produced no searchable chunks`);

  return {
    entry,
    document: {
      path: sourcePath,
      kind: "text",
      // Hash the complete file so frontmatter-only lifecycle changes invalidate the index.
      text: content,
      metadata: {
        corpus: ORG_BRAIN_CORPUS,
        orgBrain: entryMetadata(entry)
      }
    },
    chunks
  };
}

export function parseOrgBrainEntry(sourcePath: string, content: string): OrgBrainEntry {
  if (!content.startsWith("---\n")) throw new Error(`${sourcePath}: missing opening frontmatter delimiter`);
  const closing = content.indexOf("\n---\n", 4);
  if (closing === -1) throw new Error(`${sourcePath}: missing closing frontmatter delimiter`);

  const frontmatter = content.slice(4, closing);
  const values = new Map<string, unknown>();
  for (const rawLine of frontmatter.split("\n")) {
    const colon = rawLine.indexOf(":");
    if (colon <= 0) throw new Error(`${sourcePath}: invalid frontmatter line ${JSON.stringify(rawLine)}`);
    const key = rawLine.slice(0, colon).trim();
    const rawValue = rawLine.slice(colon + 1).trim();
    values.set(key, parseFrontmatterValue(sourcePath, key, rawValue));
  }

  const type = requiredEnum(values, sourcePath, "type", entryTypes);
  const declaredState = requiredEnum(values, sourcePath, "state", entryStates);
  const expiresAt = nullableString(values, sourcePath, "expires_at");
  const state: OrgBrainEntryState = declaredState === "active" && type === "status" && expiresAt && expiresAt < todayUtc()
    ? "expired"
    : declaredState;
  const bodyStartOffset = closing + "\n---\n".length;

  return {
    id: requiredString(values, sourcePath, "id"),
    name: requiredString(values, sourcePath, "name"),
    project: requiredString(values, sourcePath, "project"),
    type,
    author: requiredString(values, sourcePath, "author"),
    date: requiredString(values, sourcePath, "date"),
    verifiedAt: requiredString(values, sourcePath, "verified_at"),
    state,
    declaredState,
    supersedes: requiredStringArray(values, sourcePath, "supersedes"),
    disputes: requiredStringArray(values, sourcePath, "disputes"),
    expiresAt,
    description: requiredString(values, sourcePath, "description"),
    tags: requiredStringArray(values, sourcePath, "tags"),
    sourcePath,
    bodyStartOffset,
    body: content.slice(bodyStartOffset).trim()
  };
}

function chunkOrgBrainEntry(entry: OrgBrainEntry, content: string, options: ChunkOptions): TextChunk[] {
  const sections = splitSections(content, entry.bodyStartOffset, entry.description);
  const chunks: TextChunk[] = [];

  for (const section of sections) {
    const sectionText = content.slice(section.startOffset, section.endOffset);
    const leading = sectionText.length - sectionText.trimStart().length;
    const normalized = sectionText.trim();
    if (!normalized) continue;

    const sectionChunks = chunkDocument({
      path: entry.sourcePath,
      kind: "text",
      text: normalized,
      metadata: {}
    }, options);

    for (const chunk of sectionChunks) {
      const startOffset = section.startOffset + leading + chunk.startOffset;
      const endOffset = section.startOffset + leading + chunk.endOffset;
      const lineStart = lineNumberAt(content, startOffset);
      const lineEnd = lineNumberAt(content, Math.max(startOffset, endOffset - 1));
      const heading = section.heading;
      chunks.push({
        documentPath: entry.sourcePath,
        ordinal: chunks.length,
        text: chunk.text,
        startOffset,
        endOffset,
        metadata: {
          corpus: ORG_BRAIN_CORPUS,
          embeddingText: embeddingText(entry, heading, chunk.text),
          orgBrain: {
            ...entryMetadata(entry),
            heading,
            lineStart,
            lineEnd
          },
          sourceKind: "text"
        }
      });
    }
  }

  return chunks;
}

function splitSections(content: string, bodyStartOffset: number, fallbackHeading: string): Array<{
  heading: string;
  startOffset: number;
  endOffset: number;
}> {
  const body = content.slice(bodyStartOffset);
  const headings = Array.from(body.matchAll(/^#{1,6}\s+(.+)$/gm), (match) => ({
    heading: match[1]?.trim() || fallbackHeading,
    offset: bodyStartOffset + (match.index ?? 0)
  }));
  const sections: Array<{ heading: string; startOffset: number; endOffset: number }> = [];

  if (headings.length === 0) {
    return [{ heading: fallbackHeading, startOffset: bodyStartOffset, endOffset: content.length }];
  }
  if ((headings[0]?.offset ?? bodyStartOffset) > bodyStartOffset && content.slice(bodyStartOffset, headings[0]?.offset).trim()) {
    sections.push({ heading: fallbackHeading, startOffset: bodyStartOffset, endOffset: headings[0]?.offset ?? content.length });
  }
  for (let index = 0; index < headings.length; index += 1) {
    const heading = headings[index];
    if (!heading) continue;
    sections.push({
      heading: heading.heading,
      startOffset: heading.offset,
      endOffset: headings[index + 1]?.offset ?? content.length
    });
  }
  return sections;
}

function entryMetadata(entry: OrgBrainEntry): Record<string, unknown> {
  return {
    id: entry.id,
    name: entry.name,
    project: entry.project,
    type: entry.type,
    author: entry.author,
    date: entry.date,
    verifiedAt: entry.verifiedAt,
    state: entry.state,
    declaredState: entry.declaredState,
    supersedes: entry.supersedes,
    disputes: entry.disputes,
    expiresAt: entry.expiresAt,
    description: entry.description,
    tags: entry.tags,
    sourcePath: entry.sourcePath
  };
}

function embeddingText(entry: OrgBrainEntry, heading: string, text: string): string {
  return [
    `Title: ${entry.description}`,
    `Project: ${entry.project}`,
    `Type: ${entry.type}`,
    entry.tags.length > 0 ? `Tags: ${entry.tags.join(", ")}` : "",
    `Section: ${heading}`,
    text
  ].filter(Boolean).join("\n");
}

function validateSourcePath(entry: OrgBrainEntry): void {
  const parts = entry.sourcePath.split("/");
  if (parts.length !== 4 || parts[0] !== "knowledge") {
    throw new Error(`${entry.sourcePath}: expected knowledge/<project>/<type-directory>/<file>.md`);
  }
  if (parts[1] !== entry.project) throw new Error(`${entry.sourcePath}: project does not match its directory`);
  if (parts[2] !== directoryForType[entry.type]) throw new Error(`${entry.sourcePath}: type does not match its directory`);
  if (!parts[3]?.endsWith(".md")) throw new Error(`${entry.sourcePath}: entry must be Markdown`);
}

function parseFrontmatterValue(sourcePath: string, key: string, value: string): unknown {
  if (value === "null") return null;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    throw new Error(`${sourcePath}: ${key} must use JSON-compatible scalar or array syntax`);
  }
}

function requiredString(values: Map<string, unknown>, sourcePath: string, key: string): string {
  const value = values.get(key);
  if (typeof value !== "string" || !value) throw new Error(`${sourcePath}: ${key} must be a non-empty string`);
  return value;
}

function nullableString(values: Map<string, unknown>, sourcePath: string, key: string): string | null {
  const value = values.get(key);
  if (value === null) return null;
  if (typeof value !== "string" || !value) throw new Error(`${sourcePath}: ${key} must be a string or null`);
  return value;
}

function requiredStringArray(values: Map<string, unknown>, sourcePath: string, key: string): string[] {
  const value = values.get(key);
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new Error(`${sourcePath}: ${key} must be an array of strings`);
  }
  return value;
}

function requiredEnum<const T extends readonly string[]>(
  values: Map<string, unknown>,
  sourcePath: string,
  key: string,
  allowed: T
): T[number] {
  const value = requiredString(values, sourcePath, key);
  if (!allowed.includes(value)) throw new Error(`${sourcePath}: unsupported ${key} ${JSON.stringify(value)}`);
  return value as T[number];
}

function lineNumberAt(content: string, offset: number): number {
  let line = 1;
  for (let index = 0; index < Math.min(offset, content.length); index += 1) {
    if (content.charCodeAt(index) === 10) line += 1;
  }
  return line;
}

function normalizeRelativePath(path: string): string {
  return sep === "/" ? path : path.split(sep).join("/");
}

function todayUtc(): string {
  return new Date().toISOString().slice(0, 10);
}

async function walkMarkdown(directory: string, paths: string[]): Promise<void> {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) await walkMarkdown(path, paths);
    else if (entry.isFile() && entry.name.endsWith(".md")) paths.push(path);
  }
}
