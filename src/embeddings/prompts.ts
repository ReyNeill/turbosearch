import type { EmbeddingPromptFormat } from "../types.ts";
import type { TextEmbeddingDocument } from "./provider.ts";

// Query and document prompts each model family was trained with. Retrieval quality drops when a
// model sees the wrong one, so the format is chosen with the model, never inferred from its name.

export function formatQuery(format: EmbeddingPromptFormat, query: string, instruction = ""): string {
  if (format === "embeddinggemma") return `task: search result | query: ${query}`;
  if (format === "qwen3" && instruction.trim()) return `Instruct: ${instruction.trim()}\nQuery:${query}`;
  return query;
}

export function formatDocument(format: EmbeddingPromptFormat, document: TextEmbeddingDocument): string {
  if (format === "embeddinggemma") return `title: ${document.title ?? "none"} | text: ${document.text}`;
  return document.title ? `Title: ${document.title}\n${document.text}` : document.text;
}
