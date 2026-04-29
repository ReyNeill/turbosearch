#!/usr/bin/env bun
import { createConfig, loadEnvLocal, type AppConfig } from "./config.ts";
import { createEmbeddingProvider } from "./embeddings/provider.ts";
import { extractEvidence } from "./ingest/extractors.ts";
import { SearchEngine } from "./search/search-engine.ts";
import type { EmbeddingProviderName, SearchBackendName } from "./types.ts";

type ParsedArgs = {
  command: string | undefined;
  positionals: string[];
  flags: Record<string, string | boolean>;
};

type JsonPayload = Record<string, unknown>;

async function main(): Promise<void> {
  loadEnvLocal();
  const args = parseArgs(process.argv.slice(2));
  const payload = await readJsonPayload(args);
  const command = args.command ?? getString(payload, "command");

  if (!command || args.flags.help === true || args.flags.h === true) {
    printHelp();
    return;
  }

  const config = configFromArgs(args, payload);
  const provider = createEmbeddingProvider(config);
  const engine = new SearchEngine(config, provider);

  try {
    if (command === "ingest") {
      await runIngest(engine, args, payload);
      return;
    }

    if (command === "search") {
      await runSearch(engine, args, payload);
      return;
    }

    if (command === "extract") {
      await runExtract(args, payload);
      return;
    }

    if (command === "stats") {
      const stats = engine.stats();
      writeJson({ ok: true, command: "stats", stats });
      return;
    }

    throw new Error(`Unknown command: ${command}`);
  } finally {
    engine.close();
  }
}

async function runIngest(engine: SearchEngine, args: ParsedArgs, payload: JsonPayload): Promise<void> {
  const paths = getStringArray(payload, "paths") ?? args.positionals;
  if (paths.length === 0) {
    throw new Error("ingest requires at least one file or directory path.");
  }

  const jsonl = args.flags.jsonl === true || payload.jsonl === true;
  const summary = await engine.ingestPaths(paths, jsonl ? (event) => writeJsonLine(event) : undefined);
  if (!jsonl) writeJson({ ok: true, command: "ingest", summary });
}

async function runSearch(engine: SearchEngine, args: ParsedArgs, payload: JsonPayload): Promise<void> {
  const query = getString(payload, "query") ?? args.positionals.join(" ").trim();
  if (!query) throw new Error("search requires a query.");

  const backend = parseBackend(getValue(args.flags.backend, payload.backend));
  const k = getNumber(payload, "k") ?? Number(args.flags.k ?? 10);
  const results = await engine.search(query, { backend, k });

  writeJson({ ok: true, command: "search", query, backend, k, results });
}

async function runExtract(args: ParsedArgs, payload: JsonPayload): Promise<void> {
  const paths = getStringArray(payload, "paths") ?? maybeSinglePath(args, payload);
  if (paths.length === 0) throw new Error("extract requires a path or paths.");

  const results = [];
  for (const path of paths) {
    results.push(await extractEvidence(path));
  }

  writeJson({ ok: true, command: "extract", results });
}

function configFromArgs(args: ParsedArgs, payload: JsonPayload): AppConfig {
  return createConfig({
    dbPath: getString(payload, "db") ?? String(args.flags.db ?? ".turbosearch/index.sqlite"),
    embeddingProvider: parseEmbeddingProvider(getValue(args.flags.embedding, payload.embedding)),
    turboBits: getNumber(payload, "turboBits") ?? Number(args.flags["turbo-bits"] ?? process.env.TURBOSEARCH_TURBO_BITS ?? 4),
    turboRerankLimit: getNumber(payload, "turboRerank") ?? Number(args.flags["turbo-rerank"] ?? process.env.TURBOSEARCH_TURBO_RERANK_LIMIT ?? 80)
  });
}

function parseEmbeddingProvider(value: string | boolean | undefined): EmbeddingProviderName {
  if (value === "openai") return "openai";
  if (value === "hash" || value === undefined || value === false) return "hash";
  throw new Error(`Unsupported embedding provider: ${String(value)}`);
}

function parseBackend(value: string | boolean | undefined): SearchBackendName {
  if (value === "turbo") return "turbo";
  if (value === "exact" || value === undefined || value === false) return "exact";
  throw new Error(`Unsupported backend: ${String(value)}`);
}

function parseArgs(argv: string[]): ParsedArgs {
  const [command, ...rest] = argv;
  const positionals: string[] = [];
  const flags: Record<string, string | boolean> = {};

  for (let i = 0; i < rest.length; i += 1) {
    const arg = rest[i];
    if (!arg) continue;

    if (arg.startsWith("--")) {
      const [key, inlineValue] = arg.slice(2).split("=", 2);
      if (!key) continue;
      if (inlineValue !== undefined) {
        flags[key] = inlineValue;
        continue;
      }

      const next = rest[i + 1];
      if (next && !next.startsWith("-")) {
        flags[key] = next;
        i += 1;
      } else {
        flags[key] = true;
      }
      continue;
    }

    if (arg.startsWith("-") && arg.length === 2) {
      flags[arg.slice(1)] = true;
      continue;
    }

    positionals.push(arg);
  }

  return { command, positionals, flags };
}

function printHelp(): void {
  console.log(`TurboSearch

Usage:
  bun run src/cli.ts ingest <file-or-directory...> [--db path] [--embedding hash|openai]
  bun run src/cli.ts search <query> [--backend exact|turbo] [--k 10] [--db path]
  bun run src/cli.ts extract <file> [--json]
  bun run src/cli.ts stats [--db path]

JSON mode:
  bun run src/cli.ts ingest --json '{"paths":["docs"],"db":".turbosearch/index.sqlite"}'
  bun run src/cli.ts ingest --jsonl docs
  bun run src/cli.ts search --json '{"query":"TurboQuant search","backend":"turbo","k":5}'
  bun run src/cli.ts extract --json '{"paths":["image.png","video.mp4"]}'

Options:
  --embedding      hash by default; openai requires OPENAI_API_KEY in .env.local
  --backend        exact or turbo
  --json           read a JSON request from the next argument, or stdin when no value is provided
  --jsonl          stream ingestion events as JSON Lines
  --turbo-bits     compressed backend bit width, default 4
  --turbo-rerank   exact rerank candidate count, default 80
`);
}

async function readJsonPayload(args: ParsedArgs): Promise<JsonPayload> {
  const jsonFlag = args.flags.json;
  if (jsonFlag === undefined || jsonFlag === false) return {};

  const body = typeof jsonFlag === "string" ? jsonFlag : await Bun.stdin.text();
  const trimmed = body.trim();
  if (!trimmed) return {};

  const parsed = JSON.parse(trimmed) as unknown;
  if (!isRecord(parsed)) throw new Error("--json payload must be an object.");
  return parsed;
}

function maybeSinglePath(args: ParsedArgs, payload: JsonPayload): string[] {
  const payloadPath = getString(payload, "path");
  if (payloadPath) return [payloadPath];
  return args.positionals;
}

function getString(payload: JsonPayload, key: string): string | undefined {
  const value = payload[key];
  return typeof value === "string" ? value : undefined;
}

function getNumber(payload: JsonPayload, key: string): number | undefined {
  const value = payload[key];
  return typeof value === "number" ? value : undefined;
}

function getStringArray(payload: JsonPayload, key: string): string[] | undefined {
  const value = payload[key];
  if (!Array.isArray(value)) return undefined;
  if (!value.every((item) => typeof item === "string")) {
    throw new Error(`${key} must be an array of strings.`);
  }
  return value;
}

function getValue(primary: unknown, fallback: unknown): string | boolean | undefined {
  if (typeof primary === "string" || typeof primary === "boolean") return primary;
  if (typeof fallback === "string" || typeof fallback === "boolean") return fallback;
  return undefined;
}

function isRecord(value: unknown): value is JsonPayload {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function writeJson(value: unknown): void {
  console.log(JSON.stringify(value, null, 2));
}

function writeJsonLine(value: unknown): void {
  console.log(JSON.stringify(value));
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});
