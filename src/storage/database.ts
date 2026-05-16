import type { AppConfig } from "../config.ts";
import type { SearchStorage } from "./types.ts";
import { PostgresSearchDatabase } from "./postgres-database.ts";
import { SqliteSearchDatabase } from "./sqlite-database.ts";

export function createSearchStorage(config: AppConfig): SearchStorage {
  if (config.storageProvider === "postgres") {
    return new PostgresSearchDatabase({
      url: config.postgresUrl,
      usePsqlBm25s: config.postgresUsePsqlBm25s
    });
  }

  return new SqliteSearchDatabase(config.dbPath);
}

export type { SearchStorage };
