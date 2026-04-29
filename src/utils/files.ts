import { readdir, stat } from "node:fs/promises";
import { extname, join } from "node:path";

const ignoredDirectories = new Set([
  ".git",
  "node_modules",
  ".next",
  "dist",
  "build",
  ".turbosearch"
]);

export async function collectInputFiles(paths: string[]): Promise<string[]> {
  const files: string[] = [];

  for (const path of paths) {
    const info = await stat(path);
    if (info.isDirectory()) {
      files.push(...await collectDirectory(path));
    } else if (info.isFile()) {
      files.push(path);
    }
  }

  return files.sort();
}

export function extensionOf(path: string): string {
  return extname(path).toLowerCase();
}

async function collectDirectory(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    if (entry.name.startsWith(".") && entry.name !== ".env.example") continue;
    const path = join(directory, entry.name);

    if (entry.isDirectory()) {
      if (!ignoredDirectories.has(entry.name)) {
        files.push(...await collectDirectory(path));
      }
      continue;
    }

    if (entry.isFile()) files.push(path);
  }

  return files;
}
