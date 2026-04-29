import { createHash } from "node:crypto";

export function sha256Text(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

export function hashToUint32(value: string): number {
  const digest = createHash("sha256").update(value).digest();
  return digest.readUInt32LE(0);
}
