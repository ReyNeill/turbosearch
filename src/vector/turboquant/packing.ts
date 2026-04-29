export function packCodes(codes: Uint8Array, bits: number): Uint8Array {
  const output = new Uint8Array(Math.ceil((codes.length * bits) / 8));
  let bitOffset = 0;

  for (const code of codes) {
    writeBits(output, bitOffset, bits, code);
    bitOffset += bits;
  }

  return output;
}

export function getPackedCode(bytes: Uint8Array, index: number, bits: number): number {
  const bitOffset = index * bits;
  const byteOffset = Math.floor(bitOffset / 8);
  const shift = bitOffset % 8;
  const mask = (1 << bits) - 1;
  const current = bytes[byteOffset] ?? 0;
  const next = bytes[byteOffset + 1] ?? 0;
  return ((current >> shift) | (next << (8 - shift))) & mask;
}

function writeBits(bytes: Uint8Array, bitOffset: number, bits: number, value: number): void {
  const byteOffset = Math.floor(bitOffset / 8);
  const shift = bitOffset % 8;
  bytes[byteOffset] = (bytes[byteOffset] ?? 0) | ((value << shift) & 0xff);

  const overflow = shift + bits - 8;
  if (overflow > 0) {
    bytes[byteOffset + 1] = (bytes[byteOffset + 1] ?? 0) | (value >> (bits - overflow));
  }
}
