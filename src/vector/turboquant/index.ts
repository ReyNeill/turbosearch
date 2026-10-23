import { l2Normalize, vectorNorm } from "../math.ts";
import { createNormalLloydCodebook, quantizeScalar, type QuantCodebook } from "./codebook.ts";
import { packCodes, getPackedCode } from "./packing.ts";
import { createRotationPlan, rotateVector, type RotationPlan } from "./rotation.ts";

export type TurboQuantCodecOptions = {
  dimension: number;
  bits: number;
  seed?: string;
};

export type TurboQuantEncoded = {
  code: Uint8Array;
  norm: number;
};

// A rotated query plus, for bit widths that pack whole codes into a byte (1, 2, 4, 8), one
// 256-entry table per code byte holding that byte's partial dot product. Scoring then costs
// one lookup per byte instead of one unpack and multiply per dimension.
export type TurboQuantQuery = {
  rotated: Float32Array;
  byteTable: Float32Array | null;
};

export class TurboQuantCodec {
  readonly bits: number;
  readonly rotatedDimension: number;
  private readonly rotation: RotationPlan;
  private readonly codebook: QuantCodebook;

  constructor(options: TurboQuantCodecOptions) {
    this.bits = options.bits;
    this.rotation = createRotationPlan(options.dimension, options.seed ?? "turbosearch-v1");
    this.rotatedDimension = this.rotation.rotatedDimension;
    this.codebook = createNormalLloydCodebook(options.bits, this.rotatedDimension);
  }

  encode(vector: Float32Array): TurboQuantEncoded {
    const norm = vectorNorm(vector);
    const normalized = norm === 0 ? vector : l2Normalize(vector);
    const rotated = rotateVector(normalized, this.rotation);
    const codes = new Uint8Array(this.rotatedDimension);

    for (let i = 0; i < rotated.length; i += 1) {
      codes[i] = quantizeScalar(rotated[i] ?? 0, this.codebook);
    }

    return {
      code: packCodes(codes, this.bits),
      norm
    };
  }

  prepareQuery(query: Float32Array): TurboQuantQuery {
    const rotated = rotateVector(l2Normalize(query), this.rotation);
    return { rotated, byteTable: 8 % this.bits === 0 ? this.byteTable(rotated) : null };
  }

  scorePreparedQuery(query: TurboQuantQuery, encoded: TurboQuantEncoded): number {
    const table = query.byteTable;
    let score = 0;

    if (table) {
      for (let byte = 0; byte < encoded.code.length; byte += 1) {
        score += table[byte * 256 + (encoded.code[byte] ?? 0)] ?? 0;
      }
    } else {
      for (let i = 0; i < this.rotatedDimension; i += 1) {
        const code = getPackedCode(encoded.code, i, this.bits);
        score += (query.rotated[i] ?? 0) * (this.codebook.centroids[code] ?? 0);
      }
    }

    return score * encoded.norm;
  }

  private byteTable(rotated: Float32Array): Float32Array {
    const codesPerByte = 8 / this.bits;
    const mask = (1 << this.bits) - 1;
    const bytes = Math.ceil(this.rotatedDimension / codesPerByte);
    const table = new Float32Array(bytes * 256);

    for (let byte = 0; byte < bytes; byte += 1) {
      for (let value = 0; value < 256; value += 1) {
        let sum = 0;
        for (let slot = 0; slot < codesPerByte; slot += 1) {
          const code = (value >> (slot * this.bits)) & mask;
          sum += (rotated[byte * codesPerByte + slot] ?? 0) * (this.codebook.centroids[code] ?? 0);
        }
        table[byte * 256 + value] = sum;
      }
    }

    return table;
  }
}
