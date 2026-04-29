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

  prepareQuery(query: Float32Array): Float32Array {
    return rotateVector(l2Normalize(query), this.rotation);
  }

  scorePreparedQuery(rotatedQuery: Float32Array, encoded: TurboQuantEncoded): number {
    let score = 0;

    for (let i = 0; i < this.rotatedDimension; i += 1) {
      const code = getPackedCode(encoded.code, i, this.bits);
      score += (rotatedQuery[i] ?? 0) * (this.codebook.centroids[code] ?? 0);
    }

    return score * encoded.norm;
  }
}
