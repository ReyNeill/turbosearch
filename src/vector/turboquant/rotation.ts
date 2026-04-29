import { hashToUint32 } from "../../utils/hash.ts";

export type RotationPlan = {
  originalDimension: number;
  rotatedDimension: number;
  signs: Int8Array;
};

export function createRotationPlan(originalDimension: number, seed: string): RotationPlan {
  const rotatedDimension = nextPowerOfTwo(originalDimension);
  const signs = new Int8Array(rotatedDimension);

  for (let i = 0; i < rotatedDimension; i += 1) {
    signs[i] = (hashToUint32(`${seed}:${i}`) & 1) === 0 ? 1 : -1;
  }

  return { originalDimension, rotatedDimension, signs };
}

export function rotateVector(vector: Float32Array, plan: RotationPlan): Float32Array {
  const output = new Float32Array(plan.rotatedDimension);

  for (let i = 0; i < plan.originalDimension; i += 1) {
    output[i] = (vector[i] ?? 0) * (plan.signs[i] ?? 1);
  }

  fastHadamard(output);

  const scale = 1 / Math.sqrt(plan.rotatedDimension);
  for (let i = 0; i < output.length; i += 1) output[i] *= scale;
  return output;
}

function fastHadamard(values: Float32Array): void {
  for (let size = 1; size < values.length; size *= 2) {
    const step = size * 2;
    for (let start = 0; start < values.length; start += step) {
      for (let i = start; i < start + size; i += 1) {
        const a = values[i] ?? 0;
        const b = values[i + size] ?? 0;
        values[i] = a + b;
        values[i + size] = a - b;
      }
    }
  }
}

function nextPowerOfTwo(value: number): number {
  let power = 1;
  while (power < value) power *= 2;
  return power;
}
