export type QuantCodebook = {
  bits: number;
  levels: number;
  centroids: Float32Array;
  boundaries: Float32Array;
};

export function createNormalLloydCodebook(bits: number, dimension: number): QuantCodebook {
  if (!Number.isInteger(bits) || bits < 1 || bits > 8) {
    throw new Error("TurboQuant bit width must be an integer between 1 and 8.");
  }

  const levels = 2 ** bits;
  const centroids = new Float32Array(levels);
  const boundaries = new Float32Array(levels + 1);
  boundaries[0] = Number.NEGATIVE_INFINITY;
  boundaries[levels] = Number.POSITIVE_INFINITY;

  for (let i = 0; i < levels; i += 1) {
    centroids[i] = inverseNormalCdf((i + 0.5) / levels);
  }

  for (let iteration = 0; iteration < 60; iteration += 1) {
    for (let i = 1; i < levels; i += 1) {
      boundaries[i] = (centroids[i - 1] + centroids[i]) / 2;
    }

    for (let i = 0; i < levels; i += 1) {
      centroids[i] = conditionalNormalMean(boundaries[i] ?? 0, boundaries[i + 1] ?? 0);
    }
  }

  const scale = 1 / Math.sqrt(dimension);
  for (let i = 0; i < centroids.length; i += 1) centroids[i] *= scale;
  for (let i = 1; i < boundaries.length - 1; i += 1) boundaries[i] *= scale;

  return { bits, levels, centroids, boundaries };
}

export function quantizeScalar(value: number, codebook: QuantCodebook): number {
  let low = 0;
  let high = codebook.levels - 1;

  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    const boundary = codebook.boundaries[mid + 1] ?? Number.POSITIVE_INFINITY;
    if (value < boundary) high = mid;
    else low = mid + 1;
  }

  return low;
}

function conditionalNormalMean(a: number, b: number): number {
  const probability = normalCdf(b) - normalCdf(a);
  if (probability <= 0) return 0;
  return (normalPdf(a) - normalPdf(b)) / probability;
}

function normalPdf(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function normalCdf(x: number): number {
  if (x === Number.NEGATIVE_INFINITY) return 0;
  if (x === Number.POSITIVE_INFINITY) return 1;
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const abs = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * abs);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-abs * abs);
  return sign * y;
}

function inverseNormalCdf(p: number): number {
  if (p <= 0 || p >= 1) throw new Error("p must be in (0, 1).");

  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const low = 0.02425;
  const high = 1 - low;

  if (p < low) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }

  if (p <= high) {
    const q = p - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}
