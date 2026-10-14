// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getIncludedRows,
  type PairStatisticsFrame,
  type PairStatisticsScene
} from './pair-statistics-oracle';

const fround = Math.fround;

/** Tail of the Clark-Evans and quadrat tests. */
export type PointPatternOracleAlternative = 'two-sided' | 'greater' | 'lesser';

/** `'less'` is an alias of `'lesser'`. */
function resolveAlternative(
  alternative: PointPatternOracleAlternative | 'less' | undefined
): PointPatternOracleAlternative {
  return alternative === 'less' ? 'lesser' : (alternative ?? 'two-sided');
}

/** Per-frame options of the Clark-Evans and quadrat p-values. */
export type PointPatternOracleOptions = {
  /** Defaults to `'donnelly'`, like `getGPUPointPatternIndicesParameterValues`. */
  clarkEvansCorrection?: 'none' | 'donnelly';
  /** Defaults to `'two-sided'`. */
  alternative?: PointPatternOracleAlternative | 'less';
};

/**
 * f64 lower and upper tails `[P(a, x), Q(a, x)]` of the regularized incomplete gamma function
 * (series for `x < a + 1`, Lentz continued fraction above; Numerical Recipes `gser`/`gcf`).
 * Verified against scipy.special.gammainc / gammaincc in the node spec.
 */
export function getGammaTailsOnCPU(a: number, x: number): [number, number] {
  if (x <= 0) {
    return [0, 1];
  }
  const logGamma = getLogGammaOnCPU(a);
  const prefix = Math.exp(a * Math.log(x) - x - logGamma);
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let denominator = a + 1; Math.abs(term) > Math.abs(sum) * 1e-16; denominator++) {
      term *= x / denominator;
      sum += term;
    }
    return [sum * prefix, 1 - sum * prefix];
  }
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let iteration = 1; iteration < 100000; iteration++) {
    const an = -iteration * (iteration - a);
    b += 2;
    d = an * d + b;
    d = Math.abs(d) < tiny ? tiny : d;
    c = b + an / c;
    c = Math.abs(c) < tiny ? tiny : c;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) {
      break;
    }
  }
  return [1 - prefix * h, prefix * h];
}

/** Lanczos (g = 7, n = 9) log-gamma in f64. */
function getLogGammaOnCPU(value: number): number {
  const coefficients = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7
  ];
  const x = value - 1;
  let sum = coefficients[0];
  for (let index = 1; index < 9; index++) {
    sum += coefficients[index] / (x + index);
  }
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
}

/** Normal p-value of a standardized statistic. */
export function getNormalPValueOnCPU(
  z: number,
  alternative: PointPatternOracleAlternative
): number {
  // erfc(|z| / sqrt(2)) = Q(1/2, z^2 / 2) is the two-sided p-value; each tail is half of it.
  const twoSided = getGammaTailsOnCPU(0.5, (z * z) / 2)[1];
  if (alternative === 'two-sided') {
    return twoSided;
  }
  const smallTail = twoSided / 2;
  const isUpper = alternative === 'greater';
  return z >= 0 === isUpper ? smallTail : 1 - smallTail;
}

/** Chi-square p-value for the tail, exact in f64 for every degree of freedom. */
export function getChiSquarePValueOnCPU(
  x: number,
  degreesOfFreedom: number,
  alternative: PointPatternOracleAlternative
): number {
  const [lower, upper] = getGammaTailsOnCPU(degreesOfFreedom / 2, x / 2);
  if (alternative === 'greater') {
    return upper;
  }
  if (alternative === 'lesser') {
    return lower;
  }
  return Math.min(1, 2 * Math.min(lower, upper));
}

/** CPU result of {@link computePointPatternIndicesOnCPU}. */
export type PointPatternIndicesOracleResult = {
  /** Per row: f32 nearest-neighbor distance, NaN for excluded rows and when `n < 2`. */
  nearestNeighborDistances: number[];
  /** Per row: smallest-index nearest neighbor, `0xffffffff` when none. */
  nearestNeighborIds: number[];
  /** `[n, observed, expected, ratio, standardError, zScore, pValue]`. */
  clarkEvans: number[];
  /** Row-major quadrat counts (empty without a quadrat grid). */
  quadratCounts: number[];
  /** `[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom, pValue]`. */
  quadratStatistics: number[];
};

/**
 * f64 Clark-Evans index and quadrat statistics with the `GPUPointPatternIndices` inclusion rule,
 * f32 squared distances (ties to the smallest row index) and f32 quadrat binning.
 */
export function computePointPatternIndicesOnCPU(
  scene: PairStatisticsScene,
  frame: PairStatisticsFrame & PointPatternOracleOptions,
  quadratGrid: readonly [number, number] = [1, 1]
): PointPatternIndicesOracleResult {
  const [minX, minY, maxX, maxY] = frame.bounds.map(fround);
  const rows = scene.positions.length / 2;
  const included = getIncludedRows(scene, frame);
  const n = included.length;
  const width = fround(maxX - minX);
  const height = fround(maxY - minY);
  const area = fround(width * height);
  const nearestNeighborDistances = new Array<number>(rows).fill(NaN);
  const nearestNeighborIds = new Array<number>(rows).fill(0xffffffff);
  let distanceSum = 0;
  for (const focus of included) {
    const x = scene.positions[focus * 2];
    const y = scene.positions[focus * 2 + 1];
    let bestSquared = Infinity;
    let bestId = -1;
    for (const neighbor of included) {
      if (neighbor === focus) {
        continue;
      }
      const deltaX = fround(scene.positions[neighbor * 2] - x);
      const deltaY = fround(scene.positions[neighbor * 2 + 1] - y);
      const squared = fround(fround(deltaX * deltaX) + fround(deltaY * deltaY));
      if (squared < bestSquared) {
        bestSquared = squared;
        bestId = neighbor;
      }
    }
    if (bestId >= 0) {
      nearestNeighborDistances[focus] = fround(Math.sqrt(bestSquared));
      nearestNeighborIds[focus] = bestId;
      distanceSum += nearestNeighborDistances[focus];
    }
  }
  const alternative = resolveAlternative(frame.alternative);
  let clarkEvans = [n, NaN, NaN, NaN, NaN, NaN, NaN];
  if (n >= 2 && area > 0) {
    const observed = distanceSum / n;
    let expected = 0.5 / Math.sqrt(n / area);
    let standardError = 0.26136 / Math.sqrt((n * n) / area);
    if ((frame.clarkEvansCorrection ?? 'donnelly') === 'donnelly') {
      // Donnelly (1978), rectangular window of perimeter P.
      const perimeter = 2 * (width + height);
      expected += ((0.0514 + 0.041 / Math.sqrt(n)) * perimeter) / n;
      standardError = Math.sqrt(
        (0.0703 * area) / (n * n) + 0.037 * perimeter * Math.sqrt(area / n ** 5)
      );
    }
    const zScore = (observed - expected) / standardError;
    clarkEvans = [
      n,
      observed,
      expected,
      observed / expected,
      standardError,
      zScore,
      getNormalPValueOnCPU(zScore, alternative)
    ];
  }

  const [columns, quadratRows] = quadratGrid;
  const quadratCounts = new Array<number>(columns * quadratRows).fill(0);
  for (const row of included) {
    const x = scene.positions[row * 2];
    const y = scene.positions[row * 2 + 1];
    const column =
      width > 0
        ? Math.min(Math.floor(fround(fround(fround(x - minX) / width) * columns)), columns - 1)
        : 0;
    const quadratRow =
      height > 0
        ? Math.min(
            Math.floor(fround(fround(fround(y - minY) / height) * quadratRows)),
            quadratRows - 1
          )
        : 0;
    quadratCounts[quadratRow * columns + column]++;
  }
  const m = quadratCounts.length;
  const mean = n / m;
  const squares = quadratCounts.reduce((sum, count) => sum + (count - mean) ** 2, 0);
  const variance = m > 1 ? squares / (m - 1) : NaN;
  const ratio = m > 1 && mean > 0 ? variance / mean : NaN;
  return {
    nearestNeighborDistances,
    nearestNeighborIds,
    clarkEvans,
    quadratCounts,
    quadratStatistics: [
      m,
      mean,
      variance,
      ratio,
      (m - 1) * ratio,
      m - 1,
      m > 1 && Number.isFinite(ratio)
        ? getChiSquarePValueOnCPU((m - 1) * ratio, m - 1, alternative)
        : NaN
    ]
  };
}
