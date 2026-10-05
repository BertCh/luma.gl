// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  forEachPair,
  getDistanceBin,
  getIncludedRows,
  isNearBinEdge,
  type PairStatisticsFrame,
  type PairStatisticsScene
} from './pair-statistics-oracle';

/** CPU result of {@link computeSpatialCorrelogramOnCPU}. */
export type SpatialCorrelogramOracleResult = {
  moransI: number[];
  expectedI: number[];
  varianceI: number[];
  zScores: number[];
  pValues: number[];
  pairCounts: number[];
  /** Per band: pairs within rounding of an annulus edge that the GPU may bin next door. */
  edgePairCounts: number[];
  /** `[firstPeakBand, maximumBand]`, `0xffffffff` when none. */
  peakBands: number[];
  /** `[n, mean, variance, sumOfSquares, kurtosis]`. */
  statistics: number[];
  /** `n * max|c|^2 * 2^-25 / M2`: the fixed-point bound of `|I|` (sum over pairs averaged). */
  moransITolerance: number;
};

/** Two-sided standard normal p-value in f64 (complementary error function by continued fraction). */
export function getTwoSidedNormalPValue(z: number): number {
  const x = Math.abs(z) / Math.SQRT2;
  // erfc via the Numerical Recipes erfcc fit, accurate to 1.2e-7 relative.
  const t = 1 / (1 + 0.5 * x);
  const exponent =
    -x * x -
    1.26551223 +
    t *
      (1.00002368 +
        t *
          (0.37409196 +
            t *
              (0.09678418 +
                t *
                  (-0.18628806 +
                    t *
                      (0.27886807 +
                        t *
                          (-1.13520398 +
                            t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
  return Math.min(t * Math.exp(exponent), 1);
}

/**
 * f64 global Moran's I per distance band with the `GPUSpatialCorrelogram` inclusion and band rules
 * (offsets, distances and annuli in f32 like the GPU), and esda's analytic moments.
 */
export function computeSpatialCorrelogramOnCPU(
  scene: PairStatisticsScene & {values: Float32Array},
  frame: PairStatisticsFrame & {varianceAssumption?: 'normality' | 'randomization'},
  bandCount: number,
  bandMode: 'cumulative' | 'annulus' = 'cumulative'
): SpatialCorrelogramOracleResult {
  const included = getIncludedRows(scene, frame);
  const n = included.length;
  const mean = included.reduce((sum, row) => sum + scene.values[row], 0) / n;
  const centered = new Map<number, number>();
  let sumOfSquares = 0;
  let sumOfFourth = 0;
  let maximumAbsolute = 0;
  for (const row of included) {
    const value = scene.values[row] - mean;
    centered.set(row, value);
    sumOfSquares += value * value;
    sumOfFourth += value ** 4;
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
  }
  const annulusPairs = new Array<number>(bandCount).fill(0);
  const annulusProducts = new Array<number>(bandCount).fill(0);
  const edgePairCounts = new Array<number>(bandCount).fill(0);
  const degrees = new Map<number, number[]>();
  forEachPair(scene, frame, 'ordered', ({focus, neighbor, distance}) => {
    const band = getDistanceBin(distance, frame.maximumDistance, bandCount);
    annulusPairs[band]++;
    annulusProducts[band] += centered.get(focus)! * centered.get(neighbor)!;
    if (isNearBinEdge(distance, frame.maximumDistance, bandCount)) {
      const edge = Math.round((distance / frame.maximumDistance) * bandCount);
      // Ordered visits count each pair twice; halve when reporting unordered counts.
      edgePairCounts[edge - 1] += 0.5;
      if (bandMode === 'annulus') {
        edgePairCounts[edge] += 0.5;
      }
    }
    let focusDegrees = degrees.get(focus);
    if (!focusDegrees) {
      focusDegrees = new Array<number>(bandCount).fill(0);
      degrees.set(focus, focusDegrees);
    }
    focusDegrees[band]++;
  });
  const randomization = (frame.varianceAssumption ?? 'randomization') === 'randomization';
  const kurtosis = (n * sumOfFourth) / (sumOfSquares * sumOfSquares);
  const expected = n >= 2 ? -1 / (n - 1) : NaN;
  const result: SpatialCorrelogramOracleResult = {
    moransI: [],
    expectedI: [],
    varianceI: [],
    zScores: [],
    pValues: [],
    pairCounts: [],
    edgePairCounts,
    peakBands: [],
    statistics: [n, mean, sumOfSquares / n, sumOfSquares, kurtosis],
    moransITolerance: (n * maximumAbsolute * maximumAbsolute * 2 ** -25) / sumOfSquares
  };
  for (let band = 0; band < bandCount; band++) {
    const first = bandMode === 'cumulative' ? 0 : band;
    let s0 = 0;
    let products = 0;
    for (let annulus = first; annulus <= band; annulus++) {
      s0 += annulusPairs[annulus];
      products += annulusProducts[annulus];
    }
    let degreeSquares = 0;
    for (const focusDegrees of degrees.values()) {
      let degree = 0;
      for (let annulus = first; annulus <= band; annulus++) {
        degree += focusDegrees[annulus];
      }
      degreeSquares += degree * degree;
    }
    result.pairCounts.push(s0 / 2);
    let moransI = NaN;
    let variance = NaN;
    if (s0 > 0 && n >= 3 && sumOfSquares > 0) {
      moransI = (n / s0) * (products / sumOfSquares);
      const s1 = 2 * s0;
      const s2 = 4 * degreeSquares;
      let magnitude = 0;
      if (!randomization) {
        const firstTerm = ((n * n) / (n * n - 1)) * (s1 / (s0 * s0));
        const secondTerm = (n / (n * n - 1)) * (s2 / (s0 * s0));
        const thirdTerm = 3 / (n * n - 1);
        variance = firstTerm - secondTerm + thirdTerm - expected ** 2;
        magnitude = firstTerm + secondTerm + thirdTerm + expected ** 2;
      } else if (n >= 4) {
        const firstTerm = n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0 * s0);
        const secondTerm = kurtosis * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0 * s0);
        const denominator = (n - 1) * (n - 2) * (n - 3) * s0 * s0;
        variance = (firstTerm - secondTerm) / denominator - expected ** 2;
        magnitude = (Math.abs(firstTerm) + Math.abs(secondTerm)) / denominator + expected ** 2;
      }
      // Mirrors the GPU rule: a variance within 1e-4 of its terms' magnitude is undefined.
      if (!(variance > 1e-4 * magnitude)) {
        variance = NaN;
      }
    }
    const zScore = (moransI - expected) / Math.sqrt(variance);
    result.moransI.push(moransI);
    result.expectedI.push(expected);
    result.varianceI.push(variance);
    result.zScores.push(zScore);
    result.pValues.push(Number.isFinite(zScore) ? getTwoSidedNormalPValue(zScore) : NaN);
  }
  let firstPeak = 0xffffffff;
  let maximumBand = 0xffffffff;
  let maximumZ = 0;
  for (let band = 0; band < bandCount; band++) {
    const z = result.zScores[band];
    if (!Number.isFinite(z)) {
      continue;
    }
    if (maximumBand === 0xffffffff || z > maximumZ) {
      maximumBand = band;
      maximumZ = z;
    }
    if (firstPeak === 0xffffffff && band > 0 && band + 1 < bandCount) {
      const previous = result.zScores[band - 1];
      const next = result.zScores[band + 1];
      if (Number.isFinite(previous) && Number.isFinite(next) && z > previous && z > next) {
        firstPeak = band;
      }
    }
  }
  result.peakBands = [firstPeak, maximumBand];
  return result;
}
