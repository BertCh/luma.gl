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

/** CPU result of {@link computeVariogramOnCPU}, sector-major like `GPUVariogram`. */
export type VariogramOracleResult = {
  pairCounts: number[];
  /** Per bin: pairs within rounding of one of its edges, which the GPU may bin next door. */
  edgePairCounts: number[];
  semivariances: number[];
  meanDistances: number[];
  robustSemivariances: number[];
  /** `[n, mean, variance, minimum, maximum]`. */
  statistics: number[];
  /** `(max - min)^2 * 2^-26`: the fixed-point error bound of `gamma`. */
  semivarianceTolerance: number;
};

/**
 * f64 empirical semivariogram with the `GPUVariogram` inclusion, binning and sector rules (offsets,
 * distances and bins in f32 like the GPU).
 */
export function computeVariogramOnCPU(
  scene: PairStatisticsScene & {values: Float32Array},
  frame: PairStatisticsFrame & {azimuthOffset?: number},
  lagCount: number,
  directionCount = 1
): VariogramOracleResult {
  const binCount = lagCount * directionCount;
  const pairCounts = new Array<number>(binCount).fill(0);
  const squareSums = new Array<number>(binCount).fill(0);
  const distanceSums = new Array<number>(binCount).fill(0);
  const robustSums = new Array<number>(binCount).fill(0);
  const edgePairCounts = new Array<number>(binCount).fill(0);
  const offset = Math.fround(frame.azimuthOffset ?? 0);
  forEachPair(scene, frame, 'unordered', ({focus, neighbor, deltaX, deltaY, distance}) => {
    const lag = getDistanceBin(distance, frame.maximumDistance, lagCount);
    let sector = 0;
    if (directionCount > 1) {
      const angle = Math.atan2(deltaY, deltaX) - offset;
      const reduced = angle - Math.floor(angle / Math.PI) * Math.PI;
      sector = Math.min(
        Math.floor((Math.max(reduced, 0) / Math.PI) * directionCount),
        directionCount - 1
      );
    }
    const bin = sector * lagCount + lag;
    const difference = scene.values[neighbor] - scene.values[focus];
    pairCounts[bin]++;
    if (isNearBinEdge(distance, frame.maximumDistance, lagCount)) {
      edgePairCounts[bin]++;
      const scaled = Math.round((distance / frame.maximumDistance) * lagCount);
      const other = sector * lagCount + (scaled === lag ? lag - 1 : lag + 1);
      edgePairCounts[other]++;
    }
    squareSums[bin] += 0.5 * difference * difference;
    distanceSums[bin] += distance;
    robustSums[bin] += Math.sqrt(Math.abs(difference));
  });
  const included = getIncludedRows(scene, frame);
  const values = included.map(row => scene.values[row]);
  const n = values.length;
  const mean = values.reduce((sum, value) => sum + value, 0) / n;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / n;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  return {
    pairCounts,
    edgePairCounts,
    semivariances: pairCounts.map((count, bin) => (count ? squareSums[bin] / count : NaN)),
    meanDistances: pairCounts.map((count, bin) => (count ? distanceSums[bin] / count : NaN)),
    robustSemivariances: pairCounts.map((count, bin) =>
      count ? (robustSums[bin] / count) ** 4 / (2 * (0.457 + 0.494 / count)) : NaN
    ),
    statistics: [n, mean, variance, minimum, maximum],
    semivarianceTolerance: (maximum - minimum) ** 2 * 2 ** -26
  };
}
