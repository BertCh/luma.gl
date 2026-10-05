// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getIncludedRows,
  type PairStatisticsFrame,
  type PairStatisticsScene
} from './pair-statistics-oracle';

const fround = Math.fround;

/** CPU result of {@link computePointPatternIndicesOnCPU}. */
export type PointPatternIndicesOracleResult = {
  /** Per row: f32 nearest-neighbor distance, NaN for excluded rows and when `n < 2`. */
  nearestNeighborDistances: number[];
  /** Per row: smallest-index nearest neighbor, `0xffffffff` when none. */
  nearestNeighborIds: number[];
  /** `[n, observed, expected, ratio, standardError, zScore]`. */
  clarkEvans: number[];
  /** Row-major quadrat counts (empty without a quadrat grid). */
  quadratCounts: number[];
  /** `[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom]`. */
  quadratStatistics: number[];
};

/**
 * f64 Clark-Evans index and quadrat statistics with the `GPUPointPatternIndices` inclusion rule,
 * f32 squared distances (ties to the smallest row index) and f32 quadrat binning.
 */
export function computePointPatternIndicesOnCPU(
  scene: PairStatisticsScene,
  frame: PairStatisticsFrame,
  quadratGrid: readonly [number, number] = [1, 1]
): PointPatternIndicesOracleResult {
  const [minX, minY, maxX, maxY] = frame.bounds.map(fround);
  const rows = scene.positions.length / 2;
  const included = getIncludedRows(scene, frame);
  const n = included.length;
  const area = fround(fround(maxX - minX) * fround(maxY - minY));
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
  let clarkEvans = [n, NaN, NaN, NaN, NaN, NaN];
  if (n >= 2 && area > 0) {
    const observed = distanceSum / n;
    const expected = 0.5 / Math.sqrt(n / area);
    const standardError = 0.26136 / Math.sqrt((n * n) / area);
    clarkEvans = [
      n,
      observed,
      expected,
      observed / expected,
      standardError,
      (observed - expected) / standardError
    ];
  }

  const [columns, quadratRows] = quadratGrid;
  const quadratCounts = new Array<number>(columns * quadratRows).fill(0);
  const width = fround(maxX - minX);
  const height = fround(maxY - minY);
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
    quadratStatistics: [m, mean, variance, ratio, (m - 1) * ratio, m - 1]
  };
}
