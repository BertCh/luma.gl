// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  GPU_LOCAL_MORAN_QUADRANT,
  type GPUSpatialAutocorrelationParameters
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';

const {fround} = Math;

/** Inputs of the CPU oracle; mirrors the GPU contributors' views. */
export type SpatialAutocorrelationOracleInput = {
  positions: Float32Array;
  values: Float32Array;
  mask?: Uint32Array;
  parameters: GPUSpatialAutocorrelationParameters;
};

/** Global moments used by both statistics. */
export type SpatialAutocorrelationOracleMoments = {
  count: number;
  mean: number;
  variance: number;
  sumOfSquares: number;
};

/** Gi* oracle result. */
export type HotSpotOracleResult = {
  moments: SpatialAutocorrelationOracleMoments;
  included: boolean[];
  zScores: number[];
  pValues: number[];
  bins: number[];
  neighborCounts: number[];
};

/** Local Moran oracle result. */
export type LocalMoranOracleResult = {
  moments: SpatialAutocorrelationOracleMoments;
  included: boolean[];
  zScores: number[];
  pValues: number[];
  localI: number[];
  spatialLag: number[];
  quadrants: number[];
  neighborCounts: number[];
};

/** Deterministic `[0, 1)` generator (mulberry32). */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Two-sided normal p-value `erfc(|z| / sqrt(2))` with the same Numerical Recipes erfcc fit as the
 * WGSL, evaluated in double precision.
 */
export function getTwoSidedPValue(zScore: number): number {
  const x = Math.abs(zScore) / Math.SQRT2;
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

/** Whether row `row` takes part in the analysis (mask, finite value, finite in-bounds position). */
export function getIncludedRows(input: SpatialAutocorrelationOracleInput): boolean[] {
  const [minX, minY, maxX, maxY] = input.parameters.bounds.map(fround);
  const radius = fround(input.parameters.radius);
  const validLattice =
    radius > 0 && Number.isFinite(radius) && maxX - minX >= 0 && maxY - minY >= 0;
  return Array.from(input.values, (value, row) => {
    const x = input.positions[row * 2];
    const y = input.positions[row * 2 + 1];
    return (
      validLattice &&
      (input.mask ? input.mask[row] !== 0 : true) &&
      Number.isFinite(value) &&
      Number.isFinite(x) &&
      Number.isFinite(y) &&
      x >= minX &&
      x <= maxX &&
      y >= minY &&
      y <= maxY
    );
  });
}

/** The f32 distance-band test the WGSL evaluates. */
export function isWithinRadius(
  positions: Float32Array,
  row: number,
  other: number,
  radius: number
): boolean {
  const deltaX = fround(positions[other * 2] - positions[row * 2]);
  const deltaY = fround(positions[other * 2 + 1] - positions[row * 2 + 1]);
  const radiusSquared = fround(fround(radius) * fround(radius));
  return fround(fround(deltaX * deltaX) + fround(deltaY * deltaY)) <= radiusSquared;
}

/** Double-precision two-pass moments over included rows, or the fixed moments when given. */
export function getOracleMoments(
  input: SpatialAutocorrelationOracleInput,
  included: boolean[]
): SpatialAutocorrelationOracleMoments {
  const fixed = input.parameters.fixedMoments;
  if (fixed) {
    return {
      count: fixed.count,
      mean: fixed.mean,
      variance: fixed.variance,
      sumOfSquares: fixed.variance * fixed.count
    };
  }
  let count = 0;
  let sum = 0;
  for (const [row, isIncluded] of included.entries()) {
    if (isIncluded) {
      count++;
      sum += input.values[row];
    }
  }
  const mean = sum / count;
  let sumOfSquares = 0;
  for (const [row, isIncluded] of included.entries()) {
    if (isIncluded) {
      sumOfSquares += (input.values[row] - mean) ** 2;
    }
  }
  return {count, mean, variance: sumOfSquares / count, sumOfSquares};
}

/**
 * Benjamini-Hochberg step-up levels: for each row, the number of `alphas` (ordered from loosest to
 * strictest) at which the row's two-sided p-value is rejected. Rows with a non-finite z get 0.
 */
export function getFalseDiscoveryRateLevels(
  zScores: ArrayLike<number>,
  alphas: readonly number[]
): number[] {
  const tested = Array.from(zScores, (zScore, row) => ({row, zScore}))
    .filter(({zScore}) => Number.isFinite(zScore))
    .sort((left, right) => Math.abs(right.zScore) - Math.abs(left.zScore) || left.row - right.row);
  const testedCount = tested.length;
  const thresholds = alphas.map(alpha => {
    let threshold = 0;
    for (const [slot, {zScore}] of tested.entries()) {
      if (getTwoSidedPValue(zScore) <= ((slot + 1) / testedCount) * alpha) {
        threshold = slot + 1;
      }
    }
    return threshold;
  });
  const levels = new Array<number>(zScores.length).fill(0);
  for (const [slot, {row}] of tested.entries()) {
    levels[row] = thresholds.filter(threshold => slot + 1 <= threshold).length;
  }
  return levels;
}

/**
 * Smallest relative distance between any tested p-value and its BH bound `k * alpha / m`, used by
 * tests to make sure the f32 GPU and double CPU decisions cannot disagree by rounding.
 */
export function getFalseDiscoveryRateMargin(
  zScores: ArrayLike<number>,
  alphas: readonly number[]
): number {
  const pValues = Array.from(zScores)
    .filter(Number.isFinite)
    .map(getTwoSidedPValue)
    .sort((left, right) => left - right);
  let margin = Infinity;
  for (const alpha of alphas) {
    for (const [slot, pValue] of pValues.entries()) {
      const bound = ((slot + 1) / pValues.length) * alpha;
      margin = Math.min(margin, Math.abs(pValue - bound) / bound);
    }
  }
  return margin;
}

/** Brute-force Getis-Ord Gi* oracle in double precision with the GPU's f32 neighbor test. */
export function computeHotSpotOracle(
  input: SpatialAutocorrelationOracleInput,
  options: {falseDiscoveryRate?: boolean} = {}
): HotSpotOracleResult {
  const rows = input.values.length;
  const included = getIncludedRows(input);
  const moments = getOracleMoments(input, included);
  const zScores = new Array<number>(rows).fill(NaN);
  const neighborCounts = new Array<number>(rows).fill(0);
  const standardDeviation = Math.sqrt(moments.variance);
  for (let row = 0; row < rows; row++) {
    if (!included[row]) {
      continue;
    }
    let neighborSum = 0;
    let neighborCount = 0;
    for (let other = 0; other < rows; other++) {
      if (included[other] && isWithinRadius(input.positions, row, other, input.parameters.radius)) {
        neighborSum += input.values[other] - moments.mean;
        neighborCount++;
      }
    }
    neighborCounts[row] = neighborCount;
    const {count} = moments;
    const spread = (neighborCount * (count - neighborCount)) / (count - 1);
    if (count >= 2 && moments.variance > 0 && spread > 0) {
      zScores[row] = neighborSum / (standardDeviation * Math.sqrt(spread));
    }
  }
  const pValues = zScores.map(zScore =>
    Number.isFinite(zScore) ? getTwoSidedPValue(zScore) : NaN
  );
  const levels = options.falseDiscoveryRate
    ? getFalseDiscoveryRateLevels(zScores, GPU_HOT_SPOT_SIGNIFICANCE_LEVELS)
    : zScores.map(zScore => getUncorrectedHotSpotLevel(zScore));
  const bins = zScores.map((zScore, row) => (zScore < 0 ? -levels[row] : levels[row]) || 0);
  return {moments, included, zScores, pValues, bins, neighborCounts};
}

/** Uncorrected Gi* confidence level `0..3` of one z-score. */
export function getUncorrectedHotSpotLevel(zScore: number): number {
  if (!Number.isFinite(zScore)) {
    return 0;
  }
  return GPU_HOT_SPOT_CRITICAL_Z_SCORES.filter(critical => Math.abs(zScore) >= critical).length;
}

/**
 * Brute-force local Moran oracle in double precision with the GPU's f32 neighbor test and the
 * conditional-randomization analytic moments.
 */
export function computeLocalMoranOracle(
  input: SpatialAutocorrelationOracleInput,
  options: {falseDiscoveryRate?: boolean} = {}
): LocalMoranOracleResult {
  const rows = input.values.length;
  const included = getIncludedRows(input);
  const moments = getOracleMoments(input, included);
  const significanceLevel = input.parameters.significanceLevel ?? 0.05;
  const rowStandardized = (input.parameters.weightTransform ?? 'row') === 'row';
  const zScores = new Array<number>(rows).fill(NaN);
  const localI = new Array<number>(rows).fill(NaN);
  const spatialLag = new Array<number>(rows).fill(NaN);
  const neighborCounts = new Array<number>(rows).fill(0);
  for (let row = 0; row < rows; row++) {
    if (!included[row]) {
      continue;
    }
    const centered = input.values[row] - moments.mean;
    let neighborSum = 0;
    let neighborCount = 0;
    for (let other = 0; other < rows; other++) {
      if (
        other !== row &&
        included[other] &&
        isWithinRadius(input.positions, row, other, input.parameters.radius)
      ) {
        neighborSum += input.values[other] - moments.mean;
        neighborCount++;
      }
    }
    neighborCounts[row] = neighborCount;
    spatialLag[row] = neighborCount > 0 ? neighborSum / neighborCount : 0;
    const weightedLag = rowStandardized ? spatialLag[row] : neighborSum;
    localI[row] = ((moments.count - 1) * centered * weightedLag) / moments.sumOfSquares;
    const {mean, variance} = getConditionalLagMoments(
      centered,
      moments.count,
      moments.sumOfSquares,
      neighborCount
    );
    if (
      moments.count - 1 >= 2 &&
      centered !== 0 &&
      neighborCount > 0 &&
      neighborCount < moments.count - 1 &&
      variance > 0
    ) {
      const lagZ = (neighborSum - mean) / Math.sqrt(variance);
      zScores[row] = centered < 0 ? -lagZ : lagZ;
    }
  }
  const pValues = zScores.map(zScore =>
    Number.isFinite(zScore) ? getTwoSidedPValue(zScore) : NaN
  );
  const significant = options.falseDiscoveryRate
    ? getFalseDiscoveryRateLevels(zScores, [significanceLevel]).map(level => level > 0)
    : pValues.map(pValue => pValue <= significanceLevel);
  const quadrants = zScores.map((_, row) =>
    significant[row] ? getQuadrant(input.values[row] - moments.mean, spatialLag[row]) : 0
  );
  return {moments, included, zScores, pValues, localI, spatialLag, quadrants, neighborCounts};
}

/**
 * Exact mean and variance of `L = sum of k randomly placed centered values` when the other
 * `count - 1` values (centered values summing to `-centered`) are permuted over the other
 * locations (sampling without replacement).
 */
export function getConditionalLagMoments(
  centered: number,
  count: number,
  sumOfSquares: number,
  neighborCount: number
): {mean: number; variance: number} {
  const others = count - 1;
  const otherMean = -centered / others;
  const otherVariance = (sumOfSquares - centered * centered) / others - otherMean * otherMean;
  return {
    mean: neighborCount * otherMean,
    variance: (otherVariance * neighborCount * (others - neighborCount)) / (others - 1)
  };
}

/** esda quadrant code from the signs of the centered value and the spatial lag. */
export function getQuadrant(centered: number, lag: number): number {
  if (centered > 0 && lag > 0) return GPU_LOCAL_MORAN_QUADRANT.HIGH_HIGH;
  if (centered < 0 && lag > 0) return GPU_LOCAL_MORAN_QUADRANT.LOW_HIGH;
  if (centered < 0 && lag < 0) return GPU_LOCAL_MORAN_QUADRANT.LOW_LOW;
  if (centered > 0 && lag < 0) return GPU_LOCAL_MORAN_QUADRANT.HIGH_LOW;
  return GPU_LOCAL_MORAN_QUADRANT.NOT_SIGNIFICANT;
}

/**
 * Random points in `[0, extent]^2` with values from a few Gaussian bumps and dips plus noise, so
 * both hot and cold spots exist. Points are re-drawn until no pair distance lies within a relative
 * `1e-3` of any tested radius, so the f32 GPU and CPU neighbor sets cannot differ by rounding.
 */
export function createAutocorrelatedScene(
  seed: number,
  pointCount: number,
  radii: readonly number[],
  extent: number = 100
): {positions: Float32Array; values: Float32Array} {
  const random = createSeededRandom(seed);
  const positions = new Float32Array(pointCount * 2);
  const radiiSquared = radii.map(radius => radius * radius);
  const isAmbiguous = (row: number): boolean => {
    for (let other = 0; other < row; other++) {
      const distanceSquared =
        (positions[row * 2] - positions[other * 2]) ** 2 +
        (positions[row * 2 + 1] - positions[other * 2 + 1]) ** 2;
      if (
        radiiSquared.some(
          radiusSquared => Math.abs(distanceSquared - radiusSquared) < 1e-3 * radiusSquared
        )
      ) {
        return true;
      }
    }
    return false;
  };
  for (let row = 0; row < pointCount; row++) {
    do {
      positions[row * 2] = random() * extent;
      positions[row * 2 + 1] = random() * extent;
    } while (isAmbiguous(row));
  }
  const bumps = [
    {x: 0.25, y: 0.3, amplitude: 6},
    {x: 0.7, y: 0.65, amplitude: -5},
    {x: 0.8, y: 0.2, amplitude: 4}
  ].map(bump => ({...bump, x: bump.x * extent, y: bump.y * extent}));
  const sigma = extent * 0.1;
  const values = new Float32Array(pointCount);
  for (let row = 0; row < pointCount; row++) {
    let value = 100 + (random() - 0.5) * 4;
    for (const bump of bumps) {
      const distanceSquared =
        (positions[row * 2] - bump.x) ** 2 + (positions[row * 2 + 1] - bump.y) ** 2;
      value += bump.amplitude * Math.exp(-distanceSquared / (2 * sigma * sigma));
    }
    values[row] = value;
  }
  return {positions, values};
}
