// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUMultipleTestingMethod} from '../../../src/analysis-core/contracts';
import {getMultipleTestingAdjustedPValues} from '../../../src/gpu-spatial-analysis/multiple-testing';
import {
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  GPU_LOCAL_MORAN_QUADRANT,
  type GPUSpatialAutocorrelationSettings
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';

const {fround} = Math;

/** Host-side CSR spatial weights, the CPU twin of `GPUSpatialWeights`. */
export type OracleWeights = {
  /** `rows + 1` exclusive row offsets. */
  offsets: Uint32Array;
  /** Neighbor row IDs, ascending within each row. */
  neighbors: Uint32Array;
  /** Weight per slot. */
  weights: Float32Array;
};

/** Inputs of the CPU oracle; mirrors the GPU contributors' views. */
export type SpatialAutocorrelationOracleInput = {
  weights: OracleWeights;
  values: Float32Array;
  mask?: Uint32Array;
  parameters?: GPUSpatialAutocorrelationSettings;
  /** Gi* only: weight `w_ii` of the focal row. Defaults to 1. */
  selfWeight?: number;
  /** Island policy, as `GPUHotSpotAnalysis`. Defaults to `'exclude'`. */
  islands?: 'exclude' | 'keep';
  /** Rows with fewer active neighbors have no z-score. Defaults to 1. */
  minimumNeighbors?: number;
  /** Re-standardize each row over its remaining neighbors (`weights.information.transform === 'row'`). */
  rowStandardized?: boolean;
};

/** Options of the oracles. */
export type SpatialAutocorrelationOracleOptions = {
  /** Multiple-testing correction of the p-values. Defaults to `'none'`. */
  multipleTesting?: GPUMultipleTestingMethod;
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
  /** Rows with a finite value and nonzero mask. */
  included: boolean[];
  /** Included rows without an active neighbor among the included rows. */
  islands: boolean[];
  islandCount: number;
  zScores: number[];
  pValues: number[];
  adjustedPValues: number[];
  bins: number[];
  neighborCounts: number[];
};

/** Local Moran oracle result. */
export type LocalMoranOracleResult = {
  moments: SpatialAutocorrelationOracleMoments;
  included: boolean[];
  islands: boolean[];
  islandCount: number;
  zScores: number[];
  pValues: number[];
  adjustedPValues: number[];
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

/** Whether each row takes part in the analysis (mask nonzero and finite value). */
export function getIncludedRows(input: SpatialAutocorrelationOracleInput): boolean[] {
  return Array.from(
    input.values,
    (value, row) => (input.mask ? input.mask[row] !== 0 : true) && Number.isFinite(value)
  );
}

/**
 * Brute-force distance-band weights with the same f32 distance test as `GPUNeighborSearch`: row `i`
 * lists every row `j != i` with `|p_i - p_j| <= radius`, ascending. Optionally row-standardized
 * (`1 / k`, rounded to f32 like the GPU) or with a custom weight per pair.
 */
export function createDistanceBandWeights(
  positions: Float32Array,
  radius: number,
  options: {rowStandardize?: boolean} = {}
): OracleWeights {
  const rows = positions.length / 2;
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  const radiusSquared = fround(fround(radius) * fround(radius));
  for (let row = 0; row < rows; row++) {
    const first = neighbors.length;
    for (let other = 0; other < rows; other++) {
      const deltaX = fround(positions[other * 2] - positions[row * 2]);
      const deltaY = fround(positions[other * 2 + 1] - positions[row * 2 + 1]);
      if (
        other !== row &&
        fround(fround(deltaX * deltaX) + fround(deltaY * deltaY)) <= radiusSquared
      ) {
        neighbors.push(other);
        weights.push(1);
      }
    }
    if (options.rowStandardize) {
      const count = neighbors.length - first;
      for (let slot = first; slot < neighbors.length; slot++) {
        weights[slot] = fround(1 / count);
      }
    }
    offsets.push(neighbors.length);
  }
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(weights)
  };
}

/**
 * Islands: included rows without an active slot (a neighbor other than the row, with a positive
 * weight, that is an included row).
 */
export function getIslandRows(input: SpatialAutocorrelationOracleInput): boolean[] {
  const included = getIncludedRows(input);
  const {offsets, neighbors, weights} = input.weights;
  return included.map((isIncluded, row) => {
    if (!isIncluded) {
      return false;
    }
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      if (
        neighbor < included.length &&
        neighbor !== row &&
        weights[slot] > 0 &&
        included[neighbor]
      ) {
        return false;
      }
    }
    return true;
  });
}

/** Rows that enter the moments and can be neighbors: included rows, minus islands under 'exclude'. */
export function getMomentRows(input: SpatialAutocorrelationOracleInput): boolean[] {
  const included = getIncludedRows(input);
  if (input.islands === 'keep') {
    return included;
  }
  const islands = getIslandRows(input);
  return included.map((isIncluded, row) => isIncluded && !islands[row]);
}

/**
 * Active neighbors of `row` among `momentRows` with their weights, in slot order: positive
 * weight, not the row itself. Weights are divided by their sum when `rowStandardized`.
 */
export function getIncludedNeighbors(
  input: SpatialAutocorrelationOracleInput,
  momentRows: boolean[],
  row: number
): {neighbors: number[]; weights: number[]} {
  const {offsets, neighbors, weights} = input.weights;
  const result = {neighbors: [] as number[], weights: [] as number[]};
  for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
    const neighbor = neighbors[slot];
    if (
      neighbor < momentRows.length &&
      neighbor !== row &&
      weights[slot] > 0 &&
      momentRows[neighbor]
    ) {
      result.neighbors.push(neighbor);
      result.weights.push(weights[slot]);
    }
  }
  if (input.rowStandardized && result.weights.length > 0) {
    const total = result.weights.reduce((sum, weight) => sum + weight, 0);
    result.weights = result.weights.map(weight => weight / total);
  }
  return result;
}

/** Double-precision two-pass moments over included rows, or the fixed moments when given. */
export function getOracleMoments(
  input: SpatialAutocorrelationOracleInput,
  included: boolean[]
): SpatialAutocorrelationOracleMoments {
  const fixed = input.parameters?.fixedMoments;
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

/**
 * Adjusted two-sided p-values of GPU z-scores: the CPU reference (`getMultipleTestingAdjustedPValues`)
 * over `erfc` p-values of the given finite z-scores. Non-finite z-scores are not tested (NaN).
 */
export function getAdjustedPValuesFromZScores(
  zScores: ArrayLike<number>,
  method: GPUMultipleTestingMethod
): number[] {
  const pValues = Array.from(zScores, zScore =>
    Number.isFinite(zScore) ? getTwoSidedPValue(zScore) : NaN
  );
  return Array.from(getMultipleTestingAdjustedPValues(pValues, method));
}

/**
 * Smallest relative distance between any finite adjusted p-value and one of `levels`, used by
 * tests to make sure the f32 GPU and double CPU decisions cannot disagree by rounding.
 */
export function getAdjustedPValueMargin(
  adjustedPValues: ArrayLike<number>,
  levels: readonly number[]
): number {
  let margin = Infinity;
  for (const adjusted of Array.from(adjustedPValues)) {
    if (Number.isFinite(adjusted)) {
      for (const level of levels) {
        margin = Math.min(margin, Math.abs(adjusted - level) / level);
      }
    }
  }
  return margin;
}

function getMethod(options: SpatialAutocorrelationOracleOptions): GPUMultipleTestingMethod {
  return options.multipleTesting ?? 'none';
}

function getAdjustedPValues(
  pValues: number[],
  options: SpatialAutocorrelationOracleOptions
): number[] {
  return Array.from(getMultipleTestingAdjustedPValues(pValues, getMethod(options)));
}

/**
 * Brute-force Getis-Ord Gi* oracle in double precision. With `selfWeight > 0` the Ord-Getis Gi*
 * z-score for arbitrary weights `z = sum w (x - X) / (S sqrt((n S1 - W^2) / (n - 1)))` with the
 * focal weight `selfWeight`; with `selfWeight = 0` the exact Ord-Getis Gi, standardized with the
 * mean and variance of the other `n - 1` rows. Rows with fewer than `minimumNeighbors` active
 * neighbors have no z-score.
 */
export function computeHotSpotOracle(
  input: SpatialAutocorrelationOracleInput,
  options: SpatialAutocorrelationOracleOptions = {}
): HotSpotOracleResult {
  const rows = input.values.length;
  const included = getIncludedRows(input);
  const islands = getIslandRows(input);
  const momentRows = getMomentRows(input);
  const moments = getOracleMoments(input, momentRows);
  const selfWeight = input.selfWeight ?? 1;
  const minimumNeighbors = input.minimumNeighbors ?? 1;
  const zScores = new Array<number>(rows).fill(NaN);
  const neighborCounts = new Array<number>(rows).fill(0);
  const standardDeviation = Math.sqrt(moments.variance);
  for (let row = 0; row < rows; row++) {
    if (!momentRows[row]) {
      continue;
    }
    const {neighbors, weights} = getIncludedNeighbors(input, momentRows, row);
    neighborCounts[row] = neighbors.length;
    if (neighbors.length < minimumNeighbors || neighbors.length === 0) {
      continue;
    }
    const centered = input.values[row] - moments.mean;
    let weightedSum = 0;
    let weightSum = 0;
    let squareSum = 0;
    for (const [slot, neighbor] of neighbors.entries()) {
      weightedSum += weights[slot] * (input.values[neighbor] - moments.mean);
      weightSum += weights[slot];
      squareSum += weights[slot] * weights[slot];
    }
    const {count} = moments;
    if (selfWeight === 0) {
      const others = count - 1;
      const otherMean = -centered / others;
      const otherVariance = (moments.sumOfSquares - centered * centered) / others - otherMean ** 2;
      const spread = (others * squareSum - weightSum * weightSum) / (others - 1);
      if (others >= 2 && otherVariance > 0 && spread > 0) {
        zScores[row] = (weightedSum - weightSum * otherMean) / Math.sqrt(otherVariance * spread);
      }
    } else {
      weightedSum += selfWeight * centered;
      weightSum += selfWeight;
      squareSum += selfWeight * selfWeight;
      const spread = (count * squareSum - weightSum * weightSum) / (count - 1);
      if (count >= 2 && moments.variance > 0 && spread > 0) {
        zScores[row] = weightedSum / (standardDeviation * Math.sqrt(spread));
      }
    }
  }
  const pValues = zScores.map(zScore =>
    Number.isFinite(zScore) ? getTwoSidedPValue(zScore) : NaN
  );
  const adjustedPValues = getAdjustedPValues(pValues, options);
  const hasCorrection = getMethod(options) !== 'none';
  const levels = hasCorrection
    ? adjustedPValues.map(adjusted =>
        Number.isFinite(adjusted)
          ? GPU_HOT_SPOT_SIGNIFICANCE_LEVELS.filter(level => adjusted <= level).length
          : 0
      )
    : zScores.map(zScore => getUncorrectedHotSpotLevel(zScore));
  const bins = zScores.map((zScore, row) => (zScore < 0 ? -levels[row] : levels[row]) || 0);
  return {
    moments,
    included,
    islands,
    islandCount: islands.filter(Boolean).length,
    zScores,
    pValues,
    adjustedPValues,
    bins,
    neighborCounts
  };
}

/** Uncorrected Gi* confidence level `0..3` of one z-score. */
export function getUncorrectedHotSpotLevel(zScore: number): number {
  if (!Number.isFinite(zScore)) {
    return 0;
  }
  return GPU_HOT_SPOT_CRITICAL_Z_SCORES.filter(critical => Math.abs(zScore) >= critical).length;
}

/**
 * Brute-force local Moran oracle in double precision with the conditional-randomization analytic
 * moments, for weights used as given (or re-standardized with `rowStandardized`).
 */
export function computeLocalMoranOracle(
  input: SpatialAutocorrelationOracleInput,
  options: SpatialAutocorrelationOracleOptions = {}
): LocalMoranOracleResult {
  const rows = input.values.length;
  const included = getIncludedRows(input);
  const islands = getIslandRows(input);
  const momentRows = getMomentRows(input);
  const moments = getOracleMoments(input, momentRows);
  const significanceLevel = input.parameters?.significanceLevel ?? 0.05;
  const minimumNeighbors = input.minimumNeighbors ?? 1;
  const zScores = new Array<number>(rows).fill(NaN);
  const localI = new Array<number>(rows).fill(NaN);
  const spatialLag = new Array<number>(rows).fill(NaN);
  const neighborCounts = new Array<number>(rows).fill(0);
  for (let row = 0; row < rows; row++) {
    if (!momentRows[row]) {
      continue;
    }
    const centered = input.values[row] - moments.mean;
    const {neighbors, weights} = getIncludedNeighbors(input, momentRows, row);
    let lag = 0;
    let weightSum = 0;
    let squareSum = 0;
    for (const [slot, neighbor] of neighbors.entries()) {
      lag += weights[slot] * (input.values[neighbor] - moments.mean);
      weightSum += weights[slot];
      squareSum += weights[slot] * weights[slot];
    }
    neighborCounts[row] = neighbors.length;
    spatialLag[row] = lag;
    localI[row] = ((moments.count - 1) * centered * lag) / moments.sumOfSquares;
    const {mean, variance} = getConditionalLagMoments(
      centered,
      moments.count,
      moments.sumOfSquares,
      weightSum,
      squareSum
    );
    if (
      moments.count - 1 >= 2 &&
      centered !== 0 &&
      neighbors.length >= Math.max(1, minimumNeighbors) &&
      variance > 0
    ) {
      const lagZ = (lag - mean) / Math.sqrt(variance);
      zScores[row] = centered < 0 ? -lagZ : lagZ;
    }
  }
  const pValues = zScores.map(zScore =>
    Number.isFinite(zScore) ? getTwoSidedPValue(zScore) : NaN
  );
  const adjustedPValues = getAdjustedPValues(pValues, options);
  const significant = adjustedPValues.map(adjusted => adjusted <= significanceLevel);
  const quadrants = zScores.map((_, row) =>
    significant[row] ? getQuadrant(input.values[row] - moments.mean, spatialLag[row]) : 0
  );
  return {
    moments,
    included,
    islands,
    islandCount: islands.filter(Boolean).length,
    zScores,
    pValues,
    adjustedPValues,
    localI,
    spatialLag,
    quadrants,
    neighborCounts
  };
}

/**
 * Exact mean and variance of `L = sum_j w_j c_pi(j)` when the other `count - 1` centered values
 * (summing to `-centered`) are permuted over the other locations (sampling without replacement):
 * `E[L] = W mu`, `Var[L] = sigma^2 (N S1 - W^2) / (N - 1)` with `W = sum w`, `S1 = sum w^2`.
 */
export function getConditionalLagMoments(
  centered: number,
  count: number,
  sumOfSquares: number,
  weightSum: number,
  squareSum: number
): {mean: number; variance: number} {
  const others = count - 1;
  const otherMean = -centered / others;
  const otherVariance = (sumOfSquares - centered * centered) / others - otherMean * otherMean;
  return {
    mean: weightSum * otherMean,
    variance: (otherVariance * (others * squareSum - weightSum * weightSum)) / (others - 1)
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
