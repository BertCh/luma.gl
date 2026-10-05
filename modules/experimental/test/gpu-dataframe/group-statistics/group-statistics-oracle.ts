// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU reference for GPUGroupStatistics. Integer results (counts, fixed-point sums, extremes,
// median, percentiles, mode, unique counts) are exact; moments are computed in float64 from the
// true values and compared to the GPU with a stated float32 tolerance.

import {getScaledValue} from '../../gpu-spatial-analysis/cell-aggregation/cell-aggregation-oracle';

const f = Math.fround;

/** Deterministic mulberry32 generator in [0, 1). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Statistics of one value column within one group. */
export type OracleColumnStatistics = {
  /** Finite values. */
  count: number;
  /** Exact fixed-point sum (`roundHalfEven(fround(v * sumScale))` summed). */
  sum: bigint;
  sumValue: number;
  mean: number;
  minimum: number;
  maximum: number;
  variance: number;
  standardDeviation: number;
  skewness: number;
  kurtosis: number;
  median: number;
  percentiles: number[];
  mode: number;
  uniqueCount: number;
};

/** One output group. */
export type OracleGroup = {
  key: bigint;
  /** Valid rows (mask set, key valid). */
  count: number;
  columns: OracleColumnStatistics[];
};

/** Inputs of {@link computeGroupStatisticsOnCPU}. */
export type GroupStatisticsOracleInput = {
  keys: readonly bigint[];
  keyBits: 32 | 64;
  mask?: Uint32Array;
  columns: readonly Float32Array[];
  variance: 'sample' | 'population';
  fractions: readonly number[];
  sumScale: number;
  capacity: number;
};

/** Result of {@link computeGroupStatisticsOnCPU}. */
export type GroupStatisticsOracleResult = {
  /** Unclamped number of groups. */
  totalCount: number;
  /** Groups ascending by key, clamped to the capacity. */
  groups: OracleGroup[];
  /** Per column, per source row z-score (NaN when undefined). */
  zScores: Float64Array[];
};

/** Canonical key of a finite f32: `-0` becomes `+0`. */
export function canonicalizeValue(value: number): number {
  return value === 0 ? 0 : value;
}

/**
 * Linear-interpolation quantile of ascending canonical finite f32 values with the kernel's f32
 * arithmetic: `h = f32(n - 1) * p`, `lo + (hi - lo) * (h - floor h)`.
 */
export function interpolateQuantile(sorted: readonly number[], fraction: number): number {
  const count = sorted.length;
  if (count === 0 || !Number.isFinite(fraction)) {
    return NaN;
  }
  const p = Math.min(Math.max(f(fraction), 0), 1);
  const h = f(f(count - 1) * p);
  const floorH = Math.floor(h);
  const lowIndex = Math.min(floorH, count - 1);
  const low = sorted[lowIndex];
  if (lowIndex + 1 >= count) {
    return low;
  }
  const high = sorted[lowIndex + 1];
  return f(low + f(f(high - low) * f(h - floorH)));
}

function getVariance(variance: 'sample' | 'population', count: number, m2: number): number {
  if (variance === 'sample') {
    return count >= 2 ? m2 / (count - 1) : NaN;
  }
  return count >= 1 ? m2 / count : NaN;
}

/** Statistics of finite f32 values (any order) for one group and column. */
export function computeColumnStatistics(
  values: readonly number[],
  props: {
    variance: 'sample' | 'population';
    fractions: readonly number[];
    sumScale: number;
  }
): OracleColumnStatistics {
  const finite = values.filter(Number.isFinite).map(canonicalizeValue);
  const count = finite.length;
  let sum = 0n;
  for (const value of values.filter(Number.isFinite)) {
    sum += getScaledValue(value, props.sumScale);
  }
  const sorted = [...finite].sort((left, right) => left - right);
  const sumValue = Number(sum) / props.sumScale;
  const mean = count > 0 ? sumValue / count : NaN;
  // Float64 central moments about the true mean of the values.
  let trueMean = 0;
  for (const value of finite) {
    trueMean += value;
  }
  trueMean /= Math.max(count, 1);
  const isConstant = count === 0 || sorted[0] === sorted[count - 1];
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (const value of isConstant ? [] : finite) {
    const d = value - trueMean;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  const variance = count > 0 ? getVariance(props.variance, count, m2) : NaN;
  // Runs of equal values (ascending), longest first, smallest value on ties.
  let mode = NaN;
  let uniqueCount = 0;
  let bestLength = 0;
  for (let start = 0; start < count; ) {
    let end = start;
    while (end < count && sorted[end] === sorted[start]) {
      end++;
    }
    uniqueCount++;
    if (end - start > bestLength) {
      bestLength = end - start;
      mode = sorted[start];
    }
    start = end;
  }
  return {
    count,
    sum,
    sumValue,
    mean,
    minimum: count > 0 ? sorted[0] : NaN,
    maximum: count > 0 ? sorted[count - 1] : NaN,
    variance,
    standardDeviation: Math.sqrt(variance),
    skewness: isConstant ? NaN : m3 / count / Math.pow(m2 / count, 1.5),
    kurtosis: isConstant ? NaN : m4 / count / ((m2 / count) * (m2 / count)) - 3,
    median: interpolateQuantile(sorted, 0.5),
    percentiles: props.fractions.map(fraction => interpolateQuantile(sorted, fraction)),
    mode,
    uniqueCount
  };
}

/** Groups rows by key and computes every statistic of every column. */
export function computeGroupStatisticsOnCPU(
  input: GroupStatisticsOracleInput
): GroupStatisticsOracleResult {
  const reserved = (1n << BigInt(input.keyBits)) - 1n;
  const rows = input.keys.length;
  const rowsByKey = new Map<bigint, number[]>();
  for (let row = 0; row < rows; row++) {
    const key = input.keys[row];
    if (key === reserved || (input.mask && input.mask[row] === 0)) {
      continue;
    }
    const group = rowsByKey.get(key);
    if (group) {
      group.push(row);
    } else {
      rowsByKey.set(key, [row]);
    }
  }
  const keys = [...rowsByKey.keys()].sort((left, right) => (left < right ? -1 : 1));
  const groups: OracleGroup[] = [];
  const zScores = input.columns.map(() => new Float64Array(rows).fill(NaN));
  for (const [groupIndex, key] of keys.entries()) {
    if (groupIndex >= input.capacity) {
      break;
    }
    const groupRows = rowsByKey.get(key)!;
    const columns = input.columns.map(column =>
      computeColumnStatistics(
        groupRows.map(row => column[row]),
        input
      )
    );
    groups.push({key, count: groupRows.length, columns});
    for (const [columnIndex, column] of input.columns.entries()) {
      const statistics = columns[columnIndex];
      for (const row of groupRows) {
        if (!Number.isFinite(column[row])) {
          continue;
        }
        const {standardDeviation, count} = statistics;
        const isUndefined =
          (input.variance === 'sample' && count < 2) ||
          !(standardDeviation > 0) ||
          !Number.isFinite(standardDeviation);
        zScores[columnIndex][row] = isUndefined
          ? 0
          : (column[row] - statistics.mean) / standardDeviation;
      }
    }
  }
  return {totalCount: keys.length, groups, zScores};
}
