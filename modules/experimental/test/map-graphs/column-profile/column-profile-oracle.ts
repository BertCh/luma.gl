// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getOrderedFloat32Key} from '../../../src/map-graphs/column-classification/column-classification-shared';

const fround = Math.fround;
const NULL_CODE = 0xffffffff;

/** One oracle column. */
export type ColumnProfileOracleColumn =
  | {kind: 'numeric'; values: Float32Array}
  | {kind: 'category'; values: Uint32Array; categoryCount: number};

/** Inputs of {@link computeColumnProfileOracle}. */
export type ColumnProfileOracleInput = {
  columns: readonly ColumnProfileOracleColumn[];
  mask?: Uint32Array;
  /** `[lo, hi]` per column, NaN bounds are automatic. */
  domains?: readonly (readonly [number, number] | undefined)[];
  histogramBinCount: number;
  precision: number;
  topCategoryCount: number;
};

/** Oracle statistics of one column. */
export type ColumnProfileOracleResult = {
  count: number;
  nullCount: number;
  minimum: number;
  maximum: number;
  sum: number;
  mean: number;
  variance: number;
  sampleVariance: number;
  standardDeviation: number;
  distinctEstimate: number;
  overflowCount: number;
  trueDistinct: number;
  histogram: Uint32Array;
  registers: Uint32Array;
  topCategories: number[];
  topCategoryCounts: number[];
};

/** murmur3 fmix32. */
export function fmix32(value: number): number {
  let h = value >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

const scratchFloat = new Float32Array(1);
const scratchBits = new Uint32Array(scratchFloat.buffer);

/** f32 bits of a value with `-0` canonicalised to `+0`. */
export function getCanonicalFloat32Bits(value: number): number {
  scratchFloat[0] = value;
  return (scratchBits[0] & 0x7fffffff) === 0 ? 0 : scratchBits[0];
}

/** Standard HyperLogLog estimate with linear counting for the small range, in f64. */
export function estimateHyperLogLog(registers: ArrayLike<number>, precision: number): number {
  const m = 2 ** precision;
  const alpha = m === 16 ? 0.673 : m === 32 ? 0.697 : m === 64 ? 0.709 : 0.7213 / (1 + 1.079 / m);
  let inverseSum = 0;
  let zeros = 0;
  for (let index = 0; index < m; index++) {
    inverseSum += 2 ** -registers[index];
    if (registers[index] === 0) {
      zeros++;
    }
  }
  if (zeros === m) {
    return 0;
  }
  const raw = (alpha * m * m) / inverseSum;
  return raw <= 2.5 * m && zeros > 0 ? m * Math.log(m / zeros) : raw;
}

/** Bin of a finite value in a regular domain, mirroring the GPU with `Math.fround`. */
export function getOracleBin(value: number, lo: number, width: number, binCount: number): number {
  const difference = fround(value - lo);
  let bin = Math.floor(fround(difference / width));
  bin = Math.min(Math.max(bin, 0), binCount - 1);
  if (difference < fround(bin * width)) {
    bin -= 1;
  } else if (difference >= fround((bin + 1) * width)) {
    bin += 1;
  }
  return Math.min(Math.max(bin, 0), binCount - 1);
}

/** Resolves a histogram domain `[lo, hi]` to `{mode, width}`, mirroring the GPU. */
export function resolveOracleDomain(
  lo: number,
  hi: number,
  binCount: number
): {lo: number; hi: number; width: number; mode: 0 | 1 | 2} {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(lo <= hi)) {
    return {lo, hi, width: 0, mode: 0};
  }
  if (lo === hi) {
    return {lo, hi, width: 0, mode: 2};
  }
  const range = fround(hi - lo);
  const width = fround(range * fround(1 / binCount));
  return Number.isFinite(range) && Number.isFinite(width) && width > 0
    ? {lo, hi, width, mode: 1}
    : {lo, hi, width, mode: 0};
}

/** f64 two-pass CPU profile of every column, written independently of the GPU algorithm. */
export function computeColumnProfileOracle(
  input: ColumnProfileOracleInput
): ColumnProfileOracleResult[] {
  const {histogramBinCount, precision, topCategoryCount} = input;
  const registerCount = 2 ** precision;
  return input.columns.map((column, columnIndex) => {
    const rows = column.values.length;
    const included = (row: number) => !input.mask || input.mask[row] !== 0;
    let count = 0;
    let nullCount = 0;
    let overflowCount = 0;
    let minimumKey = Infinity;
    let maximumKey = -Infinity;
    let finiteMinimum = NaN;
    let finiteMaximum = NaN;
    let finiteMinimumKey = Infinity;
    let finiteMaximumKey = -Infinity;
    const finite: number[] = [];
    const registers = new Uint32Array(registerCount);
    const distinct = new Set<number>();
    const seed = Math.imul(columnIndex + 1, 0x9e3779b9) >>> 0;
    const categoryCounts = new Map<number, number>();
    for (let row = 0; row < rows; row++) {
      if (!included(row)) {
        continue;
      }
      const value = column.values[row];
      let bits: number;
      if (column.kind === 'numeric') {
        if (Number.isNaN(value)) {
          nullCount++;
          continue;
        }
        bits = getCanonicalFloat32Bits(value);
        const key = getOrderedFloat32Key(value);
        minimumKey = Math.min(minimumKey, key);
        maximumKey = Math.max(maximumKey, key);
        if (Number.isFinite(value)) {
          finite.push(value);
          if (key < finiteMinimumKey) {
            finiteMinimumKey = key;
            finiteMinimum = value;
          }
          if (key > finiteMaximumKey) {
            finiteMaximumKey = key;
            finiteMaximum = value;
          }
        }
      } else {
        if (value === NULL_CODE) {
          nullCount++;
          continue;
        }
        bits = value;
        minimumKey = Math.min(minimumKey, value);
        maximumKey = Math.max(maximumKey, value);
        if (value >= column.categoryCount) {
          overflowCount++;
        } else {
          categoryCounts.set(value, (categoryCounts.get(value) ?? 0) + 1);
        }
      }
      count++;
      distinct.add(bits);
      const hash = fmix32((bits ^ seed) >>> 0);
      const registerIndex = hash >>> (32 - precision);
      const rest = (hash << precision) >>> 0;
      const leadingZeros = rest === 0 ? 32 : Math.clz32(rest);
      registers[registerIndex] = Math.max(
        registers[registerIndex],
        Math.min(leadingZeros, 32 - precision) + 1
      );
    }
    // Moments: f64 two pass over the finite values.
    let sum = 0;
    for (const value of finite) {
      sum += value;
    }
    const finiteCount = finite.length;
    const mean = finiteCount > 0 ? sum / finiteCount : NaN;
    let squares = 0;
    for (const value of finite) {
      squares += (value - mean) ** 2;
    }
    const variance = finiteCount > 0 ? squares / finiteCount : NaN;
    const sampleVariance = finiteCount > 1 ? squares / (finiteCount - 1) : NaN;
    let minimum = NaN;
    let maximum = NaN;
    if (count > 0) {
      if (column.kind === 'numeric') {
        // Decode through the key order: -0 sorts below +0.
        minimum = decodeKey(minimumKey);
        maximum = decodeKey(maximumKey);
      } else {
        minimum = minimumKey;
        maximum = maximumKey;
      }
    }
    const histogram = new Uint32Array(histogramBinCount);
    if (column.kind === 'numeric') {
      const requested = input.domains?.[columnIndex] ?? [NaN, NaN];
      const lo = Number.isNaN(requested[0]) ? finiteMinimum : requested[0];
      const hi = Number.isNaN(requested[1]) ? finiteMaximum : requested[1];
      const domain = resolveOracleDomain(lo, hi, histogramBinCount);
      if (domain.mode !== 0) {
        for (const value of finite) {
          if (value < domain.lo || value > domain.hi) {
            continue;
          }
          const bin =
            domain.mode === 2 ? 0 : getOracleBin(value, domain.lo, domain.width, histogramBinCount);
          histogram[bin]++;
        }
      }
    } else {
      for (let code = 0; code < Math.min(histogramBinCount, column.categoryCount); code++) {
        histogram[code] = categoryCounts.get(code) ?? 0;
      }
    }
    const ranked = [...categoryCounts.entries()].sort(
      (left, right) => right[1] - left[1] || left[0] - right[0]
    );
    const topCategories: number[] = [];
    const topCategoryCounts: number[] = [];
    for (let slot = 0; slot < topCategoryCount; slot++) {
      topCategories.push(slot < ranked.length ? ranked[slot][0] : NULL_CODE);
      topCategoryCounts.push(slot < ranked.length ? ranked[slot][1] : 0);
    }
    return {
      count,
      nullCount,
      minimum,
      maximum,
      sum: finiteCount > 0 ? sum : NaN,
      mean,
      variance,
      sampleVariance,
      standardDeviation: Math.sqrt(variance),
      distinctEstimate: estimateHyperLogLog(registers, precision),
      overflowCount,
      trueDistinct: distinct.size,
      histogram,
      registers,
      topCategories,
      topCategoryCounts
    };
  });
}

function decodeKey(key: number): number {
  scratchBits[0] = ((key & 0x80000000) !== 0 ? key ^ 0x80000000 : ~key) >>> 0;
  return scratchFloat[0];
}
