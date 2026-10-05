// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUColumnQuantileInterpolation} from '../../../src/map-graphs/column-classification/column-quantiles-parameters';

/** Inputs of {@link computeColumnQuantilesOracle}. */
export type ColumnQuantilesOracleInput = {
  values: ArrayLike<number>;
  mask?: ArrayLike<number>;
  quantiles: ArrayLike<number>;
  interpolation?: GPUColumnQuantileInterpolation;
  filterRange?: [number, number];
  /** Output slots; probabilities beyond `quantiles.length` give NaN. Defaults to `quantiles.length`. */
  quantileCount?: number;
};

/** Result of {@link computeColumnQuantilesOracle}. */
export type ColumnQuantilesOracleResult = {
  quantiles: Float32Array;
  validCount: number;
  /** Valid values sorted ascending (`-0` before `+0`). */
  sorted: Float32Array;
  filterBounds: Float32Array;
  filterMask: Uint32Array;
};

/** Total order of non-NaN floats where `-0` sorts below `+0`. */
export function compareFloatTotal(left: number, right: number): number {
  if (left < right) {
    return -1;
  }
  if (left > right) {
    return 1;
  }
  const leftNegativeZero = Object.is(left, -0);
  const rightNegativeZero = Object.is(right, -0);
  return leftNegativeZero === rightNegativeZero ? 0 : leftNegativeZero ? -1 : 1;
}

/** Round half to even, as numpy `nearest`. */
export function roundHalfToEven(value: number): number {
  const whole = Math.floor(value);
  const fraction = value - whole;
  if (fraction < 0.5) {
    return whole;
  }
  if (fraction > 0.5) {
    return whole + 1;
  }
  return whole % 2 === 0 ? whole : whole + 1;
}

/**
 * Sort-based reference of `GPUColumnQuantiles`, written without any radix logic. Float32
 * intermediate arithmetic uses `Math.fround`, so the order statistics, ranks, and filter indices
 * equal the GPU's; the final `linear` and `midpoint` arithmetic may differ by 1 ULP because a GPU
 * can fuse the multiply and add.
 */
export function computeColumnQuantilesOracle(
  input: ColumnQuantilesOracleInput
): ColumnQuantilesOracleResult {
  const {values, mask} = input;
  const interpolation = input.interpolation ?? 'linear';
  const quantileCount = input.quantileCount ?? input.quantiles.length;
  const validValues: number[] = [];
  for (let row = 0; row < values.length; row++) {
    const value = Math.fround(values[row]);
    if (!Number.isNaN(value) && (!mask || mask[row] !== 0)) {
      validValues.push(value);
    }
  }
  validValues.sort(compareFloatTotal);
  const sorted = Float32Array.from(validValues);
  const count = sorted.length;
  const quantiles = new Float32Array(quantileCount).fill(NaN);
  for (let index = 0; index < Math.min(quantileCount, input.quantiles.length); index++) {
    const probability = Math.fround(input.quantiles[index]);
    if (count === 0 || !(probability >= 0 && probability <= 1)) {
      continue;
    }
    const height = Math.fround(Math.fround(count - 1) * probability);
    const lower = Math.floor(height);
    const upper = Math.ceil(height);
    const firstValue =
      sorted[
        interpolation === 'higher'
          ? upper
          : interpolation === 'nearest'
            ? roundHalfToEven(height)
            : lower
      ];
    let result = firstValue;
    if (interpolation === 'linear') {
      const second = sorted[Math.min(lower + 1, count - 1)];
      const fraction = height - lower;
      if (firstValue !== second && fraction !== 0) {
        result = Math.fround(firstValue + Math.fround(Math.fround(second - firstValue) * fraction));
      }
    } else if (interpolation === 'midpoint') {
      const second = sorted[upper];
      if (firstValue !== second) {
        result = Math.fround(Math.fround(firstValue + second) * 0.5);
      }
    }
    quantiles[index] = result;
  }

  const filterBounds = new Float32Array([NaN, NaN]);
  const filterMask = new Uint32Array(values.length);
  if (count > 0) {
    const [rawLower, rawUpper] = input.filterRange ?? [0, 1];
    const clampFraction = (raw: number, fallback: number) => {
      const fraction = Math.fround(raw);
      return Number.isNaN(fraction) ? fallback : Math.min(Math.max(fraction, 0), 1);
    };
    const clampIndex = (index: number) => Math.min(Math.max(index, 0), count - 1);
    const lowerIndex = clampIndex(Math.floor(Math.fround(count * clampFraction(rawLower, 0))));
    const upperIndex = clampIndex(Math.ceil(Math.fround(count * clampFraction(rawUpper, 1))) - 1);
    filterBounds[0] = sorted[lowerIndex];
    filterBounds[1] = sorted[upperIndex];
    for (let row = 0; row < values.length; row++) {
      const value = Math.fround(values[row]);
      if (Number.isNaN(value) || (mask && mask[row] === 0)) {
        continue;
      }
      filterMask[row] =
        compareFloatTotal(filterBounds[0], value) <= 0 &&
        compareFloatTotal(value, filterBounds[1]) <= 0
          ? 1
          : 0;
    }
  }
  return {quantiles, validCount: count, sorted, filterBounds, filterMask};
}

/** Returns the float32 bit pattern of a value with every NaN folded to one pattern. */
export function getFloat32BitsCanonical(value: number): number {
  const scratch = new Float32Array([value]);
  return Number.isNaN(value) ? 0x7fc00000 : new Uint32Array(scratch.buffer)[0];
}

/** Distance in float32 ULPs between two values; 0 for equal bits (NaN equals NaN), Infinity if only one is non-finite or NaN. */
export function getUlpDistance(left: number, right: number): number {
  if (getFloat32BitsCanonical(left) === getFloat32BitsCanonical(right)) {
    return 0;
  }
  if (!Number.isFinite(left) || !Number.isFinite(right)) {
    return Infinity;
  }
  const ordered = (value: number) => {
    const bits = getFloat32BitsCanonical(value);
    return bits & 0x80000000 ? -(bits & 0x7fffffff) : bits;
  };
  return Math.abs(ordered(left) - ordered(right));
}
