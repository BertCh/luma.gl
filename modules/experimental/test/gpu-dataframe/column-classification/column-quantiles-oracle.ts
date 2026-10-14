// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUColumnQuantileInterpolation} from '../../../src/gpu-dataframe/column-classification/column-quantiles-parameters';
import {
  getOrderStatisticMidpoint,
  getOrderStatisticSample,
  interpolateOrderStatistic,
  isOrderStatisticProbability
} from '../order-statistic-oracle';

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
  /** Valid (finite, unmasked) values sorted ascending, `-0` canonicalised to `+0`. */
  sorted: Float32Array;
  filterBounds: Float32Array;
  outputMask: Uint32Array;
};

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
 * Sort-based reference of `GPUColumnQuantiles`, written without any radix logic, on the shared
 * order-statistic policy (`../order-statistic-oracle.ts`): finite values only, `-0` equal to `+0`,
 * NaN for an invalid probability. Float32 arithmetic uses `Math.fround` and the kernels never fuse
 * a multiply-add, so every output is bit exact.
 */
export function computeColumnQuantilesOracle(
  input: ColumnQuantilesOracleInput
): ColumnQuantilesOracleResult {
  const {values, mask} = input;
  const interpolation = input.interpolation ?? 'linear';
  const quantileCount = input.quantileCount ?? input.quantiles.length;
  const sorted = Float32Array.from(getOrderStatisticSample(values, mask));
  const count = sorted.length;
  const quantiles = new Float32Array(quantileCount).fill(NaN);
  for (let index = 0; index < Math.min(quantileCount, input.quantiles.length); index++) {
    const probability = Math.fround(input.quantiles[index]);
    if (count === 0 || !isOrderStatisticProbability(probability)) {
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
      result = interpolateOrderStatistic(firstValue, second, Math.fround(height - lower));
    } else if (interpolation === 'midpoint') {
      result = getOrderStatisticMidpoint(firstValue, sorted[upper]);
    }
    quantiles[index] = result;
  }

  const filterBounds = new Float32Array([NaN, NaN]);
  const outputMask = new Uint32Array(values.length);
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
      if (!Number.isFinite(value) || (mask && mask[row] === 0)) {
        continue;
      }
      outputMask[row] = filterBounds[0] <= value && value <= filterBounds[1] ? 1 : 0;
    }
  }
  return {quantiles, validCount: count, sorted, filterBounds, outputMask};
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
