// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** How a quantile that falls between two order statistics is resolved (numpy `method` names). */
export type GPUColumnQuantileInterpolation = 'lower' | 'higher' | 'nearest' | 'linear' | 'midpoint';

/** Parameter-buffer code of each {@link GPUColumnQuantileInterpolation}. */
export const GPU_COLUMN_QUANTILE_INTERPOLATION_CODES: Record<
  GPUColumnQuantileInterpolation,
  number
> = {
  lower: 0,
  higher: 1,
  nearest: 2,
  linear: 3,
  midpoint: 4
};

/** Number of float32 header elements before the probabilities. */
export const GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH = 4;

/** Largest compile-time quantile count. */
export const GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT = 64;

/** Number of float32 elements of a parameter view for `quantileCount` probabilities. */
export function getGPUColumnQuantilesParameterLength(quantileCount: number): number {
  return GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH + quantileCount;
}

/** Inputs of {@link getGPUColumnQuantilesParameterValues}. */
export type GPUColumnQuantilesParameterInput = {
  /** Probabilities in `[0, 1]`; others (and NaN) produce a NaN quantile. */
  quantiles: ArrayLike<number>;
  /** Defaults to `'linear'`. */
  interpolation?: GPUColumnQuantileInterpolation;
  /** Percentile filter as fractions `[lower, upper]` in `[0, 1]`. Defaults to `[0, 1]`. */
  filterRange?: [number, number];
  /** Number of probability slots. Defaults to `quantiles.length`. Unused slots become NaN. */
  quantileCount?: number;
};

/**
 * Packs per-frame parameters of `GPUColumnQuantiles`.
 *
 * Layout (float32): `[interpolationCode, filterLower, filterUpper, 0, p_0, ..., p_{quantileCount-1}]`.
 * Probability slots beyond `quantiles.length` are NaN, so their outputs are NaN.
 *
 * @param props Probabilities, interpolation, filter range, and slot count.
 * @param target Optional destination of at least {@link getGPUColumnQuantilesParameterLength} elements.
 */
export function getGPUColumnQuantilesParameterValues(
  props: GPUColumnQuantilesParameterInput,
  target?: Float32Array
): Float32Array {
  const quantileCount = props.quantileCount ?? props.quantiles.length;
  if (
    !Number.isInteger(quantileCount) ||
    quantileCount < 1 ||
    quantileCount > GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT
  ) {
    throw new Error(
      `Column quantile count must be an integer in [1, ${GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT}]`
    );
  }
  if (props.quantiles.length > quantileCount) {
    throw new Error('Column quantile probabilities exceed quantileCount');
  }
  const length = getGPUColumnQuantilesParameterLength(quantileCount);
  const values = target ?? new Float32Array(length);
  if (values.length < length) {
    throw new Error(`Column quantile parameter target must hold ${length} elements`);
  }
  const interpolation = props.interpolation ?? 'linear';
  const code = GPU_COLUMN_QUANTILE_INTERPOLATION_CODES[interpolation];
  if (code === undefined) {
    throw new Error(`Unknown column quantile interpolation ${String(interpolation)}`);
  }
  const [filterLower, filterUpper] = props.filterRange ?? [0, 1];
  values[0] = code;
  values[1] = filterLower;
  values[2] = filterUpper;
  values[3] = 0;
  for (let index = 0; index < quantileCount; index++) {
    values[GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH + index] =
      index < props.quantiles.length ? props.quantiles[index] : NaN;
  }
  return values;
}
