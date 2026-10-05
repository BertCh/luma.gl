// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Floats per dimension in the {@link GPUAttributeCrossfilter} parameter view. */
export const GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE = 8;
/** Offset of the inclusive brush minimum within one dimension's parameter record. */
export const GPU_ATTRIBUTE_CROSSFILTER_BRUSH_MIN_OFFSET = 0;
/** Offset of the exclusive brush maximum within one dimension's parameter record. */
export const GPU_ATTRIBUTE_CROSSFILTER_BRUSH_MAX_OFFSET = 1;
/** Offset of the brush-enabled flag (`0` or `1`) within one dimension's parameter record. */
export const GPU_ATTRIBUTE_CROSSFILTER_BRUSH_ENABLED_OFFSET = 2;
/** Offset of the histogram domain minimum, read when the dimension's domain is `'parameters'`. */
export const GPU_ATTRIBUTE_CROSSFILTER_DOMAIN_MIN_OFFSET = 3;
/** Offset of the histogram domain maximum, read when the dimension's domain is `'parameters'`. */
export const GPU_ATTRIBUTE_CROSSFILTER_DOMAIN_MAX_OFFSET = 4;

/** Per-frame state of one crossfilter dimension, packed by {@link getGPUAttributeCrossfilterParameterValues}. */
export type GPUAttributeCrossfilterDimensionState = {
  /**
   * Brush `[min, max]`: rows pass when `min <= value < max` (inclusive minimum, exclusive
   * maximum). Omit or pass `null` to disable the brush. Use `Infinity` as the maximum to include
   * everything above the minimum.
   */
  brush?: readonly [number, number] | null;
  /** Histogram domain `[min, max]`, used only by dimensions declared with `domain: 'parameters'`. */
  domain?: readonly [number, number];
};

/** Returns the minimum parameter view length for `dimensionCount` dimensions. */
export function getGPUAttributeCrossfilterParameterLength(dimensionCount: number): number {
  return dimensionCount * GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE;
}

/**
 * Packs per-dimension brushes and parameter domains into the float32 layout the recipe reads.
 *
 * Per dimension, {@link GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE} floats:
 * `[brushMin, brushMax, brushEnabled, domainMin, domainMax, 0, 0, 0]`.
 * Write the result into the caller-owned parameter buffer between encodings; nothing recompiles.
 *
 * @param states One entry per dimension, in recipe dimension order.
 * @param target Optional array to fill, at least `states.length * stride` long.
 */
export function getGPUAttributeCrossfilterParameterValues(
  states: readonly GPUAttributeCrossfilterDimensionState[],
  target: Float32Array = new Float32Array(getGPUAttributeCrossfilterParameterLength(states.length))
): Float32Array {
  if (target.length < getGPUAttributeCrossfilterParameterLength(states.length)) {
    throw new Error('GPUAttributeCrossfilter parameter target is too short');
  }
  for (const [dimensionIndex, state] of states.entries()) {
    const base = dimensionIndex * GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE;
    target.fill(0, base, base + GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE);
    if (state.brush) {
      target[base + GPU_ATTRIBUTE_CROSSFILTER_BRUSH_MIN_OFFSET] = state.brush[0];
      target[base + GPU_ATTRIBUTE_CROSSFILTER_BRUSH_MAX_OFFSET] = state.brush[1];
      target[base + GPU_ATTRIBUTE_CROSSFILTER_BRUSH_ENABLED_OFFSET] = 1;
    }
    if (state.domain) {
      target[base + GPU_ATTRIBUTE_CROSSFILTER_DOMAIN_MIN_OFFSET] = state.domain[0];
      target[base + GPU_ATTRIBUTE_CROSSFILTER_DOMAIN_MAX_OFFSET] = state.domain[1];
    }
  }
  return target;
}

/** Row offsets of each dimension's bins within the dimension-major `histograms` output. */
export type GPUAttributeCrossfilterHistogramLayout = {
  /** First histogram row of each dimension. */
  offsets: number[];
  /** Total histogram rows, the minimum `histograms` view length. */
  totalBinCount: number;
};

/** Returns dimension-major histogram offsets for the given per-dimension bin counts. */
export function getGPUAttributeCrossfilterHistogramLayout(
  binCounts: readonly number[]
): GPUAttributeCrossfilterHistogramLayout {
  const offsets: number[] = [];
  let totalBinCount = 0;
  for (const binCount of binCounts) {
    offsets.push(totalBinCount);
    totalBinCount += binCount;
  }
  return {offsets, totalBinCount};
}
