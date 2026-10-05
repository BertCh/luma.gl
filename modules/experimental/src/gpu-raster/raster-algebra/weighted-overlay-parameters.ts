// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateParameterTarget} from './raster-algebra-utils';

/** Largest layer count accepted by {@link GPUWeightedOverlay}. */
export const GPU_WEIGHTED_OVERLAY_MAXIMUM_LAYER_COUNT = 16;

/** float32 elements before the per-layer records in a weighted-overlay parameter view. */
const HEADER_LENGTH = 8;
/** float32 elements per layer record. */
const LAYER_LENGTH = 8;

/** Remap of one {@link GPUWeightedOverlay} input layer. */
export type GPUWeightedOverlayLayerSettings = {
  /** Layer weight. Negative weights are allowed and subtract. */
  weight: number;
  /**
   * `'linear'` (default) maps `[inputMin, inputMax]` to `[0, 1]`, clamped. `'table'` classifies
   * the value through this layer's rows of `remapBreaks` and returns the matching `remapValues`
   * row; a NaN remap value marks the class as restricted.
   */
  mode?: 'linear' | 'table';
  /** Linear mode: input value mapped to 0. Defaults to 0. */
  inputMin?: number;
  /** Linear mode: input value mapped to 1. Defaults to 1. */
  inputMax?: number;
  /** Linear mode: use `1 - t` so lower inputs score higher. */
  invert?: boolean;
  /** Table mode: active breaks of this layer, at most the contributor's `maximumBreakCount`. */
  breakCount?: number;
  /** Table mode interval closure, as in `GPURasterReclassify`. Defaults to `'left'`. */
  closed?: 'left' | 'right';
};

/** Per-frame settings of {@link GPUWeightedOverlay}. */
export type GPUWeightedOverlaySettings = {
  /** One record per input layer, in stack order. */
  layers: readonly GPUWeightedOverlayLayerSettings[];
  /** Divide the weighted sum by the sum of absolute weights of the contributing layers. */
  normalizeWeights?: boolean;
  /**
   * `'propagate'` (default): a nodata input makes the cell nodata. `'ignore'`: nodata layers are
   * skipped (and excluded from weight normalization); a cell with no valid layer is nodata.
   */
  noDataPolicy?: 'propagate' | 'ignore';
};

/** Returns the float32 element count of a weighted-overlay parameter view for `layerCount`. */
export function getGPUWeightedOverlayParameterLength(layerCount: number): number {
  return HEADER_LENGTH + LAYER_LENGTH * layerCount;
}

/**
 * Packs per-frame {@link GPUWeightedOverlay} parameters.
 *
 * Layout (float32): header `[normalizeWeights, ignoreNoData, 0, 0, 0, 0, 0, 0]`, then per layer
 * `[weight, mode, inputMin, scale, invert, breakCount, closedRight, degenerate]` where
 * `mode` is 0 (linear) or 1 (table), `scale = 1 / (inputMax - inputMin)` is computed here so the
 * shader multiplies instead of dividing, and `degenerate` is 1 when `inputMax == inputMin` (the
 * layer then scores 1 for values `>= inputMin`, else 0).
 *
 * @param settings Layer records and global options.
 * @param target Optional destination of at least `getGPUWeightedOverlayParameterLength(layers)`.
 * @throws If the target is too short, a weight or range is not finite, or a count is invalid.
 */
export function getGPUWeightedOverlayParameterValues(
  settings: GPUWeightedOverlaySettings,
  target: Float32Array = new Float32Array(
    getGPUWeightedOverlayParameterLength(settings.layers.length)
  )
): Float32Array {
  const length = getGPUWeightedOverlayParameterLength(settings.layers.length);
  validateParameterTarget('Weighted overlay', target, length);
  target.fill(0, 0, length);
  target[0] = settings.normalizeWeights ? 1 : 0;
  target[1] = settings.noDataPolicy === 'ignore' ? 1 : 0;
  for (const [layerIndex, layer] of settings.layers.entries()) {
    const inputMin = layer.inputMin ?? 0;
    const inputMax = layer.inputMax ?? 1;
    const breakCount = layer.breakCount ?? 0;
    for (const [name, value] of [
      ['weight', layer.weight],
      ['inputMin', inputMin],
      ['inputMax', inputMax]
    ] as const) {
      if (!Number.isFinite(value)) {
        throw new Error(`Weighted overlay layer ${layerIndex} ${name} must be finite`);
      }
    }
    if (!Number.isSafeInteger(breakCount) || breakCount < 0) {
      throw new Error(
        `Weighted overlay layer ${layerIndex} breakCount must be a non-negative integer`
      );
    }
    const base = HEADER_LENGTH + layerIndex * LAYER_LENGTH;
    const degenerate = Math.fround(inputMax) === Math.fround(inputMin);
    target[base] = layer.weight;
    target[base + 1] = layer.mode === 'table' ? 1 : 0;
    target[base + 2] = inputMin;
    target[base + 3] = degenerate ? 0 : 1 / (Math.fround(inputMax) - Math.fround(inputMin));
    target[base + 4] = layer.invert ? 1 : 0;
    target[base + 5] = breakCount;
    target[base + 6] = layer.closed === 'right' ? 1 : 0;
    target[base + 7] = degenerate ? 1 : 0;
  }
  return target;
}
