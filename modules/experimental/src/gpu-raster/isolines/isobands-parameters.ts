// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateParameterTarget} from '../raster-algebra/raster-algebra-utils';

/** Number of float32 elements in a {@link GPUIsobands} parameter view. */
export const GPU_ISOBANDS_PARAMETER_LENGTH = 12;

/** Band class written by {@link GPUIsobands} for nodata samples. */
export const GPU_ISOBANDS_NO_DATA_CLASS = 0xffffffff;

/** Per-frame settings of {@link GPUIsobands}. */
export type GPUIsobandsSettings = {
  /**
   * Number of active ascending breaks, at most the recipe's `maximumBreakCount` (larger values
   * clamp on the GPU). `breakCount + 1` bands exist: band `k` covers `[b[k - 1], b[k])`.
   */
  breakCount: number;
  /** Raster extent `[minX, minY, maxX, maxY]` in world units. Row 0 is at `minY`. */
  extent: readonly [number, number, number, number];
  /** Raster width in samples (must equal the recipe's `width`). */
  width: number;
  /** Raster height in samples (must equal the recipe's `height`). */
  height: number;
  /** First band emitted as geometry. Defaults to 0. Does not affect `bandClasses`. */
  firstBand?: number;
  /** Last band (inclusive) emitted as geometry. Defaults to `breakCount` (every band). */
  lastBand?: number;
};

/**
 * Packs per-frame {@link GPUIsobands} parameters.
 *
 * Layout (float32): `[breakCount, firstBand, lastBand, 0, minX, minY, maxX, maxY, cellWidth,
 * cellHeight, 0, 0]`. The cell sizes are `(maxX - minX) / width` and `(maxY - minY) / height`
 * rounded to float32, so a CPU oracle can reproduce the GPU arithmetic.
 *
 * @param settings Break count, extent, raster size, and optional band window.
 * @param target Optional destination of at least 12 elements.
 * @throws If the target is too short, a count is not a non-negative integer, or the extent is
 * not finite.
 */
export function getGPUIsobandsParameterValues(
  settings: GPUIsobandsSettings,
  target: Float32Array = new Float32Array(GPU_ISOBANDS_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget('Isobands', target, GPU_ISOBANDS_PARAMETER_LENGTH);
  const firstBand = settings.firstBand ?? 0;
  const lastBand = settings.lastBand ?? settings.breakCount;
  for (const [name, count] of [
    ['breakCount', settings.breakCount],
    ['firstBand', firstBand],
    ['lastBand', lastBand],
    ['width', settings.width],
    ['height', settings.height]
  ] as const) {
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`Isobands ${name} must be a non-negative integer`);
    }
  }
  if (settings.width < 1 || settings.height < 1) {
    throw new Error('Isobands width and height must be positive');
  }
  const [minX, minY, maxX, maxY] = settings.extent;
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) {
    throw new Error('Isobands extent must be finite');
  }
  target[0] = settings.breakCount;
  target[1] = firstBand;
  target[2] = lastBand;
  target[3] = 0;
  target[4] = minX;
  target[5] = minY;
  target[6] = maxX;
  target[7] = maxY;
  target[8] = (maxX - minX) / settings.width;
  target[9] = (maxY - minY) / settings.height;
  target[10] = 0;
  target[11] = 0;
  return target;
}
