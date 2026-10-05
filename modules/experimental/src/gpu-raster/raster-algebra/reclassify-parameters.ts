// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateParameterTarget} from './raster-algebra-utils';

/** Number of float32 elements in a {@link GPURasterReclassify} parameter view. */
export const GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH = 4;

/** Class written by {@link GPURasterReclassify} for nodata rows. */
export const GPU_RASTER_RECLASSIFY_NO_DATA_CLASS = 0xffffffff;

/** Per-frame settings of {@link GPURasterReclassify}. */
export type GPURasterReclassifySettings = {
  /** Number of active ascending breaks, at most the contributor's `maximumBreakCount`. */
  breakCount: number;
  /**
   * Which side of each interval is closed. `'left'` (default) gives intervals `[b[k - 1], b[k])`,
   * so the class is the number of breaks `<= value`; `'right'` gives `(b[k - 1], b[k]]`, so the
   * class is the number of breaks `< value`.
   */
  closed?: 'left' | 'right';
};

/**
 * Packs per-frame {@link GPURasterReclassify} parameters.
 *
 * Layout (float32): `[breakCount, closedRight, 0, 0]`.
 *
 * @param settings Active break count and interval closure.
 * @param target Optional destination of at least 4 elements.
 * @throws If the target is too short or `breakCount` is not a non-negative integer.
 */
export function getGPURasterReclassifyParameterValues(
  settings: GPURasterReclassifySettings,
  target: Float32Array = new Float32Array(GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget('Raster reclassify', target, GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH);
  if (!Number.isSafeInteger(settings.breakCount) || settings.breakCount < 0) {
    throw new Error('Raster reclassify breakCount must be a non-negative integer');
  }
  target[0] = settings.breakCount;
  target[1] = settings.closed === 'right' ? 1 : 0;
  target[2] = 0;
  target[3] = 0;
  return target;
}
