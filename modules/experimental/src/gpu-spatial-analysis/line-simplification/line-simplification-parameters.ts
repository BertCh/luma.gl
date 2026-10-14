// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance
} from '../../analysis-core/contracts/spatial-context';
import type {GPULineSimplificationMethod} from './gpu-line-simplification';

/** Number of float32 elements in a line-simplification parameter buffer. */
export const GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineSimplification`. */
export type GPULineSimplificationSettings = {
  /**
   * Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32
   * quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not
   * change the packed values.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Douglas-Peucker tolerance in position units (the context length unit in a great-circle context). A vertex is kept when its importance is strictly
   * greater than the tolerance, which is the classic `dmax > epsilon` split rule. Line endpoints
   * are always kept. Visvalingam uses an effective-area tolerance in squared position units.
   * Must be finite and non-negative.
   */
  tolerance: number;
  /**
   * Algorithm used by the operator. Defaults to `'douglas-peucker'`; `'visvalingam'` uses an area
   * tolerance and skips the distance warning. Does not change the packed values.
   */
  method?: GPULineSimplificationMethod;
};

/**
 * Packs line-simplification parameters into the 4-element float32 layout read by
 * `GPULineSimplification`.
 *
 * Layout: `[tolerance, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between
 * encodings to change the tolerance without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If the tolerance is not finite, is negative, or `target` is too short.
 */
export function getGPULineSimplificationParameterValues(
  settings: GPULineSimplificationSettings,
  target: Float32Array = new Float32Array(GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH) {
    throw new Error(
      `Line simplification target must hold ${GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH} elements`
    );
  }
  const {tolerance} = settings;
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new Error('Line simplification tolerance must be finite and non-negative');
  }
  if ((settings.method ?? 'douglas-peucker') === 'douglas-peucker') {
    warnGPUShortLongitudeLatitudeDistance(
      'GPULineSimplification',
      settings.spatialContext,
      tolerance
    );
  }
  target.set([tolerance, 0, 0, 0]);
  return target;
}
