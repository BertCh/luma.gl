// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a `GPUGeometryPredicates` parameter buffer. */
export const GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPUGeometryPredicates`. */
export type GPUGeometryPredicatesParameters = {
  /**
   * `equalsExact` tolerance: two vertices match when their distance is at most this value, in
   * position units. Defaults to `0` (vertices must coincide).
   */
  tolerance?: number;
};

/**
 * Packs `GPUGeometryPredicates` parameters into the 4-element float32 layout `[tolerance, 0, 0, 0]`.
 * Write the result into a `GPUParameterBuffer` between encodings to change the tolerance without
 * recompiling.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the tolerance is negative or NaN, or `target` is too short.
 */
export function getGPUGeometryPredicatesParameterValues(
  parameters: GPUGeometryPredicatesParameters = {},
  target: Float32Array = new Float32Array(GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH) {
    throw new Error(
      `Geometry predicates target must hold ${GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH} elements`
    );
  }
  const tolerance = parameters.tolerance ?? 0;
  if (Number.isNaN(tolerance) || tolerance < 0) {
    throw new Error('Geometry predicates tolerance must be non-negative');
  }
  // Infinity is not exact under every WGSL backend: clamp to the largest finite f32.
  target.set([Math.min(tolerance, 3.4028234663852886e38), 0, 0, 0]);
  return target;
}
