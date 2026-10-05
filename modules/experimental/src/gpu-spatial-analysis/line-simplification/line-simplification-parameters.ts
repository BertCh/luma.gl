// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a line-simplification parameter buffer. */
export const GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineSimplification`. */
export type GPULineSimplificationParameters = {
  /**
   * Douglas-Peucker tolerance in position units. A vertex is kept when its importance is strictly
   * greater than the tolerance, which is the classic `dmax > epsilon` split rule. Line endpoints
   * are always kept. Must be finite and non-negative.
   */
  tolerance: number;
};

/**
 * Packs line-simplification parameters into the 4-element float32 layout read by
 * `GPULineSimplification`.
 *
 * Layout: `[tolerance, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between
 * encodings to change the tolerance without recompiling.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If the tolerance is not finite, is negative, or `target` is too short.
 */
export function getGPULineSimplificationParameterValues(
  parameters: GPULineSimplificationParameters,
  target: Float32Array = new Float32Array(GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH) {
    throw new Error(
      `Line simplification target must hold ${GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH} elements`
    );
  }
  const {tolerance} = parameters;
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new Error('Line simplification tolerance must be finite and non-negative');
  }
  target.set([tolerance, 0, 0, 0]);
  return target;
}
