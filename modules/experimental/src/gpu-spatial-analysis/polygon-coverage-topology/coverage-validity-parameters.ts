// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a coverage-validity parameter buffer. */
export const GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPUCoverageValidity`. */
export type GPUCoverageValidityParameters = {
  /**
   * Largest gap, in position units, that counts as an error: Shapely `gap_width`. Two segments of
   * different polygons that run nearly parallel within this distance are reported. `0` disables
   * gap detection. Must be finite and non-negative.
   */
  gapWidth: number;
};

/**
 * Packs coverage-validity parameters into the 4-element float32 layout read by
 * `GPUCoverageValidity`.
 *
 * Layout: `[gapWidth, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between encodings to
 * change the gap width without recompiling.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If the gap width is not finite, is negative, or `target` is too short.
 */
export function getGPUCoverageValidityParameterValues(
  parameters: GPUCoverageValidityParameters,
  target: Float32Array = new Float32Array(GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH) {
    throw new Error(
      `Coverage validity target must hold ${GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH} elements`
    );
  }
  const {gapWidth} = parameters;
  if (!Number.isFinite(gapWidth) || gapWidth < 0) {
    throw new Error('Coverage validity gapWidth must be finite and non-negative');
  }
  target.set([gapWidth, 0, 0, 0]);
  return target;
}
