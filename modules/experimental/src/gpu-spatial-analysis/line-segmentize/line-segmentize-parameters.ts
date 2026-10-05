// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a `GPULineSegmentize` or `GPUGreatCircleArcs` parameter buffer. */
export const GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineSegmentize`. */
export type GPULineSegmentizeParameters = {
  /**
   * Largest output segment length, in position units (`'planar'`) or sphere-radius units
   * (`'spherical'`, meters by default). `0` or `Infinity` disables densification.
   */
  maximumSegmentLength: number;
};

/** CPU description of the per-frame parameters of `GPUGreatCircleArcs`. */
export type GPUGreatCircleArcsParameters = {
  /**
   * Largest arc segment length in sphere-radius units (meters by default). `0` or `Infinity` lets
   * `minimumSegments` alone decide the resolution.
   */
  maximumSegmentLength: number;
  /**
   * Smallest number of segments per arc. Set `maximumSegmentLength` to `0` and this to `n` for a
   * fixed `n` segments per arc. Default 1.
   */
  minimumSegments?: number;
};

function getTarget(target: Float32Array, name: string): Float32Array {
  if (target.length < GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH) {
    throw new Error(`${name} target must hold ${GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} elements`);
  }
  return target;
}

function getMaximumSegmentLength(value: number, name: string): number {
  if (Number.isNaN(value) || value < 0) {
    throw new Error(`${name} maximumSegmentLength must be non-negative`);
  }
  // f32 has no exact Infinity guarantee under every WGSL backend: encode "no limit" as 0.
  return Number.isFinite(value) ? value : 0;
}

/**
 * Packs `GPULineSegmentize` parameters into the 4-element float32 layout
 * `[maximumSegmentLength, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between
 * encodings to change the resolution without recompiling.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the length is negative or NaN, or `target` is too short.
 */
export function getGPULineSegmentizeParameterValues(
  parameters: GPULineSegmentizeParameters,
  target: Float32Array = new Float32Array(GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH)
): Float32Array {
  const name = 'Line segmentize';
  getTarget(target, name);
  target.set([getMaximumSegmentLength(parameters.maximumSegmentLength, name), 0, 0, 0]);
  return target;
}

/**
 * Packs `GPUGreatCircleArcs` parameters into the 4-element float32 layout
 * `[maximumSegmentLength, minimumSegments, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the length is negative or NaN, `minimumSegments` is not a positive integer, or
 * `target` is too short.
 */
export function getGPUGreatCircleArcsParameterValues(
  parameters: GPUGreatCircleArcsParameters,
  target: Float32Array = new Float32Array(GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH)
): Float32Array {
  const name = 'Great circle arcs';
  getTarget(target, name);
  const minimumSegments = parameters.minimumSegments ?? 1;
  if (!Number.isInteger(minimumSegments) || minimumSegments < 1) {
    throw new Error(`${name} minimumSegments must be a positive integer`);
  }
  target.set([
    getMaximumSegmentLength(parameters.maximumSegmentLength, name),
    minimumSegments,
    0,
    0
  ]);
  return target;
}
