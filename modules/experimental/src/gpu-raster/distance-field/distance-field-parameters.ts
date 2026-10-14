// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getRasterLocalOrigin} from '../raster-utils';
import {validateParameterTarget} from '../raster-band-utils';
import type {GPURasterMetadata} from '../types';

/** Number of float32 values read from `GPUDistanceFieldProps.parameters`. */
export const GPU_DISTANCE_FIELD_PARAMETER_LENGTH = 8;

/** Sentinel for "no seed": unreached cells in allocation and nearest-cell outputs. */
export const GPU_DISTANCE_FIELD_NONE = 0xffffffff;

/** Largest supported grid width or height, so squared cell offsets fit in a `u32`. */
export const GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION = 32768;

/**
 * CPU description of the per-frame settings packed by {@link getGPUDistanceFieldParameterValues}.
 * The grid origin and cell size come from the grid passed next to it.
 */
export type GPUDistanceFieldSettings = {
  /**
   * Cells farther than this ground distance from every seed get distance `+Infinity`, allocation
   * {@link GPU_DISTANCE_FIELD_NONE}, and a `0` in the within-distance mask. Defaults to `Infinity`.
   */
  maximumDistance?: number;
  /**
   * Float64 point that seed positions are uploaded relative to. The packer subtracts it from the
   * grid origin in float64, so seeds snap to cells from origin-relative float32 values. Defaults
   * to `[0, 0]` (absolute seeds). Pass the raster or tile corner for projected grids: absolute
   * float32 seeds and origins are quantized to 0.5 m at a UTM northing of 4.6e6 m, so seeds within
   * that of a cell edge snap to the wrong cell; origin-relative ones keep about `1.2e-7` of their
   * distance from the origin (1.2 cm at 100 km).
   */
  localOrigin?: readonly [number, number];
};

/**
 * Packs settings into the 8-float layout read by `GPUDistanceField`:
 * `[originX, originY, cellSizeX, cellSizeY, 1 / stepX, 1 / stepY, maximumDistance, 0]`, with the
 * origin relative to `localOrigin` (subtracted in float64). The origin is the outer corner of
 * cell `(0, 0)` and the step is the signed cell step along the columns and rows (`affine[0]`,
 * `affine[4]`), so north-up grids (negative row step) and north-down grids both snap correctly.
 * Cell size and origin come from `grid`, which must be axis-aligned.
 *
 * Seed positions map to cells with `floor((position - origin) * inverseStep)` in f32, so the
 * inverse is computed once here instead of dividing on the GPU. Rewriting the values between
 * encodings never recompiles the graph.
 *
 * @param settings Optional distance limit and local origin.
 * @param grid The frame's grid; its width and height must match the contributor's.
 * @param target Optional destination of at least 8 elements.
 * @throws If the grid is rotated or its cell size is not finite and positive, or `target` is too short.
 */
export function getGPUDistanceFieldParameterValues(
  settings: GPUDistanceFieldSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_DISTANCE_FIELD_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUDistanceFieldParameterValues',
    target,
    GPU_DISTANCE_FIELD_PARAMETER_LENGTH
  );
  const [stepX, shearX, originX, shearY, stepY, originY] = grid.affine;
  if (shearX !== 0 || shearY !== 0) {
    throw new Error('getGPUDistanceFieldParameterValues needs an axis-aligned grid');
  }
  const cellSizeX = Math.fround(Math.abs(stepX));
  const cellSizeY = Math.fround(Math.abs(stepY));
  if (
    !(Number.isFinite(cellSizeX) && cellSizeX > 0 && Number.isFinite(cellSizeY) && cellSizeY > 0)
  ) {
    throw new Error('getGPUDistanceFieldParameterValues cell size must be finite and positive');
  }
  // Point pixels are addressed at their centers; the kernels want the outer corner.
  const cornerShift = grid.pixelInterpretation === 'point' ? -0.5 : 0;
  const [localOriginX, localOriginY] = getRasterLocalOrigin('Distance field', settings.localOrigin);
  target.fill(0, 0, GPU_DISTANCE_FIELD_PARAMETER_LENGTH);
  target[0] = originX + stepX * cornerShift - localOriginX;
  target[1] = originY + stepY * cornerShift - localOriginY;
  target[2] = cellSizeX;
  target[3] = cellSizeY;
  target[4] = 1 / Math.fround(stepX);
  target[5] = 1 / Math.fround(stepY);
  target[6] = settings.maximumDistance ?? Infinity;
  return target;
}
