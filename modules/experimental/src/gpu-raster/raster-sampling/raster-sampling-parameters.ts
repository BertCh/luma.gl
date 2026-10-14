// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getRasterLocalOrigin} from '../raster-utils';
import {validateParameterTarget, validateRasterGrid} from '../raster-band-utils';
import type {GPURasterMetadata} from '../types';

/** Number of float32 elements in a {@link GPURasterSampling} parameter view. */
export const GPU_RASTER_SAMPLING_PARAMETER_LENGTH = 12;
/** Number of float32 elements in a {@link GPURasterProfile} parameter view. */
export const GPU_RASTER_PROFILE_PARAMETER_LENGTH = 12;

/** Interpolation used by {@link GPURasterSampling} and {@link GPURasterProfile}. */
export type GPURasterSamplingMethod = 'nearest' | 'bilinear' | 'bicubic';
/** How interpolation treats nodata cells in its support. */
export type GPURasterSamplingNoDataPolicy = 'strict' | 'renormalize';

/**
 * Per-frame settings of {@link GPURasterSampling}. The raster extent comes from the grid passed
 * next to them.
 */
export type GPURasterSamplingSettings = {
  /** Interpolation method. Defaults to `'bilinear'`. */
  method?: GPURasterSamplingMethod;
  /** Nodata handling for bilinear and bicubic. Defaults to `'strict'`. */
  noDataPolicy?: GPURasterSamplingNoDataPolicy;
  /**
   * Float64 point that query positions (sampling points, profile line vertices) are given relative
   * to, and that profile sample positions are written relative to. Defaults to `[0, 0]` (absolute
   * positions). Pass the raster or tile corner `[minX, minY]` for projected rasters: absolute
   * float32 positions are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to
   * about `1.2e-7` of their distance from the origin (1.2 cm at 100 km).
   */
  localOrigin?: readonly [number, number];
};

/** Per-frame settings of {@link GPURasterProfile}. */
export type GPURasterProfileSettings = GPURasterSamplingSettings & {
  /** Distance between samples along a line, in grid world units. Must be finite and positive. */
  spacing: number;
};

const METHOD_CODES: Record<GPURasterSamplingMethod, number> = {nearest: 0, bilinear: 1, bicubic: 2};

function packSamplingParameters(
  name: string,
  settings: GPURasterSamplingSettings,
  grid: GPURasterMetadata,
  spacing: number,
  target: Float32Array
): Float32Array {
  validateParameterTarget(name, target, GPU_RASTER_SAMPLING_PARAMETER_LENGTH);
  validateRasterGrid(name, grid);
  const {width, height} = grid;
  const [stepX, shearX, originX, shearY, stepY, originY] = grid.affine;
  if (shearX !== 0 || shearY !== 0) {
    throw new Error(`${name} needs an axis-aligned grid`);
  }
  if (!(stepX > 0) || !(Math.abs(stepY) > 0)) {
    throw new Error(`${name} grid cell size must be finite and positive`);
  }
  // Point pixels are addressed at their centers; the sampler wants the outer corner.
  const cornerShift = grid.pixelInterpretation === 'point' ? -0.5 : 0;
  const cornerX = originX + stepX * cornerShift;
  const cornerY = originY + stepY * cornerShift;
  const minX = cornerX;
  const maxX = cornerX + stepX * width;
  // Rows advance toward larger y when stepY > 0 (row 0 at minY), toward smaller y otherwise.
  const rowEdgeY = cornerY + stepY * height;
  const minY = Math.min(cornerY, rowEdgeY);
  const maxY = Math.max(cornerY, rowEdgeY);
  const method = settings.method ?? 'bilinear';
  if (!(method in METHOD_CODES)) {
    throw new Error(`${name} method ${method} is not supported`);
  }
  const noDataPolicy = settings.noDataPolicy ?? 'strict';
  if (noDataPolicy !== 'strict' && noDataPolicy !== 'renormalize') {
    throw new Error(`${name} noDataPolicy ${noDataPolicy} is not supported`);
  }
  const [localOriginX, localOriginY] = getRasterLocalOrigin(name, settings.localOrigin);
  target.fill(0, 0, GPU_RASTER_SAMPLING_PARAMETER_LENGTH);
  target[0] = minX - localOriginX;
  target[1] = minY - localOriginY;
  target[2] = maxX - localOriginX;
  target[3] = maxY - localOriginY;
  target[4] = Math.fround(stepX);
  target[5] = Math.fround(Math.abs(stepY));
  // Reciprocals are rounded from the double quotient so the GPU multiplies instead of dividing.
  // The row reciprocal is signed: positive when row 0 is at minY, negative when it is at maxY.
  target[6] = 1 / stepX;
  target[7] = 1 / stepY;
  target[8] = METHOD_CODES[method];
  target[9] = noDataPolicy === 'renormalize' ? 1 : 0;
  target[10] = spacing;
  // y of the outer edge of row 0, which the row coordinate is measured from.
  target[11] = cornerY - localOriginY;
  return target;
}

/**
 * Packs per-frame {@link GPURasterSampling} parameters.
 *
 * Layout (float32): `[minX, minY, maxX, maxY, cellWidth, cellHeight, 1 / cellWidth,
 * 1 / rowStep, method, renormalize, 0, rowOriginY]` with method codes `nearest` 0, `bilinear` 1,
 * `bicubic` 2 and `renormalize` 1 for the `'renormalize'` policy. `rowStep` is the signed row step
 * `grid.affine[4]` and `rowOriginY` the y of the outer edge of row 0, so both north-up and
 * north-down grids index rows correctly. The extent, cell size and row direction come from `grid`
 * (axis-aligned); positions are relative to `localOrigin`, subtracted in float64 before the
 * float32 rounding.
 *
 * @param settings Method, nodata policy and optional local origin.
 * @param grid The frame's grid; its width and height must match the contributor's.
 * @param target Optional destination of at least 12 elements.
 */
export function getGPURasterSamplingParameterValues(
  settings: GPURasterSamplingSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_RASTER_SAMPLING_PARAMETER_LENGTH)
): Float32Array {
  return packSamplingParameters('getGPURasterSamplingParameterValues', settings, grid, 0, target);
}

/**
 * Packs per-frame {@link GPURasterProfile} parameters.
 *
 * Same layout as {@link getGPURasterSamplingParameterValues} with `spacing` in element 10.
 * The grid supplies the extent, cell size and row direction.
 *
 * @param settings Method, nodata policy, optional local origin and spacing.
 * @param grid The frame's grid; its width and height must match the contributor's.
 * @param target Optional destination of at least 12 elements.
 */
export function getGPURasterProfileParameterValues(
  settings: GPURasterProfileSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_RASTER_PROFILE_PARAMETER_LENGTH)
): Float32Array {
  if (!(settings.spacing > 0) || !Number.isFinite(settings.spacing)) {
    throw new Error('getGPURasterProfileParameterValues spacing must be finite and positive');
  }
  return packSamplingParameters(
    'getGPURasterProfileParameterValues',
    settings,
    grid,
    settings.spacing,
    target
  );
}
