// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPURasterSampling} parameter view. */
export const GPU_RASTER_SAMPLING_PARAMETER_LENGTH = 12;
/** Number of float32 elements in a {@link GPURasterProfile} parameter view. */
export const GPU_RASTER_PROFILE_PARAMETER_LENGTH = 12;

/** Interpolation used by {@link GPURasterSampling} and {@link GPURasterProfile}. */
export type GPURasterSamplingMethod = 'nearest' | 'bilinear' | 'bicubic';
/** How interpolation treats nodata cells in its support. */
export type GPURasterSamplingNoDataPolicy = 'strict' | 'renormalize';

/** Per-frame settings of {@link GPURasterSampling}. */
export type GPURasterSamplingSettings = {
  /** Raster width in cells; must match the contributor's `width`. */
  width: number;
  /** Raster height in cells; must match the contributor's `height`. */
  height: number;
  /** Raster extent `[minX, minY, maxX, maxY]`; row 0 is at `minY`. */
  extent: readonly [number, number, number, number];
  /** Interpolation method. Defaults to `'bilinear'`. */
  method?: GPURasterSamplingMethod;
  /** Nodata handling for bilinear and bicubic. Defaults to `'strict'`. */
  noDataPolicy?: GPURasterSamplingNoDataPolicy;
};

/** Per-frame settings of {@link GPURasterProfile}. */
export type GPURasterProfileSettings = GPURasterSamplingSettings & {
  /** Distance between samples along a path, in extent units. Must be finite and positive. */
  spacing: number;
};

const METHOD_CODES: Record<GPURasterSamplingMethod, number> = {nearest: 0, bilinear: 1, bicubic: 2};

function packSamplingParameters(
  name: string,
  settings: GPURasterSamplingSettings,
  spacing: number,
  target: Float32Array
): Float32Array {
  if (target.length < GPU_RASTER_SAMPLING_PARAMETER_LENGTH) {
    throw new Error(
      `${name} parameter target must hold ${GPU_RASTER_SAMPLING_PARAMETER_LENGTH} elements`
    );
  }
  const {width, height, extent} = settings;
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) {
    throw new Error(`${name} width and height must be positive integers`);
  }
  if (extent.length !== 4 || !extent.every(Number.isFinite)) {
    throw new Error(`${name} extent must be four finite numbers`);
  }
  const [minX, minY, maxX, maxY] = extent;
  if (!(maxX > minX) || !(maxY > minY)) {
    throw new Error(`${name} extent must have maxX > minX and maxY > minY`);
  }
  const method = settings.method ?? 'bilinear';
  if (!(method in METHOD_CODES)) {
    throw new Error(`${name} method ${method} is not supported`);
  }
  const noDataPolicy = settings.noDataPolicy ?? 'strict';
  if (noDataPolicy !== 'strict' && noDataPolicy !== 'renormalize') {
    throw new Error(`${name} noDataPolicy ${noDataPolicy} is not supported`);
  }
  const cellWidth = Math.fround((maxX - minX) / width);
  const cellHeight = Math.fround((maxY - minY) / height);
  target.fill(0, 0, GPU_RASTER_SAMPLING_PARAMETER_LENGTH);
  target[0] = minX;
  target[1] = minY;
  target[2] = maxX;
  target[3] = maxY;
  target[4] = cellWidth;
  target[5] = cellHeight;
  // Reciprocals are rounded from the double quotient so the GPU multiplies instead of dividing.
  target[6] = width / (maxX - minX);
  target[7] = height / (maxY - minY);
  target[8] = METHOD_CODES[method];
  target[9] = noDataPolicy === 'renormalize' ? 1 : 0;
  target[10] = spacing;
  return target;
}

/**
 * Packs per-frame {@link GPURasterSampling} parameters.
 *
 * Layout (float32): `[minX, minY, maxX, maxY, cellWidth, cellHeight, 1 / cellWidth,
 * 1 / cellHeight, method, renormalize, 0, 0]` with method codes `nearest` 0, `bilinear` 1,
 * `bicubic` 2 and `renormalize` 1 for the `'renormalize'` policy.
 *
 * @param settings Raster size, extent, method and nodata policy.
 * @param target Optional destination of at least 12 elements.
 */
export function getGPURasterSamplingParameterValues(
  settings: GPURasterSamplingSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_SAMPLING_PARAMETER_LENGTH)
): Float32Array {
  return packSamplingParameters('Raster sampling', settings, 0, target);
}

/**
 * Packs per-frame {@link GPURasterProfile} parameters.
 *
 * Same layout as {@link getGPURasterSamplingParameterValues} with `spacing` in element 10.
 *
 * @param settings Raster size, extent, method, nodata policy and spacing.
 * @param target Optional destination of at least 12 elements.
 */
export function getGPURasterProfileParameterValues(
  settings: GPURasterProfileSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_PROFILE_PARAMETER_LENGTH)
): Float32Array {
  if (!(settings.spacing > 0) || !Number.isFinite(settings.spacing)) {
    throw new Error('Raster profile spacing must be finite and positive');
  }
  return packSamplingParameters('Raster profile', settings, settings.spacing, target);
}
