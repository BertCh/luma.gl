// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of uint32 elements in a {@link GPUDotDensity} or {@link GPURandomPointsInPolygon} parameter view. */
export const GPU_DOT_DENSITY_PARAMETER_LENGTH = 8;

/** Per-frame settings of {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}. */
export type GPUDotDensitySettings = {
  /** Random seed. Every dot position is a pure function of the seed and its (slot, rank). */
  seed: number;
  /**
   * Dots per unit of value (the reciprocal of "people per dot"), usually zoom dependent. Ignored by
   * {@link GPURandomPointsInPolygon}. Zero or a non-finite value draws no dots. Defaults to 1.
   */
  dotsPerUnit?: number;
  /**
   * Placement `[originX, originY, cellWidth, cellHeight]` of the optional dasymetric mask raster
   * (row 0 has the smallest y). Defaults to `[0, 0, 1, 1]`; ignored without a mask.
   */
  maskExtent?: readonly [number, number, number, number];
};

/**
 * Packs per-frame parameters of {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}.
 *
 * Layout (uint32 words; floats stored as their IEEE-754 bits so one view carries both):
 * `[seed, f32 dotsPerUnit, f32 maskOriginX, f32 maskOriginY, f32 maskCellWidth,
 * f32 maskCellHeight, 0, 0]`.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least {@link GPU_DOT_DENSITY_PARAMETER_LENGTH} elements.
 */
export function getGPUDotDensityParameterValues(
  settings: GPUDotDensitySettings,
  target: Uint32Array = new Uint32Array(GPU_DOT_DENSITY_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_DOT_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `Dot density parameter target must hold ${GPU_DOT_DENSITY_PARAMETER_LENGTH} elements`
    );
  }
  if (!Number.isInteger(settings.seed) || settings.seed < 0 || settings.seed > 0xffffffff) {
    throw new Error('Dot density seed must be a uint32 integer');
  }
  const floats = new Float32Array(target.buffer, target.byteOffset, target.length);
  const maskExtent = settings.maskExtent ?? [0, 0, 1, 1];
  target[0] = settings.seed;
  floats[1] = settings.dotsPerUnit ?? 1;
  floats[2] = maskExtent[0];
  floats[3] = maskExtent[1];
  floats[4] = maskExtent[2];
  floats[5] = maskExtent[3];
  target[6] = 0;
  target[7] = 0;
  return target;
}
