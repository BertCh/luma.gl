// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUIsolines} parameter view. */
export const GPU_ISOLINES_PARAMETER_LENGTH = 8;

/** Per-frame settings of {@link GPUIsolines}. */
export type GPUIsolinesSettings = {
  /** Raster width in samples. Must match the contributor topology. */
  width: number;
  /** Raster height in samples. Must match the contributor topology. */
  height: number;
  /** Number of active leading rows of the `levels` view. Clamped by the contributor to its maximum. */
  levelCount: number;
  /**
   * World extent `[minX, minY, maxX, maxY]` of the raster. Sample `(column, row)` sits at the
   * centre of its cell, `minX + (column + 0.5) * cellWidth`; row 0 is at `minY`.
   */
  extent: readonly [number, number, number, number];
};

/**
 * Packs per-frame {@link GPUIsolines} parameters.
 *
 * Layout (float32): `[levelCount, minX, minY, cellWidth, cellHeight, 0, 0, 0]`. Cell sizes are
 * computed here in float64 and rounded once to float32, so oracles can reproduce them exactly.
 *
 * @param settings Raster size, active level count and world extent.
 * @param target Optional destination of at least 8 elements.
 */
export function getGPUIsolinesParameterValues(
  settings: GPUIsolinesSettings,
  target: Float32Array = new Float32Array(GPU_ISOLINES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_ISOLINES_PARAMETER_LENGTH) {
    throw new Error(
      `Isolines parameter target must hold ${GPU_ISOLINES_PARAMETER_LENGTH} elements`
    );
  }
  const {width, height, levelCount, extent} = settings;
  for (const [name, value] of [
    ['width', width],
    ['height', height]
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 2) {
      throw new Error(`Isolines ${name} must be an integer of at least 2`);
    }
  }
  if (!Number.isSafeInteger(levelCount) || levelCount < 0) {
    throw new Error('Isolines levelCount must be a non-negative integer');
  }
  const [minX, minY, maxX, maxY] = extent;
  if (![minX, minY, maxX, maxY].every(Number.isFinite) || maxX <= minX || maxY <= minY) {
    throw new Error('Isolines extent must be finite with maxX > minX and maxY > minY');
  }
  target.fill(0, 0, GPU_ISOLINES_PARAMETER_LENGTH);
  target[0] = levelCount;
  target[1] = minX;
  target[2] = minY;
  target[3] = (maxX - minX) / width;
  target[4] = (maxY - minY) / height;
  return target;
}
