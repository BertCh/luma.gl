// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUStreamlines} parameter view. */
export const GPU_STREAMLINES_PARAMETER_LENGTH = 12;

/** Number of uint32 elements in a {@link GPUStreamlines} word parameter view. */
export const GPU_STREAMLINES_WORD_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUStreamlines}. */
export type GPUStreamlinesSettings = {
  /** Field placement `[originX, originY, cellWidth, cellHeight]`; row 0 has the smallest y. */
  fieldExtent: readonly [number, number, number, number];
  /**
   * Occupancy grid placement `[originX, originY, cellWidth, cellHeight]`. The cell size is the
   * separation distance between streamlines; seeds and traced points stay inside the grid.
   */
  gridExtent: readonly [number, number, number, number];
  /** Integration step in world units. */
  stepLength: number;
  /** A streamline stops where the field is slower than this. Defaults to 0. */
  minimumSpeed?: number;
  /** Seed of the jittered seed lattice and the line priorities. Defaults to 0. */
  seed?: number;
  /** Streamlines with fewer points after trimming are rejected. Defaults to 2. */
  minimumPoints?: number;
};

/**
 * Packs float parameters of {@link GPUStreamlines}.
 *
 * Layout: `[fieldOriginX, fieldOriginY, fieldCellWidth, fieldCellHeight, gridOriginX,
 * gridOriginY, gridCellWidth, gridCellHeight, 1 / gridCellWidth, 1 / gridCellHeight, stepLength,
 * minimumSpeed]`. The reciprocals are rounded once here so the GPU and the CPU oracle assign
 * points to grid cells with the same correctly rounded subtraction and multiplication.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least 12 elements.
 */
export function getGPUStreamlinesParameterValues(
  settings: GPUStreamlinesSettings,
  target: Float32Array = new Float32Array(GPU_STREAMLINES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_STREAMLINES_PARAMETER_LENGTH) {
    throw new Error(
      `Streamlines parameter target must hold ${GPU_STREAMLINES_PARAMETER_LENGTH} elements`
    );
  }
  const [, , cellWidth, cellHeight] = settings.gridExtent;
  if (!(cellWidth > 0) || !(cellHeight > 0)) {
    throw new Error('Streamlines grid cells must have a positive size');
  }
  target.set([
    ...settings.fieldExtent,
    ...settings.gridExtent,
    1 / Math.fround(cellWidth),
    1 / Math.fround(cellHeight),
    settings.stepLength,
    settings.minimumSpeed ?? 0
  ]);
  return target;
}

/**
 * Packs integer parameters of {@link GPUStreamlines}. Layout: `[seed, minimumPoints, 0, 0]`.
 *
 * @param settings Per-frame settings; reads `seed` and `minimumPoints`.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPUStreamlinesWordParameterValues(
  settings: Pick<GPUStreamlinesSettings, 'seed' | 'minimumPoints'>,
  target: Uint32Array = new Uint32Array(GPU_STREAMLINES_WORD_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_STREAMLINES_WORD_PARAMETER_LENGTH) {
    throw new Error(
      `Streamlines word parameter target must hold ${GPU_STREAMLINES_WORD_PARAMETER_LENGTH} elements`
    );
  }
  const seed = settings.seed ?? 0;
  const minimumPoints = settings.minimumPoints ?? 2;
  for (const [name, value] of [
    ['seed', seed],
    ['minimumPoints', minimumPoints]
  ] as const) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error(`Streamlines ${name} must be a uint32 integer`);
    }
  }
  target.set([seed, minimumPoints, 0, 0]);
  return target;
}
