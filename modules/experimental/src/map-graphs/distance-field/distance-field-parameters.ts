// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 values read from `GPUDistanceFieldProps.settings`. */
export const GPU_DISTANCE_FIELD_PARAMETER_LENGTH = 8;

/** Sentinel for "no seed": unreached cells in allocation and nearest-cell outputs. */
export const GPU_DISTANCE_FIELD_NONE = 0xffffffff;

/** Largest supported grid width or height, so squared cell offsets fit in a `u32`. */
export const GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION = 32768;

/**
 * CPU description of the per-frame settings packed by {@link getGPUDistanceFieldParameterValues}.
 *
 * Give either `cellSize` (with an optional `origin`) or `bounds` together with `gridSize`.
 */
export type GPUDistanceFieldSettings = {
  /** Ground coordinate of the outer corner of cell `(0, 0)`, the minimum x and y. Defaults to `[0, 0]`. */
  origin?: readonly [number, number];
  /** Positive cell size `[x, y]` in ground units. Unequal sizes give an anisotropic metric. */
  cellSize?: readonly [number, number];
  /** Grid domain `[minX, minY, maxX, maxY]`; the cell size is derived from it and `gridSize`. */
  bounds?: readonly [number, number, number, number];
  /** `[width, height]` in cells, required with `bounds`. */
  gridSize?: readonly [number, number];
  /**
   * Cells farther than this ground distance from every seed get distance `+Infinity`, allocation
   * {@link GPU_DISTANCE_FIELD_NONE}, and a `0` in the within-distance mask. Defaults to `Infinity`.
   */
  maxDistance?: number;
};

/**
 * Packs settings into the 8-float layout read by `GPUDistanceField`:
 * `[originX, originY, cellSizeX, cellSizeY, 1 / cellSizeX, 1 / cellSizeY, maxDistance, 0]`.
 *
 * Seed positions map to cells with `floor((position - origin) * inverseCellSize)` in f32, so the
 * inverse is computed once here instead of dividing on the GPU. Rewriting the values between
 * encodings never recompiles the graph.
 *
 * @param settings Grid placement, cell size, and optional distance limit.
 * @param target Optional destination of at least 8 elements.
 * @throws If the cell size is not finite and positive, or `target` is too short.
 */
export function getGPUDistanceFieldParameterValues(
  settings: GPUDistanceFieldSettings,
  target: Float32Array = new Float32Array(GPU_DISTANCE_FIELD_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_DISTANCE_FIELD_PARAMETER_LENGTH) {
    throw new Error(
      `Distance field parameter target must hold ${GPU_DISTANCE_FIELD_PARAMETER_LENGTH} elements`
    );
  }
  let origin = settings.origin ?? [0, 0];
  let cellSize = settings.cellSize;
  if (settings.bounds) {
    const [minX, minY, maxX, maxY] = settings.bounds;
    const [width, height] = settings.gridSize ?? [0, 0];
    if (!(width >= 1 && height >= 1)) {
      throw new Error('Distance field bounds require a positive gridSize');
    }
    origin = [minX, minY];
    cellSize = [(maxX - minX) / width, (maxY - minY) / height];
  }
  if (!cellSize) {
    throw new Error('Distance field settings require cellSize or bounds');
  }
  const cellSizeX = Math.fround(cellSize[0]);
  const cellSizeY = Math.fround(cellSize[1]);
  if (
    !(Number.isFinite(cellSizeX) && cellSizeX > 0 && Number.isFinite(cellSizeY) && cellSizeY > 0)
  ) {
    throw new Error('Distance field cell size must be finite and positive');
  }
  target.fill(0, 0, GPU_DISTANCE_FIELD_PARAMETER_LENGTH);
  target[0] = origin[0];
  target[1] = origin[1];
  target[2] = cellSizeX;
  target[3] = cellSizeY;
  target[4] = 1 / cellSizeX;
  target[5] = 1 / cellSizeY;
  target[6] = settings.maxDistance ?? Infinity;
  return target;
}
