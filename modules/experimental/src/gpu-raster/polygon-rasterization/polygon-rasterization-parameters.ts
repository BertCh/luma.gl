// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {validateParameterTarget} from '../raster-band-utils';
import type {GPURasterMetadata} from '../types';

/** Zone ID of a cell that no polygon covers, and of a point that joins no zone. */
export const GPU_POLYGON_RASTERIZATION_NO_ZONE = 0xffffffff;

/** Number of float32 values read from `GPUPolygonRasterizationProps.parameters`. */
export const GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH = 4;

/**
 * Packs the per-frame raster placement read by `GPUPolygonRasterization` and `GPURasterJoin`
 * from the raster grid.
 *
 * Layout: `[originX, originY, cellWidth, cellHeight]` as float32, where `originY` is the smallest
 * y of the raster (the bottom edge of the southernmost row) and `|cellHeight|` is the cell height.
 * The sign of `cellHeight` gives the row direction: positive when rows advance toward larger y
 * (row 0 is the southernmost row, `grid.affine[4] > 0`), negative when they advance toward smaller
 * y (the usual north-up image, `grid.affine[4] < 0`; row 0 is the northernmost row). Cell
 * `(column, row)` covers `[originX + column * cellWidth, originX + (column + 1) * cellWidth)`
 * horizontally; the kernels count rows from the smallest y and mirror the stored row index when
 * `cellHeight` is negative. A cell size that is not finite and nonzero (or a negative cell width)
 * makes the raster empty (every cell `GPU_POLYGON_RASTERIZATION_NO_ZONE`, every point outside).
 * Point pixels are addressed at their centers, so their grid corner moves by half a cell.
 *
 * @param grid The frame's grid: axis-aligned; its width and height must match the contributor's.
 * @param target Optional destination of at least 4 elements.
 * @returns `target`, filled.
 * @throws If the grid is rotated or sheared, or `target` is too short.
 */
export function getGPUPolygonRasterizationParameterValues(
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUPolygonRasterizationParameterValues',
    target,
    GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH
  );
  const [stepX, shearX, originX, shearY, stepY, originY] = grid.affine;
  if (shearX !== 0 || shearY !== 0) {
    throw new Error('getGPUPolygonRasterizationParameterValues needs an axis-aligned grid');
  }
  const cornerShift = grid.pixelInterpretation === 'point' ? -0.5 : 0;
  const cornerX = originX + stepX * cornerShift;
  const cornerY = originY + stepY * cornerShift;
  // Rows advancing toward smaller y end at the bottom edge, which is the smallest y.
  target[0] = cornerX;
  target[1] = stepY < 0 ? cornerY + stepY * grid.height : cornerY;
  target[2] = stepX;
  target[3] = stepY;
  return target;
}

/** WGSL helpers shared by the rasterization and join kernels. Requires an `extent` f32 binding. @internal */
export const POLYGON_RASTER_EXTENT_WGSL = /* wgsl */ `
${IS_FINITE_VALUE_WGSL}
struct RasterExtent { origin: vec2<f32>, cellSize: vec2<f32>, valid: bool, flipped: bool };
fn getRasterExtent() -> RasterExtent {
  let origin = vec2<f32>(extent[extentOffset], extent[extentOffset + 1u]);
  // A negative cell height means rows advance toward smaller y: the kernels count rows from the
  // smallest y and mirror the row index when they read or write the caller's rasters.
  let signedCellHeight = extent[extentOffset + 3u];
  let cellSize = vec2<f32>(extent[extentOffset + 2u], abs(signedCellHeight));
  let valid = isFiniteValue(origin.x) && isFiniteValue(origin.y) &&
    isFiniteValue(cellSize.x) && isFiniteValue(cellSize.y) && cellSize.x > 0.0 && cellSize.y > 0.0;
  return RasterExtent(origin, cellSize, valid, signedCellHeight < 0.0);
}
// Stored row of a row counted from the smallest y. Needs a HEIGHT constant.
fn getStoredRow(row: u32, rasterExtent: RasterExtent) -> u32 {
  return select(row, HEIGHT - 1u - row, rasterExtent.flipped);
}`;

/**
 * Returns a WGSL function `${functionName}(value) -> u32` that finds the range `i` in
 * `[0, rangeCount)` of an offsets binding with `offsets[i] <= value < offsets[i + 1]`, or
 * `0xffffffffu` when no range contains `value`. Empty ranges are skipped.
 *
 * @internal
 */
export function getOffsetRangeSearchSource(
  functionName: string,
  bindingName: string,
  rangeCount: number
): string {
  return /* wgsl */ `
fn ${functionName}(value: u32) -> u32 {
  // Largest i in [0, rangeCount) with offsets[i] <= value.
  var low = 0u;
  var high = ${rangeCount}u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (${bindingName}[${bindingName}Offset + middle] <= value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u) {
    return 0xffffffffu;
  }
  let range = low - 1u;
  if (value >= ${bindingName}[${bindingName}Offset + range + 1u]) {
    return 0xffffffffu;
  }
  return range;
}`;
}
