// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Zone ID of a cell that no polygon covers, and of a point that joins no zone. */
export const GPU_POLYGON_RASTERIZATION_NO_ZONE = 0xffffffff;

/** Number of float32 elements in a raster extent parameter view. */
export const GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH = 4;

/**
 * Packs the per-frame raster extent read by `GPUPolygonRasterization` and `GPURasterJoin`.
 *
 * Layout: `[originX, originY, cellWidth, cellHeight]` as float32. Cell `(column, row)` covers
 * `[originX + column * cellWidth, originX + (column + 1) * cellWidth)` horizontally and
 * `[originY + row * cellHeight, originY + (row + 1) * cellHeight)` vertically, so row 0 is the row
 * with the smallest y. Its center is `(originX + (column + 0.5) * cellWidth, originY + (row + 0.5) *
 * cellHeight)`. A cell size that is not finite and positive makes the raster empty (every cell
 * `GPU_POLYGON_RASTERIZATION_NO_ZONE`, every point outside).
 *
 * @param originX Left edge of column 0, in the coordinate system of the polygons and points.
 * @param originY Bottom edge of row 0.
 * @param cellWidth Cell width.
 * @param cellHeight Cell height.
 * @param target Optional destination of at least 4 elements.
 * @returns `target`, filled.
 */
export function getGPUPolygonRasterizationExtentValues(
  originX: number,
  originY: number,
  cellWidth: number,
  cellHeight: number,
  target: Float32Array = new Float32Array(GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH)
): Float32Array {
  if (target.length < GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH) {
    throw new Error(
      `Polygon rasterization extent target must hold ${GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH} elements`
    );
  }
  target[0] = originX;
  target[1] = originY;
  target[2] = cellWidth;
  target[3] = cellHeight;
  return target;
}

/** WGSL helpers shared by the rasterization and join kernels. Requires an `extent` f32 binding. @internal */
export const POLYGON_RASTER_EXTENT_WGSL = /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
struct RasterExtent { origin: vec2<f32>, cellSize: vec2<f32>, valid: bool };
fn getRasterExtent() -> RasterExtent {
  let origin = vec2<f32>(extent[extentOffset], extent[extentOffset + 1u]);
  let cellSize = vec2<f32>(extent[extentOffset + 2u], extent[extentOffset + 3u]);
  let valid = isFiniteValue(origin.x) && isFiniteValue(origin.y) &&
    isFiniteValue(cellSize.x) && isFiniteValue(cellSize.y) && cellSize.x > 0.0 && cellSize.y > 0.0;
  return RasterExtent(origin, cellSize, valid);
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
