// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Default compile-time loop bound, in pixels per axis, of metric disc searches. @internal */
const DEFAULT_TERRAIN_FEATURES_MAXIMUM_RADIUS_PIXELS = 16;

/** Largest accepted `maximumRadiusPixels`; the disc loop visits up to `(2 * value + 1)^2` pixels. @internal */
const MAXIMUM_TERRAIN_FEATURES_RADIUS_PIXELS_LIMIT = 64;

/** Throws unless the loop bound is an integer in `[1, 64]`; returns it (default 16). @internal */
export function validateTerrainFeaturesMaximumRadiusPixels(
  id: string,
  maximumRadiusPixels: number | undefined
): number {
  const value = maximumRadiusPixels ?? DEFAULT_TERRAIN_FEATURES_MAXIMUM_RADIUS_PIXELS;
  if (
    !Number.isInteger(value) ||
    value < 1 ||
    value > MAXIMUM_TERRAIN_FEATURES_RADIUS_PIXELS_LIMIT
  ) {
    throw new Error(
      `${id} maximumRadiusPixels must be an integer in [1, ${MAXIMUM_TERRAIN_FEATURES_RADIUS_PIXELS_LIMIT}]`
    );
  }
  return value;
}

/** Throws unless `cellSize` is two finite positive numbers. @internal */
export function validateTerrainFeaturesCellSize(
  label: string,
  cellSize: readonly [number, number]
): void {
  if (!cellSize.every(size => Number.isFinite(size) && size > 0)) {
    throw new Error(`${label} cell size must be finite and positive`);
  }
}

/** Throws unless `value` is finite and at least `minimum` (or strictly above when `exclusive`). @internal */
export function validateTerrainFeaturesScalar(
  label: string,
  name: string,
  value: number,
  minimum: number,
  exclusive: boolean = false
): void {
  if (!Number.isFinite(value) || (exclusive ? value <= minimum : value < minimum)) {
    throw new Error(
      `${label} ${name} must be finite and ${exclusive ? 'greater than' : 'at least'} ${minimum}`
    );
  }
}

/**
 * WGSL constants shared by the terrain feature kernels: `PI`, the canonical NaN / infinity bit patterns, and `getNan(seed)` / `getInfinity(seed)`. @internal
 */
export const TERRAIN_FEATURES_WGSL_CONSTANTS = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const NAN_BITS: u32 = 0x7fc00000u;
const INFINITY_BITS: u32 = 0x7f800000u;
// WGSL rejects constant-evaluated NaN and infinity; a runtime zero (seed & 0u) defers evaluation.
fn getNan(seed: u32) -> f32 { return bitcast<f32>(NAN_BITS | (seed & 0u)); }
fn getInfinity(seed: u32) -> f32 { return bitcast<f32>(INFINITY_BITS | (seed & 0u)); }`;

/**
 * WGSL for metric disc and ring tests. Requires the kernel to define `MAXIMUM_RADIUS_PIXELS: i32`.
 *
 * - `isInDisc(pixel, centre, cell, radiusSquared)`: `((q - centre) * cell)` squared length
 *   `<= radiusSquared`, with one correctly rounded multiplication per axis.
 * - `isOnDiscRing(...)`: a disc pixel with an 8-neighbour outside the disc. The ring is geometric:
 *   grid bounds and nodata play no part, so every 8-connected path leaving the disc crosses it.
 * - `getEffectiveRadius(requested, cell)` clamps to the largest radius whose disc fits
 *   `MAXIMUM_RADIUS_PIXELS` on both axes; `isRadiusClamped` reports whether it clamped.
 * - `getDiscExtent(radius, cellSize)`: conservative pixel half-extent per axis, at most the bound.
 *
 * `dx * dx + dy * dy` may be fused into an FMA by some GPU compilers, so a pixel whose squared
 * distance lies within one ULP of `radiusSquared` can fall on either side of the boundary.
 *
 * @internal
 */
export const TERRAIN_FEATURES_DISC_WGSL = /* wgsl */ `
fn isInDisc(pixel: vec2<i32>, centre: vec2<f32>, cell: vec2<f32>, radiusSquared: f32) -> bool {
  let scaledX = (f32(pixel.x) - centre.x) * cell.x;
  let scaledY = (f32(pixel.y) - centre.y) * cell.y;
  return scaledX * scaledX + scaledY * scaledY <= radiusSquared;
}
fn isOnDiscRing(pixel: vec2<i32>, centre: vec2<f32>, cell: vec2<f32>, radiusSquared: f32) -> bool {
  for (var neighborY = -1; neighborY <= 1; neighborY++) {
    for (var neighborX = -1; neighborX <= 1; neighborX++) {
      if ((neighborX != 0 || neighborY != 0) &&
          !isInDisc(pixel + vec2<i32>(neighborX, neighborY), centre, cell, radiusSquared)) {
        return true;
      }
    }
  }
  return false;
}
fn getMaximumRadius(cell: vec2<f32>) -> f32 {
  return min(f32(MAXIMUM_RADIUS_PIXELS) * cell.x, f32(MAXIMUM_RADIUS_PIXELS) * cell.y);
}
fn isRadiusClamped(requested: f32, cell: vec2<f32>) -> bool {
  return requested > getMaximumRadius(cell);
}
fn getEffectiveRadius(requested: f32, cell: vec2<f32>) -> f32 {
  return min(requested, getMaximumRadius(cell));
}
fn getDiscExtent(radius: f32, cellSize: f32) -> i32 {
  return min(i32(floor(radius / cellSize + 0.5)) + 1, MAXIMUM_RADIUS_PIXELS);
}`;

/**
 * WGSL for disc rows by offset from the centre. Requires {@link TERRAIN_FEATURES_DISC_WGSL} and
 * `MAXIMUM_RADIUS_PIXELS: i32`.
 *
 * - `isOffsetInDisc(dx, dy, cell, radiusSquared)`: `isInDisc` for the pixel at offset `(dx, dy)`
 *   from the centre. `f32(column + dx) - f32(column)` equals `f32(dx)` exactly, so the result is
 *   bit-identical to `isInDisc` on absolute coordinates.
 * - `getDiscRowHalfWidth(dy, ...)`: largest `dx >= 0` with `(dx, dy)` in the disc, or -1 when row
 *   `dy` holds no disc pixel.
 * - `getDiscColumnHalfHeight(...)`: largest `dy >= 0` with `(0, dy)` in the disc.
 *
 * Why these are exact: `isInDisc` depends on `(dx, dy)` only through `|dx|` and `|dy|`, and it is
 * monotone non-increasing in each. `f32(|dx|) * cell.x` is a correctly rounded product of a growing
 * operand, squaring and adding rounded non-negative values keep the order (rounding is monotone),
 * and the comparison with `radiusSquared` is monotone. So a row's disc pixels are exactly
 * `|dx| <= halfWidth`, the widest row is `dy = 0` and the tallest column is `dx = 0`. The `sqrt`
 * only seeds the search; two loops correct it with the real predicate. The disc never leaves the
 * `getDiscExtent` box (radius is clamped to the bound), so no pixel beyond `MAXIMUM_RADIUS_PIXELS`
 * is considered.
 *
 * @internal
 */
export const TERRAIN_FEATURES_DISC_ROW_WGSL = /* wgsl */ `
fn isOffsetInDisc(dx: i32, dy: i32, cell: vec2<f32>, radiusSquared: f32) -> bool {
  return isInDisc(vec2<i32>(dx, dy), vec2<f32>(0.0, 0.0), cell, radiusSquared);
}
fn getDiscRowHalfWidth(dy: i32, cell: vec2<f32>, radiusSquared: f32) -> i32 {
  let scaledY = f32(dy) * cell.y;
  let remaining = max(radiusSquared - scaledY * scaledY, 0.0);
  var halfWidth = clamp(i32(floor(sqrt(remaining) / cell.x)), 0, MAXIMUM_RADIUS_PIXELS + 1);
  while (halfWidth > 0 && !isOffsetInDisc(halfWidth, dy, cell, radiusSquared)) { halfWidth--; }
  while (halfWidth <= MAXIMUM_RADIUS_PIXELS && isOffsetInDisc(halfWidth + 1, dy, cell, radiusSquared)) { halfWidth++; }
  if (!isOffsetInDisc(halfWidth, dy, cell, radiusSquared)) { return -1; }
  return halfWidth;
}
fn getDiscColumnHalfHeight(cell: vec2<f32>, radiusSquared: f32) -> i32 {
  var halfHeight = clamp(i32(floor(sqrt(radiusSquared) / cell.y)), 0, MAXIMUM_RADIUS_PIXELS + 1);
  while (halfHeight > 0 && !isOffsetInDisc(0, halfHeight, cell, radiusSquared)) { halfHeight--; }
  while (halfHeight <= MAXIMUM_RADIUS_PIXELS && isOffsetInDisc(0, halfHeight + 1, cell, radiusSquared)) { halfHeight++; }
  return halfHeight;
}`;
