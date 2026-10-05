// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUTerrainCellSizeMode} from '../terrain-analysis/gpu-terrain-derivatives';

/** Default compile-time loop bound, in pixels per axis, of metric disc searches. @internal */
export const DEFAULT_TERRAIN_FEATURES_MAXIMUM_RADIUS_PIXELS = 16;

/** Largest accepted `maximumRadiusPixels`; the disc loop visits up to `(2 * value + 1)^2` pixels. @internal */
export const MAXIMUM_TERRAIN_FEATURES_RADIUS_PIXELS_LIMIT = 64;

/** Throws unless `mode` is a known {@link GPUTerrainCellSizeMode}. @internal */
export function validateTerrainFeaturesCellSizeMode(
  id: string,
  mode: GPUTerrainCellSizeMode
): GPUTerrainCellSizeMode {
  if (mode !== 'uniform' && mode !== 'web-mercator' && mode !== 'geographic') {
    throw new Error(`${id} cellSizeMode must be 'uniform', 'web-mercator' or 'geographic'`);
  }
  return mode;
}

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
 * WGSL constants shared by the terrain feature kernels: `PI`, `DEGREES_TO_RADIANS`,
 * `METERS_PER_DEGREE`, the canonical NaN / infinity bit patterns, and `getNan(seed)` / `getInfinity(seed)`. @internal
 */
export const TERRAIN_FEATURES_WGSL_CONSTANTS = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
const METERS_PER_DEGREE: f32 = 111319.49079327357;
const NAN_BITS: u32 = 0x7fc00000u;
const INFINITY_BITS: u32 = 0x7f800000u;
// WGSL rejects constant-evaluated NaN and infinity; a runtime zero (seed & 0u) defers evaluation.
fn getNan(seed: u32) -> f32 { return bitcast<f32>(NAN_BITS | (seed & 0u)); }
fn getInfinity(seed: u32) -> f32 { return bitcast<f32>(INFINITY_BITS | (seed & 0u)); }`;

/** Settings indices that locate the cell size and row edges for {@link getTerrainFeaturesCellSizeWGSL}. @internal */
export type TerrainFeaturesCellSizeIndices = {
  /** Index of `cellSizeX`; `cellSizeY` follows it. */
  cellSizeIndex: number;
  /** Index of `northEdge`. */
  northEdgeIndex: number;
  /** Index of `southEdge`. */
  southEdgeIndex: number;
};

/**
 * Generates `fn getGroundCellSize(row: u32) -> vec2<f32>` returning ground metres per pixel.
 *
 * Mirrors the per-row model of `GPUTerrainDerivatives`: `uniform` returns the settings cell size,
 * `web-mercator` divides equatorial Web Mercator metres by `cosh(PI * (1 - 2y))` at the row's
 * normalized-y centre (`northEdge` and `southEdge` are normalized y in `[0, 1]`), and `geographic`
 * scales degrees by metres per degree with `cos(latitude)` on x (edges are latitude degrees). The
 * kernel must declare `WIDTH`, `HEIGHT`, and include {@link TERRAIN_FEATURES_WGSL_CONSTANTS}.
 *
 * @internal
 */
export function getTerrainFeaturesCellSizeWGSL(
  mode: GPUTerrainCellSizeMode,
  indices: TerrainFeaturesCellSizeIndices
): string {
  const cellSource = `let cellSize = vec2<f32>(settings[settingsOffset + ${indices.cellSizeIndex}u], settings[settingsOffset + ${indices.cellSizeIndex + 1}u]);`;
  const edgeSource = `let rowFraction = (f32(row) + 0.5) / f32(HEIGHT);
  let edge = mix(settings[settingsOffset + ${indices.northEdgeIndex}u], settings[settingsOffset + ${indices.southEdgeIndex}u], rowFraction);`;
  let groundSource: string;
  if (mode === 'uniform') {
    groundSource = 'return cellSize;';
  } else if (mode === 'web-mercator') {
    groundSource = `${edgeSource}
  // cos(latitude) = 1 / cosh(PI * (1 - 2y)) for normalized Web Mercator y.
  return cellSize / cosh(PI * (1.0 - 2.0 * edge));`;
  } else {
    groundSource = `${edgeSource}
  return cellSize * METERS_PER_DEGREE * vec2<f32>(cos(edge * DEGREES_TO_RADIANS), 1.0);`;
  }
  return `fn getGroundCellSize(row: u32) -> vec2<f32> {
  ${cellSource}
  ${groundSource}
}`;
}

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
