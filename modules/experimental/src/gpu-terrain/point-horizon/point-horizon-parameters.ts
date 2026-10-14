// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '../../gpu-raster/index';
import type {GPUTerrainHeightReference} from '../../analysis-core/contracts/vertical-datum';
import {
  GPU_EARTH_MEAN_RADIUS_METERS,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from '../../analysis-core/contracts/earth';
import {getRasterGridGeoreference} from '../../analysis-core/raster-grid';
import {validateParameterTarget} from '../../gpu-raster/raster-band-utils';

/** Number of float32 values read from `GPUPointHorizonProfileProps.parameters`. */
export const GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH = 8;

/** Number of float32 values read from `GPUPointHorizonVisibilityProps.parameters`. */
export const GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH = 12;

/** Settings slot of the projection tag in the profile layout. @internal */
export const POINT_HORIZON_PROFILE_PROJECTION_SLOT = 7;

/** Settings slot of the projection tag in the visibility layout. @internal */
export const POINT_HORIZON_VISIBILITY_PROJECTION_SLOT = 11;

/**
 * Projection tags written by the packers: `settings.projection` selects the tag, and the kernels reject a tag that disagrees with their compile-time
 * `projection`, so mismatched settings produce NaN / no-data instead of silently misread numbers.
 * Tag 0 (hand-written arrays) is accepted by both. @internal
 */
export const POINT_HORIZON_PROJECTION_TAGS = {planar: 1, 'web-mercator': 2} as const;

/**
 * Spherical earth radius in meters used for Web Mercator great-circle distances: the IUGG mean
 * radius `GPU_EARTH_MEAN_RADIUS_METERS` (6371008.8 m).
 */
export const GPU_POINT_HORIZON_EARTH_RADIUS = GPU_EARTH_MEAN_RADIUS_METERS;

/**
 * Raster projection model of the point-horizon contributors (topology: baked into WGSL).
 *
 * - `'planar'`: the raster is in projected meters (a local metric CRS such as UTM or a national
 *   grid). Rays are straight lines, a pixel is `cellSize = [x, y]` meters, distances are planar
 *   meters. The packer takes the cell size from the grid.
 * - `'web-mercator'`: the raster is a window of the Web Mercator world pixel grid (EPSG:3857, rows
 *   increase to the south). Rays follow the great circle on a sphere of radius
 *   `GPU_POINT_HORIZON_EARTH_RADIUS` (6371008.8 m), approximated piecewise: positions are linear
 *   in pixels between breakpoints whose spacing keeps the chord sagitta below
 *   `segmentTolerance` of the distance (the model of mt-image's fast horizon). Distances are
 *   spherical arc meters; the true ground size of a pixel varies with `cos(latitude)` and is
 *   handled by the projection, so `cellSize` only drives the sample spacing rule. The grid is in
 *   EPSG:3857 meters; the packer derives the world size (pixels per 360 degrees, for example
 *   `512 * 2^zoom`) and the Mercator ordinate of the window's top edge from it.
 */
export type GPUPointHorizonProjection = 'planar' | 'web-mercator';

/**
 * Direction in which raster rows increase. `'south'` (default) is the image convention: row 0 is
 * the northern edge. `'north'` is for rasters stored bottom-up. Only used by `'planar'`; Web
 * Mercator windows always increase to the south.
 */
export type GPUPointHorizonRowDirection = 'south' | 'north';

/**
 * Eye and target height model. `'ground'` (default): the eye is the bilinear ground at the
 * observer plus its height row value. `'absolute'`: the height value is the eye elevation itself.
 * Alias of the shared {@link GPUTerrainHeightReference}.
 */
export type GPUPointHorizonHeightReference = GPUTerrainHeightReference;

/**
 * Sample traversal. `'march'` evaluates every sample of the distance lattice. `'pyramid'` (default)
 * omits samples that provably cannot raise the running maximum using a min-max mip pyramid
 * (Tevs, Ihrke and Seidel 2008 "Maximum mipmaps for fast, accurate, and scalable dynamic height
 * field rendering"; Dick et al. 2009 "GPU ray-casting for scalable terrain rendering") and
 * returns results bit-identical to `'march'`.
 */
export type GPUPointHorizonTraversal = 'march' | 'pyramid';

/**
 * Per-frame settings shared by both contributors, packed by
 * {@link getGPUPointHorizonProfileParameterValues}. The georeference comes from the grid passed
 * next to them.
 *
 * Planar cell-size model: projected meters per pixel, taken from the grid cell size (both finite
 * and positive). Web Mercator cell-size model: the grid is a window of EPSG:3857 meters with
 * square pixels and rows increasing to the south; the world size `worldPixelSize` (pixels per 360
 * degrees) is `2 * pi * a / cellSize` and the Mercator ordinate of the top edge of row 0 is
 * `northEdge / a`, with `a = 6378137` m, which equals the former `originY` form
 * `pi * (1 - 2 * originY / worldPixelSize)`. Distances are spherical arc meters on
 * `R = 6371008.8` m.
 *
 * Curvature convention: the earth drops terrain at ground distance `d` by `c * d^2` with
 * `c = (1 - k) / (2 * R)`, `R = 6371008.8` m and refraction coefficient `k`. mt-image and
 * geodetic practice use `k = 0.13` (`getGPUTerrainCurvatureCoefficient()`); GDAL's
 * `gdal_viewshed -cc 0.85714` is `cc = 1 - k` with `k = 1 / 7`, i.e.
 * `getGPUTerrainCurvatureCoefficient(1 / 7)`. The default here is `0` (flat earth).
 */
export type GPUPointHorizonProfileSettings = {
  /**
   * Projection the packed grid words describe. Must equal the contributor's `projection`, else the
   * kernel reports NaN or no-data. Defaults to `'planar'`.
   */
  projection?: GPUPointHorizonProjection;
  /** Drop coefficient `c` in 1/meters. Defaults to 0. */
  curvatureCoefficient?: number;
  /** Ray length cap in meters; `<= 0` or absent means the topology `maximumDistance`. */
  maximumDistance?: number;
};

/** Visibility tolerances added to {@link GPUPointHorizonProfileSettings} for `GPUPointHorizonVisibility`. */
export type GPUPointHorizonVisibilityTolerances = {
  /** Angular tolerance in degrees of the occlusion test. Defaults to 0.02. */
  toleranceDegrees?: number;
  /** Vertical standard error in meters, widening the tolerance by `sigmaZ / d` radians. Defaults to 5. */
  sigmaZ?: number;
  /** Tolerance in degrees of the skyline test. Defaults to 0.05. */
  skylineToleranceDegrees?: number;
  /**
   * Meters before the target where the occlusion test stops: terrain in this last stretch of the
   * ray is never tested, so a target inside it cannot be hidden by it. Defaults to one ground
   * pixel along the ray at the target (library rule 4: derived from the cell size), which keeps
   * the target's own bilinear cell from occluding it. mt-image's fixed 150 m is the named preset
   * {@link GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE}.
   */
  targetIgnoreDistance?: number;
  /** Fraction of the target distance added to the ignore stretch. Defaults to 0. */
  targetIgnoreFraction?: number;
};

/**
 * mt-image's `classifyPeak` target ignore distance in meters, as a named preset for
 * {@link GPUPointHorizonVisibilityTolerances.targetIgnoreDistance}.
 */
export const GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE = 150;

/**
 * Packed `targetIgnoreDistance` meaning "one ground pixel along the ray at the target", the
 * default. @internal
 */
export const POINT_HORIZON_DERIVED_TARGET_IGNORE_DISTANCE = -1;

/** CPU-side description packed by {@link getGPUPointHorizonVisibilityParameterValues}. */
export type GPUPointHorizonVisibilitySettings = GPUPointHorizonProfileSettings &
  GPUPointHorizonVisibilityTolerances;

function packModelSettings(
  name: string,
  settings: GPUPointHorizonProfileSettings,
  grid: GPURasterMetadata,
  target: Float32Array,
  projectionSlot: number
): void {
  const projection = settings.projection ?? 'planar';
  if (projection !== 'planar' && projection !== 'web-mercator') {
    throw new Error(`${name} projection must be planar or web-mercator`);
  }
  target[projectionSlot] = POINT_HORIZON_PROJECTION_TAGS[projection];
  const {cellSize, northEdge} = getRasterGridGeoreference(grid);
  if (!cellSize.every(size => Number.isFinite(size) && size > 0)) {
    throw new Error(`${name} grid cell size must be finite and positive`);
  }
  if (projection === 'web-mercator') {
    const [cellSizeX, cellSizeY] = cellSize;
    if (Math.abs(cellSizeX - cellSizeY) > 1e-6 * cellSizeX || grid.affine[4] >= 0) {
      throw new Error(`${name} web-mercator grid needs square pixels with rows increasing south`);
    }
    // Pixels per 360 degrees and the Mercator ordinate (radians) of the top edge of row 0, in
    // float64 before the float32 pack. northEdge is EPSG:3857 meters.
    target[1] = (2 * Math.PI * GPU_WGS84_SEMI_MAJOR_AXIS_METERS) / cellSizeX;
    target[2] = northEdge / GPU_WGS84_SEMI_MAJOR_AXIS_METERS;
  } else {
    target[1] = cellSize[0];
    target[2] = cellSize[1];
  }
  target[0] = settings.curvatureCoefficient ?? 0;
  target[3] = settings.maximumDistance ?? 0;
  // Slot 4 is the opaque-zero guard: the WGSL XORs it into f32 bit patterns so the compiler cannot
  // fold the hi/lo height subtraction or fuse the sample arithmetic. It must always be exactly 0.
  target[4] = 0;
}

/**
 * Packs profile settings into
 * `[curvatureCoefficient, cellSizeX | worldPixelSize, cellSizeY | mercatorTop, maximumDistance, 0,
 * 0, 0, projectionTag]` (slot 4 is the always-zero opaque guard). Cell size (planar) or the world
 * size and top-edge Mercator ordinate (Web Mercator) are derived from `grid`, see
 * {@link GPUPointHorizonProfileSettings}. The projection tag (1 planar, 2 Web Mercator) comes from
 * `settings.projection`; a contributor whose `projection` differs reports NaN for every ray
 * instead of misreading the cell size.
 *
 * @throws If the grid cell size is invalid (a Web Mercator grid needs square pixels and rows
 *   increasing south), or `target` is too short.
 */
export function getGPUPointHorizonProfileParameterValues(
  settings: GPUPointHorizonProfileSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPUPointHorizonProfileParameterValues';
  validateParameterTarget(name, target, GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH);
  target.fill(0, 0, GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH);
  packModelSettings(name, settings, grid, target, POINT_HORIZON_PROFILE_PROJECTION_SLOT);
  return target;
}

/**
 * Packs visibility settings into
 * `[curvatureCoefficient, cellSizeX | worldPixelSize, cellSizeY | mercatorTop, maximumDistance, 0,
 * toleranceDegrees, sigmaZ, skylineToleranceDegrees, targetIgnoreDistance, targetIgnoreFraction, 0,
 * projectionTag]`. An omitted `targetIgnoreDistance` packs `-1`, which the kernel reads as one
 * ground pixel along the ray at the target. The georeference words come from `grid` exactly as in
 * {@link getGPUPointHorizonProfileParameterValues}, and a contributor whose `projection` differs
 * from `settings.projection` reports no-data.
 *
 * @throws If the grid cell size is invalid, or `target` is too short.
 */
export function getGPUPointHorizonVisibilityParameterValues(
  settings: GPUPointHorizonVisibilitySettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPUPointHorizonVisibilityParameterValues';
  validateParameterTarget(name, target, GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH);
  target.fill(0, 0, GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH);
  packModelSettings(name, settings, grid, target, POINT_HORIZON_VISIBILITY_PROJECTION_SLOT);
  target[5] = settings.toleranceDegrees ?? 0.02;
  target[6] = settings.sigmaZ ?? 5;
  target[7] = settings.skylineToleranceDegrees ?? 0.05;
  const ignoreDistance = settings.targetIgnoreDistance;
  if (ignoreDistance !== undefined && !(Number.isFinite(ignoreDistance) && ignoreDistance >= 0)) {
    throw new Error(`${name} targetIgnoreDistance must be finite and non-negative`);
  }
  target[8] = ignoreDistance ?? POINT_HORIZON_DERIVED_TARGET_IGNORE_DISTANCE;
  target[9] = settings.targetIgnoreFraction ?? 0;
  return target;
}
