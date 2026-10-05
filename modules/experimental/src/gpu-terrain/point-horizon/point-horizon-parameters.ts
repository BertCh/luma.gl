// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 values read from `GPUPointHorizonProfileProps.settings`. */
export const GPU_POINT_HORIZON_PARAMETER_LENGTH = 8;

/** Number of float32 values read from `GPUPointHorizonVisibilityProps.settings`. */
export const GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH = 12;

/** Spherical earth radius in meters used for Web Mercator great-circle distances. */
export const GPU_POINT_HORIZON_EARTH_RADIUS = 6371008.8;

/**
 * Raster projection model of the point-horizon recipes (topology: baked into WGSL).
 *
 * - `'planar'`: the raster is in projected meters (a local metric CRS such as UTM or a national
 *   grid). Rays are straight lines, a pixel is `cellSize = [x, y]` meters, distances are planar
 *   meters. Per-frame settings: `cellSize`.
 * - `'web-mercator'`: the raster is a window of the Web Mercator world pixel grid (EPSG:3857, rows
 *   increase to the south). Rays follow the great circle on a sphere of radius
 *   `GPU_POINT_HORIZON_EARTH_RADIUS` (6371008.8 m), approximated piecewise: positions are linear
 *   in pixels between breakpoints whose spacing keeps the chord sagitta below
 *   `segmentTolerance` of the distance (the model of mt-image's fast horizon). Distances are
 *   spherical arc meters; the true ground size of a pixel varies with `cos(latitude)` and is
 *   handled by the projection, so `cellSize` only drives the sample spacing rule. Per-frame
 *   settings: `worldPixelSize` (pixels per 360 degrees, for example `512 * 2^zoom`) and the world
 *   pixel row of the window's top edge.
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
 */
export type GPUPointHorizonHeightReference = 'ground' | 'absolute';

/**
 * Sample traversal. `'march'` evaluates every sample of the distance lattice. `'pyramid'` (default)
 * omits samples that provably cannot raise the running maximum using a min-max mip pyramid
 * (Tevs, Ihrke and Seidel 2008 "Maximum mipmaps for fast, accurate, and scalable dynamic height
 * field rendering"; Dick et al. 2009 "GPU ray-casting for scalable terrain rendering") and
 * returns results bit-identical to `'march'`.
 */
export type GPUPointHorizonTraversal = 'march' | 'pyramid';

/**
 * Per-frame settings shared by both recipes for a planar raster.
 *
 * Cell-size model: projected meters per pixel, `cellSize = [x, y]`, both finite and positive.
 * Curvature convention: the earth drops terrain at ground distance `d` by `c * d^2` with
 * `c = (1 - k) / (2 * R)`, `R = 6371008.8` m and refraction coefficient `k`. mt-image and
 * geodetic practice use `k = 0.13` (`getGPUTerrainCurvatureCoefficient()`); GDAL's
 * `gdal_viewshed -cc 0.85714` is `cc = 1 - k` with `k = 1 / 7`, i.e.
 * `getGPUTerrainCurvatureCoefficient(1 / 7)`. The default here is `0` (flat earth).
 */
export type GPUPointHorizonPlanarSettings = {
  /** Drop coefficient `c` in 1/meters. Defaults to 0. */
  curvatureCoefficient?: number;
  /** Ray length cap in meters; `<= 0` or absent means the topology `maximumDistance`. */
  maximumDistance?: number;
  /** `[x, y]` projected meters per pixel. */
  cellSize: readonly [number, number];
};

/**
 * Per-frame settings shared by both recipes for a Web Mercator window.
 *
 * Cell-size model: Web Mercator world pixels, `worldPixelSize` pixels per 360 degrees. Distances
 * are spherical arc meters on `R = 6371008.8` m; see {@link GPUPointHorizonPlanarSettings} for the
 * curvature convention (the same `c = (1 - k) / (2R)`).
 */
export type GPUPointHorizonMercatorSettings = {
  /** Drop coefficient `c` in 1/meters. Defaults to 0. */
  curvatureCoefficient?: number;
  /** Ray length cap in meters; `<= 0` or absent means the topology `maximumDistance`. */
  maximumDistance?: number;
  /** World size in pixels (pixels per 360 degrees), for example `512 * 2^zoom`. */
  worldPixelSize: number;
  /** World pixel row of the TOP EDGE of raster row 0 (pixel centers are at `originY + row + 0.5`). */
  originY: number;
};

/** CPU-side description packed by {@link getGPUPointHorizonParameterValues}. */
export type GPUPointHorizonSettings =
  | GPUPointHorizonPlanarSettings
  | GPUPointHorizonMercatorSettings;

/** Visibility tolerances added to {@link GPUPointHorizonSettings} for `GPUPointHorizonVisibility`. */
export type GPUPointHorizonVisibilityTolerances = {
  /** Angular tolerance in degrees of the occlusion test. Defaults to 0.02. */
  toleranceDegrees?: number;
  /** Vertical standard error in meters, widening the tolerance by `sigmaZ / d` radians. Defaults to 5. */
  sigmaZ?: number;
  /** Tolerance in degrees of the skyline test. Defaults to 0.05. */
  skylineToleranceDegrees?: number;
  /** Meters before the target where the occlusion test stops. Defaults to 150. */
  targetIgnoreDistance?: number;
  /** Fraction of the target distance added to the ignore stretch. Defaults to 0. */
  targetIgnoreFraction?: number;
};

/** CPU-side description packed by {@link getGPUPointHorizonVisibilityParameterValues}. */
export type GPUPointHorizonVisibilitySettings = GPUPointHorizonSettings &
  GPUPointHorizonVisibilityTolerances;

function packModelSettings(settings: GPUPointHorizonSettings, target: Float32Array): void {
  if ('worldPixelSize' in settings) {
    const {worldPixelSize, originY} = settings;
    if (!(Number.isFinite(worldPixelSize) && worldPixelSize > 0) || !Number.isFinite(originY)) {
      throw new Error('Point horizon worldPixelSize must be positive and originY finite');
    }
    // Mercator ordinate (radians) of the top edge of row 0, in float64 before the float32 pack.
    const mercatorTop = Math.PI * (1 - (2 * originY) / worldPixelSize);
    target[1] = worldPixelSize;
    target[2] = mercatorTop;
  } else {
    if (!settings.cellSize.every(size => Number.isFinite(size) && size > 0)) {
      throw new Error('Point horizon cell size must be finite and positive');
    }
    target[1] = settings.cellSize[0];
    target[2] = settings.cellSize[1];
  }
  target[0] = settings.curvatureCoefficient ?? 0;
  target[3] = settings.maximumDistance ?? 0;
  // Slot 4 is the opaque-zero guard: the WGSL XORs it into f32 bit patterns so the compiler cannot
  // fold the hi/lo height subtraction or fuse the sample arithmetic. It must always be exactly 0.
  target[4] = 0;
}

/**
 * Packs profile settings into
 * `[curvatureCoefficient, cellSizeX | worldPixelSize, cellSizeY | mercatorTop, maximumDistance, 0, 0, 0, 0]`
 * (slot 4 is the always-zero opaque guard).
 *
 * @throws If the cell size or world size is invalid, or `target` is too short.
 */
export function getGPUPointHorizonParameterValues(
  settings: GPUPointHorizonSettings,
  target: Float32Array = new Float32Array(GPU_POINT_HORIZON_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_POINT_HORIZON_PARAMETER_LENGTH) {
    throw new Error('Point horizon settings target must hold 8 values');
  }
  target.fill(0, 0, GPU_POINT_HORIZON_PARAMETER_LENGTH);
  packModelSettings(settings, target);
  return target;
}

/**
 * Packs visibility settings into
 * `[curvatureCoefficient, cellSizeX | worldPixelSize, cellSizeY | mercatorTop, maximumDistance, 0,
 * toleranceDegrees, sigmaZ, skylineToleranceDegrees, targetIgnoreDistance, targetIgnoreFraction, 0, 0]`.
 *
 * @throws If the cell size or world size is invalid, or `target` is too short.
 */
export function getGPUPointHorizonVisibilityParameterValues(
  settings: GPUPointHorizonVisibilitySettings,
  target: Float32Array = new Float32Array(GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH) {
    throw new Error('Point horizon visibility settings target must hold 12 values');
  }
  target.fill(0, 0, GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH);
  packModelSettings(settings, target);
  target[5] = settings.toleranceDegrees ?? 0.02;
  target[6] = settings.sigmaZ ?? 5;
  target[7] = settings.skylineToleranceDegrees ?? 0.05;
  target[8] = settings.targetIgnoreDistance ?? 150;
  target[9] = settings.targetIgnoreFraction ?? 0;
  return target;
}
