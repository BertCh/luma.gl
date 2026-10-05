// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUTerrainHorizonDirection} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {
  getTerrainSweepLineGeometry,
  getTerrainSweepMinorOffset,
  TERRAIN_SWEEP_SLOPE_SCALE,
  type TerrainSweepLineGeometry
} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import type {GPUTerrainCellSizeMode} from '../../../src/gpu-terrain/index';
import {getTerrainIlluminationGroundCellSize} from '../../../src/gpu-terrain/terrain-illumination/terrain-illumination-utils';

/** Inputs of the float64 sweep oracles. */
export type SweepOracleOptions = {
  width: number;
  height: number;
  /** Elevation per pixel; NaN marks nodata. */
  elevation: Float32Array;
  /** Sector count; ignored when `directions` is given. */
  directionCount: number;
  /** Explicit pixel-space directions, overriding the sector directions. */
  directions?: readonly (readonly [number, number])[];
  /** Radius window in pixels. Defaults to the full diagonal. */
  maximumRadius?: number;
  cellSize: [number, number];
  zFactor?: number;
  curvatureCoefficient?: number;
  maximumDistance?: number;
  northEdge?: number;
  southEdge?: number;
  cellSizeMode?: GPUTerrainCellSizeMode;
  rowDirection?: 'south' | 'north';
  /** Also compute the nadir pass (zFactor negated) into `nadirHorizon` and `negativeOpenness`. */
  nadir?: boolean;
};

/** Result shape of the sweep oracles, matching `computeTerrainHorizon`. */
export type SweepOracleResult = {
  /** Pixel-major horizon angles in degrees, NaN at invalid centres. */
  horizon: number[];
  skyViewFactor: number[];
  positiveOpenness: number[];
  validity: number[];
  /** Nadir-pass angles when requested. */
  nadirHorizon: number[];
  /** `90 - mean(nadir)` when requested. */
  negativeOpenness: number[];
  /** Hull walk instrumentation (hull oracle only). */
  statistics: {
    /** Pixels processed (valid centres, summed over sectors). */
    pixelCount: number;
    /** Steps of the unbounded tangent walk. */
    walkSteps: number;
    /** Steps of the windowed queries (chain steps plus restarts). */
    windowSteps: number;
    /** Windowed queries started. */
    windowQueries: number;
  };
};

/** Resolved per-direction inputs. */
type SweepDirection = {
  geometry: TerrainSweepLineGeometry;
};

function getDirections(options: SweepOracleOptions): readonly (readonly [number, number])[] {
  return (
    options.directions ??
    Array.from({length: options.directionCount}, (_, sector) =>
      getGPUTerrainHorizonDirection(sector, options.directionCount, options.rowDirection ?? 'south')
    )
  );
}

/** Ground step length of one major step on `row`, float64. */
function getStepLength(
  options: SweepOracleOptions,
  geometry: TerrainSweepLineGeometry,
  row: number
): number {
  const ground = getTerrainIlluminationGroundCellSize(
    options.cellSizeMode ?? 'uniform',
    options.cellSize,
    options.northEdge ?? 0,
    options.southEdge ?? 0,
    row,
    options.height
  );
  const slope = geometry.slopeFixed / TERRAIN_SWEEP_SLOPE_SCALE;
  return geometry.xMajor
    ? Math.hypot(ground[0], slope * ground[1])
    : Math.hypot(slope * ground[0], ground[1]);
}

/** Pixel on line `m` at major index `a`, or -1 outside the grid. */
function getLinePixel(
  options: SweepOracleOptions,
  geometry: TerrainSweepLineGeometry,
  line: number,
  major: number
): number {
  const majorExtent = geometry.xMajor ? options.width : options.height;
  const minorExtent = geometry.xMajor ? options.height : options.width;
  if (major < 0 || major >= majorExtent) {
    return -1;
  }
  const minor = line + getTerrainSweepMinorOffset(geometry.slopeFixed, major);
  if (minor < 0 || minor >= minorExtent) {
    return -1;
  }
  return geometry.xMajor ? minor * options.width + major : major * options.width + minor;
}

type SweepContext = {
  options: SweepOracleOptions;
  geometry: TerrainSweepLineGeometry;
  z: number;
  radius: number;
  maximumDistance: number;
  curvature: number;
  referenceStep: number;
};

function createContext(
  options: SweepOracleOptions,
  geometry: TerrainSweepLineGeometry,
  zSign: 1 | -1
): SweepContext {
  const uniform = (options.cellSizeMode ?? 'uniform') === 'uniform';
  const referenceStep = getStepLength(
    options,
    geometry,
    uniform ? 0 : Math.floor(options.height / 2)
  );
  const curvature = options.curvatureCoefficient ?? 0;
  return {
    options,
    geometry,
    z: zSign * (options.zFactor ?? 1),
    radius: Math.fround(
      options.maximumRadius ?? Math.ceil(Math.hypot(options.width, options.height))
    ),
    maximumDistance: options.maximumDistance ?? 0,
    curvature,
    referenceStep
  };
}

/** Decision metric: `z * (h_r - h_p) - c * Lref^2 * k^2`, identical in brute force and hull. */
function getRise(context: SweepContext, elevationR: number, elevationP: number, k: number): number {
  return (
    context.z * (elevationR - elevationP) -
    context.curvature * context.referenceStep * context.referenceStep * (k * k)
  );
}

function isInsideWindow(context: SweepContext, k: number, pixelStep: number): boolean {
  return (
    Math.fround(Math.fround(k) * context.geometry.stepPixelLength) <= context.radius &&
    (context.maximumDistance <= 0 || k * pixelStep <= context.maximumDistance)
  );
}

function getAngle(
  context: SweepContext,
  elevationB: number,
  elevationP: number,
  k: number,
  pixelStep: number
): number {
  const distance = k * pixelStep;
  const rise = context.z * (elevationB - elevationP) - context.curvature * distance * distance;
  return (Math.atan2(rise, distance) * 180) / Math.PI;
}

function createResult(options: SweepOracleOptions, directionCount: number): SweepOracleResult {
  const pixelCount = options.width * options.height;
  return {
    horizon: new Array(pixelCount * directionCount).fill(NaN),
    skyViewFactor: new Array(pixelCount).fill(NaN),
    positiveOpenness: new Array(pixelCount).fill(NaN),
    validity: new Array(pixelCount).fill(0),
    nadirHorizon: options.nadir ? new Array(pixelCount * directionCount).fill(NaN) : [],
    negativeOpenness: options.nadir ? new Array(pixelCount).fill(NaN) : [],
    statistics: {pixelCount: 0, walkSteps: 0, windowSteps: 0, windowQueries: 0}
  };
}

function accumulate(
  options: SweepOracleOptions,
  solve: (direction: SweepDirection, zSign: 1 | -1, sector: number) => Float64Array
): SweepOracleResult {
  const directions = getDirections(options);
  const directionCount = directions.length;
  const result = createResult(options, directionCount);
  const pixelCount = options.width * options.height;
  const sineSum = new Float64Array(pixelCount);
  const angleSum = new Float64Array(pixelCount);
  const nadirSum = new Float64Array(pixelCount);
  for (let sector = 0; sector < directionCount; sector++) {
    const geometry = getTerrainSweepLineGeometry(directions[sector], options.width, options.height);
    const zenith = solve({geometry}, 1, sector);
    const nadir = options.nadir ? solve({geometry}, -1, sector) : undefined;
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      result.horizon[pixel * directionCount + sector] = zenith[pixel];
      sineSum[pixel] += Math.sin((Math.max(zenith[pixel], 0) * Math.PI) / 180);
      angleSum[pixel] += zenith[pixel];
      if (nadir) {
        result.nadirHorizon[pixel * directionCount + sector] = nadir[pixel];
        nadirSum[pixel] += nadir[pixel];
      }
    }
  }
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    if (!Number.isFinite(options.elevation[pixel])) {
      continue;
    }
    result.skyViewFactor[pixel] = 1 - sineSum[pixel] / directionCount;
    result.positiveOpenness[pixel] = 90 - angleSum[pixel] / directionCount;
    if (options.nadir) {
      result.negativeOpenness[pixel] = 90 - nadirSum[pixel] / directionCount;
    }
    result.validity[pixel] = 1;
  }
  return result;
}

/**
 * Float64 brute force: for every valid pixel and sector, walk k = 1.. along the same digital line,
 * stop at the grid edge, skip invalid samples, apply the radius and distance windows, and take the
 * argmax of the `Lref` rise ratio (nearest wins ties), reporting the angle with the pixel's own
 * step length.
 */
export function computeTerrainHorizonSweepBruteForce(
  options: SweepOracleOptions
): SweepOracleResult {
  const {width, height, elevation} = options;
  return accumulate(options, ({geometry}, zSign) => {
    const context = createContext(options, geometry, zSign);
    const angles = new Float64Array(width * height).fill(NaN);
    const majorExtent = geometry.xMajor ? width : height;
    for (let row = 0; row < height; row++) {
      const pixelStep = getStepLength(options, geometry, row);
      for (let column = 0; column < width; column++) {
        const pixel = row * width + column;
        if (!Number.isFinite(elevation[pixel])) {
          continue;
        }
        const major = geometry.xMajor ? column : row;
        const minor = geometry.xMajor ? row : column;
        const line = minor - getTerrainSweepMinorOffset(geometry.slopeFixed, major);
        let best = -1;
        let bestK = 0;
        for (let k = 1; ; k++) {
          const other = getLinePixel(options, geometry, line, major + geometry.travelSign * k);
          if (other < 0 || major + geometry.travelSign * k >= majorExtent) {
            break;
          }
          if (!isInsideWindow(context, k, pixelStep)) {
            break;
          }
          if (!Number.isFinite(elevation[other])) {
            continue;
          }
          if (
            best < 0 ||
            getRise(context, elevation[other], elevation[pixel], k) * bestK >
              getRise(context, elevation[best], elevation[pixel], bestK) * k
          ) {
            best = other;
            bestK = k;
          }
        }
        angles[pixel] =
          best < 0 ? 0 : getAngle(context, elevation[best], elevation[pixel], bestK, pixelStep);
      }
    }
    return angles;
  });
}

/**
 * Float64 port of the hull-pointer algorithm exactly as the WGSL kernel runs it: lines processed
 * from the ahead end, `hull[]` pointers, unbounded tangent walk, windowed query. Returns the same
 * shape as the brute force plus walk statistics.
 */
export function computeTerrainHorizonSweepHull(options: SweepOracleOptions): SweepOracleResult {
  const {width, height, elevation} = options;
  const statistics = {pixelCount: 0, walkSteps: 0, windowSteps: 0, windowQueries: 0};
  const result = accumulate(options, ({geometry}, zSign) => {
    const context = createContext(options, geometry, zSign);
    const angles = new Float64Array(width * height).fill(NaN);
    const hull = new Int32Array(width * height).fill(-1);
    const majorExtent = geometry.xMajor ? width : height;
    const getMajor = (pixel: number) =>
      geometry.xMajor ? pixel % width : Math.floor(pixel / width);
    for (let index = 0; index < geometry.lineCount; index++) {
      const line = geometry.firstLine + index;
      let previous = -1;
      for (let step = 0; step < majorExtent; step++) {
        const major = geometry.travelSign > 0 ? majorExtent - 1 - step : step;
        const pixel = getLinePixel(options, geometry, line, major);
        if (pixel < 0) {
          continue;
        }
        const next = previous;
        previous = pixel;
        const firstAhead = next < 0 ? -1 : Number.isFinite(elevation[next]) ? next : hull[next];
        if (!Number.isFinite(elevation[pixel])) {
          hull[pixel] = firstAhead;
          continue;
        }
        statistics.pixelCount++;
        const k = (other: number) => Math.abs(getMajor(other) - major);
        const growing = (r: number, c: number) =>
          getRise(context, elevation[r], elevation[pixel], k(r)) * k(c) >
          getRise(context, elevation[c], elevation[pixel], k(c)) * k(r);
        let tangent = firstAhead;
        if (tangent >= 0) {
          let chain = hull[tangent];
          while (chain >= 0 && growing(chain, tangent)) {
            statistics.walkSteps++;
            tangent = chain;
            chain = hull[chain];
          }
        }
        hull[pixel] = tangent;
        const pixelStep = getStepLength(options, geometry, Math.floor(pixel / width));
        const inside = (other: number) => isInsideWindow(context, k(other), pixelStep);
        let best = -1;
        if (firstAhead >= 0 && inside(firstAhead)) {
          if (inside(tangent)) {
            best = tangent;
          } else {
            statistics.windowQueries++;
            let t = firstAhead;
            while (t >= 0 && inside(t)) {
              let c = t;
              let r = hull[c];
              while (r >= 0 && inside(r) && growing(r, c)) {
                statistics.windowSteps++;
                c = r;
                r = hull[r];
              }
              statistics.windowSteps++;
              if (best < 0 || growing(c, best)) {
                best = c;
              }
              if (r < 0 || inside(r)) {
                break;
              }
              const after = getLinePixel(
                options,
                geometry,
                line,
                getMajor(c) + geometry.travelSign
              );
              t = after < 0 ? -1 : Number.isFinite(elevation[after]) ? after : hull[after];
            }
          }
        }
        angles[pixel] =
          best < 0 ? 0 : getAngle(context, elevation[best], elevation[pixel], k(best), pixelStep);
      }
    }
    return angles;
  });
  result.statistics = statistics;
  return result;
}
