// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUTerrainHorizonDirection} from '../../../src/map-graphs/terrain-illumination/gpu-terrain-horizon';
import {
  getTerrainIlluminationGroundCellSize,
  type GPUTerrainIlluminationCellSizeMode
} from '../../../src/map-graphs/terrain-illumination/terrain-illumination-utils';

/** Inputs of the float64 horizon oracle. */
export type HorizonOracleOptions = {
  width: number;
  height: number;
  /** Elevation per pixel; NaN marks nodata. */
  elevation: Float32Array;
  directionCount: number;
  /** Ray sample distances in pixels. */
  stepDistances: ArrayLike<number>;
  cellSize: [number, number];
  zFactor?: number;
  curvatureCoefficient?: number;
  maximumDistance?: number;
  northEdge?: number;
  southEdge?: number;
  cellSizeMode?: GPUTerrainIlluminationCellSizeMode;
  rowDirection?: 'south' | 'north';
  /** Also compute nadir horizons (z factor negated) and `negativeOpenness`. */
  computeNadir?: boolean;
  /** Also compute `anisotropicSkyViewFactor` with float64 weights. */
  anisotropy?: {azimuthDegrees: number; level: number; minimumWeight: number};
};

/** Horizon oracle result. */
export type HorizonOracleResult = {
  /** Pixel-major horizon angles in degrees. */
  horizon: number[];
  skyViewFactor: number[];
  positiveOpenness: number[];
  validity: number[];
  /** Pixel-major nadir angles in degrees; present with `computeNadir`. */
  nadirHorizon?: number[];
  /** `90 - mean(nadir)`; present with `computeNadir`. */
  negativeOpenness?: number[];
  /** Present with `anisotropy`. */
  anisotropicSkyViewFactor?: number[];
};

/**
 * RVT anisotropic sky-view weight of one sector in float64:
 * `(1 - wMin) * |cos((azimuth - main) / 2)|^level + wMin`, azimuths in degrees.
 */
export function getAnisotropicWeightOracle(
  sector: number,
  directionCount: number,
  anisotropy: {azimuthDegrees: number; level: number; minimumWeight: number}
): number {
  const halfAngle = (((sector * 360) / directionCount - anisotropy.azimuthDegrees) * Math.PI) / 360;
  const cosine = Math.abs(Math.cos(halfAngle));
  const power = cosine === 0 ? (anisotropy.level === 0 ? 1 : 0) : cosine ** anisotropy.level;
  return (1 - anisotropy.minimumWeight) * power + anisotropy.minimumWeight;
}

/** Float64 twin of `encodeGPUTerrainHorizonUnorm16`: code 0 is invalid, else 1..65535. */
export function encodeHorizonUnorm16Oracle(angleDegrees: number): number {
  if (!Number.isFinite(angleDegrees)) {
    return 0;
  }
  const clamped = Math.min(Math.max(angleDegrees, -90), 90);
  return 1 + Math.floor(clamped * Math.fround(65534 / 180) + 32767.5);
}

/** Float64 twin of `decodeGPUTerrainHorizonUnorm16`. */
export function decodeHorizonUnorm16Oracle(code: number): number {
  return code === 0 ? NaN : ((code - 1) * 180) / 65534 - 90;
}

/** Straightforward float64 horizon, sky-view factor, and openness. */
export function computeTerrainHorizon(options: HorizonOracleOptions): HorizonOracleResult {
  const {width, height, elevation, directionCount, stepDistances} = options;
  const zFactor = options.zFactor ?? 1;
  const curvature = options.curvatureCoefficient ?? 0;
  const maximumDistance = options.maximumDistance ?? 0;
  const isValid = (column: number, row: number) => Number.isFinite(elevation[row * width + column]);
  const sample = (x: number, y: number): number | null => {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = Math.min(x0 + 1, width - 1);
    const y1 = Math.min(y0 + 1, height - 1);
    if (!isValid(x0, y0) || !isValid(x1, y0) || !isValid(x0, y1) || !isValid(x1, y1)) {
      return null;
    }
    const fx = x - x0;
    const fy = y - y0;
    const at = (column: number, row: number) => elevation[row * width + column];
    const top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
    const bottom = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
    return top * (1 - fy) + bottom * fy;
  };
  const pixelCount = width * height;
  const result: HorizonOracleResult = {
    horizon: new Array(pixelCount * directionCount).fill(NaN),
    skyViewFactor: new Array(pixelCount).fill(NaN),
    positiveOpenness: new Array(pixelCount).fill(NaN),
    validity: new Array(pixelCount).fill(0)
  };
  if (options.computeNadir) {
    result.nadirHorizon = new Array(pixelCount * directionCount).fill(NaN);
    result.negativeOpenness = new Array(pixelCount).fill(NaN);
  }
  if (options.anisotropy) {
    result.anisotropicSkyViewFactor = new Array(pixelCount).fill(NaN);
  }
  for (let row = 0; row < height; row++) {
    const ground = getTerrainIlluminationGroundCellSize(
      options.cellSizeMode ?? 'uniform',
      options.cellSize,
      options.northEdge ?? 0,
      options.southEdge ?? 0,
      row,
      height
    );
    for (let column = 0; column < width; column++) {
      const pixel = row * width + column;
      if (!isValid(column, row)) {
        continue;
      }
      const centerElevation = elevation[pixel];
      let sineSum = 0;
      let angleSum = 0;
      let nadirSum = 0;
      let weightedSineSum = 0;
      let weightSum = 0;
      for (let sector = 0; sector < directionCount; sector++) {
        // The march direction is part of the recipe contract (float32, snapped axis components).
        const [dx, dy] = getGPUTerrainHorizonDirection(
          sector,
          directionCount,
          options.rowDirection ?? 'south'
        );
        const groundStep = Math.hypot(dx * ground[0], dy * ground[1]);
        let best = -Infinity;
        let bestNadir = -Infinity;
        for (const distance of Array.from(stepDistances)) {
          const x = column + dx * distance;
          const y = row + dy * distance;
          if (x < 0 || y < 0 || x > width - 1 || y > height - 1) {
            break;
          }
          const groundDistance = distance * groundStep;
          if (maximumDistance > 0 && groundDistance > maximumDistance) {
            break;
          }
          const value = sample(x, y);
          if (value === null) {
            continue;
          }
          const rise =
            zFactor * (value - centerElevation) - curvature * groundDistance * groundDistance;
          best = Math.max(best, (Math.atan2(rise, groundDistance) * 180) / Math.PI);
          const nadirRise =
            -zFactor * (value - centerElevation) - curvature * groundDistance * groundDistance;
          bestNadir = Math.max(bestNadir, (Math.atan2(nadirRise, groundDistance) * 180) / Math.PI);
        }
        const angle = best === -Infinity ? 0 : best;
        result.horizon[pixel * directionCount + sector] = angle;
        sineSum += Math.sin((Math.max(angle, 0) * Math.PI) / 180);
        angleSum += angle;
        if (options.computeNadir && result.nadirHorizon) {
          const nadirAngle = bestNadir === -Infinity ? 0 : bestNadir;
          result.nadirHorizon[pixel * directionCount + sector] = nadirAngle;
          nadirSum += nadirAngle;
        }
        if (options.anisotropy) {
          const weight = getAnisotropicWeightOracle(sector, directionCount, options.anisotropy);
          weightedSineSum += weight * Math.sin((Math.max(angle, 0) * Math.PI) / 180);
          weightSum += weight;
        }
      }
      if (result.negativeOpenness) {
        result.negativeOpenness[pixel] = 90 - nadirSum / directionCount;
      }
      if (result.anisotropicSkyViewFactor) {
        result.anisotropicSkyViewFactor[pixel] = 1 - weightedSineSum / weightSum;
      }
      result.skyViewFactor[pixel] = 1 - sineSum / directionCount;
      result.positiveOpenness[pixel] = 90 - angleSum / directionCount;
      result.validity[pixel] = 1;
    }
  }
  return result;
}

/** Inputs of the float64 shadow oracle. */
export type ShadowOracleOptions = {
  pixelCount: number;
  directionCount: number;
  horizon: ArrayLike<number>;
  azimuthDegrees: number;
  altitudeDegrees: number;
  angularRadiusDegrees: number;
  sunIntensity?: number;
  ambientIntensity?: number;
  slope?: ArrayLike<number>;
  aspect?: ArrayLike<number>;
  skyViewFactor?: ArrayLike<number>;
};

/**
 * Fraction of a disk of `radius` centred at `altitude` lying above a horizontal line at `horizon`,
 * by direct numerical integration of the chord length (independent of the closed form).
 */
export function integrateVisibleDiskFraction(
  altitude: number,
  horizon: number,
  radius: number,
  sampleCount: number = 4000
): number {
  if (radius === 0) {
    return altitude > horizon ? 1 : 0;
  }
  let visible = 0;
  let total = 0;
  for (let index = 0; index < sampleCount; index++) {
    const y = -radius + ((index + 0.5) / sampleCount) * 2 * radius;
    const chord = 2 * Math.sqrt(Math.max(radius * radius - y * y, 0));
    total += chord;
    if (altitude + y > horizon) {
      visible += chord;
    }
  }
  return visible / total;
}

/** Closed-form circular segment area fraction visible above the horizon. */
export function getVisibleDiskFraction(altitude: number, horizon: number, radius: number): number {
  if (radius === 0) {
    return altitude > horizon ? 1 : 0;
  }
  const u = Math.min(Math.max((altitude - horizon) / radius, -1), 1);
  return 1 - (Math.acos(u) - u * Math.sqrt(1 - u * u)) / Math.PI;
}

/** Float64 sun visibility and illumination. */
export function computeSolarShadow(options: ShadowOracleOptions): {
  sunVisibility: number[];
  illumination: number[];
} {
  const {pixelCount, directionCount, horizon} = options;
  const sunVisibility: number[] = [];
  const illumination: number[] = [];
  const spacing = 360 / directionCount;
  const azimuth = ((options.azimuthDegrees % 360) + 360) % 360;
  const lower = Math.min(Math.floor(azimuth / spacing), directionCount - 1);
  const upper = (lower + 1) % directionCount;
  const blend = azimuth / spacing - lower;
  const altitudeRadians = (options.altitudeDegrees * Math.PI) / 180;
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    const horizonAngle =
      horizon[pixel * directionCount + lower] * (1 - blend) +
      horizon[pixel * directionCount + upper] * blend;
    if (!Number.isFinite(horizonAngle)) {
      sunVisibility.push(NaN);
      illumination.push(NaN);
      continue;
    }
    const visibility =
      options.altitudeDegrees + options.angularRadiusDegrees > 0
        ? getVisibleDiskFraction(
            options.altitudeDegrees,
            horizonAngle,
            options.angularRadiusDegrees
          )
        : 0;
    sunVisibility.push(visibility);
    if (!options.slope || !options.aspect) {
      illumination.push(NaN);
      continue;
    }
    const slope = (options.slope[pixel] * Math.PI) / 180;
    const aspect = options.aspect[pixel];
    const cosIncidence =
      aspect < 0
        ? Math.sin(altitudeRadians)
        : Math.cos(slope) * Math.sin(altitudeRadians) +
          Math.sin(slope) *
            Math.cos(altitudeRadians) *
            Math.cos(((options.azimuthDegrees - aspect) * Math.PI) / 180);
    const svf = options.skyViewFactor ? options.skyViewFactor[pixel] : 1;
    illumination.push(
      (options.ambientIntensity ?? 0) * svf +
        (options.sunIntensity ?? 1) * visibility * Math.max(cosIncidence, 0)
    );
  }
  return {sunVisibility, illumination};
}

/** Deterministic xorshift-based random generator in `[0, 1)`. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

/** Smooth random terrain: a sum of a few sinusoids plus a Gaussian peak, in meters. */
export function createSmoothTerrain(width: number, height: number, seed: number): Float32Array {
  const random = createRandom(seed);
  const waves = Array.from({length: 4}, () => ({
    amplitude: 5 + random() * 20,
    frequencyX: (random() - 0.5) * 0.5,
    frequencyY: (random() - 0.5) * 0.5,
    phase: random() * Math.PI * 2
  }));
  const peak = {
    x: random() * width,
    y: random() * height,
    height: 40 + random() * 60
  };
  return Float32Array.from({length: width * height}, (_, index) => {
    const x = index % width;
    const y = Math.floor(index / width);
    let value = 100;
    for (const wave of waves) {
      value += wave.amplitude * Math.sin(wave.frequencyX * x + wave.frequencyY * y + wave.phase);
    }
    const distanceSquared = (x - peak.x) ** 2 + (y - peak.y) ** 2;
    return value + peak.height * Math.exp(-distanceSquared / 40);
  });
}
