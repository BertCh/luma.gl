// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * One sight ray of the horizon scene on the CPU: the terrain angle at every step along a line from
 * the eye, its running maximum and the ridge that sets it. It is the same definition as the GPU
 * `GPUPointHorizonVisibility` (angle of the curvature-lowered ground, `atan((z - c d^2 - eye) / d)`)
 * evaluated for the one ray the reader clicked, so the profile chart can show what the GPU does for
 * every peak. Pure TypeScript, built on `DemProbe.sampleRay`.
 */

import type {DemProbe, TerrainDem} from './cpu-dem';

const DEGREES = 180 / Math.PI;

/** Refraction coefficient `k` of an option value; `none` is `k = 1` (no curvature drop at all). */
export function getRefractionValue(refraction: 'none' | 'mt-image' | 'gdal'): number {
  if (refraction === 'none') return 1;
  return refraction === 'gdal' ? 1 / 7 : 0.13;
}

/** The ridge that sets the highest angle in front of a target (or the skyline of a free ray). */
export type RayRidge = {
  /** Metres from the eye. */
  distanceMeters: number;
  /** Elevation angle of the ridge from the eye, degrees. */
  angleDegrees: number;
  /** Ground height at the ridge, metres. */
  heightMeters: number;
  /** `[longitude, latitude]` of the ridge. */
  lngLat: readonly [number, number];
};

/** The samples and the ridge of one sight ray. All arrays start at the first step (not the eye). */
export type SightRay = {
  distanceMeters: Float64Array;
  /** Angle of the lowered ground at each step, degrees. */
  terrainAngle: Float64Array;
  /** Highest angle so far at each step, degrees. */
  runningMaxAngle: Float64Array;
  /** Ground height at each step, metres. */
  terrainHeight: Float64Array;
  eyeElevationMeters: number;
  /** Total length of the ray, metres. */
  lengthMeters: number;
  /** The ridge before the cutoff, or null when the ray is too short to have one. */
  ridge: RayRidge | null;
  /** Angle of the last sample (the target) from the eye, degrees. */
  endAngleDegrees: number;
};

/** Options of {@link castSightRay}. */
export type SightRayOptions = {
  from: readonly [number, number];
  to: readonly [number, number];
  stepMeters: number;
  /** Eye height above the ground at `from`, metres. */
  eyeHeightMeters: number;
  /** Refraction coefficient `k` (see {@link getRefractionValue}). */
  refraction: number;
  /**
   * The ridge is searched at distances up to this, metres. For a peak it is the distance minus the
   * stretch the occlusion test ignores; omit it for a free ray (the ridge is then the skyline).
   */
  cutoffMeters?: number;
};

/** Samples one sight ray and finds its ridge. */
export function castSightRay(probe: DemProbe, options: SightRayOptions): SightRay {
  const samples = probe.sampleRay(options.from, options.to, options.stepMeters, {
    refraction: options.refraction,
    eyeHeight: options.eyeHeightMeters,
    targetHeight: 0
  });
  const count = Math.max(0, samples.count - 1);
  const distanceMeters = new Float64Array(count);
  const terrainAngle = new Float64Array(count);
  const runningMaxAngle = new Float64Array(count);
  const terrainHeight = new Float64Array(count);
  const cutoff = options.cutoffMeters ?? Number.POSITIVE_INFINITY;
  let ridgeIndex = -1;
  let ridgeAngle = -90;
  for (let index = 0; index < count; index++) {
    const source = index + 1;
    const distance = samples.distance[source];
    distanceMeters[index] = distance;
    terrainHeight[index] = samples.terrain[source];
    terrainAngle[index] =
      Math.atan2(samples.loweredTerrain[source] - samples.eyeElevation, distance) * DEGREES;
    runningMaxAngle[index] = samples.runningMaxAngle[source];
    if (distance <= cutoff && terrainAngle[index] > ridgeAngle) {
      ridgeAngle = terrainAngle[index];
      ridgeIndex = index;
    }
  }
  const ridge: RayRidge | null =
    ridgeIndex >= 0
      ? {
          distanceMeters: distanceMeters[ridgeIndex],
          angleDegrees: ridgeAngle,
          heightMeters: terrainHeight[ridgeIndex],
          lngLat: [samples.lngLat[(ridgeIndex + 1) * 2], samples.lngLat[(ridgeIndex + 1) * 2 + 1]]
        }
      : null;
  return {
    distanceMeters,
    terrainAngle,
    runningMaxAngle,
    terrainHeight,
    eyeElevationMeters: samples.eyeElevation,
    lengthMeters: samples.totalDistance,
    ridge,
    endAngleDegrees: count > 0 ? terrainAngle[count - 1] : Number.NaN
  };
}

/**
 * The far end of a ray from the eye along a bearing: `maximumMeters` out, shortened until it lies
 * inside the DEM (a ray that leaves the data stops where the data ends).
 *
 * @param unproject Layer metres to `[longitude, latitude]`.
 */
export function getRayEnd(
  dem: Pick<TerrainDem, 'width' | 'height' | 'getPixelCoordinates'>,
  eyeMeters: readonly [number, number],
  bearingDegrees: number,
  maximumMeters: number,
  unproject: (x: number, y: number) => [number, number]
): {lngLat: [number, number]; meters: [number, number]; lengthMeters: number} {
  const radians = (bearingDegrees * Math.PI) / 180;
  let length = maximumMeters;
  for (let attempt = 0; attempt < 200; attempt++) {
    const x = eyeMeters[0] + length * Math.sin(radians);
    const y = eyeMeters[1] + length * Math.cos(radians);
    const lngLat = unproject(x, y);
    const [column, row] = dem.getPixelCoordinates(lngLat[0], lngLat[1]);
    const inside = column >= 1 && row >= 1 && column <= dem.width - 1 && row <= dem.height - 1;
    if (inside || length < 100) return {lngLat, meters: [x, y], lengthMeters: length};
    length *= 0.97;
  }
  const fallback = unproject(eyeMeters[0], eyeMeters[1]);
  return {lngLat: fallback, meters: [eyeMeters[0], eyeMeters[1]], lengthMeters: 0};
}
