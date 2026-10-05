// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getVisibleDiskFraction} from './terrain-horizon-oracle';

const RADIANS = Math.PI / 180;

/** Inputs of {@link computeSolarIrradiance}. */
export type SolarIrradianceOracleOptions = {
  pixelCount: number;
  directionCount: number;
  /** Pixel-major horizon angles in degrees (decoded when the GPU reads unorm16). */
  horizon: ArrayLike<number>;
  /** Rows of `[azimuth, altitude, durationHours, directNormalIrradiance]`. */
  sunTable: ArrayLike<number>;
  sampleCount: number;
  angularRadiusDegrees: number;
  diffuseIrradiance?: number;
  slope?: ArrayLike<number>;
  aspect?: ArrayLike<number>;
  skyViewFactor?: ArrayLike<number>;
};

/** Float64 sun-hours and insolation; invalid pixels are NaN with validity 0. */
export function computeSolarIrradiance(options: SolarIrradianceOracleOptions): {
  sunHours: number[];
  insolation: number[];
  validity: number[];
} {
  const {pixelCount, directionCount, horizon, sunTable, sampleCount} = options;
  const radius = options.angularRadiusDegrees;
  const spacing = 360 / directionCount;
  const result = {sunHours: [] as number[], insolation: [] as number[], validity: [] as number[]};
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    const slope = options.slope?.[pixel];
    const aspect = options.aspect?.[pixel];
    const svf = options.skyViewFactor?.[pixel];
    const valid =
      Number.isFinite(horizon[pixel * directionCount]) &&
      (slope === undefined || (Number.isFinite(slope) && Number.isFinite(aspect))) &&
      (svf === undefined || Number.isFinite(svf));
    if (!valid) {
      result.sunHours.push(NaN);
      result.insolation.push(NaN);
      result.validity.push(0);
      continue;
    }
    let hours = 0;
    let energy = 0;
    for (let sample = 0; sample < sampleCount; sample++) {
      const [azimuthDegrees, altitude, duration, normal] = Array.from(
        {length: 4},
        (_, column) => sunTable[sample * 4 + column]
      );
      if (altitude + radius <= 0) {
        continue;
      }
      const azimuth = ((azimuthDegrees % 360) + 360) % 360;
      const lower = Math.min(Math.floor(azimuth / spacing), directionCount - 1);
      const upper = (lower + 1) % directionCount;
      const blend = azimuth / spacing - lower;
      const horizonAngle =
        horizon[pixel * directionCount + lower] * (1 - blend) +
        horizon[pixel * directionCount + upper] * blend;
      const visibility = getVisibleDiskFraction(altitude, horizonAngle, radius);
      hours += visibility * duration;
      let cosIncidence = Math.sin(altitude * RADIANS);
      if (slope !== undefined && aspect !== undefined && aspect >= 0) {
        cosIncidence =
          Math.cos(slope * RADIANS) * Math.sin(altitude * RADIANS) +
          Math.sin(slope * RADIANS) *
            Math.cos(altitude * RADIANS) *
            Math.cos((azimuthDegrees - aspect) * RADIANS);
      }
      energy += normal * visibility * Math.max(cosIncidence, 0) * duration;
      if (altitude > 0) {
        energy += (options.diffuseIrradiance ?? 0) * (svf ?? 1) * duration;
      }
    }
    result.sunHours.push(hours);
    result.insolation.push(energy);
    result.validity.push(1);
  }
  return result;
}
