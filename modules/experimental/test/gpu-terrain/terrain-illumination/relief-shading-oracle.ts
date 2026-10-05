// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_RELIEF_SHADING_CONTRAST_PIVOT,
  GPU_RELIEF_SHADING_MDOW_LIGHTS,
  type GPUReliefShadingSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {computeSobel} from '../terrain-analysis/terrain-analysis-oracle';

const RADIANS = Math.PI / 180;

/** GLSL/WGSL `smoothstep` in float64. */
export function smoothstepOracle(low: number, high: number, value: number): number {
  const t = Math.min(Math.max((value - low) / (high - low), 0), 1);
  return t * t * (3 - 2 * t);
}

/** Float64 relief shading result. */
export type ReliefShadingOracleResult = {
  hillshade: number[];
  relief: number[];
  /** Unpacked `[r, g, b, a]` bytes per pixel; zeros for invalid pixels. */
  color: number[][];
  validity: number[];
};

/** Float64 multidirectional hillshade and Imhof blend following the documented formulas. */
export function computeReliefShading(options: {
  width: number;
  height: number;
  elevation: Float32Array;
  settings: GPUReliefShadingSettings;
  skyViewFactor?: ArrayLike<number>;
  textureShade?: ArrayLike<number>;
  /** Curvature raster, positive on ridges. */
  curvature?: ArrayLike<number>;
  rowDirection?: 'south' | 'north';
}): ReliefShadingOracleResult {
  const {width, height, elevation, settings} = options;
  const gradientX = computeSobel(elevation, undefined, width, height, 'x', 'clamp');
  const gradientY = computeSobel(elevation, undefined, width, height, 'y', 'clamp');
  const lightSet = settings.lights ?? 'mdow';
  const lights = (lightSet === 'mdow' ? GPU_RELIEF_SHADING_MDOW_LIGHTS : lightSet).map(light => ({
    azimuth: light.azimuthDegrees * RADIANS,
    altitude: (light.altitudeDegrees ?? 45) * RADIANS,
    weight: light.weight ?? 1
  }));
  const weighting = settings.lightWeighting ?? (lightSet === 'mdow' ? 'aspect' : 'fixed');
  const aspectWeighting = weighting === 'aspect';
  const swingRadians =
    weighting === 'imhof-swing' ? (settings.imhofSwingDegrees ?? 65) * RADIANS : 0;
  const zFactor = settings.zFactor ?? 1;
  const rowSign = (options.rowDirection ?? 'south') === 'south' ? -1 : 1;
  const stops = settings.elevationStops ?? [];
  const warm = settings.warmColor ?? [1, 0.95, 0.85];
  const cool = settings.coolColor ?? [0.8, 0.85, 1];
  const result: ReliefShadingOracleResult = {
    hillshade: [],
    relief: [],
    color: [],
    validity: []
  };
  for (let pixel = 0; pixel < width * height; pixel++) {
    const east = (zFactor * gradientX.values[pixel]) / (8 * settings.cellSize[0]);
    const north = (rowSign * zFactor * gradientY.values[pixel]) / (8 * settings.cellSize[1]);
    const length = Math.hypot(east, north, 1);
    const normal = [-east / length, -north / length, 1 / length];
    const aspect = Math.atan2(-east, -north);
    const isFlat = east === 0 && north === 0;
    const swingFactor = smoothstepOracle(0.05, 0.4, Math.hypot(normal[0], normal[1]));
    let weighted = 0;
    let weights = 0;
    for (const light of lights) {
      const azimuth =
        weighting === 'imhof-swing'
          ? light.azimuth + swingRadians * Math.sin(aspect - light.azimuth) * swingFactor
          : light.azimuth;
      const weight = aspectWeighting
        ? isFlat
          ? 1
          : Math.sin(aspect - light.azimuth) ** 2
        : light.weight;
      const direction = [
        Math.sin(azimuth) * Math.cos(light.altitude),
        Math.cos(azimuth) * Math.cos(light.altitude),
        Math.sin(light.altitude)
      ];
      const dot = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2];
      weighted += weight * Math.max(dot, 0);
      weights += weight;
    }
    const hillshade = weights > 0 ? weighted / weights : 0;
    const gradientValid = gradientX.validity[pixel] === 1 && gradientY.validity[pixel] === 1;
    let valid = gradientValid;
    let shade = 1 + (hillshade - 1) * (settings.hillshadeStrength ?? 1);
    if (options.skyViewFactor) {
      const svf = options.skyViewFactor[pixel];
      valid &&= Number.isFinite(svf);
      shade *= 1 + (svf - 1) * (settings.skyViewStrength ?? 1);
    }
    if (options.textureShade) {
      const texture = options.textureShade[pixel];
      valid &&= Number.isFinite(texture);
      shade += (settings.textureShadeStrength ?? 0.5) * texture;
    }
    if (options.curvature) {
      const curvature = options.curvature[pixel];
      valid &&= Number.isFinite(curvature);
      shade += (settings.curvatureStrength ?? 1) * curvature;
    }
    const contrastStrength = settings.contrastStrength ?? 0;
    if (contrastStrength !== 0) {
      const low = settings.contrastLowElevation ?? 0;
      const high = settings.contrastHighElevation ?? 0;
      const ramp =
        high > low
          ? smoothstepOracle(low, high, elevation[pixel])
          : elevation[pixel] >= low
            ? 1
            : 0;
      const k = 1 - contrastStrength + contrastStrength * ramp;
      shade = GPU_RELIEF_SHADING_CONTRAST_PIVOT + (shade - GPU_RELIEF_SHADING_CONTRAST_PIVOT) * k;
    }
    const relief = Math.min(Math.max(shade * (settings.exposure ?? 1), 0), 1);
    result.hillshade.push(gradientValid ? hillshade : NaN);
    result.relief.push(valid ? relief : NaN);
    result.validity.push(valid ? 1 : 0);
    if (!valid) {
      result.color.push([0, 0, 0, 0]);
      continue;
    }
    const base = getRampColor(stops, elevation[pixel]);
    const lightness = Math.min(Math.max(2 * hillshade - 1, -1), 1);
    const tintStrength = settings.tintStrength ?? 0.3;
    const tintTarget = lightness >= 0 ? warm : cool;
    const amount = Math.abs(lightness) * tintStrength;
    const color = [0, 1, 2].map(channel => {
      const tint = 1 + (tintTarget[channel] - 1) * amount;
      return Math.round(Math.min(Math.max(base[channel] * relief * tint, 0), 1) * 255);
    });
    result.color.push([...color, 255]);
  }
  return result;
}

function getRampColor(
  stops: NonNullable<GPUReliefShadingSettings['elevationStops']>,
  elevation: number
): number[] {
  if (stops.length === 0) {
    return [0.85, 0.85, 0.85];
  }
  if (elevation <= stops[0].elevation) {
    return [...stops[0].color];
  }
  for (let index = 1; index < stops.length; index++) {
    if (elevation < stops[index].elevation) {
      const blend =
        (elevation - stops[index - 1].elevation) /
        (stops[index].elevation - stops[index - 1].elevation);
      return [0, 1, 2].map(
        channel =>
          stops[index - 1].color[channel] +
          (stops[index].color[channel] - stops[index - 1].color[channel]) * blend
      );
    }
  }
  return [...stops[stops.length - 1].color];
}

/** Unpacks `pack4x8unorm` words into `[r, g, b, a]` bytes. */
export function unpackColors(words: ArrayLike<number>): number[][] {
  return Array.from(words, word => [
    word & 0xff,
    (word >>> 8) & 0xff,
    (word >>> 16) & 0xff,
    word >>> 24
  ]);
}
