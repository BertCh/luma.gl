// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * One sight line on the CPU, with the same decision rule as the GPU contributors. The viewshed
 * asks it for every cell at once; this module asks it for one ray so the story can draw what the
 * GPU does for each cell: the profile, the steepest ground the ray passes (the horizon sample),
 * the running horizon while a scan dot walks from the eye, and the tolerance band.
 *
 * The rule (`terrain-sight-line.ts` of `gpu-terrain`): every sample at distance `d` has the slope
 * `(height - c d^2 - eye) / d`, the target the slope `T` with its own height and drop. The ray is
 * hidden when the steepest sample before the ignored final stretch exceeds `T + band`, visible
 * when it is at most `T - band`, marginal between, and the clearance the GPU reports is
 * `(T - steepest) * D` in metres: how far the target could sink and stay visible, or, when
 * negative, how much taller it would have to be.
 */

import {formatDistance, formatSigned} from '../../cartography/live-text';
import type {ChartData} from '../chart-types';
import type {RaySamples} from './cpu-dem';

/** The tolerance settings of a sight line, as the `GPUTerrainViewshed` tolerance buffer takes them. */
export type RayTolerance = {
  toleranceMeters: number;
  tolerancePerKilometer: number;
  targetIgnoreDistance: number;
  targetIgnoreFraction: number;
};

/** The three results of a sight line. */
export type RayCode = 'visible' | 'marginal' | 'hidden';

/** The outcome of {@link getRayVerdict}. */
export type RayVerdict = {
  code: RayCode;
  /** `(T - steepest) * D` in metres, or null when no sample was tested. */
  clearanceMeters: number | null;
  /** Index of the steepest sample (the horizon sample), or -1 when no sample was tested. */
  horizonIndex: number;
  /** Ground distance of the horizon sample from the eye, metres. */
  horizonDistance: number;
  /** Height of the horizon sample (ground, not lowered), metres. */
  horizonHeight: number;
  /** Half-width of the tolerance band at the target as a slope. */
  bandSlope: number;
  /** Slope from the eye to the (lowered) target. */
  targetSlope: number;
  /** Samples at or before this ground distance were tested (the rest are the ignored stretch). */
  testedDistance: number;
  /** Samples tested. */
  testedCount: number;
};

/** Slope of every sample of a ray seen from its eye (index 0 has no slope: NaN). */
export function getSampleSlopes(ray: RaySamples): Float64Array {
  const slopes = new Float64Array(ray.count).fill(Number.NaN);
  for (let index = 1; index < ray.count; index++) {
    slopes[index] = (ray.loweredTerrain[index] - ray.eyeElevation) / ray.distance[index];
  }
  return slopes;
}

/**
 * Applies the GPU decision rule to a sampled ray. Samples closer to the target than
 * `max(targetIgnoreDistance, targetIgnoreFraction * D)` are not tested (the last sample, the
 * target itself, never is).
 */
export function getRayVerdict(ray: RaySamples, tolerance: RayTolerance): RayVerdict {
  const total = ray.totalDistance;
  const targetSlope =
    total > 0
      ? (ray.sightLine[ray.count - 1] - ray.eyeElevation) / total
      : Number.NEGATIVE_INFINITY;
  const bandSlope =
    total > 0
      ? (tolerance.toleranceMeters + (tolerance.tolerancePerKilometer * total) / 1000) / total
      : 0;
  const testedDistance =
    total - Math.max(tolerance.targetIgnoreDistance, tolerance.targetIgnoreFraction * total);
  const slopes = getSampleSlopes(ray);
  let horizonIndex = -1;
  let steepest = Number.NEGATIVE_INFINITY;
  let testedCount = 0;
  for (let index = 1; index < ray.count - 1; index++) {
    if (ray.distance[index] > testedDistance) break;
    testedCount++;
    if (slopes[index] > steepest) {
      steepest = slopes[index];
      horizonIndex = index;
    }
  }
  let code: RayCode = 'marginal';
  if (steepest > targetSlope + bandSlope) code = 'hidden';
  else if (steepest <= targetSlope - bandSlope) code = 'visible';
  return {
    code,
    clearanceMeters: horizonIndex >= 0 ? (targetSlope - steepest) * total : null,
    horizonIndex,
    horizonDistance: horizonIndex >= 0 ? ray.distance[horizonIndex] : 0,
    horizonHeight: horizonIndex >= 0 ? ray.terrain[horizonIndex] : 0,
    bandSlope,
    targetSlope,
    testedDistance,
    testedCount
  };
}

/** `Visible`, `Marginal` or `Hidden` with the metres of clearance, for a tooltip row. */
export function describeVerdict(verdict: RayVerdict): string {
  const label = verdict.code[0].toUpperCase() + verdict.code.slice(1);
  if (verdict.clearanceMeters === null) return label;
  return `${label} (${formatSigned(verdict.clearanceMeters)} m)`;
}

/** Options of {@link getProfileChart}. */
export type ProfileChartOptions = {
  /** The scan position as a fraction of the ray, 0 at the eye to 1 at the target. */
  scanFraction: number;
  /** Draw the curve-lowered terrain (false for the flat-earth model). */
  showCurve: boolean;
  /** Names of the two ends, for the chart description. */
  observerName: string;
  targetName: string;
};

const KILOMETRES = (meters: number) => `${(meters / 1000).toFixed(1)} km`;

/**
 * The profile of one sight line (distance against height): the ground as a filled area, the ground
 * lowered by the earth's curve (dashed), the straight sight line from the eye to the lowered
 * target, and the horizon so far: the steepest line from the eye to any sample up to the scan
 * position. A rule marks the scan position and one the horizon sample, the ground that decides.
 */
export function getProfileChart(
  ray: RaySamples,
  verdict: RayVerdict,
  options: ProfileChartOptions
): ChartData {
  const kilometres = Array.from(ray.distance, distance => distance / 1000);
  const scanIndex = Math.min(
    ray.count - 1,
    Math.max(0, Math.round(options.scanFraction * (ray.count - 1)))
  );
  const scanDistance = ray.distance[scanIndex];
  // The horizon so far: the running maximum slope up to the scan sample, drawn from the eye.
  const slopes = getSampleSlopes(ray);
  let steepest = Number.NEGATIVE_INFINITY;
  for (let index = 1; index <= scanIndex; index++) steepest = Math.max(steepest, slopes[index]);
  const horizonLine =
    Number.isFinite(steepest) && scanIndex > 0
      ? {
          label: 'Horizon so far',
          x: [0, ray.totalDistance / 1000],
          y: [ray.eyeElevation, ray.eyeElevation + steepest * ray.totalDistance],
          dashed: true,
          color: 5,
          width: 1.25
        }
      : null;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < ray.count; index++) {
    minimum = Math.min(minimum, ray.terrain[index]);
    maximum = Math.max(maximum, ray.terrain[index], ray.sightLine[index]);
  }
  const markers: {x: number; label?: string}[] = [
    {x: scanDistance / 1000, label: scanIndex > 0 ? 'scan' : 'eye'}
  ];
  if (verdict.horizonIndex >= 0) {
    markers.push({x: verdict.horizonDistance / 1000, label: 'steepest ground'});
  }
  const clearance =
    verdict.clearanceMeters === null
      ? ''
      : `, clearance ${formatSigned(verdict.clearanceMeters)} m`;
  return {
    kind: 'line',
    height: 150,
    xLabel: 'Distance from the eye (km)',
    yLabel: 'Height (m)',
    xDomain: [0, ray.totalDistance / 1000],
    yDomain: [Math.floor(minimum / 100) * 100, Math.ceil((maximum + 40) / 100) * 100],
    formatX: value => value.toFixed(1),
    formatY: value => Math.round(value).toLocaleString('en-US'),
    series: [
      {
        label: 'Ground',
        x: kilometres,
        y: Array.from(ray.terrain),
        area: true,
        color: 4,
        width: 1.25
      },
      ...(options.showCurve
        ? [
            {
              label: 'Ground lowered by the curve',
              x: kilometres,
              y: Array.from(ray.loweredTerrain),
              dashed: true,
              color: 4,
              width: 1.25
            }
          ]
        : []),
      {label: 'Sight line', x: kilometres, y: Array.from(ray.sightLine), color: 3, width: 2},
      ...(horizonLine ? [horizonLine] : [])
    ],
    markers,
    description: `Profile of the sight line from ${options.observerName} to ${options.targetName}, ${KILOMETRES(ray.totalDistance)} long: the ${verdict.code} result${clearance}. The steepest ground is ${formatDistance(verdict.horizonDistance)} from the eye.`
  };
}
