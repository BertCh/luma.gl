// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The charts of the horizon scene. The panorama is the hero: elevation angle against bearing, the
 * skyline as an area, the classified peaks as points on or below it, named peaks as rules. The ray
 * profile shows one sight ray (angle against distance, the highest angle so far, the ridge). The
 * skyline peaks are a ranked bar list. Pure TypeScript.
 */

import type {BarChartData, ChartSeries, LineChartData} from '../scene';
import {formatBearing, type HorizonView, PANORAMA_START_DEGREES} from './horizon-style';
import type {SightRay} from './horizon-ray';

/** A catalogue peak as the panorama plots it. */
export type PanoramaPeak = {
  /** Bearing from the eye, degrees clockwise from north. */
  azimuth: number;
  /** Elevation angle of the summit from the eye, degrees. */
  angle: number;
  code: 'visible' | 'marginal' | 'hidden';
};

/** Everything {@link buildPanoramaChart} draws. */
export type PanoramaInput = {
  view: HorizonView;
  /** Rays per full circle; `skylineAngles[i]` is the angle at bearing `i * 360 / azimuthCount`. */
  azimuthCount: number;
  skylineAngles: ArrayLike<number>;
  /** Classified peaks (the `visibility` view). */
  peaks: readonly PanoramaPeak[];
  /** Named peaks drawn as labelled rules. */
  markers: readonly {azimuth: number; label: string}[];
  /** Maxima of the skyline (the `skyline-peaks` view). */
  skylinePeaks: readonly {azimuth: number; angle: number}[];
  /** Label of the linked bearing marker (the `visibility` view); undefined leaves it unlinked. */
  linkLabel?: (value: number) => string;
};

/** The x position of a bearing on the panorama: it starts at {@link PANORAMA_START_DEGREES}. */
export function getPanoramaX(azimuthDegrees: number): number {
  const wrapped = ((azimuthDegrees % 360) + 360) % 360;
  return wrapped < PANORAMA_START_DEGREES ? wrapped + 360 : wrapped;
}

const isValidAngle = (angle: number): boolean => Number.isFinite(angle) && angle > -89.5;

/**
 * Pairs of identical points separated by gaps: a series drawn with a wide round line cap shows each
 * pair as a dot, so the points need no line between them.
 */
function toDots(xs: readonly number[], ys: readonly number[]): {x: number[]; y: number[]} {
  const x: number[] = [];
  const y: number[] = [];
  xs.forEach((value, index) => {
    if (x.length) {
      x.push(Number.NaN);
      y.push(Number.NaN);
    }
    x.push(value, value);
    y.push(ys[index], ys[index]);
  });
  return {x, y};
}

/**
 * The panorama: bearing from south round to south (x) against elevation angle (y). The skyline is
 * the area; visible peaks touch it, hidden peaks sit below it. Clicking the chart in the
 * `visibility` view writes the `bearing` option.
 */
export function buildPanoramaChart(input: PanoramaInput): LineChartData {
  const {azimuthCount} = input;
  const startRay = Math.round((PANORAMA_START_DEGREES / 360) * azimuthCount);
  const x: number[] = [];
  const y: number[] = [];
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let step = 0; step <= azimuthCount; step++) {
    const angle = input.skylineAngles[(startRay + step) % azimuthCount];
    x.push(PANORAMA_START_DEGREES + (step * 360) / azimuthCount);
    if (isValidAngle(angle)) {
      y.push(angle);
      minimum = Math.min(minimum, angle);
      maximum = Math.max(maximum, angle);
    } else {
      y.push(Number.NaN);
    }
  }
  const finite = Number.isFinite(minimum);
  const series: ChartSeries[] = [{label: 'Skyline', x, y, area: true, color: 2, width: 1.75}];
  if (input.view === 'visibility') {
    const visible = input.peaks.filter(peak => peak.code === 'visible');
    const marginal = input.peaks.filter(peak => peak.code === 'marginal');
    const hidden = input.peaks.filter(peak => peak.code === 'hidden');
    if (hidden.length) {
      const dots = toDots(
        hidden.map(peak => getPanoramaX(peak.azimuth)),
        hidden.map(peak => peak.angle)
      );
      series.push({label: 'Hidden peaks', x: dots.x, y: dots.y, ghost: true, width: 5});
    }
    if (marginal.length) {
      series.push({
        label: 'Marginal peaks',
        x: marginal.map(peak => getPanoramaX(peak.azimuth)),
        y: marginal.map(peak => peak.angle),
        color: 4,
        points: true,
        width: 0
      });
    }
    if (visible.length) {
      series.push({
        label: 'Visible peaks',
        x: visible.map(peak => getPanoramaX(peak.azimuth)),
        y: visible.map(peak => peak.angle),
        color: 0,
        points: true,
        width: 0
      });
    }
  }
  if (input.view === 'skyline-peaks' && input.skylinePeaks.length) {
    series.push({
      label: 'Skyline peaks',
      x: input.skylinePeaks.map(peak => getPanoramaX(peak.azimuth)),
      y: input.skylinePeaks.map(peak => peak.angle),
      color: 1,
      points: true,
      width: 0
    });
  }
  const lower = finite ? Math.floor(Math.min(minimum, 0)) : -6;
  const upper = finite ? Math.ceil(maximum) + 1 : 12;
  return {
    kind: 'line',
    height: 176,
    series,
    xDomain: [PANORAMA_START_DEGREES, PANORAMA_START_DEGREES + 360],
    yDomain: [lower, upper],
    xLabel: 'Bearing from the eye, south to south by way of west and north',
    yLabel: 'Elevation angle (°)',
    formatX: value => formatBearing(value),
    formatY: value => `${value}°`,
    markers: input.markers.map(marker => ({x: getPanoramaX(marker.azimuth), label: marker.label})),
    ...(input.linkLabel ? {link: {option: 'bearing', label: input.linkLabel}} : {}),
    table: false,
    description:
      'Panorama from the eye: the highest elevation angle at every bearing as a filled skyline, with the named peaks marked. In the visibility view visible peaks touch the skyline and hidden peaks sit below it.'
  };
}

/** What {@link buildRayProfileChart} draws. */
export type RayProfileInput = {
  ray: SightRay;
  /** Name of the target peak, or null for a free ray (its ridge is the skyline). */
  targetName: string | null;
  /** Elevation angle of the target from the eye, degrees (null for a free ray). */
  targetAngle: number | null;
};

/**
 * One sight ray: the angle of the ground at every distance, the highest angle so far (it only
 * rises) and the ridge that sets it. A peak is hidden when its angle guide lies below the curve
 * before the peak.
 */
export function buildRayProfileChart(input: RayProfileInput): LineChartData {
  const {ray} = input;
  const distances = Array.from(ray.distanceMeters, value => value / 1000);
  const markers: {x: number; label: string}[] = [];
  if (ray.ridge) {
    markers.push({
      x: ray.ridge.distanceMeters / 1000,
      label: input.targetName ? 'Ridge' : 'Skyline'
    });
  }
  if (input.targetName) {
    markers.push({x: ray.lengthMeters / 1000, label: input.targetName});
  }
  return {
    kind: 'line',
    height: 150,
    series: [
      {label: 'Ground', x: distances, y: Array.from(ray.terrainAngle), ghost: true, width: 1},
      {
        label: 'Highest so far',
        x: distances,
        y: Array.from(ray.runningMaxAngle),
        color: 2,
        width: 2
      }
    ],
    xLabel: 'Distance from the eye (km)',
    yLabel: 'Angle (°)',
    formatY: value => `${value}°`,
    ...(input.targetAngle !== null && Number.isFinite(input.targetAngle)
      ? {guides: [{y: input.targetAngle, label: 'Peak angle'}]}
      : {}),
    markers,
    table: false,
    description: input.targetName
      ? `Elevation angle against distance along the ray to ${input.targetName}: the ground, the highest angle so far, the ridge that sets it and the angle of the peak.`
      : 'Elevation angle against distance along the ray: the ground, the highest angle so far and the skyline where it is set.'
  };
}

/** One skyline peak of the ranked list. */
export type SkylinePeakRow = {label: string; prominence: number};

/** The skyline peaks ranked by prominence, a horizontal bar list; the first bar is highlighted. */
export function buildSkylinePeaksChart(
  rows: readonly SkylinePeakRow[],
  onBarClick?: (index: number) => void
): BarChartData {
  return {
    kind: 'bars',
    horizontal: true,
    values: rows.map(row => row.prominence),
    labels: rows.map(row => row.label),
    highlight: [0],
    formatX: value => `${value.toFixed(1)}°`,
    ...(onBarClick ? {onBarClick} : {}),
    table: false,
    description:
      'Skyline peaks ranked by prominence in degrees: how far each stands above the lower of its two surrounding valleys on the skyline.'
  };
}
