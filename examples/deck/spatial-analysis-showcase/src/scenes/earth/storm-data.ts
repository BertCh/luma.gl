// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {BarChartData, LineChartData} from '../scene';
import {createAzimuthalEquidistant, createTrackSet, type TrackSet} from '../movement/b12-tracks';

/**
 * Shared helpers of the three storm scenes (cell tracks, warning verification, outage exposure):
 * the common clock, the projections, the loaders of the poopdeck severe-weather datasets and a
 * few chart builders. Every dataset stores `uint32` seconds since {@link STORM_ORIGIN_UNIX}.
 */

/** 2024-05-21T12:00:00Z, the time origin of every `poopdeck-*` severe-weather dataset. */
export const STORM_ORIGIN_UNIX = Date.UTC(2024, 4, 21, 12, 0, 0) / 1000;
/** Length of the dataset window: 12:00 UTC on 21 May to 06:00 UTC on 22 May 2024. */
export const STORM_EVENT_SECONDS = 18 * 3600;
/** The part of the window that holds storm reports and warnings: 17:30 UTC to 03:00 UTC. */
export const STORM_VERIFICATION_RANGE: readonly [number, number] = [19800, 54000];
/** Centre of the azimuthal-equidistant frame used for joins and areal sums. */
export const STORM_FRAME_CENTER: readonly [number, number] = [-92, 37];
/** Planar frame of the joins: distances from the centre are exact, tangential error is under 1% at 1,500 km. */
export const STORM_FRAME = createAzimuthalEquidistant(STORM_FRAME_CENTER as [number, number]);

/** Camera over the central and eastern US where the 21 May 2024 storms ran. */
export const STORM_VIEW = {longitude: -93.5, latitude: 38, zoom: 4.4};

/** Speed in km/h at which the speed ramps of the storm scenes end. */
export const STORM_SPEED_RAMP_KMH = 120;

const METERS_PER_DEGREE = 111194.9;
const MONTH_DAY = ['Tue 21 May', 'Wed 22 May'];

/** `Tue 21 May 18:30 UTC` for seconds since the storm origin. */
export function formatStormClock(seconds: number): string {
  const unix = STORM_ORIGIN_UNIX + Math.round(seconds);
  const date = new Date(unix * 1000);
  const day = MONTH_DAY[Math.min(1, Math.max(0, date.getUTCDate() - 21))];
  const hours = String(date.getUTCHours()).padStart(2, '0');
  const minutes = String(date.getUTCMinutes()).padStart(2, '0');
  return `${day} ${hours}:${minutes} UTC`;
}

/** `18:30` for seconds since the storm origin. */
export function formatStormHourMinute(seconds: number): string {
  const date = new Date((STORM_ORIGIN_UNIX + Math.round(seconds)) * 1000);
  return `${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')}`;
}

/** Central Daylight Time (UTC-5) clock of a storm time, which is what people in the Plains saw. */
export function formatStormCentral(seconds: number): string {
  const date = new Date((STORM_ORIGIN_UNIX + Math.round(seconds) - 5 * 3600) * 1000);
  return `${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')} CDT`;
}

/** Compass sector names, north first. */
export const COMPASS_SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'] as const;

/** One color per compass sector, hue around the circle; readable on light and dark basemaps. */
export const COMPASS_COLORS: readonly (readonly [number, number, number, number])[] = [
  [96, 150, 255, 255],
  [52, 200, 232, 255],
  [72, 202, 124, 255],
  [186, 216, 66, 255],
  [246, 190, 58, 255],
  [240, 120, 60, 255],
  [226, 78, 112, 255],
  [166, 100, 224, 255]
];

/** Storm-cell tracks in the layout of the trajectory contributors, plus the per-vertex reflectivity. */
export type StormTrackSet = TrackSet & {
  /** Cell reflectivity in dBZ per vertex. */
  reflectivity: Float32Array;
  /** Peak dBZ per track. */
  peakDbz: Float32Array;
  /** Track start and end seconds. */
  startTimes: Float32Array;
  endTimes: Float32Array;
};

/**
 * Projects every track on its own tangent plane: x east and y north in meters around the track's
 * mean latitude and first longitude. A track is at most a few hundred kilometers long, so the
 * east-west scale error stays under 3%, and every step heading is true compass north, which a
 * single continental projection cannot give.
 */
export function projectTracksLocal(lngLat: Float32Array, offsets: Uint32Array): Float32Array {
  const local = new Float32Array(lngLat.length);
  for (let track = 0; track + 1 < offsets.length; track++) {
    const first = offsets[track];
    const last = offsets[track + 1];
    let latitudeSum = 0;
    for (let vertex = first; vertex < last; vertex++) latitudeSum += lngLat[vertex * 2 + 1];
    const referenceLatitude = latitudeSum / Math.max(1, last - first);
    const referenceLongitude = lngLat[first * 2];
    const scale = Math.cos((referenceLatitude * Math.PI) / 180);
    for (let vertex = first; vertex < last; vertex++) {
      local[vertex * 2] = (lngLat[vertex * 2] - referenceLongitude) * scale * METERS_PER_DEGREE;
      local[vertex * 2 + 1] = (lngLat[vertex * 2 + 1] - referenceLatitude) * METERS_PER_DEGREE;
    }
  }
  return local;
}

/** Loads `poopdeck-mrms-precip-tracks`. `positions` are per-track tangent-plane meters (see {@link projectTracksLocal}). */
export function loadStormTracks(dataset: LoadedDataset): StormTrackSet {
  const offsets = dataset.column<Uint32Array>('pathOffsets');
  const lngLat = dataset.column<Float32Array>('vertices');
  const set = createTrackSet({
    offsets,
    positions: projectTracksLocal(lngLat, offsets),
    lngLat,
    timestamps: Float32Array.from(dataset.column<Uint32Array>('timestamp')),
    origin: STORM_FRAME_CENTER,
    project: STORM_FRAME.project,
    unproject: STORM_FRAME.unproject,
    drawInDegrees: true
  });
  return {
    ...set,
    reflectivity: dataset.column<Float32Array>('reflectivity'),
    peakDbz: dataset.column<Float32Array>('peakDbz'),
    startTimes: Float32Array.from(dataset.column<Uint32Array>('startTime')),
    endTimes: Float32Array.from(dataset.column<Uint32Array>('endTime'))
  };
}

/** Lightning flashes with the three frames the scenes need. */
export type StormFlashes = {
  count: number;
  /** Longitude and latitude degrees. */
  lngLat: Float32Array;
  /** Meters around `origin` in deck's local metric frame (the frame the density raster is drawn in). */
  local: Float32Array;
  /** Meters in the azimuthal-equidistant frame (areal joins). */
  aeqd: Float32Array;
  /** Seconds since the storm origin. */
  times: Float32Array;
  /** Optical energy of each flash, femtojoules. */
  energy: Float32Array;
  origin: readonly [number, number];
  project: (longitude: number, latitude: number) => [number, number];
  /** Fraction of all flashes in the window that the sample holds. */
  sampleFraction: number;
  flashesInWindow: number;
};

/** Loads `poopdeck-goes-glm-lightning`. */
export function loadStormFlashes(dataset: LoadedDataset): StormFlashes {
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const lngLat = dataset.column<Float32Array>('position');
  const aeqd = new Float32Array(lngLat.length);
  for (let index = 0; index < lngLat.length; index += 2) {
    const [x, y] = STORM_FRAME.project(lngLat[index], lngLat[index + 1]);
    aeqd[index] = x;
    aeqd[index + 1] = y;
  }
  const properties = dataset.properties as {sampleFraction?: number; flashesInWindow?: number};
  return {
    count: dataset.count,
    lngLat,
    local: dataset.projectColumn('position', origin),
    aeqd,
    times: Float32Array.from(dataset.column<Uint32Array>('timestamp')),
    energy: dataset.column<Float32Array>('energy'),
    origin,
    project: (longitude, latitude) => projection.project(longitude, latitude),
    sampleFraction: properties.sampleFraction ?? 1,
    flashesInWindow: properties.flashesInWindow ?? dataset.count
  };
}

/** Projects interleaved longitude and latitude degrees into the azimuthal-equidistant storm frame. */
export function projectToStormFrame(lngLat: ArrayLike<number>): Float32Array {
  const meters = new Float32Array(lngLat.length);
  for (let index = 0; index < lngLat.length; index += 2) {
    const [x, y] = STORM_FRAME.project(lngLat[index], lngLat[index + 1]);
    meters[index] = x;
    meters[index + 1] = y;
  }
  return meters;
}

/** Counts `values` into `bins` equal bins over `[low, high]`; out-of-range values go to the end bins. */
export function binValues(
  values: ArrayLike<number>,
  low: number,
  high: number,
  bins: number,
  count = values.length
): Float64Array {
  const counts = new Float64Array(bins);
  const scale = bins / (high - low);
  for (let index = 0; index < count; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    counts[Math.min(bins - 1, Math.max(0, Math.floor((value - low) * scale)))]++;
  }
  return counts;
}

/** Histogram chart over `[low, high]`. */
export function histogramChart(
  counts: ArrayLike<number>,
  low: number,
  high: number,
  options: Partial<BarChartData> & {xLabel: string; yLabel?: string}
): BarChartData {
  return {
    kind: 'histogram',
    values: counts,
    xDomain: [low, high],
    height: 120,
    yLabel: 'count',
    ...options
  };
}

/** Line chart with one or more series. */
export function seriesChart(
  series: LineChartData['series'],
  options: Partial<LineChartData> & {xLabel: string; yLabel?: string}
): LineChartData {
  return {kind: 'line', series, height: 120, ...options};
}

/** `1,234` style integer. */
export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

/** Nearest-rank quantile of the first `count` finite values. */
export function quantile(values: ArrayLike<number>, fraction: number, count = values.length) {
  const finite: number[] = [];
  for (let index = 0; index < count; index++)
    if (Number.isFinite(values[index])) finite.push(values[index]);
  finite.sort((a, b) => a - b);
  if (!finite.length) return Number.NaN;
  return finite[Math.min(finite.length - 1, Math.floor(fraction * finite.length))];
}

/** Compass bearing in degrees (0 north, clockwise) of a planar heading in radians (0 east, counterclockwise). */
export function headingToCompass(heading: number): number {
  const degrees = 90 - (heading * 180) / Math.PI;
  return ((degrees % 360) + 360) % 360;
}

/** Compass sector index (0 north, clockwise, 8 sectors) of a planar heading in radians. */
export function headingToSector(heading: number): number {
  return Math.floor((headingToCompass(heading) + 22.5) / 45) % 8;
}

/** Metric ids in the order of the compose kernel's `mode`. */
export const METRIC_MODES = [
  'outageNow',
  'outagePeak',
  'outageCount',
  'trackKm',
  'flashDensity',
  'meanDbz'
] as const;
/** Legend title and unit of each metric. */
export const METRIC_LABELS: Record<(typeof METRIC_MODES)[number], {title: string; unit: string}> = {
  outageNow: {title: 'Customers without power now', unit: 'per 1,000 residents'},
  outagePeak: {title: 'Peak customers without power', unit: 'per 1,000 residents'},
  outageCount: {title: 'Customers without power now', unit: 'customers'},
  trackKm: {title: 'Storm-cell track inside the county', unit: 'km'},
  flashDensity: {title: 'Lightning flashes (estimated)', unit: 'per 1,000 km2'},
  meanDbz: {title: 'Mean peak reflectivity of cells that crossed', unit: 'dBZ'}
};
