// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {formatNumber, type PolygonLayout} from './b3-common';

/**
 * Shared loading and statistics helpers of the wildfire scenes (`wildfire-shapes`,
 * `wildfire-seasons` and `raster/wildfire-terrain`): the `poopdeck-wildfires` dataset as a polygon
 * layout with one feature per fire, Web Mercator coordinates, the per-fire attributes and a few
 * small CPU statistics for the charts (medians, quantiles, rank correlation).
 */

/** Web Mercator sphere radius in meters. */
export const MERCATOR_RADIUS = 6378137;
/** Acres in one square meter. */
export const ACRES_PER_SQUARE_METER = 1 / 4046.8564224;
/** Days between 2020-01-01 and the end of 2023. */
export const WILDFIRE_DAY_COUNT = 1461;
/** Unix seconds of the dataset time origin (2020-01-01T00:00:00Z). */
export const WILDFIRE_TIME_ORIGIN_SECONDS = Date.UTC(2020, 0, 1) / 1000;

/** Names of the acreage classes (poopdeck calls the column `severity`; it is not burn severity). */
export const SIZE_CLASS_NAMES = [
  'under 10k acres',
  '11k to 33k acres',
  '50k to 97k acres',
  'over 300k acres'
] as const;

/** Year colors (2020 to 2023), readable on light and dark basemaps. */
export const YEAR_COLORS: readonly (readonly [number, number, number])[] = [
  [232, 90, 60],
  [240, 170, 50],
  [70, 190, 150],
  [110, 140, 235]
];

/** The wildfire layer: polygon layout plus per-fire attributes. */
export type WildfireData = {
  /** One feature per fire, parts as polygons; compatible with the Geometry chapter helpers. */
  layout: PolygonLayout;
  count: number;
  /** Interleaved Web Mercator meters of every vertex (absolute, `x, y`). */
  mercator: Float32Array;
  /** NIFC acres. */
  acres: Float32Array;
  /** Calendar year, 2020 to 2023. */
  year: Uint16Array;
  /** Index into 2020..2023. */
  yearIndex: Uint8Array;
  /** Days since 2020-01-01 of the NIFC perimeter date (float32 is exact for these values). */
  days: Float32Array;
  names: string[];
  sizeClass: Uint8Array;
  partCount: Uint16Array;
  /** Fire rows sorted by `days` (ties keep file order). */
  byDate: Uint32Array;
};

/** Longitude and latitude (degrees) to Web Mercator meters. */
export function toMercator(longitude: number, latitude: number): [number, number] {
  const clamped = Math.min(Math.max(latitude, -85), 85);
  return [
    (MERCATOR_RADIUS * longitude * Math.PI) / 180,
    MERCATOR_RADIUS * Math.log(Math.tan(Math.PI / 4 + (clamped * Math.PI) / 360))
  ];
}

/** Web Mercator meters to longitude and latitude (degrees). */
export function fromMercator(x: number, y: number): [number, number] {
  return [
    (x / MERCATOR_RADIUS) * (180 / Math.PI),
    (2 * Math.atan(Math.exp(y / MERCATOR_RADIUS)) - Math.PI / 2) * (180 / Math.PI)
  ];
}

/** Reads `poopdeck-wildfires` into a {@link WildfireData}. */
export function loadWildfires(dataset: LoadedDataset): WildfireData {
  const lngLat = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const featureOffsets = dataset.column<Uint32Array>('featurePolygonOffsets');
  const count = featureOffsets.length - 1;
  const featureRingOffsets = new Uint32Array(count + 1);
  const ringFeature = new Uint32Array(ringOffsets.length - 1);
  const featureBounds = new Float32Array(count * 4);
  for (let fire = 0; fire <= count; fire++) {
    featureRingOffsets[fire] = polygonOffsets[featureOffsets[fire]];
  }
  for (let fire = 0; fire < count; fire++) {
    let west = Infinity;
    let south = Infinity;
    let east = -Infinity;
    let north = -Infinity;
    for (let ring = featureRingOffsets[fire]; ring < featureRingOffsets[fire + 1]; ring++) {
      ringFeature[ring] = fire;
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
        west = Math.min(west, lngLat[vertex * 2]);
        east = Math.max(east, lngLat[vertex * 2]);
        south = Math.min(south, lngLat[vertex * 2 + 1]);
        north = Math.max(north, lngLat[vertex * 2 + 1]);
      }
    }
    featureBounds.set([west, south, east, north], fire * 4);
  }
  const vertexCount = lngLat.length / 2;
  const mercator = new Float32Array(vertexCount * 2);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const [x, y] = toMercator(lngLat[vertex * 2], lngLat[vertex * 2 + 1]);
    mercator[vertex * 2] = x;
    mercator[vertex * 2 + 1] = y;
  }
  const year = dataset.column<Uint16Array>('year');
  const yearIndex = new Uint8Array(count);
  const seconds = dataset.column<Uint32Array>('perimeterTime');
  const days = new Float32Array(count);
  for (let fire = 0; fire < count; fire++) {
    yearIndex[fire] = Math.min(3, Math.max(0, year[fire] - 2020));
    days[fire] = Math.floor(seconds[fire] / 86400);
  }
  const nameCodes = dataset.column<Uint8Array>('name');
  const categories = dataset.categories('name');
  const names = Array.from(nameCodes, code => String(categories[code] ?? ''));
  const byDate = Uint32Array.from({length: count}, (_, index) => index).sort(
    (a, b) => days[a] - days[b] || a - b
  );
  return {
    layout: {
      featureCount: count,
      partCount: polygonOffsets.length - 1,
      ringCount: ringOffsets.length - 1,
      vertexCount,
      lngLat,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      featureRingOffsets,
      ringFeature,
      featureBounds
    },
    count,
    mercator,
    acres: dataset.column<Float32Array>('acres'),
    year,
    yearIndex,
    days,
    names,
    sizeClass: dataset.column<Uint8Array>('sizeClass'),
    partCount: dataset.column<Uint16Array>('partCount'),
    byDate
  };
}

/** Formats days since 2020-01-01 as `12 Jun 2021`. */
export function formatWildfireDay(days: number): string {
  const date = new Date((WILDFIRE_TIME_ORIGIN_SECONDS + Math.round(days) * 86400) * 1000);
  const month = date.toLocaleString('en-GB', {month: 'short', timeZone: 'UTC'});
  return `${date.getUTCDate()} ${month} ${date.getUTCFullYear()}`;
}

/** One-line description of a fire for tooltips. */
export function describeFire(data: WildfireData, fire: number): string {
  return `${data.names[fire]} (${data.year[fire]}): ${formatNumber(data.acres[fire])} acres, perimeter dated ${formatWildfireDay(data.days[fire])}`;
}

/** Interpolated quantile of finite values (sorts a copy). `NaN` for an empty input. */
export function getFiniteQuantile(values: ArrayLike<number>, quantile: number): number {
  const finite = Array.from(values as ArrayLike<number>).filter(value => Number.isFinite(value));
  if (!finite.length) return Number.NaN;
  finite.sort((a, b) => a - b);
  const position = Math.min(1, Math.max(0, quantile)) * (finite.length - 1);
  const low = Math.floor(position);
  const high = Math.ceil(position);
  return finite[low] + (finite[high] - finite[low]) * (position - low);
}

/** Spearman rank correlation of two equally long columns; pairs with a non-finite value are skipped. */
export function getSpearmanCorrelation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const pairs: [number, number][] = [];
  for (let index = 0; index < a.length; index++) {
    if (Number.isFinite(a[index]) && Number.isFinite(b[index])) pairs.push([a[index], b[index]]);
  }
  if (pairs.length < 3) return Number.NaN;
  const rank = (selector: (pair: [number, number]) => number) => {
    const order = pairs
      .map((_, index) => index)
      .sort((x, y) => selector(pairs[x]) - selector(pairs[y]));
    const ranks = new Float64Array(pairs.length);
    for (let position = 0; position < order.length; ) {
      let end = position;
      while (
        end + 1 < order.length &&
        selector(pairs[order[end + 1]]) === selector(pairs[order[position]])
      ) {
        end++;
      }
      const average = (position + end) / 2;
      for (let k = position; k <= end; k++) ranks[order[k]] = average;
      position = end + 1;
    }
    return ranks;
  };
  const rankA = rank(pair => pair[0]);
  const rankB = rank(pair => pair[1]);
  const mean = (pairs.length - 1) / 2;
  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (let index = 0; index < pairs.length; index++) {
    covariance += (rankA[index] - mean) * (rankB[index] - mean);
    varianceA += (rankA[index] - mean) ** 2;
    varianceB += (rankB[index] - mean) ** 2;
  }
  return covariance / Math.sqrt(varianceA * varianceB);
}

/** Pearson-free helper: formats a correlation as `+0.42` (or `n/a`). */
export function formatCorrelation(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
}
