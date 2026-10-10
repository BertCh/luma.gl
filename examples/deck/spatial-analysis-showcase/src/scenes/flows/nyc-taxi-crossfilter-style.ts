// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Ink, class tables and place frames of the nyc-taxi-crossfilter story (design sheet
 * `FID/design/nyc-taxi-crossfilter.md`). Everything here is pure TypeScript: the class breaks are
 * computed ONCE from the loaded trips and never change with a brush, so a colour change on the map
 * is always a data change.
 */

import {formatBreakLabels, getClassBreaks, getClassCounts} from '../../cartography/breaks';
import {makeClassTable} from '../../cartography/class-table';
import {NYC} from '../../cartography/gazetteer';
import type {ClassTable} from '../../cartography/types';
import {getClassColors, sampleRamp, type PaletteColor} from '../../engine/ramps';
import {AREA_PRESETS, type TaxiTrips} from './nyc-taxi-data';

/** Number of fare and distance classes. */
export const TAXI_CLASS_COUNT = 6;

/** Ramp of the classed attributes on the night ground, trimmed so the lowest class is not black. */
export const TAXI_CLASS_RAMP = 'inferno' as const;
export const TAXI_CLASS_RAMP_RANGE = [0.3, 1] as const;

/** Ghost ink: rows a brush removed, never a ramp colour (design sheet: about `#5d6a80` at 0.12). */
export const GHOST_INK: PaletteColor = [93, 106, 128, 255];
export const GHOST_ALPHA = 31;

/** Trip links: straight pickup-to-dropoff segments, neutral and faint. */
export const LINK_INK: PaletteColor = [214, 221, 234, 26];

/** Ordinal party sizes: three rising greys, then orange tones so groups of four or more pop. */
export const PASSENGER_COLORS: readonly PaletteColor[] = [
  [118, 128, 145, 255],
  [150, 160, 176, 255],
  [186, 194, 207, 255],
  [255, 194, 71, 255],
  [245, 154, 35, 255],
  [232, 102, 28, 255]
];

/** Labels of the passenger classes, one per party size. */
export const PASSENGER_LABELS = ['1', '2', '3', '4', '5', '6'] as const;

/** Names of the map rectangles that are brushes. */
export type TaxiAreaId = keyof typeof AREA_PRESETS;

/** Gazetteer place that centres each area preset (no typed coordinates). */
const AREA_PLACES: Record<TaxiAreaId, string> = {
  midtown: 'midtown',
  downtown: 'lower-manhattan',
  jfk: 'jfk',
  laguardia: 'lga'
};

/** `[longitude, latitude]` of a NYC gazetteer place. */
export function getPlaceCenter(placeId: string): [number, number] {
  const place = NYC.places[placeId];
  return [place.lngLat[0], place.lngLat[1]];
}

/** `[west, south, east, north]` around NYC gazetteer places, padded by `padding` degrees. */
export function getPlacesBounds(
  placeIds: readonly string[],
  padding: number
): [number, number, number, number] {
  const points = placeIds.map(getPlaceCenter);
  return [
    Math.min(...points.map(point => point[0])) - padding,
    Math.min(...points.map(point => point[1])) - padding,
    Math.max(...points.map(point => point[0])) + padding,
    Math.max(...points.map(point => point[1])) + padding
  ];
}

/** Planar `[minX, minY, maxX, maxY]` meters of a map-area preset, centred on its gazetteer place. */
export function getAreaBoundsMeters(
  trips: TaxiTrips,
  area: TaxiAreaId
): [number, number, number, number] {
  const preset = AREA_PRESETS[area];
  const [longitude, latitude] = getPlaceCenter(AREA_PLACES[area]);
  const [centerX, centerY] = trips.project(longitude, latitude);
  return [
    centerX - preset.halfWidth,
    centerY - preset.halfHeight,
    centerX + preset.halfWidth,
    centerY + preset.halfHeight
  ];
}

/** Rounds a break up to the 0.5 USD step fares are stored in: the first fare that falls above it. */
function getEffectiveFare(value: number): number {
  return Math.ceil(value * 2 - 1e-9) / 2;
}

/** `$9`, `$5.50`: a break as the reader sees it on a legend or a tooltip. */
export function formatFare(value: number): string {
  const fare = getEffectiveFare(value);
  return `$${Number.isInteger(fare) ? fare : fare.toFixed(2)}`;
}

/** `1.5 mi`, `12 mi`. */
export function formatMiles(value: number): string {
  return `${Number.isInteger(value) ? value : value.toFixed(1)} mi`;
}

/** Cyclic hour-of-day colour (`romao`), `hour` modulo 24, so 00:00 and 24:00 meet. */
export function getHourColor(hour: number, alpha = 255): PaletteColor {
  const [red, green, blue] = sampleRamp('romao', (((hour % 24) + 24) % 24) / 24);
  return [red, green, blue, alpha];
}

/** Every class table of the story, built once. */
export type TaxiClassTables = {
  /** Six fare classes with the same number of trips each (breaks fixed at load). */
  fareQuantile: ClassTable;
  /** Six equal-width fare classes over the data range (breaks fixed at load). */
  fareEqual: ClassTable;
  /** Trips per class of {@link TaxiClassTables.fareQuantile}. */
  fareQuantileCounts: number[];
  /** Trips per class of {@link TaxiClassTables.fareEqual}. */
  fareEqualCounts: number[];
  /** Six distance classes (quantiles). */
  distance: ClassTable;
  /** Party sizes one to six, one class each. */
  passengers: ClassTable;
};

const FILTERED_OUT = {label: 'Filtered out by a brush', color: GHOST_INK} as const;

/**
 * Builds the class tables from the loaded trips: quantile and equal-interval fare classes, quantile
 * distance classes and the party-size classes. The breaks are read from the data here, once; no
 * brush, step or option changes them afterwards.
 */
export function buildTaxiClassTables(trips: TaxiTrips): TaxiClassTables {
  const getMaximum = (values: Float32Array) => {
    let maximum = 0;
    for (let row = 0; row < values.length; row++) maximum = Math.max(maximum, values[row]);
    return maximum;
  };
  const buildTable = (
    values: Float32Array,
    method: 'quantile' | 'equal-interval',
    unit: string,
    format: (value: number) => string
  ) => {
    const extent: [number, number] = [0, getMaximum(values)];
    const breaks = getClassBreaks(values, TAXI_CLASS_COUNT, method);
    const table = makeClassTable({
      breaks,
      colors: getClassColors(TAXI_CLASS_RAMP, breaks.length + 1, false, 255, TAXI_CLASS_RAMP_RANGE),
      unit,
      format,
      extent,
      labels: formatBreakLabels(breaks, extent, format),
      method: method === 'quantile' ? 'Quantiles, fixed at load' : 'Equal intervals, fixed at load',
      noData: FILTERED_OUT
    });
    return {table, counts: getClassCounts(values, breaks)};
  };
  const fareQuantile = buildTable(trips.fare, 'quantile', 'USD', formatFare);
  const fareEqual = buildTable(trips.fare, 'equal-interval', 'USD', formatFare);
  const distance = buildTable(trips.distance, 'quantile', 'miles', formatMiles);
  const passengers = makeClassTable({
    breaks: [2, 3, 4, 5, 6],
    colors: [...PASSENGER_COLORS],
    labels: [...PASSENGER_LABELS],
    unit: 'passengers',
    method: 'One class per party size',
    noData: FILTERED_OUT
  });
  return {
    fareQuantile: fareQuantile.table,
    fareEqual: fareEqual.table,
    fareQuantileCounts: fareQuantile.counts,
    fareEqualCounts: fareEqual.counts,
    distance: distance.table,
    passengers
  };
}
