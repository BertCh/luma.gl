// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The figure of the migration-timing story: a Hovmöller panel drawn on the map at sea west of
 * Africa. Columns are time buckets of the folded year (longitude), rows are 3-degree latitude
 * bands at their true latitudes, and the colour is the share, count or fastest step of the cell.
 *
 * This file holds the pure parts: the panel geometry, the class tables (one table per cell value,
 * read by the layer, the legend and the tooltip), the axes as map annotations and the segment rows
 * of the overlays. The GPU work and the layers are in `migration-timing.compute.ts`.
 */

import {getClassPalette, makeClassTable} from '../../cartography/class-table';
import {type MapGround, MAP_INK} from '../../cartography/hue-registry';
import type {ClassTable, MapAnnotation} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';
import {formatCount} from '../../cartography/live-text';
import {getBirdDensityClasses, FOLDED_MONTH_TICKS} from './movement-style';
import {DAYS_IN_YEAR} from './migration-shared';
import {
  getRowLatitude,
  getValidBuckets,
  LATITUDE_START,
  ROWS_PER_GROUP
} from './migration-timing-stats';

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

/** Longitude of day 0 (the west edge of the matrix). */
export const CHART_WEST = -52;
/** Degrees of longitude one folded year spans. */
export const CHART_WIDTH_DEGREES = 30;
/**
 * The paper panel behind the matrix. It leaves room on the west for the latitude labels, on the
 * south for the month ruler and its caption, on the north for the title, and on the east for the
 * latitude-group key.
 */
export const PANEL = {west: -60, south: -4, east: -20, north: 65} as const;
/**
 * Where the probe line starts on the Atlantic side of the coast, in degrees of longitude: far
 * enough east that the line crosses the strip between the panel and the tracks and ends on the
 * panel's row.
 */
export const PROBE_COAST_LONGITUDE = -10;
/** Longitude of the latitude-group key (a thin coloured ruler beside the matrix). */
export const GROUP_KEY_LONGITUDE = PANEL.east - 0.9;
/** Latitude of the bracket that carries the x-axis caption, below the month ruler. */
const CAPTION_LATITUDE = LATITUDE_START - 5.5;

/** Longitude of a day of the folded year (0-based, fractions allowed). */
export function dayToLongitude(day: number): number {
  return CHART_WEST + (CHART_WIDTH_DEGREES * day) / DAYS_IN_YEAR;
}

/** Day of the folded year at a longitude of the panel. */
export function longitudeToDay(longitude: number): number {
  return ((longitude - CHART_WEST) * DAYS_IN_YEAR) / CHART_WIDTH_DEGREES;
}

/** Width of one bucket column in degrees of longitude. */
export function getColumnDegrees(bucketDays: number): number {
  return (CHART_WIDTH_DEGREES * bucketDays) / DAYS_IN_YEAR;
}

/**
 * The extent of the drawn matrix: only the valid buckets (a 7-day width holds the year in 53
 * columns), so the raster stops where the data stops instead of tinting empty columns.
 */
export function getMatrixBounds(
  bucketDays: number,
  rowCount: number
): [west: number, south: number, east: number, north: number] {
  return [
    CHART_WEST,
    LATITUDE_START,
    CHART_WEST + getValidBuckets(bucketDays) * getColumnDegrees(bucketDays),
    getRowLatitude(rowCount)
  ];
}

/** The panel as a raster extent `[west, south, east, north]`. */
export function getPanelBounds(): [number, number, number, number] {
  return [PANEL.west, PANEL.south, PANEL.east, PANEL.north];
}

// ---------------------------------------------------------------------------------------------
// Inks
// ---------------------------------------------------------------------------------------------

/** The paper of the panel (the backing carries the translucency; the matrix above is opaque). */
export function getPanelFill(ground: MapGround): PaletteColor {
  return ground === 'dark' ? [20, 24, 28, 235] : [255, 255, 255, 235];
}

/** Ink, ground halo, rule and signal colours of the overlays for a ground. */
export function getOverlayInks(ground: MapGround): {
  ink: PaletteColor;
  halo: PaletteColor;
  rule: PaletteColor;
  hairline: PaletteColor;
  signal: PaletteColor;
  hatch: PaletteColor;
} {
  const tokens = MAP_INK[ground];
  const channels = (hex: string, alpha: number): PaletteColor => [
    Number.parseInt(hex.slice(1, 3), 16),
    Number.parseInt(hex.slice(3, 5), 16),
    Number.parseInt(hex.slice(5, 7), 16),
    alpha
  ];
  return {
    ink: channels(tokens.ink, 255),
    halo: channels(tokens.halo, 235),
    rule: channels(tokens.rule, Math.round(tokens.ruleAlpha * 255)),
    hairline: channels(tokens.ink, 46),
    signal: channels(tokens.signal, 255),
    hatch: ground === 'dark' ? [214, 221, 234, 130] : [31, 41, 51, 120]
  };
}

// ---------------------------------------------------------------------------------------------
// Class tables
// ---------------------------------------------------------------------------------------------

/** What a cell shows. */
export type TimingMode = 'share' | 'count' | 'speed';

/** Share class breaks as fractions of the column (1, 2.5, 5, 10, 20 and 35 percent). */
export const SHARE_BREAKS: readonly number[] = [0.01, 0.025, 0.05, 0.1, 0.2, 0.35];
const SHARE_LABELS = ['< 1', '1-2.5', '2.5-5', '5-10', '10-20', '20-35', '35 or more'];
/** Fastest-step class breaks in km/h. */
export const SPEED_BREAKS: readonly number[] = [10, 20, 35, 50];
const SPEED_LABELS = ['< 10', '10-20', '20-35', '35-50', '50 or more'];

/**
 * The seven share and count colours: the registry's `nature` row (YlGnBu on paper, inferno on
 * dark). On paper the published 7-class table starts at a near-white yellow that vanishes on the
 * white panel, so the lightest class of the 8-class table is dropped: the lowest class stays
 * visible and empty cells (transparent) read as paper.
 */
export function getShareColors(ground: MapGround): PaletteColor[] {
  return ground === 'dark'
    ? getBirdDensityClasses(ground, 7)
    : getClassPalette('YlGnBu', 8, {ground}).slice(1);
}

/**
 * The five fastest-step colours. OrRd (warm, intensity) on purpose: share and count are YlGnBu, so
 * the two kinds of cell can never be mistaken for each other. On paper the lightest class of the
 * 6-class table is dropped for the same reason as in {@link getShareColors}.
 */
export function getSpeedColors(ground: MapGround): PaletteColor[] {
  return ground === 'dark'
    ? getClassPalette('OrRd', 5, {ground})
    : getClassPalette('OrRd', 6, {ground}).slice(1);
}

/** The five 12-degree group colours of the band chart and the group key, south to north. */
export function getGroupColors(ground: MapGround): PaletteColor[] {
  // The upper five of the 7 share colours: distinct from each other and visible on the panel.
  return getShareColors(ground).slice(2);
}

/** `count` colours spread over a seven-class palette (every class when `count` is 7). */
function spreadColors(palette: readonly PaletteColor[], count: number): PaletteColor[] {
  if (count >= palette.length) return [...palette];
  if (count <= 1) return [palette[palette.length - 1]];
  return Array.from(
    {length: count},
    (_, index) => palette[Math.round((index * (palette.length - 1)) / (count - 1))]
  );
}

/**
 * The one class table of a cell value, read by the matrix layer, the legend and the tooltip.
 * Share and fastest-step breaks are fixed; count breaks are the quantiles of the occupied cells
 * of the current selection (`countBreaks`), so the classes hold similar numbers of cells.
 */
export function getTimingTable(
  mode: TimingMode,
  ground: MapGround,
  countBreaks: readonly number[]
): ClassTable {
  const noData = {label: 'No fixes in the cell'};
  if (mode === 'share') {
    return makeClassTable({
      breaks: SHARE_BREAKS,
      colors: getShareColors(ground),
      labels: SHARE_LABELS,
      unit: '%',
      extent: [0, 1],
      method: "Fixed breaks. Each column of cells adds up to 100 % of that week's fixes.",
      noData
    });
  }
  if (mode === 'speed') {
    return makeClassTable({
      breaks: SPEED_BREAKS,
      colors: getSpeedColors(ground),
      labels: SPEED_LABELS,
      unit: 'km/h',
      extent: [0, 100],
      method: 'Fixed breaks. The maximum of the slot, not a mean: the reduction keeps no sum.',
      noData
    });
  }
  const breaks = countBreaks.length > 0 ? countBreaks : [10, 25, 50, 100, 200, 400];
  return makeClassTable({
    breaks,
    colors: spreadColors(getShareColors(ground), breaks.length + 1),
    unit: 'fixes',
    extent: [0, Math.max(breaks[breaks.length - 1] * 2, 1)],
    format: formatCount,
    method: 'Quantiles of the occupied cells, refit when the species or the width changes.',
    noData
  });
}

// ---------------------------------------------------------------------------------------------
// Overlays: segment rows [x0, y0, x1, y1, ...]
// ---------------------------------------------------------------------------------------------

/** A flat list of segments `x0, y0, x1, y1`. */
export type SegmentRows = number[];

function rectangle(west: number, south: number, east: number, north: number): SegmentRows {
  return [
    west,
    south,
    east,
    south,
    east,
    south,
    east,
    north,
    east,
    north,
    west,
    north,
    west,
    north,
    west,
    south
  ];
}

/** The 0.8 px rule around the paper panel. */
export function getPanelFrameRows(): SegmentRows {
  return rectangle(PANEL.west, PANEL.south, PANEL.east, PANEL.north);
}

/**
 * Hairlines inside the matrix: a faint rule at every labelled latitude (10, 20, ... degrees north)
 * and the three quarter-year lines. Hairlines only where the reader needs them: not one per cell.
 */
export function getHairlineRows(bucketDays: number, rowCount: number): SegmentRows {
  const [west, south, east, north] = getMatrixBounds(bucketDays, rowCount);
  const rows: SegmentRows = [];
  for (let latitude = Math.ceil(south / 10) * 10; latitude < north; latitude += 10) {
    if (latitude > south) rows.push(west, latitude, east, latitude);
  }
  for (const month of [3, 6, 9]) {
    const x = dayToLongitude(FOLDED_MONTH_TICKS[month].at);
    rows.push(x, south, x, north);
  }
  return rows;
}

/** Vertical edges between the buckets, the faint grid of the "bucket width is a choice" step. */
export function getBucketEdgeRows(bucketDays: number, rowCount: number): SegmentRows {
  const [west, south, , north] = getMatrixBounds(bucketDays, rowCount);
  const rows: SegmentRows = [];
  for (let bucket = 1; bucket < getValidBuckets(bucketDays); bucket++) {
    const x = west + bucket * getColumnDegrees(bucketDays);
    rows.push(x, south, x, north);
  }
  return rows;
}

/** The outline of the row that holds the probe latitude. */
export function getProbeRowRows(bucketDays: number, rowCount: number, row: number): SegmentRows {
  const [west, , east] = getMatrixBounds(bucketDays, rowCount);
  return rectangle(west, getRowLatitude(row), east, getRowLatitude(row + 1));
}

/** The outline of one bucket column (the bucket under the playback cursor). */
export function getBucketColumnRows(
  bucketDays: number,
  rowCount: number,
  bucket: number
): SegmentRows {
  const [west, south, , north] = getMatrixBounds(bucketDays, rowCount);
  const x = west + bucket * getColumnDegrees(bucketDays);
  return rectangle(x, south, x + getColumnDegrees(bucketDays), north);
}

/** The playback cursor: one vertical line at a day of the folded year, over the whole matrix. */
export function getCursorRows(day: number, rowCount: number): SegmentRows {
  const x = dayToLongitude(day + 0.5);
  return [x, LATITUDE_START, x, getRowLatitude(rowCount)];
}

/** A polyline through the median latitudes of consecutive buckets, drawn as segments. */
export function getTraceRows(medians: ArrayLike<number>, bucketDays: number): SegmentRows {
  const rows: SegmentRows = [];
  for (let bucket = 1; bucket < medians.length; bucket++) {
    const before = medians[bucket - 1];
    const after = medians[bucket];
    if (!Number.isFinite(before) || !Number.isFinite(after)) continue;
    rows.push(
      dayToLongitude((bucket - 0.5) * bucketDays),
      before,
      dayToLongitude((bucket + 0.5) * bucketDays),
      after
    );
  }
  return rows;
}

/** The group key: one segment per 12-degree group on the east side of the matrix, south to north. */
export function getGroupKeyRows(rowCount: number): SegmentRows {
  const rows: SegmentRows = [];
  for (let first = 0; first < rowCount; first += ROWS_PER_GROUP) {
    const last = Math.min(rowCount, first + ROWS_PER_GROUP);
    rows.push(
      GROUP_KEY_LONGITUDE,
      getRowLatitude(first),
      GROUP_KEY_LONGITUDE,
      getRowLatitude(last)
    );
  }
  return rows;
}

// ---------------------------------------------------------------------------------------------
// Axes as annotations
// ---------------------------------------------------------------------------------------------

/**
 * The axes of the figure as `bracket` annotations (fixed kinds: never culled by the label
 * budget). A bracket of almost zero length is an 8 px tick with its label beyond it:
 *
 * - latitude ticks every 10 degrees on the west side, the top one with the `N`;
 * - one bracket per month under the matrix, labelled with the month's initial: a ruler;
 * - the title above the matrix and the x-axis caption below the ruler.
 */
export function getAxisAnnotations(title: string, rowCount: number): MapAnnotation[] {
  const north = getRowLatitude(rowCount);
  const annotations: MapAnnotation[] = [];
  const topTick = Math.floor(north / 10) * 10;
  for (let latitude = 10; latitude <= topTick; latitude += 10) {
    annotations.push({
      kind: 'bracket',
      id: `timing-latitude-${latitude}`,
      from: [CHART_WEST, latitude - 0.04],
      to: [CHART_WEST, latitude + 0.04],
      text: latitude === topTick ? `${latitude}° N` : `${latitude}°`,
      side: 'left'
    });
  }
  FOLDED_MONTH_TICKS.forEach((tick, index) => {
    const next = FOLDED_MONTH_TICKS[index + 1]?.at ?? DAYS_IN_YEAR;
    annotations.push({
      kind: 'bracket',
      id: `timing-month-${index}`,
      from: [dayToLongitude(tick.at), LATITUDE_START],
      to: [dayToLongitude(next), LATITUDE_START],
      text: tick.label,
      side: 'right'
    });
  });
  annotations.push(
    {
      kind: 'bracket',
      id: 'timing-title',
      from: [CHART_WEST, north],
      to: [dayToLongitude(DAYS_IN_YEAR), north],
      text: title,
      side: 'left',
      priority: 3
    },
    {
      kind: 'bracket',
      id: 'timing-x-caption',
      from: [CHART_WEST, CAPTION_LATITUDE],
      to: [dayToLongitude(DAYS_IN_YEAR), CAPTION_LATITUDE],
      text: 'Day of the folded year',
      side: 'right',
      tone: 'muted'
    }
  );
  return annotations;
}

/**
 * The probe: a dashed `--map-signal` line from the coast to the panel at the probe latitude with
 * its tag. Reads across the Atlantic to the row of the matrix, and no further.
 */
export function getProbeAnnotation(latitude: number): MapAnnotation {
  return {
    kind: 'line',
    id: 'timing-probe',
    coordinates: [
      [PROBE_COAST_LONGITUDE, latitude],
      [PANEL.east, latitude]
    ],
    text: `${Number.isInteger(latitude) ? latitude : latitude.toFixed(1)}° N`,
    dashed: true,
    widthPixels: 1.2,
    tone: 'signal'
  };
}

/** The date range of a bucket as `1-7 Sep` or `28 Aug-3 Sep` (day numbers of the folded year). */
export function formatBucketRange(bucket: number, bucketDays: number): string {
  const first = new Date(Date.UTC(2024, 0, 1) + bucket * bucketDays * 86400000);
  const last = new Date(
    Date.UTC(2024, 0, 1) + (Math.min(DAYS_IN_YEAR, (bucket + 1) * bucketDays) - 1) * 86400000
  );
  const month = (date: Date) =>
    new Intl.DateTimeFormat('en-GB', {month: 'short', timeZone: 'UTC'}).format(date);
  if (first.getUTCMonth() === last.getUTCMonth() && first.getUTCDate() !== last.getUTCDate()) {
    return `${first.getUTCDate()}-${last.getUTCDate()} ${month(first)}`;
  }
  if (first.getTime() === last.getTime()) return `${first.getUTCDate()} ${month(first)}`;
  return `${first.getUTCDate()} ${month(first)}-${last.getUTCDate()} ${month(last)}`;
}

/** Whether a longitude and latitude fall inside the matrix of the current width. */
export function isInsideMatrix(
  longitude: number,
  latitude: number,
  bucketDays: number,
  rowCount: number
): boolean {
  const [west, south, east, north] = getMatrixBounds(bucketDays, rowCount);
  return longitude >= west && longitude < east && latitude >= south && latitude < north;
}
