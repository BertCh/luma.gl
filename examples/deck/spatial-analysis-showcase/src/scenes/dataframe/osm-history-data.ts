// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import type {LoadedDataset} from '../../data/catalog';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {LocalMetricProjection} from '../../engine/projection';
import type {SpatialAnalysisColor} from '../../engine/layers';

/** Dataset id shared by the `osm-history-*` scenes. */
export const OSM_DATASET_ID = 'poopdeck-osm-nyc';

/** Attribution that every OSM history scene shows in its about text and readouts. */
export const OSM_ATTRIBUTION = '© OpenStreetMap contributors';

/** Kind codes in the order of the dataset's `kind` categories. */
export const OSM_KINDS = ['land', 'other', 'transport', 'infra', 'poi', 'building'] as const;

/** Human labels of the kinds (the grouping is poopdeck.gl's, derived from node tags). */
export const OSM_KIND_LABELS: Record<(typeof OSM_KINDS)[number], string> = {
  land: 'Land and water',
  other: 'Other tagged nodes',
  transport: 'Transport',
  infra: 'Infrastructure',
  poi: 'Places and amenities',
  building: 'Buildings'
};

/** Categorical palette (colour-blind safe hues), one per kind, readable on light and dark maps. */
export const OSM_KIND_COLORS: readonly SpatialAnalysisColor[] = [
  [86, 190, 120, 235],
  [160, 160, 185, 235],
  [240, 165, 60, 235],
  [205, 105, 220, 235],
  [80, 170, 245, 235],
  [240, 95, 95, 235]
];

const MILLISECONDS_PER_DAY = 86_400_000;
const MONTH_NAMES = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
];

/** Full-history aggregates stored in the manifest (exact, not from the sample). */
export type OsmFullHistory = {
  fullCount: number;
  sampleFraction: number;
  contributorCount: number;
  gini: number;
  /** Cumulative share of all nodes made by the top 1..200 contributors. */
  topShares: readonly number[];
  monthCount: number;
  monthTotal: readonly number[];
  monthByKind: Record<string, readonly number[]>;
  /** Share of each month's nodes made by that month's most active contributor. */
  monthTopShare: readonly number[];
  kindTotals: Record<string, number>;
};

/** The sample of OSM New York node creations, as arrays ready to upload. */
export type OsmHistory = {
  count: number;
  origin: [number, number];
  projection: LocalMetricProjection;
  /** Planar meters around `origin`, interleaved x, y. */
  positions: Float32Array;
  /** Creation time in days since `timeOriginMs` (float32, exact to about 40 s). */
  days: Float32Array;
  /** Kind code per node (uint32 for storage buffers). */
  kind: Uint32Array;
  /** Anonymous contributor rank per node, 0 = most active over the full history. */
  contributor: Uint32Array;
  timeOriginMs: number;
  /** Day of the last node (about 6,900). */
  maxDay: number;
  /** `[minX, minY, maxX, maxY]` meters of the sample. */
  bounds: [number, number, number, number];
  full: OsmFullHistory;
  /** Day (since the origin) at which each calendar month starts, `monthCount + 1` entries. */
  monthEdges: Float32Array;
};

/** Reads the dataset into upload-ready arrays (days as float32, kind and rank widened to uint32). */
export function loadOsmHistory(dataset: LoadedDataset): OsmHistory {
  const count = dataset.count;
  const props = dataset.properties;
  const timeOriginMs = props.timeOriginMs as number;
  const seconds = dataset.column<Uint32Array>('timestamp');
  const days = new Float32Array(count);
  for (let i = 0; i < count; i++) days[i] = seconds[i] / 86400;
  const kindSource = dataset.column<Uint8Array>('kind');
  const contributorSource = dataset.column<Uint16Array>('contributor');
  const kind = Uint32Array.from(kindSource);
  const contributor = Uint32Array.from(contributorSource);
  const positions = dataset.projectColumn('position');
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = positions[i * 2];
    const y = positions[i * 2 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const month = props.month as {
    count: number;
    total: number[];
    byKind: Record<string, number[]>;
    topContributorShare: number[];
  };
  const monthEdges = new Float32Array(month.count + 1);
  const origin = new Date(timeOriginMs);
  // Month 0 is the month of the origin (June 2007); edges are the first of each month, UTC.
  for (let m = 0; m <= month.count; m++) {
    const start = Date.UTC(origin.getUTCFullYear(), origin.getUTCMonth() + m, 1);
    monthEdges[m] = Math.max(0, (start - timeOriginMs) / MILLISECONDS_PER_DAY);
  }
  return {
    count,
    origin: dataset.defaultOrigin,
    projection: dataset.getProjection(),
    positions,
    days,
    kind,
    contributor,
    timeOriginMs,
    maxDay: days[count - 1],
    bounds: [minX, minY, maxX, maxY],
    monthEdges,
    full: {
      fullCount: props.fullCount as number,
      sampleFraction: props.sampleFraction as number,
      contributorCount: props.contributorCount as number,
      gini: props.giniNodesPerContributor as number,
      topShares: props.topShares as number[],
      monthCount: month.count,
      monthTotal: month.total,
      monthByKind: month.byKind,
      monthTopShare: month.topContributorShare,
      kindTotals: props.kindTotals as Record<string, number>
    }
  };
}

/** Decimal year of a day since the origin, for chart axes (2013.85). */
export function dayToYear(history: Pick<OsmHistory, 'timeOriginMs'>, day: number): number {
  const date = new Date(history.timeOriginMs + day * MILLISECONDS_PER_DAY);
  const yearStart = Date.UTC(date.getUTCFullYear(), 0, 1);
  const yearLength = Date.UTC(date.getUTCFullYear() + 1, 0, 1) - yearStart;
  return date.getUTCFullYear() + (date.getTime() - yearStart) / yearLength;
}

/** Day since the origin of a decimal year. */
export function yearToDay(history: Pick<OsmHistory, 'timeOriginMs'>, year: number): number {
  const whole = Math.floor(year);
  const start = Date.UTC(whole, 0, 1);
  const length = Date.UTC(whole + 1, 0, 1) - start;
  return (start + (year - whole) * length - history.timeOriginMs) / MILLISECONDS_PER_DAY;
}

/** `Sep 2007` for a day since the origin. */
export function formatMonthYear(history: Pick<OsmHistory, 'timeOriginMs'>, day: number): string {
  const date = new Date(history.timeOriginMs + day * MILLISECONDS_PER_DAY);
  return `${MONTH_NAMES[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}

/** `12 Sep 2007` for a day since the origin. */
export function formatDate(history: Pick<OsmHistory, 'timeOriginMs'>, day: number): string {
  const date = new Date(history.timeOriginMs + day * MILLISECONDS_PER_DAY);
  return `${date.getUTCDate()} ${MONTH_NAMES[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}

/** `Sep 2007` for a month index (0 = the origin's month). */
export function formatMonthIndex(history: Pick<OsmHistory, 'timeOriginMs'>, month: number): string {
  const origin = new Date(history.timeOriginMs);
  const date = new Date(Date.UTC(origin.getUTCFullYear(), origin.getUTCMonth() + month, 1));
  return `${MONTH_NAMES[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}

/** Share formatted as a percentage with one decimal. */
export function formatShare(value: number): string {
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

/** Returns an importer that adds a buffer to `graph` once and returns the same view afterwards. */
export function createViewImporter(graph: GPUCommandGraph<void>, prefix: string) {
  const imported = new Map<string, GraphDataView<GPUVectorFormat>>();
  return <Format extends GPUVectorFormat>(
    name: string,
    buffer: Buffer,
    format: Format,
    length?: number
  ): GraphDataView<Format> => {
    let view = imported.get(name);
    if (!view) {
      view = importGraphBuffer(graph, `${prefix}-${name}`, buffer, format, length);
      imported.set(name, view);
    }
    return view as GraphDataView<Format>;
  };
}

/** Sample counts scaled to a full-history estimate, as an integer. */
export function scaleToFull(history: OsmHistory, sampleCount: number): number {
  return Math.round(sampleCount / history.full.sampleFraction);
}
