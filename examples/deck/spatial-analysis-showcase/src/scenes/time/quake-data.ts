// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {AnalysisFrame} from '../../engine/analysis-frame';
import {LocalMetricProjection} from '../../engine/projection';
import {createAzimuthalEquidistant} from '../movement/b12-tracks';
import {
  BOUNDARY_FAMILY_OF_CLASS,
  QUAKE_DAY_ZERO_MS,
  QUAKE_REGIONS,
  type QuakeRegionId
} from './quake-regions';

/**
 * Largest local distance scale error of the analysis frame over any of the regions: an
 * azimuthal equidistant frame about the window center is exact along radii from the center and
 * off by at most +2.2% across them (computed at the window corners of all five regions).
 */
export const QUAKE_ANALYSIS_SCALE_ERROR = 0.022;

/** Name of the analysis frame, for help and readout text. */
export const QUAKE_ANALYSIS_FRAME_NAME = 'azimuthal equidistant about the window center';

/** A planar frame: `[longitude, latitude]` degrees to meters and back. */
export type QuakePlanarFrame = {
  project: (longitude: number, latitude: number) => [number, number];
  unproject: (x: number, y: number) => [number, number];
};

/**
 * Events of one region as the typed arrays the contributors read.
 *
 * The ANALYSIS frame (azimuthal equidistant about the window center `origin`, at most 2.2%
 * distance error) feeds every operator: `positions`, `analysisWindowMeters` and the boundary
 * step segments. Layers draw it directly with `analysisFrame` (the engine inverts the frame on
 * the GPU). The MERCATOR-LOCAL frame (deck.gl `METER_OFFSETS` around `origin`, -10%..+19% off in
 * these windows) remains only for the aftershocks month-by-cell grid, which is binned and drawn
 * in it: `mercatorPositions`, `windowMeters`, `projection`.
 */
export type QuakeEvents = {
  count: number;
  /** `[longitude, latitude]` of the window center, the origin of both frames. */
  origin: [number, number];
  /** Mercator-local frame (`METER_OFFSETS` meters around `origin`). */
  projection: LocalMetricProjection;
  /** The analysis frame, for the `analysisFrame` prop of the layers that draw its meters. */
  analysisFrame: AnalysisFrame;
  /** Projection of the analysis frame (azimuthal equidistant meters around `origin`). */
  analysisProjection: QuakePlanarFrame;
  /** Analysis meters around `origin`, `x, y` per event. */
  positions: Float32Array;
  /** Mercator-local meters around `origin`, `x, y` per event (the aftershocks grid only). */
  mercatorPositions: Float32Array;
  /** Event time in days since 2020-01-01 00:00 UTC, ascending. */
  days: Float32Array;
  magnitude: Float32Array;
  /** Depth in km (negative values from the catalog are clamped to 0). */
  depth: Float32Array;
  /** `[west, south, east, north]` of the window in Mercator-local meters. */
  windowMeters: [number, number, number, number];
  /** `[west, south, east, north]` of the window in analysis meters. */
  analysisWindowMeters: [number, number, number, number];
};

/** Metric window of a region: the extremes of its edge points in one planar frame. */
function getWindowMeters(
  projection: Pick<QuakePlanarFrame, 'project'>,
  bbox: readonly [number, number, number, number]
): [number, number, number, number] {
  const [west, south, east, north] = bbox;
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let step = 0; step <= 8; step++) {
    const longitude = west + ((east - west) * step) / 8;
    const latitude = south + ((north - south) * step) / 8;
    for (const [lng, lat] of [
      [longitude, south],
      [longitude, north],
      [west, latitude],
      [east, latitude]
    ]) {
      const [x, y] = projection.project(lng, lat);
      minimumX = Math.min(minimumX, x);
      minimumY = Math.min(minimumY, y);
      maximumX = Math.max(maximumX, x);
      maximumY = Math.max(maximumY, y);
    }
  }
  return [minimumX, minimumY, maximumX, maximumY];
}

/**
 * Reads the catalog rows of type `earthquake` inside a region with magnitude at or above
 * `minimumMagnitude`, in time order, projected to analysis and Mercator-local meters around the
 * window center.
 */
export function readQuakeEvents(
  dataset: LoadedDataset,
  regionId: QuakeRegionId,
  minimumMagnitude: number
): QuakeEvents {
  const region = QUAKE_REGIONS[regionId];
  const [west, south, east, north] = region.bbox;
  const origin: [number, number] = [(west + east) / 2, (south + north) / 2];
  const projection = new LocalMetricProjection(origin);
  const analysisProjection = createAzimuthalEquidistant(origin);
  const lonLat = dataset.column<Float32Array>('position');
  const seconds = dataset.column<Uint32Array>('timestamp');
  const magnitudes = dataset.column<Float32Array>('magnitude');
  const depths = dataset.column<Float32Array>('depth');
  const types = dataset.column<Uint8Array>('type');
  const timeOriginMs = Number(dataset.properties.timeOriginMs ?? QUAKE_DAY_ZERO_MS);
  const dayOffset = (timeOriginMs - QUAKE_DAY_ZERO_MS) / 86400000;
  const kept: number[] = [];
  for (let row = 0; row < seconds.length; row++) {
    const longitude = lonLat[row * 2];
    const latitude = lonLat[row * 2 + 1];
    if (
      types[row] === 0 &&
      magnitudes[row] >= minimumMagnitude - 1e-4 &&
      longitude >= west &&
      longitude <= east &&
      latitude >= south &&
      latitude <= north
    ) {
      kept.push(row);
    }
  }
  const count = kept.length;
  const events: QuakeEvents = {
    count,
    origin,
    projection,
    analysisFrame: {kind: 'azimuthal-equidistant', origin},
    analysisProjection,
    positions: new Float32Array(count * 2),
    mercatorPositions: new Float32Array(count * 2),
    days: new Float32Array(count),
    magnitude: new Float32Array(count),
    depth: new Float32Array(count),
    windowMeters: getWindowMeters(projection, region.bbox),
    analysisWindowMeters: getWindowMeters(analysisProjection, region.bbox)
  };
  kept.forEach((row, index) => {
    const [x, y] = analysisProjection.project(lonLat[row * 2], lonLat[row * 2 + 1]);
    events.positions[index * 2] = x;
    events.positions[index * 2 + 1] = y;
    const [mercatorX, mercatorY] = projection.project(lonLat[row * 2], lonLat[row * 2 + 1]);
    events.mercatorPositions[index * 2] = mercatorX;
    events.mercatorPositions[index * 2 + 1] = mercatorY;
    events.days[index] = seconds[row] / 86400 + dayOffset;
    events.magnitude[index] = magnitudes[row];
    events.depth[index] = Math.max(0, depths[row]);
  });
  return events;
}

/** Plate boundary steps of the window, in the analysis frame of the events. */
export type QuakeBoundarySteps = {
  count: number;
  /** `x0, y0, x1, y1` analysis meters per step: seeds, distances and drawing. */
  segments: Float32Array;
  /** Class code per step (index into `BOUNDARY_CLASS_NAMES`). */
  stepClass: Uint8Array;
  /** Plate convergence rate per step, mm/a. */
  convergence: Float32Array;
};

/** Reads the steps with an end point inside the window grown by `padDegrees`. */
export function readBoundarySteps(
  boundaries: LoadedDataset,
  events: Pick<QuakeEvents, 'analysisProjection'>,
  regionId: QuakeRegionId,
  padDegrees: number
): QuakeBoundarySteps {
  const [west, south, east, north] = QUAKE_REGIONS[regionId].bbox;
  const vertices = boundaries.column<Float32Array>('vertices');
  const classes = boundaries.column<Uint8Array>('stepClass');
  const convergence = boundaries.column<Float32Array>('convergence');
  const inside = (longitude: number, latitude: number) =>
    longitude >= west - padDegrees &&
    longitude <= east + padDegrees &&
    latitude >= south - padDegrees &&
    latitude <= north + padDegrees;
  const kept: number[] = [];
  for (let step = 0; step < classes.length; step++) {
    const base = step * 4;
    if (
      inside(vertices[base], vertices[base + 1]) ||
      inside(vertices[base + 2], vertices[base + 3])
    ) {
      kept.push(step);
    }
  }
  const result: QuakeBoundarySteps = {
    count: kept.length,
    segments: new Float32Array(kept.length * 4),
    stepClass: new Uint8Array(kept.length),
    convergence: new Float32Array(kept.length)
  };
  kept.forEach((step, index) => {
    const base = step * 4;
    const [x0, y0] = events.analysisProjection.project(vertices[base], vertices[base + 1]);
    const [x1, y1] = events.analysisProjection.project(vertices[base + 2], vertices[base + 3]);
    result.segments.set([x0, y0, x1, y1], index * 4);
    result.stepClass[index] = classes[step];
    result.convergence[index] = convergence[step];
  });
  return result;
}

/** Seed points (analysis meters) along boundary steps, spaced at most `spacing` meters apart. */
export type QuakeSeeds = {
  count: number;
  positions: Float32Array;
  /** Family code of the step each seed came from. */
  families: Uint32Array;
};

/**
 * Samples the steps whose class passes `accept` into seed points for `GPUDistanceField`: both end
 * points and enough interior points that no two neighbors are further apart than `spacing`.
 * Stops at `capacity` seeds and reports `overflow`.
 */
export function densifyBoundarySeeds(
  steps: QuakeBoundarySteps,
  accept: (classCode: number) => boolean,
  spacing: number,
  capacity: number
): QuakeSeeds & {overflow: boolean} {
  const positions = new Float32Array(capacity * 2);
  const families = new Uint32Array(capacity);
  let count = 0;
  let overflow = false;
  for (let step = 0; step < steps.count && !overflow; step++) {
    const code = steps.stepClass[step];
    if (!accept(code)) continue;
    const base = step * 4;
    const [x0, y0, x1, y1] = [
      steps.segments[base],
      steps.segments[base + 1],
      steps.segments[base + 2],
      steps.segments[base + 3]
    ];
    const pieces = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / spacing));
    for (let piece = 0; piece <= pieces; piece++) {
      if (count >= capacity) {
        overflow = true;
        break;
      }
      const t = piece / pieces;
      positions[count * 2] = x0 + (x1 - x0) * t;
      positions[count * 2 + 1] = y0 + (y1 - y0) * t;
      families[count] = BOUNDARY_FAMILY_OF_CLASS[code];
      count++;
    }
  }
  return {count, positions, families, overflow};
}

/** Result of a Gutenberg-Richter fit to binned magnitudes. */
export type GutenbergRichterFit = {
  /** Events at or above the completeness magnitude. */
  count: number;
  /** Aki-Utsu maximum-likelihood b-value, `NaN` when fewer than 2 events. */
  b: number;
  /** Standard error `b / sqrt(count)`. */
  standardError: number;
  /** Intercept `a` of `log10 N(>= M) = a - b M`. */
  a: number;
};

/**
 * Fits `log10 N(>= M) = a - b M` to counts per magnitude bin (`counts[i]` events of magnitude
 * `startMagnitude + i * binWidth`) above the completeness magnitude `completeness`, with the
 * Aki (1965) maximum-likelihood b-value and Utsu's correction for binned magnitudes.
 */
export function fitGutenbergRichter(
  counts: ArrayLike<number>,
  startMagnitude: number,
  binWidth: number,
  completeness: number
): GutenbergRichterFit {
  let total = 0;
  let sum = 0;
  for (let bin = 0; bin < counts.length; bin++) {
    const magnitude = startMagnitude + bin * binWidth;
    if (magnitude < completeness - 1e-6) continue;
    total += counts[bin];
    sum += counts[bin] * magnitude;
  }
  if (total < 2) return {count: total, b: Number.NaN, standardError: Number.NaN, a: Number.NaN};
  const mean = sum / total;
  const b = Math.LOG10E / (mean - (completeness - binWidth / 2));
  return {
    count: total,
    b,
    standardError: b / Math.sqrt(total),
    a: Math.log10(total) + b * completeness
  };
}

/** Median of a histogram: the bin center where the cumulative count crosses half. */
export function getHistogramMedian(counts: ArrayLike<number>, binWidth: number): number {
  let total = 0;
  for (let bin = 0; bin < counts.length; bin++) total += counts[bin];
  if (total === 0) return Number.NaN;
  let cumulative = 0;
  for (let bin = 0; bin < counts.length; bin++) {
    cumulative += counts[bin];
    if (cumulative >= total / 2) return (bin + 0.5) * binWidth;
  }
  return Number.NaN;
}
