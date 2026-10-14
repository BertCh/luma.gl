// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  CELL_COVER_H3_LATTICE_SPACING_DEGREES,
  CELL_COVER_H3_MAXIMUM_LATITUDE
} from '../../../src/gpu-spatial-analysis/cell-cover/cell-cover-h3-constants';
import {
  CELL_COVER_SCANLINE_COST_RATIO,
  CELL_COVER_SCANLINE_H3_X_SLACK_DEGREES,
  CELL_COVER_SCANLINE_H3_Y_SLACK_DEGREES,
  CELL_COVER_SCANLINE_QUADBIN_Y_SLACK_DEGREES,
  CELL_COVER_SCANLINE_X_SLACK_DEGREES,
  CELL_COVER_SCANLINE_X_SLACK_RELATIVE
} from '../../../src/gpu-spatial-analysis/cell-cover/cell-cover-scanline';
import {cellToLatLng} from 'h3-js';
import {
  bigIntToH3,
  getQuadbinTileX,
  getQuadbinTileY,
  quadbinCellToTile
} from '../cell-aggregation/cell-aggregation-oracle';
import {isInsideRings, type CoverFeature} from './cell-cover-oracle';

/**
 * CPU model of the scanline candidate enumeration of `GPUCellCover` (steps (a) to (d) of
 * `cell-cover-scanline.ts`): boundary trace spans, scanline crossings, sort by (row, x),
 * depth/parity sweep, column ranges and disjoint runs, with the kernel's f32 operations mirrored by
 * `Math.fround`. Tile and row-line latitudes come from the f64 formulas rounded to f32 (the kernel
 * uses f32 `atan`/`exp`); `latitudeJitter` perturbs them by up to that many degrees to model the
 * difference between kernels and the formulas.
 */

const f = Math.fround;

type Vertex = [number, number];

/** Lattice of one feature, with the row and column functions of the kernels. */
export type ScanlineLattice = {
  family: 'quadbin' | 'h3';
  rows: number;
  columns: number;
  /** Test line latitude of a row. */
  getRowLine: (row: number) => number;
  /** Band `[low, high]` of a row, with slack. */
  getRowBand: (row: number) => [number, number];
  /** Inclusive row range that can meet `[lowY, highY]`, clamped; empty when first > last. */
  getEdgeRows: (lowY: number, highY: number) => [number, number];
  /** Inclusive column range for an x interval, clamped; empty when first > last. */
  getColumnRange: (lowX: number, highX: number) => [number, number];
  /** Test point of a cell (no jitter). */
  getCenter: (row: number, column: number) => Vertex;
  /** Nominal rectangle around the test point that must be covered (no jitter). */
  getRectangle: (row: number, column: number) => [Vertex, Vertex];
  /** Lattice `[row, column]` that enumerates a cell key (the tile, or H3's nearest lattice point). */
  getCellPosition: (cell: bigint) => [row: number, column: number];
};

/** Result of {@link getScanlineCandidatesOnCPU}. */
export type ScanlineResult = {
  lattice: ScanlineLattice;
  /** Cells of the bounding box. */
  boundsCount: number;
  /** Edge-row slots the boundary trace needs. */
  slots: number;
  /** Whether the GPU choice rule picks the scanline enumeration (and the stage capacity fits). */
  selected: boolean;
  /** Candidates of the scanline enumeration, row-major, no duplicates (empty when not computed). */
  candidates: [row: number, column: number][];
};

function getRings(feature: CoverFeature): Vertex[][] {
  return feature.flatMap(polygon => polygon) as Vertex[][];
}

function getBounds(rings: Vertex[][]): [number, number, number, number] | undefined {
  let bounds: [number, number, number, number] | undefined;
  for (const ring of rings) {
    for (const [x, y] of ring) {
      if (Number.isFinite(x) && Number.isFinite(y)) {
        bounds = bounds
          ? [
              Math.min(bounds[0], x),
              Math.min(bounds[1], y),
              Math.max(bounds[2], x),
              Math.max(bounds[3], y)
            ]
          : [x, y, x, y];
      }
    }
  }
  return bounds;
}

function getTileLatitude(row: number, resolution: number): number {
  const t = Math.PI * (1 - 2 * (row / 2 ** resolution));
  return f(Math.atan(Math.sinh(t)) * (180 / Math.PI));
}

/** Creates the lattice of a feature like the count kernel, or `undefined` without candidates. */
export function createScanlineLattice(
  feature: CoverFeature,
  family: 'quadbin' | 'h3',
  resolution: number,
  latitudeJitter: number = 0
): ScanlineLattice | undefined {
  const bounds = getBounds(getRings(feature));
  if (!bounds || bounds[2] - bounds[0] > 180) {
    return undefined;
  }
  const jitter = (row: number) => (row % 2 === 0 ? latitudeJitter : -latitudeJitter);
  if (family === 'quadbin') {
    const x0 = getQuadbinTileX(bounds[0], resolution);
    const x1 = getQuadbinTileX(bounds[2], resolution);
    const y0 = getQuadbinTileY(bounds[3], resolution);
    const y1 = getQuadbinTileY(bounds[1], resolution);
    if (x1 < x0 || y1 < y0) {
      return undefined;
    }
    const rows = y1 - y0 + 1;
    const columns = x1 - x0 + 1;
    const tileWidth = f(45 * 2 ** (3 - resolution));
    const slackY = f(CELL_COVER_SCANLINE_QUADBIN_Y_SLACK_DEGREES);
    const latitude = (tileRow: number) =>
      f(getTileLatitude(tileRow, resolution) + jitter(Math.floor(tileRow)));
    return {
      family,
      rows,
      columns,
      getRowLine: row => latitude(f(y0 + row) + 0.5),
      getRowBand: row => [f(latitude(y0 + row + 1) - slackY), f(latitude(y0 + row) + slackY)],
      getEdgeRows: (lowY, highY) => [
        Math.max(getQuadbinTileY(f(highY + slackY), resolution) - y0 - 1, 0),
        Math.min(getQuadbinTileY(f(lowY - slackY), resolution) - y0 + 1, rows - 1)
      ],
      getColumnRange: (lowX, highX) => [
        Math.max(
          Math.floor(f(f(f(lowX - f(CELL_COVER_SCANLINE_X_SLACK_DEGREES)) + 180) / tileWidth)) -
            1 -
            x0,
          0
        ),
        Math.min(
          Math.floor(f(f(f(highX + f(CELL_COVER_SCANLINE_X_SLACK_DEGREES)) + 180) / tileWidth)) +
            1 -
            x0,
          columns - 1
        )
      ],
      getCenter: (row, column) => [
        (x0 + column + 0.5) * (360 / 2 ** resolution) - 180,
        getTileLatitude(y0 + row + 0.5, resolution)
      ],
      getRectangle: (row, column) => [
        [(x0 + column) * (360 / 2 ** resolution) - 180, getTileLatitude(y0 + row + 1, resolution)],
        [(x0 + column + 1) * (360 / 2 ** resolution) - 180, getTileLatitude(y0 + row, resolution)]
      ],
      getCellPosition: cell => {
        const tile = quadbinCellToTile(cell);
        return [tile.y - y0, tile.x - x0];
      }
    };
  }
  const spacing = CELL_COVER_H3_LATTICE_SPACING_DEGREES[resolution];
  const originLat = f(Math.max(bounds[1] - spacing, -CELL_COVER_H3_MAXIMUM_LATITUDE));
  const endLat = f(Math.min(bounds[3] + spacing, CELL_COVER_H3_MAXIMUM_LATITUDE));
  if (bounds[1] > bounds[3] || originLat > endLat) {
    return undefined;
  }
  const equatorwardLat =
    originLat <= 0 && endLat >= 0 ? 0 : Math.min(Math.abs(originLat), Math.abs(endLat));
  const spacingLng = f(spacing / Math.cos((equatorwardLat * Math.PI) / 180));
  const originLng = f(bounds[0] - spacingLng);
  const columns = Math.ceil((bounds[2] + spacingLng - originLng) / spacingLng) + 1;
  const rows = Math.ceil((endLat - originLat) / spacing) + 1;
  const line = (row: number) => f(originLat + f(f(row) * spacing));
  const halfBand = f(
    spacing * f(0.501 + f(3e-7 * rows)) + f(CELL_COVER_SCANLINE_H3_Y_SLACK_DEGREES)
  );
  const margin = f(
    f(spacingLng * f(0.501 + f(3e-7 * columns))) + f(CELL_COVER_SCANLINE_H3_X_SLACK_DEGREES)
  );
  return {
    family,
    rows,
    columns,
    getRowLine: row => f(line(row) + jitter(row)),
    getRowBand: row => [
      f(f(line(row) + jitter(row)) - halfBand),
      f(f(line(row) + jitter(row)) + halfBand)
    ],
    getEdgeRows: (lowY, highY) => [
      Math.max(Math.floor(f(f(f(lowY - halfBand) - originLat) / spacing)) - 1, 0),
      Math.min(Math.ceil(f(f(f(highY + halfBand) - originLat) / spacing)) + 1, rows - 1)
    ],
    getColumnRange: (lowX, highX) => {
      const extent = f(margin + f(CELL_COVER_SCANLINE_X_SLACK_DEGREES));
      return [
        Math.max(Math.floor(f(f(f(lowX - extent) - originLng) / spacingLng)) - 1, 0),
        Math.min(Math.ceil(f(f(f(highX + extent) - originLng) / spacingLng)) + 1, columns - 1)
      ];
    },
    getCenter: (row, column) => [originLng + column * spacingLng, originLat + row * spacing],
    getRectangle: (row, column) => [
      [originLng + (column - 0.5) * spacingLng, originLat + (row - 0.5) * spacing],
      [originLng + (column + 0.5) * spacingLng, originLat + (row + 0.5) * spacing]
    ],
    getCellPosition: cell => {
      const [centreLat, centreLng] = cellToLatLng(bigIntToH3(cell));
      return [
        Math.floor((centreLat - originLat) / spacing + 0.5),
        Math.floor((centreLng - originLng) / spacingLng + 0.5)
      ];
    }
  };
}

/** `(valid, low, high)` x extent of edge a-b inside a latitude band, with slack (WGSL `coverClipEdge`). */
function clipEdge(a: Vertex, b: Vertex, band: [number, number]): [number, number] | undefined {
  const aIsLower = a[1] <= b[1];
  const lower = aIsLower ? a : b;
  const upper = aIsLower ? b : a;
  if (upper[1] < band[0] || lower[1] > band[1]) {
    return undefined;
  }
  let x0 = lower[0];
  let x1 = upper[0];
  const dy = f(upper[1] - lower[1]);
  if (dy > 0) {
    const dx = f(upper[0] - lower[0]);
    const t0 = Math.min(Math.max(f(f(band[0] - lower[1]) / dy), 0), 1);
    const t1 = Math.min(Math.max(f(f(band[1] - lower[1]) / dy), 0), 1);
    x0 = f(lower[0] + f(dx * t0));
    x1 = f(lower[0] + f(dx * t1));
  }
  const slack = f(
    f(CELL_COVER_SCANLINE_X_SLACK_DEGREES) +
      f(f(CELL_COVER_SCANLINE_X_SLACK_RELATIVE) * Math.abs(f(upper[0] - lower[0])))
  );
  return [f(Math.min(x0, x1) - slack), f(Math.max(x0, x1) + slack)];
}

/** x where edge a-b crosses the line (WGSL `coverGetCrossing`). */
function getCrossing(a: Vertex, b: Vertex, line: number): number {
  const aIsLower = a[1] <= b[1];
  const lower = aIsLower ? a : b;
  const upper = aIsLower ? b : a;
  const t = Math.min(Math.max(f(f(line - lower[1]) / f(upper[1] - lower[1])), 0), 1);
  return f(lower[0] + f(f(upper[0] - lower[0]) * t));
}

type Event = {row: number; position: number; kind: 0 | 1 | 2};

/**
 * Runs the scanline enumeration for one feature. With `cheaperOnly` the result carries candidates
 * only when the choice rule selects the feature (the GPU behavior); otherwise candidates are always
 * computed so superset properties can be checked on any polygon.
 */
export function getScanlineCandidatesOnCPU(
  feature: CoverFeature,
  family: 'quadbin' | 'h3',
  resolution: number,
  options: {
    slotCapacity?: number;
    force?: boolean;
    latitudeJitter?: number;
    cheaperOnly?: boolean;
    /** Negative control: drop the boundary trace spans, keeping only the scanline crossings. */
    omitSpans?: boolean;
  } = {}
): ScanlineResult | undefined {
  const lattice = createScanlineLattice(feature, family, resolution, options.latitudeJitter);
  if (!lattice) {
    return undefined;
  }
  const rings = getRings(feature);
  const slotCapacity = options.slotCapacity ?? Infinity;
  const boundsCount = lattice.rows * lattice.columns;
  const edges: [Vertex, Vertex][] = [];
  for (const ring of rings) {
    for (let index = 0; index < ring.length; index++) {
      edges.push([ring[index], ring[(index + 1) % ring.length]]);
    }
  }
  const events: Event[] = [];
  let slots = 0;
  for (const [a, b] of edges) {
    const [first, last] = lattice.getEdgeRows(Math.min(a[1], b[1]), Math.max(a[1], b[1]));
    for (let row = first; row <= last; row++) {
      slots++;
      const clip = clipEdge(a, b, lattice.getRowBand(row));
      if (clip && !options.omitSpans) {
        events.push({row, position: clip[0], kind: 0}, {row, position: clip[1], kind: 1});
      }
      const line = lattice.getRowLine(row);
      if (a[1] <= line !== b[1] <= line) {
        events.push({row, position: getCrossing(a, b, line), kind: 2});
      }
    }
  }
  const cheaper = boundsCount > CELL_COVER_SCANLINE_COST_RATIO * (2 * lattice.rows + slots);
  const eligible = lattice.rows <= slotCapacity && slots + lattice.rows <= slotCapacity;
  const selected = (Boolean(options.force) || cheaper) && eligible;
  const result: ScanlineResult = {lattice, boundsCount, slots, selected, candidates: []};
  if (options.cheaperOnly && !selected) {
    return result;
  }
  events.sort((left, right) => left.row - right.row || left.position - right.position);
  let depth = 0;
  let crossings = 0;
  let previousRun: {row: number; high: number} | undefined;
  for (let index = 0; index + 1 < events.length; index++) {
    const event = events[index];
    if (index === 0 || events[index - 1].row !== event.row) {
      depth = 0;
      crossings = 0;
    }
    if (event.kind === 0) {
      depth++;
    } else if (event.kind === 1) {
      depth--;
    } else {
      crossings++;
    }
    const next = events[index + 1];
    if (next.row !== event.row || !(depth !== 0 || crossings % 2 === 1)) {
      continue;
    }
    if (!(event.position < next.position)) {
      continue;
    }
    const [low, high] = lattice.getColumnRange(event.position, next.position);
    if (low > high) {
      continue;
    }
    let start = low;
    if (previousRun?.row === event.row) {
      start = Math.max(start, previousRun.high + 1);
    }
    previousRun = {row: event.row, high};
    for (let column = start; column <= high; column++) {
      result.candidates.push([event.row, column]);
    }
  }
  return result;
}

function segmentMeetsRectangle(a: Vertex, b: Vertex, lo: Vertex, hi: Vertex): boolean {
  let lower = 0;
  let upper = 1;
  for (const axis of [0, 1]) {
    const delta = b[axis] - a[axis];
    if (delta === 0) {
      if (a[axis] < lo[axis] || a[axis] > hi[axis]) {
        return false;
      }
    } else {
      const t1 = (lo[axis] - a[axis]) / delta;
      const t2 = (hi[axis] - a[axis]) / delta;
      lower = Math.max(lower, Math.min(t1, t2));
      upper = Math.min(upper, Math.max(t1, t2));
    }
  }
  return lower <= upper;
}

/**
 * Cells (`row * columns + column`) the exact test could accept for any containment: the test point
 * is inside the polygon (f64 even-odd) or a polygon edge meets the closed nominal rectangle. This
 * is the set the scanline enumeration must contain.
 */
export function getNecessaryCellsOnCPU(
  feature: CoverFeature,
  lattice: ScanlineLattice
): Set<number> {
  const rings = getRings(feature);
  const edges: [Vertex, Vertex][] = [];
  for (const ring of rings) {
    for (let index = 0; index < ring.length; index++) {
      edges.push([ring[index], ring[(index + 1) % ring.length]]);
    }
  }
  const necessary = new Set<number>();
  for (let row = 0; row < lattice.rows; row++) {
    for (let column = 0; column < lattice.columns; column++) {
      const [lo, hi] = lattice.getRectangle(row, column);
      if (
        isInsideRings(rings, lattice.getCenter(row, column), true) ||
        edges.some(([a, b]) => segmentMeetsRectangle(a, b, lo, hi))
      ) {
        necessary.add(row * lattice.columns + column);
      }
    }
  }
  return necessary;
}
