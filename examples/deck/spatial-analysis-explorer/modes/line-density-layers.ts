// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local helpers of the line-density mode: polygon sets in the road network's meter frame
 * (a jittered district tiling and the catalog ZIP shapes moved over the roads) and a CPU
 * rasterizer that turns a polygon set into a grid of feature rows for the choropleth.
 */

import type {SpatialAnalysisPolygons} from '../spatial-analysis-data';

/** Polygons in the layout of `GPUSpatialJoinPolygons`, plus drawing and area helpers. */
export type LineDensityPolygonSet = {
  /** Label shown in the polygon-set select. */
  label: string;
  /** Flattened ring vertices `x, y` in meters of the road frame. */
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Feature name per row. */
  names: readonly string[];
  /** Area per feature in square meters (shell minus holes). */
  areas: Float32Array;
  /** Ring outline segments `x0, y0, x1, y1`. */
  outlineSegments: Float32Array;
};

/** Square-cell grid of feature rows; cells outside every polygon hold `featureCount`. */
export type FeatureRowGrid = {
  columns: number;
  rows: number;
  bounds: [number, number, number, number];
  featureRows: Uint32Array;
};

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Builds a polygon set from one list of rings per feature (one polygon per feature). */
function buildPolygonSet(
  label: string,
  features: {name: string; rings: [number, number][][]}[]
): LineDensityPolygonSet {
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  const outline: number[] = [];
  const areas = new Float32Array(features.length);
  features.forEach((feature, row) => {
    feature.rings.forEach((ring, ringIndex) => {
      const open =
        ring.length > 1 &&
        ring[0][0] === ring[ring.length - 1][0] &&
        ring[0][1] === ring[ring.length - 1][1]
          ? ring.slice(0, -1)
          : ring;
      let twiceArea = 0;
      for (let index = 0; index < open.length; index++) {
        const [x0, y0] = open[index];
        const [x1, y1] = open[(index + 1) % open.length];
        positions.push(x0, y0);
        outline.push(x0, y0, x1, y1);
        twiceArea += x0 * y1 - x1 * y0;
      }
      ringOffsets.push(positions.length / 2);
      // The first ring is the shell; later rings are holes.
      areas[row] += (ringIndex === 0 ? 1 : -1) * Math.abs(twiceArea / 2);
    });
    polygonOffsets.push(ringOffsets.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  });
  return {
    label,
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    names: features.map(feature => feature.name),
    areas,
    outlineSegments: Float32Array.from(outline)
  };
}

/**
 * A jittered quadrilateral tiling of `[minX, minY, maxX, maxY]`. The tiling partitions the box, so
 * every road inside it is clipped into exactly one district piece set and lengths must sum back to
 * the network length.
 */
export function createDistrictTiling(
  bounds: readonly [number, number, number, number],
  columns: number,
  rows: number
): LineDensityPolygonSet {
  const random = createSeededRandom(23);
  const width = (bounds[2] - bounds[0]) / columns;
  const height = (bounds[3] - bounds[1]) / rows;
  const corners: [number, number][][] = [];
  for (let row = 0; row <= rows; row++) {
    corners.push([]);
    for (let column = 0; column <= columns; column++) {
      const interior = row > 0 && row < rows && column > 0 && column < columns;
      corners[row].push([
        bounds[0] + column * width + (interior ? (random() - 0.5) * 0.6 * width : 0),
        bounds[1] + row * height + (interior ? (random() - 0.5) * 0.6 * height : 0)
      ]);
    }
  }
  const features: {name: string; rings: [number, number][][]}[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      features.push({
        name: `District ${features.length + 1}`,
        rings: [
          [
            corners[row][column],
            corners[row][column + 1],
            corners[row + 1][column + 1],
            corners[row + 1][column]
          ]
        ]
      });
    }
  }
  return buildPolygonSet('District tiling (partitions the roads)', features);
}

/**
 * The catalog ZIP polygons moved so their bounding-box center sits on `center`. The catalog has no
 * New York ZIP set, so the real (or synthetic) shapes are laid over the roads; they leave gaps.
 */
export function createMovedZipSet(
  zips: SpatialAnalysisPolygons,
  center: readonly [number, number],
  label: string
): LineDensityPolygonSet {
  const {polygonPositions, featureOffsets, polygonOffsets, ringOffsets} = zips;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < polygonPositions.length; index += 2) {
    minX = Math.min(minX, polygonPositions[index]);
    maxX = Math.max(maxX, polygonPositions[index]);
    minY = Math.min(minY, polygonPositions[index + 1]);
    maxY = Math.max(maxY, polygonPositions[index + 1]);
  }
  const shiftX = center[0] - (minX + maxX) / 2;
  const shiftY = center[1] - (minY + maxY) / 2;
  const features: {name: string; rings: [number, number][][]}[] = [];
  for (let feature = 0; feature < featureOffsets.length - 1; feature++) {
    const rings: [number, number][][] = [];
    for (
      let ring = polygonOffsets[featureOffsets[feature]];
      ring < polygonOffsets[featureOffsets[feature + 1]];
      ring++
    ) {
      const vertices: [number, number][] = [];
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
        vertices.push([
          polygonPositions[vertex * 2] + shiftX,
          polygonPositions[vertex * 2 + 1] + shiftY
        ]);
      }
      rings.push(vertices);
    }
    features.push({name: zips.featureNames[feature], rings});
  }
  return buildPolygonSet(label, features);
}

/**
 * Rasterizes a polygon set into a square-cell grid of feature rows (CPU, once). A cell takes the
 * lowest feature row whose rings contain its center (even-odd over the rings of a feature). Row 0
 * is the south edge, matching `SpatialAnalysisRasterLayer` with `rowOrigin: 'south'`.
 */
export function rasterizeFeatureRows(set: LineDensityPolygonSet, columns: number): FeatureRowGrid {
  const {positions, featureOffsets, polygonOffsets, ringOffsets} = set;
  const featureCount = featureOffsets.length - 1;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minX = Math.min(minX, positions[index]);
    maxX = Math.max(maxX, positions[index]);
    minY = Math.min(minY, positions[index + 1]);
    maxY = Math.max(maxY, positions[index + 1]);
  }
  const cellSize = (maxX - minX) / columns;
  const rows = Math.max(1, Math.ceil((maxY - minY) / cellSize));
  const featureRows = new Uint32Array(columns * rows).fill(featureCount);
  const crossings: number[][] = Array.from({length: rows}, () => []);
  for (let feature = 0; feature < featureCount; feature++) {
    for (const rowCrossings of crossings) rowCrossings.length = 0;
    for (
      let ring = polygonOffsets[featureOffsets[feature]];
      ring < polygonOffsets[featureOffsets[feature + 1]];
      ring++
    ) {
      const start = ringOffsets[ring];
      const length = ringOffsets[ring + 1] - start;
      for (let vertex = 0; vertex < length; vertex++) {
        const from = (start + vertex) * 2;
        const to = (start + ((vertex + 1) % length)) * 2;
        const x0 = positions[from];
        const y0 = positions[from + 1];
        const x1 = positions[to];
        const y1 = positions[to + 1];
        if (y0 === y1) continue;
        const rowStart = Math.max(0, Math.ceil((Math.min(y0, y1) - minY) / cellSize - 0.5));
        const rowEnd = Math.min(
          rows - 1,
          Math.ceil((Math.max(y0, y1) - minY) / cellSize - 0.5) - 1
        );
        for (let row = rowStart; row <= rowEnd; row++) {
          const centerY = minY + (row + 0.5) * cellSize;
          crossings[row].push(x0 + ((centerY - y0) / (y1 - y0)) * (x1 - x0));
        }
      }
    }
    for (let row = 0; row < rows; row++) {
      const rowCrossings = crossings[row].sort((left, right) => left - right);
      for (let index = 0; index + 1 < rowCrossings.length; index += 2) {
        const columnStart = Math.max(0, Math.ceil((rowCrossings[index] - minX) / cellSize - 0.5));
        const columnEnd = Math.min(
          columns - 1,
          Math.ceil((rowCrossings[index + 1] - minX) / cellSize - 0.5) - 1
        );
        for (let column = columnStart; column <= columnEnd; column++) {
          const cell = row * columns + column;
          if (featureRows[cell] === featureCount) featureRows[cell] = feature;
        }
      }
    }
  }
  return {
    columns,
    rows,
    bounds: [minX, minY, minX + columns * cellSize, minY + rows * cellSize],
    featureRows
  };
}

/** Returns the feature row containing `[x, y]` using the rasterized grid, or -1. */
export function getFeatureRowAt(grid: FeatureRowGrid, featureCount: number, x: number, y: number) {
  const cellWidth = (grid.bounds[2] - grid.bounds[0]) / grid.columns;
  const cellHeight = (grid.bounds[3] - grid.bounds[1]) / grid.rows;
  const column = Math.floor((x - grid.bounds[0]) / cellWidth);
  const row = Math.floor((y - grid.bounds[1]) / cellHeight);
  if (column < 0 || row < 0 || column >= grid.columns || row >= grid.rows) return -1;
  const feature = grid.featureRows[row * grid.columns + column];
  return feature >= featureCount ? -1 : feature;
}
