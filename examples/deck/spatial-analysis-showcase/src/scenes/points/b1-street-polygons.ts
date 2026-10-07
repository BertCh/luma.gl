// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {LocalMetricProjection} from '../../engine/projection';

/** Polygons in the layout of `GPUSpatialJoinPolygons`, plus drawing helpers. */
export type StreetPolygonSet = {
  id: string;
  label: string;
  /** Ring vertices as local meters `x, y`. */
  positions: Float32Array;
  /** The same vertices as longitude and latitude degrees (for the spherical graphs). */
  degrees: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  names: readonly string[];
  /** Area per feature in square meters (shell minus holes). */
  areas: Float32Array;
  /** Residents per feature. */
  population: Float32Array;
  /** Ring outline segments `x0, y0, x1, y1` in meters. */
  outlineSegments: Float32Array;
};

/** Square-cell grid of feature rows; cells outside every polygon hold `featureCount`. */
export type FeatureRowGrid = {
  columns: number;
  rows: number;
  bounds: [number, number, number, number];
  featureRows: Uint32Array;
};

/**
 * Builds a polygon set from a binary GeoArrow-style dataset (`vertices`, `ringOffsets`,
 * `polygonRingOffsets`, `partFeature`). Parts are grouped by feature, rings stay open.
 */
export function readPolygonSet(
  dataset: LoadedDataset,
  projection: LocalMetricProjection,
  options: {id: string; label: string; names: readonly string[]; population?: Float32Array}
): StreetPolygonSet {
  const vertices = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonRingOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partFeature = dataset.column<Uint32Array>('partFeature');
  const featureCount = options.names.length;
  // Rebuild the arrays so rings are open (no repeated closing vertex) and parts follow features.
  const positions: number[] = [];
  const degrees: number[] = [];
  const outline: number[] = [];
  const newRingOffsets = [0];
  const newPolygonOffsets = [0];
  const newFeatureOffsets = [0];
  const areas = new Float32Array(featureCount);
  const partsByFeature: number[][] = Array.from({length: featureCount}, () => []);
  for (let part = 0; part < partFeature.length; part++)
    partsByFeature[partFeature[part]].push(part);
  for (let feature = 0; feature < featureCount; feature++) {
    for (const part of partsByFeature[feature]) {
      for (let ring = polygonRingOffsets[part]; ring < polygonRingOffsets[part + 1]; ring++) {
        let start = ringOffsets[ring];
        let end = ringOffsets[ring + 1];
        if (
          end - start > 1 &&
          vertices[start * 2] === vertices[(end - 1) * 2] &&
          vertices[start * 2 + 1] === vertices[(end - 1) * 2 + 1]
        ) {
          end--;
        }
        const ringPoints: [number, number][] = [];
        for (let vertex = start; vertex < end; vertex++) {
          const longitude = vertices[vertex * 2];
          const latitude = vertices[vertex * 2 + 1];
          degrees.push(longitude, latitude);
          ringPoints.push(projection.project(longitude, latitude));
        }
        let twiceArea = 0;
        for (let index = 0; index < ringPoints.length; index++) {
          const [x0, y0] = ringPoints[index];
          const [x1, y1] = ringPoints[(index + 1) % ringPoints.length];
          positions.push(x0, y0);
          outline.push(x0, y0, x1, y1);
          twiceArea += x0 * y1 - x1 * y0;
        }
        newRingOffsets.push(positions.length / 2);
        // The first ring of a part is its shell; later rings are holes.
        areas[feature] += (ring === polygonRingOffsets[part] ? 1 : -1) * Math.abs(twiceArea / 2);
        start = end;
      }
      newPolygonOffsets.push(newRingOffsets.length - 1);
    }
    newFeatureOffsets.push(newPolygonOffsets.length - 1);
  }
  return {
    id: options.id,
    label: options.label,
    positions: Float32Array.from(positions),
    degrees: Float32Array.from(degrees),
    featureOffsets: Uint32Array.from(newFeatureOffsets),
    polygonOffsets: Uint32Array.from(newPolygonOffsets),
    ringOffsets: Uint32Array.from(newRingOffsets),
    names: options.names,
    areas,
    population: options.population ?? new Float32Array(featureCount),
    outlineSegments: Float32Array.from(outline)
  };
}

/**
 * Rasterizes a polygon set into a square-cell grid of feature rows (CPU, once). A cell takes the
 * lowest feature row whose rings contain its center (even-odd over the rings of a feature). Row 0
 * is the south edge, matching `SpatialAnalysisRasterLayer` with `rowOrigin: 'south'`.
 */
export function rasterizeFeatureRows(set: StreetPolygonSet, columns: number): FeatureRowGrid {
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
export function getFeatureRowAt(
  grid: FeatureRowGrid,
  featureCount: number,
  x: number,
  y: number
): number {
  const cellWidth = (grid.bounds[2] - grid.bounds[0]) / grid.columns;
  const cellHeight = (grid.bounds[3] - grid.bounds[1]) / grid.rows;
  const column = Math.floor((x - grid.bounds[0]) / cellWidth);
  const row = Math.floor((y - grid.bounds[1]) / cellHeight);
  if (column < 0 || row < 0 || column >= grid.columns || row >= grid.rows) return -1;
  const feature = grid.featureRows[row * grid.columns + column];
  return feature >= featureCount ? -1 : feature;
}
