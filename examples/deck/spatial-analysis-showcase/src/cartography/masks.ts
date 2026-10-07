// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Raster masks from polygons: city and lake masks, KDE windows, and edge-effect fade weights.
 * Scenes upload the result as a `mask` buffer. Pure TypeScript.
 */

import type {MapAnnotation} from './types';
import {getInputPolygons, type GeoJsonInput, type LngLatBounds, type PolygonRings} from './picking';

/** A raster grid in lon/lat. Same conventions as `RasterGrid` in `picking.ts`. */
export type MaskGrid = {
  width: number;
  height: number;
  /** `[west, south, east, north]` of the outer edges of the grid. */
  bounds: LngLatBounds;
  /** Which edge row 0 is on: `'north'` (default, as `DecodedRaster`) or `'south'`. */
  rowOrigin?: 'north' | 'south';
};

/**
 * Rasterises polygons to a mask: 1 where a cell centre is inside (even-odd, so holes are
 * respected; overlapping polygons are united), else 0. Row 0 follows `grid.rowOrigin`. The fill
 * is a scanline over the grid, linear in edges plus filled cells, so a 2,000 x 2,000 grid with a
 * detailed coastline takes tens of milliseconds.
 *
 * `input` is a GeoJSON collection, feature or geometry (Polygon and MultiPolygon), or the rings
 * of one polygon (outer ring first, then holes).
 *
 * @example
 * const grid = {width: 512, height: 512, bounds: [-88, 41.6, -87.5, 42.1] as const};
 * const mask = rasterizePolygonMask(chicagoBoundary, grid); // Uint8Array(512 * 512)
 * const weights = getEdgeFadeWeights(mask, grid, 8);
 */
export function rasterizePolygonMask(
  input: GeoJsonInput | PolygonRings,
  grid: MaskGrid
): Uint8Array {
  const mask = new Uint8Array(grid.width * grid.height);
  const polygons = Array.isArray(input)
    ? [input as PolygonRings]
    : getInputPolygons(input as GeoJsonInput).map(entry => entry.polygon);
  const [west, south, east, north] = grid.bounds;
  const columnScale = grid.width / (east - west);
  const rowScale = grid.height / (north - south);
  const fromSouth = grid.rowOrigin === 'south';

  for (const rings of polygons) {
    const crossings: number[][] = [];
    for (const ring of rings) {
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        // Vertices in grid space: x in columns, y in rows; cell centres sit at +0.5.
        const x0 = (ring[j][0] - west) * columnScale;
        const x1 = (ring[i][0] - west) * columnScale;
        const y0 = fromSouth ? (ring[j][1] - south) * rowScale : (north - ring[j][1]) * rowScale;
        const y1 = fromSouth ? (ring[i][1] - south) * rowScale : (north - ring[i][1]) * rowScale;
        if (y0 === y1) continue;
        const rowStart = Math.max(0, Math.ceil(Math.min(y0, y1) - 0.5));
        const rowEnd = Math.min(grid.height - 1, Math.ceil(Math.max(y0, y1) - 0.5) - 1);
        for (let row = rowStart; row <= rowEnd; row++) {
          const x = x0 + ((row + 0.5 - y0) / (y1 - y0)) * (x1 - x0);
          const rowCrossings = crossings[row] ?? [];
          rowCrossings.push(x);
          crossings[row] = rowCrossings;
        }
      }
    }
    crossings.forEach((xs, row) => {
      if (!xs) return;
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const columnStart = Math.max(0, Math.ceil(xs[k] - 0.5));
        const columnEnd = Math.min(grid.width - 1, Math.ceil(xs[k + 1] - 0.5) - 1);
        if (columnEnd >= columnStart) {
          mask.fill(1, row * grid.width + columnStart, row * grid.width + columnEnd + 1);
        }
      }
    });
  }
  return mask;
}

/** Options of {@link getEdgeFadeWeights}. */
export type EdgeFadeOptions = {
  /** Weight of the outermost cells inside the mask. Default 0.5. */
  edgeWeight?: number;
};

/**
 * Edge-effect weights for a mask (a KDE or a smoothed surface is biased down near the edge of
 * its window because neighbours are missing): 0 outside the mask, `edgeWeight` (0.5) on the
 * outermost cells inside it, rising linearly to 1 over `fadeCells` cells, and 1 deeper in.
 * Distance to the outside is a 3-4 chamfer approximation of Euclidean, in cells. The grid
 * border does not count as an edge. Multiply the surface by the weights, or draw the weights as
 * opacity to show where the estimate is less certain.
 *
 * @example
 * const fade = getEdgeFadeWeights(mask, grid, 10); // Float32Array, 0 / 0.5..1
 */
export function getEdgeFadeWeights(
  mask: ArrayLike<number>,
  grid: Pick<MaskGrid, 'width' | 'height'>,
  fadeCells: number,
  options: EdgeFadeOptions = {}
): Float32Array {
  const {width, height} = grid;
  const edgeWeight = options.edgeWeight ?? 0.5;
  const weights = new Float32Array(width * height);
  const far = 1e9;
  // Chamfer distances scaled by 3 (axis step 3, diagonal step 4).
  const distance = new Float32Array(width * height);
  for (let i = 0; i < distance.length; i++) distance[i] = mask[i] ? far : 0;
  const at = (column: number, row: number) =>
    column < 0 || column >= width || row < 0 || row >= height
      ? far
      : distance[row * width + column];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const i = row * width + column;
      if (distance[i] === 0) continue;
      distance[i] = Math.min(
        distance[i],
        at(column - 1, row) + 3,
        at(column, row - 1) + 3,
        at(column - 1, row - 1) + 4,
        at(column + 1, row - 1) + 4
      );
    }
  }
  for (let row = height - 1; row >= 0; row--) {
    for (let column = width - 1; column >= 0; column--) {
      const i = row * width + column;
      if (distance[i] === 0) continue;
      distance[i] = Math.min(
        distance[i],
        at(column + 1, row) + 3,
        at(column, row + 1) + 3,
        at(column + 1, row + 1) + 4,
        at(column - 1, row + 1) + 4
      );
    }
  }
  const span = Math.max(fadeCells, 0);
  for (let i = 0; i < weights.length; i++) {
    if (!mask[i]) continue;
    const cells = distance[i] / 3; // 1 for a cell next to the outside
    if (cells >= far / 3 || span === 0) {
      weights[i] = 1;
      continue;
    }
    const t = Math.min(1, Math.max(0, (cells - 1) / span));
    weights[i] = edgeWeight + (1 - edgeWeight) * t;
  }
  return weights;
}

/**
 * The `frame` annotation for a rectangle, to mark where data ends ("Data ends here") and clipping
 * creates an apparent edge.
 *
 * @example
 * annotations.push(getBoundsFrame([-88, 41.6, -87.5, 42.1], 'Data ends here'));
 */
export function getBoundsFrame(
  bounds: readonly [number, number, number, number],
  text?: string
): Extract<MapAnnotation, {kind: 'frame'}> {
  return text === undefined ? {kind: 'frame', bounds} : {kind: 'frame', bounds, text};
}
