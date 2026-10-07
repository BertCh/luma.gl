// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GeoJsonCollection} from '../../data/loaders';
import type {LocalMetricProjection} from '../../engine/projection';

/** Polygon zones prepared for GPU drawing. */
export type ZoneRaster = {
  /** Zone row per cell, or `zoneCount` for cells outside every zone. Row 0 is the south edge. */
  zoneIds: Uint32Array;
  columns: number;
  rows: number;
  /** `[minX, minY, maxX, maxY]` planar meters of the raster. */
  bounds: [number, number, number, number];
  cellSize: number;
  /** Number of zones; `zoneIds` uses this value for "no zone". */
  zoneCount: number;
  /** Zone boundaries as `x0, y0, x1, y1` planar meter segments. */
  outline: Float32Array;
};

type Ring = [number, number][];

function collectRings(geometry: {type: string; coordinates: unknown} | null): Ring[][] {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates as Ring[]];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates as Ring[][];
  return [];
}

/**
 * Rasterizes polygon features to a zone-id grid (even-odd scanline fill at cell centers) so a
 * `SpatialAnalysisRasterLayer` can color zones from a GPU value buffer through `valueIndices`
 * without triangulating anything, and extracts the boundary as line segments.
 *
 * @param getZone Zone row of a feature, or `-1` to skip it.
 */
export function createZoneRaster(
  collection: GeoJsonCollection,
  projection: LocalMetricProjection,
  zoneCount: number,
  getZone: (properties: Record<string, unknown>, featureIndex: number) => number,
  cellSize: number
): ZoneRaster {
  const features = collection.features.map((feature, featureIndex) => ({
    zone: getZone(feature.properties ?? {}, featureIndex),
    polygons: collectRings(feature.geometry).map(rings =>
      rings.map(ring =>
        ring.map(([longitude, latitude]) => projection.project(longitude, latitude))
      )
    )
  }));
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const {polygons} of features) {
    for (const rings of polygons) {
      for (const ring of rings) {
        for (const [x, y] of ring) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }
  }
  const columns = Math.max(1, Math.ceil((maxX - minX) / cellSize));
  const rows = Math.max(1, Math.ceil((maxY - minY) / cellSize));
  const bounds: [number, number, number, number] = [
    minX,
    minY,
    minX + columns * cellSize,
    minY + rows * cellSize
  ];
  const zoneIds = new Uint32Array(columns * rows).fill(zoneCount);
  const outline: number[] = [];
  const crossings: number[] = [];
  for (const {zone, polygons} of features) {
    if (zone < 0) continue;
    const rings = polygons.flat();
    let featureMinY = Infinity;
    let featureMaxY = -Infinity;
    for (const ring of rings) {
      for (let index = 0; index < ring.length; index++) {
        const [x, y] = ring[index];
        const [nextX, nextY] = ring[(index + 1) % ring.length];
        featureMinY = Math.min(featureMinY, y);
        featureMaxY = Math.max(featureMaxY, y);
        outline.push(x, y, nextX, nextY);
      }
    }
    const firstRow = Math.max(0, Math.floor((featureMinY - minY) / cellSize));
    const lastRow = Math.min(rows - 1, Math.ceil((featureMaxY - minY) / cellSize));
    for (let row = firstRow; row <= lastRow; row++) {
      const y = minY + (row + 0.5) * cellSize;
      crossings.length = 0;
      for (const ring of rings) {
        for (let index = 0; index < ring.length; index++) {
          const [x0, y0] = ring[index];
          const [x1, y1] = ring[(index + 1) % ring.length];
          if (y0 <= y !== y1 <= y) {
            crossings.push(x0 + ((y - y0) * (x1 - x0)) / (y1 - y0));
          }
        }
      }
      crossings.sort((a, b) => a - b);
      for (let pair = 0; pair + 1 < crossings.length; pair += 2) {
        const from = Math.max(0, Math.ceil((crossings[pair] - minX) / cellSize - 0.5));
        const to = Math.min(columns - 1, Math.floor((crossings[pair + 1] - minX) / cellSize - 0.5));
        for (let column = from; column <= to; column++) zoneIds[row * columns + column] = zone;
      }
    }
  }
  return {
    zoneIds,
    columns,
    rows,
    bounds,
    cellSize,
    zoneCount,
    outline: Float32Array.from(outline)
  };
}

/** Zone row under a planar position, or `-1` outside every zone. */
export function getZoneAt(raster: ZoneRaster, x: number, y: number): number {
  const column = Math.floor((x - raster.bounds[0]) / raster.cellSize);
  const row = Math.floor((y - raster.bounds[1]) / raster.cellSize);
  if (column < 0 || row < 0 || column >= raster.columns || row >= raster.rows) return -1;
  const zone = raster.zoneIds[row * raster.columns + column];
  return zone === raster.zoneCount ? -1 : zone;
}
