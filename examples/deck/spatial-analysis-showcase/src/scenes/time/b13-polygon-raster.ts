// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {LocalMetricProjection} from '../../engine/projection';

/**
 * GeoArrow-style polygon layout of a shipped polygon dataset, normalised so that county datasets
 * (feature to polygon offsets) and tract or community-area datasets (part to feature) look alike.
 * Coordinates are longitude and latitude degrees.
 */
export type B13PolygonSource = {
  featureCount: number;
  /** Interleaved `lng, lat` vertices. */
  vertices: Float32Array;
  /** Ring to vertex offsets. */
  ringOffsets: Uint32Array;
  /** Part (polygon) to ring offsets, with a terminal entry. */
  polygonRingOffsets: Uint32Array;
  /** Feature of every part. */
  partFeature: Uint32Array;
};

/** Reads the polygon columns of a dataset. */
export function getPolygonSource(dataset: LoadedDataset): B13PolygonSource {
  const vertices = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonRingOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partCount = polygonRingOffsets.length - 1;
  if (dataset.hasColumn('partFeature')) {
    return {
      featureCount: dataset.count,
      vertices,
      ringOffsets,
      polygonRingOffsets,
      partFeature: dataset.column<Uint32Array>('partFeature')
    };
  }
  // Counties: feature i owns parts [countyPolygonOffsets[i], countyPolygonOffsets[i + 1]).
  const featureOffsets = dataset.column<Uint32Array>('countyPolygonOffsets');
  const featureCount = featureOffsets.length - 1;
  const partFeature = new Uint32Array(partCount);
  for (let feature = 0; feature < featureCount; feature++) {
    for (let part = featureOffsets[feature]; part < featureOffsets[feature + 1]; part++) {
      partFeature[part] = feature;
    }
  }
  return {featureCount, vertices, ringOffsets, polygonRingOffsets, partFeature};
}

/** A polygon set painted into a grid that is uniform in Web Mercator. */
export type B13PolygonRaster = {
  width: number;
  height: number;
  /** Feature index of every cell, row 0 at the south; `featureCount` marks "no feature". */
  cellFeature: Uint32Array;
  /** `[minX, minY, maxX, maxY]` meters around the projection origin, for the raster layer. */
  bounds: [number, number, number, number];
  featureCount: number;
  /** Grid origin and cell size in (longitude, Mercator y) degrees, for {@link lookupFeature}. */
  origin: readonly [number, number];
  cellSize: number;
};

const toMercatorY = (latitude: number) =>
  (Math.log(Math.tan(Math.PI / 4 + (latitude * Math.PI) / 360)) * 180) / Math.PI;
const fromMercatorY = (value: number) =>
  ((2 * Math.atan(Math.exp((value * Math.PI) / 180)) - Math.PI / 2) * 180) / Math.PI;

/**
 * Paints every polygon into a regular Web Mercator grid with an even-odd scanline fill, once on
 * the CPU. The grid is uniform in Mercator space, which is what deck.gl's meter-offset
 * coordinates are linear in, so a raster layer over `bounds` lines up with the basemap. The
 * result is static: choropleth values then come from GPU buffers indexed by `cellFeature`.
 */
export function rasterizePolygons(
  source: B13PolygonSource,
  projection: LocalMetricProjection,
  bbox: readonly [number, number, number, number],
  width: number
): B13PolygonRaster {
  const [west, south, east, north] = bbox;
  const margin = 0.002 * (east - west);
  const u0 = west - margin;
  const u1 = east + margin;
  const v0 = toMercatorY(south) - margin;
  const v1 = toMercatorY(north) + margin;
  const cellSize = (u1 - u0) / width;
  const height = Math.ceil((v1 - v0) / cellSize);
  const cellFeature = new Uint32Array(width * height).fill(source.featureCount);
  const {vertices, ringOffsets, polygonRingOffsets, partFeature} = source;
  const partCount = polygonRingOffsets.length - 1;
  const crossings: number[] = [];
  const gridX = new Float64Array(vertices.length / 2);
  const gridY = new Float64Array(vertices.length / 2);
  for (let index = 0; index < gridX.length; index++) {
    gridX[index] = (vertices[index * 2] - u0) / cellSize;
    gridY[index] = (toMercatorY(vertices[index * 2 + 1]) - v0) / cellSize;
  }
  for (let part = 0; part < partCount; part++) {
    const firstRing = polygonRingOffsets[part];
    const lastRing = polygonRingOffsets[part + 1];
    let minY = Infinity;
    let maxY = -Infinity;
    for (let ring = firstRing; ring < lastRing; ring++) {
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
        minY = Math.min(minY, gridY[vertex]);
        maxY = Math.max(maxY, gridY[vertex]);
      }
    }
    const feature = partFeature[part];
    const firstRow = Math.max(0, Math.ceil(minY - 0.5));
    const lastRow = Math.min(height - 1, Math.floor(maxY - 0.5));
    for (let row = firstRow; row <= lastRow; row++) {
      const y = row + 0.5;
      crossings.length = 0;
      for (let ring = firstRing; ring < lastRing; ring++) {
        const start = ringOffsets[ring];
        const end = ringOffsets[ring + 1];
        for (let vertex = start; vertex < end; vertex++) {
          const next = vertex + 1 < end ? vertex + 1 : start;
          const ay = gridY[vertex];
          const by = gridY[next];
          if (ay === by || ay <= y === by <= y) continue;
          const t = (y - ay) / (by - ay);
          crossings.push(gridX[vertex] + t * (gridX[next] - gridX[vertex]));
        }
      }
      crossings.sort((a, b) => a - b);
      for (let pair = 0; pair + 1 < crossings.length; pair += 2) {
        const from = Math.max(0, Math.ceil(crossings[pair] - 0.5));
        const to = Math.min(width - 1, Math.ceil(crossings[pair + 1] - 0.5) - 1);
        const base = row * width;
        for (let column = from; column <= to; column++) cellFeature[base + column] = feature;
      }
    }
  }
  const southWest = projection.project(u0, fromMercatorY(v0));
  const northEast = projection.project(
    u0 + width * cellSize,
    fromMercatorY(v0 + height * cellSize)
  );
  return {
    width,
    height,
    cellFeature,
    bounds: [southWest[0], southWest[1], northEast[0], northEast[1]],
    featureCount: source.featureCount,
    origin: [u0, v0],
    cellSize
  };
}

/** Feature under a longitude and latitude, or -1 when the point is outside every polygon. */
export function lookupFeature(
  raster: B13PolygonRaster,
  longitude: number,
  latitude: number
): number {
  const column = Math.floor((longitude - raster.origin[0]) / raster.cellSize);
  const row = Math.floor((toMercatorY(latitude) - raster.origin[1]) / raster.cellSize);
  if (column < 0 || row < 0 || column >= raster.width || row >= raster.height) return -1;
  const feature = raster.cellFeature[row * raster.width + column];
  return feature >= raster.featureCount ? -1 : feature;
}

/** Closed ring outlines of a polygon set as `x0, y0, x1, y1` meter segments. */
export function getOutlineSegments(
  source: B13PolygonSource,
  projection: LocalMetricProjection
): Float32Array {
  const {vertices, ringOffsets} = source;
  const meters = new Float32Array((vertices.length / 2) * 2);
  for (let index = 0; index < vertices.length / 2; index++) {
    const [x, y] = projection.project(vertices[index * 2], vertices[index * 2 + 1]);
    meters[index * 2] = x;
    meters[index * 2 + 1] = y;
  }
  const segments: number[] = [];
  for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
    const start = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    for (let vertex = start; vertex < end; vertex++) {
      const next = vertex + 1 < end ? vertex + 1 : start;
      if (vertex === next) continue;
      segments.push(
        meters[vertex * 2],
        meters[vertex * 2 + 1],
        meters[next * 2],
        meters[next * 2 + 1]
      );
    }
  }
  return Float32Array.from(segments);
}

/** Ring outlines of a GeoJSON Polygon or MultiPolygon collection as `x0, y0, x1, y1` meters. */
export function getGeoJsonOutlineSegments(
  collection: {features: readonly {geometry: {type: string; coordinates: unknown}}[]},
  projection: LocalMetricProjection
): Float32Array {
  const segments: number[] = [];
  const addRing = (ring: readonly (readonly number[])[]) => {
    for (let index = 0; index + 1 < ring.length; index++) {
      const [x0, y0] = projection.project(ring[index][0], ring[index][1]);
      const [x1, y1] = projection.project(ring[index + 1][0], ring[index + 1][1]);
      segments.push(x0, y0, x1, y1);
    }
  };
  for (const feature of collection.features) {
    const {type, coordinates} = feature.geometry;
    if (type === 'Polygon') {
      for (const ring of coordinates as number[][][]) addRing(ring);
    } else if (type === 'MultiPolygon') {
      for (const polygon of coordinates as number[][][][])
        for (const ring of polygon) addRing(ring);
    }
  }
  return Float32Array.from(segments);
}
