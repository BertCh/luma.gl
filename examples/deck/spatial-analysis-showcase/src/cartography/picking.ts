// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Point and polygon hit testing on the CPU (G14): nearest-point index, raster cell lookup and
 * point-in-polygon, for tooltips, probes and "which feature is under the pointer" without GPU
 * readback. Pure TypeScript.
 */

import type {GeoJsonCollection, GeoJsonFeature, GeoJsonGeometry} from '../data/loaders';
import {haversineDistance} from './reference-geometry';
import type {LngLat} from './types';

/** A ring of `[lng, lat, ...]` vertices (GeoJSON coordinates). */
export type PolygonRing = readonly (readonly number[])[];

/** A polygon: the outer ring first, then holes. */
export type PolygonRings = readonly PolygonRing[];

/** `[west, south, east, north]` in degrees. */
export type LngLatBounds = readonly [number, number, number, number];

/** Anything that carries polygons: a collection, a feature or a bare geometry. */
export type GeoJsonInput = GeoJsonCollection | GeoJsonFeature | GeoJsonGeometry;

/** Metres per degree of latitude on the spherical earth of `reference-geometry.ts`. */
const METERS_PER_DEGREE = 111_195;

// ---------------------------------------------------------------------------------------------
// Polygon helpers
// ---------------------------------------------------------------------------------------------

/**
 * Returns the polygons of a GeoJSON geometry (`Polygon` gives one, `MultiPolygon` many, every
 * other type none).
 */
export function getGeometryPolygons(geometry: GeoJsonGeometry | null | undefined): PolygonRings[] {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates as PolygonRings];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates as PolygonRings[];
  return [];
}

/**
 * Returns the polygons of every feature of a GeoJSON input, with the index of the feature each
 * polygon came from (0 for a bare feature or geometry).
 */
export function getInputPolygons(
  input: GeoJsonInput
): {polygon: PolygonRings; featureIndex: number}[] {
  const result: {polygon: PolygonRings; featureIndex: number}[] = [];
  const features = getInputFeatures(input);
  features.forEach((geometry, featureIndex) => {
    for (const polygon of getGeometryPolygons(geometry)) result.push({polygon, featureIndex});
  });
  return result;
}

function getInputFeatures(input: GeoJsonInput): (GeoJsonGeometry | null)[] {
  if (input.type === 'FeatureCollection') {
    return (input as GeoJsonCollection).features.map(feature => feature.geometry);
  }
  if (input.type === 'Feature') return [(input as GeoJsonFeature).geometry];
  return [input as GeoJsonGeometry];
}

/** Returns the `[west, south, east, north]` box of rings, or `null` for no vertices. */
export function getRingsBounds(rings: PolygonRings): LngLatBounds | null {
  let west = Number.POSITIVE_INFINITY;
  let south = Number.POSITIVE_INFINITY;
  let east = Number.NEGATIVE_INFINITY;
  let north = Number.NEGATIVE_INFINITY;
  for (const ring of rings) {
    for (const vertex of ring) {
      if (vertex[0] < west) west = vertex[0];
      if (vertex[0] > east) east = vertex[0];
      if (vertex[1] < south) south = vertex[1];
      if (vertex[1] > north) north = vertex[1];
    }
  }
  return west <= east ? [west, south, east, north] : null;
}

/**
 * Even-odd point-in-polygon over all rings of one polygon: a point inside the outer ring and
 * outside every hole is inside. Points exactly on an edge may fall either way.
 *
 * @example
 * pointInPolygon([0.5, 0.5], [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]); // true
 */
export function pointInPolygon(lngLat: readonly number[], rings: PolygonRings): boolean {
  const x = lngLat[0];
  const y = lngLat[1];
  let inside = false;
  for (const ring of rings) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][0];
      const yi = ring[i][1];
      const xj = ring[j][0];
      const yj = ring[j][1];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/** The result of {@link findContainingFeature}. */
export type ContainingFeature = {feature: GeoJsonFeature; index: number};

/** Finds features containing points; build once with {@link createFeatureLocator}. */
export type FeatureLocator = {
  /** The first feature (in collection order) that contains `lngLat`, or `null`. */
  find: (lngLat: readonly number[]) => ContainingFeature | null;
  /** Every feature that contains `lngLat` (overlapping polygons), in collection order. */
  findAll: (lngLat: readonly number[]) => ContainingFeature[];
};

/**
 * Prepares a feature locator: the bounding box of every polygon is measured once, so each query
 * tests only the polygons whose box contains the point. Use it for hover (many queries); use
 * {@link findContainingFeature} for a single query.
 */
export function createFeatureLocator(geojson: GeoJsonCollection): FeatureLocator {
  const entries: {polygon: PolygonRings; bounds: LngLatBounds; featureIndex: number}[] = [];
  geojson.features.forEach((feature, featureIndex) => {
    for (const polygon of getGeometryPolygons(feature.geometry)) {
      const bounds = getRingsBounds(polygon);
      if (bounds) entries.push({polygon, bounds, featureIndex});
    }
  });
  const query = (lngLat: readonly number[], stopAtFirst: boolean): ContainingFeature[] => {
    const hits: ContainingFeature[] = [];
    let last = -1;
    for (const {polygon, bounds, featureIndex} of entries) {
      if (featureIndex === last && stopAtFirst === false) continue;
      if (
        lngLat[0] < bounds[0] ||
        lngLat[0] > bounds[2] ||
        lngLat[1] < bounds[1] ||
        lngLat[1] > bounds[3]
      ) {
        continue;
      }
      if (pointInPolygon(lngLat, polygon)) {
        hits.push({feature: geojson.features[featureIndex], index: featureIndex});
        if (stopAtFirst) return hits;
        last = featureIndex;
      }
    }
    return hits;
  };
  return {
    find: lngLat => query(lngLat, true)[0] ?? null,
    findAll: lngLat => query(lngLat, false)
  };
}

/**
 * One-shot containing-feature query (bounding-box prefilter, then even-odd test). For repeated
 * queries on the same collection use {@link createFeatureLocator}.
 *
 * @example
 * const hit = findContainingFeature(counties, [-87.63, 41.88]);
 * hit?.feature.properties?.name; // 'Cook'
 */
export function findContainingFeature(
  geojson: GeoJsonCollection,
  lngLat: readonly number[]
): ContainingFeature | null {
  return createFeatureLocator(geojson).find(lngLat);
}

// ---------------------------------------------------------------------------------------------
// Nearest-point index
// ---------------------------------------------------------------------------------------------

/** Options of {@link createNearestIndex}. */
export type NearestIndexOptions = {
  /**
   * Edge of a grid cell in degrees. Default: chosen so a cell holds about two points on average
   * (about `sqrt(2 * area / count)`), at least 0.001 degrees.
   */
  cellDegrees?: number;
};

/** A found point: its row in the positions array and the great-circle distance. */
export type NearestResult = {index: number; distanceMeters: number};

/** Spatial index over points; see {@link createNearestIndex}. */
export type NearestIndex = {
  /** Number of indexed points. */
  readonly count: number;
  /**
   * The closest point to `lngLat` or `null` when there is none within `maxDistanceMeters`
   * (default: unlimited). Ties go to the lower index.
   */
  nearest: (lngLat: readonly number[], maxDistanceMeters?: number) => NearestResult | null;
  /** Indices of every point within `radiusMeters`, closest first. */
  within: (lngLat: readonly number[], radiusMeters: number) => number[];
};

/**
 * Builds a uniform-grid index over interleaved `lng, lat` positions (a CSR bucket layout: three
 * typed arrays, no per-point objects). Building is linear; a query touches only the cells around
 * the pointer, so 3,257 airports or 905 stations answer instantly and 100,000 points answer in
 * well under a millisecond for typical hover radii. Distances are great-circle metres
 * ({@link haversineDistance}); longitude wrap at the antimeridian is not handled. Non-finite
 * positions are skipped.
 *
 * @example
 * const index = createNearestIndex(airports.position.data);
 * const hit = index.nearest([lng, lat], 50_000); // within 50 km, else null
 * if (hit) showTooltip(hit.index, hit.distanceMeters);
 * const nearby = index.within([lng, lat], 200_000); // indices, closest first
 */
export function createNearestIndex(
  positions: ArrayLike<number>,
  options: NearestIndexOptions = {}
): NearestIndex {
  const count = Math.floor(positions.length / 2);
  let west = Number.POSITIVE_INFINITY;
  let south = Number.POSITIVE_INFINITY;
  let east = Number.NEGATIVE_INFINITY;
  let north = Number.NEGATIVE_INFINITY;
  let finiteCount = 0;
  for (let i = 0; i < count; i++) {
    const lng = positions[i * 2];
    const lat = positions[i * 2 + 1];
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) continue;
    finiteCount++;
    if (lng < west) west = lng;
    if (lng > east) east = lng;
    if (lat < south) south = lat;
    if (lat > north) north = lat;
  }
  if (finiteCount === 0) {
    return {count, nearest: () => null, within: () => []};
  }

  const spanLng = Math.max(east - west, 1e-6);
  const spanLat = Math.max(north - south, 1e-6);
  let cell =
    options.cellDegrees ?? Math.max(0.001, Math.sqrt((2 * spanLng * spanLat) / finiteCount));
  // Keep the grid under about four million cells for degenerate spreads.
  while ((spanLng / cell + 1) * (spanLat / cell + 1) > 4e6) cell *= 2;
  const columns = Math.floor(spanLng / cell) + 1;
  const rows = Math.floor(spanLat / cell) + 1;

  const cellOf = new Int32Array(count).fill(-1);
  const offsets = new Uint32Array(columns * rows + 1);
  for (let i = 0; i < count; i++) {
    const lng = positions[i * 2];
    const lat = positions[i * 2 + 1];
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) continue;
    const column = Math.min(columns - 1, Math.floor((lng - west) / cell));
    const row = Math.min(rows - 1, Math.floor((lat - south) / cell));
    const id = row * columns + column;
    cellOf[i] = id;
    offsets[id + 1]++;
  }
  for (let id = 0; id < columns * rows; id++) offsets[id + 1] += offsets[id];
  const fill = offsets.slice(0, columns * rows);
  const members = new Uint32Array(finiteCount);
  for (let i = 0; i < count; i++) {
    if (cellOf[i] >= 0) members[fill[cellOf[i]]++] = i;
  }

  const distanceTo = (lngLat: readonly number[], i: number) =>
    haversineDistance(lngLat as LngLat, [positions[i * 2], positions[i * 2 + 1]]);

  const nearest = (
    lngLat: readonly number[],
    maxDistanceMeters = Number.POSITIVE_INFINITY
  ): NearestResult | null => {
    if (!Number.isFinite(lngLat[0]) || !Number.isFinite(lngLat[1])) return null;
    // Unclamped cell of the query: the ring bound below holds even for queries off the grid.
    const queryColumn = Math.floor((lngLat[0] - west) / cell);
    const queryRow = Math.floor((lngLat[1] - south) / cell);
    const maxRing =
      Math.max(
        Math.abs(queryColumn),
        Math.abs(queryColumn - columns),
        Math.abs(queryRow),
        Math.abs(queryRow - rows)
      ) + 1;
    let bestIndex = -1;
    let bestDistance = maxDistanceMeters;
    for (let ring = 0; ring <= maxRing; ring++) {
      // Everything in ring r is at least (r - 1) cells away; a cell is as narrow as its
      // longitude width at the highest latitude the ring can reach.
      const reachLat = Math.min(89.99, Math.abs(lngLat[1]) + ring * cell);
      const cellMeters =
        cell *
        METERS_PER_DEGREE *
        Math.min(1, Math.max(0.01, Math.cos((reachLat * Math.PI) / 180)));
      if ((ring - 1) * cellMeters > bestDistance) break;
      const rowMin = queryRow - ring;
      const rowMax = queryRow + ring;
      for (let row = Math.max(0, rowMin); row <= Math.min(rows - 1, rowMax); row++) {
        const edgeRow = row === rowMin || row === rowMax;
        const step = edgeRow ? 1 : Math.max(1, 2 * ring);
        for (let column = queryColumn - ring; column <= queryColumn + ring; column += step) {
          if (column < 0 || column >= columns) continue;
          const id = row * columns + column;
          for (let k = offsets[id]; k < offsets[id + 1]; k++) {
            const i = members[k];
            const distance = distanceTo(lngLat, i);
            if (
              distance < bestDistance ||
              (distance === bestDistance && (bestIndex < 0 || i < bestIndex))
            ) {
              bestDistance = distance;
              bestIndex = i;
            }
          }
        }
      }
    }
    return bestIndex < 0 ? null : {index: bestIndex, distanceMeters: bestDistance};
  };

  const within = (lngLat: readonly number[], radiusMeters: number): number[] => {
    if (!(radiusMeters >= 0) || !Number.isFinite(lngLat[0]) || !Number.isFinite(lngLat[1])) {
      return [];
    }
    const lat = lngLat[1];
    const halfLat = radiusMeters / METERS_PER_DEGREE;
    const cosLat = Math.max(
      0.01,
      Math.cos((Math.min(89.99, Math.abs(lat) + halfLat) * Math.PI) / 180)
    );
    const halfLng = radiusMeters / (METERS_PER_DEGREE * cosLat);
    const column0 = Math.max(0, Math.floor((lngLat[0] - halfLng - west) / cell));
    const column1 = Math.min(columns - 1, Math.floor((lngLat[0] + halfLng - west) / cell));
    const row0 = Math.max(0, Math.floor((lat - halfLat - south) / cell));
    const row1 = Math.min(rows - 1, Math.floor((lat + halfLat - south) / cell));
    const found: NearestResult[] = [];
    for (let row = row0; row <= row1; row++) {
      for (let column = column0; column <= column1; column++) {
        const id = row * columns + column;
        for (let k = offsets[id]; k < offsets[id + 1]; k++) {
          const distance = distanceTo(lngLat, members[k]);
          if (distance <= radiusMeters) found.push({index: members[k], distanceMeters: distance});
        }
      }
    }
    found.sort((a, b) => a.distanceMeters - b.distanceMeters || a.index - b.index);
    return found.map(result => result.index);
  };

  return {count, nearest, within};
}

// ---------------------------------------------------------------------------------------------
// Rasters
// ---------------------------------------------------------------------------------------------

/**
 * A raster grid for lookups. `values` is row-major, `width * height` long (for a time or band
 * stack pass the `subarray` of one slice).
 */
export type RasterGrid = {
  width: number;
  height: number;
  values: ArrayLike<number>;
  /** `[west, south, east, north]` of the outer edges of the grid. */
  bounds: LngLatBounds;
  /**
   * Which edge row 0 is on. `'north'` (default) is the convention of `DecodedRaster` in
   * `data/loaders.ts` ("row 0 at the north edge"); `'south'` for grids stored south-up.
   */
  rowOrigin?: 'north' | 'south';
};

/** A raster cell and its value. */
export type RasterSample = {column: number; row: number; value: number};

/**
 * Returns the raster cell containing `lngLat` and its value, or `null` outside the bounds.
 * Cells are half-open; the east and north edges belong to the last cell. `noData` handling
 * (NaN, a sentinel, scale and offset) stays with the caller, which reads `value`.
 *
 * @example
 * // Raster decoded by data/loaders.ts: row 0 is the north edge.
 * const hit = getRasterValueAt({width, height, values, bounds}, [lng, lat]);
 * if (hit) tooltip(`${hit.value.toFixed(1)} m at cell ${hit.column}, ${hit.row}`);
 */
export function getRasterValueAt(
  raster: RasterGrid,
  lngLat: readonly number[]
): RasterSample | null {
  const [west, south, east, north] = raster.bounds;
  const lng = lngLat[0];
  const lat = lngLat[1];
  if (!(lng >= west && lng <= east && lat >= south && lat <= north)) return null;
  const column = Math.min(
    raster.width - 1,
    Math.floor(((lng - west) / (east - west)) * raster.width)
  );
  const fromSouth = (lat - south) / (north - south);
  const rowFraction = raster.rowOrigin === 'south' ? fromSouth : 1 - fromSouth;
  const row = Math.min(raster.height - 1, Math.floor(rowFraction * raster.height));
  return {column, row, value: raster.values[row * raster.width + column]};
}

/**
 * Returns `[west, south, east, north]` of one raster cell, ready for a `{kind: 'box'}` highlight.
 *
 * @example
 * ctx.setHighlight({kind: 'box', bounds: getCellBounds(raster, hit.column, hit.row)});
 */
export function getCellBounds(raster: RasterGrid, column: number, row: number): LngLatBounds {
  const [west, south, east, north] = raster.bounds;
  const cellWidth = (east - west) / raster.width;
  const cellHeight = (north - south) / raster.height;
  const fromSouth = raster.rowOrigin === 'south' ? row : raster.height - 1 - row;
  return [
    west + column * cellWidth,
    south + fromSouth * cellHeight,
    west + (column + 1) * cellWidth,
    south + (fromSouth + 1) * cellHeight
  ];
}
