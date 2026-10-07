// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Data-snapping resolvers: move a hand-placed label coordinate onto the data it names (the
 * highest DEM cell near a summit, the nearest station row, the visual centre of a polygon). Pure
 * TypeScript, no GPU. All coordinates are `[longitude, latitude]` in degrees.
 */

import type {GeoJsonGeometry} from '../data/loaders';
import type {Place} from './gazetteer/types';
import {getFeatureLabelPoint} from './polygon-mesh';

/** Mean Earth radius in metres used by the haversine formula. */
const EARTH_RADIUS_METERS = 6371008.8;

/** Metres per degree of latitude (spherical Earth). */
const METERS_PER_DEGREE = (Math.PI / 180) * EARTH_RADIUS_METERS;

/**
 * Great-circle distance in metres between two `[longitude, latitude]` points.
 * Example: `haversineMeters([-87.63, 41.88], [-87.9, 41.98])` is about 24,000.
 */
export function haversineMeters(
  a: readonly [number, number],
  b: readonly [number, number]
): number {
  const radians = Math.PI / 180;
  const deltaLat = (b[1] - a[1]) * radians;
  const deltaLng = (b[0] - a[0]) * radians;
  const h =
    Math.sin(deltaLat / 2) ** 2 +
    Math.cos(a[1] * radians) * Math.cos(b[1] * radians) * Math.sin(deltaLng / 2) ** 2;
  return 2 * EARTH_RADIUS_METERS * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** A raster on a regular lng/lat grid, as the loaders return DEMs and analysis grids. */
export type SnapRaster = {
  width: number;
  height: number;
  /** `height * width` values, row-major. Non-finite values (NaN, nodata) are ignored. */
  values: ArrayLike<number>;
  /** `[west, south, east, north]` of the outer cell edges. */
  bounds: readonly [number, number, number, number];
  /**
   * Which edge row 0 is on. Default `'north'`, as in the loaders' rasters (see `b15-common.ts`:
   * "row 0 at the north edge"); pass `'south'` for rasters flipped to row 0 at the minimum y.
   */
  rowOrigin?: 'north' | 'south';
};

/** Result of a snap to a raster cell. */
export type SnappedCell = {
  /** Centre of the winning cell. */
  lngLat: [number, number];
  value: number;
};

/**
 * Moves a summit label to the highest cell of a raster within `radiusMeters` of `lngLat`
 * (cell centres are compared). Ties go to the cell nearest the input. Returns `null` when the
 * circle misses the raster or holds no finite value.
 * Example: `snapToHighestCell({width, height, values, bounds}, [7.6586, 45.9766], 200)`.
 */
export function snapToHighestCell(
  raster: SnapRaster,
  lngLat: readonly [number, number],
  radiusMeters = 160
): SnappedCell | null {
  const {width, height, values, bounds} = raster;
  const [west, south, east, north] = bounds;
  const cellWidth = (east - west) / width;
  const cellHeight = (north - south) / height;
  const fromSouth = raster.rowOrigin === 'south';
  const latitudeRadians = (lngLat[1] * Math.PI) / 180;
  const radiusLat = radiusMeters / METERS_PER_DEGREE;
  const radiusLng = radiusMeters / (METERS_PER_DEGREE * Math.max(0.01, Math.cos(latitudeRadians)));

  const firstColumn = Math.max(0, Math.floor((lngLat[0] - radiusLng - west) / cellWidth));
  const lastColumn = Math.min(width - 1, Math.floor((lngLat[0] + radiusLng - west) / cellWidth));
  const firstFromSouth = Math.max(0, Math.floor((lngLat[1] - radiusLat - south) / cellHeight));
  const lastFromSouth = Math.min(
    height - 1,
    Math.floor((lngLat[1] + radiusLat - south) / cellHeight)
  );

  let best: SnappedCell | null = null;
  let bestDistance = Infinity;
  for (let fromSouthRow = firstFromSouth; fromSouthRow <= lastFromSouth; fromSouthRow++) {
    const row = fromSouth ? fromSouthRow : height - 1 - fromSouthRow;
    const centerLat = south + (fromSouthRow + 0.5) * cellHeight;
    for (let column = firstColumn; column <= lastColumn; column++) {
      const value = values[row * width + column];
      if (!Number.isFinite(value)) continue;
      const center: [number, number] = [west + (column + 0.5) * cellWidth, centerLat];
      const distance = haversineMeters(lngLat, center);
      if (distance > radiusMeters) continue;
      if (!best || value > best.value || (value === best.value && distance < bestDistance)) {
        best = {lngLat: center, value};
        bestDistance = distance;
      }
    }
  }
  return best;
}

/** Result of a snap to a point or row. */
export type SnappedPoint = {
  /** Index of the point in the interleaved array. */
  index: number;
  lngLat: [number, number];
  distanceMeters: number;
};

/**
 * Finds the point of an interleaved `[lng, lat, lng, lat, ...]` array nearest to `lngLat`. Returns
 * `null` for an empty array or when nothing is within `maxDistanceMeters` (default: no limit).
 * Example: `snapToNearestPoint(stationPositions, [-87.63, 41.88], 500)`.
 */
export function snapToNearestPoint(
  positions: ArrayLike<number>,
  lngLat: readonly [number, number],
  maxDistanceMeters = Infinity
): SnappedPoint | null {
  const count = Math.floor(positions.length / 2);
  // Equirectangular distance picks the winner; the haversine gives the reported distance.
  const lngScale = Math.cos((lngLat[1] * Math.PI) / 180);
  let bestIndex = -1;
  let bestSquared = Infinity;
  for (let i = 0; i < count; i++) {
    const deltaLng = (positions[2 * i] - lngLat[0]) * lngScale;
    const deltaLat = positions[2 * i + 1] - lngLat[1];
    const squared = deltaLng * deltaLng + deltaLat * deltaLat;
    if (squared < bestSquared) {
      bestSquared = squared;
      bestIndex = i;
    }
  }
  if (bestIndex < 0) return null;
  const found: [number, number] = [positions[2 * bestIndex], positions[2 * bestIndex + 1]];
  const distanceMeters = haversineMeters(lngLat, found);
  if (distanceMeters > maxDistanceMeters) return null;
  return {index: bestIndex, lngLat: found, distanceMeters};
}

/** Result of a snap to a table row. */
export type SnappedRow<Row> = SnappedPoint & {row: Row};

/**
 * Finds the row (a station, a facility, a tract centroid) nearest to `lngLat`. Returns `null` for
 * an empty table or when nothing is within `maxDistanceMeters` (default: no limit).
 * Example: `snapToNearestRow(stations, [-87.63, 41.88], 800)?.row.name`.
 */
export function snapToNearestRow<Row extends {lngLat: readonly [number, number]}>(
  rows: readonly Row[],
  lngLat: readonly [number, number],
  maxDistanceMeters = Infinity
): SnappedRow<Row> | null {
  const lngScale = Math.cos((lngLat[1] * Math.PI) / 180);
  let bestIndex = -1;
  let bestSquared = Infinity;
  for (let i = 0; i < rows.length; i++) {
    const deltaLng = (rows[i].lngLat[0] - lngLat[0]) * lngScale;
    const deltaLat = rows[i].lngLat[1] - lngLat[1];
    const squared = deltaLng * deltaLng + deltaLat * deltaLat;
    if (squared < bestSquared) {
      bestSquared = squared;
      bestIndex = i;
    }
  }
  if (bestIndex < 0) return null;
  const row = rows[bestIndex];
  const distanceMeters = haversineMeters(lngLat, row.lngLat);
  if (distanceMeters > maxDistanceMeters) return null;
  return {index: bestIndex, lngLat: [row.lngLat[0], row.lngLat[1]], distanceMeters, row};
}

/**
 * Label point of a polygon given as rings (outer ring first, then holes) of `[lng, lat]`: an
 * approximate pole of inaccessibility, the same routine the polygon meshes use for their label
 * points (`getFeatureLabelPoint`). Returns `null` for fewer than three vertices.
 * Example: `getPolygonLabelPoint([[[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]])` is about `[2, 1]`.
 */
export function getPolygonLabelPoint(
  rings: readonly (readonly (readonly [number, number])[])[]
): [number, number] | null {
  const geometry: GeoJsonGeometry = {type: 'Polygon', coordinates: rings};
  return getFeatureLabelPoint(geometry);
}

/**
 * Returns copies of `places` whose `lngLat` is replaced by what `resolver` returns; places for
 * which it returns `null` keep their coordinate. The input places are not modified.
 * Example: `resolveAnchors(places, place => snapToNearestPoint(stops, place.lngLat, 300)?.lngLat ?? null)`.
 */
export function resolveAnchors(
  places: readonly Place[],
  resolver: (place: Place) => readonly [number, number] | null
): Place[] {
  return places.map(place => {
    const resolved = resolver(place);
    return resolved ? {...place, lngLat: [resolved[0], resolved[1]] as const} : {...place};
  });
}
