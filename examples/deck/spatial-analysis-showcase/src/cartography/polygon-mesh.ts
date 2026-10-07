// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GeoJSON polygons to GPU-ready arrays (triangles, outlines, label points, areas). Pure TypeScript.
 */

import earcut from 'earcut';
import type {GeoJsonCollection, GeoJsonFeature, GeoJsonGeometry} from '../data/loaders';

/** Triangulated polygon features ready for GPU upload. */
export type PolygonMesh = {
  featureCount: number;
  /** float32x2 triangle-list vertices in planar metres (3 per triangle). */
  triangles: Float32Array;
  /** uint32 feature row of every triangle vertex. */
  triangleFeatures: Uint32Array;
  /** float32x4 outline segments x0, y0, x1, y1 (every ring edge once per ring). */
  outlineSegments: Float32Array;
  /** uint32 feature row of every outline segment. */
  outlineFeatures: Uint32Array;
  /** [lng, lat] label point per feature (inside the largest polygon), for annotations. NaN when there is no polygon. */
  labelPoints: [number, number][];
  /** Area in square metres per feature (planar, after projection). 0 for filtered-out features. */
  areas: Float64Array;
};

type Ring = number[][];
type Polygon = Ring[];

/** Extracts polygons (outer ring first, then holes) from a Polygon or MultiPolygon geometry. */
function getPolygons(geometry: GeoJsonGeometry | null): Polygon[] {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates as Polygon];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates as Polygon[];
  return [];
}

/** Ring without its duplicate closing vertex. */
function openRing(ring: Ring): Ring {
  const n = ring.length;
  if (n > 1 && ring[0][0] === ring[n - 1][0] && ring[0][1] === ring[n - 1][1]) {
    return ring.slice(0, n - 1);
  }
  return ring;
}

/** Signed shoelace area of a ring given as vertices; positive when counter-clockwise. */
function getSignedArea(points: ArrayLike<readonly number[]>): number {
  let area = 0;
  for (let i = 0, n = points.length; i < n; i++) {
    const a = points[i];
    const b = points[(i + 1) % n];
    area += a[0] * b[1] - b[0] * a[1];
  }
  return area / 2;
}

/** Squared distance from point `(px, py)` to segment `a`-`b`. */
function getSegmentDistanceSquared(px: number, py: number, a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSquared = dx * dx + dy * dy;
  const t =
    lengthSquared > 0
      ? Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / lengthSquared))
      : 0;
  const cx = a[0] + t * dx - px;
  const cy = a[1] + t * dy - py;
  return cx * cx + cy * cy;
}

/** Signed distance to the polygon boundary: positive inside (even-odd over all rings), negative outside. */
function getSignedBoundaryDistance(px: number, py: number, rings: Ring[]): number {
  let inside = false;
  let best = Infinity;
  for (const ring of rings) {
    for (let i = 0, n = ring.length, j = n - 1; i < n; j = i++) {
      const a = ring[i];
      const b = ring[j];
      if (a[1] > py !== b[1] > py && px < ((b[0] - a[0]) * (py - a[1])) / (b[1] - a[1]) + a[0]) {
        inside = !inside;
      }
      best = Math.min(best, getSegmentDistanceSquared(px, py, a, b));
    }
  }
  const distance = Math.sqrt(best);
  return inside ? distance : -distance;
}

/** Area centroid of the outer ring, or its vertex mean when degenerate. */
function getRingCentroid(ring: Ring): [number, number] {
  let area = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0, n = ring.length; i < n; i++) {
    const a = ring[i];
    const b = ring[(i + 1) % n];
    const cross = a[0] * b[1] - b[0] * a[1];
    area += cross;
    cx += (a[0] + b[0]) * cross;
    cy += (a[1] + b[1]) * cross;
  }
  if (Math.abs(area) < 1e-18) {
    const n = Math.max(1, ring.length);
    return [ring.reduce((sum, p) => sum + p[0], 0) / n, ring.reduce((sum, p) => sum + p[1], 0) / n];
  }
  return [cx / (3 * area), cy / (3 * area)];
}

/**
 * Label point of a (Multi)Polygon: an approximate pole of inaccessibility of its largest polygon.
 * Coarse 12 x 12 grid search for the interior point farthest from every ring edge, refined three
 * times around the best cell; falls back to the area centroid. Computed in degrees with longitude
 * scaled by cos(latitude). Returns `null` for other geometries.
 */
export function getFeatureLabelPoint(geometry: GeoJsonGeometry | null): [number, number] | null {
  const polygons = getPolygons(geometry)
    .map(polygon => polygon.map(openRing).filter(ring => ring.length >= 3))
    .filter(polygon => polygon.length > 0);
  if (polygons.length === 0) return null;

  const outer = polygons[0][0];
  const scale = Math.max(0.05, Math.cos((outer[0][1] * Math.PI) / 180));
  const scaled = polygons.map(polygon =>
    polygon.map(ring => ring.map(point => [point[0] * scale, point[1]]))
  );
  let largest = 0;
  let largestArea = -1;
  scaled.forEach((polygon, index) => {
    const area = Math.abs(getSignedArea(polygon[0]));
    if (area > largestArea) {
      largestArea = area;
      largest = index;
    }
  });
  const rings = scaled[largest];

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of rings[0]) {
    minX = Math.min(minX, point[0]);
    maxX = Math.max(maxX, point[0]);
    minY = Math.min(minY, point[1]);
    maxY = Math.max(maxY, point[1]);
  }

  let bestX = NaN;
  let bestY = NaN;
  let bestDistance = -Infinity;
  const search = (x0: number, y0: number, x1: number, y1: number, steps: number) => {
    for (let i = 0; i < steps; i++) {
      for (let j = 0; j < steps; j++) {
        const x = x0 + ((i + 0.5) / steps) * (x1 - x0);
        const y = y0 + ((j + 0.5) / steps) * (y1 - y0);
        const distance = getSignedBoundaryDistance(x, y, rings);
        if (distance > bestDistance) {
          bestDistance = distance;
          bestX = x;
          bestY = y;
        }
      }
    }
  };
  search(minX, minY, maxX, maxY, 12);
  let halfWidth = (maxX - minX) / 12;
  let halfHeight = (maxY - minY) / 12;
  for (let round = 0; round < 3 && bestDistance > 0; round++) {
    search(bestX - halfWidth, bestY - halfHeight, bestX + halfWidth, bestY + halfHeight, 6);
    halfWidth /= 3;
    halfHeight /= 3;
  }
  if (bestDistance > 0) return [bestX / scale, bestY];
  const centroid = getRingCentroid(rings[0]);
  return [centroid[0] / scale, centroid[1]];
}

/**
 * Triangulates and outlines polygon features. The feature row is the index in the input; filtered-out
 * and non-polygon features get no triangles or outlines but keep their row so value buffers line up.
 * Rings are projected with `project` first, so earcut runs in planar metres; holes are honoured.
 */
export function buildPolygonMesh(
  collection: GeoJsonCollection | readonly GeoJsonFeature[],
  project: (longitude: number, latitude: number) => readonly [number, number],
  options: {filter?: (feature: GeoJsonFeature, index: number) => boolean} = {}
): PolygonMesh {
  const features = Array.isArray(collection)
    ? (collection as readonly GeoJsonFeature[])
    : (collection as GeoJsonCollection).features;
  const triangleValues: number[] = [];
  const triangleFeatureRows: number[] = [];
  const outlineValues: number[] = [];
  const outlineFeatureRows: number[] = [];
  const labelPoints: [number, number][] = [];
  const areas = new Float64Array(features.length);

  features.forEach((feature, row) => {
    labelPoints.push(getFeatureLabelPoint(feature.geometry) ?? [NaN, NaN]);
    if (options.filter && !options.filter(feature, row)) return;

    for (const polygon of getPolygons(feature.geometry)) {
      const rings = polygon
        .map(ring => openRing(ring).map(point => project(point[0], point[1])))
        .filter(ring => ring.length >= 3);
      if (rings.length === 0) continue;

      const flat: number[] = [];
      const holeIndices: number[] = [];
      rings.forEach((ring, ringIndex) => {
        if (ringIndex > 0) holeIndices.push(flat.length / 2);
        for (const [x, y] of ring) flat.push(x, y);
        for (let i = 0, n = ring.length; i < n; i++) {
          const a = ring[i];
          const b = ring[(i + 1) % n];
          outlineValues.push(a[0], a[1], b[0], b[1]);
          outlineFeatureRows.push(row);
        }
      });

      areas[row] += Math.abs(getSignedArea(rings[0]));
      for (let i = 1; i < rings.length; i++) areas[row] -= Math.abs(getSignedArea(rings[i]));

      const indices = earcut(flat, holeIndices.length > 0 ? holeIndices : null, 2);
      for (const index of indices) {
        triangleValues.push(flat[index * 2], flat[index * 2 + 1]);
        triangleFeatureRows.push(row);
      }
    }
  });

  return {
    featureCount: features.length,
    triangles: new Float32Array(triangleValues),
    triangleFeatures: new Uint32Array(triangleFeatureRows),
    outlineSegments: new Float32Array(outlineValues),
    outlineFeatures: new Uint32Array(outlineFeatureRows),
    labelPoints,
    areas
  };
}
