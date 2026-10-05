// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const NO_FEATURE = 0xffffffff;

/** Feature, then polygons, then rings, then `[x, y]` vertices (rings unclosed). */
export type OraclePolygonFeature = number[][][][];

type Point = [number, number];

/** Flattens polygon features into GeoArrow-style offset arrays. */
export function buildPolygonFeatureArrays(features: OraclePolygonFeature[]): {
  polygonPositions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
} {
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const feature of features) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const [x, y] of ring) {
          positions.push(x, y);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    polygonPositions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets)
  };
}

/** Classifies one point against one feature: even/odd per polygon, union across polygons. */
export function classifyPointInFeature(
  point: Point,
  feature: OraclePolygonFeature
): 'inside' | 'boundary' | 'outside' {
  const [x, y] = point;
  let inside = false;
  for (const polygon of feature) {
    let polygonInside = false;
    for (const ring of polygon) {
      for (let index = 0; index < ring.length; index++) {
        const [ax, ay] = ring[index];
        const [bx, by] = ring[(index + 1) % ring.length];
        const cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        if (
          cross === 0 &&
          x >= Math.min(ax, bx) &&
          x <= Math.max(ax, bx) &&
          y >= Math.min(ay, by) &&
          y <= Math.max(ay, by)
        ) {
          return 'boundary';
        }
        if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) {
          polygonInside = !polygonInside;
        }
      }
    }
    inside = inside || polygonInside;
  }
  return inside ? 'inside' : 'outside';
}

/** Smallest containing feature row per point, plus per-feature counts. */
export function joinPointsInPolygons(
  points: Point[],
  features: OraclePolygonFeature[],
  includeBoundary: boolean
): {featureRows: number[]; counts: number[]} {
  const counts = new Array<number>(features.length).fill(0);
  const featureRows = points.map(point => {
    if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) {
      return NO_FEATURE;
    }
    for (const [row, feature] of features.entries()) {
      const classification = classifyPointInFeature(point, feature);
      if (classification === 'inside' || (includeBoundary && classification === 'boundary')) {
        counts[row]++;
        return row;
      }
    }
    return NO_FEATURE;
  });
  return {featureRows, counts};
}

/** Distance from a point to a closed segment. */
export function pointSegmentDistance(point: Point, start: Point, end: Point): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const denominator = dx * dx + dy * dy;
  const t =
    denominator === 0
      ? 0
      : Math.min(
          Math.max(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator, 0),
          1
        );
  return Math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy));
}

/** Nearest segment within radius per point; ties go to the smallest row. */
export function joinNearestSegments(
  points: Point[],
  segments: [Point, Point][],
  radius: number
): {featureRows: number[]; distances: number[]; counts: number[]; secondDistances: number[]} {
  const counts = new Array<number>(segments.length).fill(0);
  const featureRows: number[] = [];
  const distances: number[] = [];
  const secondDistances: number[] = [];
  for (const point of points) {
    let best = NO_FEATURE;
    let bestDistance = Infinity;
    let secondDistance = Infinity;
    for (const [row, [start, end]] of segments.entries()) {
      const distance = pointSegmentDistance(point, start, end);
      if (distance > radius) {
        continue;
      }
      if (distance < bestDistance) {
        secondDistance = bestDistance;
        bestDistance = distance;
        best = row;
      } else if (distance < secondDistance) {
        secondDistance = distance;
      }
    }
    featureRows.push(best);
    distances.push(best === NO_FEATURE ? -1 : bestDistance);
    secondDistances.push(secondDistance);
    if (best !== NO_FEATURE) {
      counts[best]++;
    }
  }
  return {featureRows, distances, counts, secondDistances};
}

/** Deterministic mulberry32 generator. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

const square = (minX: number, minY: number, maxX: number, maxY: number): number[][] => [
  [minX, minY],
  [maxX, minY],
  [maxX, maxY],
  [minX, maxY]
];

/** Four features: holed square, adjacent square, two-part multipolygon, and an empty feature. */
export const POLYGON_FEATURES: OraclePolygonFeature[] = [
  [[square(0, 0, 4, 4), square(1, 1, 2, 2)]],
  [[square(4, 0, 8, 4)]],
  [[square(10, 0, 12, 2)], [square(10, 3, 12, 5)]],
  []
];

/** Ten fixture points for the point-in-polygon join. */
export const POLYGON_POINTS: Point[] = [
  [0.5, 0.5],
  [1.5, 1.5],
  [3, 3],
  [4, 2],
  [6, 1],
  [11, 1],
  [11, 4],
  [11, 2.5],
  [20, 20],
  [Number.NaN, 1]
];
