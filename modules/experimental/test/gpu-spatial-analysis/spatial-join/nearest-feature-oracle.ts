// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OraclePolygonFeature} from './spatial-join-oracle';

export type NearestPoint = [number, number];

/** Geometry sets for the k-nearest oracle, one feature per entry. */
export type NearestOracleGeometry =
  | {kind: 'points'; features: NearestPoint[]}
  | {kind: 'lines'; features: NearestPoint[][]}
  | {kind: 'polygons'; features: OraclePolygonFeature[]};

/** One usable edge: endpoints and the absolute index of its first vertex (-1 for points). */
type OracleEdge = {a: NearestPoint; b: NearestPoint; segment: number};

/** Neighbor row of the oracle. */
export type OracleNeighbor = {
  id: number;
  distance: number;
  foot: NearestPoint;
  /** Absolute vertex index of the edge start, or -1 for none. */
  segment: number;
};

/** Flat typed arrays in the layout the GPU geometry views expect. */
export type NearestOracleArrays = {
  positions: Float32Array;
  lineOffsets?: Uint32Array;
  featureOffsets?: Uint32Array;
  polygonOffsets?: Uint32Array;
  ringOffsets?: Uint32Array;
};

/** Flattens a geometry set into GeoArrow-style arrays. */
export function buildNearestArrays(geometry: NearestOracleGeometry): NearestOracleArrays {
  const positions: number[] = [];
  if (geometry.kind === 'points') {
    for (const [x, y] of geometry.features) positions.push(x, y);
    return {positions: Float32Array.from(positions)};
  }
  if (geometry.kind === 'lines') {
    const lineOffsets = [0];
    for (const line of geometry.features) {
      for (const [x, y] of line) positions.push(x, y);
      lineOffsets.push(positions.length / 2);
    }
    return {positions: Float32Array.from(positions), lineOffsets: Uint32Array.from(lineOffsets)};
  }
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const feature of geometry.features) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const [x, y] of ring) positions.push(x, y);
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets)
  };
}

/** Returns per-feature usable edges with absolute vertex indices, and polygon rings for containment. */
function getEdges(geometry: NearestOracleGeometry): {
  edges: OracleEdge[][];
  rings: NearestPoint[][][];
} {
  let vertexIndex = 0;
  const edges: OracleEdge[][] = [];
  const rings: NearestPoint[][][] = [];
  if (geometry.kind === 'points') {
    for (const point of geometry.features) {
      edges.push([{a: point, b: point, segment: -1}]);
      rings.push([]);
      vertexIndex++;
    }
  } else if (geometry.kind === 'lines') {
    for (const line of geometry.features) {
      const featureEdges: OracleEdge[] = [];
      if (line.length >= 2) {
        for (let index = 0; index + 1 < line.length; index++) {
          featureEdges.push({a: line[index], b: line[index + 1], segment: vertexIndex + index});
        }
      }
      edges.push(featureEdges);
      rings.push([]);
      vertexIndex += line.length;
    }
  } else {
    for (const feature of geometry.features) {
      const featureEdges: OracleEdge[] = [];
      const featureRings: NearestPoint[][] = [];
      for (const polygon of feature) {
        for (const ring of polygon) {
          if (ring.length >= 3) {
            featureRings.push(ring as NearestPoint[]);
            for (let index = 0; index < ring.length; index++) {
              featureEdges.push({
                a: ring[index] as NearestPoint,
                b: ring[(index + 1) % ring.length] as NearestPoint,
                segment: vertexIndex + index
              });
            }
          }
          vertexIndex += ring.length;
        }
      }
      edges.push(featureEdges);
      rings.push(featureRings);
    }
  }
  return {edges, rings};
}

function dot(a: NearestPoint, b: NearestPoint): number {
  return a[0] * b[0] + a[1] * b[1];
}

function subtract(a: NearestPoint, b: NearestPoint): NearestPoint {
  return [a[0] - b[0], a[1] - b[1]];
}

function cross(a: NearestPoint, b: NearestPoint): number {
  return a[0] * b[1] - a[1] * b[0];
}

function closestOnSegment(p: NearestPoint, a: NearestPoint, b: NearestPoint): NearestPoint {
  const ab = subtract(b, a);
  const lengthSq = dot(ab, ab);
  if (lengthSq === 0) return a;
  const t = Math.min(1, Math.max(0, dot(subtract(p, a), ab) / lengthSq));
  return [a[0] + ab[0] * t, a[1] + ab[1] * t];
}

function distanceSq(a: NearestPoint, b: NearestPoint): number {
  const d = subtract(a, b);
  return dot(d, d);
}

function orientation(a: NearestPoint, b: NearestPoint, c: NearestPoint): number {
  return cross(subtract(b, a), subtract(c, a));
}

function segmentsIntersect(a: NearestPoint, b: NearestPoint, c: NearestPoint, d: NearestPoint) {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);
  if (o1 * o2 < 0 && o3 * o4 < 0) return true;
  const on = (p: NearestPoint, q: NearestPoint, r: NearestPoint) =>
    orientation(p, q, r) === 0 &&
    r[0] >= Math.min(p[0], q[0]) &&
    r[0] <= Math.max(p[0], q[0]) &&
    r[1] >= Math.min(p[1], q[1]) &&
    r[1] <= Math.max(p[1], q[1]);
  return on(a, b, c) || on(a, b, d) || on(c, d, a) || on(c, d, b);
}

/** Distance squared between segments a-b and c-d, and the nearest point on c-d. */
function segmentPair(
  a: NearestPoint,
  b: NearestPoint,
  c: NearestPoint,
  d: NearestPoint
): {distanceSq: number; foot: NearestPoint} {
  if (segmentsIntersect(a, b, c, d)) {
    const ab = subtract(b, a);
    const cd = subtract(d, c);
    const denominator = cross(cd, ab);
    if (denominator !== 0) {
      const u = Math.min(1, Math.max(0, cross(subtract(a, c), ab) / denominator));
      return {distanceSq: 0, foot: [c[0] + cd[0] * u, c[1] + cd[1] * u]};
    }
  }
  const candidates: NearestPoint[] = [closestOnSegment(a, c, d), closestOnSegment(b, c, d), c, d];
  const references: NearestPoint[] = [a, b, closestOnSegment(c, a, b), closestOnSegment(d, a, b)];
  let best = {distanceSq: Infinity, foot: c};
  for (const [index, candidate] of candidates.entries()) {
    const value = distanceSq(candidate, references[index]);
    if (value < best.distanceSq) best = {distanceSq: value, foot: candidate};
  }
  return best;
}

function contains(rings: NearestPoint[][], q: NearestPoint): boolean {
  let inside = false;
  for (const ring of rings) {
    for (let index = 0; index < ring.length; index++) {
      const a = ring[index];
      const b = ring[(index + 1) % ring.length];
      if (a[1] > q[1] !== b[1] > q[1]) {
        const t = (q[1] - a[1]) / (b[1] - a[1]);
        if (q[0] < a[0] + t * (b[0] - a[0])) inside = !inside;
      }
    }
  }
  return inside;
}

/**
 * Brute-force k-nearest in double precision with `(distance, id)` ordering.
 *
 * `ties: 'all'` keeps every neighbor tied with the k-th distance. `maxDistance` is inclusive.
 * Polygon containment (a vertex of one geometry inside a polygon of the other) gives distance 0.
 */
export function nearestNeighborsOracle(
  queries: NearestOracleGeometry,
  features: NearestOracleGeometry,
  options: {k: number; ties?: 'lowest-id' | 'all'; maxDistance?: number; featureIds?: number[]}
): OracleNeighbor[][] {
  const queryGeometry = getEdges(queries);
  const featureGeometry = getEdges(features);
  const result: OracleNeighbor[][] = [];
  for (let queryIndex = 0; queryIndex < queryGeometry.edges.length; queryIndex++) {
    const neighbors: OracleNeighbor[] = [];
    for (let featureIndex = 0; featureIndex < featureGeometry.edges.length; featureIndex++) {
      let best: OracleNeighbor | undefined;
      for (const queryEdge of queryGeometry.edges[queryIndex]) {
        for (const featureEdge of featureGeometry.edges[featureIndex]) {
          const pair = segmentPair(queryEdge.a, queryEdge.b, featureEdge.a, featureEdge.b);
          if (!best || pair.distanceSq < best.distance ** 2) {
            best = {
              id: featureIndex,
              distance: Math.sqrt(pair.distanceSq),
              foot: pair.foot,
              segment: featureEdge.segment
            };
          }
        }
      }
      if (!best) continue;
      if (best.distance > 0) {
        const queryFirst = queryGeometry.edges[queryIndex][0]?.a;
        const featureFirst = featureGeometry.edges[featureIndex][0]?.a;
        if (queryFirst && contains(featureGeometry.rings[featureIndex], queryFirst)) {
          best = {id: featureIndex, distance: 0, foot: queryFirst, segment: -1};
        } else if (featureFirst && contains(queryGeometry.rings[queryIndex], featureFirst)) {
          best = {id: featureIndex, distance: 0, foot: featureFirst, segment: -1};
        }
      }
      if (options.maxDistance === undefined || best.distance <= options.maxDistance) {
        neighbors.push(best);
      }
    }
    neighbors.sort((left, right) => left.distance - right.distance || left.id - right.id);
    let count = Math.min(options.k, neighbors.length);
    if (options.ties === 'all' && neighbors.length >= options.k) {
      const kth = neighbors[options.k - 1].distance;
      while (count < neighbors.length && neighbors[count].distance === kth) count++;
    }
    result.push(
      neighbors
        .slice(0, count)
        .map(neighbor => ({...neighbor, id: options.featureIds?.[neighbor.id] ?? neighbor.id}))
    );
  }
  return result;
}
