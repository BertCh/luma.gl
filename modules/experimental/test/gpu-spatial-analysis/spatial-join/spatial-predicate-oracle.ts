// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {buildPolygonFeatureArrays, createRandom} from './spatial-join-oracle';

export type OraclePoint = [number, number];

/** One oracle feature. Polygons are multipolygons: polygons, then rings (shell first), then vertices. */
export type OracleFeature =
  | {kind: 'points'; vertex: OraclePoint}
  | {kind: 'lines'; vertices: OraclePoint[]}
  | {kind: 'polygons'; polygons: OraclePoint[][][]};

export type OraclePredicate = 'intersects' | 'contains' | 'within' | 'dwithin';
type Location = 'exterior' | 'interior' | 'boundary';

/**
 * The oracle decides predicates by sampling, not by the algorithm the GPU kernels use.
 *
 * Test geometry has integer or half-integer vertices and axis-aligned or 45 degree edges, so every
 * vertex of the overlay arrangement lies on the half-integer lattice and every face contains a
 * point of the 1/8 lattice. Locating each lattice point against both geometries (exact in scaled
 * integer arithmetic) and applying the DE-9IM definitions to the sampled locations is therefore
 * complete for this geometry class.
 */
const SCALE = 8;

function scaled(feature: OracleFeature): OracleFeature {
  const scalePoint = ([x, y]: OraclePoint): OraclePoint => [x * SCALE, y * SCALE];
  if (feature.kind === 'points') {
    return {kind: 'points', vertex: scalePoint(feature.vertex)};
  }
  if (feature.kind === 'lines') {
    return {kind: 'lines', vertices: feature.vertices.map(scalePoint)};
  }
  return {
    kind: 'polygons',
    polygons: feature.polygons.map(polygon => polygon.map(ring => ring.map(scalePoint)))
  };
}

function onSegment(a: OraclePoint, b: OraclePoint, q: OraclePoint): boolean {
  const cross = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]);
  return (
    cross === 0 &&
    q[0] >= Math.min(a[0], b[0]) &&
    q[0] <= Math.max(a[0], b[0]) &&
    q[1] >= Math.min(a[1], b[1]) &&
    q[1] <= Math.max(a[1], b[1])
  );
}

function locate(feature: OracleFeature, q: OraclePoint): Location {
  if (feature.kind === 'points') {
    return feature.vertex[0] === q[0] && feature.vertex[1] === q[1] ? 'interior' : 'exterior';
  }
  if (feature.kind === 'lines') {
    const {vertices} = feature;
    if (vertices.length < 2) {
      return 'exterior';
    }
    for (let index = 0; index + 1 < vertices.length; index++) {
      if (onSegment(vertices[index], vertices[index + 1], q)) {
        const first = vertices[0];
        const last = vertices[vertices.length - 1];
        const closed = first[0] === last[0] && first[1] === last[1];
        const atEnd =
          (q[0] === first[0] && q[1] === first[1]) || (q[0] === last[0] && q[1] === last[1]);
        return !closed && atEnd ? 'boundary' : 'interior';
      }
    }
    return 'exterior';
  }
  // Even/odd over every ring of every polygon, which is the union for valid multipolygons.
  let inside = false;
  for (const polygon of feature.polygons) {
    for (const ring of polygon) {
      for (let index = 0; index < ring.length; index++) {
        const a = ring[index];
        const b = ring[(index + 1) % ring.length];
        if (onSegment(a, b, q)) {
          return 'boundary';
        }
        if (a[1] > q[1] !== b[1] > q[1]) {
          const crossX = a[0] + ((q[1] - a[1]) * (b[0] - a[0])) / (b[1] - a[1]);
          if (q[0] < crossX) {
            inside = !inside;
          }
        }
      }
    }
  }
  return inside ? 'interior' : 'exterior';
}

function getVertices(feature: OracleFeature): OraclePoint[] {
  if (feature.kind === 'points') {
    return [feature.vertex];
  }
  if (feature.kind === 'lines') {
    return feature.vertices;
  }
  return feature.polygons.flat(2);
}

function getBounds(features: OracleFeature[]): [number, number, number, number] {
  const vertices = features.flatMap(getVertices);
  const xs = vertices.map(vertex => vertex[0]);
  const ys = vertices.map(vertex => vertex[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function isEmpty(feature: OracleFeature): boolean {
  if (feature.kind === 'lines') {
    return feature.vertices.length < 2;
  }
  if (feature.kind === 'polygons') {
    return !feature.polygons.some(polygon => polygon.some(ring => ring.length >= 3));
  }
  return false;
}

function distanceToSegment(p: OraclePoint, a: OraclePoint, b: OraclePoint): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  const t =
    lengthSq === 0
      ? 0
      : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lengthSq));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

function getSegments(feature: OracleFeature): [OraclePoint, OraclePoint][] {
  if (feature.kind === 'points') {
    return [[feature.vertex, feature.vertex]];
  }
  if (feature.kind === 'lines') {
    return feature.vertices.slice(1).map((vertex, index) => [feature.vertices[index], vertex]);
  }
  return feature.polygons.flatMap(polygon =>
    polygon.flatMap(ring => ring.map((vertex, index) => [vertex, ring[(index + 1) % ring.length]]))
  ) as [OraclePoint, OraclePoint][];
}

/** Evaluates one predicate for one `(left, right)` pair by lattice sampling. */
export function evaluateOraclePredicate(
  predicate: OraclePredicate,
  left: OracleFeature,
  right: OracleFeature,
  distance = 0
): boolean {
  if (isEmpty(left) || isEmpty(right)) {
    return false;
  }
  if (predicate === 'within') {
    return evaluateOraclePredicate('contains', right, left);
  }
  const scaledLeft = scaled(left);
  const scaledRight = scaled(right);
  const [minX, minY, maxX, maxY] = getBounds([scaledLeft, scaledRight]);
  let touching = false;
  let rightOutsideLeft = false;
  let interiorsMeet = false;
  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      const leftLocation = locate(scaledLeft, [x, y]);
      const rightLocation = locate(scaledRight, [x, y]);
      if (leftLocation !== 'exterior' && rightLocation !== 'exterior') {
        touching = true;
      }
      if (leftLocation === 'exterior' && rightLocation !== 'exterior') {
        rightOutsideLeft = true;
      }
      if (leftLocation === 'interior' && rightLocation === 'interior') {
        interiorsMeet = true;
      }
    }
  }
  switch (predicate) {
    case 'intersects':
      return touching;
    case 'contains':
      return !rightOutsideLeft && interiorsMeet;
    case 'dwithin': {
      if (touching) {
        return true;
      }
      let best = Infinity;
      for (const [a, b] of getSegments(left)) {
        for (const [c, d] of getSegments(right)) {
          best = Math.min(
            best,
            distanceToSegment(a, c, d),
            distanceToSegment(b, c, d),
            distanceToSegment(c, a, b),
            distanceToSegment(d, a, b)
          );
        }
      }
      return best <= distance;
    }
    default:
      throw new Error(`unknown predicate ${predicate}`);
  }
}

/** All matching `[left, right]` pairs in `(left, right)` order. */
export function joinWithOracle(
  predicate: OraclePredicate,
  lefts: OracleFeature[],
  rights: OracleFeature[],
  options: {distance?: number; excludeSameRow?: boolean} = {}
): [number, number][] {
  const pairs: [number, number][] = [];
  for (const [leftRow, left] of lefts.entries()) {
    for (const [rightRow, right] of rights.entries()) {
      if (options.excludeSameRow && leftRow === rightRow) {
        continue;
      }
      if (evaluateOraclePredicate(predicate, left, right, options.distance)) {
        pairs.push([leftRow, rightRow]);
      }
    }
  }
  return pairs;
}

/** Flat arrays for one side of the join, in the layouts `GPUSpatialPredicateJoin` accepts. */
export type OracleArrays =
  | {kind: 'points'; positions: Float32Array}
  | {kind: 'lines'; positions: Float32Array; lineOffsets: Uint32Array}
  | {
      kind: 'polygons';
      positions: Float32Array;
      featureOffsets: Uint32Array;
      polygonOffsets: Uint32Array;
      ringOffsets: Uint32Array;
    };

/** Flattens same-kind features. */
export function buildOracleArrays(
  kind: OracleFeature['kind'],
  features: OracleFeature[]
): OracleArrays {
  if (kind === 'points') {
    return {
      kind,
      positions: Float32Array.from(
        features.flatMap(feature => (feature as {vertex: OraclePoint}).vertex)
      )
    };
  }
  if (kind === 'lines') {
    const offsets = [0];
    const positions: number[] = [];
    for (const feature of features) {
      const {vertices} = feature as {vertices: OraclePoint[]};
      positions.push(...vertices.flat());
      offsets.push(positions.length / 2);
    }
    return {kind, positions: Float32Array.from(positions), lineOffsets: Uint32Array.from(offsets)};
  }
  const arrays = buildPolygonFeatureArrays(
    features.map(feature => (feature as {polygons: OraclePoint[][][]}).polygons)
  );
  return {
    kind,
    positions: arrays.polygonPositions,
    featureOffsets: arrays.featureOffsets,
    polygonOffsets: arrays.polygonOffsets,
    ringOffsets: arrays.ringOffsets
  };
}

/** Axis-aligned rectangle polygon feature. */
export function rectangle(x: number, y: number, width: number, height: number): OracleFeature {
  return {
    kind: 'polygons',
    polygons: [
      [
        [
          [x, y],
          [x + width, y],
          [x + width, y + height],
          [x, y + height]
        ]
      ]
    ]
  };
}

/** Right triangle with legs along +x and +y from `(x, y)`. */
export function triangle(x: number, y: number, size: number): OracleFeature {
  return {
    kind: 'polygons',
    polygons: [
      [
        [
          [x, y],
          [x + size, y],
          [x, y + size]
        ]
      ]
    ]
  };
}

/** Polygon from explicit rings (shell first). */
export function polygonWithRings(...rings: OraclePoint[][]): OracleFeature {
  return {kind: 'polygons', polygons: [rings]};
}

/** Linestring feature. */
export function line(...vertices: OraclePoint[]): OracleFeature {
  return {kind: 'lines', vertices};
}

/** Point feature. */
export function point(x: number, y: number): OracleFeature {
  return {kind: 'points', vertex: [x, y]};
}

/**
 * Random valid features on the half-integer lattice with axis-aligned or 45 degree edges: points,
 * polylines, and rectangles, triangles, rectangles with holes, and two-part multipolygons.
 */
export function generateRandomFeatures(
  kind: OracleFeature['kind'],
  count: number,
  seed: number
): OracleFeature[] {
  const random = createRandom(seed);
  const integer = (low: number, high: number) => low + Math.floor(random() * (high - low + 1));
  const features: OracleFeature[] = [];
  for (let index = 0; index < count; index++) {
    if (kind === 'points') {
      features.push(point(integer(0, 20) / 2, integer(0, 20) / 2));
    } else if (kind === 'lines') {
      let vertex: OraclePoint = [integer(0, 10), integer(0, 10)];
      const vertices: OraclePoint[] = [vertex];
      const segmentCount = integer(1, 4);
      for (let segment = 0; segment < segmentCount; segment++) {
        const direction = [
          [1, 0],
          [0, 1],
          [-1, 0],
          [0, -1],
          [1, 1],
          [1, -1],
          [-1, 1],
          [-1, -1]
        ][integer(0, 7)];
        const length = integer(1, 3);
        vertex = [vertex[0] + direction[0] * length, vertex[1] + direction[1] * length];
        vertices.push(vertex);
      }
      features.push(line(...vertices));
    } else {
      const shape = integer(0, 4);
      const x = integer(0, 8);
      const y = integer(0, 8);
      if (shape === 0) {
        features.push(rectangle(x, y, integer(1, 4), integer(1, 4)));
      } else if (shape === 1) {
        features.push(triangle(x, y, integer(1, 4)));
      } else if (shape === 2) {
        const outer = integer(4, 6);
        features.push(
          polygonWithRings(
            [
              [x, y],
              [x + outer, y],
              [x + outer, y + outer],
              [x, y + outer]
            ],
            [
              [x + 1, y + 1],
              [x + 1, y + outer - 1],
              [x + outer - 1, y + outer - 1],
              [x + outer - 1, y + 1]
            ]
          )
        );
      } else if (shape === 3) {
        features.push({
          kind: 'polygons',
          polygons: [
            [
              [
                [x, y],
                [x + 2, y],
                [x + 2, y + 2],
                [x, y + 2]
              ]
            ],
            [
              [
                [x + 3, y],
                [x + 5, y],
                [x + 5, y + 2],
                [x + 3, y + 2]
              ]
            ]
          ]
        });
      } else {
        // L shape.
        features.push(
          polygonWithRings([
            [x, y],
            [x + 4, y],
            [x + 4, y + 2],
            [x + 2, y + 2],
            [x + 2, y + 4],
            [x, y + 4]
          ])
        );
      }
    }
  }
  return features;
}
