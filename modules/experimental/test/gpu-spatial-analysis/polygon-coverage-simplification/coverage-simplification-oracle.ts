// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {simplifyDouglasPeucker} from '../line-simplification/line-simplification-oracle';
import {flattenPolygons, type OraclePolygons} from '../spatial-weights/spatial-weights-oracle';

/** Result of {@link computeCoverageSimplificationOracle}. */
export type CoverageSimplificationOracle = {
  /** Per input vertex keep flag. */
  keepMask: number[];
  /** Number of arcs that were simplified (owned arcs). */
  arcCount: number;
};

const pointKey = (x: number, y: number) =>
  `${Math.fround(x) === 0 ? 0 : Math.fround(x)},${Math.fround(y) === 0 ? 0 : Math.fround(y)}`;

/**
 * Independent CPU model of `GPUCoverageSimplification` with exact snapping (`snapTolerance` 0):
 * partner edges, arcs broken at ring starts and partner changes, one Douglas-Peucker run per owned
 * arc, and the point-ID sharing of decisions.
 */
export function computeCoverageSimplificationOracle(
  polygons: OraclePolygons,
  tolerance: number
): CoverageSimplificationOracle {
  const layout = flattenPolygons(polygons);
  const vertexCount = layout.positions.length / 2;
  const ringCount = layout.ringOffsets.length - 1;
  const polygonOfRing: number[] = [];
  for (let polygon = 0; polygon < layout.polygonOffsets.length - 1; polygon++) {
    for (
      let ring = layout.polygonOffsets[polygon];
      ring < layout.polygonOffsets[polygon + 1];
      ring++
    ) {
      polygonOfRing[ring] = polygon;
    }
  }
  const keys: string[] = [];
  const polygonOf: number[] = [];
  const ringBegin: number[] = [];
  const next: number[] = [];
  const previous: number[] = [];
  for (let ring = 0; ring < ringCount; ring++) {
    const begin = layout.ringOffsets[ring];
    const end = layout.ringOffsets[ring + 1];
    for (let vertex = begin; vertex < end; vertex++) {
      keys[vertex] = pointKey(layout.positions[2 * vertex], layout.positions[2 * vertex + 1]);
      polygonOf[vertex] = polygonOfRing[ring];
      ringBegin[vertex] = begin;
      next[vertex] = vertex + 1 === end ? begin : vertex + 1;
      previous[vertex] = vertex === begin ? end - 1 : vertex - 1;
    }
  }
  const edgeKey = (vertex: number) => {
    const first = keys[vertex];
    const second = keys[next[vertex]];
    return first === second
      ? undefined
      : first < second
        ? `${first}|${second}`
        : `${second}|${first}`;
  };
  const edgePolygons = new Map<string, Set<number>>();
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const key = edgeKey(vertex);
    if (key) {
      if (!edgePolygons.has(key)) edgePolygons.set(key, new Set());
      edgePolygons.get(key)!.add(polygonOf[vertex]);
    }
  }
  const NONE = Infinity;
  const partner = (vertex: number) => {
    const key = edgeKey(vertex);
    let best = NONE;
    if (key) {
      for (const other of edgePolygons.get(key)!) {
        if (other !== polygonOf[vertex]) best = Math.min(best, other);
      }
    }
    return best;
  };
  const partners = Array.from({length: vertexCount}, (_, vertex) => partner(vertex));
  const isStart: boolean[] = [];
  const isJunction: boolean[] = [];
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    isJunction[vertex] = partners[previous[vertex]] !== partners[vertex];
    isStart[vertex] = vertex === ringBegin[vertex] || isJunction[vertex];
  }
  const lineVertices: number[] = [];
  const trackOffsets = [0];
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const owned = partners[vertex] === NONE || polygonOf[vertex] < partners[vertex];
    if (!isStart[vertex] || !owned) continue;
    let walker = vertex;
    lineVertices.push(walker);
    do {
      walker = next[walker];
      lineVertices.push(walker);
    } while (!isStart[walker]);
    trackOffsets.push(lineVertices.length);
  }
  const positions = new Float32Array(lineVertices.length * 2);
  lineVertices.forEach((vertex, row) => {
    positions[2 * row] = layout.positions[2 * vertex];
    positions[2 * row + 1] = layout.positions[2 * vertex + 1];
  });
  const kept = simplifyDouglasPeucker(
    {positions, trackOffsets: new Uint32Array(trackOffsets)},
    tolerance
  );
  const keepPoints = new Set<string>();
  for (const row of kept) keepPoints.add(keys[lineVertices[row]]);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    if (isJunction[vertex]) keepPoints.add(keys[vertex]);
  }
  return {
    keepMask: keys.map(key => (keepPoints.has(key) ? 1 : 0)),
    arcCount: trackOffsets.length - 1
  };
}

/**
 * Gap-freeness check. Returns a list of violations: for every original edge shared by polygons `P`
 * and `Q`, the simplified span of `P` containing it must appear (in either direction, since ring orientation varies) as a span of `Q`.
 */
export function findCoverageGaps(polygons: OraclePolygons, keepMask: readonly number[]): string[] {
  const layout = flattenPolygons(polygons);
  const ringCount = layout.ringOffsets.length - 1;
  const polygonOfRing: number[] = [];
  for (let polygon = 0; polygon < layout.polygonOffsets.length - 1; polygon++) {
    for (
      let ring = layout.polygonOffsets[polygon];
      ring < layout.polygonOffsets[polygon + 1];
      ring++
    ) {
      polygonOfRing[ring] = polygon;
    }
  }
  const key = (vertex: number) =>
    pointKey(layout.positions[2 * vertex], layout.positions[2 * vertex + 1]);
  // Output spans per polygon: directed (from|to) pairs between consecutive kept vertices.
  const spans = new Map<number, Set<string>>();
  const originalEdges = new Map<string, Set<number>>();
  const problems: string[] = [];
  for (let ring = 0; ring < ringCount; ring++) {
    const begin = layout.ringOffsets[ring];
    const end = layout.ringOffsets[ring + 1];
    const polygon = polygonOfRing[ring];
    if (!spans.has(polygon)) spans.set(polygon, new Set());
    const keptVertices: number[] = [];
    for (let vertex = begin; vertex < end; vertex++) {
      if (keepMask[vertex]) keptVertices.push(vertex);
    }
    keptVertices.forEach((vertex, index) => {
      const following = keptVertices[(index + 1) % keptVertices.length];
      spans.get(polygon)!.add(`${key(vertex)}>${key(following)}`);
      spans.get(polygon)!.add(`${key(following)}>${key(vertex)}`);
    });
    for (let vertex = begin; vertex < end; vertex++) {
      const following = vertex + 1 === end ? begin : vertex + 1;
      const [first, second] = [key(vertex), key(following)].sort();
      const edge = `${first}|${second}`;
      if (!originalEdges.has(edge)) originalEdges.set(edge, new Set());
      originalEdges.get(edge)!.add(polygon);
    }
  }
  for (let ring = 0; ring < ringCount; ring++) {
    const begin = layout.ringOffsets[ring];
    const end = layout.ringOffsets[ring + 1];
    const polygon = polygonOfRing[ring];
    const keptVertices: number[] = [];
    for (let vertex = begin; vertex < end; vertex++) {
      if (keepMask[vertex]) keptVertices.push(vertex);
    }
    // Walk original edges; track the span (kept vertex to next kept vertex) each belongs to.
    for (let index = 0; index < keptVertices.length; index++) {
      const from = keptVertices[index];
      const to = keptVertices[(index + 1) % keptVertices.length];
      let sharedWith: number | undefined;
      let walker = from;
      let allShared = true;
      do {
        const following = walker + 1 === end ? begin : walker + 1;
        const [first, second] = [key(walker), key(following)].sort();
        const others = [...originalEdges.get(`${first}|${second}`)!].filter(
          other => other !== polygon
        );
        if (others.length === 0) allShared = false;
        else if (sharedWith === undefined) sharedWith = Math.min(...others);
        else if (!others.includes(sharedWith)) allShared = false;
        walker = following;
      } while (walker !== to && keptVertices.length > 1);
      if (allShared && sharedWith !== undefined) {
        const reversed = `${key(to)}>${key(from)}`;
        if (!spans.get(sharedWith)!.has(reversed)) {
          problems.push(
            `polygon ${polygon} span ${key(from)} to ${key(to)} has no match in ${sharedWith}`
          );
        }
      }
    }
  }
  return problems;
}

/** Exact integer image of an f32 value, scaled by 2^80 (values must be multiples of 2^-80). */
const toExact = (value: number) => BigInt(value * 2 ** 80);

function exactOrientation(
  a: readonly number[],
  b: readonly number[],
  c: readonly number[]
): -1 | 0 | 1 {
  const determinant =
    (toExact(b[0]) - toExact(a[0])) * (toExact(c[1]) - toExact(a[1])) -
    (toExact(b[1]) - toExact(a[1])) * (toExact(c[0]) - toExact(a[0]));
  return determinant > 0n ? 1 : determinant < 0n ? -1 : 0;
}

const inBox = (p: readonly number[], a: readonly number[], b: readonly number[]) =>
  p[0] >= Math.min(a[0], b[0]) &&
  p[0] <= Math.max(a[0], b[0]) &&
  p[1] >= Math.min(a[1], b[1]) &&
  p[1] <= Math.max(a[1], b[1]);

function segmentsIntersect(
  a: readonly number[],
  b: readonly number[],
  c: readonly number[],
  d: readonly number[]
): boolean {
  const o1 = exactOrientation(a, b, c);
  const o2 = exactOrientation(a, b, d);
  const o3 = exactOrientation(c, d, a);
  const o4 = exactOrientation(c, d, b);
  if (o1 !== o2 && o3 !== o4) return true;
  return (
    (o1 === 0 && inBox(c, a, b)) ||
    (o2 === 0 && inBox(d, a, b)) ||
    (o3 === 0 && inBox(a, c, d)) ||
    (o4 === 0 && inBox(b, c, d))
  );
}

/**
 * Exact count of topology violations in simplified rings: pairs of ring segments that intersect
 * although they share no endpoint coordinate. Identical segments (a shared edge, in either
 * direction) and segments that only meet at a shared endpoint are legitimate. `positions` and
 * `ringOffsets` use the layout of the contributor output (rings close implicitly).
 */
export function countCoverageCrossings(
  positions: ArrayLike<number>,
  ringOffsets: ArrayLike<number>
): number {
  type Segment = {a: [number, number]; b: [number, number]};
  const segments: Segment[] = [];
  for (let ring = 0; ring < ringOffsets.length - 1; ring++) {
    const begin = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    for (let vertex = begin; vertex < end; vertex++) {
      const next = vertex + 1 === end ? begin : vertex + 1;
      const a: [number, number] = [positions[2 * vertex], positions[2 * vertex + 1]];
      const b: [number, number] = [positions[2 * next], positions[2 * next + 1]];
      if (a[0] !== b[0] || a[1] !== b[1]) segments.push({a, b});
    }
  }
  const same = (p: readonly number[], q: readonly number[]) => p[0] === q[0] && p[1] === q[1];
  let crossings = 0;
  for (let first = 0; first < segments.length; first++) {
    const s = segments[first];
    for (let second = first + 1; second < segments.length; second++) {
      const t = segments[second];
      if (
        Math.max(s.a[0], s.b[0]) < Math.min(t.a[0], t.b[0]) ||
        Math.max(t.a[0], t.b[0]) < Math.min(s.a[0], s.b[0]) ||
        Math.max(s.a[1], s.b[1]) < Math.min(t.a[1], t.b[1]) ||
        Math.max(t.a[1], t.b[1]) < Math.min(s.a[1], s.b[1])
      ) {
        continue;
      }
      const identical = (same(s.a, t.a) && same(s.b, t.b)) || (same(s.a, t.b) && same(s.b, t.a));
      if (identical) continue;
      const shares = same(s.a, t.a) || same(s.a, t.b) || same(s.b, t.a) || same(s.b, t.b);
      if (shares) {
        // Sharing a point is fine unless the segments also overlap along a line.
        const collinear =
          exactOrientation(s.a, s.b, t.a) === 0 && exactOrientation(s.a, s.b, t.b) === 0;
        if (!collinear) continue;
        const sharedStart = same(s.a, t.a) || same(s.a, t.b) ? s.a : s.b;
        const sOther = sharedStart === s.a ? s.b : s.a;
        const tOther = same(sharedStart, t.a) ? t.b : t.a;
        const dot =
          (sOther[0] - sharedStart[0]) * (tOther[0] - sharedStart[0]) +
          (sOther[1] - sharedStart[1]) * (tOther[1] - sharedStart[1]);
        if (dot <= 0) continue;
      }
      if (segmentsIntersect(s.a, s.b, t.a, t.b)) crossings++;
    }
  }
  return crossings;
}
