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
