// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  cellsToMultiPolygon,
  directedEdgeToBoundary,
  getDirectedEdgeDestination,
  originToDirectedEdges
} from 'h3-js';
import {
  bigIntToH3,
  h3ToBigInt,
  quadbinCellToTile,
  quadbinTileToCell
} from '../cell-aggregation/cell-aggregation-oracle';
import {webMercatorTileBounds} from '../cell-indexing/cell-indexing-oracle';

/** One expected outline segment of a cell. */
export type OutlineSegment = {cell: bigint; a: [number, number]; b: [number, number]};

/** Rounds a coordinate so f32 results and f64 oracles hash to the same key. */
const ROUND = 1e3;
function roundPoint([lng, lat]: [number, number]): string {
  return `${Math.round(lng * ROUND)},${Math.round(lat * ROUND)}`;
}

/** Order-independent key of a segment, `cell` included. */
export function getSegmentKey(cell: bigint, a: [number, number], b: [number, number]): string {
  const [first, second] = [roundPoint(a), roundPoint(b)].sort();
  return `${cell.toString(16)}|${first}|${second}`;
}

/** Order-independent key of a segment without its cell. */
export function getBareSegmentKey(a: [number, number], b: [number, number]): string {
  const [first, second] = [roundPoint(a), roundPoint(b)].sort();
  return `${first}|${second}`;
}

/**
 * H3 oracle from directed edges: for every cell, every outgoing edge whose destination is outside
 * the set (or has another group) contributes the segments of `directedEdgeToBoundary`.
 */
export function outlineH3OnCPU(cells: bigint[], groups?: Map<bigint, number>): OutlineSegment[] {
  const present = new Set(cells);
  const segments: OutlineSegment[] = [];
  for (const cell of cells) {
    for (const edge of originToDirectedEdges(bigIntToH3(cell))) {
      const destination = h3ToBigInt(getDirectedEdgeDestination(edge));
      const isSameSide =
        present.has(destination) && (!groups || groups.get(destination) === groups.get(cell));
      if (isSameSide) {
        continue;
      }
      const boundary = directedEdgeToBoundary(edge, true) as [number, number][];
      for (let vertex = 0; vertex + 1 < boundary.length; vertex++) {
        segments.push({cell, a: boundary[vertex], b: boundary[vertex + 1]});
      }
    }
  }
  return segments;
}

/** Quadbin oracle: tile neighbors with wrapped columns and absent rows beyond the map. */
export function outlineQuadbinOnCPU(
  cells: bigint[],
  groups?: Map<bigint, number>
): OutlineSegment[] {
  const present = new Set(cells);
  const segments: OutlineSegment[] = [];
  for (const cell of cells) {
    const {x, y, z} = quadbinCellToTile(cell);
    const n = 2 ** z;
    const [west, south, east, north] = webMercatorTileBounds(x, y, z);
    const corners: [number, number][] = [
      [west, north],
      [east, north],
      [east, south],
      [west, south]
    ];
    const offsets = [
      [0, -1],
      [1, 0],
      [0, 1],
      [-1, 0]
    ];
    for (let edge = 0; edge < 4; edge++) {
      const neighborY = y + offsets[edge][1];
      const neighborX = (((x + offsets[edge][0]) % n) + n) % n;
      const neighbor =
        neighborY >= 0 && neighborY < n ? quadbinTileToCell(neighborX, neighborY, z) : undefined;
      const isSameSide =
        neighbor !== undefined &&
        present.has(neighbor) &&
        (!groups || groups.get(neighbor) === groups.get(cell));
      if (!isSameSide) {
        segments.push({cell, a: corners[edge], b: corners[(edge + 1) % 4]});
      }
    }
  }
  return segments;
}

/** Tolerance in degrees between f32 GPU endpoints and f64 oracle endpoints. */
export const OUTLINE_TOLERANCE_DEGREES = 2e-3;

function isNear(left: [number, number], right: [number, number]): boolean {
  return (
    Math.abs(left[0] - right[0]) < OUTLINE_TOLERANCE_DEGREES &&
    Math.abs(left[1] - right[1]) < OUTLINE_TOLERANCE_DEGREES
  );
}

/**
 * Greedy one-to-one matching of segments within a tolerance, ignoring direction. Segments match
 * only when they have the same `cell` (use 0n for cell-less comparisons). Returns descriptions of
 * the unmatched actual and expected segments (both empty on success).
 */
export function matchSegments(
  actual: OutlineSegment[],
  expected: OutlineSegment[]
): {unmatchedActual: string[]; unmatchedExpected: string[]} {
  const remaining = new Map<bigint, OutlineSegment[]>();
  for (const segment of actual) {
    remaining.set(segment.cell, [...(remaining.get(segment.cell) ?? []), segment]);
  }
  const unmatchedExpected: string[] = [];
  for (const segment of expected) {
    const candidates = remaining.get(segment.cell) ?? [];
    const found = candidates.findIndex(
      item =>
        (isNear(item.a, segment.a) && isNear(item.b, segment.b)) ||
        (isNear(item.a, segment.b) && isNear(item.b, segment.a))
    );
    if (found < 0) {
      unmatchedExpected.push(`${segment.cell.toString(16)} ${segment.a} ${segment.b}`);
    } else {
      candidates.splice(found, 1);
    }
  }
  const unmatchedActual = [...remaining.entries()].flatMap(([cell, items]) =>
    items.map(item => `${cell.toString(16)} ${item.a} ${item.b}`)
  );
  return {unmatchedActual, unmatchedExpected};
}

/** Ring edges of `cellsToMultiPolygon` as cell-less segments. */
export function getMultiPolygonSegments(cells: bigint[]): OutlineSegment[] {
  const segments: OutlineSegment[] = [];
  for (const polygon of cellsToMultiPolygon(cells.map(bigIntToH3), true)) {
    for (const ring of polygon) {
      for (let vertex = 0; vertex + 1 < ring.length; vertex++) {
        segments.push({
          cell: 0n,
          a: ring[vertex] as [number, number],
          b: ring[vertex + 1] as [number, number]
        });
      }
    }
  }
  return segments;
}
