// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {cellToLatLng, latLngToCell, polygonToCells} from 'h3-js';
import {
  CELL_COVER_H3_LATTICE_SPACING_DEGREES,
  CELL_COVER_H3_MAXIMUM_LATITUDE
} from '../../../src/gpu-spatial-analysis/cell-cover/cell-cover-h3-constants';
import {
  getQuadbinTileX,
  getQuadbinTileY,
  getQuadbinTileYFloat64,
  h3ToBigInt,
  quadbinTileToCell
} from '../cell-aggregation/cell-aggregation-oracle';

const f = Math.fround;

/** One feature: polygons, each a list of rings (shell first), each a list of `[lng, lat]`. */
export type CoverFeature = number[][][][];

/** Flattened polygon arrays as the contributor takes them. */
export type CoverPolygonArrays = {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};

/** Result in output order. */
export type CoverResult = {featureRows: number[]; cells: bigint[]};

/** Flattens nested features into the GeoArrow-style offsets arrays. */
export function flattenCoverFeatures(features: CoverFeature[]): CoverPolygonArrays {
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  let polygonCount = 0;
  let ringCount = 0;
  for (const feature of features) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const [lng, lat] of ring) {
          positions.push(lng, lat);
        }
        ringOffsets.push(positions.length / 2);
        ringCount++;
      }
      polygonOffsets.push(ringCount);
      polygonCount++;
    }
    featureOffsets.push(polygonCount);
  }
  return {
    positions: new Float32Array(positions),
    featureOffsets: new Uint32Array(featureOffsets),
    polygonOffsets: new Uint32Array(polygonOffsets),
    ringOffsets: new Uint32Array(ringOffsets)
  };
}

type Vertex = [number, number];

function getFeatureRings(arrays: CoverPolygonArrays, feature: number): Vertex[][] {
  const rings: Vertex[][] = [];
  for (
    let polygon = arrays.featureOffsets[feature];
    polygon < arrays.featureOffsets[feature + 1];
    polygon++
  ) {
    for (
      let ring = arrays.polygonOffsets[polygon];
      ring < arrays.polygonOffsets[polygon + 1];
      ring++
    ) {
      const vertices: Vertex[] = [];
      for (let vertex = arrays.ringOffsets[ring]; vertex < arrays.ringOffsets[ring + 1]; vertex++) {
        vertices.push([arrays.positions[2 * vertex], arrays.positions[2 * vertex + 1]]);
      }
      rings.push(vertices);
    }
  }
  return rings;
}

function getBounds(rings: Vertex[][]): [number, number, number, number] | undefined {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const ring of rings) {
    for (const [x, y] of ring) {
      if (Number.isFinite(x) && Number.isFinite(y)) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }
  return minX <= maxX && minY <= maxY ? [minX, minY, maxX, maxY] : undefined;
}

/**
 * Even-odd containment mirroring the WGSL `coverContains`: every operation is rounded to f32 in
 * the shader's order. With `exact` true it uses f64 instead.
 */
export function isInsideRings(rings: Vertex[][], p: Vertex, exact: boolean = false): boolean {
  const r = exact ? (value: number) => value : f;
  let inside = false;
  for (const ring of rings) {
    for (let index = 0; index < ring.length; index++) {
      const a = ring[index];
      const b = ring[(index + 1) % ring.length];
      if (a[1] > p[1] !== b[1] > p[1]) {
        const crossing = r(r(r(r(b[0] - a[0]) * r(p[1] - a[1])) / r(b[1] - a[1])) + a[0]);
        if (p[0] < crossing) {
          inside = !inside;
        }
      }
    }
  }
  return inside;
}

function segmentHitsRect(a: Vertex, b: Vertex, lo: Vertex, hi: Vertex): boolean {
  let lower = -1e30;
  let upper = 1e30;
  for (const axis of [0, 1]) {
    const d = f(b[axis] - a[axis]);
    if (d === 0) {
      if (!(lo[axis] < a[axis] && a[axis] < hi[axis])) {
        return false;
      }
    } else {
      const t1 = f(f(lo[axis] - a[axis]) / d);
      const t2 = f(f(hi[axis] - a[axis]) / d);
      lower = Math.max(lower, Math.min(t1, t2));
      upper = Math.min(upper, Math.max(t1, t2));
    }
  }
  return lower < upper && lower < 1 && upper > 0;
}

function edgesHitRect(rings: Vertex[][], lo: Vertex, hi: Vertex): boolean {
  for (const ring of rings) {
    for (let index = 0; index < ring.length; index++) {
      if (segmentHitsRect(ring[index], ring[(index + 1) % ring.length], lo, hi)) {
        return true;
      }
    }
  }
  return false;
}

/** Mercator tile edge latitude (f64 formula, rounded to f32 like the shader's result). */
function getTileLatitude(row: number, resolution: number): number {
  const t = Math.PI * (1 - 2 * (row / 2 ** resolution));
  return Math.atan(Math.sinh(t)) * (180 / Math.PI);
}

/** Candidate Quadbin tiles of one feature's bounding box, north to south then west to east. */
function getQuadbinCandidates(rings: Vertex[][], resolution: number): {x: number; y: number}[] {
  const bounds = getBounds(rings);
  if (!bounds) {
    return [];
  }
  const x0 = getQuadbinTileX(bounds[0], resolution);
  const x1 = getQuadbinTileX(bounds[2], resolution);
  const y0 = getQuadbinTileY(bounds[3], resolution);
  const y1 = getQuadbinTileY(bounds[1], resolution);
  const candidates: {x: number; y: number}[] = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      candidates.push({x, y});
    }
  }
  return candidates;
}

/**
 * Quadbin cover oracle. Candidates and the integer tile range are bit exact with the kernel;
 * predicates mirror its f32 arithmetic. Tile edge and center latitudes come from the f64 Mercator
 * formula rounded to f32, whereas the kernel uses f32 `atan`/`exp`, so results can differ only for
 * polygon edges within about 1e-4 degrees of a tile edge or center.
 */
export function coverQuadbinOnCPU(
  arrays: CoverPolygonArrays,
  resolution: number,
  containment: 'center' | 'full' | 'intersects',
  candidateCapacity: number = Infinity
): CoverResult {
  const result: CoverResult = {featureRows: [], cells: []};
  const width = 45 * 2 ** (3 - resolution);
  const featureCount = arrays.featureOffsets.length - 1;
  let candidateTotal = 0;
  for (let feature = 0; feature < featureCount; feature++) {
    const rings = getFeatureRings(arrays, feature);
    for (const {x, y} of getQuadbinCandidates(rings, resolution)) {
      if (candidateTotal++ >= candidateCapacity) {
        return result;
      }
      const centre: Vertex = [
        f(f(f(f(x) + 0.5) * f(width)) - 180),
        f(getTileLatitude(f(f(y) + 0.5), resolution))
      ];
      let accepted: boolean;
      if (containment === 'center') {
        accepted = isInsideRings(rings, centre);
      } else {
        const lo: Vertex = [f(f(f(x) * f(width)) - 180), f(getTileLatitude(y + 1, resolution))];
        const hi: Vertex = [f(f(f(x + 1) * f(width)) - 180), f(getTileLatitude(y, resolution))];
        const hit = edgesHitRect(rings, lo, hi);
        accepted =
          containment === 'full'
            ? !hit && isInsideRings(rings, centre)
            : hit || isInsideRings(rings, centre);
      }
      if (accepted) {
        result.featureRows.push(feature);
        result.cells.push(quadbinTileToCell(x, y, resolution));
      }
    }
  }
  return result;
}

/**
 * Independent f64 `center` oracle: f64 tile range (quadbin-py row formula), f64 tile centers
 * (center of the tile in Mercator space) and f64 even-odd containment.
 */
export function coverQuadbinCenterFloat64(
  arrays: CoverPolygonArrays,
  resolution: number
): CoverResult {
  const result: CoverResult = {featureRows: [], cells: []};
  const width = 360 / 2 ** resolution;
  for (let feature = 0; feature < arrays.featureOffsets.length - 1; feature++) {
    const rings = getFeatureRings(arrays, feature);
    const bounds = getBounds(rings);
    if (!bounds) {
      continue;
    }
    const x0 = getQuadbinTileX(bounds[0], resolution);
    const x1 = getQuadbinTileX(bounds[2], resolution);
    const y0 = getQuadbinTileYFloat64(bounds[3], resolution);
    const y1 = getQuadbinTileYFloat64(bounds[1], resolution);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const centre: Vertex = [(x + 0.5) * width - 180, getTileLatitude(y + 0.5, resolution)];
        if (isInsideRings(rings, centre, true)) {
          result.featureRows.push(feature);
          result.cells.push(quadbinTileToCell(x, y, resolution));
        }
      }
    }
  }
  return result;
}

/** h3-js `polygonToCells` (center containment) per feature, as BigInt cells, sorted ascending. */
export function coverH3OnCPU(features: CoverFeature[], resolution: number): CoverResult {
  const result: CoverResult = {featureRows: [], cells: []};
  for (const [feature, polygons] of features.entries()) {
    const cells = new Set<bigint>();
    for (const polygon of polygons) {
      for (const cell of polygonToCells(polygon, resolution, true)) {
        cells.add(h3ToBigInt(cell));
      }
    }
    for (const cell of [...cells].sort((left, right) =>
      left < right ? -1 : left > right ? 1 : 0
    )) {
      result.featureRows.push(feature);
      result.cells.push(cell);
    }
  }
  return result;
}

/** Groups a cover result into per-feature sets of hex strings. */
export function groupCoverByFeature(result: CoverResult): Map<number, Set<bigint>> {
  const groups = new Map<number, Set<bigint>>();
  for (const [index, feature] of result.featureRows.entries()) {
    const group = groups.get(feature) ?? new Set<bigint>();
    group.add(result.cells[index]);
    groups.set(feature, group);
  }
  return groups;
}

/** Counts cells present in exactly one of two results (symmetric difference), per feature total. */
export function countCoverDisagreements(left: CoverResult, right: CoverResult): number {
  const leftGroups = groupCoverByFeature(left);
  const rightGroups = groupCoverByFeature(right);
  let count = 0;
  for (const feature of new Set([...leftGroups.keys(), ...rightGroups.keys()])) {
    const a = leftGroups.get(feature) ?? new Set<bigint>();
    const b = rightGroups.get(feature) ?? new Set<bigint>();
    for (const cell of a) {
      count += b.has(cell) ? 0 : 1;
    }
    for (const cell of b) {
      count += a.has(cell) ? 0 : 1;
    }
  }
  return count;
}

/**
 * f64 CPU simulation of the H3 lattice algorithm with h3-js `latLngToCell` / `cellToLatLng`: a
 * lattice over the bounding box (same spacing constants as the kernel), one emit per cell from the
 * lattice point nearest its center when that center is inside the polygon. Validates the spacing
 * constants independently of f32 arithmetic. Returns the candidate (lattice point) count too.
 */
export function coverH3LatticeOnCPU(
  arrays: CoverPolygonArrays,
  resolution: number
): CoverResult & {candidateCount: number} {
  const result: CoverResult & {candidateCount: number} = {
    featureRows: [],
    cells: [],
    candidateCount: 0
  };
  const spacingLat = CELL_COVER_H3_LATTICE_SPACING_DEGREES[resolution];
  for (let feature = 0; feature < arrays.featureOffsets.length - 1; feature++) {
    const rings = getFeatureRings(arrays, feature);
    const bounds = getBounds(rings);
    if (!bounds) {
      continue;
    }
    const originLat = Math.max(bounds[1] - spacingLat, -CELL_COVER_H3_MAXIMUM_LATITUDE);
    const endLat = Math.min(bounds[3] + spacingLat, CELL_COVER_H3_MAXIMUM_LATITUDE);
    if (originLat > endLat) {
      continue;
    }
    const spacingLng =
      spacingLat / Math.cos((Math.max(Math.abs(originLat), Math.abs(endLat)) * Math.PI) / 180);
    const originLng = bounds[0] - spacingLng;
    const width = Math.ceil((bounds[2] + spacingLng - originLng) / spacingLng) + 1;
    const height = Math.ceil((endLat - originLat) / spacingLat) + 1;
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        result.candidateCount++;
        const lng = originLng + column * spacingLng;
        const lat = originLat + row * spacingLat;
        const cell = latLngToCell(lat, lng, resolution);
        const [centreLat, centreLng] = cellToLatLng(cell);
        let deltaLng = centreLng - lng;
        deltaLng -= 360 * Math.round(deltaLng / 360);
        const deltaLat = centreLat - lat;
        if (
          deltaLng >= -0.5 * spacingLng &&
          deltaLng < 0.5 * spacingLng &&
          deltaLat >= -0.5 * spacingLat &&
          deltaLat < 0.5 * spacingLat &&
          isInsideRings(rings, [centreLng, centreLat], true)
        ) {
          result.featureRows.push(feature);
          result.cells.push(h3ToBigInt(cell));
        }
      }
    }
  }
  return result;
}

/** Deterministic star-shaped polygon (simple, optionally with a concave dent) around a center. */
export function createStarPolygon(
  seed: number,
  center: [number, number],
  radius: number,
  vertexCount: number
): number[][] {
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const ring: number[][] = [];
  for (let index = 0; index < vertexCount; index++) {
    const angle = (2 * Math.PI * index) / vertexCount;
    const length = radius * (0.45 + 0.55 * random());
    ring.push([
      Math.fround(center[0] + length * Math.cos(angle)),
      Math.fround(center[1] + length * Math.sin(angle))
    ]);
  }
  return ring;
}

/** Rounds every coordinate to f32, as the GPU and the h3-js comparison both see them. */
export function roundFeatures(features: CoverFeature[]): CoverFeature[] {
  return features.map(feature =>
    feature.map(polygon => polygon.map(ring => ring.map(([lng, lat]) => [f(lng), f(lat)])))
  );
}

/**
 * Closed-ring rectangle helper: counter-clockwise `[west, south, east, north]` ring (closed for
 * h3-js, which expects a closed GeoJSON loop; the contributor's rings close implicitly either way).
 */
export function createRectangleRing([west, south, east, north]: number[]): number[][] {
  return [
    [west, south],
    [east, south],
    [east, north],
    [west, north],
    [west, south]
  ];
}

/** Minimum planar distance (degrees) from a point to any ring edge of a feature. */
export function getDistanceToFeatureBoundary(
  arrays: CoverPolygonArrays,
  feature: number,
  point: [number, number]
): number {
  let minimum = Infinity;
  for (const ring of getFeatureRings(arrays, feature)) {
    for (let index = 0; index < ring.length; index++) {
      const a = ring[index];
      const b = ring[(index + 1) % ring.length];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const length2 = dx * dx + dy * dy;
      const t =
        length2 === 0
          ? 0
          : Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2));
      minimum = Math.min(minimum, Math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy));
    }
  }
  return minimum;
}

/** Lists cells present in exactly one of two results as `{feature, cell, side}`. */
export function listCoverDisagreements(
  left: CoverResult,
  right: CoverResult
): {feature: number; cell: bigint; side: 'left' | 'right'}[] {
  const leftGroups = groupCoverByFeature(left);
  const rightGroups = groupCoverByFeature(right);
  const rows: {feature: number; cell: bigint; side: 'left' | 'right'}[] = [];
  for (const feature of new Set([...leftGroups.keys(), ...rightGroups.keys()])) {
    const a = leftGroups.get(feature) ?? new Set<bigint>();
    const b = rightGroups.get(feature) ?? new Set<bigint>();
    for (const cell of a) {
      if (!b.has(cell)) {
        rows.push({feature, cell, side: 'left'});
      }
    }
    for (const cell of b) {
      if (!a.has(cell)) {
        rows.push({feature, cell, side: 'right'});
      }
    }
  }
  return rows;
}
