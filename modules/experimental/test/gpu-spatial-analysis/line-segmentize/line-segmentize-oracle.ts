// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCentralAngle,
  interpolateGreatCircle,
  wrapLongitudeDelta
} from '../geometry-measures/geodesic-oracle';

/** Flat path layout used by the line contributor tests. */
export type FlatPaths = {
  /** Interleaved x/y (or longitude/latitude) values. */
  positions: Float32Array;
  /** `pathCount + 1` row offsets. */
  pathOffsets: Uint32Array;
};

/** f64 result of a path-emitting oracle, flattened like `GPULinePathOutput`. */
export type FlatPathResult = {
  positions: number[][];
  pathOffsets: number[];
  sourceRows: number[];
  measures: number[];
};

/** Packs nested paths into the flat layout (values rounded to f32). */
export function createFlatPaths(paths: readonly (readonly number[][])[]): FlatPaths {
  const pathOffsets = new Uint32Array(paths.length + 1);
  const values: number[] = [];
  paths.forEach((path, pathIndex) => {
    for (const vertex of path) {
      values.push(vertex[0], vertex[1]);
    }
    pathOffsets[pathIndex + 1] = pathOffsets[pathIndex] + path.length;
  });
  return {positions: new Float32Array(values), pathOffsets};
}

/** Returns the f32-rounded vertex `row` of a flat path set. */
export function getFlatVertex(paths: FlatPaths, row: number): [number, number] {
  return [paths.positions[2 * row], paths.positions[2 * row + 1]];
}

function getLength(
  a: readonly number[],
  b: readonly number[],
  spherical: boolean,
  radius: number
): number {
  return spherical ? getCentralAngle(a, b) * radius : Math.hypot(b[0] - a[0], b[1] - a[1]);
}

/**
 * f64 reference of `GPULineSegmentize`: pieces per segment
 * `clamp(ceil(length / maximumSegmentLength), 1, maximumPieces)`, input vertices kept, spherical
 * longitudes unwrapped along each path.
 */
export function segmentizePaths(
  paths: FlatPaths,
  options: {
    maximumSegmentLength: number;
    maximumPieces: number;
    spherical?: boolean;
    radius?: number;
  }
): FlatPathResult {
  const spherical = Boolean(options.spherical);
  const radius = options.radius ?? 1;
  const result: FlatPathResult = {
    positions: [],
    pathOffsets: [0],
    sourceRows: [],
    measures: []
  };
  const pathCount = paths.pathOffsets.length - 1;
  for (let path = 0; path < pathCount; path++) {
    const start = paths.pathOffsets[path];
    const end = paths.pathOffsets[path + 1];
    let measure = 0;
    let shift = 0;
    for (let row = start; row < end; row++) {
      const vertex = getFlatVertex(paths, row);
      if (row > start) {
        const previous = getFlatVertex(paths, row - 1);
        shift -= 360 * Math.round((vertex[0] - previous[0]) / 360);
      }
      const shifted = spherical ? [vertex[0] + shift, vertex[1]] : vertex;
      result.positions.push(shifted);
      result.sourceRows.push(row);
      result.measures.push(measure);
      if (row + 1 < end) {
        const next = getFlatVertex(paths, row + 1);
        const length = getLength(vertex, next, spherical, radius);
        const pieces =
          options.maximumSegmentLength > 0 && length > 0
            ? Math.min(
                Math.max(Math.ceil(length / options.maximumSegmentLength), 1),
                options.maximumPieces
              )
            : 1;
        for (let piece = 1; piece < pieces; piece++) {
          const fraction = piece / pieces;
          result.positions.push(
            spherical
              ? interpolateGreatCircle(shifted, next, fraction)
              : [
                  vertex[0] + (next[0] - vertex[0]) * fraction,
                  vertex[1] + (next[1] - vertex[1]) * fraction
                ]
          );
          result.sourceRows.push(row);
          result.measures.push(measure + length * fraction);
        }
        measure += length;
      }
    }
    result.pathOffsets.push(result.positions.length);
  }
  return result;
}

/** f64 reference of `GPUGreatCircleArcs`. */
export function tessellateGreatCircleArcs(
  sources: readonly (readonly number[])[],
  targets: readonly (readonly number[])[],
  options: {
    maximumSegmentLength: number;
    minimumSegments: number;
    maximumSegments: number;
    radius: number;
  }
): FlatPathResult {
  const result: FlatPathResult = {
    positions: [],
    pathOffsets: [0],
    sourceRows: [],
    measures: []
  };
  sources.forEach((source, pair) => {
    const target = [source[0] + wrapLongitudeDelta(targets[pair][0] - source[0]), targets[pair][1]];
    const angle = getCentralAngle(source, target);
    const distance = angle * options.radius;
    let segments =
      options.maximumSegmentLength > 0 && distance > 0
        ? Math.ceil(Math.min(distance / options.maximumSegmentLength, options.maximumSegments))
        : 1;
    if (options.minimumSegments > segments) {
      segments = Math.min(options.minimumSegments, options.maximumSegments);
    }
    segments = Math.min(Math.max(segments, 1), options.maximumSegments);
    for (let vertex = 0; vertex <= segments; vertex++) {
      const fraction = vertex / segments;
      result.positions.push(
        vertex === 0
          ? [source[0], source[1]]
          : vertex === segments
            ? target
            : interpolateGreatCircle(source, target, fraction)
      );
      result.sourceRows.push(pair);
      result.measures.push(distance * fraction);
    }
    result.pathOffsets.push(result.positions.length);
  });
  return result;
}

/** Deterministic xorshift random generator in `[0, 1)`. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}
