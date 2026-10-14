// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCentralAngle,
  interpolateGreatCircle,
  wrapLongitudeDelta
} from '../geometry-measures/geodesic-oracle';

/** Flat line layout used by the line contributor tests. */
export type FlatLines = {
  /** Interleaved x/y (or longitude/latitude) values. */
  positions: Float32Array;
  /** `lineCount + 1` row offsets. */
  lineOffsets: Uint32Array;
};

/** f64 result of a line-emitting oracle, flattened like `GPULineGeometryOutput`. */
export type FlatLineResult = {
  positions: number[][];
  lineOffsets: number[];
  sourceRows: number[];
  measures: number[];
};

/** Packs nested lines into the flat layout (values rounded to f32). */
export function createFlatLines(lines: readonly (readonly number[][])[]): FlatLines {
  const lineOffsets = new Uint32Array(lines.length + 1);
  const values: number[] = [];
  lines.forEach((line, lineIndex) => {
    for (const vertex of line) {
      values.push(vertex[0], vertex[1]);
    }
    lineOffsets[lineIndex + 1] = lineOffsets[lineIndex] + line.length;
  });
  return {positions: new Float32Array(values), lineOffsets};
}

/** Returns the f32-rounded vertex `row` of a flat line set. */
export function getFlatVertex(lines: FlatLines, row: number): [number, number] {
  return [lines.positions[2 * row], lines.positions[2 * row + 1]];
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
 * longitudes unwrapped along each line.
 */
export function segmentizeLines(
  lines: FlatLines,
  options: {
    maximumSegmentLength: number;
    maximumPieces: number;
    spherical?: boolean;
    radius?: number;
  }
): FlatLineResult {
  const spherical = Boolean(options.spherical);
  const radius = options.radius ?? 1;
  const result: FlatLineResult = {
    positions: [],
    lineOffsets: [0],
    sourceRows: [],
    measures: []
  };
  const lineCount = lines.lineOffsets.length - 1;
  for (let line = 0; line < lineCount; line++) {
    const start = lines.lineOffsets[line];
    const end = lines.lineOffsets[line + 1];
    let measure = 0;
    let shift = 0;
    for (let row = start; row < end; row++) {
      const vertex = getFlatVertex(lines, row);
      if (row > start) {
        const previous = getFlatVertex(lines, row - 1);
        shift -= 360 * Math.round((vertex[0] - previous[0]) / 360);
      }
      const shifted = spherical ? [vertex[0] + shift, vertex[1]] : vertex;
      result.positions.push(shifted);
      result.sourceRows.push(row);
      result.measures.push(measure);
      if (row + 1 < end) {
        const next = getFlatVertex(lines, row + 1);
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
    result.lineOffsets.push(result.positions.length);
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
    maximumSegmentCount: number;
    radius: number;
  }
): FlatLineResult {
  const result: FlatLineResult = {
    positions: [],
    lineOffsets: [0],
    sourceRows: [],
    measures: []
  };
  sources.forEach((source, pair) => {
    const target = [source[0] + wrapLongitudeDelta(targets[pair][0] - source[0]), targets[pair][1]];
    const angle = getCentralAngle(source, target);
    const distance = angle * options.radius;
    let segments =
      options.maximumSegmentLength > 0 && distance > 0
        ? Math.ceil(Math.min(distance / options.maximumSegmentLength, options.maximumSegmentCount))
        : 1;
    if (options.minimumSegments > segments) {
      segments = Math.min(options.minimumSegments, options.maximumSegmentCount);
    }
    segments = Math.min(Math.max(segments, 1), options.maximumSegmentCount);
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
    result.lineOffsets.push(result.positions.length);
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
