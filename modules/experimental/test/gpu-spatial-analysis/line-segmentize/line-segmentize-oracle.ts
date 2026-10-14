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

/**
 * Absolute degree error bound for an interior f32 lerp of exact f32 endpoints.
 * With u = 2^-24, division contributes at most 2.5 ULP <= 5u to the fraction.
 * For a + t*d, budget u|raw| for subtraction, u|d| for longitude wrapping,
 * 5u|d| for the fraction, u|d| for multiplication and u(|a|+|d|) for addition.
 * Also cover mix(a,b,t) = a*(1-t)+b*t: 5u|d| + u(2|a|+|b|+max(|a|,|b|)).
 * The denominator bounds higher-order rounding products; fused arithmetic needs no extra error.
 * A final longitude wrap is exact by Sterbenz (180 < |longitude| <= 360).
 */
export function getLinearInterpolationErrorBound(
  start: number,
  end: number,
  longitude: boolean = false
): number {
  const unitRoundoff = 2 ** -24;
  const rawStep = end - start;
  const step = Math.abs(longitude ? wrapLongitudeDelta(rawStep) : rawStep);
  const differenceForm = Math.abs(rawStep) + (longitude ? 8 : 7) * step + Math.abs(start);
  const weightedForm =
    5 * step + 2 * Math.abs(start) + Math.abs(end) + Math.max(Math.abs(start), Math.abs(end));
  return (unitRoundoff * Math.max(differenceForm, weightedForm)) / (1 - 8 * unitRoundoff);
}

function getLength(
  a: readonly number[],
  b: readonly number[],
  spherical: boolean,
  radius: number
): number {
  return spherical ? getCentralAngle(a, b) * radius : Math.hypot(b[0] - a[0], b[1] - a[1]);
}

/** Binary64 reference for open/closed, length/degree-step and wrapped-turn segmentization. */
export function segmentizeLines(
  lines: FlatLines,
  options: {
    maximumSegmentLength: number;
    maximumPieces: number;
    spherical?: boolean;
    radius?: number;
    closedLines?: boolean;
    pieceCountModel?: 'length' | 'degree-step';
    maximumSegmentDegrees?: number;
    interpolation?: 'planar' | 'linear-degrees' | 'great-circle';
    wrapOutputLongitudes?: boolean;
  }
): FlatLineResult & {turns: number[]; approximation: number} {
  const spherical = Boolean(options.spherical);
  const interpolation = options.interpolation ?? (spherical ? 'great-circle' : 'planar');
  const geographic = interpolation !== 'planar';
  const radius = options.radius ?? 1;
  const result: FlatLineResult & {turns: number[]; approximation: number} = {
    positions: [],
    lineOffsets: [0],
    sourceRows: [],
    measures: [],
    turns: [],
    approximation: 0
  };
  const positionBits = new Uint32Array(
    lines.positions.buffer,
    lines.positions.byteOffset,
    lines.positions.length
  );
  for (let line = 0; line < lines.lineOffsets.length - 1; line++) {
    const first = lines.lineOffsets[line];
    let last = lines.lineOffsets[line + 1];
    if (
      options.closedLines &&
      last > first + 1 &&
      positionBits[2 * first] === positionBits[2 * (last - 1)] &&
      positionBits[2 * first + 1] === positionBits[2 * (last - 1) + 1]
    ) {
      last--;
    }
    let measure = 0;
    let turn = 0;
    for (let row = first; row < last; row++) {
      const vertex = getFlatVertex(lines, row);
      if (row > first && geographic) {
        const previous = getFlatVertex(lines, row - 1);
        turn -= Math.round((vertex[0] - previous[0]) / 360);
      }
      const hasEdge = row + 1 < last || Boolean(options.closedLines && last > first + 1);
      const next = getFlatVertex(lines, row + 1 < last ? row + 1 : first);
      const length = hasEdge ? getLength(vertex, next, spherical, radius) : 0;
      const longitudeStep = wrapLongitudeDelta(next[0] - vertex[0]);
      const extent =
        options.pieceCountModel === 'degree-step'
          ? Math.max(
              Math.abs(longitudeStep),
              interpolation === 'great-circle'
                ? (getCentralAngle(vertex, next) * 180) / Math.PI
                : Math.abs(next[1] - vertex[1])
            )
          : length;
      const maximumStep =
        options.pieceCountModel === 'degree-step'
          ? (options.maximumSegmentDegrees ?? options.maximumSegmentLength)
          : options.maximumSegmentLength;
      const requested =
        hasEdge && maximumStep > 0 ? Math.max(Math.ceil(extent / maximumStep), 1) : 1;
      const pieces = Math.min(requested, options.maximumPieces);
      if (requested > options.maximumPieces) result.approximation = 1;
      for (let piece = 0; piece < pieces; piece++) {
        const fraction = piece / pieces;
        const position =
          piece === 0
            ? [...vertex]
            : interpolation === 'great-circle'
              ? interpolateGreatCircle(vertex, next, fraction)
              : [
                  vertex[0] + (geographic ? longitudeStep : next[0] - vertex[0]) * fraction,
                  vertex[1] + (next[1] - vertex[1]) * fraction
                ];
        let outputTurn = geographic ? turn : 0;
        if (options.wrapOutputLongitudes) {
          if (position[0] < -180 || position[0] > 180) {
            const localTurn = Math.floor((position[0] + 180) / 360);
            position[0] -= 360 * localTurn;
            outputTurn += localTurn;
          }
        } else if (outputTurn !== 0) {
          position[0] += 360 * outputTurn;
        }
        result.positions.push(position);
        result.turns.push(outputTurn);
        result.sourceRows.push(row);
        result.measures.push(measure + length * fraction);
      }
      measure += length;
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
