// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCentralAngle, interpolateGreatCircle} from '../geometry-measures/geodesic-oracle';
import {getFlatVertex, type FlatLineResult, type FlatLines} from './line-segmentize-oracle';

type Piece = {line: number; start: number; end: number; copy: boolean};

/**
 * f64 reference of `GPULineChunk`: chunks of `chunkLength` (`'chunk'`) or one
 * `[startMeasure, endMeasure]` range per line (`'substring'`).
 */
export function chunkLines(
  lines: FlatLines,
  options: {
    mode: 'chunk' | 'substring';
    chunkLength?: number;
    startMeasure?: number;
    endMeasure?: number;
    lineCapacity: number;
    spherical?: boolean;
    radius?: number;
  }
): FlatLineResult & {sourceLineIndices: number[]; pieceTotal: number} {
  const spherical = Boolean(options.spherical);
  const radius = options.radius ?? 6371008.8;
  const lineCount = lines.lineOffsets.length - 1;
  const measures: number[] = [];
  const shifted: number[][] = [];
  for (let line = 0; line < lineCount; line++) {
    let measure = 0;
    let shift = 0;
    for (let row = lines.lineOffsets[line]; row < lines.lineOffsets[line + 1]; row++) {
      const vertex = getFlatVertex(lines, row);
      if (row > lines.lineOffsets[line]) {
        const previous = getFlatVertex(lines, row - 1);
        measure += spherical
          ? getCentralAngle(previous, vertex) * radius
          : Math.hypot(vertex[0] - previous[0], vertex[1] - previous[1]);
        shift -= 360 * Math.round((vertex[0] - previous[0]) / 360);
      }
      measures[row] = measure;
      shifted[row] = spherical ? [vertex[0] + shift, vertex[1]] : vertex;
    }
  }
  const pieces: Piece[] = [];
  for (let line = 0; line < lineCount; line++) {
    const start = lines.lineOffsets[line];
    const end = lines.lineOffsets[line + 1];
    if (end <= start) {
      continue;
    }
    const length = measures[end - 1];
    if (!(length > 0)) {
      pieces.push({line, start: 0, end: 0, copy: true});
      continue;
    }
    if (options.mode === 'chunk') {
      const chunkLength = options.chunkLength ?? 0;
      const count =
        chunkLength > 0
          ? Math.min(Math.max(Math.ceil(length / chunkLength), 1), options.lineCapacity + 1)
          : 1;
      for (let piece = 0; piece < count; piece++) {
        pieces.push({
          line,
          start: chunkLength > 0 ? piece * chunkLength : 0,
          end: piece + 1 < count ? (piece + 1) * chunkLength : length,
          copy: false
        });
      }
    } else {
      const clamp = (value: number) => Math.min(Math.max(value, 0), length);
      pieces.push({
        line,
        start: clamp(options.startMeasure ?? 0),
        end: clamp(options.endMeasure ?? length),
        copy: false
      });
    }
  }
  const interpolate = (row: number, measure: number): number[] => {
    const segmentMeasure = measures[row + 1] - measures[row];
    const fraction =
      segmentMeasure > 0 ? Math.min(Math.max((measure - measures[row]) / segmentMeasure, 0), 1) : 0;
    const a = shifted[row];
    const b = shifted[row + 1];
    return spherical
      ? interpolateGreatCircle(a, b, fraction)
      : [a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction];
  };
  const result = {
    positions: [] as number[][],
    lineOffsets: [0],
    sourceRows: [] as number[],
    measures: [] as number[],
    sourceLineIndices: [] as number[],
    pieceTotal: pieces.length
  };
  for (const piece of pieces.slice(0, options.lineCapacity)) {
    const start = lines.lineOffsets[piece.line];
    const end = lines.lineOffsets[piece.line + 1];
    const push = (vertex: number[], measure: number, row: number) => {
      result.positions.push(vertex);
      result.measures.push(measure);
      result.sourceRows.push(row);
    };
    if (piece.copy) {
      for (let row = start; row < end; row++) {
        push(shifted[row], measures[row], row);
      }
    } else if (piece.start <= piece.end) {
      let firstInterior = start;
      while (firstInterior < end && measures[firstInterior] <= piece.start) {
        firstInterior++;
      }
      let interiorEnd = start;
      while (interiorEnd < end && measures[interiorEnd] < piece.end) {
        interiorEnd++;
      }
      interiorEnd = Math.max(interiorEnd, firstInterior);
      const clampRow = (row: number) => Math.min(Math.max(row, start + 1), end - 1) - 1;
      const startSegment = clampRow(firstInterior);
      push(interpolate(startSegment, piece.start), piece.start, startSegment);
      for (let row = firstInterior; row < interiorEnd; row++) {
        push(shifted[row], measures[row], row);
      }
      const endSegment = clampRow(interiorEnd);
      push(interpolate(endSegment, piece.end), piece.end, endSegment);
    }
    result.lineOffsets.push(result.positions.length);
    result.sourceLineIndices.push(piece.line);
  }
  while (result.lineOffsets.length < options.lineCapacity + 1) {
    result.lineOffsets.push(result.positions.length);
    result.sourceLineIndices.push(0xffffffff);
  }
  return result;
}
