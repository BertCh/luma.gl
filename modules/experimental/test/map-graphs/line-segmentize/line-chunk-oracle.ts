// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCentralAngle, interpolateGreatCircle} from '../geometry-measures/geodesic-oracle';
import {getFlatVertex, type FlatPathResult, type FlatPaths} from './line-segmentize-oracle';

type Piece = {path: number; start: number; end: number; copy: boolean};

/**
 * f64 reference of `GPULineChunk`: chunks of `chunkLength` (`'chunk'`) or one
 * `[startMeasure, endMeasure]` range per path (`'substring'`).
 */
export function chunkPaths(
  paths: FlatPaths,
  options: {
    mode: 'chunk' | 'substring';
    chunkLength?: number;
    startMeasure?: number;
    endMeasure?: number;
    pathCapacity: number;
    spherical?: boolean;
    radius?: number;
  }
): FlatPathResult & {sourcePaths: number[]; pieceTotal: number} {
  const spherical = Boolean(options.spherical);
  const radius = options.radius ?? 6371008.8;
  const pathCount = paths.pathOffsets.length - 1;
  const measures: number[] = [];
  const shifted: number[][] = [];
  for (let path = 0; path < pathCount; path++) {
    let measure = 0;
    let shift = 0;
    for (let row = paths.pathOffsets[path]; row < paths.pathOffsets[path + 1]; row++) {
      const vertex = getFlatVertex(paths, row);
      if (row > paths.pathOffsets[path]) {
        const previous = getFlatVertex(paths, row - 1);
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
  for (let path = 0; path < pathCount; path++) {
    const start = paths.pathOffsets[path];
    const end = paths.pathOffsets[path + 1];
    if (end <= start) {
      continue;
    }
    const length = measures[end - 1];
    if (!(length > 0)) {
      pieces.push({path, start: 0, end: 0, copy: true});
      continue;
    }
    if (options.mode === 'chunk') {
      const chunkLength = options.chunkLength ?? 0;
      const count =
        chunkLength > 0
          ? Math.min(Math.max(Math.ceil(length / chunkLength), 1), options.pathCapacity + 1)
          : 1;
      for (let piece = 0; piece < count; piece++) {
        pieces.push({
          path,
          start: chunkLength > 0 ? piece * chunkLength : 0,
          end: piece + 1 < count ? (piece + 1) * chunkLength : length,
          copy: false
        });
      }
    } else {
      const clamp = (value: number) => Math.min(Math.max(value, 0), length);
      pieces.push({
        path,
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
    pathOffsets: [0],
    sourceRows: [] as number[],
    measures: [] as number[],
    sourcePaths: [] as number[],
    pieceTotal: pieces.length
  };
  for (const piece of pieces.slice(0, options.pathCapacity)) {
    const start = paths.pathOffsets[piece.path];
    const end = paths.pathOffsets[piece.path + 1];
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
    result.pathOffsets.push(result.positions.length);
    result.sourcePaths.push(piece.path);
  }
  while (result.pathOffsets.length < options.pathCapacity + 1) {
    result.pathOffsets.push(result.positions.length);
    result.sourcePaths.push(0xffffffff);
  }
  return result;
}
