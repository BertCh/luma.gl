// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getFlatVertex, type FlatLineResult, type FlatLines} from './line-segmentize-oracle';

function smoothOnce(line: number[][], ratio: number, closed: boolean): number[][] {
  const result: number[][] = closed ? [] : [line[0]];
  const edgeCount = closed ? line.length : line.length - 1;
  for (let edge = 0; edge < edgeCount; edge++) {
    const start = line[edge];
    const end = line[(edge + 1) % line.length];
    result.push(
      [start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio],
      [start[0] + (end[0] - start[0]) * (1 - ratio), start[1] + (end[1] - start[1]) * (1 - ratio)]
    );
  }
  if (!closed) {
    result.push(line[line.length - 1]);
  }
  return result;
}

/** f64 reference of `GPULineSmooth` (Chaikin corner cutting, `iterations` times). */
export function smoothLines(
  lines: FlatLines,
  options: {iterationCount: number; ratio: number; closed: boolean}
): FlatLineResult {
  const result: FlatLineResult = {positions: [], lineOffsets: [0], sourceRows: [], measures: []};
  for (let line = 0; line + 1 < lines.lineOffsets.length; line++) {
    let vertices: number[][] = [];
    for (let row = lines.lineOffsets[line]; row < lines.lineOffsets[line + 1]; row++) {
      vertices.push(getFlatVertex(lines, row));
    }
    if (options.closed && vertices.length >= 2) {
      const first = vertices[0];
      const last = vertices[vertices.length - 1];
      if (first[0] === last[0] && first[1] === last[1]) {
        vertices = vertices.slice(0, -1);
      }
    }
    const smoothable = options.closed ? vertices.length >= 3 : vertices.length >= 2;
    if (!smoothable) {
      vertices = [];
      for (let row = lines.lineOffsets[line]; row < lines.lineOffsets[line + 1]; row++) {
        vertices.push(getFlatVertex(lines, row));
      }
    } else {
      for (let iteration = 0; iteration < options.iterationCount; iteration++) {
        vertices = smoothOnce(vertices, options.ratio, options.closed);
      }
      if (options.closed) {
        vertices.push(vertices[0]);
      }
    }
    result.positions.push(...vertices);
    result.lineOffsets.push(result.positions.length);
  }
  return result;
}
