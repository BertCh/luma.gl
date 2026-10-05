// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getFlatVertex, type FlatPathResult, type FlatPaths} from './line-segmentize-oracle';

function smoothOnce(path: number[][], ratio: number, closed: boolean): number[][] {
  const result: number[][] = closed ? [] : [path[0]];
  const edgeCount = closed ? path.length : path.length - 1;
  for (let edge = 0; edge < edgeCount; edge++) {
    const start = path[edge];
    const end = path[(edge + 1) % path.length];
    result.push(
      [start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio],
      [start[0] + (end[0] - start[0]) * (1 - ratio), start[1] + (end[1] - start[1]) * (1 - ratio)]
    );
  }
  if (!closed) {
    result.push(path[path.length - 1]);
  }
  return result;
}

/** f64 reference of `GPULineSmooth` (Chaikin corner cutting, `iterations` times). */
export function smoothPaths(
  paths: FlatPaths,
  options: {iterations: number; ratio: number; closed: boolean}
): FlatPathResult {
  const result: FlatPathResult = {positions: [], pathOffsets: [0], sourceRows: [], measures: []};
  for (let path = 0; path + 1 < paths.pathOffsets.length; path++) {
    let vertices: number[][] = [];
    for (let row = paths.pathOffsets[path]; row < paths.pathOffsets[path + 1]; row++) {
      vertices.push(getFlatVertex(paths, row));
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
      for (let row = paths.pathOffsets[path]; row < paths.pathOffsets[path + 1]; row++) {
        vertices.push(getFlatVertex(paths, row));
      }
    } else {
      for (let iteration = 0; iteration < options.iterations; iteration++) {
        vertices = smoothOnce(vertices, options.ratio, options.closed);
      }
      if (options.closed) {
        vertices.push(vertices[0]);
      }
    }
    result.positions.push(...vertices);
    result.pathOffsets.push(result.positions.length);
  }
  return result;
}
