// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

export type Point = [number, number];

export type SplitResult = {
  /** Source line of each written piece. */
  lineIds: number[];
  /** Vertex run of each written piece. */
  pieces: Point[][];
  count: number;
  vertexCount: number;
  overflow: number;
  candidateOverflow: number;
  requiredCount: number;
  requiredVertexCount: number;
};

export type SplitOptions = {
  intersectionCapacity?: number;
  pieceCapacity?: number;
  vertexCapacity?: number;
};

/** Flattens lines into GeoArrow arrays. */
export function flattenLines(lines: readonly (readonly Point[])[]) {
  const positions: number[] = [];
  const lineOffsets = [0];
  for (const line of lines) {
    for (const point of line) {
      positions.push(point[0], point[1]);
    }
    lineOffsets.push(positions.length / 2);
  }
  return {positions: new Float32Array(positions), lineOffsets: new Uint32Array(lineOffsets)};
}

/** Runs `GPULineSplit` on a headless device and reads the written pieces back. */
export async function runLineSplit(
  device: Device,
  lines: readonly (readonly Point[])[],
  options: SplitOptions = {}
): Promise<SplitResult> {
  const graph = new GPUCommandGraph(device, {id: 'line-split-test'});
  const buffers: Buffer[] = [];
  const arrays = flattenLines(lines);
  const input = (
    name: string,
    data: Float32Array | Uint32Array,
    format: 'float32x2' | 'uint32'
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      format === 'uint32' ? data.length : data.length / 2
    ) as never;
  };
  const pieceCapacity = options.pieceCapacity ?? 256;
  const vertexCapacity = options.vertexCapacity ?? 2048;
  const output = (name: string, length: number, format: 'uint32' | 'float32x2' = 'uint32') => {
    const buffer = createOutputBuffer(device, format === 'uint32' ? length : length * 2);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length) as never};
  };
  const lineIds = output('piece-lines', pieceCapacity);
  const offsets = output('piece-offsets', pieceCapacity + 1);
  const positions = output('piece-positions', vertexCapacity, 'float32x2');
  const count = output('count', 1);
  const vertexCount = output('vertex-count', 1);
  const overflow = output('overflow', 1);
  const candidateOverflow = output('candidate-overflow', 1);
  const requiredCount = output('total-count', 1);
  const requiredVertexCount = output('total-vertex-count', 1);
  graph.add(
    new GPULineSplit({
      lines: {
        kind: 'lines',
        positions: input('positions', arrays.positions, 'float32x2'),
        lineOffsets: input('line-offsets', arrays.lineOffsets, 'uint32')
      },
      intersectionCapacity: options.intersectionCapacity ?? 512,
      pieces: {
        geometry: {
          kind: 'lines',
          positions: positions.view,
          lineOffsets: offsets.view
        },
        sourceIds: lineIds.view,
        status: {
          count: count.view,
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view,
          requiredCount: requiredCount.view
        },
        vertexCount: vertexCount.view,
        requiredVertexCount: requiredVertexCount.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [pieceCount] = await readUint32(count.buffer, 1);
  const [vertexTotal] = await readUint32(vertexCount.buffer, 1);
  const offsetValues = await readUint32(offsets.buffer, pieceCapacity + 1);
  const positionValues = await readFloat32(positions.buffer, vertexCapacity * 2);
  const ids = await readUint32(lineIds.buffer, pieceCapacity);
  const pieces: Point[][] = [];
  for (let piece = 0; piece < pieceCount; piece++) {
    const run: Point[] = [];
    for (let vertex = offsetValues[piece]; vertex < offsetValues[piece + 1]; vertex++) {
      run.push([positionValues[vertex * 2], positionValues[vertex * 2 + 1]]);
    }
    pieces.push(run);
  }
  const result: SplitResult = {
    lineIds: ids.slice(0, pieceCount),
    pieces,
    count: pieceCount,
    vertexCount: vertexTotal,
    overflow: (await readUint32(overflow.buffer, 1))[0],
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    requiredCount: (await readUint32(requiredCount.buffer, 1))[0],
    requiredVertexCount: (await readUint32(requiredVertexCount.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}
