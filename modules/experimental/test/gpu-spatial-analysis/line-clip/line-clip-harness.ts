// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPULineClipByPolygon,
  GPUSharedPaths
} from '../../../src/gpu-spatial-analysis/line-clip/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

export type Point = [number, number];

/** Features of polygons of rings of vertices. */
export type PolygonFeatures = readonly (readonly (readonly (readonly (readonly number[])[])[])[])[];

export type ClipResult = {
  lineIds: number[];
  pieces: Point[][];
  count: number;
  vertexCount: number;
  overflow: number;
  requiredCount: number;
  requiredVertexCount: number;
  uncertainCount: number;
};

export type ClipOptions = {
  mode?: 'inside' | 'outside';
  intersectionCapacity?: number;
  candidateCapacity?: number;
  pieceCapacity?: number;
  vertexCapacity?: number;
};

/** Flattens lines into GeoArrow arrays. */
export function flattenLines(lines: readonly (readonly (readonly number[])[])[]) {
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

/** Flattens polygon features into GeoArrow arrays. */
export function flattenPolygons(features: PolygonFeatures) {
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const feature of features) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const point of ring) {
          positions.push(point[0], point[1]);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: new Float32Array(positions),
    featureOffsets: new Uint32Array(featureOffsets),
    polygonOffsets: new Uint32Array(polygonOffsets),
    ringOffsets: new Uint32Array(ringOffsets)
  };
}

type Harness = {
  buffers: Buffer[];
  graph: GPUCommandGraph;
  input: (name: string, data: Float32Array | Uint32Array, format: 'float32x2' | 'uint32') => never;
  output: (
    name: string,
    length: number,
    format?: 'uint32' | 'float32x2'
  ) => {buffer: Buffer; view: never};
};

function createHarness(device: Device, id: string): Harness {
  const graph = new GPUCommandGraph(device, {id});
  const buffers: Buffer[] = [];
  return {
    buffers,
    graph,
    input: (name, data, format) => {
      const buffer = createInputBuffer(device, data);
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        format === 'uint32' ? data.length : data.length / 2
      ) as never;
    },
    output: (name, length, format = 'uint32') => {
      const buffer = createOutputBuffer(device, format === 'uint32' ? length : length * 2);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length) as never};
    }
  };
}

async function readRuns(
  offsets: Buffer,
  positions: Buffer,
  runCount: number,
  runCapacity: number,
  vertexCapacity: number
): Promise<Point[][]> {
  const offsetValues = await readUint32(offsets, runCapacity + 1);
  const positionValues = await readFloat32(positions, vertexCapacity * 2);
  const runs: Point[][] = [];
  for (let run = 0; run < runCount; run++) {
    const points: Point[] = [];
    for (let vertex = offsetValues[run]; vertex < offsetValues[run + 1]; vertex++) {
      points.push([positionValues[vertex * 2], positionValues[vertex * 2 + 1]]);
    }
    runs.push(points);
  }
  return runs;
}

/** Runs `GPULineClipByPolygon` on a headless device and reads the pieces back. */
export async function runLineClip(
  device: Device,
  lines: readonly (readonly (readonly number[])[])[],
  features: PolygonFeatures,
  options: ClipOptions = {}
): Promise<ClipResult> {
  const harness = createHarness(device, 'line-clip-test');
  const {graph, buffers, input, output} = harness;
  const lineArrays = flattenLines(lines);
  const polygonArrays = flattenPolygons(features);
  const pieceCapacity = options.pieceCapacity ?? 256;
  const vertexCapacity = options.vertexCapacity ?? 2048;
  const lineIds = output('piece-lines', pieceCapacity);
  const offsets = output('piece-offsets', pieceCapacity + 1);
  const positions = output('piece-positions', vertexCapacity, 'float32x2');
  const count = output('count', 1);
  const vertexCount = output('vertex-count', 1);
  const overflow = output('overflow', 1);
  const requiredCount = output('total-count', 1);
  const requiredVertexCount = output('total-vertex-count', 1);
  const uncertainCount = output('uncertain-count', 1);
  graph.add(
    new GPULineClipByPolygon({
      lines: {
        kind: 'lines',
        positions: input('positions', lineArrays.positions, 'float32x2'),
        lineOffsets: input('line-offsets', lineArrays.lineOffsets, 'uint32')
      },
      polygons: {
        kind: 'polygons',
        positions: input('polygon-positions', polygonArrays.positions, 'float32x2'),
        featureOffsets: input('feature-offsets', polygonArrays.featureOffsets, 'uint32'),
        polygonOffsets: input('polygon-offsets', polygonArrays.polygonOffsets, 'uint32'),
        ringOffsets: input('ring-offsets', polygonArrays.ringOffsets, 'uint32')
      },
      mode: options.mode,
      intersectionCapacity: options.intersectionCapacity ?? 2048,
      candidateCapacity: options.candidateCapacity ?? 4096,
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
          requiredCount: requiredCount.view
        },
        vertexCount: vertexCount.view,
        requiredVertexCount: requiredVertexCount.view
      },
      uncertainCount: uncertainCount.view
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [pieceCount] = await readUint32(count.buffer, 1);
  const ids = await readUint32(lineIds.buffer, pieceCapacity);
  const result: ClipResult = {
    lineIds: ids.slice(0, pieceCount),
    pieces: await readRuns(
      offsets.buffer,
      positions.buffer,
      pieceCount,
      pieceCapacity,
      vertexCapacity
    ),
    count: pieceCount,
    vertexCount: (await readUint32(vertexCount.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    requiredCount: (await readUint32(requiredCount.buffer, 1))[0],
    requiredVertexCount: (await readUint32(requiredVertexCount.buffer, 1))[0],
    uncertainCount: (await readUint32(uncertainCount.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

export type SharedResult = {
  leftLineIds: number[];
  rightLineIds: number[];
  forward: number[];
  runs: Point[][];
  count: number;
  vertexCount: number;
  overflow: number;
  requiredCount: number;
};

export type SharedOptions = {
  intersectionCapacity?: number;
  runCapacity?: number;
  vertexCapacity?: number;
};

/** Runs `GPUSharedPaths` on a headless device and reads the runs back. */
export async function runSharedPaths(
  device: Device,
  left: readonly (readonly (readonly number[])[])[],
  right: readonly (readonly (readonly number[])[])[],
  options: SharedOptions = {}
): Promise<SharedResult> {
  const harness = createHarness(device, 'shared-paths-test');
  const {graph, buffers, input, output} = harness;
  const leftArrays = flattenLines(left);
  const rightArrays = flattenLines(right);
  const runCapacity = options.runCapacity ?? 128;
  const vertexCapacity = options.vertexCapacity ?? 1024;
  const leftLineIds = output('left-line-ids', runCapacity);
  const rightLineIds = output('right-line-ids', runCapacity);
  const forward = output('forward', runCapacity);
  const offsets = output('run-offsets', runCapacity + 1);
  const positions = output('run-positions', vertexCapacity, 'float32x2');
  const count = output('count', 1);
  const vertexCount = output('vertex-count', 1);
  const overflow = output('overflow', 1);
  const requiredCount = output('total-count', 1);
  graph.add(
    new GPUSharedPaths({
      left: {
        kind: 'lines',
        positions: input('left-positions', leftArrays.positions, 'float32x2'),
        lineOffsets: input('left-offsets', leftArrays.lineOffsets, 'uint32')
      },
      right: {
        kind: 'lines',
        positions: input('right-positions', rightArrays.positions, 'float32x2'),
        lineOffsets: input('right-offsets', rightArrays.lineOffsets, 'uint32')
      },
      intersectionCapacity: options.intersectionCapacity ?? 1024,
      runs: {
        leftLineIds: leftLineIds.view,
        rightLineIds: rightLineIds.view,
        forward: forward.view,
        offsets: offsets.view,
        positions: positions.view,
        count: count.view,
        vertexCount: vertexCount.view,
        overflow: overflow.view,
        requiredCount: requiredCount.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [runTotal] = await readUint32(count.buffer, 1);
  const result: SharedResult = {
    leftLineIds: (await readUint32(leftLineIds.buffer, runCapacity)).slice(0, runTotal),
    rightLineIds: (await readUint32(rightLineIds.buffer, runCapacity)).slice(0, runTotal),
    forward: (await readUint32(forward.buffer, runCapacity)).slice(0, runTotal),
    runs: await readRuns(offsets.buffer, positions.buffer, runTotal, runCapacity, vertexCapacity),
    count: runTotal,
    vertexCount: (await readUint32(vertexCount.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    requiredCount: (await readUint32(requiredCount.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}
