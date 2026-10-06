// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  formatGPUSpatialRelate,
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicate
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {buildOracleArrays, type OracleFeature} from './spatial-predicate-oracle';

const INTERSECTS_PATTERNS = ['T********', '*T*******', '***T*****', '****T****'];

export type JoinResult = {
  pairs: string[];
  matrices: string[];
  overflow: number;
  uncertainCount: number;
  usesRelateEngine: boolean;
  usesWorkgroupDistance: boolean;
};

function createGeometry(
  device: Device,
  graph: GPUCommandGraph,
  name: string,
  features: OracleFeature[],
  buffers: Buffer[]
): GPUSpatialJoinGeometry {
  const arrays = buildOracleArrays(features[0].kind, features);
  const upload = (data: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const positions = importGraphBuffer(
    graph,
    `${name}-positions`,
    upload(arrays.positions),
    'float32x2',
    arrays.positions.length / 2
  );
  const offsets = (suffix: string, data: Uint32Array) =>
    importGraphBuffer(graph, `${name}-${suffix}`, upload(data), 'uint32', data.length);
  if (arrays.kind === 'points') {
    return {kind: 'points', positions};
  }
  if (arrays.kind === 'lines') {
    return {kind: 'lines', positions, lineOffsets: offsets('line-offsets', arrays.lineOffsets)};
  }
  return {
    kind: 'polygons',
    positions,
    featureOffsets: offsets('feature-offsets', arrays.featureOffsets),
    polygonOffsets: offsets('polygon-offsets', arrays.polygonOffsets),
    ringOffsets: offsets('ring-offsets', arrays.ringOffsets)
  };
}

export async function runScaleJoin(
  device: Device,
  lefts: OracleFeature[],
  rights: OracleFeature[],
  predicate: GPUSpatialPredicate,
  options: {engine?: 'auto' | 'fast' | 'relate'; matrix?: boolean; distance?: number}
): Promise<JoinResult> {
  const graph = new GPUCommandGraph(device, {id: 'relate-scale'});
  const buffers: Buffer[] = [];
  const left = createGeometry(device, graph, 'left', lefts, buffers);
  const right = createGeometry(device, graph, 'right', rights, buffers);
  const capacity = lefts.length * rights.length;
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const leftIds = output('left-ids', capacity);
  const rightIds = output('right-ids', capacity);
  const count = output('count', 1);
  const overflow = output('overflow', 1);
  const relate = output('relate', capacity);
  const uncertain = output('uncertain', 1);
  const join = new GPUSpatialPredicateJoin({
    left,
    right,
    predicate,
    pattern: predicate === 'relate' ? INTERSECTS_PATTERNS : undefined,
    engine: options.engine,
    distance: options.distance,
    candidateCapacity: capacity,
    pairs: {
      leftIds: leftIds.view,
      rightIds: rightIds.view,
      count: count.view,
      overflow: overflow.view
    },
    relate: options.matrix ? relate.view : undefined,
    uncertainCount: uncertain.view
  });
  graph.add(join);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [countValue] = await readUint32(count.buffer, 1);
  const lefted = (await readUint32(leftIds.buffer, capacity)).slice(0, countValue);
  const righted = (await readUint32(rightIds.buffer, capacity)).slice(0, countValue);
  const matrices = (await readUint32(relate.buffer, capacity)).slice(0, countValue);
  const result: JoinResult = {
    pairs: lefted.map((row, slot) => `${row},${righted[slot]}`),
    matrices: options.matrix ? matrices.map(formatGPUSpatialRelate) : [],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    uncertainCount: (await readUint32(uncertain.buffer, 1))[0],
    usesRelateEngine: join.usesRelateEngine,
    usesWorkgroupDistance: join.usesWorkgroupDistance
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

export function comparePairs(first: string, second: string): number {
  const [firstLeft, firstRight] = first.split(',').map(Number);
  const [secondLeft, secondRight] = second.split(',').map(Number);
  return firstLeft - secondLeft || firstRight - secondRight;
}
