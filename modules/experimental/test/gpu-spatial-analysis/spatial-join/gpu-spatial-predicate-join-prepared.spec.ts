// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSpatialJoinCandidates,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicate
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildOracleArrays} from './spatial-predicate-oracle';
import {
  PREPARED_SELF_JOIN_FEATURES,
  PREPARED_SELF_JOIN_OVERLAPS,
  PREPARED_SELF_JOIN_PATTERN
} from './spatial-join-prepared-fixture';

async function runPreparedSelfJoin(
  device: Device,
  spatialSort: boolean,
  candidateCapacity: number,
  predicate: GPUSpatialPredicate,
  engine: 'fast' | 'relate' = 'relate'
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const features = PREPARED_SELF_JOIN_FEATURES;
  const arrays = buildOracleArrays('polygons', features);
  if (arrays.kind !== 'polygons') throw new Error('Expected polygon arrays');
  const uploadOffsets = (name: string, data: Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'uint32', data.length);
  };
  const positions = createInputBuffer(device, arrays.positions);
  buffers.push(positions);
  const geometry: GPUSpatialJoinGeometry = {
    kind: 'polygons',
    positions: importGraphBuffer(
      graph,
      'positions',
      positions,
      'float32x2',
      arrays.positions.length / 2
    ),
    featureOffsets: uploadOffsets('feature-offsets', arrays.featureOffsets),
    polygonOffsets: uploadOffsets('polygon-offsets', arrays.polygonOffsets),
    ringOffsets: uploadOffsets('ring-offsets', arrays.ringOffsets)
  };
  const outputs = new Map<string, {buffer: Buffer; length: number}>();
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs.set(name, {buffer, length});
    return importGraphBuffer(graph, name, buffer, 'uint32', length);
  };
  const pairOutput = (name: string) => ({
    leftIds: output(`${name}-left`, candidateCapacity),
    rightIds: output(`${name}-right`, candidateCapacity),
    status: {
      count: output(`${name}-count`, 1),
      requiredCount: output(`${name}-required`, 1),
      overflow: output(`${name}-overflow`, 1)
    }
  });
  const prepared = new GPUSpatialJoinPrepared({geometry, spatialSort});
  graph.add(prepared);
  graph.add(
    new GPUSpatialJoinCandidates({
      id: 'candidates',
      left: geometry,
      right: geometry,
      prepared,
      output: pairOutput('candidates')
    })
  );
  const common = {
    left: geometry,
    right: geometry,
    prepared,
    predicate,
    engine,
    ...(predicate === 'relate' ? {pattern: PREPARED_SELF_JOIN_PATTERN} : {}),
    excludeSameRow: true,
    candidateCapacity
  };
  const weightBuffer = createOutputBuffer(device, candidateCapacity);
  buffers.push(weightBuffer);
  outputs.set('weights', {buffer: weightBuffer, length: candidateCapacity});
  graph.add(
    new GPUSpatialPredicateJoin({
      ...common,
      id: 'inner',
      output: {
        pairs: pairOutput('pairs'),
        ...(engine === 'relate' ? {relate: output('matrices', candidateCapacity)} : {}),
        weights: {
          offsets: output('offsets', features.length + 1),
          neighbors: output('neighbors', candidateCapacity),
          weights: importGraphBuffer(graph, 'weights', weightBuffer, 'float32', candidateCapacity)
        },
        status: {candidateCount: output('inner-candidates', 1)}
      }
    })
  );
  graph.add(
    new GPUSpatialPredicateJoin({
      ...common,
      id: 'anti',
      how: 'anti',
      output: {
        unmatched: {
          ids: output('unmatched', features.length),
          status: {
            count: output('unmatched-count', 1),
            overflow: output('unmatched-overflow', 1)
          }
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    const results: Record<string, number[]>[] = [];
    // The second encoding reuses the prepared index and must not read stale heap entries.
    for (let encoding = 0; encoding < 2; encoding++) {
      submitGraph(device, compiled, undefined);
      const result: Record<string, number[]> = {};
      for (const [name, {buffer, length}] of outputs) {
        result[name] = await readUint32(buffer, length);
      }
      results.push(result);
    }
    expect(prepared.encodedBuildCount).toBe(1);
    return results;
  } finally {
    compiled.destroy();
    prepared.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function expectMatchingOutputs(
  sorted: Record<string, number[]>[],
  unsorted: Record<string, number[]>[],
  context: string
) {
  expect(sorted.length, context).toBe(unsorted.length);
  for (const [encoding, result] of sorted.entries()) {
    expect(Object.keys(result), context).toEqual(Object.keys(unsorted[encoding]));
    for (const [name, values] of Object.entries(result)) {
      expect(values, `${context}, encoding ${encoding}, ${name}`).toEqual(unsorted[encoding][name]);
    }
  }
  expect(sorted[1], `${context}, sorted repeated encoding`).toEqual(sorted[0]);
  expect(unsorted[1], `${context}, unsorted repeated encoding`).toEqual(unsorted[0]);
}

it('spatially sorted prepared self joins preserve pairs, matrices, CSR, anti rows and overflow prefixes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const predicate of ['overlaps', 'touches', 'relate'] as const) {
    for (const candidateCapacity of [1, 7, 64]) {
      const unsorted = await runPreparedSelfJoin(device, false, candidateCapacity, predicate);
      const sorted = await runPreparedSelfJoin(device, true, candidateCapacity, predicate);
      expectMatchingOutputs(sorted, unsorted, `${predicate}, capacity ${candidateCapacity}`);
      if (predicate === 'overlaps' && candidateCapacity === 64) {
        for (const result of [...sorted, ...unsorted]) {
          expect(result['pairs-count']).toEqual([10]);
          expect(result['pairs-required']).toEqual([10]);
          expect(result['pairs-overflow']).toEqual([0]);
          expect(result['pairs-left'].slice(0, 10)).toEqual(
            PREPARED_SELF_JOIN_OVERLAPS.map(([left]) => left)
          );
          expect(result['pairs-right'].slice(0, 10)).toEqual(
            PREPARED_SELF_JOIN_OVERLAPS.map(([, right]) => right)
          );
          // Row 5 overlaps rows 1 and 3; only the isolated row 4 has no overlap.
          expect(result['unmatched-count']).toEqual([1]);
          expect(result['unmatched'].slice(0, 1)).toEqual([4]);
          expect(result['unmatched-overflow']).toEqual([0]);
        }
      }
    }
  }
});

it('spatially sorted prepared self joins preserve the fast predicate path', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const unsorted = await runPreparedSelfJoin(device, false, 64, 'intersects', 'fast');
  const sorted = await runPreparedSelfJoin(device, true, 64, 'intersects', 'fast');
  expectMatchingOutputs(sorted, unsorted, 'fast intersects, capacity 64');
  for (const result of [...sorted, ...unsorted]) {
    // excludeSameRow removes row 4's reflexive intersects match.
    expect(result['unmatched-count']).toEqual([1]);
    expect(result['unmatched'].slice(0, 1)).toEqual([4]);
    expect(result['unmatched-overflow']).toEqual([0]);
  }
});
