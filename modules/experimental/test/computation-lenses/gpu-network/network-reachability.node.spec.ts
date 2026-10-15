// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {createNetworkLensGraph} from '../../../src/computation-lenses/scenarios/gpu-network/network-lens-support';
import '../../../src/computation-lenses/scenarios/gpu-network/index';
import {validateGPUComputationRecord, type GPUComputationCapture} from '@luma.gl/gpgpu';
import {
  describeStructure,
  makeReachabilityStamp,
  selectReachabilitySnapshots
} from '../../../src/computation-lenses/scenarios/gpu-network/network-reachability.lens';
import {describeStructure as describeChained} from '../../../src/computation-lenses/scenarios/gpu-network/network-reachability.chained.lens';
import {
  extractLensExample,
  findNonPublicImports
} from '../../../src/computation-lenses/lens-example';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const [chained, describe] of [
  [false, describeStructure],
  [true, describeChained]
] as const)
  test(`reachability structure on a null device (chained=${chained})`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const record = await describe(device);
      expect(validateGPUComputationRecord(record).issues).toEqual([]);
      const relax = record.graphs[0].nodes.filter(node => /^reachability-relax-\d+$/.test(node.id));
      expect(relax.map(node => node.id)).toEqual(
        Array.from({length: 256}, (_, iteration) => `reachability-relax-${iteration}`)
      );
      for (const [iteration, node] of relax.entries()) {
        expect(node.loop).toEqual({
          id: 'reachability-relax',
          iteration,
          maximumIterations: 256,
          phase: 'gated'
        });
        expect(node.datum).toEqual({keySpace: 'node', relation: 'frontier-queue'});
        expect(node.attrs['chaotic']).toBe(true);
      }
      expect(record.choices[0].value).toBe(chained ? 32 : 1);
      expect(record.graphs[0].attrs?.['localIterations']).toBe(chained ? 32 : 1);
      expect(
        record.axes.relations.find(relation => relation.id === 'frontier-queue')
      ).toMatchObject({
        timeDependent: true,
        kind: 'buffer',
        ref: {layout: 'index', resource: 'reachability-queues'}
      });
    } finally {
      device.destroy();
    }
  });

test('the shared example is short, public and ends with the displayed outputs', () => {
  const source = readFileSync(
    new URL(
      '../../../src/computation-lenses/scenarios/gpu-network/network-reachability.example.ts',
      import.meta.url
    ),
    'utf8'
  );
  expect(findNonPublicImports(source)).toEqual([]);
  expect(extractLensExample(source).code).toMatch(
    /const \{costs, bands, predecessors\} = reachability.props.output;$/
  );
});

function capture(afterNode: string, values: number[], name = 'costs'): GPUComputationCapture {
  return {id: afterNode, name, afterNode, elementType: 'f32', provenance: 'gpu-readback', values};
}

test('snapshot rounds count completed relax nodes, retain seed and final, and omit duplicate tails', () => {
  const captures = [
    capture('reachability-initialize', [Infinity, Infinity]),
    capture('reachability-seed', [0, Infinity]),
    capture('reachability-relax-0', [0, 3]),
    capture('reachability-relax-1', [0, 2]),
    capture('reachability-relax-2', [0, 2]),
    capture('reachability-relax-3', [0, 2]),
    capture('reachability-bands', [0, 2])
  ];
  const snapshots = selectReachabilitySnapshots(captures, 3);
  expect(snapshots.map(snapshot => snapshot.iteration)).toEqual([0, 1, 3]);
  expect(snapshots.at(-1)?.afterNode).toBe('reachability-relax-2');
});

test('reachability snapshots keep log-spaced key frames and the final round', () => {
  const captures = [
    capture('reachability-seed', [0]),
    ...Array.from({length: 11}, (_, round) => capture(`reachability-relax-${round}`, [round + 1]))
  ];
  const snapshots = selectReachabilitySnapshots(captures, 11);
  expect(snapshots.map(snapshot => snapshot.iteration)).toEqual([0, 1, 2, 4, 8, 11]);
  expect(snapshots.every(snapshot => snapshot.meaning?.includes('6 other executed rounds'))).toBe(
    true
  );
});

test('stamp arithmetic includes seed round zero and rejects stamps overwritten by a tie phase', () => {
  const queued = capture('end', [1, 2, 4, 0], 'queuedRound');
  expect(Array.from(makeReachabilityStamp(queued, 4, false).values)).toEqual([0, 1, 3, NaN]);
  expect(makeReachabilityStamp(queued, 4, true).meaning).toBe(
    'last global-queue round (chained hops inside a round are not stamped)'
  );
  expect(() => makeReachabilityStamp(capture('end', [7]), 4, false)).toThrow();
});

test('only observed transients receive copy-source usage in the lens graph', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = createNetworkLensGraph(device, 'usage', ['observed']);
    expect(
      graph.createTransientBuffer({id: 'observed', byteLength: 4, usage: Buffer.STORAGE}).usage
    ).toBe(Buffer.STORAGE | Buffer.COPY_SRC);
    expect(
      graph.createTransientBuffer({id: 'scratch', byteLength: 4, usage: Buffer.STORAGE}).usage
    ).toBe(Buffer.STORAGE);
  } finally {
    device.destroy();
  }
});
