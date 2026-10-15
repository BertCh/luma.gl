// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/engine';
import {GPUCommandGraph, createTransientView, getViewBinding} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGPUCommandGraph,
  createGPUComputationCaptures,
  createGPUComputationRecord,
  validateGPUComputationRecord,
  validateGPUComputationSidecars
} from '@luma.gl/gpgpu';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';

function createChain(device: Device) {
  const graph = new GPUCommandGraph(device, {id: 'capture-chain', recording: {retain: ['state']}});
  // The nonzero view offset also exercises capture source range resolution.
  const storage = createTransientView(
    graph,
    'state',
    'uint32',
    2,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const state = graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset: 4});
  for (let iteration = 1; iteration <= 3; iteration++) {
    graph.addComputePass({
      id: `iteration-${iteration}`,
      publication: {
        id: `result-${iteration}`,
        completeness: iteration === 3 ? 'complete' : 'partial'
      },
      resources: [{buffer: state, usage: 'storage-read-write'}],
      compile: () => {
        const kernel = new Kernel(device, {
          source: `@group(0) @binding(0) var<storage, read_write> state: array<u32>;
@compute @workgroup_size(1) fn main() { state[1] = state[1] + 1u; }`,
          shaderLayout: {bindings: [{name: 'state', type: 'storage', group: 0, location: 0}]}
        });
        return {
          encode: ({computePass, getBuffer}) =>
            kernel.dispatch(computePass, {
              bindings: {state: getViewBinding(state, getBuffer)},
              x: 1
            }),
          destroy: () => kernel.destroy()
        };
      }
    });
  }
  return {graph, state};
}

for (const at of ['publication', 'step', 'end'] as const) {
  test(`captures a retained iterative state at ${at} boundaries`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const {graph, state} = createChain(device);
    const compiled = graph.compile();
    try {
      const record = createGPUComputationRecord({
        id: 'chain',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-core'],
        graphs: [compiled.describe()]
      });
      expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
      const result = await captureGPUCommandGraph(compiled, {
        encodeOptions: {parameters: undefined},
        views: {state},
        at,
        runId: record.runId
      });
      expect(result.stepCount).toBe(3);
      expect(result.snapshotCount).toBe(at === 'end' ? 1 : 3);
      expect(result.captures.map(capture => Array.from(capture.values))).toEqual(
        at === 'end' ? [[3]] : [[1], [2], [3]]
      );
      expect(result.captures.map(capture => capture.afterNode)).toEqual(
        at === 'end' ? ['iteration-3'] : ['iteration-1', 'iteration-2', 'iteration-3']
      );
      expect(
        result.captures.every(
          capture => capture.elementType === 'u32' && capture.resource === 'state'
        )
      ).toBe(true);
      const captures = createGPUComputationCaptures({
        runId: record.runId,
        snapshots: result.captures
      });
      expect(validateGPUComputationSidecars(record, {captures})).toMatchObject({valid: true});
    } finally {
      compiled.destroy();
    }
  });
}

test('observation tap preserves iteration two while later iterations overwrite state', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {graph, state} = createChain(device);
  const sinkBuffer = device.createBuffer({byteLength: 4, usage: Buffer.COPY_DST | Buffer.COPY_SRC});
  const sink = graph.importBuffer(
    {id: 'tap-sink', byteLength: 4, usage: sinkBuffer.usage},
    sinkBuffer
  );
  graph.addObservationTap(state, {afterNode: 'iteration-2', sink, id: 'observe-two'});
  const compiled = graph.compile();
  try {
    const result = await captureGPUCommandGraph(compiled, {
      encodeOptions: {parameters: undefined},
      views: {state},
      at: 'publication'
    });
    expect(result.captures.map(capture => capture.values[0])).toEqual([1, 2, 3]);
    const bytes = await sinkBuffer.readAsync();
    expect(new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0]).toBe(2);
  } finally {
    compiled.destroy();
    sinkBuffer.destroy();
  }
});
