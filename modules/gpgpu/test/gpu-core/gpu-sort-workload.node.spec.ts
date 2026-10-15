// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  GPUGridBinning,
  GPUGridIndex,
  GPUSort,
  GraphVectorView,
  createTransientView,
  type GPUCommandNode
} from '@luma.gl/gpgpu/gpu-core';
import {NullDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';

function createFixture(recording = true) {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 134217728,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeWorkgroupStorageSize: 16384
    }
  });
  return {device, graph: new GPUCommandGraph(device, recording ? {recording: {}} : {})};
}

function expectWorkloads(nodes: readonly GPUCommandNode[], operation: string) {
  expect(nodes.length).toBeGreaterThan(0);
  for (const node of nodes) {
    expect(node.workload?.operation, node.id).toBe(operation);
    expect(node.workload?.variant, node.id).toBeTruthy();
    expect(node.workload?.commandCount, node.id).toBe(1);
    expect(node.workload?.maximumWorkgroupCount, node.id).toBeGreaterThan(0);
    expect(node.workload?.maximumInvocationCount, node.id).toBeGreaterThan(0);
  }
}

test.each([
  [1, 'bitonic'],
  [4, 'bitonic'],
  [1025, 'bitonic'],
  [1025, 'radix']
] as const)('recorded sort workloads cover length %s with %s', (length, algorithm) => {
  const {device, graph} = createFixture();
  try {
    const sort = new GPUSort({
      keys: createTransientView(graph, 'keys', 'uint32', length),
      values: createTransientView(graph, 'values', 'uint32', length),
      outputKeys: createTransientView(graph, 'output-keys', 'uint32', length),
      outputValues: createTransientView(graph, 'output-values', 'uint32', length),
      algorithm
    });
    const nodes = sort.getCommandNodes(graph);
    expectWorkloads(nodes, 'GPUSort');
    if (length === 4) {
      expect(nodes[0].workload).toMatchObject({
        variant: 'bitonic-local',
        maximumInvocationCount: 4,
        readByteLength: 32,
        writeByteLength: 32
      });
    }
  } finally {
    device.destroy();
  }
});

test('recorded chunked sort workloads include merge stages', () => {
  const {device, graph} = createFixture();
  const vector = (id: string) =>
    new GraphVectorView({
      id,
      name: id,
      format: 'uint32',
      length: 4,
      valueLength: 4,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [
        createTransientView(graph, `${id}-0`, 'uint32', 2),
        createTransientView(graph, `${id}-1`, 'uint32', 2)
      ]
    });
  try {
    const nodes = new GPUSort({
      keys: vector('keys'),
      values: vector('values'),
      outputKeys: vector('output-keys'),
      outputValues: vector('output-values')
    }).getCommandNodes(graph);
    expectWorkloads(nodes, 'GPUSort');
    expect(nodes.some(node => node.workload?.variant === 'chunk-merge')).toBe(true);
  } finally {
    device.destroy();
  }
});

test.each([4, 400])('recorded grid workloads cover %s cells', cellCount => {
  const {device, graph} = createFixture();
  try {
    const positions = createTransientView(graph, 'positions', 'float32x2', 10);
    const gridSize = [cellCount, 1] as const;
    const bounds = [0, 0, 1, 1] as const;
    expectWorkloads(
      new GPUGridBinning({
        positions,
        gridSize,
        bounds,
        output: createTransientView(graph, 'bins', 'uint32', cellCount)
      }).getCommandNodes(graph),
      'GPUGridBinning'
    );
    expectWorkloads(
      new GPUGridIndex({
        positions,
        gridSize,
        bounds,
        cellOffsets: createTransientView(graph, 'offsets', 'uint32', cellCount + 1),
        objectIds: createTransientView(graph, 'ids', 'uint32', 10),
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }).getCommandNodes(graph),
      'GPUGridIndex'
    );
  } finally {
    device.destroy();
  }
});

test('production sort does not allocate recording workloads', () => {
  const {device, graph} = createFixture(false);
  try {
    const nodes = new GPUSort({
      keys: createTransientView(graph, 'keys', 'uint32', 4),
      values: createTransientView(graph, 'values', 'uint32', 4),
      outputKeys: createTransientView(graph, 'output-keys', 'uint32', 4),
      outputValues: createTransientView(graph, 'output-values', 'uint32', 4)
    }).getCommandNodes(graph);
    expect(nodes[0].workload).toBeUndefined();
  } finally {
    device.destroy();
  }
});
