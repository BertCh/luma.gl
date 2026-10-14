// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphBufferHandle,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUNeighborhoodSummary,
  GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS,
  type GPUNeighborhoodSummaryStatistic
} from '../../../src/gpu-spatial-analysis/neighborhood-summary';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

type ResourceNode = {resources?: {buffer?: GraphDataView | GraphBufferHandle}[]};

function getBufferBytes(buffer: GraphDataView | GraphBufferHandle): number {
  return 'rowByteLength' in buffer
    ? Math.max(buffer.length, 1) * buffer.rowByteLength
    : buffer.byteLength;
}

function expectKernelsWithinCoreLimits(nodes: readonly unknown[]): void {
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels.length).toBeGreaterThan(0);
  for (const kernel of kernels) {
    expect(getStorageBindingCount(kernel.source), `${kernel.id} storage`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(kernel.source), `${kernel.id} private`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(kernel.source), `${kernel.id} workgroup`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
}

function expectBindingsWithinLimit(nodes: readonly unknown[], limit: number): void {
  let checked = 0;
  for (const node of nodes as ResourceNode[]) {
    for (const use of node.resources ?? []) {
      if (use.buffer) {
        checked++;
        expect(getBufferBytes(use.buffer)).toBeLessThanOrEqual(limit);
      }
    }
  }
  expect(checked).toBeGreaterThan(0);
}

const ALL_STATISTICS: GPUNeighborhoodSummaryStatistic[] = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
];

function build(rows: number, slots: number, maximumNeighbors: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'neighborhood-core-limits'});
  const view = <Format extends 'uint32' | 'float32'>(
    format: Format,
    length: number,
    name: string
  ) => createTransientView(graph, name, format, length);
  const weights = {
    offsets: view('uint32', rows + 1, 'offsets'),
    neighbors: view('uint32', slots, 'neighbors'),
    weights: view('float32', slots, 'weights')
  };
  const contributor = new GPUNeighborhoodSummary({
    weights,
    values: view('float32', rows, 'values'),
    categories: view('uint32', rows, 'categories'),
    mask: view('uint32', rows, 'mask'),
    includeFocal: true,
    focalWeight: 0.5,
    statistics: ALL_STATISTICS,
    maximumNeighbors,
    output: {
      table: view('float32', rows * ALL_STATISTICS.length, 'output'),
      modes: view('uint32', rows, 'modes'),
      entropy: view('float32', rows, 'entropy'),
      status: {overflow: view('uint32', 1, 'overflow')}
    }
  });
  return {device, contributor, graph};
}

describe('GPUNeighborhoodSummary at the core device limits', () => {
  for (const maximumNeighbors of [1, 32, GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS]) {
    it(`stays within storage, private and workgroup limits at maximumNeighbors ${maximumNeighbors}`, () => {
      const {contributor, graph} = build(100, 800, maximumNeighbors);
      const nodes = contributor.getCommandNodes(graph);
      expectKernelsWithinCoreLimits(nodes);
    });
  }

  it('keeps every bound view within maxStorageBufferBindingSize at a large row count', () => {
    const {device, contributor, graph} = build(1_000_000, 8_000_000, 64);
    const nodes = contributor.getCommandNodes(graph);
    expectBindingsWithinLimit(nodes, device.limits.maxStorageBufferBindingSize);
  });
});
