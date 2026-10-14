// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUMultipleTestingAdjustment,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS
} from '../../../src/gpu-spatial-analysis/multiple-testing';
import type {GPUMultipleTestingMethod} from '../../../src/analysis-core/contracts';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

const METHODS: GPUMultipleTestingMethod[] = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
];

type Bufferish = {byteLength?: number; length?: number; rowByteLength?: number};

function getResourceBytes(buffer: Bufferish): number {
  return buffer.byteLength ?? (buffer.length ?? 0) * (buffer.rowByteLength ?? 4);
}

/** Largest bound or transient resource of the nodes, in bytes. */
function getLargestResourceBytes(nodes: readonly unknown[]): number {
  let largest = 0;
  for (const node of nodes as {resources?: {buffer: Bufferish}[]}[]) {
    for (const resource of node.resources ?? []) {
      largest = Math.max(largest, getResourceBytes(resource.buffer));
    }
  }
  return largest;
}

function expectKernelsWithinCoreLimits(nodes: readonly unknown[]): void {
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(0);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), `${id} private bytes`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
}

function build(method: GPUMultipleTestingMethod, rows: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `mt-${method}-${rows}-${format}-${length}`, format, length);
  const nodes = new GPUMultipleTestingAdjustment({
    pValues: view('float32', rows),
    method,
    minimumFamilySize: GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1,
    output: {
      adjustedPValues: createTransientView(graph, `mt-out-${method}-${rows}`, 'float32', rows),
      testedCount: view('uint32', 1)
    }
  }).getCommandNodes(graph);
  return {device, nodes};
}

for (const method of METHODS) {
  it(`GPUMultipleTestingAdjustment ${method} at maximal options fits the core limits`, () => {
    const {device, nodes} = build(method, 70000);
    expectKernelsWithinCoreLimits(nodes);
    device.destroy();
  });
}

it('GPUMultipleTestingAdjustment at the documented maximum row count binds within maxStorageBufferBindingSize', () => {
  // Rows are below 2^24, so one float32 view is at most 64 MiB: below the 128 MiB binding limit.
  const rows = GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1;
  for (const method of METHODS) {
    const {device, nodes} = build(method, rows);
    expect(getLargestResourceBytes(nodes), method).toBeGreaterThanOrEqual(rows * 4);
    expect(getLargestResourceBytes(nodes), method).toBeLessThanOrEqual(
      device.limits.maxStorageBufferBindingSize
    );
    device.destroy();
  }
});
