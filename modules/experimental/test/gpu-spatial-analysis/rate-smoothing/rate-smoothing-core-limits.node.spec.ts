// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUEmpiricalBayesRates,
  GPUSpatialEmpiricalBayesRates,
  GPU_EMPIRICAL_BAYES_SUMMARY
} from '../../../src/gpu-spatial-analysis/rate-smoothing';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

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

const ROWS = 300;
const SLOTS = 6 * ROWS;

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `rate-smoothing-${serial++}`, format, length);
  return {device, graph, view};
}

for (const clampPriorVariance of [false, true]) {
  it(`GPUEmpiricalBayesRates with every output (clamp ${clampPriorVariance}) fits the core limits`, () => {
    const {device, graph, view} = createContext();
    const nodes = new GPUEmpiricalBayesRates({
      events: view('float32', ROWS),
      populations: view('float32', ROWS),
      mask: view('uint32', ROWS),
      output: {
        standardizedRates: view('float32', ROWS),
        smoothedRates: view('float32', ROWS),
        rawRates: view('float32', ROWS),
        summary: view('float32', GPU_EMPIRICAL_BAYES_SUMMARY.length)
      },
      clampPriorVariance
    }).getCommandNodes(graph);
    expectKernelsWithinCoreLimits(nodes);
    device.destroy();
  });
}

it('GPUSpatialEmpiricalBayesRates with every output fits the core limits', () => {
  const {device, graph, view} = createContext();
  const nodes = new GPUSpatialEmpiricalBayesRates({
    events: view('float32', ROWS),
    populations: view('float32', ROWS),
    weights: {
      offsets: view('uint32', ROWS + 1),
      neighbors: view('uint32', SLOTS),
      weights: view('float32', SLOTS)
    },
    mask: view('uint32', ROWS),
    output: {
      spatialRates: view('float32', ROWS),
      smoothedRates: view('float32', ROWS),
      priorMeans: view('float32', ROWS),
      priorVariances: view('float32', ROWS)
    }
  }).getCommandNodes(graph);
  expectKernelsWithinCoreLimits(nodes);
  device.destroy();
});

it('rate smoothing at large row counts binds within maxStorageBufferBindingSize', () => {
  // Documented sizes: float32 per-row views only; 2^24 rows is 64 MiB.
  const rows = 1 << 24;
  const {device, graph, view} = createContext();
  const nodes = new GPUEmpiricalBayesRates({
    events: view('float32', rows),
    populations: view('float32', rows),
    output: {
      smoothedRates: view('float32', rows),
      summary: view('float32', GPU_EMPIRICAL_BAYES_SUMMARY.length)
    }
  }).getCommandNodes(graph);
  expect(getLargestResourceBytes(nodes)).toBeGreaterThanOrEqual(rows * 4);
  expect(getLargestResourceBytes(nodes)).toBeLessThanOrEqual(
    device.limits.maxStorageBufferBindingSize
  );
  device.destroy();
});
