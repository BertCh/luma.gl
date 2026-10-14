// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
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

const METHODS: GPUMultipleTestingMethod[] = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
];

type Variant = {
  gridWidth: number;
  gridHeight: number;
  sliceCount: number;
  multipleTesting: GPUMultipleTestingMethod;
  finalSlice: 'require' | 'last-valid';
  valueFormat: 'float32' | 'uint32';
  weightsMode?: boolean;
  maximumRadius?: number;
  withoutPValues?: boolean;
};

function build(variant: Variant) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `emerging-limits-${serial++}`, format, length);
  const cells = variant.gridWidth * variant.gridHeight;
  const bins = cells * variant.sliceCount;
  const spatial = variant.weightsMode
    ? {
        weights: {
          offsets: view('uint32', cells + 1),
          neighbors: view('uint32', cells * 8),
          weights: view('float32', cells * 8)
        },
        selfWeight: 1
      }
    : {gridWidth: variant.gridWidth, gridHeight: variant.gridHeight};
  const nodes = new GPUEmergingHotSpots({
    values: variant.valueFormat === 'float32' ? view('float32', bins) : view('uint32', bins),
    ...spatial,
    sliceCount: variant.sliceCount,
    maximumRadius: variant.maximumRadius,
    mask: view('uint32', cells),
    parameters: view('float32', GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH),
    multipleTesting: variant.multipleTesting,
    finalSlice: variant.finalSlice,
    output: {
      giZScores: view('float32', bins),
      pValues: variant.withoutPValues ? undefined : view('float32', bins),
      adjustedPValues: variant.withoutPValues ? undefined : view('float32', bins),
      lastValidSlice: view('uint32', cells),
      trendZ: view('float32', cells),
      trendP: view('float32', cells),
      trendS: view('sint32', cells),
      category: view('uint32', cells),
      hotSliceCount: view('uint32', cells),
      coldSliceCount: view('uint32', cells),
      globalStatistics: view('float32', GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH)
    }
  }).getCommandNodes(graph);
  return {device, nodes, bins};
}

for (const multipleTesting of METHODS) {
  for (const finalSlice of ['require', 'last-valid'] as const) {
    it(`GPUEmergingHotSpots lattice ${multipleTesting}/${finalSlice} at maximal slices and radius fits the core limits`, () => {
      const {device, nodes} = build({
        gridWidth: 8,
        gridHeight: 8,
        sliceCount: GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
        maximumRadius: GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
        multipleTesting,
        finalSlice,
        valueFormat: 'float32'
      });
      expectKernelsWithinCoreLimits(nodes);
      device.destroy();
    });
  }
}

it('GPUEmergingHotSpots lattice with a uint32 cube fits the core limits', () => {
  const {device, nodes} = build({
    gridWidth: 8,
    gridHeight: 8,
    sliceCount: GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
    maximumRadius: GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
    multipleTesting: 'benjamini-yekutieli',
    finalSlice: 'last-valid',
    valueFormat: 'uint32'
  });
  expectKernelsWithinCoreLimits(nodes);
  device.destroy();
});

for (const multipleTesting of METHODS) {
  it(`GPUEmergingHotSpots weights mode ${multipleTesting} at maximal slices fits the core limits`, () => {
    const {device, nodes} = build({
      gridWidth: 64,
      gridHeight: 1,
      sliceCount: GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
      multipleTesting,
      finalSlice: 'last-valid',
      valueFormat: 'float32',
      weightsMode: true
    });
    expectKernelsWithinCoreLimits(nodes);
    device.destroy();
  });
}

for (const multipleTesting of METHODS) {
  it(`GPUEmergingHotSpots ${multipleTesting} just below 2^24 bins binds within maxStorageBufferBindingSize`, () => {
    // 2048 x 2047 cells x 4 slices = 16,769,024 bins, the largest family a correction accepts.
    const {device, nodes, bins} = build({
      gridWidth: 2048,
      gridHeight: 2047,
      sliceCount: 4,
      multipleTesting,
      finalSlice: 'require',
      valueFormat: 'float32'
    });
    expect(bins).toBeLessThan(2 ** 24);
    expect(getLargestResourceBytes(nodes)).toBeGreaterThanOrEqual(bins * 4);
    expect(getLargestResourceBytes(nodes)).toBeLessThanOrEqual(
      device.limits.maxStorageBufferBindingSize
    );
    device.destroy();
  });
}

it('GPUEmergingHotSpots none at 2^25 bins (the binding limit of float32) binds within the limit', () => {
  // `none` without correction accepts up to 2^31 - 1 bins; one float32 binding holds 2^25.
  const {device, nodes, bins} = build({
    gridWidth: 2048,
    gridHeight: 2048,
    sliceCount: 8,
    multipleTesting: 'none',
    finalSlice: 'require',
    valueFormat: 'float32',
    withoutPValues: true
  });
  expect(bins).toBe(2 ** 25);
  expect(getLargestResourceBytes(nodes)).toBeLessThanOrEqual(
    device.limits.maxStorageBufferBindingSize
  );
  device.destroy();
});
