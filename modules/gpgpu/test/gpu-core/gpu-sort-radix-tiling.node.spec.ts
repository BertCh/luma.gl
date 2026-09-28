// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  GPUBatchSort,
  GPUCommandGraph,
  GPUSort,
  GraphVectorView,
  getGPUSortRadixPlan,
  type GPUSortDigitBits,
  type GPUSortDirection
} from '@luma.gl/gpgpu/gpu-core';
import {NullDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';
import {getGPUSortCommandNodesWithDispatchLimit} from '../../src/gpu-core/gpu-sort';

const WORKGROUP_SIZE = 256;

it('getGPUSortRadixPlan tiles eight keys per thread by default', () => {
  const plan = getGPUSortRadixPlan({
    keys: {length: 1_700_000},
    keyBits: 16,
    digitBits: 4,
    elementsPerThread: 8
  });

  expect(plan.tileSize, 'one workgroup covers 256 threads x 8 keys').toBe(2048);
  expect(plan.workgroupCount, '1.7M keys dispatch 831 workgroups').toBe(831);
  expect(plan.histogramLength, 'the scanned histogram holds 16 buckets x 831 tiles').toBe(13_296);
  expect(plan.passCount, 'a 16-bit key needs four four-bit passes').toBe(4);
});

it('getGPUSortRadixPlan reports the single-element tiling it replaces', () => {
  const plan = getGPUSortRadixPlan({
    keys: {length: 1_700_000},
    keyBits: 16,
    digitBits: 4,
    elementsPerThread: 1
  });

  expect(plan.tileSize, 'one key per thread covers only the workgroup').toBe(WORKGROUP_SIZE);
  expect(plan.workgroupCount, 'the same rows need 6641 workgroups').toBe(6_641);
  expect(plan.histogramLength, 'and a 106256-entry histogram to scan').toBe(106_256);
});

it('getGPUSortRadixPlan halves the pass count for eight-bit digits', () => {
  const wide = getGPUSortRadixPlan({
    keys: {length: 1_000_000},
    keyBits: 32,
    digitBits: 8,
    elementsPerThread: 8
  });
  const narrow = getGPUSortRadixPlan({
    keys: {length: 1_000_000},
    keyBits: 32,
    digitBits: 4,
    elementsPerThread: 8
  });

  expect(wide.passCount, 'four eight-bit passes cover a full 32-bit key').toBe(4);
  expect(narrow.passCount, 'the same key needs eight four-bit passes').toBe(8);
  expect(wide.bucketCount, 'eight-bit digits scan 256 buckets').toBe(256);
  expect(
    wide.workgroupStorageBytes,
    'the 256-bucket ballot mask and cursors fit the guaranteed 16KB of workgroup storage'
  ).toBeLessThanOrEqual(16_384);
});

it('getGPUSortRadixPlan keeps at least one tile for an empty range', () => {
  const plan = getGPUSortRadixPlan({
    keys: {length: 0},
    keyBits: 16,
    digitBits: 4,
    elementsPerThread: 8
  });

  expect(plan.workgroupCount, 'an empty range still reports a single tile').toBe(1);
});

it('getGPUSortRadixPlan covers a partial final digit with one extra pass', () => {
  const plan = getGPUSortRadixPlan({
    keys: {length: 5000},
    keyBits: 12,
    digitBits: 8,
    elementsPerThread: 8
  });

  expect(plan.passCount, 'twelve bits need an eight-bit pass and a four-bit remainder').toBe(2);
  expect(plan.workgroupCount, '5000 keys fill three 2048-key tiles').toBe(3);
});

it('GPUSort radix emits eight-bit descending passes and a narrowed final digit', () => {
  const fixture = createRadixFixture(5000, {
    keyBits: 12,
    digitBits: 8,
    direction: 'descending'
  });
  try {
    const passes = compileRadixPasses(fixture, 65_535);
    const histograms = passes.filter(pass => pass.id.endsWith('-histogram'));

    expect(
      histograms.map(pass => pass.id),
      'one eight-bit pass at bit 0, then the four remaining bits at bit 8'
    ).toEqual(['tiled-sort-radix-digit-0-histogram', 'tiled-sort-radix-digit-8-histogram']);
    expect(histograms[0].source, 'the full digit scans 256 buckets').toContain(
      'const BUCKET_COUNT: u32 = 256u;'
    );
    expect(histograms[1].source, 'the partial final digit scans only 16').toContain(
      'const BUCKET_COUNT: u32 = 16u;'
    );
    expect(histograms[1].source, 'and masks only its four significant bits').toContain(
      'const DIGIT_MASK: u32 = 15u;'
    );
    for (const pass of passes.filter(pass => /-(histogram|scatter)$/.test(pass.id))) {
      expect(pass.source, `${pass.id} inverts buckets for descending order`).toContain(
        'DIGIT_MASK - digit'
      );
      expect(pass.source, `${pass.id} tiles eight keys per thread`).toContain(
        'const TILE_SIZE: u32 = 2048u;'
      );
    }
    expect(passes.at(-1)?.id, 'the last digit scatters straight into the output').toBe(
      'tiled-sort-radix-digit-8-scatter'
    );
  } finally {
    fixture.device.destroy();
  }
});

it('GPUSort radix plans its dispatch per tile, not per key, under a small dispatch limit', () => {
  // With at most two workgroups per dimension a 3D dispatch holds eight workgroups. One key per
  // thread would cap the sort at 8 * 256 = 2048 keys; eight keys per thread lifts it to 16384.
  const fitting = createRadixFixture(5000, {keyBits: 16});
  try {
    const passes = compileRadixPasses(fitting, 2);
    const scatter = passes.find(pass => pass.id === 'tiled-sort-radix-digit-0-scatter');
    expect(scatter?.source, 'three tiles fold into a 2 x 2 x 1 dispatch').toContain(
      'const WORKGROUP_COUNT: u32 = 3u;'
    );
    expect(scatter?.source, 'indexed across the second dimension').toContain(
      'workgroupId.z * 2u + workgroupId.y) * 2u + workgroupId.x'
    );
  } finally {
    fitting.device.destroy();
  }

  const largest = createRadixFixture(8 * 2048, {keyBits: 16});
  try {
    expect(
      () => compileRadixPasses(largest, 2),
      'eight full tiles exactly fill the bounded dispatch'
    ).not.toThrow();
  } finally {
    largest.device.destroy();
  }

  const overflowing = createRadixFixture(8 * 2048 + 1, {keyBits: 16});
  try {
    expect(() => compileRadixPasses(overflowing, 2), 'one key more needs a ninth tile').toThrow(
      /exceeding the 3D dispatch limit/
    );
  } finally {
    overflowing.device.destroy();
  }

  const untiled = createRadixFixture(8 * 256 + 1, {keyBits: 16, elementsPerThread: 1});
  try {
    expect(
      () => compileRadixPasses(untiled, 2),
      'one key per thread still hits the limit at 2049 keys'
    ).toThrow(/exceeding the 3D dispatch limit/);
  } finally {
    untiled.device.destroy();
  }
});

it('GPUBatchSort forwards radix tiling to every chunk sort', () => {
  const device = createWebGPUNullDevice();
  device.limits.maxComputeWorkgroupsPerDimension = 65_535;
  const graph = new GPUCommandGraph(device, {id: 'tiled-batch-sort-graph'});
  const lengths = [3000, 700];
  const sort = new GPUBatchSort({
    id: 'tiled-batch-sort',
    keys: createVector(graph, 'keys', lengths),
    values: createVector(graph, 'values', lengths),
    outputKeys: createVector(graph, 'output-keys', lengths),
    outputValues: createVector(graph, 'output-values', lengths),
    algorithm: 'radix',
    digitBits: 8,
    elementsPerThread: 4
  });
  const addComputePass = vi.spyOn(graph, 'addComputePass');
  try {
    graph.add(sort);
    const identifiers = addComputePass.mock.calls.map(([pass]) => pass.id);
    expect(
      identifiers.filter(identifier => identifier?.endsWith('-histogram')),
      'each chunk sorts a 32-bit key in four eight-bit passes'
    ).toHaveLength(8);
    expect(identifiers).toContain('tiled-batch-sort-chunk-0-radix-digit-24-scatter');
    expect(identifiers).toContain('tiled-batch-sort-chunk-1-radix-digit-24-scatter');
  } finally {
    addComputePass.mockRestore();
    device.destroy();
  }
});

type RadixFixture = {device: NullDevice; graph: GPUCommandGraph; sort: GPUSort};

function createWebGPUNullDevice(): NullDevice {
  const device = new NullDevice({id: 'tiled-sort-node-device'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  return device;
}

function createRadixFixture(
  length: number,
  options: {
    keyBits: number;
    digitBits?: GPUSortDigitBits;
    direction?: GPUSortDirection;
    elementsPerThread?: number;
  }
): RadixFixture {
  const device = createWebGPUNullDevice();
  const graph = new GPUCommandGraph(device, {id: 'tiled-sort-node-graph'});
  const sort = new GPUSort({
    id: 'tiled-sort',
    keys: createView(graph, 'keys', length),
    values: createView(graph, 'values', length),
    outputKeys: createView(graph, 'output-keys', length),
    outputValues: createView(graph, 'output-values', length),
    algorithm: 'radix',
    ...options
  });
  return {device, graph, sort};
}

/**
 * Plans a radix sort under a synthetic dispatch limit and captures each pass's WGSL.
 *
 * NullDevice cannot build compute pipelines, but it does see the shader source first, which is
 * all these tests inspect.
 */
function compileRadixPasses(
  fixture: RadixFixture,
  maxComputeWorkgroupsPerDimension: number
): {id: string; source: string}[] {
  const nodes = getGPUSortCommandNodesWithDispatchLimit(
    fixture.sort,
    fixture.graph,
    maxComputeWorkgroupsPerDimension
  );
  const createShader = vi.spyOn(fixture.device, 'createShader');
  try {
    return nodes.map(node => {
      createShader.mockClear();
      try {
        (node as unknown as {compile: (props: {device: NullDevice}) => unknown}).compile({
          device: fixture.device
        });
      } catch {
        // Expected: NullDevice rejects the compute pipeline after the shader is created.
      }
      return {id: node.id, source: String(createShader.mock.calls[0]?.[0]?.source ?? '')};
    });
  } finally {
    createShader.mockRestore();
  }
}

function createView(graph: GPUCommandGraph, id: string, length: number) {
  const buffer = graph.createTransientBuffer({
    id,
    byteLength: Math.max(length, 1) * Uint32Array.BYTES_PER_ELEMENT,
    usage: Buffer.STORAGE
  });
  return graph.createDataView(buffer, {format: 'uint32', length});
}

function createVector(
  graph: GPUCommandGraph,
  id: string,
  lengths: readonly number[]
): GraphVectorView<'uint32'> {
  const length = lengths.reduce((total, chunkLength) => total + chunkLength, 0);
  return new GraphVectorView({
    id,
    name: id,
    format: 'uint32',
    length,
    valueLength: length,
    stride: 1,
    byteStride: Uint32Array.BYTES_PER_ELEMENT,
    rowByteLength: Uint32Array.BYTES_PER_ELEMENT,
    data: lengths.map((chunkLength, chunkIndex) =>
      createView(graph, `${id}-chunk-${chunkIndex}`, chunkLength)
    )
  });
}
