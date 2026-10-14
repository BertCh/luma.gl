// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphBufferHandle,
  GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUSpatialScanStatistic,
  type GPUSpatialScanModel
} from '../../../src/gpu-spatial-analysis/scan-statistics/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

const MAXIMUM_BINDING_BYTES = 134217728;

type Formats = 'float32' | 'uint32' | 'float32x2';

function getBindingBytes(buffer: GraphDataView | GraphBufferHandle): number {
  return buffer instanceof GraphDataView
    ? buffer.byteOffset + Math.max(buffer.length - 1, 0) * buffer.byteStride + buffer.rowByteLength
    : buffer.byteLength;
}

function buildScan(
  model: GPUSpatialScanModel,
  options: {
    zones: number;
    timeBuckets: number;
    maximumPermutations: number;
    maximumClusters: number;
    maximumWindowZones?: number;
  }
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `scan-limits-${model}`});
  const transients: GraphBufferHandle[] = [];
  const createTransientBuffer = graph.createTransientBuffer.bind(graph);
  graph.createTransientBuffer = descriptor => {
    const handle = createTransientBuffer(descriptor);
    transients.push(handle);
    return handle;
  };
  const {zones, timeBuckets, maximumPermutations, maximumClusters} = options;
  const cells = zones * timeBuckets;
  const view = <Format extends Formats>(name: string, format: Format, length: number) =>
    createTransientView(graph, `scan-${name}`, format, length);
  const statistic = new GPUSpatialScanStatistic({
    positions: view('positions', 'float32x2', zones),
    model,
    cases: view('cases', 'uint32', cells),
    ...(model === 'poisson' ? {baseline: view('baseline', 'float32', cells)} : {}),
    ...(model === 'bernoulli' ? {controls: view('controls', 'uint32', cells)} : {}),
    timeBuckets,
    maximumWindowZones: options.maximumWindowZones,
    maximumPermutations,
    maximumClusters,
    parameters: view('parameters', 'uint32', 8),
    output: {
      clusterIndices: view('cluster-indices', 'uint32', maximumClusters * 4),
      clusterStatistics: view('cluster-statistics', 'float32', maximumClusters * 8),
      statistics: view('statistics', 'float32', maximumPermutations + 1),
      summary: view('summary', 'uint32', 4),
      zoneStatistics: view('zone-statistics', 'float32', zones)
    }
  });
  const nodes = statistic.getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  const boundBytes = nodes.flatMap(node =>
    (node.resources ?? []).flatMap(resource =>
      'buffer' in resource
        ? [{node: node.id, bytes: getBindingBytes(resource.buffer as GraphDataView)}]
        : []
    )
  );
  device.destroy();
  return {statistic, nodes, sources, transients, boundBytes};
}

function expectKernelsWithinCoreLimits(sources: {id: string; source: string}[]) {
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

const MODELS: GPUSpatialScanModel[] = ['poisson', 'bernoulli', 'space-time-permutation'];

describe('GPUSpatialScanStatistic core limits', () => {
  it.each(MODELS)('%s at maximal options fits the core kernel limits', model => {
    const {nodes, sources} = buildScan(model, {
      zones: 40,
      timeBuckets: 32,
      maximumPermutations: 999,
      maximumClusters: 64,
      maximumWindowZones: 40
    });
    expect(nodes.length).toBeGreaterThan(0);
    expectKernelsWithinCoreLimits(sources);
  });

  it.each(MODELS)('%s at the documented count-table maximum binds <= 128 MiB', model => {
    // (batchReplicates + 1) * zones * timeBuckets == 2^25 words, with more replicates than one
    // batch (the old single-pass cap of (P + 1) * cells <= 2^25 allowed only 2047 here).
    const {statistic, sources, transients, boundBytes} = buildScan(model, {
      zones: 512,
      timeBuckets: 32,
      maximumPermutations: 20000,
      maximumClusters: 64,
      maximumWindowZones: 32
    });
    expect(statistic.batchReplicates).toBe(2047);
    expect(statistic.batchCount).toBe(10);
    expectKernelsWithinCoreLimits(sources);
    expect(boundBytes.length).toBeGreaterThan(0);
    // The count table itself is 2^25 words: exactly the binding limit.
    expect(Math.max(...transients.map(handle => handle.byteLength))).toBeGreaterThanOrEqual(
      2 ** 27
    );
    for (const handle of transients) {
      expect(handle.byteLength, handle.id).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
    }
    for (const {node, bytes} of boundBytes) {
      expect(bytes, node).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
    }
  });

  it.each(MODELS)('%s at 99999 replicates fits the core kernel limits', model => {
    const {statistic, sources, transients} = buildScan(model, {
      zones: 40,
      timeBuckets: 32,
      maximumPermutations: 99999,
      maximumClusters: 64,
      maximumWindowZones: 40
    });
    expect(statistic.batchCount).toBe(25);
    expectKernelsWithinCoreLimits(sources);
    for (const handle of transients) {
      expect(handle.byteLength, handle.id).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
    }
  });

  it.each(MODELS)('%s at the documented neighbor-table maximum binds <= 128 MiB', model => {
    // zones * zones == 2^24 words per neighbor table.
    const {sources, transients, boundBytes} = buildScan(model, {
      zones: 4096,
      timeBuckets: 1,
      maximumPermutations: 2047,
      maximumClusters: 64
    });
    expectKernelsWithinCoreLimits(sources);
    for (const handle of transients) {
      expect(handle.byteLength, handle.id).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
    }
    for (const {node, bytes} of boundBytes) {
      expect(bytes, node).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
    }
  });
});
