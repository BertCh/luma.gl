// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  type GPUCommandNode,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import {describe, expect, it} from 'vitest';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  type GPUGlobalSpatialStatistic
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
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
let viewSerial = 0;

type InspectedGraph = {
  graph: GPUCommandGraph;
  transients: GraphBufferHandle[];
  view: <Format extends VertexFormat>(
    format: Format,
    length: number,
    name?: string
  ) => GraphDataView<Format>;
};

/** A graph on the core-limits device that records every transient buffer it creates. */
function createInspectedGraph(id: string): InspectedGraph {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id});
  const transients: GraphBufferHandle[] = [];
  const createTransientBuffer = graph.createTransientBuffer.bind(graph);
  graph.createTransientBuffer = descriptor => {
    const handle = createTransientBuffer(descriptor);
    transients.push(handle);
    return handle;
  };
  return {
    graph,
    transients,
    view: (format, length, name = 'v') =>
      createTransientView(graph, `${id}-${name}-${viewSerial++}`, format, length)
  };
}

function getBindingBytes(buffer: GraphDataView | GraphBufferHandle): number {
  return buffer instanceof GraphDataView
    ? buffer.byteOffset + Math.max(buffer.length - 1, 0) * buffer.byteStride + buffer.rowByteLength
    : buffer.byteLength;
}

/**
 * Asserts the three core kernel limits for every kernel of the nodes and that every transient and
 * every bound view fits one storage binding.
 */
function expectWithinCoreLimits(inspected: InspectedGraph, nodes: readonly GPUCommandNode[]) {
  expect(nodes.length).toBeGreaterThan(0);
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
  for (const handle of inspected.transients) {
    expect(handle.byteLength, `transient ${handle.id}`).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
  }
  for (const node of nodes) {
    for (const resource of node.resources ?? []) {
      if ('buffer' in resource) {
        expect(
          getBindingBytes(resource.buffer as GraphDataView),
          `${node.id} bound view`
        ).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
      }
    }
  }
}

const ALL_STATISTICS: GPUGlobalSpatialStatistic[] = [
  'moran',
  'geary',
  'getisOrdG',
  'bivariateMoran',
  'joinCount'
];
/** Rows at which the `27 * rows * 4` byte matrix transient equals the binding limit. */
const MAXIMUM_ROWS = Math.floor(MAXIMUM_BINDING_BYTES / (27 * 4));

function buildGlobalStatistics(
  rows: number,
  capacity: number,
  options: {islands?: 'keep' | 'exclude'; alternative?: 'greater' | 'lesser' | 'two-sided'} = {}
) {
  const inspected = createInspectedGraph('global-limits');
  const {view, graph} = inspected;
  const nodes = new GPUGlobalSpatialStatistics({
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', capacity),
      weights: view('float32', capacity),
      distances: view('float32', capacity),
      status: {overflow: view('uint32', 1)}
    },
    values: view('float32', rows),
    secondValues: view('float32', rows),
    mask: view('uint32', rows),
    statistics: ALL_STATISTICS,
    output: {
      summary: view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length),
      joinCounts: view('uint32', 3)
    },
    ...options
  }).getCommandNodes(graph);
  return {inspected, nodes};
}

describe('GPUGlobalSpatialStatistics core limits', () => {
  for (const islands of ['exclude', 'keep'] as const) {
    it(`all statistics and optional views at modest rows (islands ${islands})`, () => {
      const {inspected, nodes} = buildGlobalStatistics(500, 4000, {islands});
      expectWithinCoreLimits(inspected, nodes);
    });
  }

  it('the documented row maximum (matrix equals one binding) stays within the binding limit', () => {
    const {inspected, nodes} = buildGlobalStatistics(MAXIMUM_ROWS, 8 * MAXIMUM_ROWS);
    expectWithinCoreLimits(inspected, nodes);
  });

  it('a neighbor-slot capacity of one full binding (2^25 slots) stays within the binding limit', () => {
    const {inspected, nodes} = buildGlobalStatistics(100000, 2 ** 25);
    expectWithinCoreLimits(inspected, nodes);
  });

  it('one row above the maximum is rejected with an error, not a binding overflow', () => {
    expect(() => buildGlobalStatistics(MAXIMUM_ROWS + 1, 8 * MAXIMUM_ROWS)).toThrow();
  });
});
