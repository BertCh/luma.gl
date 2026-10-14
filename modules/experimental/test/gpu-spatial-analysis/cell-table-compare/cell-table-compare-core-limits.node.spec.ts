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
import type {GPUCellTable} from '../../../src/gpu-spatial-analysis/cell-aggregation/cell-table';
import {GPUCellTableCompare} from '../../../src/gpu-spatial-analysis/cell-table-compare/index';
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

function createTable({view}: InspectedGraph, capacity: number): GPUCellTable {
  return {
    cells: view('uint32x2', capacity),
    counts: view('uint32', capacity),
    sums: view('uint32x2', capacity),
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
}

function buildCompare(
  measure: 'count' | 'sum',
  zScore: 'poisson' | 'standardized',
  beforeCapacity: number,
  afterCapacity: number,
  outputCapacity: number
) {
  const inspected = createInspectedGraph('compare-limits');
  const {view, graph} = inspected;
  const nodes = new GPUCellTableCompare({
    before: createTable(inspected, beforeCapacity),
    after: createTable(inspected, afterCapacity),
    measure,
    zScore,
    sumScale: 1024,
    output: {
      cells: view('uint32x2', outputCapacity),
      presence: view('uint32', outputCapacity),
      mask: view('uint32', outputCapacity),
      before: view('float32', outputCapacity),
      after: view('float32', outputCapacity),
      delta: view('float32', outputCapacity),
      ratio: view('float32', outputCapacity),
      percentChange: view('float32', outputCapacity),
      zScore: view('float32', outputCapacity),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1)
      }
    }
  }).getCommandNodes(graph);
  return {inspected, nodes};
}

const VARIANTS = [
  ['count', 'poisson'],
  ['count', 'standardized'],
  ['sum', 'poisson'],
  ['sum', 'standardized']
] as const;

describe('GPUCellTableCompare core limits', () => {
  for (const [measure, zScore] of VARIANTS) {
    it(`${measure} measure with ${zScore} z-scores and every output`, () => {
      const {inspected, nodes} = buildCompare(measure, zScore, 1000, 1500, 2000);
      expectWithinCoreLimits(inspected, nodes);
    });

    it(`${measure} measure with ${zScore} z-scores at 2^24 cells (one full binding of uint32x2)`, () => {
      const capacity = 2 ** 24;
      const {inspected, nodes} = buildCompare(measure, zScore, capacity, capacity, capacity);
      expectWithinCoreLimits(inspected, nodes);
    });
  }
});
