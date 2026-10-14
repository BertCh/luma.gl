// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUGridIndexView,
  type GraphBufferHandle,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUPickRegionMask,
  GPURegionMask,
  GPURegionStatistics,
  type GPURegionSelection,
  type GPURegionStatisticsProps
} from '../../../src/gpu-spatial-analysis/region-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

type ResourceNode = {id: string; resources?: {buffer?: GraphDataView | GraphBufferHandle}[]};

function getBufferBytes(buffer: GraphDataView | GraphBufferHandle): number {
  return 'rowByteLength' in buffer
    ? Math.max(buffer.length, 1) * buffer.rowByteLength
    : buffer.byteLength;
}

/** Asserts storage-binding count, private and workgroup memory of every generated kernel. */
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

/** Asserts every transient and bound view of the nodes fits one storage binding. */
function expectBindingsWithinLimit(nodes: readonly unknown[], limit: number): void {
  let checked = 0;
  for (const node of nodes as ResourceNode[]) {
    for (const use of node.resources ?? []) {
      if (use.buffer) {
        checked++;
        expect(getBufferBytes(use.buffer), `${node.id} binding`).toBeLessThanOrEqual(limit);
      }
    }
  }
  expect(checked).toBeGreaterThan(0);
}

const ROWS = 4096;
const POLYGON_CAPACITY = 1024;

function createGraph() {
  const device = createNullWebGPUDevice();
  return {device, graph: new GPUCommandGraph(device, {id: 'region-core-limits'})};
}

function createIndex(graph: GPUCommandGraph, rows: number): GPUGridIndexView {
  return {
    gridSize: [64, 64],
    bounds: [0, 0, 10, 10],
    cellOffsets: createTransientView(graph, 'cell-offsets', 'uint32', 64 * 64 + 1),
    objectIds: createTransientView(graph, 'object-ids', 'uint32', rows),
    count: createTransientView(graph, 'index-count', 'uint32', 1),
    overflow: createTransientView(graph, 'index-overflow', 'uint32', 1)
  };
}

function createSelection(
  graph: GPUCommandGraph,
  kind: GPURegionSelection['kind'],
  screen: boolean
): GPURegionSelection {
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number,
    name: string
  ) => createTransientView(graph, name, format, length);
  const screenTransform = screen ? view('float32', 20, 'transform') : undefined;
  switch (kind) {
    case 'rectangle':
      return {kind, bounds: view('float32', 4, 'bounds'), screenTransform};
    case 'polygon':
      return {
        kind,
        vertices: view('float32x2', POLYGON_CAPACITY, 'vertices'),
        vertexCount: view('uint32', 1, 'vertex-count'),
        screenTransform
      };
    case 'radius':
      return {kind, circle: view('float32', 3, 'circle')};
    case 'pick-region':
      return {kind, picks: view('uint32', 2 + 2 * ROWS, 'pick-result'), batchIndex: 3};
    case 'mask':
      return {kind, mask: view('uint32', ROWS, 'selection-mask')};
  }
}

function createProps(
  graph: GPUCommandGraph,
  rows: number,
  selection: GPURegionSelection,
  overrides: Partial<GPURegionStatisticsProps> = {}
): GPURegionStatisticsProps {
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number,
    name: string
  ) => createTransientView(graph, name, format, length);
  const binCount = 65536;
  const needsPositions = selection.kind !== 'pick-region' && selection.kind !== 'mask';
  return {
    region: selection,
    positions: needsPositions ? view('float32x2', rows, 'positions') : undefined,
    values: view('float32', rows, 'values'),
    sourceIds: view('uint32', rows, 'source-ids'),
    histogram: {binCount, domain: view('float32', 2, 'domain')},
    output: {
      summary: view('uint32', 8 + binCount, 'summary'),
      mask: selection.kind === 'mask' ? undefined : view('uint32', rows, 'output-mask'),
      selectedRows: {
        ids: view('uint32', rows, 'ids'),
        status: {count: view('uint32', 1, 'count'), overflow: view('uint32', 1, 'overflow')}
      },
      drawInstanceCount: view('uint32', 1, 'draw-count')
    },
    ...overrides
  } as GPURegionStatisticsProps;
}

describe('GPURegionStatistics at the core device limits', () => {
  const variants: {name: string; kind: GPURegionSelection['kind']; screen: boolean}[] = [
    {name: 'rectangle', kind: 'rectangle', screen: false},
    {name: 'screen rectangle', kind: 'rectangle', screen: true},
    {name: 'polygon', kind: 'polygon', screen: false},
    {name: 'screen polygon', kind: 'polygon', screen: true},
    {name: 'radius', kind: 'radius', screen: false},
    {name: 'pick-region', kind: 'pick-region', screen: false},
    {name: 'mask', kind: 'mask', screen: false}
  ];
  for (const {name, kind, screen} of variants) {
    it(`${name} selection with every output and a 65536-bin histogram`, () => {
      const {graph} = createGraph();
      const props = createProps(graph, ROWS, createSelection(graph, kind, screen));
      const nodes = new GPURegionStatistics(props).getCommandNodes(graph);
      expectKernelsWithinCoreLimits(nodes);
    });
  }

  for (const kind of ['rectangle', 'polygon', 'radius'] as const) {
    it(`${kind} selection through the grid spatialIndex`, () => {
      const {graph} = createGraph();
      const selection = createSelection(graph, kind, false);
      const props = createProps(graph, ROWS, selection, {
        spatialIndex: {kind: 'grid', index: createIndex(graph, ROWS), candidateCapacity: ROWS}
      });
      const nodes = new GPURegionStatistics(props).getCommandNodes(graph);
      expectKernelsWithinCoreLimits(nodes);
    });
  }

  it('keeps every bound view within maxStorageBufferBindingSize at two million rows', () => {
    const {device, graph} = createGraph();
    const rows = 2_000_000;
    const props = createProps(graph, rows, createSelection(graph, 'rectangle', false));
    const nodes = new GPURegionStatistics(props).getCommandNodes(graph);
    expectBindingsWithinLimit(nodes, device.limits.maxStorageBufferBindingSize);
  });
});

describe('GPURegionMask and GPUPickRegionMask at the core device limits', () => {
  for (const kind of ['rectangle', 'polygon'] as const) {
    for (const screen of [false, true]) {
      it(`GPURegionMask ${kind}${screen ? ' in screen space' : ''}`, () => {
        const {graph} = createGraph();
        const region = createSelection(graph, kind, screen) as GPURegionSelection & {
          kind: 'rectangle' | 'polygon';
        };
        const nodes = new GPURegionMask({
          positions: createTransientView(graph, 'positions', 'float32x2', ROWS),
          region,
          output: {
            mask: createTransientView(graph, 'mask', 'uint32', ROWS),
            status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
          }
        }).getCommandNodes(graph);
        expectKernelsWithinCoreLimits(nodes);
      });
    }
  }

  it('GPUPickRegionMask with a batch filter', () => {
    const {graph} = createGraph();
    const nodes = new GPUPickRegionMask({
      picks: createTransientView(graph, 'result', 'uint32', 2 + 2 * ROWS),
      batchIndex: 0x7fffffff,
      output: {
        mask: createTransientView(graph, 'mask', 'uint32', ROWS),
        status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
      }
    }).getCommandNodes(graph);
    expectKernelsWithinCoreLimits(nodes);
  });
});
