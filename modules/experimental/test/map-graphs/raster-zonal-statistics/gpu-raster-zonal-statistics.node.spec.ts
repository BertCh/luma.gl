// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_RASTER_ZONAL_STATISTICS_NO_ZONE,
  GPURasterZonalStatistics,
  type GPURasterZonalStatisticsProps
} from '../../../src/map-graphs/raster-zonal-statistics';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

const WIDTH = 4;
const HEIGHT = 3;
const CAPACITY = 5;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPURasterZonalStatisticsProps> = {}
): GPURasterZonalStatisticsProps {
  return {
    id: 'zonal',
    width: WIDTH,
    height: HEIGHT,
    zones: createTransientView(graph, 'zones', 'uint32', WIDTH * HEIGHT),
    values: {
      id: 'values',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: createTransientView(graph, 'values', 'float32', WIDTH * HEIGHT)
      }
    },
    zoneCapacity: CAPACITY,
    output: {means: createTransientView(graph, 'means', 'float32', CAPACITY)},
    ...overrides
  };
}

function getNodeIds(props: GPURasterZonalStatisticsProps, graph: GPUCommandGraph): string[] {
  return new GPURasterZonalStatistics(props).getCommandNodes(graph).map(node => node.id);
}

it('GPURasterZonalStatistics prefixes IDs and schedules only requested columns', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(GPU_RASTER_ZONAL_STATISTICS_NO_ZONE).toBe(0xffffffff);
  const meanIds = getNodeIds(createProps(graph), graph);
  expect(meanIds.every(id => id.startsWith('zonal-'))).toBe(true);
  expect(meanIds).toContain('zonal-mask');
  expect(meanIds.some(id => id.startsWith('zonal-means'))).toBe(true);
  expect(meanIds.some(id => id.startsWith('zonal-overflow'))).toBe(false);

  // cellCounts alone needs no band canonicalization and no value mask.
  const countGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const countIds = getNodeIds(
    createProps(countGraph, {
      output: {
        cellCounts: createTransientView(countGraph, 'cell-counts', 'uint32', CAPACITY)
      }
    }),
    countGraph
  );
  expect(countIds.every(id => id.startsWith('zonal-'))).toBe(true);
  expect(countIds.some(id => id.includes('values-elevation'))).toBe(false);
  expect(countIds.some(id => id.startsWith('zonal-means'))).toBe(false);
  expect(countIds.length).toBeLessThan(meanIds.length);

  const fullGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const fullIds = getNodeIds(
    createProps(fullGraph, {
      output: {
        cellCounts: createTransientView(fullGraph, 'o0', 'uint32', CAPACITY),
        valueCounts: createTransientView(fullGraph, 'o1', 'uint32', CAPACITY),
        sums: createTransientView(fullGraph, 'o2', 'float32', CAPACITY),
        means: createTransientView(fullGraph, 'o3', 'float32', CAPACITY),
        minimums: createTransientView(fullGraph, 'o4', 'float32', CAPACITY),
        maximums: createTransientView(fullGraph, 'o5', 'float32', CAPACITY)
      },
      overflow: createTransientView(fullGraph, 'overflow', 'uint32', 1)
    }),
    fullGraph
  );
  expect(fullIds).toContain('zonal-overflow-reset');
  for (const name of ['cellCounts', 'valueCounts', 'sums', 'means', 'minimums', 'maximums']) {
    expect(fullIds.some(id => id.startsWith(`zonal-${name}`))).toBe(true);
  }
});

function expectConstructorToThrow(
  createOverrides: (graph: GPUCommandGraph) => Partial<GPURasterZonalStatisticsProps>,
  pattern?: RegExp
): void {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPURasterZonalStatistics(createProps(graph, createOverrides(graph)))).toThrow(
    pattern
  );
}

it('GPURasterZonalStatistics validates its props', () => {
  expectConstructorToThrow(() => ({width: 0}), /dimensions/);
  expectConstructorToThrow(
    graph => ({
      zones: createTransientView(graph, 'short-zones', 'uint32', 3)
    }),
    /zones/
  );
  expectConstructorToThrow(() => ({output: {}}), /at least one output/);
  expectConstructorToThrow(() => ({zoneCapacity: 0}), /zoneCapacity/);
  expectConstructorToThrow(() => ({zoneCapacity: 2 ** 32}), /zoneCapacity/);
  expectConstructorToThrow(() => ({ignoredZone: -1}), /ignoredZone/);
  expectConstructorToThrow(() => ({ignoredZone: 1.5}), /ignoredZone/);
  expectConstructorToThrow(
    graph => ({
      output: {
        means: createTransientView(graph, 'wrong-length', 'float32', CAPACITY + 1)
      }
    }),
    /zoneCapacity rows/
  );
  expectConstructorToThrow(graph => ({
    output: {
      means: createTransientView(graph, 'wrong-format', 'uint32', CAPACITY) as never
    }
  }));
  expectConstructorToThrow(
    graph => ({
      output: {sums: createTransientView(graph, 'sums', 'float32', CAPACITY)},
      overflow: createTransientView(graph, 'overflow-two', 'uint32', 2)
    }),
    /overflow/
  );
});

it('GPURasterZonalStatistics rejects aliasing and foreign graphs', () => {
  expectConstructorToThrow(graph => {
    const shared = createTransientView(graph, 'shared', 'float32', CAPACITY);
    return {output: {sums: shared, means: shared}};
  }, /share buffers/);
  expectConstructorToThrow(graph => {
    const zones = createTransientView(graph, 'aliased-zones', 'uint32', WIDTH * HEIGHT);
    return {
      zones,
      zoneCapacity: WIDTH * HEIGHT,
      output: {cellCounts: zones}
    };
  }, /share buffers/);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() =>
    new GPURasterZonalStatistics(createProps(graph)).getCommandNodes(otherGraph)
  ).toThrow(/target graph/);
});

it('GPURasterZonalStatistics sumOrder defaults to atomic and sorted adds the sort nodes', () => {
  const outputs = (graph: GPUCommandGraph) => ({
    sums: createTransientView(graph, 'o-sums', 'float32', CAPACITY),
    means: createTransientView(graph, 'o-means', 'float32', CAPACITY)
  });
  const defaultGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const defaultIds = getNodeIds(
    createProps(defaultGraph, {output: outputs(defaultGraph)}),
    defaultGraph
  );
  const atomicGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const atomicIds = getNodeIds(
    createProps(atomicGraph, {
      output: outputs(atomicGraph),
      sumOrder: 'atomic'
    }),
    atomicGraph
  );
  expect(atomicIds).toEqual(defaultIds);
  expect(defaultIds.some(id => id.includes('-sorted-'))).toBe(false);

  const sortedGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const sortedIds = getNodeIds(
    createProps(sortedGraph, {
      output: outputs(sortedGraph),
      sumOrder: 'sorted'
    }),
    sortedGraph
  );
  expect(sortedIds.some(id => id.startsWith('zonal-sorted-sort'))).toBe(true);
  expect(sortedIds).toContain('zonal-sorted-gather-sums');
  expect(sortedIds).toContain('zonal-means');
  expect(sortedIds.length).toBeGreaterThan(defaultIds.length);

  // Minimum-only requests never need the sort even when sorted is requested.
  const minGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const minIds = getNodeIds(
    createProps(minGraph, {
      output: {
        minimums: createTransientView(minGraph, 'o-min', 'float32', CAPACITY)
      },
      sumOrder: 'sorted'
    }),
    minGraph
  );
  expect(minIds.some(id => id.includes('-sorted-'))).toBe(false);

  expectConstructorToThrow(
    () => ({sumOrder: 'fast' as never}),
    /sumOrder must be atomic or sorted/
  );
});
