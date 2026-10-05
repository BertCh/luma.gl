// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSpatialClusteringParameterValues,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_NOISE,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  type GPUSpatialClusteringProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {clusterPointsOracle, ORACLE_NOISE} from './spatial-clustering-oracle';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUSpatialClusteringProps> = {},
  rows: number = 16
): GPUSpatialClusteringProps {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    gridSize: [8, 8],
    labels: createTransientView(graph, 'labels', 'uint32', rows),
    ...overrides
  };
}

function createCluster(graph: GPUCommandGraph, capacity: number) {
  return {
    ids: createTransientView(graph, 'cluster-ids', 'uint32', capacity),
    count: createTransientView(graph, 'cluster-ids-count', 'uint32', 1),
    overflow: createTransientView(graph, 'cluster-ids-overflow', 'uint32', 1)
  };
}

it('getGPUSpatialClusteringParameterValues packs and validates', () => {
  expect(GPU_SPATIAL_CLUSTERING_NOISE).toBe(ORACLE_NOISE);
  const values = getGPUSpatialClusteringParameterValues({
    bounds: [1, 2, 3, 4],
    epsilon: 0.5,
    minimumPoints: 7
  });
  expect(Array.from(values)).toEqual([1, 2, 3, 4, 0.5, 7, 0, 0]);
  expect(values.length).toBe(GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH);
  const target = new Float32Array(10).fill(9);
  const returned = getGPUSpatialClusteringParameterValues(
    {bounds: [0, 0, 1, 1], epsilon: 1, minimumPoints: 1},
    target
  );
  expect(returned).toBe(target);
  expect(Array.from(target.slice(6))).toEqual([0, 0, 9, 9]);

  const valid = {bounds: [0, 0, 1, 1], epsilon: 1, minimumPoints: 2} as const;
  expect(() => getGPUSpatialClusteringParameterValues({...valid, epsilon: 0})).toThrow(/epsilon/);
  expect(() => getGPUSpatialClusteringParameterValues({...valid, epsilon: -1})).toThrow(/epsilon/);
  expect(() => getGPUSpatialClusteringParameterValues({...valid, epsilon: NaN})).toThrow(/finite/);
  expect(() =>
    getGPUSpatialClusteringParameterValues({
      ...valid,
      bounds: [0, 0, Infinity, 1]
    })
  ).toThrow(/finite/);
  for (const minimumPoints of [0, 1.5, -2, 2 ** 24 + 1, NaN]) {
    expect(() => getGPUSpatialClusteringParameterValues({...valid, minimumPoints})).toThrow(
      /minimumPoints/
    );
  }
  expect(() => getGPUSpatialClusteringParameterValues(valid, new Float32Array(7))).toThrow(
    /target/
  );
});

it('clusterPointsOracle labels a tiny fixture canonically', () => {
  // Row 0 is noise, rows 1-3 form the first cluster by root order, row 4 is a border point.
  const result = clusterPointsOracle(
    [9, 9, 0, 0, 1, 0, 0, 1, 2.2, 0],
    {bounds: [0, 0, 10, 10], epsilon: 1.5, minimumPoints: 3},
    [10, 11, 12, 13, 14]
  );
  expect(result.coreFlags).toEqual([0, 1, 1, 1, 0]);
  expect(result.labels).toEqual([ORACLE_NOISE, 0, 0, 0, 0]);
  expect(result.rootRows).toEqual([ORACLE_NOISE, 1, 1, 1, 1]);
  expect(result.clusterRoots).toEqual([11]);
  expect(result.clusterSizes).toEqual([4]);
});

it('GPUSpatialClustering schedules the label pipeline with prefixed unique IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUSpatialClustering(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids[0]).toBe('spatial-clustering-cell-keys');
  expect(ids.at(-1)).toBe('spatial-clustering-labels');
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids.every(id => id.startsWith('spatial-clustering-'))).toBe(true);
  expect(ids.some(id => id.endsWith('-publish'))).toBe(false);
  const prefixes = [
    'spatial-clustering-cell-keys',
    'spatial-clustering-cell-counts',
    'spatial-clustering-cell-starts',
    'spatial-clustering-sort',
    'spatial-clustering-core',
    'spatial-clustering-parents-init',
    'spatial-clustering-union',
    'spatial-clustering-resolve',
    'spatial-clustering-border',
    'spatial-clustering-root-flags',
    'spatial-clustering-cluster-offsets',
    'spatial-clustering-labels'
  ];
  let previous = -1;
  for (const prefix of prefixes) {
    const index = ids.findIndex(id => id === prefix || id.startsWith(`${prefix}-`));
    expect(index, prefix).toBeGreaterThan(previous);
    previous = index;
  }
  device.destroy();
});

it('GPUSpatialClustering orders cluster outputs and publishes last', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUSpatialClustering({
    ...createProps(graph, {}, 16),
    id: 'dbscan',
    clusters: createCluster(graph, 4),
    clusterSizes: createTransientView(graph, 'sizes', 'uint32', 4),
    clusterCentroids: createTransientView(graph, 'centroids', 'float32x2', 4)
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.every(id => id.startsWith('dbscan-'))).toBe(true);
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids.at(-1)).toBe('dbscan-publish');
  const order = [
    'labels',
    'cluster-roots',
    'cluster-sizes',
    'cluster-sums-x',
    'cluster-sums-y',
    'cluster-centroids',
    'publish'
  ];
  let previous = -1;
  for (const name of order) {
    const index = ids.findIndex(id => id === `dbscan-${name}` || id.startsWith(`dbscan-${name}-`));
    expect(index, name).toBeGreaterThan(previous);
    previous = index;
  }
  device.destroy();
});

it('GPUSpatialClustering schedules only scalar and cluster nodes for empty input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUSpatialClustering({
    ...createProps(graph, {}, 0),
    clusters: createCluster(graph, 2)
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toEqual(['spatial-clustering-cluster-count', 'spatial-clustering-publish']);
  device.destroy();
});

it('GPUSpatialClustering validates props', () => {
  const device = createNullWebGPUDevice();
  const expectThrow = (
    build: (graph: GPUCommandGraph) => Partial<GPUSpatialClusteringProps>,
    pattern: RegExp
  ) => {
    const graph = new GPUCommandGraph(device);
    expect(() => new GPUSpatialClustering(createProps(graph, build(graph)))).toThrow(pattern);
  };

  expectThrow(() => ({gridSize: [0, 4]}), /gridSize/);
  expectThrow(() => ({gridSize: [2.5, 4]}), /gridSize/);
  expectThrow(() => ({gridSize: [65536, 65536]}), /gridSize/);
  expectThrow(
    graph => ({
      parameters: createTransientView(graph, 'short', 'float32', 7)
    }),
    /parameters must hold 8/
  );
  expectThrow(
    graph => ({
      positions: createTransientView(graph, 'p3', 'float32', 16) as never
    }),
    /positions/
  );
  expectThrow(
    graph => ({
      labels: createTransientView(graph, 'short-labels', 'uint32', 15)
    }),
    /labels length/
  );
  expectThrow(
    graph => ({
      rootRows: createTransientView(graph, 'short-roots', 'uint32', 3)
    }),
    /rootRows length/
  );
  expectThrow(
    graph => ({
      coreFlags: createTransientView(graph, 'short-core', 'uint32', 3)
    }),
    /coreFlags length/
  );
  expectThrow(
    graph => ({
      sourceIds: createTransientView(graph, 'short-ids', 'uint32', 3)
    }),
    /sourceIds length/
  );
  expectThrow(
    graph => ({
      clusterCount: createTransientView(graph, 'empty-count', 'uint32', 0)
    }),
    /clusterCount/
  );
  expectThrow(
    graph => ({
      clusterSizes: createTransientView(graph, 'orphan-sizes', 'uint32', 4)
    }),
    /require clusters/
  );
  expectThrow(
    graph => ({
      clusterCentroids: createTransientView(graph, 'orphan-centroids', 'float32x2', 4)
    }),
    /require clusters/
  );
  expectThrow(
    graph => ({
      clusters: createCluster(graph, 4),
      clusterSizes: createTransientView(graph, 'wrong-sizes', 'uint32', 3)
    }),
    /clusterSizes length/
  );
  expectThrow(
    graph => ({
      clusters: createCluster(graph, 4),
      clusterCentroids: createTransientView(graph, 'wrong-centroids', 'float32x2', 5)
    }),
    /clusterCentroids length/
  );
  expectThrow(graph => ({clusters: createCluster(graph, 0)}), /at least one row/);
  expectThrow(graph => {
    const positions = createTransientView(graph, 'shared', 'float32x2', 16);
    return {
      positions,
      labels: graph.createDataView(positions.buffer, {
        format: 'uint32',
        length: 16
      })
    };
  }, /outputs must not share buffers with inputs/);
  device.destroy();
});

it('GPUSpatialClustering rejects views from another graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const otherGraph = new GPUCommandGraph(device);
  const contributor = new GPUSpatialClustering(createProps(graph));
  expect(() => contributor.getCommandNodes(otherGraph)).toThrow(/must belong to the target graph/);
  device.destroy();
});

it('GPUSpatialClustering schedules sort nodes for centroids only with sumOrder sorted', () => {
  const device = createNullWebGPUDevice();
  const getIds = (overrides: Partial<GPUSpatialClusteringProps>) => {
    const graph = new GPUCommandGraph(device);
    return new GPUSpatialClustering({
      ...createProps(graph, {}, 16),
      id: 'dbscan',
      clusters: createCluster(graph, 4),
      clusterCentroids: createTransientView(graph, 'centroids', 'float32x2', 4),
      ...overrides
    })
      .getCommandNodes(graph)
      .map(node => node.id);
  };
  const defaults = getIds({});
  expect(getIds({sumOrder: 'atomic'})).toEqual(defaults);
  const sorted = getIds({sumOrder: 'sorted'});
  expect(new Set(sorted).size).toBe(sorted.length);
  expect(sorted.at(-1)).toBe('dbscan-publish');
  expect(defaults.some(id => id.startsWith('dbscan-cluster-sums-sort'))).toBe(false);
  expect(sorted.some(id => id.startsWith('dbscan-cluster-sums-sort'))).toBe(true);
  expect(sorted.some(id => id === 'dbscan-cluster-sums-reduce-x')).toBe(true);
  expect(sorted.some(id => id === 'dbscan-cluster-sums-reduce-y')).toBe(true);
  expect(sorted.some(id => id === 'dbscan-cluster-sums-x')).toBe(false);
  expect(sorted.indexOf('dbscan-cluster-sums-reduce-y')).toBeLessThan(
    sorted.indexOf('dbscan-cluster-centroids')
  );
  expect(sorted.length).toBeGreaterThan(defaults.length);
  const graph = new GPUCommandGraph(device);
  expect(
    () => new GPUSpatialClustering(createProps(graph, {sumOrder: 'fast' as unknown as 'sorted'}))
  ).toThrow(/sumOrder must be atomic or sorted/);
  device.destroy();
});
