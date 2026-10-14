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
  type GPUSpatialClusteringOutput,
  type GPUSpatialClusteringProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {getSpatialClusteringSharedWGSL} from '../../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-kernels';
import {
  clusterGeographicPointsOracle,
  clusterPointsOracle,
  createGeographicBlobScene,
  createSeededRandom,
  getHaversineDistance,
  ORACLE_EARTH_RADIUS,
  ORACLE_NOISE
} from './spatial-clustering-oracle';

/** Input props and output views mixed in one flat object, split by {@link createProps}. */
type ClusteringOverrides = Partial<Omit<GPUSpatialClusteringProps, 'output'>> &
  Partial<GPUSpatialClusteringOutput>;

const OUTPUT_KEYS = [
  'labels',
  'rootRows',
  'coreFlags',
  'clusterCount',
  'clusters',
  'drawInstanceCount',
  'clusterSizes',
  'clusterCentroids'
] as const;

function createProps(
  graph: GPUCommandGraph,
  overrides: ClusteringOverrides = {},
  rows: number = 16
): GPUSpatialClusteringProps {
  const inputs: Record<string, unknown> = {};
  const output: Record<string, unknown> = {
    labels: createTransientView(graph, 'labels', 'uint32', rows)
  };
  for (const [key, value] of Object.entries(overrides)) {
    if ((OUTPUT_KEYS as readonly string[]).includes(key)) {
      output[key] = value;
    } else {
      inputs[key] = value;
    }
  }
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    gridSize: [8, 8],
    ...inputs,
    output: output as GPUSpatialClusteringOutput
  };
}

function createCluster(graph: GPUCommandGraph, capacity: number) {
  return {
    ids: createTransientView(graph, 'cluster-ids', 'uint32', capacity),
    status: {
      count: createTransientView(graph, 'cluster-ids-count', 'uint32', 1),
      overflow: createTransientView(graph, 'cluster-ids-overflow', 'uint32', 1)
    }
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
  expect(ids[0]).toBe('spatial-clustering-grid-bounds');
  expect(ids.at(-1)).toBe('spatial-clustering-labels');
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids.every(id => id.startsWith('spatial-clustering-'))).toBe(true);
  expect(ids.some(id => id.endsWith('-publish'))).toBe(false);
  const prefixes = [
    'spatial-clustering-grid-bounds',
    'spatial-clustering-grid-positions',
    'spatial-clustering-grid-index',
    'spatial-clustering-cell-positions',
    'spatial-clustering-core',
    'spatial-clustering-cell-core-flags',
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
  const ids = new GPUSpatialClustering(
    createProps(
      graph,
      {
        id: 'dbscan',
        sumOrder: 'atomic',
        clusters: createCluster(graph, 4),
        clusterSizes: createTransientView(graph, 'sizes', 'uint32', 4),
        clusterCentroids: createTransientView(graph, 'centroids', 'float32x2', 4)
      },
      16
    )
  )
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
  const ids = new GPUSpatialClustering(createProps(graph, {clusters: createCluster(graph, 2)}, 0))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toEqual(['spatial-clustering-cluster-count', 'spatial-clustering-publish']);
  device.destroy();
});

it('GPUSpatialClustering validates props', () => {
  const device = createNullWebGPUDevice();
  const expectThrow = (build: (graph: GPUCommandGraph) => ClusteringOverrides, pattern: RegExp) => {
    const graph = new GPUCommandGraph(device);
    expect(() => new GPUSpatialClustering(createProps(graph, build(graph)))).toThrow(pattern);
  };

  expectThrow(() => ({gridSize: [0, 4]}), /gridSize/);
  expectThrow(() => ({gridSize: [2.5, 4]}), /gridSize/);
  // The device-derived grid maximum is checked at declaration.
  const gridGraph = new GPUCommandGraph(device);
  const largeGrid = new GPUSpatialClustering(createProps(gridGraph, {gridSize: [65536, 65536]}));
  expect(() => largeGrid.getCommandNodes(gridGraph)).toThrow(
    /spatial-clustering gridSize cells must be at most 33554431/
  );
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
  const getIds = (overrides: ClusteringOverrides) => {
    const graph = new GPUCommandGraph(device);
    return new GPUSpatialClustering(
      createProps(
        graph,
        {
          id: 'dbscan',
          clusters: createCluster(graph, 4),
          clusterCentroids: createTransientView(graph, 'centroids', 'float32x2', 4),
          ...overrides
        },
        16
      )
    )
      .getCommandNodes(graph)
      .map(node => node.id);
  };
  const defaults = getIds({sumOrder: 'atomic'});
  expect(getIds({})).toEqual(getIds({sumOrder: 'sorted'}));
  const sorted = getIds({});
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

it('GPUSpatialClustering resolves spatialContext: planar or great-circle only', () => {
  let serial = 0;
  const makeGraph = () =>
    new GPUCommandGraph(createNullWebGPUDevice(), {id: `clustering-context-${serial++}`});
  const build = (overrides: ClusteringOverrides = {}) => {
    const graph = makeGraph();
    return {graph, producer: new GPUSpatialClustering(createProps(graph, overrides))};
  };
  const planar = build().producer;
  expect(planar.spatialContext.coordinateSpace).toBe('planar');
  expect(planar.geographicRadius).toBeUndefined();
  const geographic = build({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
  }).producer;
  expect(geographic.geographicRadius).toBeCloseTo(ORACLE_EARTH_RADIUS, 6);
  const custom = build({
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      sphereRadius: 1000
    }
  }).producer;
  expect(custom.geographicRadius).toBe(1000);
  expect(
    () =>
      build({
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
      }).producer
  ).toThrow(/great-circle/);
  expect(
    () =>
      build({
        spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
      }).producer
  ).toThrow(/does not support/);
  expect(
    () =>
      build({
        denseBoxShortcut: true,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      }).producer
  ).toThrow(/denseBoxShortcut is planar only/);
  const geographicBuilt = build({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
  });
  const geographicNodes = geographicBuilt.producer
    .getCommandNodes(geographicBuilt.graph)
    .map(node => node.id);
  expect(geographicNodes).toContain('spatial-clustering-core');
  expect(getSpatialClusteringSharedWGSL(8, 8, 6371008.8)).toContain('geodesicHaversineTerm(');
  expect(getSpatialClusteringSharedWGSL(8, 8)).not.toContain('geodesicHaversineTerm(');
});

// Oracle: scikit-learn 1.9.1 DBSCAN(eps=150 / 6371008.8, min_samples=5, metric='haversine') on the
// f32 scene dumped by fix/work/C/U7/dump-geo.ts, via fix/work/C/U7/dbscan-haversine.py: 5 clusters,
// 93 noise points, identical core flags, noise set and core partition.
it('geographic DBSCAN oracle matches scikit-learn haversine DBSCAN on the pinned scene', () => {
  const positions = createGeographicBlobScene();
  const result = clusterGeographicPointsOracle(positions, {
    bounds: [-180, -90, 180, 90],
    epsilon: 150,
    minimumPoints: 5
  });
  expect(result.clusterCount).toBe(5);
  expect(result.labels.filter(label => label === ORACLE_NOISE).length).toBe(93);
  const sizes = new Array<number>(5).fill(0);
  for (const label of result.labels) if (label !== ORACLE_NOISE) sizes[label]++;
  expect(sizes.sort((a, b) => a - b)).toEqual([4, 7, 23, 32, 41]);
});

it('geographic lattice cells bound every epsilon neighbor pair', () => {
  // The cell height is epsilon / R of latitude and the cell width 2 asin(sin(theta / 2) / cos(lat))
  // of longitude at the largest |latitude| in the bounds; no pair within epsilon may exceed either.
  const random = createSeededRandom(5);
  for (const maxLatitude of [0, 30, 60, 80, 89]) {
    for (const epsilon of [10, 1000, 100000]) {
      const theta = epsilon / ORACLE_EARTH_RADIUS;
      const cosLatitude = Math.cos((maxLatitude * Math.PI) / 180);
      const ratio = Math.sin(theta / 2) / cosLatitude;
      const widthDegrees = ratio >= 0.99 ? 360 : (2 * Math.asin(ratio) * 180) / Math.PI;
      const heightDegrees = (theta * 180) / Math.PI;
      for (let trial = 0; trial < 400; trial++) {
        const a: [number, number] = [random() * 20, (random() * 2 - 1) * maxLatitude];
        // Random direction at a random distance up to epsilon, converted to a nearby point.
        const distance = random() * epsilon;
        const bearing = random() * 2 * Math.PI;
        const latitude = (a[1] * Math.PI) / 180;
        const lat2 = Math.asin(
          Math.sin(latitude) * Math.cos(distance / ORACLE_EARTH_RADIUS) +
            Math.cos(latitude) * Math.sin(distance / ORACLE_EARTH_RADIUS) * Math.cos(bearing)
        );
        const lon2 =
          (a[0] * Math.PI) / 180 +
          Math.atan2(
            Math.sin(bearing) * Math.sin(distance / ORACLE_EARTH_RADIUS) * Math.cos(latitude),
            Math.cos(distance / ORACLE_EARTH_RADIUS) - Math.sin(latitude) * Math.sin(lat2)
          );
        const b: [number, number] = [(lon2 * 180) / Math.PI, (lat2 * 180) / Math.PI];
        if (Math.abs(b[1]) > maxLatitude) continue;
        expect(getHaversineDistance(a, b)).toBeLessThanOrEqual(epsilon * (1 + 1e-9));
        expect(Math.abs(b[1] - a[1])).toBeLessThanOrEqual(heightDegrees * (1 + 1e-9));
        expect(Math.abs(b[0] - a[0])).toBeLessThanOrEqual(widthDegrees * (1 + 1e-9));
      }
    }
  }
});

it('GPUSpatialClustering neighbor scans read cell-ordered copies (coalesced reads)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUSpatialClustering(
    createProps(graph, {denseBoxShortcut: true}, 64)
  ).getCommandNodes(graph);
  const kernels = getGPUCommandNodeKernelSources(nodes);
  getParsedKernelSources(nodes);
  const sourceOf = (id: string) => {
    // Pipeline IDs carry the node ID plus a suffix; match the node ID exactly up to that suffix.
    const kernel = kernels.find(entry => new RegExp(`(^|[^\\w-])${id}(?![\\w-])`).test(entry.id));
    expect(kernel, id).toBeDefined();
    return kernel?.source ?? '';
  };
  for (const id of [
    'spatial-clustering-core',
    'spatial-clustering-union',
    'spatial-clustering-border'
  ]) {
    const source = sourceOf(id);
    // The scan reads positions by cell slot, never by row.
    expect(source, id).toMatch(/cellPositions\[cellPositionsOffset \+ slot \* 2u\]/);
    expect(source, id).not.toMatch(/var<storage[^;]*>\s+positions\s*:/);
  }
  for (const id of ['spatial-clustering-union', 'spatial-clustering-border']) {
    const source = sourceOf(id);
    expect(source, id).toMatch(/cellCoreFlags\[cellCoreFlagsOffset \+ slot\]/);
    expect(source, id).not.toMatch(/var<storage[^;]*>\s+coreFlags\s*:/);
  }
  for (const kernel of kernels) {
    expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  }
  device.destroy();
});
