// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUHierarchicalDensityClustering,
  type GPUHierarchicalDensityClusteringProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {
  computeHierarchicalDensityModel,
  getAdjustedRandIndex,
  HIERARCHICAL_DENSITY_FIXTURE
} from './hierarchical-density-clustering-oracle';

import {
  createDensitySelectionInput,
  createParallelDensityTree,
  createSerialDensityTree,
  selectDensityTreeParallel,
  selectDensityTreeSerial
} from './hierarchical-density-tree-model';

let serial = 0;

/**
 * Smallest adjusted Rand index accepted against scikit-learn. With tied mutual reachability
 * distances the merge order of equal heights is implementation-defined; on the fixture scenes the
 * model differs from scikit-learn in at most 12 of 1300 points (index 0.98), and scenes without
 * effective ties match exactly.
 */
const TIED_SCENE_MINIMUM_RAND_INDEX = 0.97;

function createProps(
  graph: GPUCommandGraph,
  rows: number,
  overrides: Partial<GPUHierarchicalDensityClusteringProps> = {}
): GPUHierarchicalDensityClusteringProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', rows),
    minimumClusterSize: 15,
    minimumSamples: 32,
    output: {
      labels: view('uint32', rows),
      coreDistances: view('float32', rows),
      status: {count: view('uint32', 1)}
    },
    ...overrides
  };
}

function getNodes(rows: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  return new GPUHierarchicalDensityClustering(createProps(graph, rows)).getCommandNodes(graph);
}

it('GPUHierarchicalDensityClustering kernels parse and fit the core limits', () => {
  const nodes = getNodes(1000);
  for (const [kernelId, source] of getParsedKernelSources(nodes)) {
    expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), kernelId).toBeLessThanOrEqual(8192);
    expect(getWorkgroupMemoryBytes(source), kernelId).toBeLessThanOrEqual(16384);
  }
  const ids = nodes.map(node => node.id);
  const order = [
    'hierarchical-density-clustering-neighbors',
    'hierarchical-density-clustering-init',
    'hierarchical-density-clustering-node-data',
    'hierarchical-density-clustering-gate-0',
    'hierarchical-density-clustering-search-0',
    'hierarchical-density-clustering-add-0',
    'hierarchical-density-clustering-edge-sort-2',
    'hierarchical-density-clustering-dendrogram-splice',
    'hierarchical-density-clustering-dendrogram-finish',
    'hierarchical-density-clustering-pointer-init',
    'hierarchical-density-clustering-cluster-offsets',
    'hierarchical-density-clustering-stability',
    'hierarchical-density-clustering-selection-depth-init',
    'hierarchical-density-clustering-selection-chain-jump-0-0',
    'hierarchical-density-clustering-selection-representatives-jump-0',
    'hierarchical-density-clustering-selection-labels',
    'hierarchical-density-clustering-labels'
  ];
  let previous = -1;
  for (const prefix of order) {
    const index = ids.findIndex(id => id === prefix || id.startsWith(`${prefix}-`));
    expect(index, prefix).toBeGreaterThan(previous);
    previous = index;
  }
});

it('GPUHierarchicalDensityClustering rounds add nodes but no kernel variants (rule 12)', () => {
  // 1000 and 5000 rows unroll 11 and 14 Boruvka rounds and different pointer-jump counts.
  const small = getNodes(1000);
  const large = getNodes(5000);
  expect(large.length).toBeGreaterThan(small.length);
  const roundPattern =
    /-(gate|resolve|leaf-flags|search|tie-low|tie-high|add)-\d+|-(tour|ancestors|depth|representatives)-(gate|jump|copy)-\d+|selection-(level-gate|functions|chain-gate|chain-jump|chain-copy|best|advance)-\d+/;
  const countVariants = (nodes: readonly unknown[]) =>
    getGPUCommandNodePipelineKeys(nodes, {idPattern: roundPattern}).size;
  expect(countVariants(large)).toBe(countVariants(small));
});

it('tree contraction gates iterative work and never walks the whole tree in one invocation', () => {
  const nodes = getNodes(1025);
  const iterative = nodes.filter(node =>
    /-(tour|ancestors|depth|representatives|chain)-(jump|copy)-|selection-(functions|best)-/.test(
      node.id
    )
  );
  expect(iterative.length).toBeGreaterThan(100);
  for (const node of iterative) {
    expect(node.condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
    expect(node.workload?.maximumInvocationCount, node.id).toBeGreaterThan(1);
  }
  for (const node of nodes.filter(node =>
    /-dendrogram-splice|-selection-candidates/.test(node.id)
  )) {
    expect(node.workload?.maximumInvocationCount, node.id).toBeGreaterThan(1);
  }
  for (const [kernelId, source] of getParsedKernelSources(
    nodes.filter(node => /-(dendrogram|condensing|selection)-/.test(node.id))
  )) {
    expect(source, kernelId).not.toMatch(/for \(var (rank|cluster|step) =/);
  }
  // Crossing a power of two adds exactly one pointer round, without adding any pipeline family.
  const small = getNodes(1024);
  expect(nodes.length).toBeGreaterThan(small.length);
  const pattern = /-(dendrogram|condensing|selection)-/;
  expect(getGPUCommandNodePipelineKeys(nodes, {idPattern: pattern}).size).toBe(
    getGPUCommandNodePipelineKeys(small, {idPattern: pattern}).size
  );
});

it('GPUHierarchicalDensityClustering validates props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 64);
  expect(() => new GPUHierarchicalDensityClustering({...props, minimumClusterSize: 1})).toThrow(
    /minimumClusterSize/
  );
  expect(() => new GPUHierarchicalDensityClustering({...props, minimumSamples: 33})).toThrow(
    /minimumSamples/
  );
  expect(
    () =>
      new GPUHierarchicalDensityClustering({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow();
  expect(
    () =>
      new GPUHierarchicalDensityClustering({
        ...props,
        output: {labels: createTransientView(graph, 'short-labels', 'uint32', 3)}
      })
  ).toThrow(/labels/);
  // minimumSamples defaults to min(minimumClusterSize, 32).
  expect(
    new GPUHierarchicalDensityClustering({
      ...props,
      minimumSamples: undefined,
      minimumClusterSize: 50
    }).minimumSamples
  ).toBe(32);
});

it('the GPU formulation (model) reproduces scikit-learn HDBSCAN partitions', () => {
  for (const scene of HIERARCHICAL_DENSITY_FIXTURE.scenes) {
    const model = computeHierarchicalDensityModel(
      scene.positions,
      scene.minimumClusterSize,
      scene.minimumSamples
    );
    expect(
      createParallelDensityTree(scene.positions.length, model.sortedEdges),
      scene.name
    ).toEqual(createSerialDensityTree(scene.positions.length, model.sortedEdges));
    const selection = createDensitySelectionInput(
      scene.positions.length,
      model.sortedEdges,
      scene.minimumClusterSize
    );
    expect(selectDensityTreeParallel(selection), `${scene.name} exact EOM parity`).toEqual(
      selectDensityTreeSerial(selection)
    );
    // The spanning tree is a minimum one: its sorted weights equal scikit-learn's merge heights.
    expect(model.spanningTreeWeights.length, scene.name).toBe(scene.spanningTreeWeights.length);
    model.spanningTreeWeights.forEach((weight, index) => {
      const expected = scene.spanningTreeWeights[index];
      expect(Math.abs(weight - expected), `${scene.name} weight ${index}`).toBeLessThanOrEqual(
        1e-9 * Math.max(1, expected)
      );
    });
    // Tied merges may condense a few points differently (see the generator); the partitions agree.
    expect(getAdjustedRandIndex(model.labels, scene.labels), scene.name).toBeGreaterThanOrEqual(
      TIED_SCENE_MINIMUM_RAND_INDEX
    );
    expect(Math.max(...model.labels), scene.name).toBe(Math.max(...scene.labels));
    model.coreDistances.forEach((core, row) => {
      const expected = scene.coreDistances[row];
      if (Number.isFinite(expected)) {
        expect(Math.abs(core - expected), `${scene.name} core ${row}`).toBeLessThanOrEqual(1e-9);
      }
    });
  }
}, 600000);

it('distance measurements reject Lambert frames and accept AEQD meters', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const lambert = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'local-frame', kind: 'local-lambert-equal-area'}
  } as const;
  const create = (spatialContext: GPUSpatialContext) =>
    new GPUHierarchicalDensityClustering(createProps(graph, 100, {spatialContext}));
  expect(() => create(lambert)).toThrow(/measures distance/);
  expect(() =>
    create({...lambert, frame: {...lambert.frame, kind: 'local-azimuthal-equidistant'}})
  ).not.toThrow();
  device.destroy();
});
