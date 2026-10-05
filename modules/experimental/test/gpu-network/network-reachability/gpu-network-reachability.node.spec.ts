// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkReachability,
  type GPUNetworkReachabilityProps
} from '../../../src/gpu-network/network-reachability';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkReachabilityProps> = {}
): GPUNetworkReachabilityProps {
  return {
    offsets: createTransientView(graph, 'offsets', 'uint32', 9),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 10),
    weights: createTransientView(graph, 'weights', 'float32', 10),
    sources: createTransientView(graph, 'sources', 'uint32', 2),
    costs: createTransientView(graph, 'costs', 'float32', 8),
    ...overrides
  };
}

it('GPUNetworkReachability schedules one gated relax node per round and optional outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUNetworkReachability({
    ...createProps(graph),
    id: 'reach',
    maxIterations: 2,
    predecessors: createTransientView(graph, 'predecessors', 'uint32', 8),
    bandThresholds: createTransientView(graph, 'thresholds', 'float32', 2),
    bands: createTransientView(graph, 'bands', 'uint32', 8),
    bandCounts: createTransientView(graph, 'band-counts', 'uint32', 2),
    converged: createTransientView(graph, 'converged', 'uint32', 1),
    iterationCount: createTransientView(graph, 'iteration-count', 'uint32', 1)
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.slice(0, 15)).toEqual([
    'reach-initialize',
    'reach-seed',
    'reach-relax-0',
    'reach-relax-1',
    'reach-tie-initialize',
    'reach-predecessors',
    'reach-tie-roots',
    'reach-tie-seed',
    'reach-tie-level-0',
    'reach-tie-level-1',
    'reach-tie-level-2',
    'reach-tie-level-3',
    'reach-tie-predecessors',
    'reach-finalize',
    'reach-bands'
  ]);
  expect(ids.slice(15).every(id => id.startsWith('reach-band-counts'))).toBe(true);
  const relax = nodes.find(node => node.id === 'reach-relax-0');
  expect(relax?.condition).toMatchObject({source: 'gpu', mode: 'indirect'});
  expect(
    relax?.resources?.some(resource => 'usage' in resource && resource.usage === 'indirect')
  ).toBe(true);

  const minimalGraph = new GPUCommandGraph(device);
  const minimal = new GPUNetworkReachability(createProps(minimalGraph));
  expect(minimal.id).toBe('network-reachability');
  const minimalIds = minimal.getCommandNodes(minimalGraph).map(node => node.id);
  expect(minimalIds.slice(0, 2)).toEqual([
    'network-reachability-initialize',
    'network-reachability-seed'
  ]);
  // Without predecessors there is no tie phase.
  expect(minimalIds.length).toBe(2 + 64);

  const tieGraph = new GPUCommandGraph(device);
  const tieIds = new GPUNetworkReachability({
    ...createProps(tieGraph),
    id: 'tie',
    maxIterations: 2,
    maxTieIterations: 7,
    predecessors: createTransientView(tieGraph, 'predecessors', 'uint32', 8)
  })
    .getCommandNodes(tieGraph)
    .map(node => node.id);
  expect(tieIds.filter(id => id.startsWith('tie-tie-level-')).length).toBe(7);
  expect(tieIds.length).toBe(2 + 2 + 1 + 4 + 7);

  const emptyGraph = new GPUCommandGraph(device);
  const emptyIds = new GPUNetworkReachability(
    createProps(emptyGraph, {
      sources: createTransientView(emptyGraph, 'no-sources', 'uint32', 0)
    })
  )
    .getCommandNodes(emptyGraph)
    .map(node => node.id);
  expect(emptyIds).not.toContain('network-reachability-seed');
  device.destroy();
});

it('GPUNetworkReachability validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUNetworkReachabilityProps>) =>
    new GPUNetworkReachability({...base, ...overrides});
  expect(() => create({offsets: createTransientView(graph, 'offsets8', 'uint32', 8)})).toThrow(
    /offsets must contain one more row/
  );
  expect(() => create({weights: createTransientView(graph, 'weights9', 'float32', 9)})).toThrow(
    /weights length/
  );
  expect(() =>
    create({
      sourceCosts: createTransientView(graph, 'source-costs', 'float32', 3)
    })
  ).toThrow(/sourceCosts length/);
  for (const maxIterations of [0, 1025, 1.5]) {
    expect(() => create({maxIterations})).toThrow(/maxIterations/);
  }
  for (const maxTieIterations of [0, 1025, 1.5]) {
    expect(() => create({maxTieIterations})).toThrow(/maxTieIterations/);
  }
  for (const localIterations of [0, 65, 2.5]) {
    expect(() => create({localIterations})).toThrow(/localIterations/);
  }
  expect(() => create({bands: createTransientView(graph, 'bands', 'uint32', 8)})).toThrow(
    /bandThresholds/
  );
  const thresholds = createTransientView(graph, 'thresholds', 'float32', 2);
  expect(() => create({bandThresholds: thresholds})).toThrow(/bandThresholds/);
  expect(() =>
    create({
      bandThresholds: thresholds,
      bandCounts: createTransientView(graph, 'band-counts', 'uint32', 3)
    })
  ).toThrow(/bandCounts length/);
  expect(() =>
    create({
      costs: createTransientView(graph, 'uint-costs', 'uint32', 8) as never
    })
  ).toThrow(/costs/);
  const shared = graph.createTransientBuffer({
    id: 'shared',
    byteLength: 64,
    usage: 128
  });
  expect(() =>
    create({
      weights: graph.createDataView(shared, {format: 'float32', length: 10}),
      costs: graph.createDataView(shared, {format: 'float32', length: 8})
    })
  ).toThrow(/separate buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUNetworkReachability(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});

it('GPUNetworkReachability graph size is one node per round for the explorer prop set', () => {
  // Same props as the reachability explorer mode: no predecessors.
  const countNodes = (maxIterations: number) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const ids = new GPUNetworkReachability({
      ...createProps(graph),
      id: 'explorer',
      maxIterations,
      costLimit: createTransientView(graph, 'cost-limit', 'float32', 1),
      bandThresholds: createTransientView(graph, 'thresholds', 'float32', 4),
      bands: createTransientView(graph, 'bands', 'uint32', 8),
      bandCounts: createTransientView(graph, 'band-counts', 'uint32', 4),
      converged: createTransientView(graph, 'converged', 'uint32', 1),
      iterationCount: createTransientView(graph, 'iteration-count', 'uint32', 1)
    })
      .getCommandNodes(graph)
      .map(node => node.id);
    device.destroy();
    expect(ids.filter(id => id.includes('-gate-'))).toEqual([]);
    expect(ids.filter(id => id.startsWith('explorer-relax-')).length).toBe(maxIterations);
    return ids.length;
  };
  // Rounds are graph nodes: the frontier path is taken, and the budget sets the size.
  expect(countNodes(640) - countNodes(48)).toBe(640 - 48);
  expect(countNodes(48)).toBeLessThan(60);
});
