// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkNeighborhood,
  type GPUNetworkNeighborhoodProps
} from '../../../src/map-graphs/network-analysis/gpu-network-neighborhood';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkNeighborhoodProps> = {}
): GPUNetworkNeighborhoodProps {
  return {
    offsets: createTransientView(graph, 'offsets', 'uint32', 9),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 10),
    seeds: createTransientView(graph, 'seeds', 'uint32', 2),
    hops: createTransientView(graph, 'hops', 'uint32', 1),
    hopDistances: createTransientView(graph, 'hop-distances', 'uint32', 8),
    ...overrides
  };
}

function createOutput(graph: GPUCommandGraph, name: string, capacity: number) {
  return {
    ids: createTransientView(graph, `${name}-ids`, 'uint32', capacity),
    count: createTransientView(graph, `${name}-count`, 'uint32', 1),
    overflow: createTransientView(graph, `${name}-overflow`, 'uint32', 1)
  };
}

it('GPUNetworkNeighborhood schedules nodes in order', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recipe = new GPUNetworkNeighborhood({
    ...createProps(graph),
    id: 'hood',
    maxHops: 2,
    nodeMask: createTransientView(graph, 'node-mask', 'uint32', 8),
    edgeMask: createTransientView(graph, 'edge-mask', 'uint32', 10),
    nodes: createOutput(graph, 'nodes', 4),
    edges: createOutput(graph, 'edges', 20)
  });
  expect(recipe.recipe).toBe('network-neighborhood');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  const positionOf = (id: string) => ids.indexOf(id);
  expect(ids[0]).toBe('hood-unit-weights');
  expect(ids.slice(1, 4)).toEqual([
    'hood-reachability-initialize',
    'hood-reachability-seed',
    'hood-reachability-relax-0'
  ]);
  const order = [
    'hood-reachability-relax-1',
    'hood-hops',
    'hood-edge-clear',
    'hood-edge-mask',
    'hood-node-publish',
    'hood-edge-publish'
  ].map(positionOf);
  expect(order.every(position => position >= 0)).toBe(true);
  expect([...order].sort((left, right) => left - right)).toEqual(order);
  expect(ids.some(id => id.startsWith('hood-node-visibility'))).toBe(true);
  expect(ids.some(id => id.startsWith('hood-edge-visibility'))).toBe(true);
  expect(ids.filter(id => id.startsWith('hood-reachability-relax')).length).toBe(2);
  device.destroy();
});

it('GPUNetworkNeighborhood omits optional nodes and handles empty edge sets', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const minimalIds = new GPUNetworkNeighborhood(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(minimalIds).toContain('network-neighborhood-hops');
  for (const absent of ['edge-clear', 'edge-mask', 'node-publish', 'edge-publish']) {
    expect(minimalIds).not.toContain(`network-neighborhood-${absent}`);
  }

  const emptyGraph = new GPUCommandGraph(device);
  const emptyIds = new GPUNetworkNeighborhood({
    ...createProps(emptyGraph, {
      neighbors: createTransientView(emptyGraph, 'no-neighbors', 'uint32', 0)
    }),
    id: 'empty',
    edges: createOutput(emptyGraph, 'edges', 4)
  })
    .getCommandNodes(emptyGraph)
    .map(node => node.id);
  expect(emptyIds).not.toContain('empty-unit-weights');
  expect(emptyIds).not.toContain('empty-edge-mask');
  expect(emptyIds).toContain('empty-edge-total');
  expect(emptyIds).toContain('empty-edge-publish');
  device.destroy();
});

it('GPUNetworkNeighborhood validates props', () => {
  const device = createNullWebGPUDevice();
  const create = (
    build: (
      graph: GPUCommandGraph,
      base: GPUNetworkNeighborhoodProps
    ) => Partial<GPUNetworkNeighborhoodProps>
  ) => {
    const graph = new GPUCommandGraph(device);
    const base = createProps(graph);
    return new GPUNetworkNeighborhood({...base, ...build(graph, base)});
  };
  const view = (graph: GPUCommandGraph, length: number, format: 'uint32' | 'float32' = 'uint32') =>
    createTransientView(graph, 'override', format, length);
  expect(() => create(graph => ({offsets: view(graph, 8)}))).toThrow(/offsets/);
  expect(() => create(graph => ({hops: view(graph, 2)}))).toThrow(/hops/);
  expect(() => create(graph => ({seedCount: view(graph, 2)}))).toThrow(/seedCount/);
  expect(() => create(() => ({maxHops: 0}))).toThrow(/maxHops/);
  expect(() => create(() => ({maxHops: 1025}))).toThrow(/maxHops/);
  expect(() => create(() => ({maxHops: 1.5}))).toThrow(/maxHops/);
  expect(() => create(graph => ({nodeIds: view(graph, 7)}))).toThrow(/nodeIds/);
  expect(() => create(graph => ({edgeIds: view(graph, 9)}))).toThrow(/edgeIds/);
  expect(() => create(graph => ({edgeMask: view(graph, 9)}))).toThrow(/edgeMask/);
  expect(() => create(graph => ({hopDistances: view(graph, 8, 'float32') as never}))).toThrow(
    /hopDistances/
  );
  expect(() => create((_, base) => ({nodeMask: base.hopDistances}))).toThrow(/separate buffers/);
  expect(() =>
    create((graph, base) => ({nodes: {...createOutput(graph, 'alias', 2), ids: base.seeds}}))
  ).toThrow(/outputs/);
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  expect(() => new GPUNetworkNeighborhood(createProps(other)).getCommandNodes(graph)).toThrow(
    /target graph/
  );
  device.destroy();
});
