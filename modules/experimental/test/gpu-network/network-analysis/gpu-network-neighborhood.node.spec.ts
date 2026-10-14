// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkNeighborhood,
  type GPUNetworkNeighborhoodOutput,
  type GPUNetworkNeighborhoodProps
} from '../../../src/gpu-network/network-analysis/gpu-network-neighborhood';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkNeighborhoodProps> = {}
): GPUNetworkNeighborhoodProps {
  return {
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 10)
    },
    startNodes: createTransientView(graph, 'seeds', 'uint32', 2),
    hops: createTransientView(graph, 'hops', 'uint32', 1),
    output: {hopDistances: createTransientView(graph, 'hop-distances', 'uint32', 8)},
    ...overrides
  };
}

/** Props with extra output views merged into the default output bag. */
function createPropsWithOutput(
  graph: GPUCommandGraph,
  output: Partial<GPUNetworkNeighborhoodOutput>,
  overrides: Partial<GPUNetworkNeighborhoodProps> = {}
): GPUNetworkNeighborhoodProps {
  const props = createProps(graph, overrides);
  return {...props, output: {...props.output, ...output}};
}

function createOutput(graph: GPUCommandGraph, name: string, capacity: number) {
  return {
    ids: createTransientView(graph, `${name}-ids`, 'uint32', capacity),
    status: {
      count: createTransientView(graph, `${name}-count`, 'uint32', 1),
      overflow: createTransientView(graph, `${name}-overflow`, 'uint32', 1)
    }
  };
}

it('GPUNetworkNeighborhood schedules nodes in order', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUNetworkNeighborhood({
    ...createPropsWithOutput(graph, {
      nodeMask: createTransientView(graph, 'node-mask', 'uint32', 8),
      edgeMask: createTransientView(graph, 'edge-mask', 'uint32', 10),
      nodes: createOutput(graph, 'nodes', 4),
      edges: createOutput(graph, 'edges', 20)
    }),
    id: 'hood',
    maximumHops: 2
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
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
    ...createPropsWithOutput(
      emptyGraph,
      {edges: createOutput(emptyGraph, 'edges', 4)},
      {
        topology: {
          offsets: createTransientView(emptyGraph, 'empty-offsets', 'uint32', 9),
          neighbors: createTransientView(emptyGraph, 'no-neighbors', 'uint32', 0)
        }
      }
    ),
    id: 'empty'
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
  const view = <Format extends 'uint32' | 'float32' = 'uint32'>(
    graph: GPUCommandGraph,
    length: number,
    format: Format = 'uint32' as Format
  ) => createTransientView(graph, 'override', format, length);
  const withOutput = (
    build: (
      graph: GPUCommandGraph,
      base: GPUNetworkNeighborhoodProps
    ) => GPUNetworkNeighborhoodOutput
  ) => create((graph, base) => ({output: build(graph, base)}));
  expect(() =>
    create((graph, base) => ({topology: {...base.topology, offsets: view(graph, 8)}}))
  ).toThrow(/topology must contain one node per output.hopDistances row/);
  expect(() =>
    create((graph, base) => ({topology: {...base.topology, offsets: view(graph, 1)}}))
  ).toThrow(/topology.offsets/);
  expect(() =>
    create((graph, base) => ({
      topology: {...base.topology, weights: view(graph, 9, 'float32')}
    }))
  ).toThrow(/topology.weights length/);
  expect(() => create(graph => ({hops: view(graph, 2)}))).toThrow(/hops/);
  expect(() => create(graph => ({startCount: view(graph, 2)}))).toThrow(/startCount/);
  expect(() => create(() => ({startCount: 3}))).toThrow(
    /startCount must be an integer in \[0, 2\]/
  );
  expect(() => create(() => ({maximumHops: 0}))).toThrow(/maximumHops/);
  expect(() => create(() => ({maximumHops: 1025}))).toThrow(/maximumHops/);
  expect(() => create(() => ({maximumHops: 1.5}))).toThrow(/maximumHops/);
  expect(() => create(graph => ({nodeIds: view(graph, 7)}))).toThrow(/nodeIds/);
  expect(() => create(graph => ({edgeIds: view(graph, 9)}))).toThrow(/edgeIds/);
  expect(() => withOutput((graph, base) => ({...base.output, edgeMask: view(graph, 9)}))).toThrow(
    /edgeMask/
  );
  expect(() => withOutput(graph => ({hopDistances: view(graph, 8, 'float32') as never}))).toThrow(
    /hopDistances/
  );
  expect(() =>
    withOutput((_, base) => ({...base.output, nodeMask: base.output.hopDistances}))
  ).toThrow(/separate buffers/);
  expect(() =>
    withOutput((graph, base) => ({
      ...base.output,
      nodes: {...createOutput(graph, 'alias', 2), ids: base.startNodes}
    }))
  ).toThrow(/outputs/);
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  expect(() => new GPUNetworkNeighborhood(createProps(other)).getCommandNodes(graph)).toThrow(
    /target graph/
  );
  device.destroy();
});

it('GPUNetworkNeighborhood bakes a numeric startCount without a binding', () => {
  const device = createNullWebGPUDevice();
  const countResources = (
    startCount: (graph: GPUCommandGraph) => number | GraphDataView<'uint32'>
  ) => {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUNetworkNeighborhood({
      ...createProps(graph),
      id: 'hood',
      startCount: startCount(graph)
    }).getCommandNodes(graph);
    return (nodes.find(node => node.id === 'hood-reachability-seed')?.resources ?? []).length;
  };
  const constant = countResources(() => 1);
  const view = countResources(graph => createTransientView(graph, 'start-count', 'uint32', 1));
  expect(view - constant).toBe(1);
  device.destroy();
});

it('GPUNetworkNeighborhood keeps every written view in the output bag', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(Object.keys(props.output)).toEqual(['hopDistances']);
  expect('hopDistances' in props).toBe(false);
  device.destroy();
});
