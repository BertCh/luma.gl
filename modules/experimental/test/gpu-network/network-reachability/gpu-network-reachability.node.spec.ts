// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkReachability,
  type GPUNetworkReachabilityOutput,
  type GPUNetworkReachabilityProps
} from '../../../src/gpu-network/network-reachability';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkReachabilityProps> = {}
): GPUNetworkReachabilityProps {
  return {
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 10),
      weights: createTransientView(graph, 'weights', 'float32', 10)
    },
    startNodes: createTransientView(graph, 'sources', 'uint32', 2),
    output: {costs: createTransientView(graph, 'costs', 'float32', 8)},
    ...overrides
  };
}

/** Props with extra output views merged into the default output bag. */
function createPropsWithOutput(
  graph: GPUCommandGraph,
  output: Partial<GPUNetworkReachabilityOutput>,
  overrides: Partial<GPUNetworkReachabilityProps> = {}
): GPUNetworkReachabilityProps {
  const props = createProps(graph, overrides);
  return {...props, output: {...props.output, ...output}};
}

it('GPUNetworkReachability schedules one gated relax node per round and optional outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUNetworkReachability({
    ...createPropsWithOutput(graph, {
      predecessors: createTransientView(graph, 'predecessors', 'uint32', 8),
      bands: createTransientView(graph, 'bands', 'uint32', 8),
      bandCounts: createTransientView(graph, 'band-counts', 'uint32', 2),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1),
        iterationCount: createTransientView(graph, 'iteration-count', 'uint32', 1)
      }
    }),
    id: 'reach',
    maximumIterations: 2,
    costBreaks: createTransientView(graph, 'thresholds', 'float32', 2)
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
    ...createPropsWithOutput(tieGraph, {
      predecessors: createTransientView(tieGraph, 'predecessors', 'uint32', 8)
    }),
    id: 'tie',
    maximumIterations: 2,
    maximumTieIterations: 7
  })
    .getCommandNodes(tieGraph)
    .map(node => node.id);
  expect(tieIds.filter(id => id.startsWith('tie-tie-level-')).length).toBe(7);
  expect(tieIds.length).toBe(2 + 2 + 1 + 4 + 7);

  const emptyGraph = new GPUCommandGraph(device);
  const emptyIds = new GPUNetworkReachability(
    createProps(emptyGraph, {
      startNodes: createTransientView(emptyGraph, 'no-sources', 'uint32', 0)
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
  const withOutput = (output: Partial<GPUNetworkReachabilityOutput>) =>
    create({output: {...base.output, ...output}});
  const withTopology = (topology: Partial<GPUNetworkReachabilityProps['topology']>) =>
    create({topology: {...base.topology, ...topology}});
  expect(() =>
    withTopology({offsets: createTransientView(graph, 'offsets8', 'uint32', 8)})
  ).toThrow(/topology must contain one node per output.costs row/);
  expect(() =>
    withTopology({weights: createTransientView(graph, 'weights9', 'float32', 9)})
  ).toThrow(/topology.weights length/);
  expect(() => withTopology({weights: undefined as never})).toThrow(/topology.weights is required/);
  expect(() =>
    create({
      startCosts: createTransientView(graph, 'source-costs', 'float32', 3)
    })
  ).toThrow(/startCosts length/);
  for (const maximumIterations of [0, 1025, 1.5]) {
    expect(() => create({maximumIterations})).toThrow(/maximumIterations/);
  }
  for (const maximumTieIterations of [0, 1025, 1.5]) {
    expect(() => create({maximumTieIterations})).toThrow(/maximumTieIterations/);
  }
  for (const localIterations of [0, 65, 2.5]) {
    expect(() => create({localIterations})).toThrow(/localIterations/);
  }
  expect(() => withOutput({bands: createTransientView(graph, 'bands', 'uint32', 8)})).toThrow(
    /costBreaks/
  );
  const thresholds = createTransientView(graph, 'thresholds', 'float32', 2);
  expect(() => create({costBreaks: thresholds})).toThrow(/costBreaks/);
  expect(() =>
    create({
      costBreaks: thresholds,
      output: {
        ...base.output,
        bandCounts: createTransientView(graph, 'band-counts', 'uint32', 3)
      }
    })
  ).toThrow(/bandCounts length/);
  expect(() =>
    withOutput({costs: createTransientView(graph, 'uint-costs', 'uint32', 8) as never})
  ).toThrow(/costs/);
  const shared = graph.createTransientBuffer({
    id: 'shared',
    byteLength: 64,
    usage: 128
  });
  expect(() =>
    create({
      topology: {
        ...base.topology,
        weights: graph.createDataView(shared, {format: 'float32', length: 10})
      },
      output: {costs: graph.createDataView(shared, {format: 'float32', length: 8})}
    })
  ).toThrow(/separate buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUNetworkReachability(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});

it('GPUNetworkReachability validates topology views', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (topology: unknown) =>
    new GPUNetworkReachability({...base, topology: topology as never});
  expect(() => create(undefined)).toThrow(/topology is required/);
  expect(() =>
    create({...base.topology, offsets: createTransientView(graph, 'float-offsets', 'float32', 9)})
  ).toThrow(/topology.offsets/);
  expect(() =>
    create({
      ...base.topology,
      nodePositions: createTransientView(graph, 'positions', 'float32x2', 7)
    })
  ).toThrow(/topology.nodePositions must contain one row per node/);
  device.destroy();
});

it('GPUNetworkReachability bakes a numeric startCount without a binding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const constantNodes = new GPUNetworkReachability({
    ...createProps(graph),
    id: 'constant',
    maximumIterations: 1,
    startCount: 1
  }).getCommandNodes(graph);
  const viewGraph = new GPUCommandGraph(device);
  const viewNodes = new GPUNetworkReachability({
    ...createProps(viewGraph),
    id: 'view',
    maximumIterations: 1,
    startCount: createTransientView(viewGraph, 'start-count', 'uint32', 1)
  }).getCommandNodes(viewGraph);
  const bindingNames = (nodes: typeof constantNodes) =>
    (nodes.find(node => node.id.endsWith('-seed'))?.resources ?? []).length;
  // The constant adds no resource for the count, the view adds one.
  expect(bindingNames(viewNodes) - bindingNames(constantNodes)).toBe(1);
  expect(constantNodes.map(node => node.id.replace('constant', 'x'))).toEqual(
    viewNodes.map(node => node.id.replace('view', 'x'))
  );
  expect(
    () => new GPUNetworkReachability({...createProps(new GPUCommandGraph(device)), startCount: 3})
  ).toThrow(/startCount must be an integer in \[0, 2\]/);
  expect(
    () => new GPUNetworkReachability({...createProps(new GPUCommandGraph(device)), startCount: 1.5})
  ).toThrow(/startCount/);
  device.destroy();
});

it('GPUNetworkReachability keeps outputs in the output bag', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(Object.keys(props.output)).toEqual(['costs']);
  expect('costs' in props).toBe(false);
  expect(
    () =>
      new GPUNetworkReachability({
        ...props,
        output: {
          ...props.output,
          unresolvedCount: createTransientView(graph, 'unresolved', 'uint32', 1)
        }
      })
  ).toThrow(/output.unresolvedCount requires output.predecessors/);
  device.destroy();
});

it('GPUNetworkReachability graph size is one node per round for the explorer prop set', () => {
  // Same props as the reachability explorer mode: no predecessors.
  const countNodes = (maximumIterations: number) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const ids = new GPUNetworkReachability({
      ...createPropsWithOutput(graph, {
        bands: createTransientView(graph, 'bands', 'uint32', 8),
        bandCounts: createTransientView(graph, 'band-counts', 'uint32', 4),
        status: {
          converged: createTransientView(graph, 'converged', 'uint32', 1),
          iterationCount: createTransientView(graph, 'iteration-count', 'uint32', 1)
        }
      }),
      id: 'explorer',
      maximumIterations,
      costLimit: createTransientView(graph, 'cost-limit', 'float32', 1),
      costBreaks: createTransientView(graph, 'thresholds', 'float32', 4)
    })
      .getCommandNodes(graph)
      .map(node => node.id);
    device.destroy();
    expect(ids.filter(id => id.includes('-gate-'))).toEqual([]);
    expect(ids.filter(id => id.startsWith('explorer-relax-')).length).toBe(maximumIterations);
    return ids.length;
  };
  // Rounds are graph nodes: the frontier path is taken, and the budget sets the size.
  expect(countNodes(640) - countNodes(48)).toBe(640 - 48);
  expect(countNodes(48)).toBeLessThan(60);
});

it('GPUNetworkReachability lanes share one CSR and validate their layout', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  // 8 nodes per lane, 3 lanes: offsets describe one lane, costs hold every lane.
  const lanes = createProps(graph, {
    laneCount: 3,
    output: {costs: createTransientView(graph, 'lane-costs', 'float32', 24)}
  });
  const nodes = new GPUNetworkReachability({...lanes, id: 'lanes'}).getCommandNodes(graph);
  expect(nodes.some(node => node.id === 'lanes-relax-0')).toBe(true);
  expect(
    () =>
      new GPUNetworkReachability(
        createProps(new GPUCommandGraph(device), {
          laneCount: 3,
          output: {
            costs: createTransientView(new GPUCommandGraph(device), 'bad-costs', 'float32', 25)
          }
        })
      )
  ).toThrow(/multiple of laneCount/);
  expect(
    () =>
      new GPUNetworkReachability({
        ...lanes,
        output: {
          ...lanes.output,
          predecessors: createTransientView(graph, 'lane-predecessors', 'uint32', 24)
        }
      })
  ).toThrow(/does not support output.predecessors/);
  // Expanded offsets (the old layout) no longer match once lanes share the CSR.
  expect(
    () =>
      new GPUNetworkReachability({
        ...lanes,
        topology: {
          ...lanes.topology,
          offsets: createTransientView(graph, 'expanded-offsets', 'uint32', 25)
        }
      })
  ).toThrow(/topology must contain one node per output.costs row per lane/);
});
