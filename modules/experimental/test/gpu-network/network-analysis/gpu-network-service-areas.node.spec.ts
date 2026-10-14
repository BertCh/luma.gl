// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkServiceAreas,
  type GPUNetworkServiceAreasOutput,
  type GPUNetworkServiceAreasProps
} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkServiceAreasProps> = {}
): GPUNetworkServiceAreasProps {
  return {
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 10),
      weights: createTransientView(graph, 'weights', 'float32', 10)
    },
    facilities: createTransientView(graph, 'facilities', 'uint32', 3),
    output: {assignments: createTransientView(graph, 'assignments', 'uint32', 8)},
    ...overrides
  };
}

/** Props with extra output views merged into the default output bag. */
function createPropsWithOutput(
  graph: GPUCommandGraph,
  output: Partial<GPUNetworkServiceAreasOutput>,
  overrides: Partial<GPUNetworkServiceAreasProps> = {}
): GPUNetworkServiceAreasProps {
  const props = createProps(graph, overrides);
  return {...props, output: {...props.output, ...output}};
}

it('GPUNetworkServiceAreas schedules reachability, label propagation, and optional outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const service = new GPUNetworkServiceAreas({
    ...createPropsWithOutput(graph, {
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1)
      },
      facilityNodeCounts: createTransientView(graph, 'node-counts', 'uint32', 3),
      facilityCostSums: createTransientView(graph, 'cost-sums', 'float32', 3)
    }),
    id: 'service',
    maximumIterations: 2
  });
  const ids = service.getCommandNodes(graph).map(node => node.id);
  const reachabilityIds = ids.filter(id => id.startsWith('service-reachability-'));
  expect(ids.slice(0, reachabilityIds.length)).toEqual(reachabilityIds);
  expect(reachabilityIds[0]).toBe('service-reachability-initialize');
  const labelStart = reachabilityIds.length;
  expect(ids.slice(labelStart, labelStart + 8)).toEqual([
    'service-label-initialize',
    'service-label-seed',
    'service-label-tight-edges',
    'service-label-frontier-seed',
    'service-label-round-0',
    'service-label-round-1',
    'service-finalize',
    expect.stringMatching(/^service-facility-node-counts/)
  ]);
  expect(ids.some(id => id.startsWith('service-facility-cost-sums'))).toBe(true);

  const minimalGraph = new GPUCommandGraph(device);
  const minimalIds = new GPUNetworkServiceAreas(createProps(minimalGraph))
    .getCommandNodes(minimalGraph)
    .map(node => node.id);
  expect(minimalIds).toContain('network-service-areas-label-seed');
  expect(minimalIds).not.toContain('network-service-areas-finalize');
  expect(minimalIds.filter(id => id.includes('-label-round-')).length).toBe(32);
  expect(minimalIds.some(id => id.includes('-label-gate-'))).toBe(false);
  // Default props must stay within the graph node budget (was 198 with hop-per-round gating).
  expect(minimalIds.length).toBeLessThanOrEqual(120);
  expect(minimalIds.some(id => id.includes('facility-'))).toBe(false);

  const emptyGraph = new GPUCommandGraph(device);
  const emptyIds = new GPUNetworkServiceAreas(
    createProps(emptyGraph, {facilities: createTransientView(emptyGraph, 'none', 'uint32', 0)})
  )
    .getCommandNodes(emptyGraph)
    .map(node => node.id);
  expect(emptyIds).not.toContain('network-service-areas-label-seed');
  device.destroy();
});

it('GPUNetworkServiceAreas validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUNetworkServiceAreasProps>) =>
    new GPUNetworkServiceAreas({...base, ...overrides});
  const withOutput = (output: Partial<GPUNetworkServiceAreasOutput>) =>
    create({output: {...base.output, ...output}});
  const withTopology = (topology: Partial<GPUNetworkServiceAreasProps['topology']>) =>
    create({topology: {...base.topology, ...topology}});
  expect(() =>
    withTopology({offsets: createTransientView(graph, 'offsets8', 'uint32', 8)})
  ).toThrow(/topology must contain one node per output.assignments row/);
  expect(() =>
    withTopology({weights: createTransientView(graph, 'weights9', 'float32', 9)})
  ).toThrow(/topology.weights length/);
  expect(() => withTopology({weights: undefined as never})).toThrow(/topology.weights is required/);
  expect(() =>
    create({facilityCosts: createTransientView(graph, 'facility-costs', 'float32', 4)})
  ).toThrow(/facilityCosts length/);
  expect(() =>
    withOutput({facilityNodeCounts: createTransientView(graph, 'node-counts', 'uint32', 4)})
  ).toThrow(/output.facilityNodeCounts length/);
  expect(() =>
    withOutput({facilityCostSums: createTransientView(graph, 'cost-sums', 'float32', 2)})
  ).toThrow(/output.facilityCostSums length/);
  expect(() => create({costLimit: createTransientView(graph, 'limit', 'float32', 2)})).toThrow(
    /costLimit must contain exactly one row/
  );
  expect(() => withOutput({costs: createTransientView(graph, 'costs7', 'float32', 7)})).toThrow(
    /output.costs length/
  );
  for (const maximumIterations of [0, 1025, 1.5]) {
    expect(() => create({maximumIterations})).toThrow(/maximumIterations/);
    expect(() => create({maximumLabelIterations: maximumIterations})).toThrow(
      /maximumLabelIterations/
    );
  }
  for (const localIterations of [0, 65, 1.5]) {
    expect(() => create({localIterations})).toThrow(/localIterations/);
  }
  expect(() =>
    withOutput({costs: createTransientView(graph, 'uint-costs', 'uint32', 8) as never})
  ).toThrow(/costs/);
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      topology: {
        ...base.topology,
        weights: graph.createDataView(shared, {format: 'float32', length: 10})
      },
      output: {
        ...base.output,
        costs: graph.createDataView(shared, {format: 'float32', length: 8})
      }
    })
  ).toThrow(/separate buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUNetworkServiceAreas(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});

it('GPUNetworkServiceAreas derives predecessors inside each service area after labelling', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUNetworkServiceAreas({
    ...createPropsWithOutput(graph, {
      predecessors: createTransientView(graph, 'predecessors', 'uint32', 8),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1)
      }
    }),
    id: 'areas',
    maximumTieIterations: 3
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  // Review B-12: the search itself computes no predecessors; a same-label predecessor phase
  // follows the label rounds, and its truncation feeds `converged`.
  expect(ids.some(id => id.startsWith('areas-reachability-') && id.includes('tie'))).toBe(false);
  const lastLabelRound = ids.lastIndexOf(
    'areas-label-round-' + (ids.filter(id => id.startsWith('areas-label-round-')).length - 1)
  );
  expect(ids.indexOf('areas-predecessors')).toBeGreaterThan(lastLabelRound);
  expect(ids.filter(id => id.startsWith('areas-tie-level-'))).toHaveLength(3);
  expect(ids.indexOf('areas-tie-predecessors')).toBeLessThan(ids.indexOf('areas-finalize'));
  const sources = getParsedKernelSources(nodes);
  expect(sources.get('areas-predecessors')).toContain(
    'if (labels[labelsOffset + index] != labels[labelsOffset + targetNode]) { continue; }'
  );
  expect(sources.get('areas-tie-roots')).toContain('if (labels[labelsOffset + node] != index)');
  expect(sources.get('areas-finalize')).toMatch(/control\[controlOffset \+ \d+u\] == 0u/);
});

it('GPUNetworkServiceAreas bakes a numeric facilityCount without a binding', () => {
  const device = createNullWebGPUDevice();
  const getSeedSource = (facilityCount: GPUNetworkServiceAreasProps['facilityCount']) => {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUNetworkServiceAreas({
      ...createProps(graph),
      id: 'areas',
      facilityCount:
        typeof facilityCount === 'function' ? undefined : (facilityCount as number | undefined)
    }).getCommandNodes(graph);
    return getParsedKernelSources(nodes);
  };
  const constant = getSeedSource(2);
  for (const name of ['areas-reachability-seed', 'areas-label-seed']) {
    expect(constant.get(name)).toContain('if (index >= 2u) { return; }');
    expect(constant.get(name)).not.toContain('facilityCount[');
    expect(constant.get(name)).not.toContain('sourceCount[');
  }
  const viewGraph = new GPUCommandGraph(device);
  const viewSources = getParsedKernelSources(
    new GPUNetworkServiceAreas({
      ...createProps(viewGraph),
      id: 'areas',
      facilityCount: createTransientView(viewGraph, 'facility-count', 'uint32', 1)
    }).getCommandNodes(viewGraph)
  );
  expect(viewSources.get('areas-label-seed')).toContain(
    'if (index >= facilityCount[facilityCountOffset]) { return; }'
  );
  expect(viewSources.get('areas-reachability-seed')).toContain(
    'if (index >= sourceCount[sourceCountOffset]) { return; }'
  );
  expect(
    () =>
      new GPUNetworkServiceAreas({
        ...createProps(new GPUCommandGraph(device)),
        facilityCount: 4
      })
  ).toThrow(/facilityCount must be an integer in \[0, 3\]/);
  device.destroy();
});

it('GPUNetworkServiceAreas keeps every written view in the output bag', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(Object.keys(props.output)).toEqual(['assignments']);
  expect('assignments' in props).toBe(false);
  device.destroy();
});
