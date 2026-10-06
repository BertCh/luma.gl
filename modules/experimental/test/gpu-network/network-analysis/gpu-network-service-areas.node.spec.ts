// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkServiceAreas,
  type GPUNetworkServiceAreasProps
} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkServiceAreasProps> = {}
): GPUNetworkServiceAreasProps {
  return {
    offsets: createTransientView(graph, 'offsets', 'uint32', 9),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 10),
    weights: createTransientView(graph, 'weights', 'float32', 10),
    facilities: createTransientView(graph, 'facilities', 'uint32', 3),
    assignments: createTransientView(graph, 'assignments', 'uint32', 8),
    ...overrides
  };
}

it('GPUNetworkServiceAreas schedules reachability, label propagation, and optional outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const service = new GPUNetworkServiceAreas({
    ...createProps(graph),
    id: 'service',
    maxIterations: 2,
    converged: createTransientView(graph, 'converged', 'uint32', 1),
    facilityNodeCounts: createTransientView(graph, 'node-counts', 'uint32', 3),
    facilityCostSums: createTransientView(graph, 'cost-sums', 'float32', 3)
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
  expect(() => create({offsets: createTransientView(graph, 'offsets8', 'uint32', 8)})).toThrow(
    /offsets must contain one more row/
  );
  expect(() => create({weights: createTransientView(graph, 'weights9', 'float32', 9)})).toThrow(
    /weights length/
  );
  expect(() =>
    create({facilityCosts: createTransientView(graph, 'facility-costs', 'float32', 4)})
  ).toThrow(/facilityCosts length/);
  expect(() =>
    create({facilityNodeCounts: createTransientView(graph, 'node-counts', 'uint32', 4)})
  ).toThrow(/facilityNodeCounts length/);
  expect(() =>
    create({facilityCostSums: createTransientView(graph, 'cost-sums', 'float32', 2)})
  ).toThrow(/facilityCostSums length/);
  expect(() => create({costLimit: createTransientView(graph, 'limit', 'float32', 2)})).toThrow(
    /costLimit must contain exactly one row/
  );
  expect(() => create({costs: createTransientView(graph, 'costs7', 'float32', 7)})).toThrow(
    /costs length/
  );
  for (const maxIterations of [0, 1025, 1.5]) {
    expect(() => create({maxIterations})).toThrow(/maxIterations/);
    expect(() => create({labelIterations: maxIterations})).toThrow(/labelIterations/);
  }
  for (const localIterations of [0, 65, 1.5]) {
    expect(() => create({localIterations})).toThrow(/localIterations/);
  }
  expect(() =>
    create({costs: createTransientView(graph, 'uint-costs', 'uint32', 8) as never})
  ).toThrow(/costs/);
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      weights: graph.createDataView(shared, {format: 'float32', length: 10}),
      costs: graph.createDataView(shared, {format: 'float32', length: 8})
    })
  ).toThrow(/separate buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUNetworkServiceAreas(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});
