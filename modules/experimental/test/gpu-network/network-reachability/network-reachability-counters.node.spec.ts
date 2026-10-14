// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUNetworkReachabilityInflation,
  GPUNetworkReachability,
  type GPUNetworkReachabilityProps
} from '../../../src/gpu-network/network-reachability';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

const NODE_COUNT = 8;
const EDGE_COUNT = 10;

function createCountedProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkReachabilityProps> = {},
  counters: {relaxation?: boolean; reached?: boolean} = {relaxation: true, reached: true}
): GPUNetworkReachabilityProps {
  return {
    id: 'counted',
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', NODE_COUNT + 1),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', EDGE_COUNT),
      weights: createTransientView(graph, 'weights', 'float32', EDGE_COUNT)
    },
    startNodes: createTransientView(graph, 'sources', 'uint32', 2),
    output: {
      costs: createTransientView(graph, 'costs', 'float32', NODE_COUNT),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1),
        ...(counters.relaxation
          ? {relaxationCount: createTransientView(graph, 'relaxation-count', 'uint32', 2)}
          : {}),
        ...(counters.reached
          ? {reachedEdgeCount: createTransientView(graph, 'reached-edge-count', 'uint32', 2)}
          : {})
      }
    },
    ...overrides
  };
}

function buildNodes(maximumIterations: number, localIterations = 16) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `counted-${maximumIterations}`});
  return new GPUNetworkReachability(
    createCountedProps(graph, {maximumIterations, localIterations})
  ).getCommandNodes(graph);
}

it('GPUNetworkReachability counters add a reached-edges node and keep kernels within 8 bindings', () => {
  const nodes = buildNodes(3);
  const ids = nodes.map(node => node.id);
  expect(ids).toContain('counted-reached-edges');
  expect(ids.indexOf('counted-reached-edges')).toBeGreaterThan(ids.indexOf('counted-relax-2'));
  expect(ids.indexOf('counted-finalize')).toBeGreaterThan(ids.indexOf('counted-reached-edges'));
  for (const node of nodes) {
    expect(node.resources?.length ?? 0, node.id).toBeLessThanOrEqual(8 + 1);
  }
  // The relax kernel keeps its eight storage bindings; the counters live in the control buffer.
  const sources = getParsedKernelSources(nodes);
  for (const [id, source] of sources) {
    const bindingCount = (source.match(/@binding\(/g) ?? []).length;
    expect(bindingCount, id).toBeLessThanOrEqual(8);
  }
  expect(sources.get('counted-relax-0')).toContain('addCount(');
  expect(sources.get('counted-relax-0')).toContain('counterLow');
});

it('GPUNetworkReachability without counters compiles no counting code', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'plain'});
  const nodes = new GPUNetworkReachability(createCountedProps(graph, {}, {})).getCommandNodes(
    graph
  );
  expect(nodes.map(node => node.id)).not.toContain('counted-reached-edges');
  const sources = getParsedKernelSources(nodes);
  expect(sources.get('counted-relax-0')).not.toContain('addCount');
});

it('GPUNetworkReachability counters can be requested one at a time', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'relax-only'});
  const nodes = new GPUNetworkReachability(
    createCountedProps(graph, {}, {relaxation: true})
  ).getCommandNodes(graph);
  expect(nodes.map(node => node.id)).not.toContain('counted-reached-edges');
  const reachedGraph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'reached-only'});
  const reachedNodes = new GPUNetworkReachability(
    createCountedProps(reachedGraph, {}, {reached: true})
  ).getCommandNodes(reachedGraph);
  expect(reachedNodes.map(node => node.id)).toContain('counted-reached-edges');
  expect(getParsedKernelSources(reachedNodes).get('counted-relax-0')).not.toContain('addCount');
});

it('GPUNetworkReachability counters keep the pipeline count independent of the round count', () => {
  for (const localIterations of [1, 16, 32]) {
    const small = buildNodes(8, localIterations);
    const large = buildNodes(9, localIterations);
    const count = (nodes: readonly unknown[]) =>
      getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size;
    expect(count(large)).toBe(count(small));
    expect(
      getGPUCommandNodePipelineKeys(small, {includeShaderLayout: true, idPattern: /-relax-\d+$/})
        .size
    ).toBe(1);
  }
});

it('GPUNetworkReachability validates the counter views', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createCountedProps(graph);
  const create = (status: Record<string, unknown>) =>
    new GPUNetworkReachability({
      ...base,
      output: {...base.output, status: {...base.output.status, ...status}}
    });
  for (const name of ['relaxationCount', 'reachedEdgeCount']) {
    expect(() => create({[name]: createTransientView(graph, `${name}-1`, 'uint32', 1)})).toThrow(
      /exactly two rows/
    );
    expect(() => create({[name]: createTransientView(graph, `${name}-3`, 'uint32', 3)})).toThrow(
      /exactly two rows/
    );
    expect(() => create({[name]: createTransientView(graph, `${name}-f`, 'float32', 2)})).toThrow(
      new RegExp(name)
    );
  }
  const shared = base.output.status?.relaxationCount;
  expect(() => create({reachedEdgeCount: shared})).toThrow(/separate buffers/);
  device.destroy();
});

it('getGPUNetworkReachabilityInflation combines low and high words', () => {
  expect(getGPUNetworkReachabilityInflation([30, 0], [10, 0])).toBe(3);
  expect(getGPUNetworkReachabilityInflation(Uint32Array.from([0, 2]), [0, 1])).toBe(2);
  expect(getGPUNetworkReachabilityInflation([5, 0], [0, 0])).toBeNaN();
});
