// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUNetworkKFunctionParameterValues,
  GPUNetworkKFunction
} from '../../../src/gpu-network/network-k-function/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(graph: GPUCommandGraph, overrides: Record<string, unknown> = {}) {
  return {
    id: 'k',
    points: createTransientView(graph, 'points', 'float32x2', 10),
    nodePositions: createTransientView(graph, 'positions', 'float32x2', 6),
    offsets: createTransientView(graph, 'offsets', 'uint32', 7),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 12),
    weights: createTransientView(graph, 'weights', 'float32', 12),
    maxDistance: createTransientView(graph, 'max-distance', 'float32', 1),
    networkLength: createTransientView(graph, 'network-length', 'float32', 1),
    parameters: createTransientView(graph, 'parameters', 'uint32', 4),
    bandCount: 8,
    simulationCount: 3,
    rowsPerBlock: 15,
    kValues: createTransientView(graph, 'k-values', 'float32', 32),
    ...overrides
  };
}

it('GPUNetworkKFunction schedules snapping, simulation, one search block per row block and outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUNetworkKFunction({
    ...createProps(graph),
    pairCounts: createTransientView(graph, 'pair-counts', 'uint32', 32),
    envelope: createTransientView(graph, 'envelope', 'float32', 24),
    converged: createTransientView(graph, 'converged', 'uint32', 1),
    maxIterations: 2
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  // 40 rows in blocks of 15 rows: three blocks (15, 15, 10).
  expect(ids.filter(id => /^k-block-\d+-count$/.test(id))).toEqual([
    'k-block-0-count',
    'k-block-1-count',
    'k-block-2-count'
  ]);
  expect(ids).toContain('k-simulate');
  expect(ids).toContain('k-quantize');
  expect(ids.slice(-3)).toEqual(['k-finalize', 'k-envelope', 'k-converged']);
});

it('GPUNetworkKFunction skips the simulation nodes without simulations', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUNetworkKFunction({
    ...createProps(graph),
    simulationCount: 0,
    rowsPerBlock: 10,
    kValues: createTransientView(graph, 'k-values-single', 'float32', 8)
  }).getCommandNodes(graph);
  expect(nodes.some(node => node.id === 'k-simulate')).toBe(false);
});

it('GPUNetworkKFunction validates views and parameters', () => {
  const create = (overrides: Record<string, unknown>) => () =>
    new GPUNetworkKFunction({
      ...createProps(new GPUCommandGraph(createNullWebGPUDevice()), overrides)
    });
  expect(create({})).not.toThrow();
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    () =>
      new GPUNetworkKFunction(
        createProps(graph, {kValues: createTransientView(graph, 'short', 'float32', 8)})
      )
  ).toThrow(/kValues length must be 32/);
  expect(create({bandCount: 1})).toThrow(/bandCount/);
  expect(() => getGPUNetworkKFunctionParameterValues({seed: -1, activeSimulations: 1})).toThrow(
    /seed/
  );
  expect(
    Array.from(getGPUNetworkKFunctionParameterValues({seed: 2 ** 32 + 5, activeSimulations: 7}))
  ).toEqual([5, 1, 7, 0]);
});
