// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUNetworkLineGraphParameterValues,
  GPUNetworkLineGraph
} from '../../../src/gpu-network/network-line-graph/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPUNetworkLineGraph schedules the count, scan, fill and clamp nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUNetworkLineGraph({
    id: 'line',
    offsets: createTransientView(graph, 'offsets', 'uint32', 5),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
    weights: createTransientView(graph, 'weights', 'float32', 6),
    nodePositions: createTransientView(graph, 'positions', 'float32x2', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    bannedTurns: createTransientView(graph, 'banned', 'uint32', 4),
    lineOffsets: createTransientView(graph, 'line-offsets', 'uint32', 7),
    lineNeighbors: createTransientView(graph, 'line-neighbors', 'uint32', 20),
    lineWeights: createTransientView(graph, 'line-weights', 'float32', 20),
    arcCount: createTransientView(graph, 'arc-count', 'uint32', 1)
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.slice(0, 2)).toEqual(['line-edge-vectors', 'line-count']);
  expect(ids.slice(-4)).toEqual([
    'line-publish',
    'line-fill-neighbors',
    'line-fill-weights',
    'line-clamp'
  ]);
});

it('GPUNetworkLineGraph fuses the weight fill into the neighbor fill when the bindings fit', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const ids = new GPUNetworkLineGraph({
    id: 'line',
    offsets: createTransientView(graph, 'offsets', 'uint32', 5),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
    weights: createTransientView(graph, 'weights', 'float32', 6),
    nodePositions: createTransientView(graph, 'positions', 'float32x2', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    lineOffsets: createTransientView(graph, 'line-offsets', 'uint32', 7),
    lineNeighbors: createTransientView(graph, 'line-neighbors', 'uint32', 20),
    lineWeights: createTransientView(graph, 'line-weights', 'float32', 20)
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  // Eight storage bindings without bans fit the default limit.
  expect(ids).not.toContain('line-fill-weights');
  expect(ids.slice(-2)).toEqual(['line-fill-neighbors', 'line-clamp']);
});

it('GPUNetworkLineGraph validates its views and parameters', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 5),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
    weights: createTransientView(graph, 'weights', 'float32', 6),
    nodePositions: createTransientView(graph, 'positions', 'float32x2', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    lineOffsets: createTransientView(graph, 'line-offsets', 'uint32', 7),
    lineNeighbors: createTransientView(graph, 'line-neighbors', 'uint32', 20),
    lineWeights: createTransientView(graph, 'line-weights', 'float32', 20)
  };
  expect(() => new GPUNetworkLineGraph({...props, lineOffsets: props.offsets})).toThrow(
    /edgeCount \+ 1/
  );
  expect(
    () =>
      new GPUNetworkLineGraph({
        ...props,
        lineWeights: createTransientView(graph, 'short', 'float32', 3)
      })
  ).toThrow(/lineWeights length/);
  expect(() => getGPUNetworkLineGraphParameterValues({angleCost: -1})).toThrow(/angleCost/);
  expect(getGPUNetworkLineGraphParameterValues({bannedTurnCount: 3})[5]).toBe(3);
});
