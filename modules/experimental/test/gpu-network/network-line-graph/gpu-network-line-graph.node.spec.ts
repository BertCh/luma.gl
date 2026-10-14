// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUNetworkLineGraphParameterValues,
  GPUNetworkLineGraph
} from '../../../src/gpu-network/network-line-graph/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('GPUNetworkLineGraph schedules the count, scan, fill and clamp nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUNetworkLineGraph({
    id: 'line',
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
      weights: createTransientView(graph, 'weights', 'float32', 6),
      nodePositions: createTransientView(graph, 'positions', 'float32x2', 4)
    },
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    bannedTurns: createTransientView(graph, 'banned', 'uint32', 4),
    output: {
      topology: {
        offsets: createTransientView(graph, 'line-offsets', 'uint32', 7),
        neighbors: createTransientView(graph, 'line-neighbors', 'uint32', 20),
        weights: createTransientView(graph, 'line-weights', 'float32', 20)
      },
      arcCount: createTransientView(graph, 'arc-count', 'uint32', 1),
      status: {
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  // Bans are sorted by fromEdge first (review B-19), then turned into interleaved pairs.
  expect(ids[0]).toBe('line-ban-keys');
  const pairs = ids.indexOf('line-ban-pairs');
  expect(ids.slice(1, pairs).every(id => id.startsWith('line-ban-sort'))).toBe(true);
  expect(ids.slice(pairs + 1, pairs + 3)).toEqual(['line-edge-vectors', 'line-count']);
  expect(ids.slice(-6)).toEqual([
    'line-publish',
    'line-clear-line-neighbors',
    'line-clear-line-weights',
    'line-fill-neighbors',
    'line-fill-weights',
    'line-clamp'
  ]);
});

it('GPUNetworkLineGraph fuses the weight fill into the neighbor fill when the bindings fit', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const ids = new GPUNetworkLineGraph({
    id: 'line',
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
      weights: createTransientView(graph, 'weights', 'float32', 6),
      nodePositions: createTransientView(graph, 'positions', 'float32x2', 4)
    },
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    output: {
      topology: {
        offsets: createTransientView(graph, 'line-offsets', 'uint32', 7),
        neighbors: createTransientView(graph, 'line-neighbors', 'uint32', 20),
        weights: createTransientView(graph, 'line-weights', 'float32', 20)
      },
      status: {
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  // Eight storage bindings without bans fit the default limit.
  expect(ids).not.toContain('line-fill-weights');
  expect(ids.slice(-4)).toEqual([
    'line-clear-line-neighbors',
    'line-clear-line-weights',
    'line-fill-neighbors',
    'line-clamp'
  ]);
});

it('GPUNetworkLineGraph validates its views and parameters', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = {
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
      weights: createTransientView(graph, 'weights', 'float32', 6),
      nodePositions: createTransientView(graph, 'positions', 'float32x2', 4)
    },
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    output: {
      topology: {
        offsets: createTransientView(graph, 'line-offsets', 'uint32', 7),
        neighbors: createTransientView(graph, 'line-neighbors', 'uint32', 20),
        weights: createTransientView(graph, 'line-weights', 'float32', 20)
      },
      status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
    }
  };
  expect(
    () =>
      new GPUNetworkLineGraph({
        ...props,
        output: {
          ...props.output,
          topology: {...props.output.topology, offsets: props.topology.offsets}
        }
      })
  ).toThrow(/edgeCount \+ 1/);
  expect(
    () =>
      new GPUNetworkLineGraph({
        ...props,
        output: {
          ...props.output,
          topology: {
            ...props.output.topology,
            weights: createTransientView(graph, 'short', 'float32', 3)
          }
        }
      })
  ).toThrow(/output\.topology\.weights length/);
  expect(
    () =>
      new GPUNetworkLineGraph({
        ...props,
        topology: {...props.topology, weights: createTransientView(graph, 'w5', 'float32', 5)}
      })
  ).toThrow(/topology\.weights length/);
  expect(
    () =>
      new GPUNetworkLineGraph({
        ...props,
        topology: {...props.topology, nodePositions: undefined} as never
      })
  ).toThrow(/topology\.nodePositions is required/);
  expect(() => getGPUNetworkLineGraphParameterValues({angleCost: -1})).toThrow(/angleCost/);
  expect(getGPUNetworkLineGraphParameterValues({bannedTurnCount: 3})[5]).toBe(3);
});

it('GPUNetworkLineGraph skips CSR tail rows and binary-searches sorted bans', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUNetworkLineGraph({
    id: 'tail',
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
      weights: createTransientView(graph, 'weights', 'float32', 6),
      nodePositions: createTransientView(graph, 'positions', 'float32x2', 4)
    },
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    bannedTurns: createTransientView(graph, 'banned', 'uint32', 8),
    output: {
      topology: {
        offsets: createTransientView(graph, 'line-offsets', 'uint32', 7),
        neighbors: createTransientView(graph, 'line-neighbors', 'uint32', 20),
        weights: createTransientView(graph, 'line-weights', 'float32', 20)
      },
      status: {
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  }).getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  // Review B-09: an edge row past offsets[NODE_COUNT] is not a line node.
  for (const kernel of ['tail-count', 'tail-fill-neighbors']) {
    expect(sources.get(kernel)).toContain('isLiveEdge(index)');
  }
  // Review B-19: the ban test reads the sorted pairs with a binary search in both passes.
  for (const kernel of ['tail-count', 'tail-fill-neighbors', 'tail-fill-weights']) {
    const source = sources.get(kernel) ?? '';
    expect(source).toContain('sortedBans');
    expect(source).not.toContain('bannedTurns[');
  }
  expect(sources.get('tail-ban-keys')).toContain('bannedTurns[');
});
