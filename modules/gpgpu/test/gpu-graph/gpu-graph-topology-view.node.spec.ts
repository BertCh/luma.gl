// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGraphConnectedComponents,
  GPUGraphCoreNumber,
  GPUGraphDegree,
  GPUGraphLabelPropagation,
  GPUGraphPageRank,
  GPUGraphTopologyView
} from '@luma.gl/gpgpu/gpu-graph';
import {NullDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';

const VERTEX_COUNT = 5;
const EDGE_COUNT = 8;

/** A CPU-only device the command graph accepts for declaration-only tests. */
function createDevice(): Device {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxComputeWorkgroupsPerDimension: 65535}
  });
  return device;
}

function createTopology(graph: GPUCommandGraph, prefix = 'csr', directed = false) {
  const adjacency = (name: string) => ({
    offsets: createTransientView(graph, `${prefix}-${name}-offsets`, 'uint32', VERTEX_COUNT + 1),
    neighbors: createTransientView(graph, `${prefix}-${name}-neighbors`, 'uint32', EDGE_COUNT)
  });
  return new GPUGraphTopologyView({
    vertexCount: VERTEX_COUNT,
    forward: adjacency('forward'),
    reverse: directed ? adjacency('reverse') : undefined
  });
}

it('GPUGraphTopologyView validates view shapes and directedness', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  const offsets = createTransientView(graph, 'offsets', 'uint32', VERTEX_COUNT + 1);
  const neighbors = createTransientView(graph, 'neighbors', 'uint32', EDGE_COUNT);
  const topology = new GPUGraphTopologyView({
    vertexCount: VERTEX_COUNT,
    forward: {offsets, neighbors}
  });
  expect(topology.graph).toEqual({
    vertexCount: VERTEX_COUNT,
    directed: false,
    edgeCount: EDGE_COUNT
  });
  expect(createTopology(graph, 'directed', true).graph.directed).toBe(true);

  expect(() => new GPUGraphTopologyView({vertexCount: -1, forward: {offsets, neighbors}})).toThrow(
    /vertexCount/
  );
  expect(
    () => new GPUGraphTopologyView({vertexCount: VERTEX_COUNT + 1, forward: {offsets, neighbors}})
  ).toThrow(/forward offsets must contain exactly 7 uint32 rows/);
  expect(
    () =>
      new GPUGraphTopologyView({
        vertexCount: VERTEX_COUNT,
        forward: {offsets, neighbors, overflow: createTransientView(graph, 'o2', 'uint32', 2)}
      })
  ).toThrow(/overflow must contain exactly 1/);
  expect(
    () =>
      new GPUGraphTopologyView({
        vertexCount: VERTEX_COUNT,
        forward: {offsets, neighbors: createTransientView(graph, 'f', 'float32', 4) as never}
      })
  ).toThrow(/neighbors must be packed/);
  expect(
    () =>
      new GPUGraphTopologyView({
        vertexCount: VERTEX_COUNT,
        directed: false,
        forward: {offsets, neighbors},
        reverse: {offsets, neighbors}
      })
  ).toThrow(/reverse adjacency requires a directed graph/);
  const otherGraph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPUGraphTopologyView({
        vertexCount: VERTEX_COUNT,
        forward: {
          offsets,
          neighbors: createTransientView(otherGraph, 'foreign', 'uint32', EDGE_COUNT)
        }
      })
  ).toThrow(/one command graph/);
  device.destroy();
});

it('gpu-graph algorithms accept view topologies and view outputs', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  const topology = createTopology(graph);
  const column = (name: string, format: 'uint32' | 'float32' = 'uint32', length = VERTEX_COUNT) =>
    createTransientView(graph, name, format, length) as never;

  new GPUGraphDegree({topology, output: column('degree')}).addToGraph(graph);
  new GPUGraphPageRank({
    topology,
    output: column('page-rank', 'float32'),
    residual: column('residual', 'float32', 1)
  }).addToGraph(graph);
  new GPUGraphCoreNumber({
    topology,
    output: column('core'),
    converged: column('core-converged', 'uint32', 1),
    degeneracy: column('core-degeneracy', 'uint32', 1)
  }).addToGraph(graph);
  new GPUGraphConnectedComponents({topology, output: column('components')}).addToGraph(graph);
  new GPUGraphLabelPropagation({topology, output: column('communities')}).addToGraph(graph);

  // Omitted overflow words become graph-owned zero words cleared before first use.
  const ids = graph.nodes.map(node => node.id);
  expect(ids).toContain('gpu-graph-page-rank-forward-overflow-exact-clear');
  expect(ids.indexOf('gpu-graph-page-rank-forward-overflow-exact-clear')).toBeLessThan(
    ids.findIndex(id => id.startsWith('gpu-graph-page-rank') && !id.includes('overflow'))
  );
  device.destroy();
});

it('gpu-graph algorithms validate view columns', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  const topology = createTopology(graph);

  expect(
    () => new GPUGraphDegree({topology, output: createTransientView(graph, 'short', 'uint32', 3)})
  ).toThrow(/output must contain exactly 5 uint32 rows/);
  expect(
    () =>
      new GPUGraphPageRank({
        topology,
        output: createTransientView(graph, 'rank-u32', 'uint32', VERTEX_COUNT) as never
      })
  ).toThrow(/must be packed, uint32-aligned float32/);
  expect(
    () =>
      new GPUGraphCoreNumber({
        topology,
        output: createTransientView(graph, 'core', 'uint32', VERTEX_COUNT),
        converged: createTransientView(graph, 'converged-2', 'uint32', 2)
      })
  ).toThrow(/converged must contain exactly 1 uint32 rows/);

  // A view from another command graph is rejected when the algorithm is added.
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUGraphDegree({
      topology,
      output: createTransientView(otherGraph, 'foreign-degree', 'uint32', VERTEX_COUNT)
    }).addToGraph(graph)
  ).toThrow(/must belong to command graph/);

  // Outputs may not alias an input handle or each other.
  expect(() =>
    new GPUGraphDegree({topology, output: topology.forward.offsets as never}).addToGraph(graph)
  ).toThrow();
  const labels = createTransientView(graph, 'labels', 'uint32', VERTEX_COUNT);
  const aliasedConverged = graph.createDataView(labels.buffer, {format: 'uint32', length: 1});
  expect(() =>
    new GPUGraphConnectedComponents({
      id: 'aliased',
      topology,
      output: labels,
      converged: aliasedConverged
    }).addToGraph(graph)
  ).toThrow(/converged must use a buffer distinct/);
  device.destroy();
});
