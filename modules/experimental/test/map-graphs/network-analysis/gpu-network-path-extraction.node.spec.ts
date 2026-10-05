// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkPathExtraction,
  type GPUNetworkPathExtractionProps
} from '../../../src/map-graphs/network-analysis/gpu-network-path-extraction';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

function createOutput(graph: GPUCommandGraph, prefix: string, capacity: number) {
  return {
    ids: createTransientView(graph, `${prefix}-ids`, 'uint32', capacity),
    count: createTransientView(graph, `${prefix}-count`, 'uint32', 1),
    overflow: createTransientView(graph, `${prefix}-overflow`, 'uint32', 1)
  };
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkPathExtractionProps> = {}
): GPUNetworkPathExtractionProps {
  return {
    predecessors: createTransientView(graph, 'predecessors', 'uint32', 8),
    costs: createTransientView(graph, 'costs', 'float32', 8),
    targets: createTransientView(graph, 'targets', 'uint32', 4),
    output: createOutput(graph, 'out', 16),
    ...overrides
  };
}

function createEdges(graph: GPUCommandGraph) {
  return {
    offsets: createTransientView(graph, 'offsets', 'uint32', 9),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 10),
    weights: createTransientView(graph, 'weights', 'float32', 10),
    output: createOutput(graph, 'edge-out', 16)
  };
}

it('GPUNetworkPathExtraction schedules nodes without and with edges', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recipe = new GPUNetworkPathExtraction(createProps(graph));
  expect(recipe.recipe).toBe('network-path-extraction');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids.slice(0, 2)).toEqual([
    'network-path-extraction-clear',
    'network-path-extraction-measure'
  ]);
  expect(ids.slice(-3)).toEqual([
    'network-path-extraction-totals',
    'network-path-extraction-write-nodes',
    'network-path-extraction-publish'
  ]);
  expect(ids.some(id => id.startsWith('network-path-extraction-node-scan'))).toBe(true);
  expect(ids.some(id => id.includes('edge'))).toBe(false);

  const edgeGraph = new GPUCommandGraph(device);
  const edgeIds = new GPUNetworkPathExtraction({
    ...createProps(edgeGraph),
    id: 'path',
    edges: createEdges(edgeGraph)
  })
    .getCommandNodes(edgeGraph)
    .map(node => node.id);
  expect(edgeIds).toContain('path-edge-lengths');
  expect(edgeIds.slice(-4)).toEqual([
    'path-link-clear',
    'path-write-links',
    'path-write-edges',
    'path-edge-publish'
  ]);
  expect(edgeIds.indexOf('path-publish')).toBeLessThan(edgeIds.indexOf('path-link-clear'));

  const emptyGraph = new GPUCommandGraph(device);
  const emptyIds = new GPUNetworkPathExtraction({
    ...createProps(emptyGraph, {
      targets: createTransientView(emptyGraph, 'no-targets', 'uint32', 0)
    }),
    id: 'path',
    edges: createEdges(emptyGraph)
  })
    .getCommandNodes(emptyGraph)
    .map(node => node.id);
  expect(emptyIds).toEqual(['path-totals', 'path-publish', 'path-edge-publish']);
  device.destroy();
});

it('GPUNetworkPathExtraction validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUNetworkPathExtractionProps>) =>
    new GPUNetworkPathExtraction({...base, ...overrides});
  expect(() => create({costs: createTransientView(graph, 'costs7', 'float32', 7)})).toThrow(
    /costs length/
  );
  expect(() => create({nodeIds: createTransientView(graph, 'node-ids', 'uint32', 7)})).toThrow(
    /nodeIds length/
  );
  expect(() =>
    create({targetCount: createTransientView(graph, 'target-count', 'uint32', 2)})
  ).toThrow(/targetCount/);
  expect(() => create({pathOffsets: createTransientView(graph, 'offsets4', 'uint32', 4)})).toThrow(
    /pathOffsets/
  );
  expect(() => create({pathCosts: createTransientView(graph, 'path-costs', 'float32', 5)})).toThrow(
    /pathCosts length/
  );
  expect(() => create({pathFound: createTransientView(graph, 'path-found', 'uint32', 5)})).toThrow(
    /pathFound length/
  );
  for (const maxPathLength of [0, 65537, 2.5]) {
    expect(() => create({maxPathLength})).toThrow(/maxPathLength/);
  }
  const edges = createEdges(graph);
  expect(() =>
    create({edges: {...edges, offsets: createTransientView(graph, 'offsets8', 'uint32', 8)}})
  ).toThrow(/edges.offsets/);
  expect(() =>
    create({edges: {...edges, weights: createTransientView(graph, 'weights9', 'float32', 9)}})
  ).toThrow(/edges.weights/);
  expect(() =>
    create({edges: {...edges, edgeIds: createTransientView(graph, 'edge-ids9', 'uint32', 9)}})
  ).toThrow(/edges.edgeIds/);
  expect(() =>
    create({costs: createTransientView(graph, 'uint-costs', 'uint32', 8) as never})
  ).toThrow(/costs/);
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      costs: graph.createDataView(shared, {format: 'float32', length: 8}),
      pathCosts: graph.createDataView(shared, {format: 'float32', length: 4})
    })
  ).toThrow(/separate buffers/);
  expect(() =>
    create({
      output: {
        ...createOutput(graph, 'shared-out', 16),
        ids: graph.createDataView(shared, {format: 'uint32', length: 16}),
        count: graph.createDataView(shared, {format: 'uint32', length: 1})
      }
    })
  ).toThrow(/separate buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUNetworkPathExtraction(createProps(otherGraph)).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
