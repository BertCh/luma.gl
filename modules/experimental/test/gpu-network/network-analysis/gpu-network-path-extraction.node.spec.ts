// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getResourceBuffer} from '../network-resource-buffers';
import {expect, it} from 'vitest';
import {
  GPUNetworkPathExtraction,
  type GPUNetworkPathExtractionEdges,
  type GPUNetworkPathExtractionOutput,
  type GPUNetworkPathExtractionProps
} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createOutput(graph: GPUCommandGraph, prefix: string, capacity: number) {
  return {
    ids: createTransientView(graph, `${prefix}-ids`, 'uint32', capacity),
    status: {
      count: createTransientView(graph, `${prefix}-count`, 'uint32', 1),
      overflow: createTransientView(graph, `${prefix}-overflow`, 'uint32', 1)
    }
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

function createEdges(graph: GPUCommandGraph): GPUNetworkPathExtractionEdges {
  return {
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 10),
      weights: createTransientView(graph, 'weights', 'float32', 10)
    },
    output: createOutput(graph, 'edge-out', 16)
  };
}

/** Props with extra output views merged into the default output bag. */
function createPropsWithOutput(
  graph: GPUCommandGraph,
  output: Partial<GPUNetworkPathExtractionOutput>,
  overrides: Partial<GPUNetworkPathExtractionProps> = {}
): GPUNetworkPathExtractionProps {
  const props = createProps(graph, overrides);
  return {...props, output: {...props.output, ...output}};
}

it('GPUNetworkPathExtraction schedules nodes without and with edges', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUNetworkPathExtraction(createProps(graph));
  expect(contributor.id).toBe('network-path-extraction');
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
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
  expect(() => create({targetCount: 5})).toThrow(/targetCount must be an integer in \[0, 4\]/);
  const withOutput = (output: Partial<GPUNetworkPathExtractionOutput>) =>
    create({output: {...base.output, ...output}});
  expect(() =>
    withOutput({lineOffsets: createTransientView(graph, 'offsets4', 'uint32', 4)})
  ).toThrow(/output.lineOffsets/);
  expect(() =>
    withOutput({pathCosts: createTransientView(graph, 'path-costs', 'float32', 5)})
  ).toThrow(/output.pathCosts length/);
  expect(() =>
    withOutput({pathFound: createTransientView(graph, 'path-found', 'uint32', 5)})
  ).toThrow(/output.pathFound length/);
  for (const maximumPathLength of [0, 65537, 2.5]) {
    expect(() => create({maximumPathLength})).toThrow(/maximumPathLength/);
  }
  const edges = createEdges(graph);
  const withTopology = (topology: Partial<GPUNetworkPathExtractionEdges['topology']>) =>
    create({edges: {...edges, topology: {...edges.topology, ...topology}}});
  expect(() =>
    withTopology({offsets: createTransientView(graph, 'offsets8', 'uint32', 8)})
  ).toThrow(/edges.topology must contain one node per predecessors row/);
  expect(() =>
    withTopology({weights: createTransientView(graph, 'weights9', 'float32', 9)})
  ).toThrow(/edges topology.weights length/);
  expect(() => withTopology({weights: undefined as never})).toThrow(
    /edges topology.weights is required/
  );
  expect(() =>
    create({edges: {...edges, edgeIds: createTransientView(graph, 'edge-ids9', 'uint32', 9)}})
  ).toThrow(/edges.edgeIds/);
  expect(() =>
    create({
      edges: {
        ...edges,
        output: {
          ...edges.output,
          lineOffsets: createTransientView(graph, 'edge-offsets4', 'uint32', 4)
        }
      }
    })
  ).toThrow(/edges.output.lineOffsets/);
  expect(() =>
    create({costs: createTransientView(graph, 'uint-costs', 'uint32', 8) as never})
  ).toThrow(/costs/);
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      costs: graph.createDataView(shared, {format: 'float32', length: 8}),
      output: {
        ...base.output,
        pathCosts: graph.createDataView(shared, {format: 'float32', length: 4})
      }
    })
  ).toThrow(/separate buffers/);
  const sharedOutput = createOutput(graph, 'shared-out', 16);
  expect(() =>
    create({
      output: {
        ...sharedOutput,
        ids: graph.createDataView(shared, {format: 'uint32', length: 16}),
        status: {
          ...sharedOutput.status,
          count: graph.createDataView(shared, {format: 'uint32', length: 1})
        }
      }
    })
  ).toThrow(/separate buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUNetworkPathExtraction(createProps(otherGraph)).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUNetworkPathExtraction reports no path found when the search did not converge', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const converged = createTransientView(graph, 'converged', 'uint32', 1);
  const nodes = new GPUNetworkPathExtraction(
    createPropsWithOutput(
      graph,
      {
        pathFound: createTransientView(graph, 'found', 'uint32', 4),
        pathCosts: createTransientView(graph, 'path-costs', 'float32', 4)
      },
      {
        id: 'paths',
        predecessorsComplete: converged,
        targetCount: createTransientView(graph, 'target-count', 'uint32', 1)
      }
    )
  ).getCommandNodes(graph);
  // Review B-24: a NONE predecessor is a source only when the search converged, so the status
  // reset copies the flag and the measure pass requires it before reporting a path found.
  const status = nodes.find(node => node.id === 'paths-clear');
  expect(status?.resources?.map(resource => getResourceBuffer(resource))).toContain(converged);
  const sources = getParsedKernelSources(nodes);
  expect(sources.get('paths-measure')).toContain(
    'state == 1u && atomicLoad(&status[statusOffset + 1u]) != 0u'
  );
  const other = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    () =>
      new GPUNetworkPathExtraction(
        createProps(other, {predecessorsComplete: createTransientView(other, 'two', 'uint32', 2)})
      )
  ).toThrow(/predecessorsComplete must contain exactly one row/);
});

it('GPUNetworkPathExtraction bakes a numeric targetCount without a binding', () => {
  const device = createNullWebGPUDevice();
  const getSource = (targetCount: (graph: GPUCommandGraph) => number | GraphDataView<'uint32'>) => {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUNetworkPathExtraction(
      createProps(graph, {id: 'paths', targetCount: targetCount(graph)})
    ).getCommandNodes(graph);
    return getParsedKernelSources(nodes).get('paths-measure');
  };
  expect(getSource(() => 2)).toContain('index < 2u');
  expect(getSource(() => 2)).not.toContain('targetCount[');
  expect(getSource(graph => createTransientView(graph, 'target-count', 'uint32', 1))).toContain(
    'index < targetCount[targetCountOffset]'
  );
  device.destroy();
});
