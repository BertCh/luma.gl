// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, ComputePipeline} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {
  GPUGraph,
  GPUGraphTopology,
  GPUGraphTopologyView,
  GPUGraphDegree,
  GPUGraphBreadthFirstSearch,
  GPUGraphSingleSourceShortestPath,
  GPUGraphConnectedComponents,
  GPUGraphCoreNumber,
  GPUGraphLabelPropagation,
  GPUGraphLocalClusteringCoefficient,
  GPUGraphModularity,
  GPUGraphModularityOptimization,
  GPUGraphPageRank,
  GPUGraphForceLayout,
  GPUGraphSpatialForceLayout
} from '@luma.gl/gpgpu/gpu-graph';
import {NullDevice} from '@luma.gl/test-utils';
import {afterEach, expect, test, vi} from 'vitest';
import * as computationHash from '../../src/computation-record/hash-computation-text';
import type {CompiledNode} from '../../src/gpu-core/gpu-command-graph-compiler';
import {getGPUGraphPassAnnotation} from '../../src/gpu-graph/gpu-graph-computation-record';

const cleanups: (() => void)[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const cleanup of cleanups.splice(0).reverse()) cleanup();
});

class RecordingPipeline extends ComputePipeline {
  setBindings(): void {}
}

function createFixture(recording: boolean) {
  const device = new NullDevice({id: 'recording-device'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxStorageBuffersPerShaderStage: 8
    }
  });
  cleanups.push(() => device.destroy());
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => new RecordingPipeline(device, props!)
  );
  const commandGraph = new GPUCommandGraph(device, {
    id: 'recording-graph',
    ...(recording && {recording: {}})
  });
  function vector<Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number,
    values?: number[]
  ): GPUVector<Format> {
    const components = format === 'float32x2' ? 2 : 1;
    // Exercise nonzero binding element offsets, including two-component layout columns.
    const data =
      format === 'uint32' ? new Uint32Array(length + 1) : new Float32Array(length * components + 1);
    if (values) data.set(values, 1);
    const buffer = device.createBuffer({
      id: name,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX
    });
    const result = new GPUVector({
      type: 'data',
      name,
      format,
      data: [new GPUData({buffer, format, length, byteOffset: 4, ownsBuffer: false})]
    });
    cleanups.push(
      () => buffer.destroy(),
      () => result.destroy()
    );
    return result;
  }
  const graph = new GPUGraph({
    vertexCount: 6,
    directed: true,
    sourceVertices: vector('sources', 'uint32', 6, [0, 0, 1, 2, 3, 4]),
    targetVertices: vector('targets', 'uint32', 6, [1, 2, 2, 3, 4, 5]),
    edgeWeights: vector('weights', 'float32', 6, [1, 2, 1, 1, 2, 1])
  });
  const adjacency = (name: string, reverse: boolean) => {
    const offsets = reverse ? [0, 0, 1, 3, 4, 5, 6] : [0, 2, 3, 4, 5, 6, 6];
    const neighbors = reverse ? [0, 0, 1, 2, 3, 4] : [1, 2, 2, 3, 4, 5];
    return {
      offsets: vector(`${name}-offsets`, 'uint32', 7, offsets),
      neighbors: vector(`${name}-neighbors`, 'uint32', 6, neighbors),
      edgeIds: vector(`${name}-edge-ids`, 'uint32', 6, [0, 1, 2, 3, 4, 5]),
      edgeWeights: vector(`${name}-weights`, 'float32', 6, [1, 2, 1, 1, 2, 1]),
      count: vector(`${name}-count`, 'uint32', 1, [6]),
      overflow: vector(`${name}-overflow`, 'uint32', 1)
    };
  };
  const topology = new GPUGraphTopology({
    id: 'contributor',
    graph,
    forward: adjacency('forward', false),
    reverse: adjacency('reverse', true),
    invalidEdgeCount: vector('invalid-edges', 'uint32', 1)
  });
  const layout = () =>
    new GPUGraphForceLayout({
      id: 'contributor',
      topology,
      iterationsPerFrame: 2,
      positions: vector('positions', 'float32x2', 6, [0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 3, 2]),
      velocities: vector('velocities', 'float32x2', 6),
      reset: vector('reset', 'uint32', 1),
      pinned: vector('pinned', 'uint32', 6)
    });
  return {device, commandGraph, vector, graph, topology, layout};
}

type Fixture = ReturnType<typeof createFixture>;
type Contributor = {id: string; addToGraph(graph: GPUCommandGraph): void};
type Case = {operation: string; iterative?: boolean; create: (fixture: Fixture) => Contributor};
const cases: Case[] = [
  {operation: 'GPUGraphTopology', create: ({topology}) => topology},
  {
    operation: 'GPUGraphDegree',
    create: ({topology, vector}) =>
      new GPUGraphDegree({id: 'contributor', topology, output: vector('degree', 'uint32', 6)})
  },
  {
    operation: 'GPUGraphBreadthFirstSearch',
    iterative: true,
    create: ({topology, vector}) =>
      new GPUGraphBreadthFirstSearch({
        id: 'contributor',
        topology,
        seeds: vector('seeds', 'uint32', 1, [0]),
        distances: vector('distances', 'uint32', 6),
        predecessors: vector('predecessors', 'uint32', 6),
        maxDepth: 2,
        direction: 'both'
      })
  },
  {
    operation: 'GPUGraphSingleSourceShortestPath',
    iterative: true,
    create: ({topology, vector}) =>
      new GPUGraphSingleSourceShortestPath({
        id: 'contributor',
        topology,
        sourceVertex: 0,
        distances: vector('distances', 'float32', 6),
        predecessors: vector('predecessors', 'uint32', 6),
        maxIterations: 2
      })
  },
  ...[GPUGraphConnectedComponents, GPUGraphCoreNumber, GPUGraphLabelPropagation].map(
    ContributorClass => ({
      operation: ContributorClass.name,
      iterative: true,
      create: ({topology, vector}: Fixture) =>
        new ContributorClass({
          id: 'contributor',
          topology,
          output: vector('labels', 'uint32', 6),
          converged: vector('converged', 'uint32', 1),
          iterations: 2
        })
    })
  ),
  ...(['direct', 'canonical'] as const).map(algorithm => ({
    operation: 'GPUGraphLocalClusteringCoefficient',
    create: ({topology, vector}: Fixture) =>
      new GPUGraphLocalClusteringCoefficient({
        id: 'contributor',
        topology,
        output: vector('clustering', 'float32', 6),
        algorithm
      })
  })),
  {
    operation: 'GPUGraphModularity',
    create: ({graph, vector}) =>
      new GPUGraphModularity({
        id: 'contributor',
        graph,
        communities: vector('communities', 'uint32', 6),
        output: vector('modularity', 'float32', 1)
      })
  },
  {
    operation: 'GPUGraphModularityOptimization',
    iterative: true,
    create: ({topology, vector}) =>
      new GPUGraphModularityOptimization({
        id: 'contributor',
        topology,
        output: vector('communities', 'uint32', 6),
        modularity: vector('modularity', 'float32', 1),
        iterations: 2
      })
  },
  {
    operation: 'GPUGraphPageRank',
    iterative: true,
    create: ({topology, vector}) =>
      new GPUGraphPageRank({
        id: 'contributor',
        topology,
        output: vector('ranks', 'float32', 6),
        residual: vector('residual', 'float32', 1),
        iterations: 2
      })
  },
  {operation: 'GPUGraphForceLayout', iterative: true, create: ({layout}) => layout()},
  {
    operation: 'GPUGraphSpatialForceLayout',
    iterative: true,
    create: ({layout, vector}) =>
      new GPUGraphSpatialForceLayout({
        id: 'contributor',
        layout: layout(),
        gridSize: [2, 2],
        bounds: [-1, -1, 4, 4],
        cellOffsets: vector('cell-offsets', 'uint32', 5),
        vertexIds: vector('vertex-ids', 'uint32', 6),
        cellCenters: vector('cell-centers', 'float32x2', 4),
        count: vector('count', 'uint32', 1),
        overflow: vector('overflow', 'uint32', 1)
      })
  }
];

function build(recording: boolean, create: Case['create']) {
  const fixture = createFixture(recording);
  const shaders = vi.spyOn(fixture.device, 'createShader');
  const contributor = create(fixture);
  contributor.addToGraph(fixture.commandGraph);
  const compiled = fixture.commandGraph.compile();
  cleanups.push(() => compiled.destroy());
  const nodes = Reflect.get(compiled, 'compiledNodes') as CompiledNode<void>[];
  return {
    compiled,
    nodes,
    description: compiled.describe(),
    shaderSources: shaders.mock.calls.map(([shader]) => shader.source)
  };
}

test.each(cases)('$operation records kernels without changing the compiled graph', ({
  operation,
  iterative,
  create
}) => {
  const hash = vi.spyOn(computationHash, 'hashGPUComputationText');
  const production = build(false, create);
  expect(hash).not.toHaveBeenCalled();
  const recorded = build(true, create);
  expect(recorded.compiled.stats).toEqual(production.compiled.stats);
  expect(recorded.shaderSources).toEqual(production.shaderSources);
  expect(recorded.description.graph.resources).toEqual(production.description.graph.resources);
  expect(
    recorded.description.graph.nodes.map(({id, reads, writes, condition}) => ({
      id,
      reads,
      writes,
      condition
    }))
  ).toEqual(
    production.description.graph.nodes.map(({id, reads, writes, condition}) => ({
      id,
      reads,
      writes,
      condition
    }))
  );
  expect(production.nodes.every(({node}) => !Object.hasOwn(node, 'annotation'))).toBe(true);
  expect(production.description.sources).toEqual([]);
  const sources = new Map(recorded.description.sources.map(source => [source.id, source]));
  const resources = new Map(
    recorded.description.graph.resources?.map(resource => [resource.id, resource])
  );
  // Exclude gpu-core scan/sort/reduction/index nodes; contributor scopes still include them.
  const nodes = recorded.description.graph.nodes.filter(
    node =>
      node.kind === 'compute' &&
      (node.namespace === 'contributor' ||
        node.namespace === 'contributor/contributor-final-modularity') &&
      !/-index-|-(forward|reverse)-scan/.test(node.id)
  );
  expect(nodes.length).toBeGreaterThan(0);
  for (const node of nodes) {
    // Modularity optimization nests a final modularity contributor with its own namespace/op.
    expect(node.op).toBe(
      node.namespace.includes('/contributor-final-modularity') ? 'GPUGraphModularity' : operation
    );
    expect(node.namespace.startsWith('contributor')).toBe(true);
    const source = sources.get(node.source!);
    expect(source?.text).toContain('@compute');
    expect(source?.moduleId).toBe(node.op);
    expect(source?.hash).toBe(computationHash.hashGPUComputationText(source!.text!));
    expect(node.workgroupSize).toEqual([
      Number(source!.text!.match(/@workgroup_size\((\d+)\)/)![1]),
      1,
      1
    ]);
    expect(node.dispatch?.workgroups).toHaveLength(3);
    for (const [guard, keySpace] of [
      ['EDGE_COUNT', 'edge'],
      ['CELL_COUNT', 'cell'],
      ['VERTEX_COUNT', 'vertex']
    ]) {
      if (source!.text!.includes(`if (index >= ${guard}`))
        expect(node.datum?.keySpace).toBe(keySpace);
    }
    expect(node.bindings?.length).toBeGreaterThan(0);
    for (const binding of node.bindings!) {
      expect(resources.has(binding.resource)).toBe(true);
      expect(source!.text).toContain(` ${binding.name}:`);
      expect(['read', 'read_write']).toContain(binding.access);
      expect(['u32', 'i32', 'f32']).toContain(binding.elementType);
      expect(binding.elementOffset).toBe(
        resources.get(binding.resource)?.ownership === 'imported' ? 1 : 0
      );
      expect(binding.length).toBeGreaterThan(0);
    }
  }
  expect(nodes.some(node => node.datum?.keySpace === 'vertex')).toBe(true);
  if (iterative) {
    const loopNodes = nodes.filter(node => node.loop);
    expect(new Set(loopNodes.map(node => node.loop!.iteration))).toEqual(new Set([0, 1]));
    for (const node of loopNodes) {
      expect(node.loop).toMatchObject({
        id: `contributor-${operation === 'GPUGraphBreadthFirstSearch' ? 'depths' : 'iterations'}`,
        maximumIterations: 2,
        phase: operation === 'GPUGraphBreadthFirstSearch' ? 'gated' : 'unconditional'
      });
    }
  }
});

test('exact topology views annotate clear passes and preserve nested instance namespaces', () => {
  const {commandGraph, topology, vector} = createFixture(true);
  const importAdjacency = (name: string, adjacency: typeof topology.forward) => ({
    offsets: commandGraph.importGPUVector(`${name}-offsets`, adjacency.offsets).data[0],
    neighbors: commandGraph.importGPUVector(`${name}-neighbors`, adjacency.neighbors).data[0]
  });
  const view = new GPUGraphTopologyView({
    vertexCount: 6,
    directed: true,
    forward: importAdjacency('forward', topology.forward),
    reverse: importAdjacency('reverse', topology.reverse!)
  });
  commandGraph.withRecordScope({namespace: 'outer'}, () => {
    for (const id of ['first', 'second'])
      new GPUGraphPageRank({
        id,
        topology: view,
        output: vector(`${id}-output`, 'float32', 6),
        iterations: 2
      }).addToGraph(commandGraph);
  });
  const compiled = commandGraph.compile();
  cleanups.push(() => compiled.destroy());
  const {graph, sources} = compiled.describe();
  const clears = graph.nodes.filter(node => node.id.endsWith('-exact-clear'));
  expect(clears).toHaveLength(4);
  for (const node of clears) {
    expect(node.op).toBe('GPUGraphPageRank');
    expect(node.namespace).toBe(`outer/${node.id.startsWith('first') ? 'first' : 'second'}`);
    expect(node.workgroupSize).toEqual([1, 1, 1]);
    expect(node.dispatch).toMatchObject({workgroups: [1, 1, 1], invocationCount: 1});
    expect(node.datum).toBeUndefined();
    expect(sources.find(source => source.id === node.source)?.text).toContain('@compute');
  }
});

test('annotation helper returns before reading source or bindings when recording is off', () => {
  const {commandGraph} = createFixture(false);
  expect(
    getGPUGraphPassAnnotation(commandGraph, {
      operation: 'test',
      workgroupSize: 1,
      dispatch: {x: 1, y: 1, z: 1},
      get source(): string {
        throw new Error('source must not be read');
      },
      get bindings(): never {
        throw new Error('bindings must not be read');
      }
    })
  ).toBeUndefined();
});
