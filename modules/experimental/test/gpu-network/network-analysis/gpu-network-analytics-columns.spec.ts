// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, GPUCOOToCSR} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {
  GPUGraph,
  GPUGraphConnectedComponents,
  GPUGraphCoreNumber,
  GPUGraphDegree,
  GPUGraphLabelPropagation,
  GPUGraphPageRank,
  GPUGraphTopology,
  type GPUGraphAdjacency
} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkAnalyticsColumns,
  type GPUNetworkAnalyticsColumnsProps
} from '../../../src/gpu-network/network-analysis/gpu-network-analytics-columns';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createRandomNetwork,
  F1_EDGES,
  type NetworkCSR,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';
import {
  buildReverseCSR,
  componentsOracle,
  coreNumberOracle,
  createGridEdges,
  createSymmetricEdges,
  degreeOracle,
  extentOracle,
  normalizeOracle,
  pageRankOracle
} from './network-analytics-oracle';

type Columns = {
  degree: number[];
  inDegree?: number[];
  pageRank: number[];
  coreNumber: number[];
  components: number[];
  communities: number[];
};

type FixtureOptions = {
  /** Directed edge list that becomes the forward CSR. */
  edges: readonly NetworkEdge[];
  nodeCount: number;
  directed?: boolean;
  normalize?: boolean;
  damping?: number;
  pageRankIterations?: number;
  iterations?: number;
};

type Fixture = {
  device: Device;
  graph: GPUCommandGraph;
  buffers: Buffer[];
  offsetsBuffer: Buffer;
  neighborsBuffer: Buffer;
  reverseOffsetsBuffer?: Buffer;
  reverseNeighborsBuffer?: Buffer;
  csr: NetworkCSR;
  reverse?: NetworkCSR;
  outputs: Record<string, Buffer>;
  recipe: GPUNetworkAnalyticsColumns;
  nodeCount: number;
};

function createFixture(device: Device, options: FixtureOptions): Fixture {
  const {nodeCount, edges} = options;
  const graph = new GPUCommandGraph(device, {id: 'analytics'});
  const csr = buildCSR(nodeCount, edges);
  const reverse = options.directed ? buildReverseCSR(nodeCount, csr) : undefined;
  const buffers: Buffer[] = [];
  const outputs: Record<string, Buffer> = {};
  const capacityOf = (values: Uint32Array) => (values.length ? values : new Uint32Array(1));
  const offsetsBuffer = createInputBuffer(device, csr.offsets);
  const neighborsBuffer = createInputBuffer(device, capacityOf(csr.neighbors));
  buffers.push(offsetsBuffer, neighborsBuffer);
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs[name] = buffer;
    return importGraphBuffer(graph, `analytics-${name}`, buffer, format, length);
  };
  const props: GPUNetworkAnalyticsColumnsProps = {
    id: 'analytics',
    offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', nodeCount + 1),
    neighbors: importGraphBuffer(
      graph,
      'neighbors',
      neighborsBuffer,
      'uint32',
      csr.neighbors.length
    ),
    degree: {
      output: output('degree', 'uint32', nodeCount),
      normalized: options.normalize ? output('degree-normalized', 'float32', nodeCount) : undefined,
      extent: options.normalize ? output('degree-extent', 'uint32', 2) : undefined
    },
    pageRank: {
      output: output('pageRank', 'float32', nodeCount),
      normalized: options.normalize
        ? output('pageRank-normalized', 'float32', nodeCount)
        : undefined,
      damping: options.damping,
      iterations: options.pageRankIterations,
      residual: output('pageRank-residual', 'float32', 1)
    },
    coreNumber: {
      output: output('coreNumber', 'uint32', nodeCount),
      normalized: options.normalize ? output('core-normalized', 'float32', nodeCount) : undefined,
      iterations: options.iterations,
      converged: output('core-converged', 'uint32', 1),
      degeneracy: output('core-degeneracy', 'uint32', 1)
    },
    components: {
      output: output('components', 'uint32', nodeCount),
      iterations: options.iterations,
      converged: output('components-converged', 'uint32', 1)
    },
    communities: {
      output: output('communities', 'uint32', nodeCount),
      iterations: options.iterations,
      converged: output('communities-converged', 'uint32', 1)
    }
  };
  let reverseOffsetsBuffer: Buffer | undefined;
  let reverseNeighborsBuffer: Buffer | undefined;
  if (reverse) {
    reverseOffsetsBuffer = createInputBuffer(device, reverse.offsets);
    reverseNeighborsBuffer = createInputBuffer(device, capacityOf(reverse.neighbors));
    buffers.push(reverseOffsetsBuffer, reverseNeighborsBuffer);
    props.reverseOffsets = importGraphBuffer(
      graph,
      'reverse-offsets',
      reverseOffsetsBuffer,
      'uint32',
      nodeCount + 1
    );
    props.reverseNeighbors = importGraphBuffer(
      graph,
      'reverse-neighbors',
      reverseNeighborsBuffer,
      'uint32',
      reverse.neighbors.length
    );
    props.inDegree = {output: output('inDegree', 'uint32', nodeCount)};
  }
  const recipe = new GPUNetworkAnalyticsColumns(props);
  graph.add(recipe);
  return {
    device,
    graph,
    buffers,
    offsetsBuffer,
    neighborsBuffer,
    reverseOffsetsBuffer,
    reverseNeighborsBuffer,
    csr,
    reverse,
    outputs,
    recipe,
    nodeCount
  };
}

function getOutput(fixture: Fixture, name: string): Buffer {
  return fixture.outputs[name];
}

function destroyFixture(fixture: Fixture): void {
  fixture.recipe.destroy();
  for (const buffer of fixture.buffers) buffer.destroy();
}

async function readColumns(fixture: Fixture): Promise<Columns> {
  const {nodeCount} = fixture;
  return {
    degree: await readUint32(getOutput(fixture, 'degree'), nodeCount),
    inDegree: getOutput(fixture, 'inDegree')
      ? await readUint32(getOutput(fixture, 'inDegree'), nodeCount)
      : undefined,
    pageRank: await readFloat32(getOutput(fixture, 'pageRank'), nodeCount),
    coreNumber: await readUint32(getOutput(fixture, 'coreNumber'), nodeCount),
    components: await readUint32(getOutput(fixture, 'components'), nodeCount),
    communities: await readUint32(getOutput(fixture, 'communities'), nodeCount)
  };
}

function expectPageRankClose(actual: number[], expected: ArrayLike<number>): void {
  expect(actual.length).toBe(expected.length);
  for (const [node, value] of actual.entries()) {
    expect(Math.abs(value - expected[node])).toBeLessThanOrEqual(1e-6 + 1e-4 * expected[node]);
  }
}

/** Runs the oracle for the fixture's CSR. */
function getExpected(fixture: Fixture, damping = 0.85, iterations = 40) {
  const {csr, reverse, nodeCount} = fixture;
  return {
    degree: Array.from(degreeOracle(csr.offsets)),
    inDegree: reverse ? Array.from(degreeOracle(reverse.offsets)) : undefined,
    pageRank: pageRankOracle(nodeCount, csr, reverse ?? csr, damping, iterations),
    coreNumber: Array.from(coreNumberOracle(nodeCount, csr, reverse)),
    components: Array.from(componentsOracle(nodeCount, csr))
  };
}

async function expectMatchesOracle(fixture: Fixture, damping?: number, iterations?: number) {
  const columns = await readColumns(fixture);
  const expected = getExpected(fixture, damping, iterations);
  expect(columns.degree).toEqual(expected.degree);
  expect(columns.inDegree).toEqual(expected.inDegree);
  expectPageRankClose(columns.pageRank, expected.pageRank);
  expect(columns.coreNumber).toEqual(expected.coreNumber);
  expect(columns.components).toEqual(expected.components);
  expect(await readUint32(fixture.outputs['components-converged'], 1)).toEqual([1]);
  expect(await readUint32(fixture.outputs['core-converged'], 1)).toEqual([1]);
  expect(await readUint32(fixture.outputs['core-degeneracy'], 1)).toEqual([
    Math.max(...expected.coreNumber)
  ]);
  return columns;
}

/** Builds the conventional GPUGraphTopology path from COO edges and runs the same algorithms. */
async function runConventional(
  device: Device,
  nodeCount: number,
  edges: readonly NetworkEdge[],
  directed: boolean,
  iterations: number,
  pageRankOptions: {damping?: number; iterations?: number} = {}
): Promise<Columns> {
  const buffers: Buffer[] = [];
  const createVector = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: ArrayLike<number> | number
  ): GPUVector<Format> => {
    const length = typeof values === 'number' ? values : values.length;
    const buffer = device.createBuffer({
      id: `conventional-${name}`,
      byteLength: Math.max(length, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    if (typeof values !== 'number' && length > 0) {
      buffer.write(Uint32Array.from(values));
    }
    buffers.push(buffer);
    return new GPUVector<Format>({type: 'buffer', name, buffer, format, length});
  };
  const sourceVertices = createVector(
    'source',
    'uint32',
    edges.map(edge => edge[0])
  );
  const targetVertices = createVector(
    'target',
    'uint32',
    edges.map(edge => edge[1])
  );
  const graph = new GPUGraph({vertexCount: nodeCount, sourceVertices, targetVertices, directed});
  const createAdjacency = (name: string, capacity: number): GPUGraphAdjacency => ({
    offsets: createVector(`${name}-offsets`, 'uint32', nodeCount + 1),
    neighbors: createVector(`${name}-neighbors`, 'uint32', capacity),
    edgeIds: createVector(`${name}-edge-ids`, 'uint32', capacity),
    count: createVector(`${name}-count`, 'uint32', 1),
    overflow: createVector(`${name}-overflow`, 'uint32', 1)
  });
  const topology = new GPUGraphTopology({
    graph,
    forward: createAdjacency('forward', directed ? edges.length : 2 * edges.length),
    reverse: directed ? createAdjacency('reverse', edges.length) : undefined,
    invalidEdgeCount: createVector('invalid', 'uint32', 1)
  });
  const degree = createVector('degree', 'uint32', nodeCount);
  const inDegree = directed ? createVector('in-degree', 'uint32', nodeCount) : undefined;
  const pageRank = createVector('page-rank', 'float32', nodeCount);
  const coreNumber = createVector('core-number', 'uint32', nodeCount);
  const components = createVector('components', 'uint32', nodeCount);
  const communities = createVector('communities', 'uint32', nodeCount);
  const commandGraph = new GPUCommandGraph(device, {id: 'conventional'});
  topology.addToGraph(commandGraph);
  new GPUGraphDegree({id: 'c-degree', topology, output: degree}).addToGraph(commandGraph);
  if (inDegree) {
    new GPUGraphDegree({
      id: 'c-in-degree',
      topology,
      output: inDegree,
      direction: 'incoming'
    }).addToGraph(commandGraph);
  }
  new GPUGraphPageRank({
    id: 'c-page-rank',
    topology,
    output: pageRank,
    ...pageRankOptions
  }).addToGraph(commandGraph);
  new GPUGraphCoreNumber({id: 'c-core', topology, output: coreNumber, iterations}).addToGraph(
    commandGraph
  );
  new GPUGraphConnectedComponents({
    id: 'c-components',
    topology,
    output: components,
    iterations
  }).addToGraph(commandGraph);
  new GPUGraphLabelPropagation({
    id: 'c-communities',
    topology,
    output: communities,
    iterations
  }).addToGraph(commandGraph);
  const compiled = commandGraph.compile();
  submitGraph(device, compiled, undefined);
  const read = async (vector: GPUVector<'uint32'> | GPUVector<'float32'> | undefined) => {
    if (!vector) return undefined;
    const bytes = await (vector.data[0].buffer as Buffer).readAsync();
    const values =
      vector.format === 'float32'
        ? new Float32Array(bytes.buffer, bytes.byteOffset, vector.length)
        : new Uint32Array(bytes.buffer, bytes.byteOffset, vector.length);
    return Array.from(values);
  };
  const result: Columns = {
    degree: (await read(degree))!,
    inDegree: await read(inDegree),
    pageRank: (await read(pageRank))!,
    coreNumber: (await read(coreNumber))!,
    components: (await read(components))!,
    communities: (await read(communities))!
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

/** Undirected COO list: each road once. */
function getRoadList(symmetricEdges: readonly NetworkEdge[]): NetworkEdge[] {
  return symmetricEdges.filter(edge => edge[0] < edge[1]);
}

async function expectMatchesConventional(
  fixture: Fixture,
  edges: readonly NetworkEdge[],
  directed: boolean,
  iterations: number,
  pageRankOptions: {damping?: number; iterations?: number} = {}
) {
  const columns = await readColumns(fixture);
  const conventional = await runConventional(
    fixture.device,
    fixture.nodeCount,
    edges,
    directed,
    iterations,
    pageRankOptions
  );
  expect(columns.degree).toEqual(conventional.degree);
  expect(columns.inDegree).toEqual(conventional.inDegree);
  expectPageRankClose(columns.pageRank, conventional.pageRank);
  expect(columns.coreNumber).toEqual(conventional.coreNumber);
  expect(columns.components).toEqual(conventional.components);
  expect(columns.communities).toEqual(conventional.communities);
}

it('GPUNetworkAnalyticsColumns matches the oracle and conventional topology on an undirected grid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const edges = createGridEdges(6, 6);
  const fixture = createFixture(device, {nodeCount: 36, edges, normalize: true, iterations: 64});
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const columns = await expectMatchesOracle(fixture);
  // Corners 2, edges 3, interior 4.
  expect(columns.degree[0]).toBe(2);
  expect(columns.degree[1]).toBe(3);
  expect(columns.degree[7]).toBe(4);
  expect(columns.components).toEqual(new Array(36).fill(0));
  expect(columns.coreNumber).toEqual(new Array(36).fill(2));
  expect(await readUint32(fixture.outputs['degree-extent'], 2)).toEqual([2, 4]);
  expect(await readFloat32(fixture.outputs['degree-normalized'], 36)).toEqual(
    Array.from(normalizeOracle(columns.degree))
  );
  const [pageRankLow, pageRankHigh] = extentOracle(columns.pageRank);
  const pageRankNormalized = await readFloat32(fixture.outputs['pageRank-normalized'], 36);
  const expectedPageRank = normalizeOracle(columns.pageRank);
  expect(Math.min(...pageRankNormalized)).toBe(0);
  expect(Math.max(...pageRankNormalized)).toBeCloseTo(1, 6);
  expect(pageRankHigh).toBeGreaterThan(pageRankLow);
  for (const [node, value] of pageRankNormalized.entries()) {
    expect(Math.abs(value - expectedPageRank[node])).toBeLessThan(1e-5);
  }
  expect(await readFloat32(fixture.outputs['core-normalized'], 36)).toEqual(new Array(36).fill(0));
  const [residual] = await readFloat32(fixture.outputs['pageRank-residual'], 1);
  expect(residual).toBeGreaterThanOrEqual(0);
  expect(residual).toBeLessThan(1e-3);
  await expectMatchesConventional(fixture, getRoadList(edges), false, 64);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkAnalyticsColumns handles a directed network with a reverse CSR', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [nodeCount, edges] of [
    [8, F1_EDGES],
    [300, createRandomNetwork(11, 300, 900)]
  ] as const) {
    const fixture = createFixture(device, {nodeCount, edges, directed: true, iterations: 128});
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    await expectMatchesOracle(fixture);
    await expectMatchesConventional(fixture, edges, true, 128);
    compiled.destroy();
    destroyFixture(fixture);
  }
});

it('GPUNetworkAnalyticsColumns handles a random symmetric network with custom PageRank options', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 600;
  const edges = createSymmetricEdges(createRandomNetwork(5, nodeCount, 700));
  const fixture = createFixture(device, {
    nodeCount,
    edges,
    iterations: 256,
    damping: 0.7,
    pageRankIterations: 25
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const columns = await expectMatchesOracle(fixture, 0.7, 25);
  // The sparse random graph has isolated nodes (dangling mass) and several components.
  expect(new Set(columns.components).size).toBeGreaterThan(1);
  expect(columns.degree).toContain(0);
  await expectMatchesConventional(fixture, getRoadList(edges), false, 256, {
    damping: 0.7,
    iterations: 25
  });
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkAnalyticsColumns normalizes a constant column to zeros', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A ring: every node has degree 2.
  const nodeCount = 10;
  const edges: NetworkEdge[] = [];
  for (let node = 0; node < nodeCount; node++) {
    const next = (node + 1) % nodeCount;
    edges.push([node, next, 1], [next, node, 1]);
  }
  const fixture = createFixture(device, {nodeCount, edges, normalize: true});
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(getOutput(fixture, 'degree'), nodeCount)).toEqual(new Array(10).fill(2));
  expect(await readUint32(fixture.outputs['degree-extent'], 2)).toEqual([2, 2]);
  expect(await readFloat32(fixture.outputs['degree-normalized'], nodeCount)).toEqual(
    new Array(10).fill(0)
  );
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkAnalyticsColumns handles single-node and edgeless networks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const nodeCount of [1, 5]) {
    const fixture = createFixture(device, {nodeCount, edges: [], normalize: true});
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const columns = await readColumns(fixture);
    expect(columns.degree).toEqual(new Array(nodeCount).fill(0));
    expect(columns.coreNumber).toEqual(new Array(nodeCount).fill(0));
    expect(columns.components).toEqual(Array.from({length: nodeCount}, (_, node) => node));
    expect(columns.communities).toEqual(Array.from({length: nodeCount}, (_, node) => node));
    expectPageRankClose(columns.pageRank, new Array(nodeCount).fill(1 / nodeCount));
    expect(await readFloat32(fixture.outputs['degree-normalized'], nodeCount)).toEqual(
      new Array(nodeCount).fill(0)
    );
    compiled.destroy();
    destroyFixture(fixture);
  }
});

it('GPUNetworkAnalyticsColumns recomputes after the CSR changes without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Two 4-node paths joined by the road 3-4.
  const roads: [number, number][] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
  ];
  const toEdges = (list: [number, number][]) =>
    list.flatMap(([from, to]): NetworkEdge[] => [
      [from, to, 1],
      [to, from, 1]
    ]);
  const fixture = createFixture(device, {nodeCount: 8, edges: toEdges(roads), iterations: 64});
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  let columns = await expectMatchesOracle(fixture);
  expect(columns.components).toEqual(new Array(8).fill(0));
  expect(columns.degree[3]).toBe(2);

  // Close road 3-4: same buffers and capacity, new contents.
  const closed = buildCSR(8, toEdges(roads.filter(road => road[0] !== 3)));
  fixture.offsetsBuffer.write(closed.offsets);
  const padded = new Uint32Array(fixture.csr.neighbors.length);
  padded.set(closed.neighbors);
  fixture.neighborsBuffer.write(padded);
  submitGraph(device, compiled, undefined);
  columns = await readColumns(fixture);
  expect(columns.components).toEqual([0, 0, 0, 0, 4, 4, 4, 4]);
  expect(columns.degree[3]).toBe(1);
  expect(columns.degree[4]).toBe(1);
  expect(columns.components).toEqual(Array.from(componentsOracle(8, closed)));
  expectPageRankClose(columns.pageRank, pageRankOracle(8, closed, closed));
  expect(columns.coreNumber).toEqual(Array.from(coreNumberOracle(8, closed)));

  // Reopen it.
  fixture.offsetsBuffer.write(fixture.csr.offsets);
  fixture.neighborsBuffer.write(fixture.csr.neighbors);
  submitGraph(device, compiled, undefined);
  columns = await readColumns(fixture);
  expect(columns.components).toEqual(new Array(8).fill(0));
  compiled.destroy();
  destroyFixture(fixture);
});

/** Builds a CSR in-graph with `GPUCOOToCSR` from a row-sorted COO edge list. */
function addTransientCSR(
  device: Device,
  graph: GPUCommandGraph,
  name: string,
  nodeCount: number,
  edges: readonly NetworkEdge[],
  buffers: Buffer[]
) {
  const sorted = [...edges].sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  const edgeCount = sorted.length;
  const rows = createInputBuffer(device, Uint32Array.from(sorted.map(edge => edge[0])));
  const columns = createInputBuffer(device, Uint32Array.from(sorted.map(edge => edge[1])));
  const values = createInputBuffer(device, Float32Array.from(sorted.map(edge => edge[2])));
  buffers.push(rows, columns, values);
  const offsets = createTransientView(graph, `${name}-offsets`, 'uint32', nodeCount + 1);
  const neighbors = createTransientView(graph, `${name}-neighbors`, 'uint32', edgeCount);
  graph.add(
    new GPUCOOToCSR({
      id: `${name}-coo-to-csr`,
      rows: nodeCount,
      rowIndices: importGraphBuffer(graph, `${name}-rows`, rows, 'uint32', edgeCount),
      columnIndices: importGraphBuffer(graph, `${name}-columns`, columns, 'uint32', edgeCount),
      values: importGraphBuffer(graph, `${name}-values`, values, 'float32', edgeCount),
      rowOffsets: offsets,
      outputColumnIndices: neighbors,
      outputValues: createTransientView(graph, `${name}-csr-values`, 'float32', edgeCount)
    })
  );
  return {offsets, neighbors, csr: buildCSR(nodeCount, sorted)};
}

it('GPUNetworkAnalyticsColumns analyzes transient CSRs built by GPUCOOToCSR', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const directed of [false, true]) {
    const nodeCount = 30;
    // A 5x5 grid plus a separate 5-node path, so components and cores differ.
    const roads = createSymmetricEdges([
      ...getRoadList(createGridEdges(5, 5)),
      [25, 26, 1],
      [26, 27, 1],
      [27, 28, 1],
      [28, 29, 1]
    ]);
    const edges = directed ? createRandomNetwork(11, nodeCount, 90) : roads;
    const buffers: Buffer[] = [];
    const graph = new GPUCommandGraph(device, {id: `transient-analytics-${directed}`});
    const forward = addTransientCSR(device, graph, 'forward', nodeCount, edges, buffers);
    const reverseEdges = edges.map(([from, to, weight]) => [to, from, weight] as NetworkEdge);
    const reverse = directed
      ? addTransientCSR(device, graph, 'reverse', nodeCount, reverseEdges, buffers)
      : undefined;
    const outputs: Record<string, Buffer> = {};
    const output = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      outputs[name] = buffer;
      return importGraphBuffer(graph, `out-${name}`, buffer, format, length);
    };
    graph.add(
      new GPUNetworkAnalyticsColumns({
        id: 'transient-analytics',
        offsets: forward.offsets,
        neighbors: forward.neighbors,
        reverseOffsets: reverse?.offsets,
        reverseNeighbors: reverse?.neighbors,
        degree: {
          output: output('degree', 'uint32', nodeCount),
          // A transient raw column still feeds the normalized output.
          normalized: output('degree-normalized', 'float32', nodeCount)
        },
        inDegree: reverse ? {output: output('inDegree', 'uint32', nodeCount)} : undefined,
        pageRank: {output: output('pageRank', 'float32', nodeCount)},
        coreNumber: {
          output: createTransientView(graph, 'core-scratch', 'uint32', nodeCount),
          normalized: output('core-normalized', 'float32', nodeCount),
          iterations: 64,
          converged: output('core-converged', 'uint32', 1)
        },
        components: {
          output: output('components', 'uint32', nodeCount),
          iterations: 64,
          converged: output('components-converged', 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const degree = degreeOracle(forward.csr.offsets);
    expect(await readUint32(outputs.degree, nodeCount)).toEqual(Array.from(degree));
    const normalized = await readFloat32(outputs['degree-normalized'], nodeCount);
    const expectedNormalized = normalizeOracle(degree);
    for (const [node, value] of normalized.entries()) {
      expect(value).toBeCloseTo(expectedNormalized[node], 6);
    }
    if (reverse) {
      expect(await readUint32(outputs.inDegree, nodeCount)).toEqual(
        Array.from(degreeOracle(reverse.csr.offsets))
      );
    }
    expectPageRankClose(
      await readFloat32(outputs.pageRank, nodeCount),
      pageRankOracle(nodeCount, forward.csr, reverse?.csr ?? forward.csr, 0.85, 40)
    );
    const cores = coreNumberOracle(nodeCount, forward.csr, reverse?.csr);
    const coreNormalized = await readFloat32(outputs['core-normalized'], nodeCount);
    const expectedCoreNormalized = normalizeOracle(cores);
    for (const [node, value] of coreNormalized.entries()) {
      expect(value).toBeCloseTo(expectedCoreNormalized[node], 6);
    }
    expect(await readUint32(outputs['core-converged'], 1)).toEqual([1]);
    expect(await readUint32(outputs.components, nodeCount)).toEqual(
      Array.from(componentsOracle(nodeCount, forward.csr))
    );
    expect(await readUint32(outputs['components-converged'], 1)).toEqual([1]);
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
