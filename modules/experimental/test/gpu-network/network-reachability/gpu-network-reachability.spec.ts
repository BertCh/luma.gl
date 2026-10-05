// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, GPUCOOToCSR} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkReachability,
  type GPUNetworkReachabilityProps
} from '../../../src/gpu-network/network-reachability';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  bandOracle,
  buildCSR,
  checkPredecessorWalks,
  createGridNetwork,
  createPathNetwork,
  createRandomNetwork,
  dijkstra,
  F1_EDGES,
  NONE,
  predecessorOracle,
  tieLevelOracle,
  type NetworkCSR,
  type NetworkEdge
} from './network-reachability-oracle';

type ReachabilityOptions = {
  sources: number[];
  sourceCosts?: number[];
  sourceCount?: number;
  thresholds?: number[];
  maxIterations?: number;
  localIterations?: number;
  maxTieIterations?: number;
  activeIterations?: number;
  costLimit?: number;
  id?: string;
};

type ReachabilityFixture = {
  graph: GPUCommandGraph;
  nodeCount: number;
  ownedBuffers: Buffer[];
  parameters: GPUParameterBuffer[];
  weightsBuffer: Buffer;
  sources: GPUParameterBuffer<'uint32'>;
  sourceCosts?: GPUParameterBuffer<'float32'>;
  sourceCount?: GPUParameterBuffer<'uint32'>;
  thresholds?: GPUParameterBuffer<'float32'>;
  activeIterations?: GPUParameterBuffer<'uint32'>;
  costLimit?: GPUParameterBuffer<'float32'>;
  costsBuffer: Buffer;
  predecessorsBuffer: Buffer;
  bandsBuffer?: Buffer;
  bandCountsBuffer?: Buffer;
  convergedBuffer: Buffer;
  iterationCountBuffer: Buffer;
  props: GPUNetworkReachabilityProps;
};

function createReachabilityFixture(
  device: Device,
  csr: NetworkCSR,
  nodeCount: number,
  options: ReachabilityOptions,
  graph: GPUCommandGraph = new GPUCommandGraph(device, {id: 'reachability'})
): ReachabilityFixture {
  const prefix = options.id ?? 'reach';
  const offsetsBuffer = createInputBuffer(device, csr.offsets);
  const neighborsBuffer = createInputBuffer(
    device,
    csr.neighbors.length ? csr.neighbors : new Uint32Array(1)
  );
  const weightsBuffer = createInputBuffer(
    device,
    csr.weights.length ? csr.weights : new Float32Array(1)
  );
  const costsBuffer = createOutputBuffer(device, nodeCount);
  const predecessorsBuffer = createOutputBuffer(device, nodeCount);
  const convergedBuffer = createOutputBuffer(device, 1);
  const iterationCountBuffer = createOutputBuffer(device, 1);
  const parameters: GPUParameterBuffer[] = [];
  const createParameter = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: number[]
  ) => {
    const parameter = new GPUParameterBuffer(device, {
      id: `${prefix}-${name}`,
      format,
      length: values.length,
      values: format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    });
    parameters.push(parameter);
    return parameter;
  };
  const sources = createParameter('sources', 'uint32', options.sources);
  const sourceCosts = options.sourceCosts
    ? createParameter('source-costs', 'float32', options.sourceCosts)
    : undefined;
  const sourceCount =
    options.sourceCount !== undefined
      ? createParameter('source-count', 'uint32', [options.sourceCount])
      : undefined;
  const thresholds = options.thresholds
    ? createParameter('thresholds', 'float32', options.thresholds)
    : undefined;
  const activeIterations =
    options.activeIterations !== undefined
      ? createParameter('active-iterations', 'uint32', [options.activeIterations])
      : undefined;
  const costLimit =
    options.costLimit !== undefined
      ? createParameter('cost-limit', 'float32', [options.costLimit])
      : undefined;
  const bandsBuffer = thresholds ? createOutputBuffer(device, nodeCount) : undefined;
  const bandCountsBuffer = thresholds
    ? createOutputBuffer(device, options.thresholds!.length)
    : undefined;
  const edgeCount = csr.neighbors.length;
  const props: GPUNetworkReachabilityProps = {
    id: prefix,
    offsets: importGraphBuffer(graph, `${prefix}-offsets`, offsetsBuffer, 'uint32', nodeCount + 1),
    neighbors: importGraphBuffer(
      graph,
      `${prefix}-neighbors`,
      neighborsBuffer,
      'uint32',
      edgeCount
    ),
    weights: importGraphBuffer(graph, `${prefix}-weights`, weightsBuffer, 'float32', edgeCount),
    sources: sources.importToGraph(graph),
    sourceCosts: sourceCosts?.importToGraph(graph),
    sourceCount: sourceCount?.importToGraph(graph),
    maxIterations: options.maxIterations ?? 16,
    localIterations: options.localIterations,
    maxTieIterations: options.maxTieIterations,
    activeIterations: activeIterations?.importToGraph(graph),
    costLimit: costLimit?.importToGraph(graph),
    costs: importGraphBuffer(graph, `${prefix}-costs`, costsBuffer, 'float32', nodeCount),
    predecessors: importGraphBuffer(
      graph,
      `${prefix}-predecessors`,
      predecessorsBuffer,
      'uint32',
      nodeCount
    ),
    bandThresholds: thresholds?.importToGraph(graph),
    bands: bandsBuffer
      ? importGraphBuffer(graph, `${prefix}-bands`, bandsBuffer, 'uint32', nodeCount)
      : undefined,
    bandCounts: bandCountsBuffer
      ? importGraphBuffer(
          graph,
          `${prefix}-band-counts`,
          bandCountsBuffer,
          'uint32',
          options.thresholds!.length
        )
      : undefined,
    converged: importGraphBuffer(graph, `${prefix}-converged`, convergedBuffer, 'uint32', 1),
    iterationCount: importGraphBuffer(
      graph,
      `${prefix}-iterations`,
      iterationCountBuffer,
      'uint32',
      1
    )
  };
  return {
    graph,
    nodeCount,
    ownedBuffers: [
      offsetsBuffer,
      neighborsBuffer,
      weightsBuffer,
      costsBuffer,
      predecessorsBuffer,
      convergedBuffer,
      iterationCountBuffer,
      ...(bandsBuffer ? [bandsBuffer] : []),
      ...(bandCountsBuffer ? [bandCountsBuffer] : [])
    ],
    parameters,
    weightsBuffer,
    sources,
    sourceCosts,
    sourceCount,
    thresholds,
    activeIterations,
    costLimit,
    costsBuffer,
    predecessorsBuffer,
    bandsBuffer,
    bandCountsBuffer,
    convergedBuffer,
    iterationCountBuffer,
    props
  };
}

function destroyFixture(fixture: ReachabilityFixture): void {
  for (const parameter of fixture.parameters) parameter.destroy();
  for (const buffer of fixture.ownedBuffers) buffer.destroy();
}

async function readCosts(fixture: ReachabilityFixture): Promise<number[]> {
  return readFloat32(fixture.costsBuffer, fixture.nodeCount);
}

const F1 = buildCSR(8, F1_EDGES);

it('GPUNetworkReachability computes costs, predecessors, and bands on a cyclic network', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createReachabilityFixture(device, F1, 8, {
    sources: [0],
    thresholds: [2, 5]
  });
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readCosts(fixture)).toEqual([0, 3, 1, 4, 7, Infinity, Infinity, Infinity]);
  expect(await readUint32(fixture.predecessorsBuffer, 8)).toEqual([
    NONE,
    2,
    0,
    1,
    3,
    NONE,
    NONE,
    NONE
  ]);
  expect(await readUint32(fixture.bandsBuffer!, 8)).toEqual([0, 1, 0, 1, NONE, NONE, NONE, NONE]);
  expect(await readUint32(fixture.bandCountsBuffer!, 2)).toEqual([2, 2]);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);
  const [iterations] = await readUint32(fixture.iterationCountBuffer, 1);
  expect(iterations).toBeGreaterThanOrEqual(1);
  expect(iterations).toBeLessThanOrEqual(16);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkReachability updates sources, thresholds, and weights without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createReachabilityFixture(device, F1, 8, {
    sources: [0, 5],
    sourceCosts: [0, 2.5],
    sourceCount: 1,
    thresholds: [2, 5]
  });
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();

  fixture.sources.write(Uint32Array.from([5, 0]));
  fixture.sourceCosts!.write(Float32Array.from([0, 0]));
  submitGraph(device, compiled, undefined);
  expect(await readCosts(fixture)).toEqual(Array.from(dijkstra(F1, 8, [{node: 5, cost: 0}])));

  fixture.sources.write(Uint32Array.from([0, 5]));
  fixture.sourceCosts!.write(Float32Array.from([0, 2.5]));
  fixture.sourceCount!.write(Uint32Array.from([2]));
  submitGraph(device, compiled, undefined);
  expect(await readCosts(fixture)).toEqual([0, 3, 1, 4, 7, 2.5, 3.5, Infinity]);
  const predecessors = await readUint32(fixture.predecessorsBuffer, 8);
  expect(predecessors[5]).toBe(NONE);
  expect(predecessors[6]).toBe(5);

  fixture.sourceCount!.write(Uint32Array.from([1]));
  submitGraph(device, compiled, undefined);
  expect(await readCosts(fixture)).toEqual([0, 3, 1, 4, 7, Infinity, Infinity, Infinity]);

  fixture.thresholds!.write(Float32Array.from([3.5, 100]));
  submitGraph(device, compiled, undefined);
  const costs = Float32Array.from(await readCosts(fixture));
  expect(await readUint32(fixture.bandsBuffer!, 8)).toEqual(
    Array.from(bandOracle(costs, [3.5, 100]))
  );

  const heavier: NetworkEdge[] = F1_EDGES.map(edge =>
    edge[0] === 0 && edge[1] === 2 ? [0, 2, 9] : edge
  );
  const heavierCSR = buildCSR(8, heavier);
  fixture.weightsBuffer.write(heavierCSR.weights);
  submitGraph(device, compiled, undefined);
  expect(await readCosts(fixture)).toEqual(
    Array.from(dijkstra(heavierCSR, 8, [{node: 0, cost: 0}]))
  );

  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkReachability honors a per-frame cost limit on a random network', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 2000;
  const csr = buildCSR(nodeCount, createRandomNetwork(7, nodeCount, 8000));
  const fixture = createReachabilityFixture(device, csr, nodeCount, {
    sources: [0, 1000],
    thresholds: [5, 10, 20],
    maxIterations: 256,
    costLimit: 10
  });
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();
  const seeds = [
    {node: 0, cost: 0},
    {node: 1000, cost: 0}
  ];

  submitGraph(device, compiled, undefined);
  expect(await readCosts(fixture)).toEqual(Array.from(dijkstra(csr, nodeCount, seeds, 10)));

  fixture.costLimit!.write(Float32Array.from([Infinity]));
  submitGraph(device, compiled, undefined);
  const expected = dijkstra(csr, nodeCount, seeds);
  expect(await readCosts(fixture)).toEqual(Array.from(expected));
  expect(await readUint32(fixture.predecessorsBuffer, nodeCount)).toEqual(
    Array.from(predecessorOracle(csr, expected, seeds))
  );
  expect(await readUint32(fixture.bandsBuffer!, nodeCount)).toEqual(
    Array.from(bandOracle(expected, [5, 10, 20]))
  );
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkReachability reports convergence against the iteration bound', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const pathEdges: NetworkEdge[] = Array.from({length: 19}, (_, node) => [node, node + 1, 1]);
  const path = buildCSR(20, pathEdges);
  const expected = Array.from(dijkstra(path, 20, [{node: 0, cost: 0}]));
  const fixture = createReachabilityFixture(device, path, 20, {
    sources: [0],
    maxIterations: 32,
    localIterations: 1,
    activeIterations: 2
  });
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();

  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([0]);
  expect(await readUint32(fixture.iterationCountBuffer, 1)).toEqual([2]);
  let costs = await readCosts(fixture);
  expect(Number.isFinite(costs[1]) && Number.isFinite(costs[2])).toBe(true);
  for (const [node, cost] of costs.entries()) {
    if (Number.isFinite(cost)) {
      expect(cost).toBe(expected[node]);
    }
  }

  fixture.activeIterations!.write(Uint32Array.from([32]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);
  expect(await readCosts(fixture)).toEqual(expected);
  expect((await readUint32(fixture.iterationCountBuffer, 1))[0]).toBeLessThanOrEqual(21);

  fixture.activeIterations!.write(Uint32Array.from([0]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([0]);
  expect(await readUint32(fixture.iterationCountBuffer, 1)).toEqual([0]);
  costs = await readCosts(fixture);
  expect(costs[0]).toBe(0);
  expect(costs.slice(1).every(cost => cost === Infinity)).toBe(true);
  compiled.destroy();
  destroyFixture(fixture);

  const bounded = createReachabilityFixture(device, path, 20, {
    sources: [0],
    maxIterations: 3,
    localIterations: 1
  });
  bounded.graph.add(new GPUNetworkReachability(bounded.props));
  const boundedCompiled = bounded.graph.compile();
  submitGraph(device, boundedCompiled, undefined);
  expect(await readUint32(bounded.convergedBuffer, 1)).toEqual([0]);
  expect(await readUint32(bounded.iterationCountBuffer, 1)).toEqual([3]);
  boundedCompiled.destroy();
  destroyFixture(bounded);
});

it('GPUNetworkReachability handles high degree, invalid data, and fractional weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const starEdges: NetworkEdge[] = [];
  for (let leaf = 1; leaf <= 300; leaf++) {
    starEdges.push([0, leaf, (leaf % 7) + 1]);
  }
  for (let leaf = 1; leaf <= 300; leaf++) {
    starEdges.push([leaf, 301, 1]);
  }
  const star = buildCSR(302, starEdges);
  const starFixture = createReachabilityFixture(device, star, 302, {
    sources: [0]
  });
  starFixture.graph.add(new GPUNetworkReachability(starFixture.props));
  let compiled = starFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const starCosts = dijkstra(star, 302, [{node: 0, cost: 0}]);
  expect(await readCosts(starFixture)).toEqual(Array.from(starCosts));
  const starPredecessors = await readUint32(starFixture.predecessorsBuffer, 302);
  expect(starPredecessors).toEqual(
    Array.from(predecessorOracle(star, starCosts, [{node: 0, cost: 0}]))
  );
  expect(starPredecessors[301]).toBe(7);
  compiled.destroy();
  destroyFixture(starFixture);

  const invalid = buildCSR(8, [...F1_EDGES, [0, 3, -1], [0, 4, Number.NaN], [1, 99, 1]]);
  const invalidFixture = createReachabilityFixture(device, invalid, 8, {
    sources: [0, 0, 12345]
  });
  invalidFixture.graph.add(new GPUNetworkReachability(invalidFixture.props));
  compiled = invalidFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const costs = await readCosts(invalidFixture);
  expect(costs).toEqual([0, 3, 1, 4, 7, Infinity, Infinity, Infinity]);
  const predecessors = await readUint32(invalidFixture.predecessorsBuffer, 8);
  for (let node = 0; node < 8; node++) {
    let current = node;
    for (
      let step = 0;
      step < 8 && Number.isFinite(costs[current]) && predecessors[current] !== NONE;
      step++
    ) {
      expect(costs[predecessors[current]]).toBeLessThan(costs[current]);
      current = predecessors[current];
    }
    if (Number.isFinite(costs[node])) {
      expect(current).toBe(0);
    }
  }
  compiled.destroy();
  destroyFixture(invalidFixture);

  if (isSoftwareDevice(device)) {
    return;
  }
  const nodeCount = 2000;
  const fractional = createRandomNetwork(7, nodeCount, 8000).map(
    ([from, to, weight]) => [from, to, weight * 0.37] as NetworkEdge
  );
  const fractionalCSR = buildCSR(nodeCount, fractional);
  const fractionalFixture = createReachabilityFixture(device, fractionalCSR, nodeCount, {
    sources: [0, 1000],
    maxIterations: 256
  });
  fractionalFixture.graph.add(new GPUNetworkReachability(fractionalFixture.props));
  compiled = fractionalFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const expected = dijkstra(fractionalCSR, nodeCount, [
    {node: 0, cost: 0},
    {node: 1000, cost: 0}
  ]);
  const actual = await readCosts(fractionalFixture);
  for (const [node, cost] of actual.entries()) {
    if (Number.isFinite(expected[node])) {
      expect(Math.abs(cost - expected[node])).toBeLessThanOrEqual(
        1e-5 * Math.max(1, expected[node])
      );
    } else {
      expect(cost).toBe(Infinity);
    }
  }
  compiled.destroy();
  destroyFixture(fractionalFixture);
});

it('GPUNetworkReachability composes with GPUCOOToCSR and a second instance', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const sorted = [...F1_EDGES].sort((left, right) => left[0] - right[0]);
  const rowIndicesBuffer = createInputBuffer(device, Uint32Array.from(sorted.map(edge => edge[0])));
  const columnIndicesBuffer = createInputBuffer(
    device,
    Uint32Array.from(sorted.map(edge => edge[1]))
  );
  const valuesBuffer = createInputBuffer(device, Float32Array.from(sorted.map(edge => edge[2])));
  const costsBuffer = createOutputBuffer(device, 8);
  const sourcesBuffer = createInputBuffer(device, Uint32Array.from([0]));
  const graph = new GPUCommandGraph(device, {id: 'coo-reachability'});
  const edgeCount = sorted.length;
  const rowOffsets = createTransientView(graph, 'row-offsets', 'uint32', 9);
  const outputColumnIndices = createTransientView(graph, 'csr-columns', 'uint32', edgeCount);
  const outputValues = createTransientView(graph, 'csr-values', 'float32', edgeCount);
  graph.add(
    new GPUCOOToCSR({
      rows: 8,
      rowIndices: importGraphBuffer(graph, 'coo-rows', rowIndicesBuffer, 'uint32', edgeCount),
      columnIndices: importGraphBuffer(
        graph,
        'coo-columns',
        columnIndicesBuffer,
        'uint32',
        edgeCount
      ),
      values: importGraphBuffer(graph, 'coo-values', valuesBuffer, 'float32', edgeCount),
      rowOffsets,
      outputColumnIndices,
      outputValues
    })
  );
  graph.add(
    new GPUNetworkReachability({
      offsets: rowOffsets,
      neighbors: outputColumnIndices,
      weights: outputValues,
      sources: importGraphBuffer(graph, 'sources', sourcesBuffer, 'uint32', 1),
      costs: importGraphBuffer(graph, 'costs', costsBuffer, 'float32', 8),
      maxIterations: 16
    })
  );
  const second = createReachabilityFixture(device, F1, 8, {id: 'reach-b', sources: [5]}, graph);
  graph.add(new GPUNetworkReachability(second.props));
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readFloat32(costsBuffer, 8)).toEqual([0, 3, 1, 4, 7, Infinity, Infinity, Infinity]);
  expect(await readCosts(second)).toEqual(Array.from(dijkstra(F1, 8, [{node: 5, cost: 0}])));
  compiled.destroy();
  destroyFixture(second);
  for (const buffer of [
    rowIndicesBuffer,
    columnIndicesBuffer,
    valuesBuffer,
    costsBuffer,
    sourcesBuffer
  ]) {
    buffer.destroy();
  }
});

const LOCAL_ITERATION_VALUES = [1, 2, 16, 64];

type ParityCase = {
  name: string;
  nodeCount: number;
  edges: NetworkEdge[];
  sources: number[];
  /** Rounds that converge for every value in `localIterationValues`. */
  maxIterations: number;
  localIterationValues?: number[];
};

function createParityCases(): ParityCase[] {
  const grid = createGridNetwork(11, 24, 24);
  const starEdges: NetworkEdge[] = [];
  for (let leaf = 1; leaf <= 700; leaf++) {
    starEdges.push([0, leaf, (leaf % 7) + 1], [leaf, 701, 1], [701, leaf, 2]);
  }
  return [
    {
      name: 'random',
      nodeCount: 1500,
      edges: createRandomNetwork(3, 1500, 6000),
      sources: [0, 700],
      maxIterations: 256
    },
    {
      name: 'cyclic',
      nodeCount: 8,
      edges: F1_EDGES,
      sources: [0, 5],
      maxIterations: 16
    },
    {
      name: 'disconnected',
      nodeCount: 600,
      edges: createRandomNetwork(5, 300, 1200),
      sources: [1, 2],
      maxIterations: 128
    },
    {
      name: 'high-degree',
      nodeCount: 702,
      edges: starEdges,
      sources: [0],
      maxIterations: 16
    },
    {
      name: 'long chain',
      nodeCount: 2000,
      edges: createPathNetwork(2000),
      sources: [0],
      maxIterations: 130,
      localIterationValues: [16, 64]
    },
    {
      name: 'short chain',
      nodeCount: 100,
      edges: createPathNetwork(100),
      sources: [0],
      maxIterations: 110,
      localIterationValues: [1, 2]
    },
    {
      name: 'grid',
      nodeCount: 576,
      edges: grid,
      sources: [0, 575],
      maxIterations: 256
    }
  ];
}

it('GPUNetworkReachability matches Dijkstra for every localIterations on varied networks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const testCase of createParityCases()) {
    const csr = buildCSR(testCase.nodeCount, testCase.edges);
    const expected = dijkstra(
      csr,
      testCase.nodeCount,
      testCase.sources.map(node => ({node, cost: 0}))
    );
    const expectedPredecessors = Array.from(
      predecessorOracle(
        csr,
        expected,
        testCase.sources.map(node => ({node, cost: 0}))
      )
    );
    for (const localIterations of testCase.localIterationValues ?? LOCAL_ITERATION_VALUES) {
      const fixture = createReachabilityFixture(device, csr, testCase.nodeCount, {
        sources: testCase.sources,
        maxIterations: testCase.maxIterations,
        localIterations
      });
      fixture.graph.add(new GPUNetworkReachability(fixture.props));
      const compiled = fixture.graph.compile();
      submitGraph(device, compiled, undefined);
      const label = `${testCase.name} localIterations ${localIterations}`;
      expect(await readCosts(fixture), label).toEqual(Array.from(expected));
      expect(await readUint32(fixture.predecessorsBuffer, testCase.nodeCount), label).toEqual(
        expectedPredecessors
      );
      expect(await readUint32(fixture.convergedBuffer, 1), label).toEqual([1]);
      compiled.destroy();
      destroyFixture(fixture);
    }
  }
});

it('GPUNetworkReachability needs about ceil(hops / localIterations) rounds on a chain', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 120;
  const path = buildCSR(nodeCount, createPathNetwork(nodeCount));
  const rounds: Record<number, number> = {};
  for (const localIterations of [1, 8, 64]) {
    const fixture = createReachabilityFixture(device, path, nodeCount, {
      sources: [0],
      maxIterations: Math.ceil((nodeCount - 1) / localIterations) + 3,
      localIterations
    });
    fixture.graph.add(new GPUNetworkReachability(fixture.props));
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);
    [rounds[localIterations]] = await readUint32(fixture.iterationCountBuffer, 1);
    compiled.destroy();
    destroyFixture(fixture);
  }
  // Hop-per-round: n - 1 hops plus the final empty-improvement round.
  expect(rounds[1]).toBe(nodeCount);
  for (const localIterations of [8, 64]) {
    const lowerBound = Math.ceil((nodeCount - 1) / localIterations);
    expect(rounds[localIterations]).toBeGreaterThanOrEqual(lowerBound);
    expect(rounds[localIterations]).toBeLessThanOrEqual(lowerBound + 2);
  }
});

it('GPUNetworkReachability reports a truncated chain as not converged with exact partial costs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 400;
  const path = buildCSR(nodeCount, createPathNetwork(nodeCount));
  const expected = Array.from(dijkstra(path, nodeCount, [{node: 0, cost: 0}]));
  for (const [localIterations, maxIterations] of [
    [1, 10],
    [8, 10],
    [4, 3]
  ]) {
    const fixture = createReachabilityFixture(device, path, nodeCount, {
      sources: [0],
      maxIterations,
      localIterations
    });
    fixture.graph.add(new GPUNetworkReachability(fixture.props));
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([0]);
    expect(await readUint32(fixture.iterationCountBuffer, 1)).toEqual([maxIterations]);
    const costs = await readCosts(fixture);
    const reached = costs.filter(cost => Number.isFinite(cost)).length;
    // Every round covers at most localIterations hops, and some progress was made.
    expect(reached).toBeGreaterThan(1);
    expect(reached).toBeLessThanOrEqual(maxIterations * localIterations + 1);
    for (const [node, cost] of costs.entries()) {
      if (Number.isFinite(cost)) {
        expect(cost).toBe(expected[node]);
      }
    }
    compiled.destroy();
    destroyFixture(fixture);
  }
});

it('GPUNetworkReachability changes limit and cost limit per frame without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 300;
  const path = buildCSR(nodeCount, createPathNetwork(nodeCount));
  const expected = Array.from(dijkstra(path, nodeCount, [{node: 0, cost: 0}]));
  const fixture = createReachabilityFixture(device, path, nodeCount, {
    sources: [0],
    maxIterations: 128,
    localIterations: 4,
    activeIterations: 3,
    costLimit: 100
  });
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();
  const frames: [number, number][] = [
    [3, 100],
    [128, 100],
    [0, 100],
    [1, Infinity],
    [128, 40],
    [128, Infinity]
  ];
  for (const [activeIterations, costLimit] of frames) {
    fixture.activeIterations!.write(Uint32Array.of(activeIterations));
    fixture.costLimit!.write(Float32Array.of(costLimit));
    submitGraph(device, compiled, undefined);
    const costs = await readCosts(fixture);
    const [converged] = await readUint32(fixture.convergedBuffer, 1);
    const [iterations] = await readUint32(fixture.iterationCountBuffer, 1);
    const label = `limit ${activeIterations} cost limit ${costLimit}`;
    if (activeIterations === 0) {
      expect(converged, label).toBe(0);
      expect(iterations, label).toBe(0);
      expect(costs[0], label).toBe(0);
      expect(
        costs.slice(1).every(cost => cost === Infinity),
        label
      ).toBe(true);
      continue;
    }
    for (const [node, cost] of costs.entries()) {
      if (Number.isFinite(cost)) {
        expect(cost, label).toBe(expected[node]);
        expect(cost, label).toBeLessThanOrEqual(costLimit);
      }
    }
    if (converged) {
      const reachable = expected.map(cost => (cost <= costLimit ? cost : Infinity));
      expect(costs, label).toEqual(reachable);
    } else {
      expect(activeIterations, label).toBeLessThan(128);
      expect(iterations, label).toBe(activeIterations);
    }
  }
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkReachability honors sourceCount and sourceCosts with local hops', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 1500;
  const csr = buildCSR(nodeCount, createRandomNetwork(9, nodeCount, 6000));
  const fixture = createReachabilityFixture(device, csr, nodeCount, {
    sources: [10, 20, 30],
    sourceCosts: [0, 3, 1.5],
    sourceCount: 2,
    maxIterations: 128,
    localIterations: 16
  });
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readCosts(fixture)).toEqual(
    Array.from(
      dijkstra(csr, nodeCount, [
        {node: 10, cost: 0},
        {node: 20, cost: 3}
      ])
    )
  );
  fixture.sourceCount!.write(Uint32Array.of(3));
  submitGraph(device, compiled, undefined);
  expect(await readCosts(fixture)).toEqual(
    Array.from(
      dijkstra(csr, nodeCount, [
        {node: 10, cost: 0},
        {node: 20, cost: 3},
        {node: 30, cost: 1.5}
      ])
    )
  );
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkReachability keeps two instances with different local hops independent', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 800;
  const csr = buildCSR(nodeCount, createRandomNetwork(21, nodeCount, 3000));
  const graph = new GPUCommandGraph(device, {id: 'two-instances'});
  const first = createReachabilityFixture(
    device,
    csr,
    nodeCount,
    {id: 'first', sources: [0], maxIterations: 128, localIterations: 1},
    graph
  );
  const second = createReachabilityFixture(
    device,
    csr,
    nodeCount,
    {id: 'second', sources: [5, 6], maxIterations: 64, localIterations: 16},
    graph
  );
  graph.add(new GPUNetworkReachability(first.props));
  graph.add(new GPUNetworkReachability(second.props));
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readCosts(first)).toEqual(
    Array.from(dijkstra(csr, nodeCount, [{node: 0, cost: 0}]))
  );
  expect(await readCosts(second)).toEqual(
    Array.from(
      dijkstra(csr, nodeCount, [
        {node: 5, cost: 0},
        {node: 6, cost: 0}
      ])
    )
  );
  compiled.destroy();
  destroyFixture(first);
  destroyFixture(second);
});

type TieCase = {
  name: string;
  nodeCount: number;
  edges: readonly NetworkEdge[];
  sources: number[];
  sourceCosts?: number[];
  sourceCount?: number;
  costLimit?: number;
  maxIterations?: number;
};

/** Host copy of the seed kernel's acceptance rules, for the oracle and the independent walk check. */
function getAcceptedSources(testCase: TieCase): {node: number; cost: number}[] {
  const count = Math.min(testCase.sourceCount ?? testCase.sources.length, testCase.sources.length);
  return testCase.sources
    .slice(0, count)
    .map((node, row) => ({node, cost: testCase.sourceCosts?.[row] ?? 0}));
}

/**
 * Runs one case on the GPU and checks costs against Dijkstra, predecessors exactly against the tie
 * oracle, the independent walk property, and `converged`. Returns the oracle levels' maximum.
 */
async function expectTieParity(
  device: Device,
  testCase: TieCase,
  localIterations: number,
  maxTieIterations: number,
  expectedConverged = 1,
  oracleMaxLevel = Infinity
): Promise<{predecessors: number[]; converged: number}> {
  const csr = buildCSR(testCase.nodeCount, testCase.edges);
  const fixture = createReachabilityFixture(device, csr, testCase.nodeCount, {
    sources: testCase.sources,
    sourceCosts: testCase.sourceCosts,
    sourceCount: testCase.sourceCount,
    costLimit: testCase.costLimit,
    maxIterations: localIterations === 1 ? (testCase.maxIterations ?? 24) : 24,
    localIterations,
    maxTieIterations
  });
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const accepted = getAcceptedSources(testCase);
  const label = `${testCase.name} localIterations ${localIterations}`;
  const expectedCosts = dijkstra(csr, testCase.nodeCount, accepted, testCase.costLimit);
  const costs = Float32Array.from(await readCosts(fixture));
  expect(Array.from(costs), label).toEqual(Array.from(expectedCosts));
  const predecessors = await readUint32(fixture.predecessorsBuffer, testCase.nodeCount);
  expect(predecessors, label).toEqual(
    Array.from(
      tieLevelOracle(csr, costs, accepted, testCase.costLimit, oracleMaxLevel).predecessors
    )
  );
  const rootNodes = new Set(
    accepted
      .filter(
        ({node, cost}) =>
          node < testCase.nodeCount &&
          cost >= 0 &&
          Math.fround(cost) <= Math.fround(testCase.costLimit ?? Infinity) &&
          Math.fround(cost) === costs[node]
      )
      .map(({node}) => node)
  );
  if (expectedConverged === 1) {
    expect(
      checkPredecessorWalks(csr, costs, predecessors, node => rootNodes.has(node)),
      label
    ).toBeUndefined();
  }
  const [converged] = await readUint32(fixture.convergedBuffer, 1);
  expect(converged, label).toBe(expectedConverged);
  compiled.destroy();
  destroyFixture(fixture);
  return {predecessors, converged};
}

/** Tie-phase rounds that always suffice for the case's deepest plateau. */
function getTieRounds(testCase: TieCase, localIterations: number): number {
  const csr = buildCSR(testCase.nodeCount, testCase.edges);
  const accepted = getAcceptedSources(testCase);
  const costs = dijkstra(csr, testCase.nodeCount, accepted, testCase.costLimit);
  const {levels} = tieLevelOracle(csr, costs, accepted, testCase.costLimit);
  let deepest = 0;
  for (const level of levels) {
    if (level !== NONE) deepest = Math.max(deepest, level);
  }
  return Math.min(1024, Math.ceil(deepest / localIterations) + 2);
}

function createRandomZeroNetwork(
  seed: number,
  nodeCount: number,
  edgeCount: number,
  zeroFraction: number
): NetworkEdge[] {
  let state = (seed * 7919) >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  return createRandomNetwork(seed, nodeCount, edgeCount).map(
    ([from, to, weight]) => [from, to, next() < zeroFraction ? 0 : weight] as NetworkEdge
  );
}

function createTieCases(): TieCase[] {
  const zeroChain = (length: number): NetworkEdge[] =>
    Array.from({length}, (_, node) => [node, node + 1, 0] as NetworkEdge);
  return [
    {
      name: 'zero chain of 100',
      nodeCount: 101,
      edges: zeroChain(100),
      sources: [0],
      maxIterations: 110
    },
    {
      name: 'zero chain behind a positive entry',
      nodeCount: 60,
      edges: [
        [0, 1, 3] as NetworkEdge,
        ...zeroChain(58).map(([a, b, w]) => [a + 1, b + 1, w] as NetworkEdge)
      ],
      sources: [0],
      maxIterations: 70
    },
    {
      name: 'negative zero weights',
      nodeCount: 4,
      edges: [
        [0, 1, -0],
        [1, 2, -0],
        [2, 3, 0]
      ],
      sources: [0]
    },
    {
      name: 'zero 2-cycle',
      nodeCount: 4,
      edges: [
        [0, 1, 5],
        [1, 2, 0],
        [2, 1, 0],
        [2, 3, 1]
      ],
      sources: [0]
    },
    {
      name: 'zero 3-cycle behind a positive entry',
      nodeCount: 5,
      edges: [
        [0, 1, 2],
        [1, 2, 0],
        [2, 3, 0],
        [3, 1, 0],
        [3, 4, 4]
      ],
      sources: [0]
    },
    {
      name: 'zero cycle containing the source',
      nodeCount: 4,
      edges: [
        [0, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
        [2, 3, 2]
      ],
      sources: [0]
    },
    {
      name: 'two equal sources joined by zero edges',
      nodeCount: 8,
      edges: [
        [3, 5, 0],
        [5, 3, 0],
        [5, 6, 1],
        [3, 7, 0]
      ],
      sources: [3, 5]
    },
    {
      name: 'mixed plateau with several entries',
      nodeCount: 10,
      edges: [
        [0, 1, 3],
        [0, 2, 3],
        [1, 3, 2],
        [2, 6, 2],
        [2, 4, 2],
        [3, 4, 0],
        [4, 5, 0],
        [6, 5, 0],
        [5, 3, 0],
        [5, 7, 0],
        [7, 8, 1],
        [8, 9, 0]
      ],
      sources: [0]
    },
    {
      name: 'tiny weight rounds away at large cost',
      nodeCount: 4,
      edges: [
        [0, 1, 1],
        [1, 2, 1],
        [2, 3, 16]
      ],
      sources: [0],
      sourceCosts: [1e8]
    },
    {
      name: 'source rows, costLimit, and sourceCount',
      nodeCount: 12,
      edges: [
        [0, 1, 0],
        [1, 4, 0],
        [4, 5, 0],
        [2, 3, 0],
        [7, 8, 0],
        [6, 9, 0],
        [0, 6, 0]
      ],
      // Rows: accepted root, over the cost limit, past sourceCount, negative, NaN, out of range,
      // and duplicate rows of node 7 where the cheaper one wins.
      sources: [0, 4, 2, 3, 10, 99, 7, 7, 6],
      sourceCosts: [0, 7, 0, -1, Number.NaN, 0, 3, 1, 2],
      sourceCount: 9,
      costLimit: 6
    },
    {
      name: 'sourceCount hides a root',
      nodeCount: 6,
      edges: [
        [0, 1, 0],
        [2, 3, 0],
        [3, 4, 0]
      ],
      sources: [0, 2],
      sourceCount: 1
    },
    ...[1, 2, 3].map(seed => ({
      name: `random 30% zero weights, seed ${seed}`,
      nodeCount: 1200,
      edges: createRandomZeroNetwork(seed, 1200, 4800, 0.3),
      sources: [0, 600],
      maxIterations: 64
    }))
  ];
}

it('GPUNetworkReachability gives zero-weight plateaus acyclic predecessors that match the tie oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const testCase of createTieCases()) {
    for (const localIterations of [1, 16]) {
      const tieRounds = getTieRounds(testCase, localIterations);
      await expectTieParity(device, testCase, localIterations, tieRounds);
    }
  }
}, 120000);

it('GPUNetworkReachability chains 64 hops of a 100-edge zero chain with the default tie rounds and reports truncation', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const edges = Array.from({length: 100}, (_, node) => [node, node + 1, 0] as NetworkEdge);
  // Default localIterations is 16, so 4 tie rounds chain 64 hops and 100 hops are truncated.
  const csr = buildCSR(101, edges);
  const fixture = createReachabilityFixture(device, csr, 101, {sources: [0]});
  expect(fixture.props.maxTieIterations).toBeUndefined();
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const predecessors = await readUint32(fixture.predecessorsBuffer, 101);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([0]);
  expect(predecessors.slice(0, 40)).toEqual([NONE, ...Array.from({length: 39}, (_, i) => i)]);
  compiled.destroy();
  destroyFixture(fixture);
  await expectTieParity(
    device,
    {name: 'default depth', nodeCount: 101, edges, sources: [0]},
    16,
    8
  );
});

it('GPUNetworkReachability reports a truncated tie phase with only deeper plateau nodes unreached', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const edges = Array.from({length: 20}, (_, node) => [node, node + 1, 0] as NetworkEdge);
  const testCase: TieCase = {name: 'truncated plateau', nodeCount: 21, edges, sources: [0]};
  const {predecessors, converged} = await expectTieParity(device, testCase, 1, 5, 0, 5);
  expect(converged).toBe(0);
  expect(predecessors).toEqual([
    NONE,
    ...Array.from({length: 5}, (_, i) => i),
    ...Array.from({length: 15}, () => NONE)
  ]);
  // More rounds finish the plateau and set the flag.
  await expectTieParity(device, testCase, 1, 21);
});

it('GPUNetworkReachability recomputes ties when weights change without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 600;
  const sources = [0, 300];
  const accepted = sources.map(node => ({node, cost: 0}));
  const frames = [0.3, 0, 0.6, 0.3].map((zeroFraction, frame) =>
    buildCSR(nodeCount, createRandomZeroNetwork(20 + frame, nodeCount, 2400, zeroFraction))
  );
  // Same structure is not required: only the weights buffer is rewritten, so keep the first CSR's
  // topology and re-randomize weights.
  const base = frames[0];
  const fixture = createReachabilityFixture(device, base, nodeCount, {
    sources,
    maxIterations: 64,
    localIterations: 4,
    maxTieIterations: 32
  });
  fixture.graph.add(new GPUNetworkReachability(fixture.props));
  const compiled = fixture.graph.compile();
  let state = 99;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  for (const zeroFraction of [0.3, 0, 0.6, 0.3, 1]) {
    const weights = Float32Array.from(base.weights, () =>
      next() < zeroFraction ? 0 : 1 + Math.floor(next() * 9)
    );
    fixture.weightsBuffer.write(weights);
    submitGraph(device, compiled, undefined);
    const csr = {...base, weights};
    const costs = Float32Array.from(await readCosts(fixture));
    expect(Array.from(costs)).toEqual(Array.from(dijkstra(csr, nodeCount, accepted)));
    const predecessors = await readUint32(fixture.predecessorsBuffer, nodeCount);
    expect(predecessors, `zeroFraction ${zeroFraction}`).toEqual(
      Array.from(predecessorOracle(csr, costs, accepted))
    );
    expect(
      checkPredecessorWalks(csr, costs, predecessors, node => sources.includes(node))
    ).toBeUndefined();
    expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);
  }
  compiled.destroy();
  destroyFixture(fixture);
}, 120000);
