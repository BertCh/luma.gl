// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraph, GPUGraphModularity} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  decodeGPUNetworkStatistics,
  encodeGPUNetworkStatisticsParameters,
  getGPUNetworkStatisticsLength,
  GPUNetworkStatistics
} from '../../../src/gpu-network/network-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSymmetricEdges} from '../network-analysis/network-analytics-oracle';
import {
  buildCSR,
  createRandomNetwork,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';
import {
  computeNetworkStatisticsOracle,
  type NetworkStatisticsOracleOptions
} from './network-statistics-oracle';

type FixtureOptions = {
  nodeCount: number;
  edges: readonly NetworkEdge[];
  directed?: boolean;
  countSelfLoopsTwice?: boolean;
  vertexMask?: Uint32Array;
  edgeMask?: Uint32Array;
  communities?: Uint32Array;
  parameters?: {resolution?: number; degreeBinWidth?: number};
  binCount?: number;
  binning?: 'linear' | 'log2';
};

class Fixture {
  readonly graph: GPUCommandGraph;
  readonly csr: ReturnType<typeof buildCSR>;
  readonly buffers: Buffer[] = [];
  readonly outputBuffer: Buffer;
  readonly contributor: GPUNetworkStatistics;
  readonly binCount: number;
  compiled?: CompiledGPUCommandGraph<void>;
  private vertexMaskBuffer?: Buffer;
  private edgeMaskBuffer?: Buffer;
  private parametersBuffer?: Buffer;

  constructor(
    readonly device: Device,
    readonly options: FixtureOptions
  ) {
    const {nodeCount} = options;
    this.binCount = options.binCount ?? 32;
    this.graph = new GPUCommandGraph(device, {id: 'statistics'});
    this.csr = buildCSR(nodeCount, options.edges);
    const slotCount = this.csr.neighbors.length;
    const track = (buffer: Buffer) => {
      this.buffers.push(buffer);
      return buffer;
    };
    const offsets = track(createInputBuffer(device, this.csr.offsets));
    const neighbors = track(createInputBuffer(device, this.csr.neighbors));
    this.outputBuffer = track(
      createOutputBuffer(device, getGPUNetworkStatisticsLength(this.binCount))
    );
    const importView = (name: string, buffer: Buffer, length: number) =>
      importGraphBuffer(this.graph, `stats-${name}`, buffer, 'uint32', length);
    const optional = (name: string, values: Uint32Array | undefined) =>
      values
        ? importView(name, track(createInputBuffer(device, values)), values.length)
        : undefined;
    if (options.vertexMask)
      this.vertexMaskBuffer = track(createInputBuffer(device, options.vertexMask));
    if (options.edgeMask) this.edgeMaskBuffer = track(createInputBuffer(device, options.edgeMask));
    if (options.parameters) {
      this.parametersBuffer = track(
        createInputBuffer(device, encodeGPUNetworkStatisticsParameters(options.parameters))
      );
    }
    this.contributor = new GPUNetworkStatistics({
      id: 'stats',
      offsets: importView('offsets', offsets, nodeCount + 1),
      neighbors: importView('neighbors', neighbors, slotCount),
      directed: options.directed,
      countSelfLoopsTwice: options.countSelfLoopsTwice,
      vertexMask:
        this.vertexMaskBuffer && importView('vertex-mask', this.vertexMaskBuffer, nodeCount),
      edgeMask: this.edgeMaskBuffer && importView('edge-mask', this.edgeMaskBuffer, slotCount),
      communities: optional('communities', options.communities),
      parameters: this.parametersBuffer && importView('parameters', this.parametersBuffer, 2),
      degreeBinCount: this.binCount,
      degreeBinning: options.binning,
      componentIterations: 64,
      output: importView('output', this.outputBuffer, getGPUNetworkStatisticsLength(this.binCount))
    });
    this.graph.add(this.contributor);
  }

  /** Encodes the same compiled graph and decodes the raw words. */
  async run(): Promise<number[]> {
    this.compiled ??= this.graph.compile();
    submitGraph(this.device, this.compiled, undefined);
    return readUint32(this.outputBuffer, getGPUNetworkStatisticsLength(this.binCount));
  }

  setVertexMask(values: Uint32Array): void {
    this.vertexMaskBuffer!.write(values);
  }

  setParameters(parameters: {resolution?: number; degreeBinWidth?: number}): void {
    this.parametersBuffer!.write(encodeGPUNetworkStatisticsParameters(parameters));
  }

  oracle(overrides: Partial<NetworkStatisticsOracleOptions> = {}): Uint32Array {
    const {options} = this;
    return computeNetworkStatisticsOracle({
      nodeCount: options.nodeCount,
      csr: this.csr,
      directed: options.directed,
      countSelfLoopsTwice: options.countSelfLoopsTwice,
      vertexMask: options.vertexMask,
      edgeMask: options.edgeMask,
      communities: options.communities,
      resolution: options.parameters?.resolution,
      binWidth: options.parameters?.degreeBinWidth,
      binCount: this.binCount,
      binning: options.binning,
      ...overrides
    });
  }

  destroy(): void {
    this.compiled?.destroy();
    this.contributor.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}

/** Exact equality for every word except modularity, which is compared with a tolerance. */
function expectMatches(actual: number[], expected: Uint32Array): void {
  const MODULARITY = 10;
  const actualWords = actual.map((value, index) => (index === MODULARITY ? 0 : value));
  const expectedWords = Array.from(expected, (value, index) => (index === MODULARITY ? 0 : value));
  if (JSON.stringify(actualWords) !== JSON.stringify(expectedWords)) {
    throw new Error(
      `header actual ${actualWords.slice(0, 16)} expected ${expectedWords.slice(0, 16)}; histograms differ at ${actualWords.findIndex((v, i) => v !== expectedWords[i])}`
    );
  }
  expect(actualWords).toEqual(expectedWords);
  const decodedActual = decodeGPUNetworkStatistics(actual, {
    degreeBinCount: (actual.length - 16) / 3
  });
  const decodedExpected = decodeGPUNetworkStatistics(expected, {
    degreeBinCount: (actual.length - 16) / 3
  });
  expect(Math.abs(decodedActual.modularity - decodedExpected.modularity)).toBeLessThan(1e-5);
}

function createLabels(nodeCount: number, communityCount: number, seed = 7): Uint32Array {
  let state = seed;
  return Uint32Array.from({length: nodeCount}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) % communityCount;
  });
}

function createMask(length: number, deadFraction: number, seed: number): Uint32Array {
  let state = seed;
  return Uint32Array.from({length}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32 < deadFraction ? 0 : 1;
  });
}

function createUndirectedEdges(seed: number, nodeCount: number, edgeCount: number): NetworkEdge[] {
  return createSymmetricEdges(createRandomNetwork(seed, nodeCount, edgeCount));
}

it('GPUNetworkStatistics matches the oracle on an undirected random graph with communities', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 400;
  const edges = [...createUndirectedEdges(3, nodeCount, 600), [5, 5, 1] as NetworkEdge];
  const fixture = new Fixture(device, {
    nodeCount,
    edges,
    communities: createLabels(nodeCount, 12),
    parameters: {resolution: 1.5}
  });
  const words = await fixture.run();
  expectMatches(words, fixture.oracle());
  const decoded = decodeGPUNetworkStatistics(words);
  expect(decoded.liveVertexCount).toBe(nodeCount);
  expect(decoded.selfLoopSlotCount).toBe(1);
  expect(decoded.componentsConverged).toBe(true);
  expect(decoded.modularityValid).toBe(true);
  expect(decoded.componentCount).toBeGreaterThan(1);
  fixture.destroy();
});

it('GPUNetworkStatistics matches the oracle on a directed random graph with masks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 300;
  const edges = createRandomNetwork(11, nodeCount, 900);
  const slotCount = edges.length;
  const fixture = new Fixture(device, {
    nodeCount,
    edges,
    directed: true,
    vertexMask: createMask(nodeCount, 0.15, 5),
    edgeMask: createMask(slotCount, 0.2, 9),
    communities: createLabels(nodeCount, 8),
    parameters: {resolution: 0.75},
    binning: 'log2',
    binCount: 8
  });
  const words = await fixture.run();
  expectMatches(words, fixture.oracle());
  const decoded = decodeGPUNetworkStatistics(words, {degreeBinCount: 8});
  expect(decoded.liveEdgeCount).toBe(decoded.liveSlotCount);
  expect(decoded.liveVertexCount).toBeLessThan(nodeCount);
  expect(decoded.maxTotalDegree).toBeGreaterThanOrEqual(decoded.maxOutDegree);
  fixture.destroy();
});

it('GPUNetworkStatistics matches the oracle without masks or communities (directed and undirected)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const directed of [false, true]) {
    const nodeCount = 257;
    const edges = directed
      ? createRandomNetwork(21, nodeCount, 700)
      : createUndirectedEdges(21, nodeCount, 400);
    const fixture = new Fixture(device, {
      nodeCount,
      edges,
      directed,
      binCount: 5
    });
    const words = await fixture.run();
    expectMatches(words, fixture.oracle());
    const decoded = decodeGPUNetworkStatistics(words, {degreeBinCount: 5});
    expect(decoded.modularityValid).toBe(false);
    expect(decoded.modularity).toBe(0);
    fixture.destroy();
  }
});

it('GPUNetworkStatistics changes the bin width and masks between encodings without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 300;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createUndirectedEdges(31, nodeCount, 700),
    vertexMask: new Uint32Array(nodeCount).fill(1),
    communities: createLabels(nodeCount, 6),
    parameters: {resolution: 1, degreeBinWidth: 1},
    binCount: 6
  });
  expectMatches(await fixture.run(), fixture.oracle());
  const compiled = fixture.compiled;

  fixture.setParameters({resolution: 2, degreeBinWidth: 3});
  expectMatches(await fixture.run(), fixture.oracle({resolution: 2, binWidth: 3}));

  const mask = createMask(nodeCount, 0.3, 77);
  fixture.setVertexMask(mask);
  fixture.setParameters({resolution: 0.5, degreeBinWidth: 2});
  const words = await fixture.run();
  expectMatches(words, fixture.oracle({vertexMask: mask, resolution: 0.5, binWidth: 2}));
  expect(decodeGPUNetworkStatistics(words, {degreeBinCount: 6}).liveVertexCount).toBe(
    mask.reduce((sum, value) => sum + value, 0)
  );
  expect(fixture.compiled).toBe(compiled);
  fixture.destroy();
});

it('GPUNetworkStatistics splits a component when a bridge vertex is masked', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Triangle {0,1,2}, bridge 3, triangle {4,5,6}.
  const roads: [number, number][] = [
    [0, 1],
    [1, 2],
    [2, 0],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 4]
  ];
  const edges = roads.flatMap(([from, to]) => [
    [from, to, 1] as NetworkEdge,
    [to, from, 1] as NetworkEdge
  ]);
  const fixture = new Fixture(device, {
    nodeCount: 7,
    edges,
    vertexMask: new Uint32Array(7).fill(1),
    communities: Uint32Array.from([0, 0, 0, 0, 1, 1, 1])
  });
  let decoded = decodeGPUNetworkStatistics(await fixture.run());
  expect(decoded.componentCount).toBe(1);
  expect(decoded.largestComponentSize).toBe(7);
  expect(decoded.liveEdgeCount).toBe(8);

  const mask = Uint32Array.from([1, 1, 1, 0, 1, 1, 1]);
  fixture.setVertexMask(mask);
  const words = await fixture.run();
  expectMatches(words, fixture.oracle({vertexMask: mask}));
  decoded = decodeGPUNetworkStatistics(words);
  expect(decoded.componentCount).toBe(2);
  expect(decoded.largestComponentSize).toBe(3);
  expect(decoded.liveVertexCount).toBe(6);
  expect(decoded.liveEdgeCount).toBe(6);
  expect(decoded.isolatedVertexCount).toBe(0);
  expect(decoded.componentsConverged).toBe(true);
  fixture.destroy();
});

it('GPUNetworkStatistics reports invalid modularity for out-of-range labels and dead graphs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 20;
  const edges = createUndirectedEdges(41, nodeCount, 40);
  const labels = createLabels(nodeCount, 3);
  labels[4] = nodeCount;
  const fixture = new Fixture(device, {
    nodeCount,
    edges,
    communities: labels
  });
  const words = await fixture.run();
  expectMatches(words, fixture.oracle());
  const decoded = decodeGPUNetworkStatistics(words);
  expect(decoded.modularityValid).toBe(false);
  expect(decoded.modularity).toBe(0);
  fixture.destroy();

  const dead = new Fixture(device, {
    nodeCount,
    edges,
    vertexMask: new Uint32Array(nodeCount),
    communities: createLabels(nodeCount, 3)
  });
  const deadDecoded = decodeGPUNetworkStatistics(await dead.run());
  expect(deadDecoded.liveVertexCount).toBe(0);
  expect(deadDecoded.componentCount).toBe(0);
  expect(deadDecoded.largestComponentSize).toBe(0);
  expect(deadDecoded.modularityValid).toBe(false);
  dead.destroy();
});

it('GPUNetworkStatistics undirected modularity matches GPUGraphModularity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 200;
  const edges = createUndirectedEdges(51, nodeCount, 500);
  const labels = createLabels(nodeCount, 7);
  const fixture = new Fixture(device, {
    nodeCount,
    edges,
    communities: labels
  });
  const decoded = decodeGPUNetworkStatistics(await fixture.run());

  const roads = edges.filter(edge => edge[0] < edge[1]);
  const buffers: Buffer[] = [];
  const vector = (values: Uint32Array | number) => {
    const length = typeof values === 'number' ? values : values.length;
    const buffer = device.createBuffer({
      byteLength: Math.max(length, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    if (typeof values !== 'number') buffer.write(values);
    buffers.push(buffer);
    return {buffer, length};
  };
  const make = <Format extends 'uint32' | 'float32'>(
    values: Uint32Array | number,
    format: Format
  ) => {
    const {buffer, length} = vector(values);
    return new GPUVector<Format>({
      type: 'buffer',
      name: 'v',
      buffer,
      format,
      length
    });
  };
  const graph = new GPUGraph({
    vertexCount: nodeCount,
    sourceVertices: make(
      Uint32Array.from(roads, edge => edge[0]),
      'uint32'
    ),
    targetVertices: make(
      Uint32Array.from(roads, edge => edge[1]),
      'uint32'
    ),
    directed: false
  });
  const output = make(1, 'float32');
  const commandGraph = new GPUCommandGraph(device, {
    id: 'reference-modularity'
  });
  new GPUGraphModularity({
    id: 'reference',
    graph,
    communities: make(labels, 'uint32'),
    output
  }).addToGraph(commandGraph);
  const compiled = commandGraph.compile();
  submitGraph(device, compiled, undefined);
  const bytes = await (output.data[0].buffer as Buffer).readAsync();
  const reference = new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0];
  expect(Math.abs(decoded.modularity - reference)).toBeLessThan(1e-5);
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  fixture.destroy();
});

it('GPUNetworkStatistics undirected modularity matches GPUGraphModularity with self-loops counted twice', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 200;
  const edges = [
    ...createUndirectedEdges(51, nodeCount, 500),
    [5, 5, 1] as NetworkEdge,
    [9, 9, 1] as NetworkEdge
  ];
  const labels = createLabels(nodeCount, 7);
  const fixture = new Fixture(device, {
    nodeCount,
    edges,
    communities: labels,
    countSelfLoopsTwice: true
  });
  const words = await fixture.run();
  expectMatches(words, fixture.oracle());
  const decoded = decodeGPUNetworkStatistics(words);
  expect(decoded.selfLoopSlotCount).toBe(2);

  const roads = edges.filter(edge => edge[0] <= edge[1]);
  const buffers: Buffer[] = [];
  const vector = (values: Uint32Array | number) => {
    const length = typeof values === 'number' ? values : values.length;
    const buffer = device.createBuffer({
      byteLength: Math.max(length, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    if (typeof values !== 'number') buffer.write(values);
    buffers.push(buffer);
    return {buffer, length};
  };
  const make = <Format extends 'uint32' | 'float32'>(
    values: Uint32Array | number,
    format: Format
  ) => {
    const {buffer, length} = vector(values);
    return new GPUVector<Format>({
      type: 'buffer',
      name: 'v',
      buffer,
      format,
      length
    });
  };
  const graph = new GPUGraph({
    vertexCount: nodeCount,
    sourceVertices: make(
      Uint32Array.from(roads, edge => edge[0]),
      'uint32'
    ),
    targetVertices: make(
      Uint32Array.from(roads, edge => edge[1]),
      'uint32'
    ),
    directed: false
  });
  const output = make(1, 'float32');
  const commandGraph = new GPUCommandGraph(device, {
    id: 'reference-modularity'
  });
  new GPUGraphModularity({
    id: 'reference',
    graph,
    communities: make(labels, 'uint32'),
    output
  }).addToGraph(commandGraph);
  const compiled = commandGraph.compile();
  submitGraph(device, compiled, undefined);
  const bytes = await (output.data[0].buffer as Buffer).readAsync();
  const reference = new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0];
  expect(Math.abs(decoded.modularity - reference)).toBeLessThan(1e-5);
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  fixture.destroy();
});

it('GPUNetworkStatistics times a 100k-vertex, 400k-slot encoding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 100_000;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createUndirectedEdges(61, nodeCount, 200_000),
    vertexMask: createMask(nodeCount, 0.05, 3),
    communities: createLabels(nodeCount, 500),
    binning: 'log2',
    binCount: 16
  });
  expect(fixture.csr.neighbors.length).toBeGreaterThan(300_000);
  const first = await fixture.run();
  expectMatches(first, fixture.oracle());
  const samples: number[] = [];
  for (let sample = 0; sample < 7; sample++) {
    const start = performance.now();
    await fixture.run();
    samples.push(performance.now() - start);
  }
  samples.sort((left, right) => left - right);
  // eslint-disable-next-line no-console
  console.log(
    `GPUNetworkStatistics 100k vertices / ${fixture.csr.neighbors.length} slots: median ${samples[3].toFixed(2)} ms (min ${samples[0].toFixed(2)}, max ${samples[6].toFixed(2)}) encode+readback`
  );
  fixture.destroy();
}, 120_000);

it('GPUNetworkStatistics matches the oracle on a hub-and-spoke graph across many workgroups', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // 1500 vertices span six workgroups; vertex 0 is a hub with a row far longer than the others,
  // and degree 0 vertices stay isolated. Log2 bins and linear bins both run.
  const nodeCount = 1500;
  const edges: NetworkEdge[] = [];
  for (let leaf = 1; leaf < 1200; leaf++) edges.push([0, leaf, 1]);
  for (let vertex = 1200; vertex < 1400; vertex += 2) edges.push([vertex, vertex + 1, 1]);
  const vertexMask = createMask(nodeCount, 0.05, 3);
  vertexMask[0] = 1; // keep the hub live
  for (const binning of ['linear', 'log2'] as const) {
    for (const binCount of [8, 400]) {
      // 400 bins exceed the workgroup-memory histogram, so that case runs the global path.
      const fixture = new Fixture(device, {
        nodeCount,
        edges: createSymmetricEdges(edges),
        communities: createLabels(nodeCount, 9),
        vertexMask,
        binCount,
        binning
      });
      const words = await fixture.run();
      expectMatches(words, fixture.oracle());
      expect(
        decodeGPUNetworkStatistics(words, {degreeBinCount: binCount}).maxTotalDegree
      ).toBeGreaterThan(1000);
      fixture.destroy();
    }
  }
});
