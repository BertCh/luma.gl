// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  decodeGPUNetworkCoarseningSummary,
  GPUNetworkCoarsening
} from '../../../src/gpu-network/network-coarsening';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCoarseningCSR,
  computeCoarseningOracle,
  createRandom,
  createRandomCoarseningEdges,
  type CoarseningCSR,
  type CoarseningOracleResult
} from './network-coarsening-oracle';

type FixtureOptions = {
  nodeCount: number;
  csr: CoarseningCSR;
  labels: Uint32Array;
  groupCapacity: number;
  edgeCapacity: number;
  directed?: boolean;
  vertexMask?: Uint32Array;
  edgeMask?: Uint32Array;
  positions?: Float32Array;
  vertexValues?: Float32Array;
  weighted?: boolean;
};

type Result = {
  groupVertexCount: number[];
  groupIntraEdgeCount: number[];
  groupIntraWeight: number[];
  groupCentroid: number[];
  groupBounds: number[];
  groupValueSum: number[];
  count: number;
  total: number;
  overflow: number;
  edgeIds: number[];
  edgeTargets: number[];
  edgeCounts: number[];
  edgeWeights: number[];
  summary: number[];
};

class Fixture {
  readonly graph: GPUCommandGraph;
  readonly buffers: Buffer[] = [];
  readonly contributor: GPUNetworkCoarsening;
  readonly out: Record<string, Buffer> = {};
  compiled?: CompiledGPUCommandGraph<void>;

  constructor(
    readonly device: Device,
    readonly options: FixtureOptions
  ) {
    const {nodeCount, csr, groupCapacity, edgeCapacity} = options;
    this.graph = new GPUCommandGraph(device, {id: 'coarsening'});
    const view = <Format extends GPUVectorFormat>(
      name: string,
      format: Format,
      buffer: Buffer,
      length: number
    ) => {
      this.buffers.push(buffer);
      return importGraphBuffer(this.graph, `c-${name}`, buffer, format, length);
    };
    const input = (name: string, format: GPUVectorFormat, values: Uint32Array | Float32Array) =>
      view(name, format as 'uint32', createInputBuffer(device, values), values.length);
    const output = (name: string, format: GPUVectorFormat, rows: number, words: number) => {
      const buffer = createOutputBuffer(device, rows * words);
      this.out[name] = buffer;
      return view(name, format as 'uint32', buffer, rows);
    };
    const {positions, vertexValues} = options;
    this.contributor = new GPUNetworkCoarsening({
      id: 'coarse',
      offsets: input('offsets', 'uint32', csr.offsets) as never,
      neighbors: input('neighbors', 'uint32', csr.neighbors) as never,
      directed: options.directed,
      vertexMask:
        options.vertexMask && (input('vertex-mask', 'uint32', options.vertexMask) as never),
      edgeMask: options.edgeMask && (input('edge-mask', 'uint32', options.edgeMask) as never),
      weights: options.weighted ? (input('weights', 'float32', csr.weights) as never) : undefined,
      labels: input('labels', 'uint32', options.labels) as never,
      positions:
        positions &&
        (view('positions', 'float32x2', createInputBuffer(device, positions), nodeCount) as never),
      vertexValues: vertexValues && (input('values', 'float32', vertexValues) as never),
      groupCapacity,
      groupVertexCount: output('group-counts', 'uint32', groupCapacity, 1) as never,
      groupIntraEdgeCount: output('intra-count', 'uint32', groupCapacity, 1) as never,
      groupIntraWeight: output('intra-weight', 'float32', groupCapacity, 1) as never,
      groupCentroid: positions && (output('centroid', 'float32x2', groupCapacity, 2) as never),
      groupBounds: positions && (output('bounds', 'float32x4', groupCapacity, 4) as never),
      groupValueSum: vertexValues && (output('value-sum', 'float32', groupCapacity, 1) as never),
      edges: {
        ids: output('edge-ids', 'uint32', edgeCapacity, 1) as never,
        count: output('edge-count', 'uint32', 1, 1) as never,
        overflow: output('edge-overflow', 'uint32', 1, 1) as never,
        totalCount: output('edge-total', 'uint32', 1, 1) as never
      },
      edgeTargets: output('edge-targets', 'uint32', edgeCapacity, 1) as never,
      edgeCounts: output('edge-counts', 'uint32', edgeCapacity, 1) as never,
      edgeWeights: output('edge-weights', 'float32', edgeCapacity, 1) as never,
      summary: output('summary', 'uint32', 8, 1) as never
    });
    this.graph.add(this.contributor);
  }

  async run(): Promise<Result> {
    this.compiled ??= this.graph.compile();
    submitGraph(this.device, this.compiled, undefined);
    const {groupCapacity: groups, edgeCapacity: edges} = this.options;
    const u = (name: string, length: number) => readUint32(this.out[name], length);
    const f = (name: string, length: number) => readFloat32(this.out[name], length);
    return {
      groupVertexCount: await u('group-counts', groups),
      groupIntraEdgeCount: await u('intra-count', groups),
      groupIntraWeight: await f('intra-weight', groups),
      groupCentroid: this.out['centroid'] ? await f('centroid', groups * 2) : [],
      groupBounds: this.out['bounds'] ? await f('bounds', groups * 4) : [],
      groupValueSum: this.out['value-sum'] ? await f('value-sum', groups) : [],
      count: (await u('edge-count', 1))[0],
      total: (await u('edge-total', 1))[0],
      overflow: (await u('edge-overflow', 1))[0],
      edgeIds: await u('edge-ids', edges),
      edgeTargets: await u('edge-targets', edges),
      edgeCounts: await u('edge-counts', edges),
      edgeWeights: await f('edge-weights', edges),
      summary: await u('summary', 8)
    };
  }

  oracle(): CoarseningOracleResult {
    const {options} = this;
    return computeCoarseningOracle({
      nodeCount: options.nodeCount,
      csr: options.csr,
      labels: options.labels,
      groupCapacity: options.groupCapacity,
      directed: options.directed,
      vertexMask: options.vertexMask,
      edgeMask: options.edgeMask,
      positions: options.positions,
      vertexValues: options.vertexValues,
      unweighted: !options.weighted,
      edgeCapacity: options.edgeCapacity
    });
  }

  destroy(): void {
    this.compiled?.destroy();
    for (const buffer of [...this.buffers, ...Object.values(this.out)]) buffer.destroy();
  }
}

function close(actual: number[], expected: ArrayLike<number>, relative = 2e-6): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < actual.length; index++) {
    const tolerance = relative * Math.max(1, Math.abs(expected[index]));
    expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(tolerance);
  }
}

/** Compares a GPU result to the oracle, honoring the superedge capacity. */
function expectMatches(actual: Result, fixture: Fixture): CoarseningOracleResult {
  const expected = fixture.oracle();
  const {edgeCapacity} = fixture.options;
  expect(actual.groupVertexCount).toEqual(Array.from(expected.groupVertexCount));
  expect(actual.groupIntraEdgeCount).toEqual(Array.from(expected.groupIntraEdgeCount));
  close(actual.groupIntraWeight, expected.groupIntraWeight);
  if (fixture.options.positions) {
    close(actual.groupCentroid, expected.groupCentroid);
    expect(actual.groupBounds).toEqual(Array.from(expected.groupBounds));
  }
  if (fixture.options.vertexValues) close(actual.groupValueSum, expected.groupValueSum);
  expect(actual.summary).toEqual(expected.summary);
  expect(actual.total).toBe(expected.edges.length);
  expect(actual.count).toBe(Math.min(expected.edges.length, edgeCapacity));
  expect(actual.overflow).toBe(expected.overflow ? 1 : 0);
  const kept = expected.edges.slice(0, edgeCapacity);
  expect(actual.edgeIds.slice(0, actual.count)).toEqual(kept.map(edge => edge.source));
  expect(actual.edgeTargets.slice(0, actual.count)).toEqual(kept.map(edge => edge.target));
  expect(actual.edgeCounts.slice(0, actual.count)).toEqual(kept.map(edge => edge.count));
  close(
    actual.edgeWeights.slice(0, actual.count),
    kept.map(edge => edge.weight)
  );
  // Rows past the count are zero.
  expect(actual.edgeIds.slice(actual.count).every(value => value === 0)).toBe(true);
  expect(actual.edgeCounts.slice(actual.count).every(value => value === 0)).toBe(true);
  return expected;
}

function createLabels(nodeCount: number, groups: number, seed: number, wild = 0): Uint32Array {
  const random = createRandom(seed);
  return Uint32Array.from({length: nodeCount}, () =>
    random() < wild ? 1000 + Math.floor(random() * 50) : Math.floor(random() * groups)
  );
}

function createMask(length: number, deadFraction: number, seed: number): Uint32Array {
  const random = createRandom(seed);
  return Uint32Array.from({length}, () => (random() < deadFraction ? 0 : 1));
}

function createPositions(nodeCount: number, seed: number): Float32Array {
  const random = createRandom(seed);
  return Float32Array.from({length: nodeCount * 2}, () => (random() - 0.5) * 360);
}

function createValues(nodeCount: number, seed: number): Float32Array {
  const random = createRandom(seed);
  return Float32Array.from({length: nodeCount}, () => Math.round((random() - 0.3) * 40) / 8);
}

function createCase(
  device: Device,
  seed: number,
  directed: boolean,
  overrides: Partial<FixtureOptions> = {}
): Fixture {
  const nodeCount = 300;
  const edges = createRandomCoarseningEdges(seed, nodeCount, 700, directed);
  const csr = buildCoarseningCSR(nodeCount, edges);
  return new Fixture(device, {
    nodeCount,
    csr,
    labels: createLabels(nodeCount, 16, seed + 1),
    groupCapacity: 16,
    edgeCapacity: 400,
    directed,
    positions: createPositions(nodeCount, seed + 2),
    vertexValues: createValues(nodeCount, seed + 3),
    weighted: true,
    ...overrides
  });
}

for (const directed of [false, true]) {
  it(`GPUNetworkCoarsening matches the oracle on a random ${directed ? 'directed' : 'undirected'} graph`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const fixture = createCase(device, 11, directed);
    const result = await fixture.run();
    const expected = expectMatches(result, fixture);
    expect(expected.edges.length).toBeGreaterThan(20);
    expect(expected.summary[4]).toBeGreaterThan(0);
    const summary = decodeGPUNetworkCoarseningSummary(result.summary);
    expect(summary.liveVertexCount).toBe(300);
    expect(summary.groupCount).toBe(16);
    if (!directed) {
      expect(result.edgeIds.slice(0, result.count).every((s, i) => s < result.edgeTargets[i])).toBe(
        true
      );
    }
    fixture.destroy();
  });
}

it('GPUNetworkCoarsening honors vertex and edge masks and is repeatable', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 300;
  const slotCount = buildCoarseningCSR(
    nodeCount,
    createRandomCoarseningEdges(21, nodeCount, 700, true)
  ).neighbors.length;
  const fixture = createCase(device, 21, true, {
    vertexMask: createMask(nodeCount, 0.2, 5),
    edgeMask: createMask(slotCount, 0.3, 6)
  });
  const first = await fixture.run();
  const expected = expectMatches(first, fixture);
  expect(expected.summary[0]).toBeLessThan(nodeCount);
  const second = await fixture.run();
  expect(second).toEqual(first);
  fixture.destroy();
});

it('GPUNetworkCoarsening handles self-loops and unweighted undirected graphs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const csr = buildCoarseningCSR(5, [
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 2, 1],
    [2, 1, 1],
    [3, 4, 1],
    [4, 3, 1]
  ]);
  const fixture = new Fixture(device, {
    nodeCount: 5,
    csr,
    labels: Uint32Array.from([0, 0, 1, 2, 2]),
    groupCapacity: 3,
    edgeCapacity: 4
  });
  const result = await fixture.run();
  expectMatches(result, fixture);
  expect(result.groupIntraEdgeCount).toEqual([3, 0, 1]);
  expect(result.edgeIds.slice(0, result.count)).toEqual([0]);
  expect(result.edgeTargets.slice(0, result.count)).toEqual([1]);
  expect(result.edgeCounts.slice(0, result.count)).toEqual([1]);
  expect(result.edgeWeights[0]).toBe(1);
  fixture.destroy();
});

it('GPUNetworkCoarsening reports label overflow and excludes those vertices', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const fixture = createCase(device, 31, false, {
    labels: createLabels(300, 16, 32, 0.1)
  });
  const result = await fixture.run();
  const expected = expectMatches(result, fixture);
  expect(expected.summary[1]).toBeGreaterThan(0);
  expect(expected.summary[3]).toBeGreaterThan(0);
  expect(result.overflow).toBe(1);
  fixture.destroy();
});

it('GPUNetworkCoarsening clamps the superedge list and keeps the smallest keys', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const fixture = createCase(device, 41, false, {edgeCapacity: 25});
  const result = await fixture.run();
  const expected = expectMatches(result, fixture);
  expect(expected.edges.length).toBeGreaterThan(25);
  expect(result.count).toBe(25);
  expect(result.total).toBe(expected.edges.length);
  expect(result.overflow).toBe(1);
  fixture.destroy();
});

it('GPUNetworkCoarsening is exact on a larger graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 5000;
  const csr = buildCoarseningCSR(
    nodeCount,
    createRandomCoarseningEdges(51, nodeCount, 20000, false)
  );
  const fixture = new Fixture(device, {
    nodeCount,
    csr,
    labels: createLabels(nodeCount, 200, 52),
    groupCapacity: 256,
    edgeCapacity: 4096,
    positions: createPositions(nodeCount, 53),
    vertexValues: createValues(nodeCount, 54),
    weighted: true
  });
  expectMatches(await fixture.run(), fixture);
  fixture.destroy();
});
