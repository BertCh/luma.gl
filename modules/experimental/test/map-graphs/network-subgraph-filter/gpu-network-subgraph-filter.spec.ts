// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  decodeGPUNetworkSubgraphFilterCounts,
  getGPUNetworkSubgraphFilterParameterValues,
  GPUNetworkSubgraphFilter,
  type GPUNetworkSubgraphFilterState
} from '../../../src/map-graphs/network-subgraph-filter';
import {
  getGPUTimeWindowWordParameterValues,
  getInt64TimeWords
} from '../../../src/map-graphs/time-window-filter/time-words';
import {createInputBuffer, createOutputBuffer, readUint32} from '../map-graph-test-utils';
import {createSymmetricEdges} from '../network-analysis/network-analytics-oracle';
import {
  buildCSR,
  createRandomNetwork,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';
import {computeSubgraphOracle, type SubgraphOracleResult} from './network-subgraph-filter-oracle';

type FixtureOptions = {
  nodeCount: number;
  edges: readonly NetworkEdge[];
  directed?: boolean;
  vertexColumnCount?: number;
  edgeColumnCount?: number;
  vertexMask?: Uint32Array;
  edgeMask?: Uint32Array;
  edgeTimes?: boolean;
  edgeTimeWords?: boolean;
  dropIsolated?: boolean;
  /** Edge columns hold equal values in both slots of an undirected edge. */
  symmetric?: boolean;
  pairUndirectedSlots?: boolean;
  liveVertexCapacity?: number;
  liveSlotCapacity?: number;
  inducedCapacity?: number;
};

let randomState = 1;
function random(): number {
  randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
  return randomState / 2 ** 32;
}

function createMask(length: number, deadFraction: number, seed: number): Uint32Array {
  randomState = seed;
  return Uint32Array.from({length}, () => (random() < deadFraction ? 0 : 1));
}

class Fixture {
  readonly graph: GPUCommandGraph;
  readonly csr: ReturnType<typeof buildCSR>;
  readonly buffers: Buffer[] = [];
  readonly recipe: GPUNetworkSubgraphFilter;
  readonly slotCount: number;
  readonly vertexColumns: Float32Array[] = [];
  readonly edgeColumns: Float32Array[] = [];
  readonly edgeTimes?: Float32Array;
  readonly edgeTimeWords?: BigInt64Array;
  readonly buffer: Record<string, Buffer> = {};
  compiled?: CompiledGPUCommandGraph<void>;
  private readonly layout: {
    vertexColumnCount: number;
    edgeColumnCount: number;
    hasEdgeTimes: boolean;
  };

  constructor(
    readonly device: Device,
    readonly options: FixtureOptions
  ) {
    const {nodeCount} = options;
    this.graph = new GPUCommandGraph(device, {id: 'subgraph'});
    this.csr = buildCSR(nodeCount, options.edges);
    const slotCount = (this.slotCount = this.csr.neighbors.length);
    const track = (name: string, buffer: Buffer) => {
      this.buffers.push(buffer);
      this.buffer[name] = buffer;
      return buffer;
    };
    const input = (name: string, values: Uint32Array | Float32Array) =>
      track(name, createInputBuffer(device, values));
    const view = <F extends 'uint32' | 'float32' | 'uint32x2'>(
      name: string,
      buffer: Buffer,
      length: number,
      format: F
    ) => importGraphBuffer(this.graph, `sub-${name}`, buffer, format, length);
    const output = (name: string, length: number) =>
      view(name, track(name, createOutputBuffer(device, length)), length, 'uint32');

    // Per-slot source vertex, for symmetric columns.
    const sources = new Uint32Array(slotCount);
    for (let u = 0; u < nodeCount; u++) {
      for (let slot = this.csr.offsets[u]; slot < this.csr.offsets[u + 1]; slot++)
        sources[slot] = u;
    }
    const pairValue = (u: number, v: number, salt: number) => {
      randomState = (Math.min(u, v) * 7919 + Math.max(u, v) * 104729 + salt * 1299709 + 17) >>> 0;
      random();
      return Math.fround(random() * 100);
    };
    randomState = 99;
    for (let i = 0; i < (options.vertexColumnCount ?? 0); i++) {
      this.vertexColumns.push(Float32Array.from({length: nodeCount}, () => random() * 100));
    }
    for (let i = 0; i < (options.edgeColumnCount ?? 0); i++) {
      this.edgeColumns.push(
        Float32Array.from({length: slotCount}, (_, slot) =>
          options.symmetric ? pairValue(sources[slot], this.csr.neighbors[slot], i) : random() * 100
        )
      );
    }
    if (options.edgeTimes) {
      this.edgeTimes = Float32Array.from({length: slotCount}, (_, slot) =>
        options.symmetric ? pairValue(sources[slot], this.csr.neighbors[slot], 77) : random() * 100
      );
    }
    if (options.edgeTimeWords) {
      // Epoch-millisecond scale values that f32 cannot hold exactly.
      this.edgeTimeWords = BigInt64Array.from(
        {length: slotCount},
        (_, slot) =>
          1_700_000_000_000n +
          BigInt(Math.floor(pairValue(sources[slot], this.csr.neighbors[slot], 5) * 10))
      );
    }
    this.layout = {
      vertexColumnCount: this.vertexColumns.length,
      edgeColumnCount: this.edgeColumns.length,
      hasEdgeTimes: Boolean(this.edgeTimes)
    };
    const parameterLength = Math.max(
      4,
      (this.vertexColumns.length + this.edgeColumns.length + (this.edgeTimes ? 1 : 0)) * 4
    );
    track('parameters', createInputBuffer(device, new Float32Array(parameterLength)));
    if (options.edgeTimeWords) {
      track('timeWordParameters', createInputBuffer(device, new Uint32Array(8)));
    }

    const liveVertexCapacity = options.liveVertexCapacity ?? nodeCount;
    const liveSlotCapacity = options.liveSlotCapacity ?? slotCount;
    const inducedCapacity = options.inducedCapacity ?? slotCount;
    const compact = (name: string, capacity: number) => ({
      ids: output(`${name}-ids`, capacity),
      count: output(`${name}-count`, 1),
      overflow: output(`${name}-overflow`, 1),
      totalCount: output(`${name}-total`, 1)
    });
    this.recipe = new GPUNetworkSubgraphFilter({
      id: 'sub',
      offsets: view('offsets', input('offsets', this.csr.offsets), nodeCount + 1, 'uint32'),
      neighbors: view('neighbors', input('neighbors', this.csr.neighbors), slotCount, 'uint32'),
      directed: options.directed,
      vertexMask: options.vertexMask
        ? view('vertex-mask', input('vertexMaskIn', options.vertexMask), nodeCount, 'uint32')
        : undefined,
      edgeMask: options.edgeMask
        ? view('edge-mask', input('edgeMaskIn', options.edgeMask), slotCount, 'uint32')
        : undefined,
      vertexColumns: this.vertexColumns.map((values, i) =>
        view(`vc${i}`, input(`vc${i}`, values), nodeCount, 'float32')
      ),
      edgeColumns: this.edgeColumns.map((values, i) =>
        view(`ec${i}`, input(`ec${i}`, values), slotCount, 'float32')
      ),
      edgeTimes: this.edgeTimes
        ? view('times', input('times', this.edgeTimes), slotCount, 'float32')
        : undefined,
      edgeTimeWords: this.edgeTimeWords
        ? view(
            'words',
            input('words', getInt64TimeWords(this.edgeTimeWords)),
            slotCount,
            'uint32x2'
          )
        : undefined,
      parameters: parameterLength
        ? view('parameters', this.buffer.parameters, parameterLength, 'float32')
        : undefined,
      timeWordParameters: options.edgeTimeWords
        ? view('twp', this.buffer.timeWordParameters, 8, 'uint32')
        : undefined,
      dropIsolated: options.dropIsolated,
      pairUndirectedSlots: options.pairUndirectedSlots,
      output: {
        vertexMask: output('vertex-mask-out', nodeCount),
        edgeMask: output('edge-mask-out', slotCount),
        counts: output('counts', 4),
        liveVertices: compact('lv', liveVertexCapacity),
        liveEdgeSlots: compact('ls', liveSlotCapacity),
        inducedCSR: {
          offsets: output('io', nodeCount + 1),
          neighbors: output('in', inducedCapacity),
          sourceSlots: output('is', inducedCapacity),
          overflow: output('iov', 1)
        }
      }
    });
    this.graph.add(this.recipe);
  }

  setState(state: GPUNetworkSubgraphFilterState): void {
    this.buffer.parameters.write(getGPUNetworkSubgraphFilterParameterValues(this.layout, state));
    this.state = state;
  }

  setTimeWindow(window: {start: number | bigint; end: number | bigint}): void {
    this.buffer.timeWordParameters.write(getGPUTimeWindowWordParameterValues(window));
    this.window = window;
  }

  state: GPUNetworkSubgraphFilterState = {};
  window?: {start: number | bigint; end: number | bigint};

  oracle(): SubgraphOracleResult {
    const {options, state} = this;
    return computeSubgraphOracle({
      nodeCount: options.nodeCount,
      offsets: this.csr.offsets,
      neighbors: this.csr.neighbors,
      directed: options.directed,
      vertexMask: options.vertexMask,
      edgeMask: options.edgeMask,
      vertexColumns: this.vertexColumns,
      vertexRanges: state.vertexRanges,
      edgeColumns: this.edgeColumns,
      edgeRanges: state.edgeRanges,
      edgeTimes: this.edgeTimes,
      edgeTimeWindow: state.edgeTimeWindow,
      edgeTimeWords: this.edgeTimeWords,
      timeWordWindow: this.window,
      dropIsolated: options.dropIsolated,
      pairUndirectedSlots: options.pairUndirectedSlots
    });
  }

  /** Encodes the same compiled graph once and returns every output. */
  async run() {
    this.compiled ??= this.graph.compile();
    submitGraph(this.device, this.compiled, undefined);
    const {nodeCount} = this.options;
    const {slotCount} = this;
    const read = (name: string, length: number) => readUint32(this.buffer[name], length);
    const capacity = (name: string) => this.buffer[name].byteLength / 4;
    const [lvCount] = await read('lv-count', 1);
    const [lsCount] = await read('ls-count', 1);
    return {
      vertexMask: await read('vertex-mask-out', nodeCount),
      edgeMask: await read('edge-mask-out', slotCount),
      counts: await read('counts', 4),
      lv: {
        ids: (await read('lv-ids', capacity('lv-ids'))).slice(0, lvCount),
        count: lvCount,
        overflow: (await read('lv-overflow', 1))[0],
        total: (await read('lv-total', 1))[0]
      },
      ls: {
        ids: (await read('ls-ids', capacity('ls-ids'))).slice(0, lsCount),
        count: lsCount,
        overflow: (await read('ls-overflow', 1))[0],
        total: (await read('ls-total', 1))[0]
      },
      inducedOffsets: await read('io', nodeCount + 1),
      inducedNeighbors: await read('in', capacity('in')),
      inducedSlots: await read('is', capacity('is')),
      inducedOverflow: (await read('iov', 1))[0]
    };
  }

  destroy(): void {
    this.compiled?.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}

type RunResult = Awaited<ReturnType<Fixture['run']>>;

function expectMatches(actual: RunResult, expected: SubgraphOracleResult, caps = {}): void {
  expect(actual.vertexMask).toEqual(Array.from(expected.vertexMask));
  expect(actual.edgeMask).toEqual(Array.from(expected.edgeMask));
  expect(actual.counts).toEqual(expected.counts);
  expect(decodeGPUNetworkSubgraphFilterCounts(actual.counts).liveSlotCount).toBe(
    expected.liveSlotIds.length
  );
  const lvCapacity = (caps as {lv?: number}).lv ?? Infinity;
  const lsCapacity = (caps as {ls?: number}).ls ?? Infinity;
  expect(actual.lv.total).toBe(expected.liveVertexIds.length);
  expect(actual.lv.count).toBe(Math.min(lvCapacity, expected.liveVertexIds.length));
  expect(actual.lv.overflow).toBe(expected.liveVertexIds.length > lvCapacity ? 1 : 0);
  expect(actual.lv.ids).toEqual(expected.liveVertexIds.slice(0, actual.lv.count));
  expect(actual.ls.total).toBe(expected.liveSlotIds.length);
  expect(actual.ls.count).toBe(Math.min(lsCapacity, expected.liveSlotIds.length));
  expect(actual.ls.overflow).toBe(expected.liveSlotIds.length > lsCapacity ? 1 : 0);
  expect(actual.ls.ids).toEqual(expected.liveSlotIds.slice(0, actual.ls.count));
  expect(actual.inducedOffsets).toEqual(Array.from(expected.inducedOffsets));
  const inducedCapacity = actual.inducedNeighbors.length;
  const written = Math.min(inducedCapacity, expected.inducedNeighbors.length);
  expect(actual.inducedOverflow).toBe(expected.inducedNeighbors.length > inducedCapacity ? 1 : 0);
  expect(actual.inducedNeighbors.slice(0, written)).toEqual(
    expected.inducedNeighbors.slice(0, written)
  );
  expect(actual.inducedSlots.slice(0, written)).toEqual(expected.inducedSlots.slice(0, written));
}

const RANGES: GPUNetworkSubgraphFilterState = {
  vertexRanges: [
    [10, 90],
    [0, 70]
  ],
  edgeRanges: [[5, 95], null, [20, Infinity]],
  edgeTimeWindow: [10, 80]
};

it('GPUNetworkSubgraphFilter matches the oracle on a directed random graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 300;
  const edges = [...createRandomNetwork(11, nodeCount, 900), [4, 4, 1] as NetworkEdge];
  const fixture = new Fixture(device, {
    nodeCount,
    edges,
    directed: true,
    vertexColumnCount: 2,
    edgeColumnCount: 3,
    edgeTimes: true,
    vertexMask: createMask(nodeCount, 0.1, 5),
    edgeMask: createMask(edges.length, 0.1, 9)
  });
  fixture.setState(RANGES);
  const actual = await fixture.run();
  const expected = fixture.oracle();
  expectMatches(actual, expected);
  expect(expected.counts[0]).toBeGreaterThan(0);
  expect(expected.counts[0]).toBeLessThan(nodeCount);
  expect(expected.counts[1]).toBeGreaterThan(0);
  expect(expected.counts[1]).toBeLessThan(edges.length);
  // Every live slot has two live endpoints.
  for (const slot of expected.liveSlotIds) {
    expect(actual.vertexMask[fixture.csr.neighbors[slot]]).toBe(1);
  }
  fixture.destroy();
});

it('GPUNetworkSubgraphFilter matches the oracle on an undirected graph with consistent slots', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 250;
  const edges = [
    ...createSymmetricEdges(createRandomNetwork(3, nodeCount, 500)),
    [7, 7, 1] as NetworkEdge
  ];
  for (const dropIsolated of [false, true]) {
    const fixture = new Fixture(device, {
      nodeCount,
      edges,
      vertexColumnCount: 2,
      edgeColumnCount: 3,
      edgeTimes: true,
      symmetric: true,
      dropIsolated
    });
    fixture.setState({
      vertexRanges: [[0, 85], null],
      edgeRanges: [[10, 90], [0, 100], null],
      edgeTimeWindow: [5, 95]
    });
    const actual = await fixture.run();
    const expected = fixture.oracle();
    expectMatches(actual, expected);
    // Both slots of every undirected edge agree.
    const {offsets, neighbors} = fixture.csr;
    const slotOf = new Map<string, number>();
    for (let u = 0; u < nodeCount; u++) {
      for (let slot = offsets[u]; slot < offsets[u + 1]; slot++) {
        slotOf.set(`${u}:${neighbors[slot]}`, slot);
      }
    }
    for (const [key, slot] of slotOf) {
      const [u, v] = key.split(':').map(Number);
      expect(actual.edgeMask[slot]).toBe(actual.edgeMask[slotOf.get(`${v}:${u}`)!]);
    }
    const counts = decodeGPUNetworkSubgraphFilterCounts(actual.counts);
    expect(counts.selfLoopSlotCount).toBeLessThanOrEqual(1);
    expect(counts.liveEdgeCount).toBe(
      (counts.liveSlotCount - counts.selfLoopSlotCount) / 2 + counts.selfLoopSlotCount
    );
    if (dropIsolated) {
      // No live vertex is isolated.
      const degree = new Uint32Array(nodeCount);
      for (let u = 0; u < nodeCount; u++) {
        for (let slot = offsets[u]; slot < offsets[u + 1]; slot++) {
          if (actual.edgeMask[slot]) degree[u]++;
        }
      }
      for (let v = 0; v < nodeCount; v++) {
        if (actual.vertexMask[v]) expect(degree[v]).toBeGreaterThan(0);
      }
    }
    fixture.destroy();
  }
});

it('GPUNetworkSubgraphFilter changes ranges between encodings without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 200;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createRandomNetwork(31, nodeCount, 600),
    directed: true,
    vertexColumnCount: 1,
    edgeColumnCount: 1,
    edgeTimes: true,
    dropIsolated: true
  });
  const states: GPUNetworkSubgraphFilterState[] = [
    {},
    {vertexRanges: [[20, 60]]},
    {edgeRanges: [[0, 30]], edgeTimeWindow: [0, 50]},
    {
      vertexRanges: [[0, 100.5]],
      edgeRanges: [[40, 100]],
      edgeTimeWindow: [25, 75]
    },
    {vertexRanges: [[90, 10]]},
    {}
  ];
  let compiled: unknown;
  const liveCounts: number[] = [];
  for (const state of states) {
    fixture.setState(state);
    const actual = await fixture.run();
    expectMatches(actual, fixture.oracle());
    liveCounts.push(actual.counts[1]);
    compiled ??= fixture.compiled;
    expect(fixture.compiled).toBe(compiled);
  }
  expect(new Set(liveCounts).size).toBeGreaterThan(3);
  expect(liveCounts[4]).toBe(0);
  expect(liveCounts[0]).toBe(liveCounts[5]);
  fixture.destroy();
});

it('GPUNetworkSubgraphFilter filters exact Int64 word times and edits the window', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 150;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createRandomNetwork(41, nodeCount, 500),
    directed: true,
    edgeTimeWords: true
  });
  const base = 1_700_000_000_000;
  const windows = [
    {start: base + 100, end: base + 600},
    {start: BigInt(base) + 250n, end: BigInt(base) + 251n},
    {start: base - 5, end: base + 2000},
    {start: base + 3000, end: base + 4000},
    {start: base + 300.5, end: base + 700}
  ];
  const liveCounts: number[] = [];
  for (const window of windows) {
    fixture.setTimeWindow(window);
    const actual = await fixture.run();
    expectMatches(actual, fixture.oracle());
    liveCounts.push(actual.counts[1]);
  }
  expect(liveCounts[0]).toBeGreaterThan(0);
  expect(liveCounts[2]).toBeGreaterThanOrEqual(liveCounts[0]);
  expect(liveCounts[3]).toBe(0);
  fixture.destroy();
});

it('GPUNetworkSubgraphFilter handles more columns than one kernel can bind', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 120;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createRandomNetwork(51, nodeCount, 700),
    directed: true,
    vertexColumnCount: 8,
    edgeColumnCount: 9
  });
  fixture.setState({
    vertexRanges: Array.from({length: 8}, () => [5, 99] as const),
    edgeRanges: Array.from({length: 9}, () => [3, 99] as const)
  });
  const actual = await fixture.run();
  expectMatches(actual, fixture.oracle());
  expect(actual.counts[1]).toBeGreaterThan(0);
  fixture.destroy();
});

it('GPUNetworkSubgraphFilter reports compact and induced overflow without corrupting results', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 200;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createRandomNetwork(61, nodeCount, 600),
    directed: true,
    edgeColumnCount: 1,
    liveVertexCapacity: 20,
    liveSlotCapacity: 50,
    inducedCapacity: 40
  });
  fixture.setState({edgeRanges: [[0, 80]]});
  const actual = await fixture.run();
  const expected = fixture.oracle();
  expectMatches(actual, expected, {lv: 20, ls: 50});
  expect(actual.lv.overflow).toBe(1);
  expect(actual.ls.overflow).toBe(1);
  expect(actual.inducedOverflow).toBe(1);
  expect(actual.lv.count).toBe(20);
  expect(actual.ls.total).toBeGreaterThan(50);
  fixture.destroy();
});

function getSlotMap(fixture: Fixture): Map<string, number[]> {
  const {offsets, neighbors} = fixture.csr;
  const slots = new Map<string, number[]>();
  for (let u = 0; u < fixture.options.nodeCount; u++) {
    for (let slot = offsets[u]; slot < offsets[u + 1]; slot++) {
      const key = `${u}:${neighbors[slot]}`;
      slots.set(key, [...(slots.get(key) ?? []), slot]);
    }
  }
  return slots;
}

it('GPUNetworkSubgraphFilter pairs undirected slots whose per-slot values differ', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 120;
  // Asymmetric columns, caller edge masks and a parallel edge pair.
  const edges = [
    ...createSymmetricEdges(createRandomNetwork(71, nodeCount, 300)),
    [3, 9, 1] as NetworkEdge,
    [9, 3, 1] as NetworkEdge,
    [3, 9, 1] as NetworkEdge,
    [9, 3, 1] as NetworkEdge,
    [6, 6, 1] as NetworkEdge
  ];
  const slotCount = edges.length;
  for (const pairUndirectedSlots of [true, false]) {
    const fixture = new Fixture(device, {
      nodeCount,
      edges,
      edgeColumnCount: 1,
      edgeTimes: true,
      edgeMask: createMask(slotCount, 0.1, 13),
      pairUndirectedSlots
    });
    fixture.setState({edgeRanges: [[10, 90]], edgeTimeWindow: [10, 90]});
    const actual = await fixture.run();
    expectMatches(actual, fixture.oracle());
    const slots = getSlotMap(fixture);
    let disagreements = 0;
    for (const [key, forward] of slots) {
      const [u, v] = key.split(':').map(Number);
      const reverse = slots.get(`${v}:${u}`) ?? [];
      for (const [i, slot] of forward.entries()) {
        if (u !== v && actual.edgeMask[slot] !== actual.edgeMask[reverse[i]]) disagreements++;
      }
    }
    // Paired: always symmetric. Unpaired: asymmetric inputs leak through.
    expect(disagreements === 0).toBe(pairUndirectedSlots);
    fixture.destroy();
  }
});
