// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  computeAdjacencyMatrixOrder,
  encodeGPUAdjacencyMatrixWindow,
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder
} from '../../../src/gpu-network/adjacency-matrix';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildMatrixCSR,
  computeAdjacencyMatrixOracle,
  createMatrixEdges,
  type MatrixOracleOptions
} from './adjacency-matrix-oracle';

type FixtureOptions = {
  nodeCount: number;
  edges: [number, number, number][];
  resolution: number;
  directed?: boolean;
  mirrorSlots?: boolean;
  vertexMask?: Uint32Array;
  edgeMask?: Uint32Array;
  order?: Uint32Array;
  window?: MatrixOracleOptions['window'];
  withTexture?: boolean;
};

class Fixture {
  readonly graph: GPUCommandGraph;
  readonly csr: ReturnType<typeof buildMatrixCSR>;
  readonly buffers: Buffer[] = [];
  readonly counts: Buffer;
  readonly weightSums: Buffer;
  readonly maxCount: Buffer;
  readonly maxWeightSum: Buffer;
  readonly windowBuffer: Buffer;
  readonly textureReadback: Buffer;
  readonly textureValidity: Buffer;
  readonly texture?: Texture;
  readonly contributor: GPUAdjacencyMatrix;
  compiled?: CompiledGPUCommandGraph<void>;

  constructor(
    readonly device: Device,
    readonly options: FixtureOptions
  ) {
    const {nodeCount, resolution} = options;
    const cells = resolution * resolution;
    this.graph = new GPUCommandGraph(device, {id: 'matrix'});
    this.csr = buildMatrixCSR(nodeCount, options.edges);
    const track = (buffer: Buffer) => {
      this.buffers.push(buffer);
      return buffer;
    };
    const importUint = (name: string, buffer: Buffer, length: number) =>
      importGraphBuffer(this.graph, `m-${name}`, buffer, 'uint32', length);
    const input = (name: string, values: Uint32Array | undefined) =>
      values && importUint(name, track(createInputBuffer(device, values)), values.length);
    this.counts = track(createOutputBuffer(device, cells));
    this.weightSums = track(createOutputBuffer(device, cells));
    this.maxCount = track(createOutputBuffer(device, 1));
    this.maxWeightSum = track(createOutputBuffer(device, 1));
    this.windowBuffer = track(
      createInputBuffer(
        device,
        encodeGPUAdjacencyMatrixWindow(
          options.window ?? {
            rowStart: 0,
            rowEnd: nodeCount,
            colStart: 0,
            colEnd: nodeCount
          },
          resolution
        )
      )
    );
    this.textureReadback = track(createOutputBuffer(device, cells));
    this.textureValidity = track(createOutputBuffer(device, cells));
    const weights = track(createInputBuffer(device, this.csr.weights));
    let textureView;
    if (options.withTexture) {
      this.texture = device.createTexture({
        format: 'r32float',
        width: resolution,
        height: resolution,
        usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
      });
      textureView = this.graph.createTextureView(
        this.graph.importTexture(
          {
            id: 'm-texture',
            format: 'r32float',
            width: resolution,
            height: resolution,
            usage: this.texture.props.usage
          },
          this.texture
        ),
        {mipLevelCount: 1}
      );
    }
    this.contributor = new GPUAdjacencyMatrix({
      id: 'matrix',
      offsets: input('offsets', this.csr.offsets)!,
      neighbors: input('neighbors', this.csr.neighbors)!,
      weights: importGraphBuffer(
        this.graph,
        'm-weights',
        weights,
        'float32',
        this.csr.weights.length
      ),
      directed: options.directed,
      mirrorSlots: options.mirrorSlots,
      vertexMask: input('vertex-mask', options.vertexMask),
      edgeMask: input('edge-mask', options.edgeMask),
      order: input('order', options.order),
      window: importUint('window', this.windowBuffer, 4),
      resolution,
      output: {
        counts: importUint('counts', this.counts, cells),
        weightSums: importUint('weight-sums', this.weightSums, cells),
        maxCount: importUint('max-count', this.maxCount, 1),
        maxWeightSum: importUint('max-weight-sum', this.maxWeightSum, 1),
        texture: textureView as never,
        textureStatistic: 'count'
      }
    });
    this.graph.add(this.contributor);
    if (textureView) {
      new GPURasterTextureToBuffer({
        id: 'texture-readback',
        input: {
          id: 'texture-band',
          format: 'float32',
          storage: {kind: 'texture', view: textureView}
        },
        output: importGraphBuffer(
          this.graph,
          'm-texture-readback',
          this.textureReadback,
          'float32',
          cells
        ),
        outputValidity: importUint('texture-validity', this.textureValidity, cells)
      }).addToGraph(this.graph);
    }
  }

  async run() {
    this.compiled ??= this.graph.compile();
    submitGraph(this.device, this.compiled, undefined);
    const cells = this.options.resolution ** 2;
    return {
      counts: await readUint32(this.counts, cells),
      weightSums: await readUint32(this.weightSums, cells),
      maxCount: (await readUint32(this.maxCount, 1))[0],
      maxWeightSum: (await readUint32(this.maxWeightSum, 1))[0]
    };
  }

  setWindow(window: NonNullable<MatrixOracleOptions['window']>): void {
    this.windowBuffer.write(encodeGPUAdjacencyMatrixWindow(window, this.options.resolution));
  }

  oracle(overrides: Partial<MatrixOracleOptions> = {}) {
    const {options} = this;
    return computeAdjacencyMatrixOracle({
      nodeCount: options.nodeCount,
      csr: this.csr,
      resolution: options.resolution,
      directed: options.directed,
      mirrorSlots: options.mirrorSlots,
      vertexMask: options.vertexMask,
      edgeMask: options.edgeMask,
      order: options.order,
      window: options.window,
      ...overrides
    });
  }

  destroy(): void {
    this.compiled?.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.texture?.destroy();
  }
}

function expectMatches(
  actual: Awaited<ReturnType<Fixture['run']>>,
  expected: ReturnType<Fixture['oracle']>
) {
  expect(actual.counts).toEqual(Array.from(expected.counts));
  expect(actual.weightSums).toEqual(Array.from(expected.weightSums));
  expect(actual.maxCount).toBe(expected.maxCount);
  expect(actual.maxWeightSum).toBe(expected.maxWeightSum);
}

/** Lists every non-self-loop edge in both directions, like an undirected CSR. */
function createBothDirections(edges: [number, number, number][]): [number, number, number][] {
  return edges.flatMap(([a, b, w]) =>
    a === b
      ? [[a, b, w] as [number, number, number]]
      : [
          [a, b, w],
          [b, a, w]
        ]
  );
}

function createPermutation(length: number, seed: number): Uint32Array {
  const order = Uint32Array.from({length}, (_, index) => index);
  let state = seed;
  for (let index = length - 1; index > 0; index--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const other = (state >>> 8) % (index + 1);
    [order[index], order[other]] = [order[other], order[index]];
  }
  return order;
}

function createMask(length: number, deadFraction: number, seed: number): Uint32Array {
  let state = seed;
  return Uint32Array.from({length}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32 < deadFraction ? 0 : 1;
  });
}

it('GPUAdjacencyMatrix matches the oracle with the identity order, undirected and symmetric', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 100;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createBothDirections(createMatrixEdges(3, nodeCount, 400)),
    resolution: 16
  });
  const result = await fixture.run();
  expectMatches(result, fixture.oracle());
  expect(result.maxCount).toBeGreaterThan(0);
  for (let row = 0; row < 16; row++) {
    for (let column = 0; column < 16; column++) {
      expect(result.counts[row * 16 + column]).toBe(result.counts[column * 16 + row]);
      expect(result.weightSums[row * 16 + column]).toBe(result.weightSums[column * 16 + row]);
    }
  }
  fixture.destroy();
});

it('GPUAdjacencyMatrix matches the oracle for directed graphs with a permutation and R not dividing N', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 257;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createMatrixEdges(11, nodeCount, 900),
    resolution: 13,
    directed: true,
    order: createPermutation(nodeCount, 5)
  });
  const result = await fixture.run();
  expectMatches(result, fixture.oracle());
  expect(result.counts.reduce((a, b) => a + b, 0)).toBe(900);
  fixture.destroy();
});

it('GPUAdjacencyMatrix handles masks on a both-directions CSR', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 120;
  const edges = createBothDirections(createMatrixEdges(21, nodeCount, 300));
  const fixture = new Fixture(device, {
    nodeCount,
    edges,
    resolution: 7,
    vertexMask: createMask(nodeCount, 0.2, 5),
    edgeMask: createMask(edges.length, 0.25, 9),
    order: createPermutation(nodeCount, 8)
  });
  const result = await fixture.run();
  expectMatches(result, fixture.oracle());
  expect(result.counts.reduce((a, b) => a + b, 0)).toBeGreaterThan(0);
  fixture.destroy();
});

it('GPUAdjacencyMatrix mirrorSlots makes an edge-list CSR symmetric and equals the both-directions CSR', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 90;
  const edges = createMatrixEdges(51, nodeCount, 350);
  const order = createPermutation(nodeCount, 4);
  const mirrored = new Fixture(device, {
    nodeCount,
    edges,
    resolution: 11,
    mirrorSlots: true,
    order
  });
  const both = new Fixture(device, {
    nodeCount,
    edges: createBothDirections(edges),
    resolution: 11,
    order
  });
  const result = await mirrored.run();
  expectMatches(result, mirrored.oracle());
  const reference = await both.run();
  expect(result.counts).toEqual(reference.counts);
  expect(result.weightSums).toEqual(reference.weightSums);
  for (let row = 0; row < 11; row++) {
    for (let column = 0; column < 11; column++) {
      expect(result.counts[row * 11 + column]).toBe(result.counts[column * 11 + row]);
    }
  }
  mirrored.destroy();
  both.destroy();
});

it('GPUAdjacencyMatrix rewrites bins for a new zoom window without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 200;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createMatrixEdges(31, nodeCount, 1200),
    resolution: 9,
    order: createPermutation(nodeCount, 2)
  });
  expectMatches(await fixture.run(), fixture.oracle());
  const compiled = fixture.compiled;
  for (const window of [
    {rowStart: 20, rowEnd: 83, colStart: 100, colEnd: 190},
    {rowStart: 0, rowEnd: 7, colStart: 0, colEnd: 200},
    {rowStart: 150, rowEnd: 260, colStart: 150, colEnd: 260},
    {rowStart: 10, rowEnd: 10, colStart: 0, colEnd: 200}
  ]) {
    fixture.setWindow(window);
    const result = await fixture.run();
    expectMatches(result, fixture.oracle({window}));
    if (window.rowEnd === window.rowStart) {
      expect(result.maxCount).toBe(0);
    }
  }
  expect(fixture.compiled).toBe(compiled);
  fixture.destroy();
});

it('GPUAdjacencyMatrix culls rows outside a zoom window, including mirrored rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 200;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createMatrixEdges(61, nodeCount, 1500),
    resolution: 8,
    mirrorSlots: true,
    order: createPermutation(nodeCount, 5)
  });
  for (const window of [
    // Row and column ranges that overlap, are disjoint, and sit past the node count.
    {rowStart: 10, rowEnd: 60, colStart: 40, colEnd: 120},
    {rowStart: 0, rowEnd: 30, colStart: 150, colEnd: 200},
    {rowStart: 180, rowEnd: 260, colStart: 0, colEnd: 25}
  ]) {
    fixture.setWindow(window);
    expectMatches(await fixture.run(), fixture.oracle({window}));
  }
  fixture.destroy();
});

it('GPUAdjacencyMatrix writes an r32float texture of the counts', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const nodeCount = 64;
  const fixture = new Fixture(device, {
    nodeCount,
    edges: createMatrixEdges(41, nodeCount, 200),
    resolution: 8,
    directed: true,
    withTexture: true
  });
  const result = await fixture.run();
  expectMatches(result, fixture.oracle());
  expect(await readFloat32(fixture.textureReadback, 64)).toEqual(result.counts);
  fixture.destroy();
});

it('GPUAdjacencyMatrixOrder matches the CPU helper for group and tie keys', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const [count, withTies] of [
    [40, true],
    [700, true],
    [700, false]
  ] as const) {
    let state = 17;
    const next = (limit: number) => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return (state >>> 8) % limit;
    };
    const groups = Uint32Array.from({length: count}, () => next(9));
    const ties = Uint32Array.from({length: count}, () => next(5));
    const graph = new GPUCommandGraph(device, {id: 'order'});
    const groupBuffer = createInputBuffer(device, groups);
    const tieBuffer = createInputBuffer(device, ties);
    const orderBuffer = createOutputBuffer(device, count);
    graph.add(
      new GPUAdjacencyMatrixOrder({
        groups: importGraphBuffer(graph, 'groups', groupBuffer, 'uint32', count),
        tieKeys: withTies
          ? importGraphBuffer(graph, 'ties', tieBuffer, 'uint32', count)
          : undefined,
        order: importGraphBuffer(graph, 'order', orderBuffer, 'uint32', count)
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    expect(await readUint32(orderBuffer, count)).toEqual(
      Array.from(computeAdjacencyMatrixOrder(groups, withTies ? ties : undefined))
    );
    compiled.destroy();
    groupBuffer.destroy();
    tieBuffer.destroy();
    orderBuffer.destroy();
  }
});
