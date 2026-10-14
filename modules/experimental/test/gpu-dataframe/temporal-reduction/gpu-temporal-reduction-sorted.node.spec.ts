// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTemporalReduction,
  type GPUTemporalReductionProps
} from '../../../src/gpu-dataframe/temporal-reduction';
import {
  reduceTemporalBucketsOnCPU,
  type GPUTemporalReductionCPUResult
} from '../../../src/gpu-dataframe/temporal-reduction/temporal-reduction-cpu';
import {
  countDistinctPipelines,
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  ADVERSARIAL_BUCKET_COUNT,
  ADVERSARIAL_WIDTHS,
  createAdversarialEdgeScene
} from './adversarial-edges';
import {searchFloatTemporalRange, searchWordTemporalRange} from './sorted-range-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  rows: number,
  overrides: Partial<GPUTemporalReductionProps> & {words?: boolean; slots?: number} = {}
): GPUTemporalReductionProps {
  const {words, slots = 12, ...rest} = overrides;
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `sorted-view-${serial++}`, format, length);
  return {
    sorted: true,
    cellIds: view('uint32', rows),
    timestamps: words ? view('uint32x2', rows) : view('float32', rows),
    values: view('float32', rows),
    mask: view('uint32', rows),
    parameters: words ? view('uint32', 4) : view('float32', 2),
    cellCount: 3,
    bucketCount: slots / 3,
    output: {
      counts: view('uint32', slots),
      minimums: view('float32', slots),
      maximums: view('float32', slots),
      first: view('float32', slots),
      last: view('float32', slots),
      occupiedSlots: {
        ids: view('uint32', slots),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    },
    ...rest
  } as GPUTemporalReductionProps;
}

it('GPUTemporalReduction sorted gates the row kernels on a GPU-written range', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUTemporalReduction(createProps(graph, 100)).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.slice(0, 3)).toEqual([
    'temporal-reduction-range-search',
    'temporal-reduction-range-dispatch',
    'temporal-reduction-init'
  ]);
  for (const gated of ['classify', 'extremes', 'tie-rows']) {
    const node = nodes.find(candidate => candidate.id === `temporal-reduction-${gated}`) as
      | {condition?: {source: string; mode: string}}
      | undefined;
    expect(node?.condition, gated).toMatchObject({source: 'gpu', mode: 'indirect'});
  }
  // The unsorted default has none of these.
  const plainGraph = new GPUCommandGraph(device);
  const plain = new GPUTemporalReduction(
    createProps(plainGraph, 100, {sorted: undefined})
  ).getCommandNodes(plainGraph);
  expect(plain.map(node => node.id).some(id => id.includes('range'))).toBe(false);
  device.destroy();
});

it('GPUTemporalReduction sorted kernels parse and stay within the storage binding limit', () => {
  const device = createNullWebGPUDevice();
  // 12 slots uses the privatized extreme kernel, 3000 slots the global-atomics one.
  for (const [words, slots] of [
    [false, 12],
    [true, 12],
    [false, 3000],
    [true, 3000]
  ] as const) {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUTemporalReduction(
      createProps(graph, 100, {words, slots, bucketCount: slots / 3})
    ).getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    for (const [id, source] of sources) {
      expect(getStorageBindingCount(source), `${id} ${words} ${slots}`).toBeLessThanOrEqual(8);
    }
    const search = sources.get('temporal-reduction-range-search')!;
    expect(search).toContain('loop {');
    expect(search.includes('isQuotientAtLeastBucketCount')).toBe(words);
    // The row index offset comes from a GPU word, not from WGSL text.
    expect(sources.get('temporal-reduction-classify')).toContain('rowSlots[rowSlotsOffset]');
    expect(sources.get('temporal-reduction-tie-rows')).toContain('rowSlots[rowSlotsOffset]');
  }
  device.destroy();
});

it('GPUTemporalReduction sorted builds the same pipelines at N and N + 1 rows', () => {
  const device = createNullWebGPUDevice();
  const count = (rows: number, words: boolean) => {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUTemporalReduction(createProps(graph, rows, {words})).getCommandNodes(
      graph
    );
    return countDistinctPipelines(nodes);
  };
  // Rule 12: the row count and the first row never reach WGSL text; both are GPU words or overrides.
  expect(count(1000, false)).toBe(count(1001, false));
  expect(count(1000, true)).toBe(count(1001, true));
  device.destroy();
});

function expectSameReduction(
  actual: GPUTemporalReductionCPUResult,
  expected: GPUTemporalReductionCPUResult
) {
  expect(Array.from(actual.count)).toEqual(Array.from(expected.count));
  for (const name of ['minimums', 'maximums', 'first', 'last'] as const) {
    const bits = (values: Float32Array) =>
      Array.from(new Uint32Array(values.buffer)).map(bit => (bit === 0x7fc00000 ? 'nan' : bit));
    expect(bits(actual[name]), name).toEqual(bits(expected[name]));
  }
}

/** Reduces the range rows only, as the sorted kernels do, and compares with reducing all rows. */
function expectRangeEqualsFullReduction(
  scene: {
    cellIds: Uint32Array;
    timestamps: Float32Array | BigInt64Array;
    values: Float32Array;
    cellCount: number;
    bucketCount: number;
  },
  range: {firstRow: number; rowCount: number},
  origin: number | bigint,
  width: number | bigint
) {
  const end = range.firstRow + range.rowCount;
  const full = reduceTemporalBucketsOnCPU({...scene, origin, width});
  const sliced = reduceTemporalBucketsOnCPU({
    ...scene,
    cellIds: scene.cellIds.subarray(range.firstRow, end),
    timestamps: scene.timestamps.subarray(range.firstRow, end) as never,
    values: scene.values.subarray(range.firstRow, end),
    origin,
    width
  });
  expectSameReduction(sliced, full);
  // Every row of the range lands in a bucket, so the counts add up to the range size exactly
  // when no row is skipped for another reason (these scenes have none).
  expect(full.count.reduce((sum, value) => sum + value, 0)).toBe(range.rowCount);
}

it('sorted range oracle: float range equals the rows the bucket assignment keeps (adversarial edges)', () => {
  for (const width of ADVERSARIAL_WIDTHS) {
    const scene = createAdversarialEdgeScene(width);
    for (const bucketCount of [1, 7, 100, ADVERSARIAL_BUCKET_COUNT - 1]) {
      // Origins at edges, between them, negative, and past the data.
      for (const origin of [
        0,
        width,
        Math.fround(50 * width),
        Math.fround(50 * width) + width / 3,
        -3 * width,
        1e6
      ]) {
        const range = searchFloatTemporalRange(scene.timestamps, bucketCount, origin, width);
        expectRangeEqualsFullReduction({...scene, bucketCount}, range, origin, width);
      }
    }
  }
});

it('sorted range oracle: float edge cases', () => {
  const times = Float32Array.from([-4, -1, -0, 0, 0.5, 1, 1, 2, 3.999, 4, 8]);
  const make = (bucketCount: number, origin: number, width: number) => {
    const range = searchFloatTemporalRange(times, bucketCount, origin, width);
    expectRangeEqualsFullReduction(
      {
        cellIds: new Uint32Array(times.length),
        timestamps: times,
        values: Float32Array.from(times, (_, row) => row + 1),
        cellCount: 1,
        bucketCount
      },
      range,
      origin,
      width
    );
    return range;
  };
  expect(make(4, 0, 1)).toEqual({firstRow: 2, rowCount: 7}); // -0 and 0 are in; 4 is out
  expect(make(4, 4, 1).rowCount).toBe(1); // 4 is in; 8 is exactly 4 buckets past the origin
  expect(make(1000, -100, 1).rowCount).toBe(times.length); // all in
  expect(make(1, 1000, 1).rowCount).toBe(0); // all below
  expect(make(1, -1000, 1).rowCount).toBe(0); // all above
  expect(make(4, 0, 0).rowCount).toBe(0); // invalid width
  expect(make(4, 0, -1).rowCount).toBe(0);
  expect(make(4, 0, Infinity).rowCount).toBe(0);
  expect(make(4, NaN, 1).rowCount).toBe(0);
  expect(make(4, 0, 1e-45).rowCount).toBe(2); // a denormal width keeps only -0 and 0
});

it('sorted range oracle: Int64 word range with negative times and duplicates', () => {
  const times = BigInt64Array.from([
    -(2n ** 62n),
    -5000n,
    -3000n,
    -3000n,
    0n,
    0n,
    999n,
    1000n,
    1000n,
    4_294_967_295n,
    4_294_967_296n,
    2n ** 62n
  ]);
  const scene = (bucketCount: number) => ({
    cellIds: new Uint32Array(times.length),
    timestamps: times,
    values: Float32Array.from(times, (_, row) => row + 1),
    cellCount: 1,
    bucketCount
  });
  for (const [bucketCount, origin, width] of [
    [4, -3000n, 1000n],
    [4, 0n, 1000n],
    [1, 1000n, 1n],
    [2, -(2n ** 62n), 2n ** 61n],
    [10, 4_294_967_290n, 3n],
    [100, -(2n ** 63n), 0xffffffffn],
    [3, 2n ** 62n, 7n],
    [3, 2n ** 62n + 1n, 7n]
  ] as const) {
    const range = searchWordTemporalRange(times, bucketCount, origin, width);
    expectRangeEqualsFullReduction(scene(bucketCount), range, origin, width);
  }
  expect(searchWordTemporalRange(times, 4, 0n, 1000n)).toEqual({firstRow: 4, rowCount: 5});
  expect(searchWordTemporalRange(times, 4, 0n, 0n).rowCount).toBe(0);
});
