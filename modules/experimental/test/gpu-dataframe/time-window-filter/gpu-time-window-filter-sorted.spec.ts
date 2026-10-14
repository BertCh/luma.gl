// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUTimeWindowFilterParameterValues,
  getGPUTimeWindowFilterWordParameterValues,
  getInt64TimeWords,
  GPUTimeWindowFilter,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
  GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,
  splitTimestamps
} from '../../../src/gpu-dataframe/time-window-filter';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

type Window = {start: number; end: number};

type RunResult = {
  ids: number[];
  count: number;
  overflow: number;
  requiredCount: number;
  mask?: number[];
  rowRange?: [number, number];
  drawInstanceCount?: number;
  drawFirstInstance?: number;
  orderViolations?: number;
};

type Fixture = {
  run(window: Window): Promise<RunResult>;
  destroy(): void;
};

type FixtureOptions = {
  sorted: boolean;
  /** Float times (with optional double-single low parts) or Int64 words. */
  times: {high: Float32Array; low?: Float32Array} | {words: BigInt64Array};
  capacity: number;
  withMask?: boolean;
  withSourceIds?: boolean;
  validateOrder?: boolean;
  /** Skip the sorted-only outputs (the mask path rejects them). */
  withRange?: boolean;
};

function createFixture(device: Device, options: FixtureOptions): Fixture {
  const isWords = 'words' in options.times;
  const rows = isWords
    ? (options.times as {words: BigInt64Array}).words.length
    : (options.times as {high: Float32Array}).high.length;
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: `time-window-sorted-${options.sorted}`});
  const window = new GPUParameterBuffer(device, {
    id: 'window',
    format: isWords ? 'uint32' : 'float32',
    length: isWords
      ? GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH
      : GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH
  });
  const outputs = {
    ids: output(options.capacity),
    count: output(1),
    overflow: output(1),
    total: output(1),
    mask: output(rows),
    range: output(2),
    violations: output(1)
  };
  const drawCommands = new DrawCommandBuffer(device, {
    id: 'time-window-draw',
    type: 'draw',
    commands: [{vertexCount: 6, instanceCount: 0, firstInstance: 0}]
  });
  // The record's firstInstance is its last word; imported as a packed one-row view like the count.
  const firstInstanceByteOffset = drawCommands.getCommandByteOffset(0) + 12;
  const firstInstanceData = new GPUData({
    buffer: drawCommands.buffer,
    format: 'uint32',
    length: 1,
    byteOffset: firstInstanceByteOffset,
    byteStride: 4,
    rowByteLength: 4,
    ownsBuffer: false
  });
  const timestamps = isWords
    ? importGraphBuffer(
        graph,
        'times',
        input(getInt64TimeWords((options.times as {words: BigInt64Array}).words)),
        'uint32x2',
        rows
      )
    : importGraphBuffer(
        graph,
        'times',
        input((options.times as {high: Float32Array}).high),
        'float32',
        rows
      );
  const low = !isWords ? (options.times as {low?: Float32Array}).low : undefined;
  const withRange = options.sorted && options.withRange !== false;
  graph.add(
    new GPUTimeWindowFilter({
      id: 'filter',
      sorted: options.sorted,
      validateOrder: options.validateOrder,
      timestamps: timestamps as never,
      timestampsLow: low ? importGraphBuffer(graph, 'low', input(low), 'float32', rows) : undefined,
      sourceIds: options.withSourceIds
        ? importGraphBuffer(
            graph,
            'source-ids',
            input(Uint32Array.from({length: rows}, (_, row) => 1000 + row)),
            'uint32',
            rows
          )
        : undefined,
      window: window.importToGraph(graph) as never,
      output: {
        ids: importGraphBuffer(graph, 'ids', outputs.ids, 'uint32', options.capacity),
        status: {
          count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'total', outputs.total, 'uint32', 1)
        },
        outputMask: options.withMask
          ? importGraphBuffer(graph, 'mask', outputs.mask, 'uint32', rows)
          : undefined,
        rowRange: withRange
          ? importGraphBuffer(graph, 'range', outputs.range, 'uint32', 2)
          : undefined,
        drawInstanceCount: graph.importGPUData('draw-count', drawCommands.getInstanceCountData(0)),
        drawFirstInstance: withRange
          ? graph.importGPUData('draw-first', firstInstanceData)
          : undefined,
        orderViolations: options.validateOrder
          ? importGraphBuffer(graph, 'violations', outputs.violations, 'uint32', 1)
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  const readDrawWord = async (byteOffset: number) => {
    const bytes = await drawCommands.buffer.readAsync(byteOffset, 4);
    return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
  };
  return {
    async run({start, end}) {
      window.write(
        isWords
          ? getGPUTimeWindowFilterWordParameterValues({start, end})
          : getGPUTimeWindowFilterParameterValues({start, end})
      );
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      const result: RunResult = {
        ids: await readUint32(outputs.ids, count),
        count,
        overflow: (await readUint32(outputs.overflow, 1))[0],
        requiredCount: (await readUint32(outputs.total, 1))[0],
        drawInstanceCount: await readDrawWord(drawCommands.getInstanceCountByteOffset(0))
      };
      if (options.withMask) {
        result.mask = await readUint32(outputs.mask, rows);
      }
      if (withRange) {
        const [firstRow, rowCount] = await readUint32(outputs.range, 2);
        result.rowRange = [firstRow, rowCount];
        result.drawFirstInstance = await readDrawWord(firstInstanceByteOffset);
      }
      if (options.validateOrder) {
        result.orderViolations = (await readUint32(outputs.violations, 1))[0];
      }
      return result;
    },
    destroy() {
      compiled.destroy();
      window.destroy();
      drawCommands.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Sorted float times with heavy duplication, including a run of equal values at the edges. */
function createSortedFloats(rows: number, seed: number): Float32Array {
  const random = createRandom(seed);
  const values = Float32Array.from({length: rows}, () => Math.floor(random() * 40) / 2);
  return values.sort();
}

const FLOAT_WINDOWS: Window[] = [
  {start: 3, end: 9},
  {start: 3.5, end: 3.5}, // duplicates at both edges
  {start: 0, end: 0}, // first value only
  {start: 19.5, end: 100}, // last value onward
  {start: -50, end: 100}, // all in
  {start: 100, end: 200}, // all out above
  {start: -200, end: -100}, // all out below
  {start: 9, end: 3}, // reversed
  {start: 4.25, end: 4.4} // between values: empty
];

function expectSameAcceptance(sorted: RunResult, mask: RunResult, capacity: number) {
  expect(sorted.count).toBe(mask.count);
  expect(sorted.requiredCount).toBe(mask.requiredCount);
  expect(sorted.overflow).toBe(mask.overflow);
  expect(sorted.ids).toEqual(mask.ids);
  expect(sorted.drawInstanceCount).toBe(mask.count);
  expect(sorted.rowRange![1]).toBe(mask.requiredCount);
  expect(sorted.count).toBe(Math.min(sorted.rowRange![1], capacity));
  if (sorted.rowRange![1] > 0) {
    expect(sorted.drawFirstInstance).toBe(sorted.rowRange![0]);
  }
}

it('GPUTimeWindowFilter sorted equals the mask path on sorted float data', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const high = createSortedFloats(500, 7);
  for (const capacity of [8, 64, 500]) {
    const maskPath = createFixture(device, {
      sorted: false,
      times: {high},
      capacity,
      withMask: true,
      withSourceIds: true
    });
    const rangePath = createFixture(device, {
      sorted: true,
      times: {high},
      capacity,
      withMask: true,
      withSourceIds: true,
      validateOrder: true
    });
    for (const window of FLOAT_WINDOWS) {
      const expected = await maskPath.run(window);
      const actual = await rangePath.run(window);
      expectSameAcceptance(actual, expected, capacity);
      expect(actual.mask, JSON.stringify(window)).toEqual(expected.mask);
      expect(actual.orderViolations).toBe(0);
    }
    maskPath.destroy();
    rangePath.destroy();
  }
});

it('GPUTimeWindowFilter sorted equals the mask path with double-single low parts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Times near 1e8 where one f32 ulp is 8: the low parts order rows that share a high part.
  const base = 1e8;
  const random = createRandom(11);
  const exact = Array.from({length: 300}, () => base + Math.floor(random() * 24) - 4).sort(
    (left, right) => left - right
  );
  const {high, low} = splitTimestamps(exact);
  const maskPath = createFixture(device, {sorted: false, times: {high, low}, capacity: 300});
  const rangePath = createFixture(device, {sorted: true, times: {high, low}, capacity: 300});
  for (const window of [
    {start: base, end: base + 3},
    {start: base - 4, end: base - 4},
    {start: base + 2.5, end: base + 19},
    {start: base - 1000, end: base + 1000},
    {start: base + 5, end: base + 4}
  ]) {
    expectSameAcceptance(await rangePath.run(window), await maskPath.run(window), 300);
  }
  maskPath.destroy();
  rangePath.destroy();
});

it('GPUTimeWindowFilter sorted equals the mask path on Int64 words with negative times', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(23);
  const values = Array.from({length: 400}, () => BigInt(Math.floor(random() * 60) - 30) * 1000n);
  // Straddle the low-word wrap and the sign change.
  values.push(4_294_967_295n, 4_294_967_296n, 4_294_967_296n, -4_294_967_297n, -(2n ** 40n));
  values.sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
  const words = BigInt64Array.from(values);
  const maskPath = createFixture(device, {sorted: false, times: {words}, capacity: 32});
  const rangePath = createFixture(device, {sorted: true, times: {words}, capacity: 32});
  for (const window of [
    {start: -5000, end: 5000},
    {start: 0, end: 0},
    {start: -4_294_967_297, end: 4_294_967_296},
    {start: -(2 ** 40), end: -(2 ** 40)},
    {start: -1e12, end: 1e12},
    {start: 2e12, end: 3e12},
    {start: 5000, end: -5000}
  ]) {
    expectSameAcceptance(await rangePath.run(window), await maskPath.run(window), 32);
  }
  maskPath.destroy();
  rangePath.destroy();
});

it('GPUTimeWindowFilter sorted handles empty and single-row inputs and counts order violations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const single = createFixture(device, {
    sorted: true,
    times: {high: Float32Array.of(5)},
    capacity: 4
  });
  expect((await single.run({start: 0, end: 10})).rowRange).toEqual([0, 1]);
  expect((await single.run({start: 6, end: 10})).rowRange![1]).toBe(0);
  single.destroy();

  // validateOrder counts adjacent descents on unsorted input: 3 -> 1 and 9 -> 2.
  const unsorted = createFixture(device, {
    sorted: true,
    times: {high: Float32Array.of(1, 3, 1, 9, 2, 10)},
    capacity: 4,
    validateOrder: true
  });
  expect((await unsorted.run({start: 0, end: 10})).orderViolations).toBe(2);
  unsorted.destroy();
});

it('GPUTimeWindowFilter sorted A/B: mask path versus range mode on 1e7 sorted rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 10_000_000;
  const high = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    high[row] = row / 16;
  }
  const capacity = 1024;
  const maskPath = createFixture(device, {sorted: false, times: {high}, capacity});
  const rangePath = createFixture(device, {sorted: true, times: {high}, capacity});
  const window = {start: 250_000, end: 250_500};
  const time = async (fixture: Fixture) => {
    await fixture.run(window); // warm up pipelines and bind groups
    const samples: number[] = [];
    for (let iteration = 0; iteration < 10; iteration++) {
      const begin = performance.now();
      await fixture.run({start: window.start + iteration, end: window.end + iteration});
      samples.push(performance.now() - begin);
    }
    return samples.sort((left, right) => left - right)[5];
  };
  const maskMilliseconds = await time(maskPath);
  const rangeMilliseconds = await time(rangePath);
  const expected = await maskPath.run(window);
  expectSameAcceptance(await rangePath.run(window), expected, capacity);
  // Timings include a small readback in both paths; the numbers are logged, not asserted.
  console.log(
    `GPUTimeWindowFilter A/B ${rows} rows: mask ${maskMilliseconds.toFixed(2)} ms, ` +
      `range ${rangeMilliseconds.toFixed(2)} ms`
  );
  maskPath.destroy();
  rangePath.destroy();
});
