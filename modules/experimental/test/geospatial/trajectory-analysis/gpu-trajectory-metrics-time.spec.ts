// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {splitTimestamps} from '../../../src/gpu-dataframe/time-window-filter/time-window-parameters';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter/time-words';
import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  type GPUTrajectoryStopParameters
} from '../../../src/geospatial/trajectory-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  computeTrajectoryOracle,
  createBigIntOracleTime,
  createDoubleOracleTime,
  type TrajectoryOracleResult,
  type TrajectoryOracleTime
} from './trajectory-metrics-oracle';

const SENTINEL = 0xffffffff;
const WRAP_BASE = 397n * 2n ** 32n; // a low-word wrap in the 2023 epoch-ms range
const EPOCH_BASE = 1_700_000_000_000n;

type TimeInput =
  | {kind: 'float32'; values: Float32Array}
  | {kind: 'double-single'; high: Float32Array; low: Float32Array}
  | {kind: 'words'; words: Uint32Array};

type Row = [x: number, y: number, time: bigint];

type TimeResult = {
  trackLengths: number[];
  trackDurations: number[];
  averageSpeeds: number[];
  maximumSpeeds: number[];
  trackStopCounts: number[];
  count: number;
  total: number;
  ids: number[];
  startRows: number[];
  endRows: number[];
  durations: number[];
  centroids: number[];
};

/** Packs `[x, y, time]` rows (BigInt times) into positions, offsets, and each time representation. */
function packRows(trackRows: Row[][]) {
  const positions: number[] = [];
  const times: bigint[] = [];
  const trackOffsets = [0];
  for (const rows of trackRows) {
    for (const [x, y, time] of rows) {
      positions.push(x, y);
      times.push(time);
    }
    trackOffsets.push(times.length);
  }
  const numbers = times.map(Number);
  const split = splitTimestamps(numbers);
  return {
    positions: Float32Array.from(positions),
    positionsTyped: {
      positions: Float32Array.from(positions),
      timestamps: new Float32Array(times.length),
      trackOffsets
    },
    trackOffsets,
    times,
    numbers,
    input: {
      words: {kind: 'words', words: getInt64TimeWords(BigInt64Array.from(times))} as TimeInput,
      doubleSingle: {kind: 'double-single', high: split.high, low: split.low} as TimeInput,
      float32: {kind: 'float32', values: Float32Array.from(numbers)} as TimeInput
    }
  };
}

/** Runs one all-outputs graph over the given time input and reads everything back. */
async function runMetrics(
  device: Device,
  packed: ReturnType<typeof packRows>,
  time: TimeInput,
  parameters: GPUTrajectoryStopParameters,
  capacity: number
): Promise<TimeResult> {
  const rowCount = packed.times.length;
  const trackCount = packed.trackOffsets.length - 1;
  const buffers: Buffer[] = [];
  const track = <B extends Buffer>(buffer: B): B => {
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = track(createOutputBuffer(device, length));
    buffer.write(new Uint32Array(Math.max(length, 1)).fill(0x7f7f7f7f));
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'trajectory-time-test'});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2'>(
    id: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, id, buffer, format, length);
  const positionsBuffer = track(createInputBuffer(device, packed.positions));
  const offsetsBuffer = track(createInputBuffer(device, Uint32Array.from(packed.trackOffsets)));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'trajectory-time-parameters',
    format: 'float32',
    length: GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
  });
  let timeProps: {
    timestamps: ReturnType<typeof view<'float32'>> | ReturnType<typeof view<'uint32x2'>>;
    timestampsLow?: ReturnType<typeof view<'float32'>>;
  };
  if (time.kind === 'words') {
    const buffer = track(createInputBuffer(device, time.words));
    timeProps = {timestamps: view('timestamps', buffer, 'uint32x2', rowCount)};
  } else if (time.kind === 'double-single') {
    const high = track(createInputBuffer(device, time.high));
    const low = track(createInputBuffer(device, time.low));
    timeProps = {
      timestamps: view('timestamps', high, 'float32', rowCount),
      timestampsLow: view('timestamps-low', low, 'float32', rowCount)
    };
  } else {
    const buffer = track(createInputBuffer(device, time.values));
    timeProps = {timestamps: view('timestamps', buffer, 'float32', rowCount)};
  }
  const out = {
    lengths: output(trackCount),
    durations: output(trackCount),
    averages: output(trackCount),
    maxima: output(trackCount),
    stopCounts: output(trackCount),
    stopIds: output(capacity),
    stopStarts: output(capacity),
    stopEnds: output(capacity),
    stopDurations: output(capacity),
    stopCentroids: output(2 * capacity),
    count: output(1),
    overflow: output(1),
    total: output(1)
  };
  graph.add(
    new GPUTrajectoryMetrics({
      positions: view('positions', positionsBuffer, 'float32x2', rowCount),
      ...timeProps,
      trackOffsets: view('offsets', offsetsBuffer, 'uint32', trackCount + 1),
      parameters: parameterBuffer.importToGraph(graph),
      trackLengths: view('lengths', out.lengths, 'float32', trackCount),
      trackDurations: view('durations', out.durations, 'float32', trackCount),
      averageSpeeds: view('averages', out.averages, 'float32', trackCount),
      maximumSpeeds: view('maxima', out.maxima, 'float32', trackCount),
      trackStopCounts: view('stop-counts', out.stopCounts, 'uint32', trackCount),
      stops: {
        output: {
          ids: view('stop-ids', out.stopIds, 'uint32', capacity),
          count: view('count', out.count, 'uint32', 1),
          overflow: view('overflow', out.overflow, 'uint32', 1),
          totalCount: view('total', out.total, 'uint32', 1)
        },
        startRows: view('stop-starts', out.stopStarts, 'uint32', capacity),
        endRows: view('stop-ends', out.stopEnds, 'uint32', capacity),
        durations: view('stop-durations', out.stopDurations, 'float32', capacity),
        centroids: view('stop-centroids', out.stopCentroids, 'float32x2', capacity)
      }
    })
  );
  const compiled = graph.compile();
  parameterBuffer.write(getGPUTrajectoryMetricsParameterValues(parameters));
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(out.count, 1);
  const [total] = await readUint32(out.total, 1);
  const result: TimeResult = {
    trackLengths: await readFloat32(out.lengths, trackCount),
    trackDurations: await readFloat32(out.durations, trackCount),
    averageSpeeds: await readFloat32(out.averages, trackCount),
    maximumSpeeds: await readFloat32(out.maxima, trackCount),
    trackStopCounts: await readUint32(out.stopCounts, trackCount),
    count,
    total,
    ids: await readUint32(out.stopIds, capacity),
    startRows: await readUint32(out.stopStarts, capacity),
    endRows: await readUint32(out.stopEnds, capacity),
    durations: await readFloat32(out.stopDurations, capacity),
    centroids: await readFloat32(out.stopCentroids, 2 * capacity)
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectCloseRelative(actual: number[], expected: number[], tolerance: number): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
      tolerance * Math.max(1, Math.abs(value))
    );
  }
}

function expectMatchesOracle(
  result: TimeResult,
  expected: TrajectoryOracleResult,
  capacity: number,
  durationTolerance: number = 0
): void {
  expectCloseRelative(result.trackLengths, expected.trackLengths, 1e-4);
  expectCloseRelative(result.maximumSpeeds, expected.maximumSpeeds, 1e-4);
  expectCloseRelative(result.averageSpeeds, expected.averageSpeeds, 1e-4 + durationTolerance);
  expectCloseRelative(result.trackDurations, expected.trackDurations, durationTolerance);
  expect(result.trackStopCounts).toEqual(expected.trackStopCounts);
  const count = Math.min(expected.stops.length, capacity);
  expect(result.total).toBe(expected.stops.length);
  expect(result.count).toBe(count);
  for (let index = 0; index < capacity; index++) {
    const stop = expected.stops[index];
    if (index >= count) {
      expect(result.ids[index]).toBe(SENTINEL);
      expect(result.durations[index]).toBe(0);
      continue;
    }
    expect(result.ids[index]).toBe(stop.track);
    expect(result.startRows[index]).toBe(stop.startRow);
    expect(result.endRows[index]).toBe(stop.endRow);
    expectCloseRelative([result.durations[index]], [stop.duration], durationTolerance);
    for (const axis of [0, 1]) {
      expect(
        Math.abs(result.centroids[2 * index + axis] - stop.centroid[axis])
      ).toBeLessThanOrEqual(0.02 + 1e-4 * Math.abs(stop.centroid[axis]));
    }
  }
}

/**
 * A run of `count` samples at `x`, 1 ms apart except where `gaps` gives the dwell step times.
 * Dwell steps are stationary; a following move is 10 units per ms.
 */
function dwell(x: number, startTime: bigint, stepTimes: number[]): Row[] {
  const rows: Row[] = [[x, 0, startTime]];
  let time = startTime;
  for (const step of stepTimes) {
    time += BigInt(step);
    rows.push([x, 0, time]);
  }
  return rows;
}

function move(fromRow: Row, deltaTime: number, deltaX: number): Row {
  return [fromRow[0] + deltaX, fromRow[1], fromRow[2] + BigInt(deltaTime)];
}

/**
 * Tracks with ms-scale steps. With `stopMinimumDuration` 5 the dwells last 4 ms (no stop), 5 ms
 * (exact boundary stop), 6 ms (stop) and the first and last dwell of the wrap track straddle the
 * low-word wrap.
 */
function buildTracks(): Row[][] {
  const tracks: Row[][] = [];
  for (const base of [EPOCH_BASE, WRAP_BASE - 3n, WRAP_BASE - 7n]) {
    const rows: Row[] = [];
    let cursor = base;
    for (const [dwellSteps, moveTime] of [
      [[1, 1, 1, 1], 2], // 4 ms: not a stop
      [[1, 2, 2], 1], // 5 ms: stop
      [[2, 2, 2], 3], // 6 ms: stop
      [[1, 1, 1, 1, 1], 1] // 5 ms: stop
    ] as [number[], number][]) {
      const x = rows.length > 0 ? rows[rows.length - 1][0] + 50 : 0;
      const block = dwell(x, cursor, dwellSteps);
      rows.push(...block);
      const moved = move(block[block.length - 1], moveTime, 50);
      cursor = moved[2];
    }
    rows.push([rows[rows.length - 1][0] + 50, 0, cursor]);
    tracks.push(rows);
  }
  return tracks;
}

const PARAMETERS: GPUTrajectoryStopParameters = {stopSpeedThreshold: 1, stopMinimumDuration: 5};

async function expectWordParity(
  device: Device,
  trackRows: Row[][],
  parameters: GPUTrajectoryStopParameters = PARAMETERS
) {
  const packed = packRows(trackRows);
  const oracleTime = createBigIntOracleTime(packed.times);
  const expected = computeTrajectoryOracle(
    packed.positionsTyped,
    parameters.stopSpeedThreshold,
    parameters.stopMinimumDuration,
    oracleTime
  );
  const capacity = expected.stops.length + 3;
  const result = await runMetrics(device, packed, packed.input.words, parameters, capacity);
  expectMatchesOracle(result, expected, capacity);
  return {packed, expected, result, capacity};
}

it('GPUTrajectoryMetrics Int64 words are exact for ms steps near 1.7e12 and across a low-word wrap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {packed, expected, result} = await expectWordParity(device, buildTracks());
  // The data has teeth: 3 tracks x 3 qualifying dwells, none of the 4 ms dwells.
  expect(expected.stops.length).toBe(9);
  expect(expected.stops.map(stop => stop.duration)).toEqual([5, 6, 5, 5, 6, 5, 5, 6, 5]);
  expect(result.durations.slice(0, 9)).toEqual(expected.stops.map(stop => stop.duration));
  expect(packed.times[packed.trackOffsets[1]] < WRAP_BASE).toBe(true);
  expect(packed.times[packed.trackOffsets[2] - 1] > WRAP_BASE).toBe(true);

  // Raising the minimum by 1 ms drops exactly the boundary dwells.
  await expectWordParity(device, buildTracks(), {stopSpeedThreshold: 1, stopMinimumDuration: 6});
  await expectWordParity(device, buildTracks(), {stopSpeedThreshold: 1, stopMinimumDuration: 4});
  device.destroy?.();
});

it('GPUTrajectoryMetrics f32 absolute timestamps get the same data wrong (control)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const packed = packRows(buildTracks());
  const expected = computeTrajectoryOracle(
    packed.positionsTyped,
    PARAMETERS.stopSpeedThreshold,
    PARAMETERS.stopMinimumDuration,
    createBigIntOracleTime(packed.times)
  );
  const capacity = 16;
  const wrong = await runMetrics(device, packed, packed.input.float32, PARAMETERS, capacity);
  // f32 spacing at 1.7e12 is 131072 ms: every ms-scale duration collapses to 0 or +-131072.
  expect(wrong.trackDurations).not.toEqual(expected.trackDurations);
  expect(wrong.total).not.toBe(expected.stops.length);
  const exact = await runMetrics(device, packed, packed.input.words, PARAMETERS, capacity);
  expect(exact.trackDurations).toEqual(expected.trackDurations);
  expect(exact.total).toBe(expected.stops.length);
  device.destroy?.();
});

it('GPUTrajectoryMetrics double-single timestamps match the double oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const packed = packRows(buildTracks());
  const oracleTime: TrajectoryOracleTime = createDoubleOracleTime(packed.numbers);
  for (const parameters of [
    PARAMETERS,
    {stopSpeedThreshold: 1, stopMinimumDuration: 6},
    {stopSpeedThreshold: 1, stopMinimumDuration: 4}
  ]) {
    const expected = computeTrajectoryOracle(
      packed.positionsTyped,
      parameters.stopSpeedThreshold,
      parameters.stopMinimumDuration,
      oracleTime
    );
    const capacity = expected.stops.length + 3;
    const result = await runMetrics(
      device,
      packed,
      packed.input.doubleSingle,
      parameters,
      capacity
    );
    expectMatchesOracle(result, expected, capacity);
  }
  device.destroy?.();
});

it('GPUTrajectoryMetrics handles pre-1970 and zero-crossing Int64 words', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const pre1970 = -1_000_000_000_000n;
  const trackRows = [
    // Negative epoch-ms with a stop that crosses a 2^32 boundary of the two's complement words.
    dwell(0, pre1970, [2, 2, 2]).concat([[50, 0, pre1970 + 7n]]),
    // Crosses zero and the low-word wrap of -2^32.
    dwell(0, -3n, [1, 2, 2]).concat([[50, 0, 4n]]),
    dwell(0, -(2n ** 32n) - 2n, [3, 3]).concat([[50, 0, -(2n ** 32n) + 5n]])
  ];
  const {expected} = await expectWordParity(device, trackRows);
  expect(expected.stops.map(stop => stop.duration)).toEqual([6, 5, 6]);
  device.destroy?.();
});

it('GPUTrajectoryMetrics Int64 words handle equal and backwards timestamps exactly', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const base = WRAP_BASE - 2n;
  const trackRows: Row[][] = [
    [
      [0, 0, base],
      [0, 0, base], // equal time, zero distance: slow, delta exactly 0
      [0, 0, base + 3n],
      [0, 0, base + 5n], // dwell rows 0..3, duration 5
      [100, 0, base + 5n], // equal time with movement: not slow (speed undefined), no speed spike
      [100, 0, base + 3n], // backwards, zero distance: not slow
      [100, 0, base + 4n],
      [100, 0, base + 9n]
    ],
    [
      [0, 0, EPOCH_BASE + 10n],
      [10, 0, EPOCH_BASE + 4n], // backwards in time
      [10, 0, EPOCH_BASE + 4n],
      [10, 0, EPOCH_BASE + 9n]
    ]
  ];
  const {expected, result, packed} = await expectWordParity(device, trackRows);
  expect(expected.stops.length).toBeGreaterThan(0);
  // Equal times give a duration of exactly 0 and never a speed spike.
  expect(result.maximumSpeeds.every(speed => Number.isFinite(speed))).toBe(true);
  expect(expected.trackDurations[1]).toBe(-1);
  expect(result.trackDurations[1]).toBe(-1);
  expect(packed.times.length).toBe(12);
  device.destroy?.();
});

it('GPUTrajectoryMetrics Int64 words keep multi-day durations within one f32 rounding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const day = 86_400_000n;
  const trackRows: Row[][] = [
    [
      [0, 0, EPOCH_BASE],
      [5, 0, EPOCH_BASE + 3n * day + 1n],
      [5, 0, EPOCH_BASE + 4n * day + 3n]
    ]
  ];
  const packed = packRows(trackRows);
  const expected = computeTrajectoryOracle(
    packed.positionsTyped,
    1,
    5,
    createBigIntOracleTime(packed.times)
  );
  const result = await runMetrics(device, packed, packed.input.words, PARAMETERS, 4);
  expectMatchesOracle(result, expected, 4, 2 ** -22);
  device.destroy?.();
});
