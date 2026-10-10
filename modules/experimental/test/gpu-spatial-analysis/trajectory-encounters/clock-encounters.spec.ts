// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter/time-words';
import {addClockEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {
  getGPUTrajectoryClockParameterValues,
  getGPUTrajectoryClockWordParameterValues,
  GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createRandom,
  packTracks,
  type FixtureRow
} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import type {OracleTracks} from '../trajectory-interpolation/trajectory-interpolation-oracle';
import {computeEncountersOracle} from './encounters-oracle';

const BOUNDS = [0, 0, 1000, 1000] as const;
const BUCKET_COUNT = 24;
const PAIR_CAPACITY = 1024;

/** Random walks with integer times and different, overlapping windows (some empty or single). */
function createWindowedTracks(seed: number, trackCount: number): FixtureRow[][] {
  const random = createRandom(seed);
  const tracks: FixtureRow[][] = [];
  for (let track = 0; track < trackCount; track++) {
    const rowCount = track % 19 === 0 ? 0 : track % 11 === 0 ? 1 : 3 + Math.floor(random() * 14);
    let [x, y, time] = [random() * 1000, random() * 1000, Math.floor(random() * 60)];
    const rows: FixtureRow[] = [];
    for (let row = 0; row < rowCount; row++) {
      rows.push([x, y, time]);
      time += random() < 0.1 ? 0 : 1 + Math.floor(random() * 12);
      x = Math.min(1000, Math.max(0, x + (random() - 0.5) * 90));
      y = Math.min(1000, Math.max(0, y + (random() - 0.5) * 90));
    }
    tracks.push(rows);
  }
  return tracks;
}

/** CPU clock resample: linear interpolation at `start + k * step`, NaN outside the track window. */
function computeClockSamples(
  tracks: OracleTracks,
  start: number,
  step: number,
  bucketCount: number
): Float32Array {
  const trackCount = tracks.trackOffsets.length - 1;
  const samples = new Float32Array(trackCount * bucketCount * 2).fill(Number.NaN);
  const time = (row: number) => Number(tracks.times.values[row]);
  for (let track = 0; track < trackCount; track++) {
    const [first, end] = [tracks.trackOffsets[track], tracks.trackOffsets[track + 1]];
    for (let bucket = 0; bucket < bucketCount && end > first; bucket++) {
      const target = start + bucket * step;
      if (target < time(first) || target > time(end - 1)) {
        continue;
      }
      let upper = first + 1;
      while (upper < end && time(upper) <= target) {
        upper++;
      }
      const index = 2 * (track * bucketCount + bucket);
      if (upper >= end) {
        samples.set(tracks.positions.slice(2 * (end - 1), 2 * end), index);
        continue;
      }
      const fraction = (target - time(upper - 1)) / (time(upper) - time(upper - 1));
      for (let axis = 0; axis < 2; axis++) {
        const a = tracks.positions[2 * (upper - 1) + axis];
        const b = tracks.positions[2 * upper + axis];
        samples[index + axis] = a + (b - a) * fraction;
      }
    }
  }
  return samples;
}

type ClockRun = {
  samples: number[];
  count: number;
  overflow: number;
  ids: number[];
  partners: number[];
  firstBuckets: number[];
  bucketCounts: number[];
  firstTimes: number[];
  bucketTimes: number[];
};

/** Compiles one clock-encounter graph and runs it once per `(start, step)` without recompiling. */
async function runClockEncounters(
  device: Device,
  tracks: OracleTracks,
  clocks: readonly {start: number; step: number}[],
  distance: number,
  wordBase = 0n
): Promise<ClockRun[]> {
  const isWordMode = tracks.times.kind === 'words';
  const trackCount = tracks.trackOffsets.length - 1;
  const rowCount = tracks.positions.length / 2;
  const graph = new GPUCommandGraph(device, {id: 'clock-encounters-test'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffer.write(new Uint32Array(Math.max(length, 1)).fill(0x7f7f7f7f));
    buffers.push(buffer);
    return buffer;
  };
  const clock = new GPUParameterBuffer(device, {
    id: 'clock',
    format: isWordMode ? 'uint32' : 'float32',
    length: GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH
  });
  const sampleCount = trackCount * BUCKET_COUNT;
  const out = {
    samples: output(2 * sampleCount),
    bucketTimes: output(BUCKET_COUNT),
    ids: output(PAIR_CAPACITY),
    partners: output(PAIR_CAPACITY),
    first: output(PAIR_CAPACITY),
    counts: output(PAIR_CAPACITY),
    times: output(PAIR_CAPACITY),
    count: output(1),
    overflow: output(1),
    candidateOverflow: output(1)
  };
  const timestampData = isWordMode
    ? getInt64TimeWords(BigInt64Array.from(tracks.times.values as bigint[]))
    : (tracks.times.values as Float32Array);
  const timestamps = isWordMode
    ? importGraphBuffer(graph, 'timestamps', input(timestampData), 'uint32x2', rowCount)
    : importGraphBuffer(graph, 'timestamps', input(timestampData), 'float32', rowCount);
  addClockEncounters(graph, {
    id: 'clock-encounters',
    positions: importGraphBuffer(
      graph,
      'positions',
      input(tracks.positions),
      'float32x2',
      rowCount
    ),
    timestamps,
    trackOffsets: importGraphBuffer(
      graph,
      'offsets',
      input(Uint32Array.from(tracks.trackOffsets)),
      'uint32',
      trackCount + 1
    ),
    clock: clock.importToGraph(graph),
    bucketCount: BUCKET_COUNT,
    samples: importGraphBuffer(graph, 'o-samples', out.samples, 'float32x2', sampleCount),
    bucketTimes: importGraphBuffer(
      graph,
      'o-bucket-times',
      out.bucketTimes,
      'float32',
      BUCKET_COUNT
    ),
    distance: importGraphBuffer(graph, 'distance', input(Float32Array.of(distance)), 'float32', 1),
    cellSize: Math.max(distance, 40),
    bounds: BOUNDS,
    hitCapacity: 16384,
    pairs: {
      candidateOverflow: importGraphBuffer(
        graph,
        'o-candidate-overflow',
        out.candidateOverflow,
        'uint32',
        1
      ),
      output: {
        ids: importGraphBuffer(graph, 'o-ids', out.ids, 'uint32', PAIR_CAPACITY),
        count: importGraphBuffer(graph, 'o-count', out.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'o-overflow', out.overflow, 'uint32', 1)
      },
      partners: importGraphBuffer(graph, 'o-partners', out.partners, 'uint32', PAIR_CAPACITY),
      firstBuckets: importGraphBuffer(graph, 'o-first', out.first, 'uint32', PAIR_CAPACITY),
      bucketCounts: importGraphBuffer(graph, 'o-counts', out.counts, 'uint32', PAIR_CAPACITY),
      firstTimes: importGraphBuffer(graph, 'o-times', out.times, 'float32', PAIR_CAPACITY)
    }
  });
  const compiled = graph.compile();
  const runs: ClockRun[] = [];
  for (const {start, step} of clocks) {
    if (isWordMode) {
      clock.write(
        getGPUTrajectoryClockWordParameterValues({start: wordBase + BigInt(start), step})
      );
    } else {
      clock.write(getGPUTrajectoryClockParameterValues({start, step}));
    }
    submitGraph(device, compiled, undefined);
    const [count] = await readUint32(out.count, 1);
    runs.push({
      samples: await readFloat32(out.samples, 2 * sampleCount),
      count,
      overflow: (await readUint32(out.overflow, 1))[0],
      ids: await readUint32(out.ids, count),
      partners: await readUint32(out.partners, count),
      firstBuckets: await readUint32(out.first, count),
      bucketCounts: await readUint32(out.counts, count),
      firstTimes: await readFloat32(out.times, count),
      bucketTimes: await readFloat32(out.bucketTimes, BUCKET_COUNT)
    });
  }
  compiled.destroy();
  clock.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return runs;
}

function expectSamplesMatch(actual: number[], expected: Float32Array, label: string): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `${label} absent ${index}`).toBe(true);
    } else {
      expect(Math.abs(actual[index] - value), `${label} sample ${index}`).toBeLessThanOrEqual(
        1e-2 + 1e-5 * Math.abs(value)
      );
    }
  }
}

it('addClockEncounters resamples to a common clock and matches the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const trackCount = 160;
  const tracks = packTracks(createWindowedTracks(21, trackCount));
  const clocks = [
    {start: 0, step: 4},
    {start: 30, step: 2}
  ];
  const distance = 45;
  const runs = await runClockEncounters(device, tracks, clocks, distance);
  for (const [index, clock] of clocks.entries()) {
    const label = `clock ${index}`;
    const run = runs[index];
    const expectedSamples = computeClockSamples(tracks, clock.start, clock.step, BUCKET_COUNT);
    expectSamplesMatch(run.samples, expectedSamples, label);
    // Teeth: absent samples (windows differ) and present ones both occur.
    const absent = expectedSamples.filter(Number.isNaN).length / 2;
    expect(absent).toBeGreaterThan(100);
    expect(absent).toBeLessThan(trackCount * BUCKET_COUNT - 100);
    // The encounters read the GPU samples, so compare against the oracle over those.
    const expected = computeEncountersOracle(
      Float32Array.from(run.samples),
      trackCount,
      BUCKET_COUNT,
      distance,
      undefined,
      BOUNDS
    );
    expect(expected.length).toBeGreaterThan(5);
    expect(run.overflow).toBe(0);
    expect(run.count).toBe(expected.length);
    for (const [pairIndex, pair] of expected.entries()) {
      expect([run.ids[pairIndex], run.partners[pairIndex]], `${label} pair ${pairIndex}`).toEqual([
        pair.track,
        pair.partner
      ]);
      expect(run.firstBuckets[pairIndex]).toBe(pair.firstBucket);
      expect(run.bucketCounts[pairIndex]).toBe(pair.bucketCount);
      expect(run.firstTimes[pairIndex]).toBe(pair.firstBucket * clock.step);
    }
    expect(run.bucketTimes).toEqual(Array.from({length: BUCKET_COUNT}, (_, k) => k * clock.step));
  }
  // Moving the clock changed the result.
  expect(runs[0].samples).not.toEqual(runs[1].samples);
  device.destroy?.();
});

it('addClockEncounters with Int64 word timestamps matches float32 timestamps', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = createWindowedTracks(22, 80);
  const relative = packTracks(rows);
  const wordBase = 397n * 2n ** 32n - 20n;
  const words = packTracks(rows, wordBase);
  const clocks = [{start: 10, step: 3}];
  const [floatRun] = await runClockEncounters(device, relative, clocks, 50);
  const [wordRun] = await runClockEncounters(device, words, clocks, 50, wordBase);
  expect(floatRun.count).toBeGreaterThan(0);
  expect(wordRun.samples.length).toBe(floatRun.samples.length);
  expect(wordRun.count).toBe(floatRun.count);
  expect(wordRun.ids).toEqual(floatRun.ids);
  expect(wordRun.partners).toEqual(floatRun.partners);
  device.destroy?.();
});
