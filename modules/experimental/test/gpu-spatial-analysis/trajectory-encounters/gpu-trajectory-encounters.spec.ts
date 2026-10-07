// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {computeEncountersOracle} from './encounters-oracle';

const BOUNDS = [0, 0, 1000, 1000] as const;

/** Random walks on a common clock with some absent samples. */
function createSamples(seed: number, trackCount: number, bucketCount: number): Float32Array {
  const random = createRandom(seed);
  const samples = new Float32Array(trackCount * bucketCount * 2);
  for (let track = 0; track < trackCount; track++) {
    let x = random() * 1000;
    let y = random() * 1000;
    for (let bucket = 0; bucket < bucketCount; bucket++) {
      const row = track * bucketCount + bucket;
      samples[2 * row] = x;
      samples[2 * row + 1] = y;
      x = Math.min(1000, Math.max(0, x + (random() - 0.5) * 60));
      y = Math.min(1000, Math.max(0, y + (random() - 0.5) * 60));
      if (random() < 0.03) {
        samples[2 * row] = Number.NaN;
        samples[2 * row + 1] = Number.NaN;
      }
    }
  }
  return samples;
}

type RunOptions = {
  distance: number;
  cellSize: number;
  hitCapacity: number;
  pairCapacity: number;
  trackValid?: Uint32Array;
};

async function runEncounters(
  device: Device,
  samples: Float32Array,
  trackCount: number,
  bucketCount: number,
  options: RunOptions
) {
  const {pairCapacity} = options;
  const graph = new GPUCommandGraph(device, {id: 'encounters-test'});
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
  const bucketTimes = Float32Array.from({length: bucketCount}, (_, bucket) => 100 + bucket * 10);
  const out = {
    ids: output(pairCapacity),
    partners: output(pairCapacity),
    first: output(pairCapacity),
    distances: output(pairCapacity),
    counts: output(pairCapacity),
    times: output(pairCapacity),
    count: output(1),
    overflow: output(1),
    total: output(1)
  };
  graph.add(
    new GPUTrajectoryEncounters({
      id: 'encounters',
      samples: importGraphBuffer(graph, 'samples', input(samples), 'float32x2', samples.length / 2),
      trackCount,
      bucketCount,
      trackValid: options.trackValid
        ? importGraphBuffer(graph, 'valid', input(options.trackValid), 'uint32', trackCount)
        : undefined,
      distance: importGraphBuffer(
        graph,
        'distance',
        input(Float32Array.of(options.distance)),
        'float32',
        1
      ),
      cellSize: options.cellSize,
      bounds: BOUNDS,
      hitCapacity: options.hitCapacity,
      bucketTimes: importGraphBuffer(
        graph,
        'bucket-times',
        input(bucketTimes),
        'float32',
        bucketCount
      ),
      pairs: {
        output: {
          ids: importGraphBuffer(graph, 'o-ids', out.ids, 'uint32', pairCapacity),
          count: importGraphBuffer(graph, 'o-count', out.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'o-overflow', out.overflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'o-total', out.total, 'uint32', 1)
        },
        partners: importGraphBuffer(graph, 'o-partners', out.partners, 'uint32', pairCapacity),
        firstBuckets: importGraphBuffer(graph, 'o-first', out.first, 'uint32', pairCapacity),
        minimumDistances: importGraphBuffer(
          graph,
          'o-distances',
          out.distances,
          'float32',
          pairCapacity
        ),
        bucketCounts: importGraphBuffer(graph, 'o-counts', out.counts, 'uint32', pairCapacity),
        firstTimes: importGraphBuffer(graph, 'o-times', out.times, 'float32', pairCapacity)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(out.count, 1);
  const [overflow] = await readUint32(out.overflow, 1);
  const [totalCount] = await readUint32(out.total, 1);
  const result = {
    count,
    overflow,
    totalCount,
    ids: await readUint32(out.ids, pairCapacity),
    partners: await readUint32(out.partners, pairCapacity),
    first: await readUint32(out.first, pairCapacity),
    distances: await readFloat32(out.distances, pairCapacity),
    counts: await readUint32(out.counts, pairCapacity),
    times: await readFloat32(out.times, pairCapacity),
    bucketTimes
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUTrajectoryEncounters matches the all-pairs oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const trackCount = 220;
  const bucketCount = 12;
  const samples = createSamples(5, trackCount, bucketCount);
  const trackValid = new Uint32Array(trackCount).fill(1);
  trackValid[3] = 0;
  trackValid[100] = 0;
  for (const [distance, cellSize] of [
    [25, 25],
    [18, 40]
  ]) {
    const expected = computeEncountersOracle(
      samples,
      trackCount,
      bucketCount,
      distance,
      trackValid,
      BOUNDS
    );
    // Teeth: many pairs, some repeated across buckets, never an ignored track.
    expect(expected.length).toBeGreaterThan(40);
    expect(expected.some(pair => pair.bucketCount > 1)).toBe(true);
    expect(expected.some(pair => pair.track === 3 || pair.partner === 3)).toBe(false);
    const actual = await runEncounters(device, samples, trackCount, bucketCount, {
      distance,
      cellSize,
      hitCapacity: 8192,
      pairCapacity: 1024,
      trackValid
    });
    expect(actual.overflow).toBe(0);
    expect(actual.count).toBe(expected.length);
    expect(actual.totalCount).toBe(expected.length);
    for (const [index, pair] of expected.entries()) {
      const label = `d=${distance} pair ${index} (${pair.track}, ${pair.partner})`;
      expect([actual.ids[index], actual.partners[index]], label).toEqual([
        pair.track,
        pair.partner
      ]);
      expect(actual.first[index], label).toBe(pair.firstBucket);
      expect(actual.counts[index], label).toBe(pair.bucketCount);
      expect(actual.times[index], label).toBe(actual.bucketTimes[pair.firstBucket]);
      expect(Math.abs(actual.distances[index] - pair.minimumDistance), label).toBeLessThan(1e-3);
    }
    // Sentinels after the last pair.
    expect(actual.ids[expected.length]).toBe(0xffffffff);
    expect(actual.counts[expected.length]).toBe(0);
  }
  device.destroy?.();
});

it('GPUTrajectoryEncounters clamps the distance to the cell size and flags overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const trackCount = 220;
  const bucketCount = 12;
  const samples = createSamples(5, trackCount, bucketCount);
  const clamped = await runEncounters(device, samples, trackCount, bucketCount, {
    distance: 100,
    cellSize: 20,
    hitCapacity: 8192,
    pairCapacity: 1024
  });
  const expected = computeEncountersOracle(samples, trackCount, bucketCount, 20, undefined, BOUNDS);
  expect(clamped.count).toBe(expected.length);
  expect(expected.length).toBeGreaterThan(20);
  // Too-small pair capacity: count is clamped, total is not.
  const small = await runEncounters(device, samples, trackCount, bucketCount, {
    distance: 20,
    cellSize: 20,
    hitCapacity: 8192,
    pairCapacity: 8
  });
  expect(small.count).toBe(8);
  expect(small.totalCount).toBe(expected.length);
  expect(small.overflow).toBe(1);
  // Too-small hit scratch.
  const starved = await runEncounters(device, samples, trackCount, bucketCount, {
    distance: 20,
    cellSize: 20,
    hitCapacity: 8,
    pairCapacity: 1024
  });
  expect(starved.overflow).toBe(1);
  device.destroy?.();
});

it('GPUTrajectoryEncounters falls back to the three-pass sort when the packed key exceeds 32 bits', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 12 track bits + 12 partner bits + 9 bucket bits = 33 > 32, so the stable LSD chain runs.
  const trackCount = 2100;
  const bucketCount = 260;
  const samples = createSamples(11, trackCount, bucketCount);
  const expected = computeEncountersOracle(samples, trackCount, bucketCount, 3, undefined, BOUNDS);
  expect(expected.length).toBeGreaterThan(100);
  const actual = await runEncounters(device, samples, trackCount, bucketCount, {
    distance: 3,
    cellSize: 10,
    hitCapacity: 1 << 17,
    pairCapacity: 1 << 14
  });
  expect(actual.overflow).toBe(0);
  expect(actual.count).toBe(expected.length);
  for (const [index, pair] of expected.entries()) {
    const label = `pair ${index} (${pair.track}, ${pair.partner})`;
    expect([actual.ids[index], actual.partners[index]], label).toEqual([pair.track, pair.partner]);
    expect(actual.first[index], label).toBe(pair.firstBucket);
    expect(actual.counts[index], label).toBe(pair.bucketCount);
  }
  device.destroy?.();
}, 120_000);
