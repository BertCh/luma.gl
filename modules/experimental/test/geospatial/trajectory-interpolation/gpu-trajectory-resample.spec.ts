// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  createRandomTracks,
  EDGE_CASE_TRACKS,
  packTracks
} from './trajectory-interpolation-fixtures';
import {expectClose, runResample} from './trajectory-interpolation-harness';
import {computeResampleOracle, type ResampleOracleResult} from './trajectory-interpolation-oracle';

function expectResampleParity(
  actual: ResampleOracleResult,
  expected: ResampleOracleResult,
  label: string
): void {
  expectClose(actual.samples, expected.samples, 1e-3, 1e-5, `${label} samples`);
  expectClose(actual.sampleElevations, expected.sampleElevations, 1e-4, 1e-5, `${label} z`);
  expectClose(actual.sampleTimes, expected.sampleTimes, 1e-3, 1e-5, `${label} times`);
}

it('GPUTrajectoryResample matches the oracle on edge cases in time and arc-length spacing', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tracks = packTracks(EDGE_CASE_TRACKS);
  for (const spacing of ['time', 'arc-length'] as const) {
    for (const sampleCount of [1, 2, 5, 17]) {
      const expected = computeResampleOracle(tracks, sampleCount, spacing);
      const actual = await runResample(device, tracks, sampleCount, spacing);
      expectResampleParity(actual, expected, `${spacing} n=${sampleCount}`);
    }
  }
  // Teeth: track 0 at 5 samples hits every stored vertex in time spacing, and the arc-length
  // spacing moves the middle sample to the midpoint of the long edge.
  const time = await runResample(device, tracks, 5, 'time');
  expect(time.samples.slice(0, 10)).toEqual([0, 0, 10, 0, 10, 20, 5, 20, 0, 20]);
  const arc = await runResample(device, tracks, 5, 'arc-length');
  expect(arc.samples.slice(0, 10)).toEqual([0, 0, 10, 0, 10, 10, 10, 20, 0, 20]);
  // Single point repeats, empty tracks are zero, duplicates end at the last duplicate.
  expect(time.samples.slice(20, 30)).toEqual([3, 4, 3, 4, 3, 4, 3, 4, 3, 4]);
  expect(time.samples.slice(30, 40)).toEqual(new Array(10).fill(0));
  expect(time.samples.slice(18, 20)).toEqual([22, 9]);
  device.destroy?.();
});

it('GPUTrajectoryResample Int64 words match the relative float32 result', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const relative = packTracks(EDGE_CASE_TRACKS);
  const words = packTracks(EDGE_CASE_TRACKS, 397n * 2n ** 32n - 20n);
  for (const spacing of ['time', 'arc-length'] as const) {
    const expected = computeResampleOracle(relative, 9, spacing);
    expect(computeResampleOracle(words, 9, spacing)).toEqual(expected);
    expectResampleParity(
      await runResample(device, words, 9, spacing),
      expected,
      `words ${spacing}`
    );
  }
  device.destroy?.();
});

/**
 * Element-wise parity that also accepts the oracle evaluated with interior targets shifted by up
 * to 4 f32 ulps, the GPU division error that matters at discontinuities of the resampled path.
 */
function expectResampleParityWithinUlps(
  actual: ResampleOracleResult,
  tracks: ReturnType<typeof packTracks>,
  sampleCount: number,
  spacing: 'time' | 'arc-length',
  label: string
): number {
  const candidates = [0, -1, 1, -2, 2, -3, 3, -4, 4].map(offset =>
    computeResampleOracle(tracks, sampleCount, spacing, offset)
  );
  const isClose = (value: number, expected: number, absolute: number) =>
    Math.abs(value - expected) <= absolute + 1e-5 * Math.abs(expected);
  let shiftedCount = 0;
  for (let index = 0; index < actual.sampleTimes.length; index++) {
    const matches = candidates.findIndex(
      candidate =>
        isClose(actual.samples[2 * index], candidate.samples[2 * index], 1e-3) &&
        isClose(actual.samples[2 * index + 1], candidate.samples[2 * index + 1], 1e-3) &&
        isClose(actual.sampleElevations[index], candidate.sampleElevations[index], 1e-4) &&
        isClose(actual.sampleTimes[index], candidate.sampleTimes[index], 1e-3)
    );
    if (matches < 0) {
      expect.fail(
        `${label} sample ${index}: (${actual.samples[2 * index]}, ${actual.samples[2 * index + 1]}, t=${actual.sampleTimes[index]}) matches no oracle target within 4 ulps (expected (${candidates[0].samples[2 * index]}, ${candidates[0].samples[2 * index + 1]}, t=${candidates[0].sampleTimes[index]}))`
      );
    }
    shiftedCount += matches > 0 ? 1 : 0;
  }
  return shiftedCount;
}

it('GPUTrajectoryResample matches the oracle on random tracks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const seed of [4, 5]) {
    const tracks = packTracks(createRandomTracks(seed, 400));
    for (const spacing of ['time', 'arc-length'] as const) {
      const actual = await runResample(device, tracks, 32, spacing);
      const shiftedCount = expectResampleParityWithinUlps(
        actual,
        tracks,
        32,
        spacing,
        `seed ${seed} ${spacing}`
      );
      // Discontinuity flips are rare: well under 1% of the 12,800 samples.
      expect(shiftedCount).toBeLessThan(128);
    }
  }
  device.destroy?.();
});
