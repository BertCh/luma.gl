// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_TRAJECTORY_PLAYHEAD_STATUS as STATUS} from '../../../src/gpu-spatial-analysis/trajectory-interpolation';
import {
  createRandomTracks,
  EDGE_CASE_TRACKS,
  packTracks
} from './trajectory-interpolation-fixtures';
import {createPlayheadFixture, expectPlayheadParity} from './trajectory-interpolation-harness';
import {computePlayheadOracle} from './trajectory-interpolation-oracle';

/** A playhead base where the edge-case rows straddle a low-word wrap of the Int64 words. */
const WRAP_BASE = 397n * 2n ** 32n - 20n;
const EDGE_PLAYHEADS = [
  -5, 0, 0.5, 4.75, 5, 7.5, 10, 12, 15, 17.25, 20, 30, 40, 50, 100.5, 101, 105, 110, 200
];
const SAMPLE_TIMES = [0, 5, 10, 12, 15, 20, 40, 101, 110];

it('GPUTrajectoryPlayhead matches the oracle on edge cases without rebuilding per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tracks = packTracks(EDGE_CASE_TRACKS);
  const capacity = 3;
  const fixture = createPlayheadFixture(device, tracks, capacity);
  for (const maxGap of [0, 10]) {
    for (const playhead of EDGE_PLAYHEADS) {
      const expected = computePlayheadOracle(tracks, {kind: 'float32', value: playhead}, maxGap);
      const actual = await fixture.run(playhead, maxGap);
      expectPlayheadParity(actual, expected, capacity, `t=${playhead} gap=${maxGap}`);
      if (SAMPLE_TIMES.includes(playhead)) {
        // Exact sample times reproduce stored positions bit-exactly.
        expect(actual.positions, `exact t=${playhead}`).toEqual(expected.positions);
      }
    }
  }
  // Teeth: the sweep covers every status, and the gap rule flips track 4 at t=50.
  const gap = await fixture.run(50, 10);
  expect(gap.status).toEqual([
    STATUS.afterEnd,
    STATUS.afterEnd,
    STATUS.afterEnd,
    STATUS.empty,
    STATUS.gap,
    STATUS.beforeStart,
    STATUS.empty
  ]);
  expect(gap.positions.slice(8, 10)).toEqual([1, 0]);
  const open = await fixture.run(50, 0);
  expect(open.status[4]).toBe(STATUS.active);
  // Interpolated (not a stored sample): exact up to the GPU's f32 division.
  expect(open.positions[8]).toBeCloseTo(50, 4);
  expect(open.positions[9]).toBe(0);
  expect(open.activeTracks).toEqual([4]);
  const duplicate = await fixture.run(5);
  expect(duplicate.positions.slice(2, 4)).toEqual([9, 9]);
  // One compiled graph served 40 playheads and gap limits.
  expect(fixture.getBuildCount()).toBe(1);
  fixture.destroy();
  device.destroy?.();
});

it('GPUTrajectoryPlayhead Int64 words are exact across a low-word wrap with fractional playheads', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const floatTracks = packTracks(EDGE_CASE_TRACKS);
  const tracks = packTracks(EDGE_CASE_TRACKS, WRAP_BASE);
  expect(tracks.wordTimes[0] % 2n ** 32n > tracks.wordTimes[3] % 2n ** 32n).toBe(true);
  const capacity = 7;
  const fixture = createPlayheadFixture(device, tracks, capacity);
  for (const maxGap of [0, 10]) {
    for (const playhead of EDGE_PLAYHEADS) {
      const integer = Math.floor(playhead);
      const fraction = playhead - integer;
      const expected = computePlayheadOracle(
        tracks,
        {kind: 'words', integer: WRAP_BASE + BigInt(integer), fraction},
        maxGap
      );
      // The word path at ~7.3e12 reproduces the relative float32 path at ~0.
      const relative = computePlayheadOracle(
        floatTracks,
        {kind: 'float32', value: playhead},
        maxGap
      );
      expect(expected.status).toEqual(relative.status);
      expect(expected.positions).toEqual(relative.positions);
      const actual = await fixture.run(
        fraction === 0 ? WRAP_BASE + BigInt(integer) : Number(WRAP_BASE) + playhead,
        maxGap
      );
      expectPlayheadParity(actual, expected, capacity, `words t=${playhead} gap=${maxGap}`);
    }
  }
  // A fraction just above a sample time leaves the exact-fix branch.
  const atFix = await fixture.run(WRAP_BASE + 101n, 10);
  const pastFix = await fixture.run(Number(WRAP_BASE) + 1.5, 10);
  expect(atFix.status[4]).toBe(STATUS.active);
  expect(pastFix.status[4]).toBe(STATUS.gap);
  expect(fixture.getBuildCount()).toBe(1);
  fixture.destroy();
  device.destroy?.();
});

it('GPUTrajectoryPlayhead matches the oracle on random tracks with duplicates and overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [seed, wordBase] of [
    [1, undefined],
    [2, undefined],
    [3, 1_700_000_000_000n]
  ] as const) {
    const rows = createRandomTracks(seed, 600);
    const tracks = packTracks(rows, wordBase);
    const capacity = 150;
    const fixture = createPlayheadFixture(device, tracks, capacity);
    let activeSeen = 0;
    for (const [playhead, maxGap] of [
      [50, 0],
      [137, 25],
      [400, 30],
      [811, 0],
      [-1, 0]
    ]) {
      const expected = computePlayheadOracle(
        tracks,
        wordBase === undefined
          ? {kind: 'float32', value: playhead}
          : {
              kind: 'words',
              integer: wordBase + BigInt(playhead),
              fraction: 0
            },
        maxGap
      );
      const actual = await fixture.run(
        wordBase === undefined ? playhead : wordBase + BigInt(playhead),
        maxGap
      );
      expectPlayheadParity(actual, expected, capacity, `seed ${seed} t=${playhead}`);
      activeSeen = Math.max(activeSeen, expected.activeTracks.length);
    }
    // At least one frame overflows the 150-track capacity.
    expect(activeSeen).toBeGreaterThan(capacity);
    expect(fixture.getBuildCount()).toBe(1);
    fixture.destroy();
  }
  device.destroy?.();
});
