// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A/B (algorithms roadmap P3 #4): exact closest-approach encounters against the clock preview on
// the same scene, 2000 random tracks of 50 samples. Prints the median wall time of 5 encodings
// (one-word readback) of the exact contributor and the pair counts of both modes. Time the clock
// mode with `gpu-trajectory-encounters.spec.ts` on the resampled table for the comparison.
//   npx vitest run --project headless modules/experimental/test/gpu-spatial-analysis/trajectory-encounters/closest-approach-ab.spec.ts

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {runClosestApproach} from './closest-approach-harness';
import type {ClosestApproachTrack} from './closest-approach-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPUTrajectoryClosestApproach A/B: 2000 tracks x 50 samples', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(99);
  const tracks: ClosestApproachTrack[] = Array.from({length: 2000}, () => {
    let [x, y, time] = [random() * 5000, random() * 5000, random() * 600];
    const positions: [number, number][] = [];
    const times: number[] = [];
    for (let sample = 0; sample < 50; sample++) {
      positions.push([Math.fround(x), Math.fround(y)]);
      times.push(Math.fround(time));
      x += (random() - 0.5) * 60;
      y += (random() - 0.5) * 60;
      time += 5 + random() * 20;
    }
    return {positions, times};
  });
  const result = await runClosestApproach(device, tracks, {
    distance: 25,
    cellSize: 100,
    timeCellSize: 60,
    entryCapacity: 1 << 21,
    hitCapacity: 1 << 20,
    pairCapacity: 1 << 16,
    timedEncodings: 5
  });
  expect(result.entryOverflow).toBe(0);
  // eslint-disable-next-line no-console
  console.log(
    `closest-approach-ab tracks=2000 pairs=${result.requiredCount} median=${result.medianEncodingMilliseconds?.toFixed(2)} ms`
  );
}, 600000);
