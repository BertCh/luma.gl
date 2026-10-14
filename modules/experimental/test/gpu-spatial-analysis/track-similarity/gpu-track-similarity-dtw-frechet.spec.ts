// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_TRACK_SIMILARITY_STATUS as STATUS} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {
  createDynamicProgramHarness,
  DYNAMIC_PROGRAM_FIXTURE,
  type DynamicProgramCase
} from './track-similarity-dtw-frechet-harness';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: test/oracles/track-similarity-dtw-frechet.py (similaritymeasures, shapely, tslearn and
// dtaidistance cross-checks; versions and tolerances in the fixture).
const {tolerances} = DYNAMIC_PROGRAM_FIXTURE;
const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

function getRadii(cases: readonly DynamicProgramCase[]): number[] {
  return [...new Set(cases.flatMap(entry => entry.banded.map(band => band.radius)))];
}

async function expectCases(cases: readonly DynamicProgramCase[], isGreatCircle: boolean) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const frechetAbsolute = isGreatCircle
    ? tolerances.frechetAbsoluteMeters
    : tolerances.frechetAbsolutePlanar;
  const stepAbsolute = isGreatCircle
    ? tolerances.dynamicTimeWarpingAbsoluteMetersPerStep
    : tolerances.dynamicTimeWarpingAbsolutePlanar;
  const setA = cases.map(entry => entry.a);
  const setB = cases.map(entry => entry.b);
  // Unbanded run: Frechet distances, every decision epsilon, unbanded dynamic time warping.
  const unbanded = createDynamicProgramHarness(device, setA, setB, {
    spatialContext: isGreatCircle ? GREAT_CIRCLE : undefined
  });
  try {
    const first = await unbanded.run(0);
    cases.forEach((entry, index) => {
      expect(first.status[index], entry.name).toBe(0);
      expect(
        Math.abs(first.frechet[index] - entry.frechet),
        `${entry.name} frechet`
      ).toBeLessThanOrEqual(frechetAbsolute + tolerances.frechetRelative * entry.frechet);
      const steps = entry.a.length + entry.b.length - 1;
      expect(
        Math.abs(first.dynamicTimeWarping[index] - entry.dynamicTimeWarping),
        `${entry.name} dynamic time warping`
      ).toBeLessThanOrEqual(
        steps * stepAbsolute +
          steps * tolerances.dynamicTimeWarpingRelativePerStep * entry.dynamicTimeWarping
      );
    });
    for (const [index, entry] of cases.entries()) {
      for (const decision of entry.decisions) {
        const result = await unbanded.run(decision.epsilon);
        expect(result.within[index], `${entry.name} epsilon ${decision.epsilon}`).toBe(
          decision.within
        );
      }
    }
    // Unusable epsilons decide 0.
    for (const epsilon of [NaN, -1, Infinity]) {
      expect(
        (await unbanded.run(epsilon)).within.every(value => value === 0),
        `${epsilon}`
      ).toBe(true);
    }
  } finally {
    unbanded.destroy();
  }
  for (const radius of getRadii(cases)) {
    const banded = createDynamicProgramHarness(device, setA, setB, {
      spatialContext: isGreatCircle ? GREAT_CIRCLE : undefined,
      bandRadius: radius
    });
    try {
      const result = await banded.run(0);
      cases.forEach((entry, index) => {
        const expected = entry.banded.find(band => band.radius === radius);
        if (!expected) {
          return;
        }
        const steps = entry.a.length + entry.b.length - 1;
        expect(
          Math.abs(result.dynamicTimeWarping[index] - expected.dynamicTimeWarping),
          `${entry.name} radius ${radius}`
        ).toBeLessThanOrEqual(
          steps * stepAbsolute +
            steps * tolerances.dynamicTimeWarpingRelativePerStep * expected.dynamicTimeWarping
        );
      });
    } finally {
      banded.destroy();
    }
  }
  device.destroy?.();
}

it('GPUTrackSimilarity Frechet decision and banded DTW match the planar oracle', async () => {
  // Teeth: the band changes the cost of at least one case, and decisions go both ways.
  const bandCase = DYNAMIC_PROGRAM_FIXTURE.planar.find(entry => entry.name === 'band-forces-path');
  expect(bandCase?.banded[0].dynamicTimeWarping).toBeGreaterThan(
    (bandCase?.dynamicTimeWarping ?? 0) * 2
  );
  await expectCases(DYNAMIC_PROGRAM_FIXTURE.planar, false);
}, 120000);

it('GPUTrackSimilarity Frechet decision and banded DTW match the great-circle oracle', async () => {
  await expectCases(DYNAMIC_PROGRAM_FIXTURE.greatCircle, true);
}, 120000);

it('GPUTrackSimilarity Frechet decision exits early on long tracks and agrees with the distance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A 2000-vertex track against 40-vertex tracks: one close (within), one that leaves the corridor
  // halfway (the free-space program dies there), one with a far endpoint (lower bound decides).
  const long = Array.from({length: 2000}, (_, index) => [index * 0.05, Math.sin(index * 0.01)]);
  const close = Array.from({length: 40}, (_, index) => [index * 2.5, Math.sin(index * 0.25) + 0.2]);
  const away = close.map(([x, y], index) => [x, index > 20 ? y + 30 : y]);
  const farEnd = close.map(([x, y], index) => [x, index === 39 ? y + 100 : y]);
  const harness = createDynamicProgramHarness(device, [long, long, long], [close, away, farEnd]);
  try {
    const distances = (await harness.run(0)).frechet;
    for (const epsilon of [1, 3, 10, 50]) {
      const result = await harness.run(epsilon);
      distances.forEach((distance, index) => {
        // Decisions agree with the distance kernel away from its f32 rounding.
        if (Math.abs(distance - epsilon) > 1e-3 * epsilon) {
          expect(result.within[index], `pair ${index} epsilon ${epsilon}`).toBe(
            distance <= epsilon ? 1 : 0
          );
        }
      });
    }
  } finally {
    harness.destroy();
  }
  // Cap exceeded: both tracks longer than the shorter-track cap.
  const capped = createDynamicProgramHarness(device, [long], [long], {
    maximumFrechetVertexCount: 64
  });
  try {
    const result = await capped.run(1e9);
    expect(result.status[0] & STATUS.frechetCapExceeded).toBeTruthy();
    expect(result.within[0]).toBe(0);
    expect(Number.isNaN(result.dynamicTimeWarping[0])).toBe(true);
  } finally {
    capped.destroy();
  }
  device.destroy?.();
}, 120000);

it('decision-only pruning preserves the independent oracle decisions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [cases, spatialContext] of [
    [DYNAMIC_PROGRAM_FIXTURE.planar, undefined],
    [DYNAMIC_PROGRAM_FIXTURE.greatCircle, GREAT_CIRCLE]
  ] as const) {
    for (const entry of cases) {
      const harness = createDynamicProgramHarness(device, [entry.a], [entry.b], {
        outputMode: 'decision',
        spatialContext
      });
      try {
        for (const {epsilon, within} of entry.decisions) {
          expect((await harness.run(epsilon)).within, entry.name).toEqual([within]);
        }
      } finally {
        harness.destroy();
      }
    }
  }
}, 300000);
