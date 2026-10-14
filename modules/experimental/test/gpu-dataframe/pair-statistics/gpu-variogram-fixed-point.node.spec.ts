// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getPairHistogramReadWGSL,
  quantizePairAmountOnCPU,
  quantizeScaledTermOnCPU
} from '../../../src/gpu-dataframe/pair-statistics/pair-histogram';
import {getPairStatisticsUnitVarianceOnCPU} from './pair-statistics-unit-oracle';
import {forEachPair, getDistanceBin, isNearBinEdge} from './pair-statistics-oracle';
import {createVariogramOutlierScene, VARIOGRAM_OUTLIER_FRAME} from './variogram-outlier-fixture';
import {VARIOGRAM_OUTLIER_REFERENCES} from './variogram-outlier-reference';

const fround = Math.fround;

type Replay = {
  /** Matheron semivariance per lag from the current fixed-point arithmetic. */
  current: number[];
  /** The former arithmetic: unit `2^24 / (0.5 * range^2)`, one channel. */
  former: number[];
  counts: number[];
  edgeCounts: number[];
  /** `u * 2^-25`, the documented absolute bound of one bin mean. */
  bound: number;
  overflow: boolean;
};

/**
 * Replays the kernel arithmetic of `GPUVariogram` over every pair (f32 offsets, centered values,
 * `0.5 * d * d * scale` in f32, `accumulateScaled` routing) with exact BigInt sums.
 */
function replayMatheron(scene: {positions: Float32Array; values: Float32Array}): Replay {
  const {lagCount, maximumDistance, bounds} = VARIOGRAM_OUTLIER_FRAME;
  const frame = {bounds, maximumDistance};
  const rows = scene.values.length;
  let mean = 0;
  for (const value of scene.values) {
    mean += value;
  }
  mean /= rows;
  const centered = Float32Array.from(scene.values, value => value - mean);
  const unit = getPairStatisticsUnitVarianceOnCPU(scene.values);
  const scale = fround(16777216 / unit);
  const range = Math.max(...scene.values) - Math.min(...scene.values);
  const formerScale = fround(16777216 / (0.5 * range * range));
  const small = new Array<bigint>(lagCount).fill(0n);
  const large = new Array<bigint>(lagCount).fill(0n);
  const former = new Array<bigint>(lagCount).fill(0n);
  const counts = new Array<number>(lagCount).fill(0);
  const edgeCounts = new Array<number>(lagCount).fill(0);
  let overflow = false;
  forEachPair(scene, frame, 'unordered', ({focus, neighbor, distance}) => {
    const lag = getDistanceBin(distance, maximumDistance, lagCount);
    const difference = fround(centered[neighbor] - centered[focus]);
    const term = fround(fround(fround(0.5 * difference) * difference) * scale);
    const routed = quantizeScaledTermOnCPU(term);
    if (routed.channel === 'small') {
      small[lag] += routed.amount;
    } else if (routed.channel === 'large') {
      large[lag] += routed.amount;
    }
    overflow ||= routed.overflow;
    former[lag] += BigInt(
      quantizePairAmountOnCPU(fround(fround(fround(0.5 * difference) * difference) * formerScale))
    );
    counts[lag]++;
    if (isNearBinEdge(distance, maximumDistance, lagCount)) {
      edgeCounts[lag]++;
    }
  });
  return {
    current: counts.map((count, lag) =>
      count ? (Number(small[lag]) + Number(large[lag]) * 4096) / scale / count : NaN
    ),
    former: counts.map((count, lag) => (count ? Number(former[lag]) / formerScale / count : NaN)),
    counts,
    edgeCounts,
    bound: unit * 2 ** -25,
    overflow
  };
}

const SCENES = {
  clean: createVariogramOutlierScene(),
  'isolated-2e4': createVariogramOutlierScene(2e4, 'isolated'),
  'isolated-1e5': createVariogramOutlierScene(1e5, 'isolated'),
  'embedded-2e4': createVariogramOutlierScene(2e4, 'embedded'),
  'embedded-1e5': createVariogramOutlierScene(1e5, 'embedded')
};

it('Matheron fixed-point arithmetic stays within its bound of scikit-gstat despite an outlier', () => {
  for (const [name, scene] of Object.entries(SCENES)) {
    const reference = VARIOGRAM_OUTLIER_REFERENCES[name];
    const replay = replayMatheron(scene);
    expect(replay.overflow, name).toBe(false);
    let compared = 0;
    for (let lag = 0; lag < reference.semivariances.length; lag++) {
      // f32 distances may move a pair lying on a lag edge into the next lag.
      if (Math.abs(replay.counts[lag] - reference.pairCounts[lag]) > replay.edgeCounts[lag]) {
        throw new Error(`${name} lag ${lag}: pair count ${replay.counts[lag]}`);
      }
      if (replay.counts[lag] !== reference.pairCounts[lag]) {
        continue;
      }
      compared++;
      const expected = reference.semivariances[lag];
      // Fixed-point rounding (u * 2^-25) plus f32 term/centering rounding and the second channel.
      const tolerance = replay.bound + 2e-6 * expected;
      expect(Math.abs(replay.current[lag] - expected), `${name} lag ${lag}`).toBeLessThanOrEqual(
        tolerance
      );
    }
    expect(compared, name).toBeGreaterThanOrEqual(6);
  }
}, 120000);

it('the former range-pegged unit lost the typical terms to an isolated outlier', () => {
  // Documents what the test above guards: with the outlier isolated, the exact semivariance is
  // about 1 in every lag, and a unit of 0.5 * range^2 / 2^24 rounded every typical term away.
  for (const name of ['isolated-2e4', 'isolated-1e5']) {
    const reference = VARIOGRAM_OUTLIER_REFERENCES[name];
    const replay = replayMatheron(SCENES[name as keyof typeof SCENES]);
    const lag = 3;
    expect(replay.counts[lag]).toBe(reference.pairCounts[lag]);
    const formerError = Math.abs(replay.former[lag] - reference.semivariances[lag]);
    const currentError = Math.abs(replay.current[lag] - reference.semivariances[lag]);
    expect(formerError / reference.semivariances[lag], name).toBeGreaterThan(0.5);
    expect(currentError, name).toBeLessThan(replay.bound + 2e-6 * reference.semivariances[lag]);
  }
}, 120000);

it('pair histogram read helpers saturate the pair count and expose the overflow word', () => {
  const plain = getPairHistogramReadWGSL('accumulators', 4);
  // A nonzero high word saturates to 0xffffffff (matches the Ripley and correlogram counts).
  expect(plain).toMatch(
    /fn readPairCount\(slot: u32, channel: u32\) -> u32 \{[^}]*select\(\s*accumulators\[accumulatorsOffset \+ word\],\s*0xffffffffu,\s*accumulators\[accumulatorsOffset \+ word \+ 1u\] != 0u/
  );
  expect(plain).not.toContain('readPairOverflow');
  // The overflow word follows the slotCount * channelCount * 2 accumulator words.
  const withOverflow = getPairHistogramReadWGSL('accumulators', 4, '', 8);
  expect(withOverflow).toContain('accumulators[accumulatorsOffset + 64u] != 0u');
});

it('quantizeScaledTermOnCPU routes terms by magnitude and clamps beyond 2^63 coarse units', () => {
  expect(quantizeScaledTermOnCPU(0)).toEqual({channel: 'none', amount: 0n, overflow: false});
  expect(quantizeScaledTermOnCPU(NaN).channel).toBe('none');
  expect(quantizeScaledTermOnCPU(2.4)).toEqual({channel: 'small', amount: 2n, overflow: false});
  expect(quantizeScaledTermOnCPU(2.6).amount).toBe(3n);
  expect(quantizeScaledTermOnCPU(4294967040).channel).toBe('small');
  // 2^40 fixed-point units: 2^28 coarse units of 2^12.
  expect(quantizeScaledTermOnCPU(2 ** 40)).toEqual({
    channel: 'large',
    amount: 2n ** 28n,
    overflow: false
  });
  // 2^70 units is 2^58 coarse units: needs the high word.
  expect(quantizeScaledTermOnCPU(2 ** 70).amount).toBe(2n ** 58n);
  const clamped = quantizeScaledTermOnCPU(2 ** 80);
  expect(clamped.overflow).toBe(true);
  expect(clamped.amount).toBe(2n ** 63n);
  expect(quantizeScaledTermOnCPU(Infinity).overflow).toBe(true);
});

it('getPairStatisticsUnitVarianceOnCPU ignores outliers and stays at most the variance', () => {
  const clean = SCENES.clean.values;
  expect(getPairStatisticsUnitVarianceOnCPU(clean)).toBeGreaterThan(0.9);
  expect(getPairStatisticsUnitVarianceOnCPU(clean)).toBeLessThanOrEqual(1.05);
  // One sentinel inflates the variance to about 2.5e6 but not the unit.
  const unit = getPairStatisticsUnitVarianceOnCPU(SCENES['isolated-1e5'].values);
  expect(unit).toBeGreaterThan(0.9);
  expect(unit).toBeLessThan(1.1);
  // Mostly tied values: the interquartile range is 0, the unit takes its floor.
  const tied = new Float32Array(1000);
  tied[0] = 8;
  const tiedVariance = (8 * 8) / 1000 - (8 / 1000) ** 2;
  expect(getPairStatisticsUnitVarianceOnCPU(tied)).toBeCloseTo(tiedVariance / 4096, 12);
  expect(getPairStatisticsUnitVarianceOnCPU(new Float32Array(10).fill(3))).toBe(0);
  expect(getPairStatisticsUnitVarianceOnCPU([NaN])).toBe(0);
});
