// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  getGPUSpatialScanStatisticLayout,
  getGPUSpatialScanStatisticParameterValues,
  GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS
} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {
  computeScanOracle,
  createRegionalClusterScene,
  getOracleBernoulliLogLikelihoodRatio,
  getOracleLogLikelihoodRatio
} from './scan-statistic-oracle';

// Pinned from <scratchpad>/fix/work/B/scan_models.py (numpy 2.5.4, scipy 1.18.1): the Bernoulli LLR of
// Kulldorff (1997) written out in the script and cross-checked with the G statistic of
// scipy.stats.chi2_contingency; the space-time permutation LLR of Kulldorff et al. (2005).
const BERNOULLI = [
  {c: 30, n: 100, C: 80, N: 1000, llr: 24.579701904333405, high: true},
  {c: 5, n: 200, C: 80, N: 1000, llr: 6.484678091601779, high: false},
  {c: 12, n: 40, C: 50, N: 400, llr: 4.909912833225064, high: true},
  {c: 1, n: 60, C: 50, N: 400, llr: 5.416855326540684, high: false}
];
const SPACE_TIME_PERMUTATION = {
  c: 12,
  e: 10.24,
  C: 25,
  llr: 0.252631359879397,
  llrLow: 0.6462331346283131
};

describe('GPUSpatialScanStatistic parameters', () => {
  test('packs defaults: time windows default to half the study period (word 0)', () => {
    const values = getGPUSpatialScanStatisticParameterValues({seed: 5, permutationCount: 99});
    expect(values.length).toBe(8);
    expect(values[2]).toBe(99);
    expect(new Float32Array(values.buffer)[3]).toBe(0.5);
    expect(values[4]).toBe(2 ** 31);
    expect(values[5]).toBe(0);
    expect(values[6]).toBe(0);
    expect(values[7]).toBe(0);
  });

  test('packs explicit limits, shape and rate direction', () => {
    const values = getGPUSpatialScanStatisticParameterValues({
      seed: 2 ** 33 + 7,
      permutationCount: 10,
      maximumTimeBuckets: 3,
      maximumWindowZones: 40,
      windowShape: 'nearest',
      rateDirection: 'both'
    });
    expect(values[0]).toBe(7);
    expect(values[1]).toBe(2);
    expect(values[4]).toBe(40);
    expect(values[5]).toBe(3);
    expect(values[6]).toBe(1);
    expect(values[7]).toBe(2);
  });

  test('validates seed, permutationCount and the population fraction', () => {
    expect(() =>
      getGPUSpatialScanStatisticParameterValues({seed: -1, permutationCount: 9})
    ).toThrow();
    expect(() =>
      getGPUSpatialScanStatisticParameterValues({seed: 1, permutationCount: 0})
    ).toThrow();
    expect(() =>
      getGPUSpatialScanStatisticParameterValues({
        seed: 1,
        permutationCount: 9,
        maximumPopulationFraction: 0
      })
    ).toThrow();
  });
});

describe('GPUSpatialScanStatistic layout', () => {
  test('defaults to every zone while the tables fit', () => {
    expect(getGPUSpatialScanStatisticLayout(100)).toEqual({usableZones: 100, listLength: 100});
    expect(getGPUSpatialScanStatisticLayout(4096)).toEqual({usableZones: 4096, listLength: 4096});
  });

  test('a smaller capacity adds one look-ahead row', () => {
    expect(getGPUSpatialScanStatisticLayout(100, 32)).toEqual({usableZones: 32, listLength: 33});
    expect(getGPUSpatialScanStatisticLayout(100, 500)).toEqual({usableZones: 100, listLength: 100});
  });

  test('bounds the table memory', () => {
    const layout = getGPUSpatialScanStatisticLayout(20000);
    expect(layout.listLength * 20000).toBeLessThanOrEqual(
      GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS
    );
    expect(layout.usableZones).toBeLessThan(20000);
    expect(() => getGPUSpatialScanStatisticLayout(20000, 20000)).toThrow();
    expect(() => getGPUSpatialScanStatisticLayout(0)).toThrow();
    expect(() => getGPUSpatialScanStatisticLayout(10, 0)).toThrow();
  });
});

describe('scan statistic oracle against closed-form LLRs', () => {
  test('Bernoulli LLR', () => {
    for (const {c, n, C, N, llr, high} of BERNOULLI) {
      const direction = high ? 'high' : 'low';
      expect(getOracleBernoulliLogLikelihoodRatio(c, n, C, N, direction)).toBeCloseTo(llr, 9);
      expect(getOracleBernoulliLogLikelihoodRatio(c, n, C, N, 'both')).toBeCloseTo(llr, 9);
      expect(getOracleBernoulliLogLikelihoodRatio(c, n, C, N, high ? 'low' : 'high')).toBe(0);
    }
  });

  test('space-time permutation LLR uses the margin expectation', () => {
    const {c, e, C, llr, llrLow} = SPACE_TIME_PERMUTATION;
    expect(getOracleLogLikelihoodRatio(c, e, C, C)).toBeCloseTo(llr, 9);
    expect(getOracleLogLikelihoodRatio(c, e, C, C, 'low')).toBe(0);
    expect(getOracleLogLikelihoodRatio(1, 2.5, 20, 20, 'low')).toBeCloseTo(llrLow, 9);
  });
});

describe('scan statistic oracle semantics', () => {
  test('a regional cluster of 60 zones is one cluster, not fragments', () => {
    const scene = createRegionalClusterScene(12, 5, 11);
    const result = computeScanOracle(scene);
    const primary = result.clusters[0];
    // The population fraction (0.5 of 144 zones = 72) governs, far above the old 32-zone cap.
    expect(primary.zoneCount).toBeGreaterThanOrEqual(50);
    expect(primary.zoneCount).toBeLessThanOrEqual(72);
    expect(primary.pValue).toBeLessThanOrEqual(0.05);
    expect(primary.truncatedByZoneCap).toBe(false);
    expect(result.overflow).toBe(0);
    // Everything in the elevated block belongs to the primary window or is not significant.
    for (const cluster of result.clusters.slice(1)) {
      expect(cluster.pValue).toBeGreaterThan(0.05);
    }
  });

  test('a 32-zone capacity fragments the same scene and reports the truncation', () => {
    const scene = {...createRegionalClusterScene(12, 5, 11), maximumWindowZones: 32};
    const result = computeScanOracle(scene);
    expect(result.clusters[0].zoneCount).toBeLessThanOrEqual(32);
    expect(result.clusters[0].truncatedByZoneCap).toBe(true);
    expect(result.overflow).toBeGreaterThan(0);
  });

  test('the Bernoulli model finds a planted cluster and conditions on the totals', () => {
    const base = createRegionalClusterScene(8, 2, 3);
    const controls = new Uint32Array(base.cases.length);
    for (let zone = 0; zone < controls.length; zone++) controls[zone] = 100 - base.cases[zone];
    const result = computeScanOracle({
      ...base,
      model: 'bernoulli',
      baseline: undefined,
      controls,
      permutationCount: 49
    });
    expect(result.clusters[0].center).toBeLessThan(24);
    expect(result.clusters[0].pValue).toBeLessThanOrEqual(0.05);
    expect(result.totalWeight).toBe(64 * 100);
  });

  test('the space-time permutation model ignores a trend shared by every zone', () => {
    const zones = 25;
    const buckets = 4;
    const positions = new Float32Array(zones * 2);
    const cases = new Uint32Array(zones * buckets);
    for (let zone = 0; zone < zones; zone++) {
      positions[zone * 2] = (zone % 5) * 10;
      positions[zone * 2 + 1] = Math.floor(zone / 5) * 10;
      for (let bucket = 0; bucket < buckets; bucket++)
        cases[zone * buckets + bucket] = 4 * (bucket + 1);
    }
    const result = computeScanOracle({
      positions,
      cases,
      timeBuckets: buckets,
      model: 'space-time-permutation',
      maximumPopulationFraction: 0.5,
      windowShape: 'circle',
      maximumClusters: 2,
      seed: 3,
      permutationCount: 19
    });
    // Every cell equals its margin expectation exactly, so no window has a positive LLR.
    expect(result.clusters.length).toBe(0);
    expect(result.maxima[0]).toBeLessThan(1e-9);
    // Replicates permute the times, so they do vary (with both margins fixed).
    expect(Math.max(...result.maxima.slice(1))).toBeGreaterThan(0);
  });
});
