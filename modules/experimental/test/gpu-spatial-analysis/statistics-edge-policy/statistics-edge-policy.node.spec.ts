// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Edge-policy tables (rule 22) against the CPU oracles and the parameter packer. Oracle:
// test/oracles/statistics-edge-policy.py (esda on the analysis rows, policy values where esda
// divides by zero). The GPU kernels are checked against the same rows in
// statistics-edge-policy.spec.ts.

import {expect, it} from 'vitest';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {computeGlobalSpatialStatisticsOracle} from '../global-spatial-statistics/global-spatial-statistics-oracle';
import {
  computeHotSpotOracle,
  computeLocalMoranOracle
} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {
  getEdgePolicyMask,
  getEdgePolicyWeights,
  isWithinEdgePolicyTolerance,
  STATISTICS_EDGE_POLICY
} from './statistics-edge-policy-fixture';

const {tolerances} = STATISTICS_EDGE_POLICY;

it('distinguishes zero General G variance from undefined denominators and small samples', () => {
  const weights = getEdgePolicyWeights();
  for (const {value, count, variance} of [
    {value: 2, count: weights.offsets.length - 1, variance: 0},
    {value: 0, count: weights.offsets.length - 1, variance: NaN},
    {value: 2, count: 3, variance: NaN}
  ]) {
    const values = new Float32Array(weights.offsets.length - 1).fill(NaN);
    values.fill(value, 0, count);
    const {getisOrdG} = computeGlobalSpatialStatisticsOracle({weights, values, islands: 'keep'});
    expect(getisOrdG.varianceNormality).toBe(variance);
    expect(getisOrdG.varianceRandomization).toBe(variance);
    expect(getisOrdG.zNormality).toBeNaN();
    expect(getisOrdG.zRandomization).toBeNaN();
    expect(getisOrdG.pNormality).toBeNaN();
    expect(getisOrdG.pRandomization).toBeNaN();
    if (value === 0) {
      expect(getisOrdG.statistic).toBeNaN();
    } else {
      expect(getisOrdG.statistic).toBeCloseTo(getisOrdG.expected, 14);
    }
  }
});

it('the global oracle follows the edge-policy table', () => {
  const weights = getEdgePolicyWeights();
  for (const testCase of STATISTICS_EDGE_POLICY.global) {
    const oracle = computeGlobalSpatialStatisticsOracle({
      weights,
      values: Float32Array.from(testCase.values),
      mask: getEdgePolicyMask(testCase.mask),
      islands: 'keep'
    });
    const label = testCase.name;
    expect(oracle.count, label).toBe(testCase.expected.count);
    for (const statistic of ['moran', 'geary', 'getisOrdG'] as const) {
      const expected = testCase.expected[statistic];
      const actual = oracle[statistic] as unknown as Record<string, number>;
      for (const [field, value] of Object.entries(expected)) {
        const tolerance = field.startsWith('p')
          ? tolerances.pValue
          : field.startsWith('z')
            ? tolerances.zScore
            : tolerances.statistic;
        expect(
          isWithinEdgePolicyTolerance(actual[field], value, tolerance),
          `${label} ${statistic}.${field}: oracle ${actual[field]} != table ${value}`
        ).toBe(true);
      }
    }
    expect(oracle.blackCount, `${label} blacks`).toBe(testCase.expected.blackCount);
    const joinCount = oracle.joinCount as unknown as Record<string, number>;
    for (const [field, value] of Object.entries(testCase.expected.joinCount)) {
      expect(joinCount[field], `${label} join ${field}`).toBe(value);
    }
  }
});

it('the local Gi* and local Moran oracles follow the edge-policy table', () => {
  const weights = getEdgePolicyWeights();
  for (const testCase of STATISTICS_EDGE_POLICY.local) {
    const input = {
      weights,
      values: Float32Array.from(testCase.values),
      mask: getEdgePolicyMask(testCase.mask),
      islands: 'keep' as const
    };
    const hotSpot = computeHotSpotOracle({...input, selfWeight: 1});
    const localMoran = computeLocalMoranOracle(input);
    for (let row = 0; row < testCase.values.length; row++) {
      expect(
        isWithinEdgePolicyTolerance(
          hotSpot.zScores[row],
          testCase.expected.giStar[row],
          tolerances.zScore
        ),
        `${testCase.name} row ${row} Gi* ${hotSpot.zScores[row]} != ${testCase.expected.giStar[row]}`
      ).toBe(true);
      expect(
        isWithinEdgePolicyTolerance(
          localMoran.localI[row],
          testCase.expected.localI[row],
          tolerances.statistic
        ),
        `${testCase.name} row ${row} I ${localMoran.localI[row]} != ${testCase.expected.localI[row]}`
      ).toBe(true);
    }
  }
});

it('the spatial-autocorrelation packer refuses out-of-range significance levels', () => {
  const {rejectedSignificanceLevels} = STATISTICS_EDGE_POLICY.parameters;
  expect(rejectedSignificanceLevels.length).toBeGreaterThan(4);
  for (const significanceLevel of rejectedSignificanceLevels) {
    expect(
      () => getGPUSpatialAutocorrelationParameterValues({significanceLevel}),
      `significanceLevel ${significanceLevel}`
    ).toThrow(/significanceLevel/);
  }
});
