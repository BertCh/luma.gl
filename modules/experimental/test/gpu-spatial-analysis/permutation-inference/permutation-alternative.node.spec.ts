// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getGPUPermutationMetadata,
  getGPUPermutationParameterValues
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  getPermutationPValue,
  resolvePermutationAlternative
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-alternative';

const R = 20;

// Oracle: fix/work/A/oracle-u2-alternative.py (esda 2.10.0, esda.significance.calculate_significance)
// on 20 simulated values (`greater`/`lesser` are #{sim >= observed} / #{sim <= observed}).
// `tied` marks fixtures where a simulation equals the observed value.
const ESDA_PINS = {
  distinct_high: {
    greater: 1,
    lesser: 20,
    directed: 0.09523809523809523,
    greaterP: 0.09523809523809523,
    lesserP: 1,
    tied: false
  },
  distinct_mid: {
    greater: 12,
    lesser: 8,
    directed: 0.42857142857142855,
    greaterP: 0.6190476190476191,
    lesserP: 0.42857142857142855,
    tied: false
  },
  distinct_extreme: {
    greater: 0,
    lesser: 20,
    directed: 0.047619047619047616,
    greaterP: 0.047619047619047616,
    lesserP: 1,
    tied: false
  },
  partial_tie_none: {
    greater: 5,
    lesser: 15,
    directed: 0.2857142857142857,
    greaterP: 0.2857142857142857,
    lesserP: 0.7619047619047619,
    tied: false
  },
  distinct_low: {
    greater: 18,
    lesser: 3,
    directed: 0.14285714285714285,
    greaterP: 0.9047619047619048,
    lesserP: 0.19047619047619047,
    tied: true
  },
  partial_tie_zero: {
    greater: 14,
    lesser: 11,
    directed: 0.3333333333333333,
    greaterP: 0.7142857142857143,
    lesserP: 0.5714285714285714,
    tied: true
  },
  all_tied: {
    greater: 20,
    lesser: 20,
    directed: 0.047619047619047616,
    greaterP: 1,
    lesserP: 1,
    tied: true
  }
};

describe('permutation alternatives', () => {
  it("defaults to 'two-sided' and accepts the 'less' alias", () => {
    expect(resolvePermutationAlternative('test', undefined)).toBe('two-sided');
    expect(resolvePermutationAlternative('test', 'less')).toBe('lesser');
    expect(() => resolvePermutationAlternative('test', 'sideways' as 'greater')).toThrow(
      /alternative/
    );
    expect(getGPUPermutationMetadata({seed: 1, permutations: 99}).alternative).toBe('two-sided');
    expect(
      getGPUPermutationMetadata({seed: 1, permutations: 99}, {alternative: 'less'}).alternative
    ).toBe('lesser');
  });

  it('matches esda greater and lesser exactly, and directed except where simulations tie the observed value', () => {
    for (const [name, pin] of Object.entries(ESDA_PINS)) {
      const counts = {greater: pin.greater, lesser: pin.lesser};
      expect(getPermutationPValue('greater', counts, R).pValue, name).toBeCloseTo(pin.greaterP, 12);
      expect(getPermutationPValue('lesser', counts, R).pValue, name).toBeCloseTo(pin.lesserP, 12);
      const directed = getPermutationPValue('directed', counts, R).pValue;
      if (pin.tied) {
        // esda counts the ties only in g; min(g, l) counts them on both sides, so it is never smaller.
        expect(directed, name).toBeGreaterThanOrEqual(pin.directed - 1e-12);
      } else {
        expect(directed, name).toBeCloseTo(pin.directed, 12);
      }
    }
  });

  it('does not report the smallest attainable p when every simulation ties the observed value', () => {
    const counts = {greater: R, lesser: R};
    // esda 2.10 gives directed p = 1/21 = 0.047619 here (and 1.95 for 'two-sided', above 1).
    expect(ESDA_PINS.all_tied.directed).toBeCloseTo(1 / 21, 12);
    expect(getPermutationPValue('directed', counts, R).pValue).toBe(1);
    expect(getPermutationPValue('two-sided', counts, R).pValue).toBe(1);
    expect(getPermutationPValue(undefined, counts, R).pValue).toBe(1);
    // The documented deviations from esda in the partially tied fixtures.
    expect(getPermutationPValue('directed', {greater: 14, lesser: 11}, R).pValue).toBeCloseTo(
      12 / 21,
      12
    );
    expect(getPermutationPValue('directed', {greater: 18, lesser: 3}, R).pValue).toBeCloseTo(
      4 / 21,
      12
    );
  });

  it('two-sided doubles the smaller tail and caps at 1', () => {
    expect(getPermutationPValue('two-sided', {greater: 1, lesser: 20}, R)).toEqual({
      exceedances: 1,
      pValue: 4 / 21
    });
    expect(getPermutationPValue('two-sided', {greater: 12, lesser: 12}, R).pValue).toBe(1);
  });
});

describe('permutation parameters', () => {
  it('rejects a count above the contributor maximum instead of clamping it', () => {
    expect(
      getGPUPermutationParameterValues({seed: 1, permutations: 99}, {maximumPermutations: 99})[2]
    ).toBe(99);
    expect(() =>
      getGPUPermutationParameterValues({seed: 1, permutations: 100}, {maximumPermutations: 99})
    ).toThrow(/maximumPermutations/);
    expect(() => getGPUPermutationParameterValues({seed: 1, permutations: 0})).toThrow(
      /permutations/
    );
    expect(() =>
      getGPUPermutationMetadata({seed: 1, permutations: 500}, {maximumPermutations: 99})
    ).toThrow(/maximumPermutations/);
  });

  it('rejects a significance level outside (0, 1)', () => {
    for (const significanceLevel of [0, 1, -0.1, Number.NaN]) {
      expect(() =>
        getGPUPermutationParameterValues({seed: 1, permutations: 9, significanceLevel})
      ).toThrow(/significanceLevel/);
    }
  });
});
