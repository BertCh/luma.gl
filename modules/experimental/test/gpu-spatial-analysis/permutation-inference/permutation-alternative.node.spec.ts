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
import {
  PERMUTATION_ALTERNATIVE_REFERENCE,
  PERMUTATION_ALTERNATIVE_TOLERANCES
} from './permutation-alternative-reference';

const R = PERMUTATION_ALTERNATIVE_REFERENCE.simulatedCount;

const ESDA_PINS = PERMUTATION_ALTERNATIVE_REFERENCE.alternatives;
const {pValueDecimals} = PERMUTATION_ALTERNATIVE_TOLERANCES;

describe('permutation alternatives', () => {
  it("defaults to 'two-sided' and rejects the removed 'less' alias", () => {
    expect(resolvePermutationAlternative('test', undefined)).toBe('two-sided');
    expect(resolvePermutationAlternative('test', 'lesser')).toBe('lesser');
    expect(() => resolvePermutationAlternative('test', 'less' as 'lesser')).toThrow(/alternative/);
    expect(() => resolvePermutationAlternative('test', 'sideways' as 'greater')).toThrow(
      /alternative/
    );
    expect(getGPUPermutationMetadata({seed: 1, permutationCount: 99}).alternative).toBe(
      'two-sided'
    );
    expect(
      getGPUPermutationMetadata({seed: 1, permutationCount: 99}, {alternative: 'lesser'})
        .alternative
    ).toBe('lesser');
  });

  it('matches esda greater and lesser exactly, and directed except where simulations tie the observed value', () => {
    for (const [name, pin] of Object.entries(ESDA_PINS)) {
      const counts = {greater: pin.greater, lesser: pin.lesser};
      expect(getPermutationPValue('greater', counts, R).pValue, name).toBeCloseTo(
        pin.greaterP,
        pValueDecimals
      );
      expect(getPermutationPValue('lesser', counts, R).pValue, name).toBeCloseTo(
        pin.lesserP,
        pValueDecimals
      );
      const directed = getPermutationPValue('directed', counts, R).pValue;
      if (pin.minRuleDiffers) {
        // esda counts the ties only in g; min(g, l) counts them on both sides, so it is never smaller.
        expect(directed, name).toBeGreaterThanOrEqual(pin.directed - 1e-12);
      } else {
        expect(directed, name).toBeCloseTo(pin.directed, pValueDecimals);
      }
    }
  });

  it('does not report the smallest attainable p when every simulation ties the observed value', () => {
    const counts = {greater: R, lesser: R};
    // esda 2.10 gives directed p = 1/21 = 0.047619 here (and 1.95 for 'two-sided', above 1).
    expect(ESDA_PINS['all_tied'].directed).toBeCloseTo(1 / 21, pValueDecimals);
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
      getGPUPermutationParameterValues(
        {seed: 1, permutationCount: 99},
        {maximumPermutations: 99}
      )[2]
    ).toBe(99);
    expect(() =>
      getGPUPermutationParameterValues({seed: 1, permutationCount: 100}, {maximumPermutations: 99})
    ).toThrow(/maximumPermutations/);
    expect(() => getGPUPermutationParameterValues({seed: 1, permutationCount: 0})).toThrow(
      /permutationCount/
    );
    expect(() =>
      getGPUPermutationMetadata({seed: 1, permutationCount: 500}, {maximumPermutations: 99})
    ).toThrow(/maximumPermutations/);
  });

  it('rejects a significance level outside (0, 1)', () => {
    for (const significanceLevel of [0, 1, -0.1, Number.NaN]) {
      expect(() =>
        getGPUPermutationParameterValues({seed: 1, permutationCount: 9, significanceLevel})
      ).toThrow(/significanceLevel/);
    }
  });
});
