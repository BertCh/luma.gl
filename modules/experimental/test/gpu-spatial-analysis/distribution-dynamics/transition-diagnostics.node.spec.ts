// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeHomogeneityOracle, computeSteadyStateOracle} from './transition-diagnostics-oracle';
import * as giddy from './markov-giddy-fixture';

function expectCloseOrNaN(actual: number[], expected: number[], relative: number): void {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    if (Number.isNaN(value)) {
      expect(actual[index], `[${index}]`).toBeNaN();
    } else {
      expect(
        Math.abs(actual[index] - value),
        `[${index}] ${actual[index]} vs ${value}`
      ).toBeLessThanOrEqual(1e-9 + relative * Math.abs(value));
    }
  });
}

it('homogeneity oracle reproduces giddy Q, LR, dof and p-values', () => {
  for (const [counts, classCount, conditionCount, expected] of [
    [giddy.A_COUNTS, 3, 3, giddy.A_HOMOGENEITY],
    [giddy.B_GOOD_COUNTS, 4, 3, giddy.B_GOOD_HOMOGENEITY],
    [giddy.B_ALL_COUNTS, 4, 5, giddy.B_ALL_HOMOGENEITY]
  ] as const) {
    const actual = computeHomogeneityOracle(counts, classCount, conditionCount);
    expect(actual.degreesOfFreedom).toBe(expected.dof);
    expectCloseOrNaN([actual.q, actual.likelihoodRatio], [expected.Q, expected.LR], 1e-9);
    expectCloseOrNaN(
      [actual.qPValue, actual.likelihoodRatioPValue],
      [expected.Qp, expected.LRp],
      1e-6
    );
  }
});

it('the documented dof rule differs from giddy exactly by the empty pooled rows', () => {
  // B_* have one from-state (3) without any transition: giddy adds (0 - 1) * (0 - 1) = 1 for it.
  expect(giddy.B_GOOD_GIDDY_HOMOGENEITY.dof - giddy.B_GOOD_HOMOGENEITY.dof).toBe(1);
  expect(giddy.B_ALL_GIDDY_HOMOGENEITY.dof - giddy.B_ALL_HOMOGENEITY.dof).toBe(1);
  expect(giddy.A_HOMOGENEITY.dof).toBe(12);
});

it('steady-state oracle reproduces giddy steady_state and the NaN rule', () => {
  const chains = (counts: readonly number[], classCount: number, conditionCount: number) => {
    const pooled = new Array<number>(classCount * classCount).fill(0);
    const result: number[] = [];
    for (let condition = 0; condition < conditionCount; condition++) {
      const slice = counts.slice(
        condition * classCount * classCount,
        (condition + 1) * classCount * classCount
      );
      slice.forEach((value, index) => (pooled[index] += value));
    }
    result.push(...computeSteadyStateOracle(pooled, classCount));
    for (let condition = 0; condition < conditionCount; condition++) {
      result.push(
        ...computeSteadyStateOracle(
          counts.slice(
            condition * classCount * classCount,
            (condition + 1) * classCount * classCount
          ),
          classCount
        )
      );
    }
    return result;
  };
  expectCloseOrNaN(chains(giddy.A_COUNTS, 3, 3), giddy.A_STEADY_STATES, 1e-9);
  expectCloseOrNaN(chains(giddy.B_GOOD_COUNTS, 4, 3), giddy.B_GOOD_STEADY_STATES, 1e-9);
  expectCloseOrNaN(chains(giddy.B_ALL_COUNTS, 4, 5), giddy.B_ALL_STEADY_STATES, 1e-9);
});
