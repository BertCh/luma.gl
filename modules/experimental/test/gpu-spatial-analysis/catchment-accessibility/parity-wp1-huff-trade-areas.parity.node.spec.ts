// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import fixture from '../../oracles/fixtures/parity-wp1-huff-trade-areas.json';

it('preserves all groceries, complete CSR probabilities, transpose and zero/tie policies', () => {
  expect(fixture.versions.scipy).toBeTruthy();
  expect(fixture.cases[0].provenance.sourceCount).toBe(105808);
  expect(fixture.cases[0].attractiveness.length).toBe(1255);
  for (const testCase of fixture.cases) {
    for (const metric of ['weights', 'attractiveness', 'demand'] as const) {
      const values = Float32Array.from(testCase[metric]);
      expect(createHash('sha256').update(new Uint8Array(values.buffer)).digest('hex')).toBe(
        testCase.inputFloat32Sha256[metric]
      );
    }
    expect(testCase.expected.probabilities.length).toBe(testCase.weights.length);
    const transposed = new Map<string, number>();
    for (let facility = 0; facility < testCase.attractiveness.length; facility++) {
      for (
        let slot = testCase.facilityOffsets[facility];
        slot < testCase.facilityOffsets[facility + 1];
        slot++
      ) {
        transposed.set(
          `${testCase.facilityNeighbors[slot]}:${facility}`,
          testCase.facilityWeights[slot]
        );
      }
    }
    for (let row = 0; row < testCase.n; row++) {
      const neighbors = testCase.neighbors.slice(testCase.offsets[row], testCase.offsets[row + 1]);
      expect(
        neighbors.every((neighbor, index) => index === 0 || neighbor > neighbors[index - 1])
      ).toBe(true);
      let probability = 0;
      for (let slot = testCase.offsets[row]; slot < testCase.offsets[row + 1]; slot++) {
        expect(transposed.get(`${row}:${testCase.neighbors[slot]}`)).toBe(testCase.weights[slot]);
        probability += testCase.expected.probabilities[slot];
      }
      expect(probability).toBeCloseTo(testCase.expected.served[row], 12);
    }
    const servedDemand = testCase.demand.reduce(
      (sum, value, row) => sum + value * testCase.expected.served[row],
      0
    );
    expect(testCase.expected.expectedDemand.reduce((sum, value) => sum + value, 0)).toBeCloseTo(
      servedDemand,
      7
    );
  }
  expect(fixture.cases[1].expected.tradeArea).toEqual([1, 4294967295, 4294967295, 2, 1]);
});
