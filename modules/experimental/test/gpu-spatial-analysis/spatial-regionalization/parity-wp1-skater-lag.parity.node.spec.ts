// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import fixture from '../../oracles/fixtures/parity-wp1-skater-lag.json';

it('preserves full tract/place provenance, forest support and independent lag/Skater arrays', () => {
  expect(fixture.skater[0].n).toBe(791);
  expect(fixture.skater[0].provenance.sourcePlaces).toBe(105808);
  expect(fixture.versions.spopt).toBe('0.8.0');
  for (const testCase of [...fixture.skater, ...fixture.lag]) {
    const values = Float32Array.from(testCase.values);
    expect(createHash('sha256').update(new Uint8Array(values.buffer)).digest('hex')).toBe(
      testCase.inputFloat32Sha256
    );
    expect(testCase.values).toHaveLength(testCase.n * testCase.columns);
    expect(testCase.offsets).toHaveLength(testCase.n + 1);
    expect(testCase.offsets[testCase.n]).toBe(testCase.neighbors.length);
    for (let row = 0; row < testCase.n; row++) {
      const neighbors = testCase.neighbors.slice(testCase.offsets[row], testCase.offsets[row + 1]);
      expect(neighbors).toEqual([...new Set(neighbors)].sort((first, second) => first - second));
    }
  }
  for (const testCase of fixture.skater) {
    expect(testCase.treeFlags.reduce((sum, value) => sum + value, 0)).toBe(
      testCase.n - testCase.componentCount
    );
    for (const run of testCase.runs) {
      expect(run.labels).toHaveLength(testCase.n);
      expect(run.sizes.reduce((sum, value) => sum + value, 0)).toBe(testCase.n);
      expect(Math.min(...run.sizes)).toBeGreaterThanOrEqual(testCase.floor);
      expect(run.scores.reduce((sum, value) => sum + value, 0)).toBeCloseTo(run.objective, 10);
    }
  }
  for (const testCase of fixture.lag) {
    expect(testCase.expected).toHaveLength(testCase.n * testCase.columns);
    testCase.mask.forEach((included, row) => {
      if (!included)
        expect(
          testCase.expected.slice(row * testCase.columns, (row + 1) * testCase.columns)
        ).toEqual(Array(testCase.columns).fill(0));
    });
  }
});
