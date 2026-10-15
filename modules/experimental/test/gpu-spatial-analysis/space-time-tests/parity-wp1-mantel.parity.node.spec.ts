// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import fixture from '../../oracles/fixtures/parity-wp1-mantel.json';
import {makeMantelPairs} from './parity-wp1-mantel-input';

it('preserves independent Mantel provenance and every reconstructed f32 input distance', () => {
  expect(fixture.provenance.sourceCount).toBe(48636);
  expect(fixture.cases[0].n).toBe(2048);
  expect(fixture.versions['scikit-bio']).toBe('0.7.4');
  for (const testCase of fixture.cases) {
    const pairs = makeMantelPairs(testCase.positions);
    const arrays = {
      positions: Float32Array.from(testCase.positions),
      times: Float32Array.from(testCase.times),
      distances: pairs.distances
    };
    for (const name of ['positions', 'times', 'distances'] as const) {
      expect(createHash('sha256').update(new Uint8Array(arrays[name].buffer)).digest('hex')).toBe(
        testCase.inputFloat32Sha256[name]
      );
    }
    expect(pairs.offsets[testCase.n]).toBe((testCase.n * (testCase.n - 1)) / 2);
    expect(testCase.times.length).toBe(testCase.n);
    expect(testCase.null.every(value => Number.isFinite(value) && Math.abs(value) <= 1)).toBe(true);
  }
  expect(fixture.cases[1].null).toHaveLength(4095);
  expect(fixture.cases[2].null).toHaveLength(720);
  expect(fixture.cases[2].exactNull).toBe(true);
  expect(fixture.cases[3].null.every(value => value === 0)).toBe(true);
});
