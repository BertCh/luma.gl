// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import fixtureJson from '../../oracles/fixtures/parity-wp1-emerging-hot-spots.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

it('preserves expanded-cube provenance, full arrays and tied/missing trend cases', () => {
  const fixture = decodeOracleFixture<typeof fixtureJson>(fixtureJson);
  expect(fixture.classificationChecks).toBe(22);
  expect(fixture.versions.esda).toBeTruthy();
  expect(fixture.cases[0].provenance.sourceCount).toBe(43557);
  expect(fixture.cases[0].values.reduce<number>((sum, value) => sum + Number(value), 0)).toBe(
    43557
  );
  for (const testCase of fixture.cases) {
    const values = Float32Array.from(testCase.values.map(Number));
    expect(createHash('sha256').update(new Uint8Array(values.buffer)).digest('hex')).toBe(
      testCase.inputFloat32Sha256
    );
    const cellCount = testCase.width * testCase.height;
    expect(values.length).toBe(cellCount * testCase.sliceCount);
    for (const [metric, expected] of Object.entries(testCase.expected)) {
      expect(expected.length).toBe(
        ['giZScores', 'pValues', 'adjustedPValues'].includes(metric)
          ? values.length
          : metric === 'globalStatistics'
            ? 4
            : cellCount
      );
    }
    expect(testCase.expected.category.every(value => value >= 0 && value <= 17)).toBe(true);
  }
  expect(
    [...new Set(fixture.cases.flatMap(testCase => testCase.expected.category))].sort(
      (first, second) => first - second
    )
  ).toEqual(Array.from({length: 18}, (_, index) => index));
  const synthetic = fixture.cases[1];
  expect(Number.isNaN(synthetic.values[2 * 12 + 11])).toBe(true);
  expect(synthetic.expected.lastValidSlice[5]).toBe(4294967295);
  expect(synthetic.expected.category[2]).toBe(17);
});
