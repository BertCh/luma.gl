// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import fixtureJson from '../../oracles/fixtures/parity-wp1-inverse-distance-weighting.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

it('preserves native GDAL provenance, full station inputs, nodata and both row orders', () => {
  const fixture = decodeOracleFixture<typeof fixtureJson>(fixtureJson);
  expect(fixture.versions.gdal).toBe('3.12.4');
  expect(fixture.datasets.ghcn.values.length).toBe(6135);
  for (const dataset of Object.values(fixture.datasets)) {
    for (const metric of ['positions', 'values'] as const) {
      const values = Float32Array.from(dataset[metric].map(Number));
      expect(createHash('sha256').update(new Uint8Array(values.buffer)).digest('hex')).toBe(
        dataset.inputFloat32Sha256[metric]
      );
    }
    expect(dataset.positions.length).toBe(dataset.values.length * 2);
  }
  for (const testCase of fixture.cases) {
    expect(testCase.expected.values.length).toBe(testCase.width * testCase.height);
    expect(testCase.expected.counts.length).toBe(testCase.n);
    if (testCase.neighborCount) {
      expect(Math.max(...testCase.expected.counts)).toBe(testCase.neighborCount);
      expect(testCase.adaptedKCutoffQueries.length).toBeGreaterThan(0);
    }
    expect(testCase.expected.values.some(Number.isNaN)).toBe(true);
  }
  for (const north of fixture.cases.filter(
    testCase => testCase.dataset === 'synthetic' && testCase.rowDirection === 'north'
  )) {
    const south = fixture.cases.find(
      testCase => testCase.name === north.name.replace(/north$/, 'south')
    )!;
    expect(north.expected.values.slice(0, 3)).toEqual([10, 30, 50]);
    for (const metric of ['values', 'counts'] as const) {
      for (let row = 0; row < north.height; row++) {
        expect(north.expected[metric].slice(row * north.width, (row + 1) * north.width)).toEqual(
          south.expected[metric].slice(
            (north.height - row - 1) * north.width,
            (north.height - row) * north.width
          )
        );
      }
    }
  }
});
