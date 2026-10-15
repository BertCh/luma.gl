// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import fixtureJson from '../../oracles/fixtures/parity-wp1-spatial-clustering.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

const fixture = decodeOracleFixture<typeof fixtureJson>(fixtureJson);

it('DBSCAN fixture preserves full showcase inputs, provenance and exact core partitions', () => {
  expect(fixture.versions.sklearn).toBeTruthy();
  expect(fixture.versions.pyproj).toBeTruthy();
  expect(fixture.tolerance).toEqual({profile: 'E', absolute: 0, relative: 0});
  expect(fixture.cases[0].n).toBe(43557);
  expect(fixture.cases[0].provenance.sourceSha256).toMatch(/^[a-f0-9]{64}$/);
  expect(fixture.cases[0].boundaryShellPairs).toBe(0);
  for (const testCase of fixture.cases) {
    expect(testCase.positions).toHaveLength(testCase.n * 2);
    expect(testCase.labels).toHaveLength(testCase.n);
    expect(testCase.coreMask).toHaveLength(testCase.n);
    const uploaded = Float32Array.from(testCase.positions);
    expect(Array.from(uploaded)).toEqual(testCase.positions);
    expect(createHash('sha256').update(new Uint8Array(uploaded.buffer)).digest('hex')).toBe(
      testCase.inputFloat32Sha256
    );
    const coreLabels = new Set(testCase.labels.filter((_, row) => testCase.coreMask[row] === 1));
    expect([...coreLabels].sort((left, right) => left - right)).toEqual(
      Array.from({length: testCase.clusterCount}, (_, index) => index)
    );
    for (const [row, allowed] of Object.entries(testCase.allowedBorderLabels)) {
      expect(testCase.coreMask[Number(row)]).toBe(0);
      expect(allowed).toContain(testCase.labels[Number(row)]);
      expect(allowed.length).toBeGreaterThan(1);
      for (const label of allowed) expect(coreLabels.has(label)).toBe(true);
    }
  }
  expect(fixture.cases[1].allowedBorderLabels).toEqual({'8': [0, 1]});
  expect(fixture.cases[2].labels).toEqual([0, 0, 1]);
  expect(fixture.cases[3].labels).toEqual([-1, -1, -1]);
});
