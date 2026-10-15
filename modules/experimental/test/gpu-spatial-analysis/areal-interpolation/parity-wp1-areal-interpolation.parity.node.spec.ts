// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import fixtureJson from '../../oracles/fixtures/parity-wp1-areal-interpolation.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

const fixture = decodeOracleFixture<typeof fixtureJson>(fixtureJson);

it('areal fixture preserves full-data provenance and polygonized support contracts', () => {
  expect(fixture.versions.tobler).toBeTruthy();
  expect(fixture.versions.shapely).toBeTruthy();
  expect(fixture.cases[0].provenance.placesCount).toBe(105808);
  expect(fixture.cases[0].sourceZoneCount).toBe(791);
  for (const testCase of fixture.cases) {
    const hash = createHash('sha256');
    for (const values of [
      Uint32Array.from(testCase.sourceZones),
      Uint32Array.from(testCase.targetZones),
      Float32Array.from(testCase.sourceValues),
      Float32Array.from(testCase.sourceRates)
    ]) {
      hash.update(new Uint8Array(values.buffer));
    }
    expect(hash.digest('hex')).toBe(testCase.inputSha256);
    expect(testCase.sourceZones).toHaveLength(testCase.width * testCase.height);
    expect(testCase.targetZones).toHaveLength(testCase.width * testCase.height);
    expect(testCase.offsets).toHaveLength(testCase.targetZoneCount + 1);
    expect(testCase.offsets.at(-1)).toBe(testCase.neighbors.length);
    expect(testCase.areas).toHaveLength(testCase.neighbors.length);
    expect(testCase.areas.every(value => Number.isInteger(value) && value > 0)).toBe(true);
    expect(testCase.sourceAreas.filter(value => value === 0)).toHaveLength(testCase.invalidCount);
    for (const reference of Object.values(testCase.references)) {
      expect(reference.extensive).toHaveLength(testCase.targetZoneCount);
      expect(reference.intensive).toHaveLength(testCase.targetZoneCount);
      expect(reference.extensiveWeights).toHaveLength(testCase.neighbors.length);
      expect(reference.intensiveWeights).toHaveLength(testCase.neighbors.length);
      expect(reference.sourceTransferredFractions.map(Number.isNaN)).toEqual(
        testCase.sourceAreas.map(value => value === 0)
      );
      expect(reference.massTotal + reference.massLost).toBeCloseTo(
        testCase.sourceValues.reduce((sum, value) => sum + value, 0),
        8
      );
    }
    expect(testCase.references['overlap-overlap'].massTotal).toBeGreaterThan(
      testCase.references['zone-overlap'].massTotal
    );
  }
  const edgeCase = fixture.cases[1];
  expect(edgeCase.targetUncoveredFractions.slice(3)).toEqual([1, 1]);
  expect(edgeCase.references['zone-overlap'].extensive.slice(3)).toEqual([0, 0]);
  expect(Number.isNaN(edgeCase.references['zone-overlap'].sourceTransferredFractions[3])).toBe(
    true
  );
});
