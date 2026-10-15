// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/parity-wp2-nearest-weights.json';

it('unbounded radius survives fixture decoding as an omitted field, never NaN', () => {
  // The shared decoder reserves JSON null for NaN, not absent optional values.
  expect(decodeOracleFixture<{radius: number}>({radius: null}).radius).toBeNaN();
  const fixture = decodeOracleFixture<{cases: {radius?: number; expected: {offsets: number[]}}[]}>(
    fixtureJson
  );
  expect(fixture.cases[0].radius).toBeUndefined();
  expect(fixture.cases[0].expected.offsets).toEqual(Array.from({length: 877}, (_, row) => row * 8));
  expect(fixture.cases.slice(1).map(entry => entry.radius)).toEqual([1500, 1, 1]);
});
