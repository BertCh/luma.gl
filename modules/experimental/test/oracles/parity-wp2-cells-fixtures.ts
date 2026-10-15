// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {decodeOracleFixture} from './oracle-fixture';
import type {CellsInput, CellsExpected, CellsComparison} from './parity-wp2-cells-harness';
import fixture0 from './fixtures/parity-wp2-cells-edges-compare.json';
import fixture1 from './fixtures/parity-wp2-cells-edges-expected.json';
import fixture2 from './fixtures/parity-wp2-cells-edges-input-0.json';
import fixture3 from './fixtures/parity-wp2-cells-nature-compare.json';
import fixture4 from './fixtures/parity-wp2-cells-nature-expected.json';
import fixture5 from './fixtures/parity-wp2-cells-nature-input-0.json';
import fixture6 from './fixtures/parity-wp2-cells-nature-input-1.json';
import fixture7 from './fixtures/parity-wp2-cells-nature-input-2.json';
import fixture8 from './fixtures/parity-wp2-cells-nature-input-3.json';
import fixture9 from './fixtures/parity-wp2-cells-nature-input-4.json';
import fixture10 from './fixtures/parity-wp2-cells-nature-input-5.json';
export const cellFixtures = [
  (() => {
    const chunks = [fixture5, fixture6, fixture7, fixture8, fixture9, fixture10].map(value =>
      decodeOracleFixture<CellsInput>(value)
    );
    return {
      expected: decodeOracleFixture<CellsExpected>(fixture4),
      comparison: decodeOracleFixture<CellsComparison>(fixture3),
      input: {
        keys: chunks.flatMap(chunk => chunk.keys),
        values: chunks.flatMap(chunk => chunk.values),
        masks: [0, 1, 2, 3].map(index => chunks.flatMap(chunk => chunk.masks[index]))
      }
    };
  })(),
  (() => {
    const chunks = [fixture2].map(value => decodeOracleFixture<CellsInput>(value));
    return {
      expected: decodeOracleFixture<CellsExpected>(fixture1),
      comparison: decodeOracleFixture<CellsComparison>(fixture0),
      input: {
        keys: chunks.flatMap(chunk => chunk.keys),
        values: chunks.flatMap(chunk => chunk.values),
        masks: [0, 1, 2, 3].map(index => chunks.flatMap(chunk => chunk.masks[index]))
      }
    };
  })()
];
