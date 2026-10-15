// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import syntheticInput from '../../oracles/fixtures/parity-wp3-cost-matrix-synthetic-input.json';
import syntheticZero from '../../oracles/fixtures/parity-wp3-cost-matrix-synthetic-zero.json';
import syntheticBoundary from '../../oracles/fixtures/parity-wp3-cost-matrix-synthetic-boundary.json';
import syntheticUnlimited from '../../oracles/fixtures/parity-wp3-cost-matrix-synthetic-unlimited.json';
import chicagoForwardInput from '../../oracles/fixtures/parity-wp3-cost-matrix-chicago-forward-input.json';
import chicagoForwardFiveMinutes from '../../oracles/fixtures/parity-wp3-cost-matrix-chicago-forward-five-minutes.json';
import chicagoForwardUnlimited from '../../oracles/fixtures/parity-wp3-cost-matrix-chicago-forward-unlimited.json';
import chicagoReverseInput from '../../oracles/fixtures/parity-wp3-cost-matrix-chicago-reverse-input.json';
import chicagoReverseFiveMinutes from '../../oracles/fixtures/parity-wp3-cost-matrix-chicago-reverse-five-minutes.json';
import chicagoReverseUnlimited from '../../oracles/fixtures/parity-wp3-cost-matrix-chicago-reverse-unlimited.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

export type CostMatrixInput = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  nodeCount: number;
  rowCount: number;
  startsPerRow: number;
  offsets: number[];
  neighbors: number[];
  weights: number[];
  startNodes: number[];
  startCosts: number[];
  startRows: number[];
  targets: number[];
};
export type CostMatrixCase = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  budget: number;
  costs: number[];
  compact: number[];
};
export const costMatrixFixtures = [
  {
    input: decodeOracleFixture<CostMatrixInput>(syntheticInput),
    cases: [syntheticZero, syntheticBoundary, syntheticUnlimited].map(value =>
      decodeOracleFixture<CostMatrixCase>(value)
    )
  },
  {
    input: decodeOracleFixture<CostMatrixInput>(chicagoForwardInput),
    cases: [chicagoForwardFiveMinutes, chicagoForwardUnlimited].map(value =>
      decodeOracleFixture<CostMatrixCase>(value)
    )
  },
  {
    input: decodeOracleFixture<CostMatrixInput>(chicagoReverseInput),
    cases: [chicagoReverseFiveMinutes, chicagoReverseUnlimited].map(value =>
      decodeOracleFixture<CostMatrixCase>(value)
    )
  }
];
