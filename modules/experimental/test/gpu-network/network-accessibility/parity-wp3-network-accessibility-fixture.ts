// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import showcase from '../../oracles/fixtures/parity-wp3-accessibility-showcase.json';
import syntheticInput from '../../oracles/fixtures/parity-wp3-network-accessibility-synthetic-input.json';
import syntheticZero from '../../oracles/fixtures/parity-wp3-network-accessibility-synthetic-zero.json';
import syntheticBoundary from '../../oracles/fixtures/parity-wp3-network-accessibility-synthetic-boundary.json';
import syntheticExponential from '../../oracles/fixtures/parity-wp3-network-accessibility-synthetic-exponential.json';
import syntheticPower from '../../oracles/fixtures/parity-wp3-network-accessibility-synthetic-power.json';
import syntheticUnlimited from '../../oracles/fixtures/parity-wp3-network-accessibility-synthetic-unlimited.json';
import chicagoInput from '../../oracles/fixtures/parity-wp3-network-accessibility-chicago-input.json';
import chicagoZero from '../../oracles/fixtures/parity-wp3-network-accessibility-chicago-zero.json';
import chicagoFiveMinutes from '../../oracles/fixtures/parity-wp3-network-accessibility-chicago-five-minutes.json';
import chicagoExponential from '../../oracles/fixtures/parity-wp3-network-accessibility-chicago-exponential.json';
import chicagoPower from '../../oracles/fixtures/parity-wp3-network-accessibility-chicago-power.json';
import chicagoUnlimited from '../../oracles/fixtures/parity-wp3-network-accessibility-chicago-unlimited.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import type {GPUNetworkAccessibilitySettings} from '../../../src/gpu-network/network-accessibility';
import type {Tolerance} from '../parity-wp3-comparison';

export type AccessibilityInput = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  originCount: number;
  facilityNodes: number[];
  costs: number[];
  supply: number[];
  demand: number[];
  source: {fullNodeCount?: number; fullEdgeCount?: number; sha256?: Record<string, string>};
};
export type OutputName = 'cumulative' | 'gravity' | 'catchment' | 'ratios';
export type AccessibilityCase = {
  method?: '2sfca' | '3sfca';
  case: string;
  reference: string;
  settings: GPUNetworkAccessibilitySettings;
  expected: Record<OutputName, number[]>;
  tolerances: Record<OutputName, Tolerance>;
};

export const ACCESSIBILITY_FIXTURES = [
  {
    input: decodeOracleFixture<AccessibilityInput>(syntheticInput),
    cases: [
      syntheticZero,
      syntheticBoundary,
      syntheticExponential,
      syntheticPower,
      syntheticUnlimited
    ].map(fixture => decodeOracleFixture<AccessibilityCase>(fixture))
  },
  {
    input: decodeOracleFixture<AccessibilityInput>(chicagoInput),
    cases: [
      chicagoZero,
      chicagoFiveMinutes,
      chicagoExponential,
      chicagoPower,
      chicagoUnlimited
    ].map(fixture => decodeOracleFixture<AccessibilityCase>(fixture))
  }
];

export const showcaseAccessibility = decodeOracleFixture<{
  input: AccessibilityInput;
  cases: AccessibilityCase[];
}>(showcase);
