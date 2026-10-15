// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import alpsFirst from '../../oracles/fixtures/parity-wp5-hydrology-alps-0.json';
import alpsSecond from '../../oracles/fixtures/parity-wp5-hydrology-alps-1.json';
import alpsThird from '../../oracles/fixtures/parity-wp5-hydrology-alps-2.json';
import synthetic from '../../oracles/fixtures/parity-wp5-hydrology-synthetic.json';

export const INDEX_OUTPUTS = [
  'slope',
  'specificCatchmentArea',
  'wetnessIndex',
  'streamPowerIndex'
] as const;
export type HydrologyMetric =
  | (typeof INDEX_OUTPUTS)[number]
  | 'heightAboveDrainage'
  | 'drainageCells'
  | 'streamOrder';
type Tolerance = {profile: string; absolute: number; relative: number};
export type HydrologyCase = {
  name: string;
  width: number;
  height: number;
  cellSize: [number, number];
  minimumSlope: number;
  source: {
    kind: string;
    fullGraphAccumulationVerified?: boolean;
    fullGraphCellCount?: number;
    catchmentCells?: number;
  };
  elevation: number[];
  directions: number[];
  streams: number[];
  accumulation: number[];
  expected: Record<HydrologyMetric, number[]>;
  tolerances: Record<'heightAboveDrainage' | (typeof INDEX_OUTPUTS)[number], Tolerance>;
  pyshedsOrder: number[];
  pyshedsOrderMismatchCells: number[];
};
export const HYDROLOGY_FIXTURES = [alpsFirst, alpsSecond, alpsThird, synthetic].map(fixture =>
  decodeOracleFixture<{
    versions: Record<string, string>;
    generatorSha256: string;
    cases: HydrologyCase[];
  }>(fixture)
);
export const HYDROLOGY_CASES = HYDROLOGY_FIXTURES.flatMap(fixture => fixture.cases);
export const EXACT_HYDROLOGY_TOLERANCE = {profile: 'E', absolute: 0, relative: 0};

export function compareHydrologyValues(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  tolerance: Tolerance
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < expected.length; index++) {
    const value = actual[index];
    const target = expected[index];
    if (!Number.isFinite(value) || !Number.isFinite(target)) {
      if (!(Number.isNaN(value) && Number.isNaN(target)) && value !== target) mismatches++;
      continue;
    }
    const difference = Math.abs(value - target);
    maxAbs = Math.max(maxAbs, difference);
    maxRel = Math.max(maxRel, difference / Math.max(Math.abs(target), Number.EPSILON));
    if (difference > tolerance.absolute + tolerance.relative * Math.abs(target)) mismatches++;
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: caseName,
      reference,
      n: expected.length,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance
    })
  );
  expect.soft(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
}
