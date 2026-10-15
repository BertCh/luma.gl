// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import watershedsFirst from '../oracles/fixtures/parity-wp5-watersheds-alps-0.json';
import watershedsSecond from '../oracles/fixtures/parity-wp5-watersheds-alps-1.json';
import watershedsThird from '../oracles/fixtures/parity-wp5-watersheds-alps-2.json';
import watershedsSynthetic from '../oracles/fixtures/parity-wp5-watersheds-synthetic.json';
import reliefFirst from '../oracles/fixtures/parity-wp5-relief-alps-0-0.json';
import reliefSecond from '../oracles/fixtures/parity-wp5-relief-alps-0-1.json';
import reliefThird from '../oracles/fixtures/parity-wp5-relief-alps-1-0.json';
import reliefFourth from '../oracles/fixtures/parity-wp5-relief-alps-1-1.json';
import reliefSynthetic from '../oracles/fixtures/parity-wp5-relief-synthetic.json';
import type {GPULocalDominanceGeometry} from '../../src/gpu-terrain/relief-visualization/gpu-local-dominance';
import type {GPUMultiScaleReliefScales} from '../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';

type Tolerance = {profile: string; absolute: number; relative: number};
type Fixture<Case> = {versions: Record<string, string>; generatorSha256: string; cases: Case[]};
type WatershedCase = {
  name: string;
  width: number;
  height: number;
  source: {kind: string; fullGraphAccumulationVerified?: boolean; catchmentCells?: number};
  sourceFixture: string;
  sourceFixtureSha256: string;
  directions: number[];
  pourPoints: number[];
  basinLabels: number[];
  pourPointLabels: number[];
  catchments: {pointIndex: number; mask: number[]}[];
};
type ReliefCase = {
  name: string;
  width: number;
  height: number;
  source: {
    kind: string;
    fullFilterVerified?: boolean;
    unitedDominanceVerified?: boolean;
    sha256?: string;
    coreOrigin?: [number, number];
    halo?: number;
  };
  scales: GPUMultiScaleReliefScales;
  geometry: GPULocalDominanceGeometry;
  radii: number[];
  observerHeight: number;
  verticalExaggeration: number;
  elevation: number[];
  validity: number[];
  expectedValidity: number[];
  expected: {dominance: number[]; relief: number[]};
  tolerances: {dominance: Tolerance; relief: Tolerance};
  rvtCrossCheck: {maxAbs: number; derivedAbsoluteBound: number};
};
export const WATERSHED_FIXTURES = [
  watershedsFirst,
  watershedsSecond,
  watershedsThird,
  watershedsSynthetic
].map(fixture => decodeOracleFixture<Fixture<WatershedCase>>(fixture));
export const WATERSHED_CASES = WATERSHED_FIXTURES.flatMap(fixture => fixture.cases);
export const RELIEF_FIXTURES = [
  reliefFirst,
  reliefSecond,
  reliefThird,
  reliefFourth,
  reliefSynthetic
].map(fixture => decodeOracleFixture<Fixture<ReliefCase>>(fixture));
export const RELIEF_CASES = RELIEF_FIXTURES.flatMap(fixture => fixture.cases);
export const EXACT_TOLERANCE = {profile: 'E', absolute: 0, relative: 0};

export function compareRound4Values(
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
