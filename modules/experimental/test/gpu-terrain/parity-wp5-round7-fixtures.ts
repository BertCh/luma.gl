// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import snapFirst from '../oracles/fixtures/parity-wp5-round7-snap-alps-0.json';
import snapSecond from '../oracles/fixtures/parity-wp5-round7-snap-alps-1.json';
import snapSynthetic from '../oracles/fixtures/parity-wp5-round7-snap-synthetic.json';
import summitsFirst from '../oracles/fixtures/parity-wp5-round7-summits-alps-0.json';
import summitsSecond from '../oracles/fixtures/parity-wp5-round7-summits-alps-1.json';
import summitsSynthetic from '../oracles/fixtures/parity-wp5-round7-summits-synthetic.json';

export const EXACT_TOLERANCE = {profile: 'E', absolute: 0, relative: 0};
// D profile: local ground distances here are < 100 m; 8 ULPs stay below 1 mm.
export const DISTANCE_TOLERANCE = {profile: 'D', absolute: 1e-3, relative: 2e-6};
export type Tolerance = {profile: string; absolute: number; relative: number};
type RasterCase = {
  name: string;
  width: number;
  height: number;
  elevation: number[];
  elevationSha256: string;
  validity: number[];
  scale: number;
  offset: number;
  cellSize: [number, number];
  maximumRadiusPixels: number;
  source: {
    kind: string;
    sha256?: string;
    manifestSha256?: string;
    fullShape?: number[];
    coreSize?: number;
    halo?: number;
    fullReferenceVerified?: boolean;
  };
};
type SnapCase = RasterCase & {
  candidates: number[];
  candidateHeights?: number[];
  candidateRadii?: number[];
  settings: {radius: number; maximumMove: number; maximumHeightChange: number; interior: boolean};
  expected: {
    positions: number[];
    heights: number[];
    status: number[];
    snapDistance: number[];
    radiusClamped: number;
    heightTolerance: Tolerance;
  };
};
type SummitsCase = RasterCase & {
  radius: number;
  minimumDrop: number;
  policy: 'reject' | 'ignore';
  radiusClamped: number;
  expected: {mask: number[]; drop: number[]; ids: number[]};
};
type Fixture<Case> = {
  versions: Record<string, string>;
  conventions: string;
  generatorSha256: string;
  cases: Case[];
};
export const SNAP_FIXTURES = [snapFirst, snapSecond, snapSynthetic].map(fixture =>
  decodeOracleFixture<Fixture<SnapCase>>(fixture)
);
export const SUMMITS_FIXTURES = [summitsFirst, summitsSecond, summitsSynthetic].map(fixture =>
  decodeOracleFixture<Fixture<SummitsCase>>(fixture)
);
export const SNAP_CASES = SNAP_FIXTURES.flatMap(fixture => fixture.cases);
export const SUMMITS_CASES = SUMMITS_FIXTURES.flatMap(fixture => fixture.cases);
export const SNAP_REFERENCE =
  'Composite scipy.spatial.cKDTree.query_ball_point + scipy.interpolate.RegularGridInterpolator + scipy.ndimage.binary_erosion';
export const SUMMITS_REFERENCE =
  'Composite scipy.ndimage.maximum_filter + binary_erosion (rank maxima and ring drop)';

export function compareValues(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  tolerance: Tolerance = EXACT_TOLERANCE
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (const [index, target] of expected.entries()) {
    const value = actual[index];
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
