// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import alpsVisibility from '../oracles/fixtures/parity-wp5-round8-alps-dem-visibility.json';
import canyonVisibility from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-visibility.json';
import ridgeVisibility from '../oracles/fixtures/parity-wp5-round8-ridge-visibility.json';
import alpsGeomorphons from '../oracles/fixtures/parity-wp5-deep-alps-dem-geomorphons.json';
import canyonGeomorphons from '../oracles/fixtures/parity-wp5-deep-grand-canyon-dem-geomorphons.json';
import ridgeGeomorphons from '../oracles/fixtures/parity-wp5-deep-ridge-geomorphons.json';
import planeGeomorphons from '../oracles/fixtures/parity-wp5-deep-plane-geomorphons.json';
import alpsHorizon from '../oracles/fixtures/parity-wp5-round10-alps-dem-horizon.json';
import canyonHorizon from '../oracles/fixtures/parity-wp5-round10-grand-canyon-dem-horizon.json';
import ridgeHorizon from '../oracles/fixtures/parity-wp5-round10-ridge-horizon.json';
import alpsShadow from '../oracles/fixtures/parity-wp5-round8-alps-dem-shadow.json';
import canyonShadow from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-shadow.json';
import ridgeShadow from '../oracles/fixtures/parity-wp5-round8-ridge-shadow.json';

export type RasterInput = {
  name: string;
  width: number;
  height: number;
  cellSize: number;
  elevation: number[];
  elevationSha256: string;
};
export type Fixture<Case> = {
  versions: Record<string, string>;
  conventions: string;
  generatorSha256: string;
  commands: string[][];
  input: RasterInput;
  cases: Case[];
};
export type VisibilityCase = {
  name: string;
  observerHeight: number;
  targetHeight: number;
  curvatureCoefficient: number;
  maximumDistance: number;
  observers: [number, number][];
  grass: number[][];
  gdal: number[][];
  grassCounts: number[];
  gdalCounts: number[];
  referenceDisagreements: number[][];
};
export type GeomorphonCase = {
  name: string;
  comparison: 'anglev1' | 'anglev2' | 'anglev2-distance';
  searchRadius: number;
  skipRadius: number;
  flatThresholdDegrees: number;
  flatDistance: number;
  rawForms: number[];
  rawTernary: number[];
  forms: number[];
  ternary: number[];
};
export type HorizonCase = {
  name: string;
  maximumRadius: number;
  curvatureCoefficient: number;
  horizon: number[];
  skyViewFactor: number[];
  point: {x: number; y: number; horizons: {azimuth: number; angle: number; distance: number}[]}[];
};
export type ShadowCase = {
  name: string;
  azimuthDegrees: number;
  altitudeDegrees: number;
  sunVisibility: number[];
};
export const VISIBILITY_FIXTURES = [alpsVisibility, canyonVisibility, ridgeVisibility].map(value =>
  decodeOracleFixture<Fixture<VisibilityCase>>(value)
);
export const GEOMORPHON_FIXTURES = [
  alpsGeomorphons,
  canyonGeomorphons,
  ridgeGeomorphons,
  planeGeomorphons
].map(value => decodeOracleFixture<Fixture<GeomorphonCase>>(value));
export const HORIZON_FIXTURES = [alpsHorizon, canyonHorizon, ridgeHorizon].map(value =>
  decodeOracleFixture<Fixture<HorizonCase>>(value)
);
export const SHADOW_FIXTURES = [alpsShadow, canyonShadow, ridgeShadow].map(value =>
  decodeOracleFixture<Fixture<ShadowCase>>(value)
);

/** Report all references before asserting; diagnostics never substitute a loose equality bound. */
export function measureComparison(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  profile: 'E' | 'D' | 'R' | 'L' = 'E',
  override?: {absolute: number; relative: number}
) {
  const finite = expected.filter(Number.isFinite);
  const range = finite.length ? Math.max(...finite) - Math.min(...finite) : 0;
  const tolerance = {
    profile,
    absolute:
      profile === 'E'
        ? 0
        : profile === 'D'
          ? 1e-4
          : (profile === 'L' ? 1e-6 : 1e-5) * Math.max(1, range),
    relative: profile === 'E' || profile === 'D' ? 0 : 1e-4
  };
  if (override) Object.assign(tolerance, override);
  let mismatches = Math.abs(actual.length - expected.length);
  let maxAbs = 0;
  let maxRel = 0;
  for (const [index, value] of expected.entries()) {
    const observed = actual[index];
    if (!Number.isFinite(value) || !Number.isFinite(observed)) {
      if (!(Number.isNaN(value) && Number.isNaN(observed)) && value !== observed) mismatches++;
      continue;
    }
    const error = Math.abs(observed - value);
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, error / Math.max(Math.abs(value), Number.EPSILON));
    if (error > tolerance.absolute + tolerance.relative * Math.abs(value)) mismatches++;
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
  return mismatches;
}
export function assertComparisons(mismatches: number[]) {
  for (const count of mismatches) expect(count).toBe(0);
}
