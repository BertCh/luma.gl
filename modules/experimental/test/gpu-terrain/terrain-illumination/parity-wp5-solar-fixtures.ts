// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import irradianceWest from '../../oracles/fixtures/parity-wp5-solar-irradiance-alps-0.json';
import irradianceEast from '../../oracles/fixtures/parity-wp5-solar-irradiance-alps-1.json';
import irradianceSynthetic from '../../oracles/fixtures/parity-wp5-solar-irradiance-synthetic.json';
import shadowWest from '../../oracles/fixtures/parity-wp5-solar-shadow-alps-0.json';
import shadowEast from '../../oracles/fixtures/parity-wp5-solar-shadow-alps-1.json';
import shadowSynthetic from '../../oracles/fixtures/parity-wp5-solar-shadow-synthetic.json';
import position from '../../oracles/fixtures/parity-wp5-solar-noaa-position.json';
import meinel from '../../oracles/fixtures/parity-wp5-solar-meinel.json';
import type {GPUSolarShadowMaskSettings} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask';

export type SolarTolerance = {profile: string; absolute: number; relative: number};
export type SolarFixture<Case> = {
  versions: Record<string, string>;
  generatorSha256: string;
  cases: Case[];
};
type SolarGeometry = {
  name: string;
  width: number;
  height: number;
  directionCount: number;
  slope: number[];
  aspect: number[];
  skyViewFactor: number[];
  horizon: number[];
  reference: string;
};
export type IrradianceCase = SolarGeometry & {
  sunTable: number[];
  sampleCount: number;
  diffuseIrradiance: number;
  angularRadiusDegrees: number;
  expected: Record<'sunHours' | 'direct' | 'diffuse' | 'total', number[]>;
  expectedValidity: number[];
  tolerances: Record<'sunHours' | 'direct' | 'diffuse' | 'total', SolarTolerance>;
};
export type ShadowCase = SolarGeometry & {
  settings: GPUSolarShadowMaskSettings;
  expected: Record<'sunVisibility' | 'illumination', number[]>;
  tolerances: Record<'sunVisibility' | 'illumination', SolarTolerance>;
};
export type PositionCase = {
  name: string;
  timestamp: number;
  representedTimestamp: string;
  parameters: number[];
  positions: number[];
  expected: Record<'azimuth' | 'altitude' | 'zenith' | 'daylight', number[]>;
  spaReference: Record<'azimuth' | 'altitude' | 'zenith' | 'daylight', number[]>;
  angleBounds: number[];
  directionBound: number;
  reference: string;
  tolerance: SolarTolerance;
};

export const IRRADIANCE_FIXTURES = [irradianceWest, irradianceEast, irradianceSynthetic].map(
  fixture => decodeOracleFixture<SolarFixture<IrradianceCase>>(fixture)
);
export const SHADOW_FIXTURES = [shadowWest, shadowEast, shadowSynthetic].map(fixture =>
  decodeOracleFixture<SolarFixture<ShadowCase>>(fixture)
);
export const POSITION_FIXTURE = decodeOracleFixture<
  SolarFixture<PositionCase> & {
    events: {name: string; noaaSeconds: number; spaSeconds: number; boundSeconds: number}[];
  }
>(position);
export const MEINEL_FIXTURE =
  decodeOracleFixture<SolarFixture<{name: string; altitudes: number[]; directNormal: number[]}>>(
    meinel
  );
export const IRRADIANCE_CASES = IRRADIANCE_FIXTURES.flatMap(fixture => fixture.cases);
export const SHADOW_CASES = SHADOW_FIXTURES.flatMap(fixture => fixture.cases);
export const EXACT_SOLAR_TOLERANCE = {profile: 'E', absolute: 0, relative: 0};

/** Full arrays, circular azimuth differences, exact nonfinite masks; report before asserting. */
export function compareSolarValues(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  tolerance: SolarTolerance,
  absoluteBounds?: readonly number[],
  assertEquality = true
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < expected.length; index++) {
    const value = actual[index];
    const target = expected[index];
    if (!Number.isFinite(value) || !Number.isFinite(target)) {
      if (!(Number.isNaN(value) && Number.isNaN(target)) && value !== target) {
        mismatches++;
      }
      continue;
    }
    const difference =
      metric === 'azimuth'
        ? Math.abs(((((value - target + 180) % 360) + 360) % 360) - 180)
        : Math.abs(value - target);
    maxAbs = Math.max(maxAbs, difference);
    maxRel = Math.max(maxRel, difference / Math.max(Math.abs(target), Number.EPSILON));
    if (
      difference >
      (absoluteBounds?.[index] ?? tolerance.absolute) + tolerance.relative * Math.abs(target)
    ) {
      mismatches++;
    }
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
      tolerance: absoluteBounds
        ? {
            ...tolerance,
            perRowAbsoluteBounds: true,
            maximumAbsolute: Math.max(...absoluteBounds.filter(Number.isFinite))
          }
        : tolerance
    })
  );
  // Soft assertions ensure a failing angle does not hide the other output comparisons.
  if (assertEquality) expect.soft(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
}

/** Model error is reported separately and is never folded into the f32 budget. */
export function reportSolarModelDifference(
  className: string,
  fixture: PositionCase,
  azimuth: readonly number[],
  altitude: readonly number[],
  daylight: readonly number[]
): void {
  for (const [metric, values] of [
    ['azimuth', azimuth],
    ['altitude', altitude],
    ['daylight', daylight]
  ] as const) {
    compareSolarValues(
      className,
      fixture.name,
      'NREL SPA / pvlib (model accuracy only; published timing bounds checked separately)',
      metric,
      values,
      fixture.spaReference[metric],
      {
        profile: 'model difference, report only (mismatches count nonzero differences)',
        absolute: 0,
        relative: 0
      },
      undefined,
      false
    );
  }
}
