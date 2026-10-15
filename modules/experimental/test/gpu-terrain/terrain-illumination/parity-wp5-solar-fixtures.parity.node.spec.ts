// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, readdirSync} from 'node:fs';
import {expect, it} from 'vitest';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {getGPUSolarDirectNormalIrradiance} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import {getGPUSolarPositionParameterValues} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';
import {
  compareSolarValues,
  EXACT_SOLAR_TOLERANCE,
  IRRADIANCE_CASES,
  IRRADIANCE_FIXTURES,
  MEINEL_FIXTURE,
  POSITION_FIXTURE,
  SHADOW_CASES,
  SHADOW_FIXTURES
} from './parity-wp5-solar-fixtures';
import type {SolarFixture} from './parity-wp5-solar-fixtures';

type RawInputCase = {name: string} & Partial<
  Record<'horizon' | 'slope' | 'aspect' | 'skyViewFactor', number[]>
>;

it('WP5 solar references preserve versions, complete arrays and float32 inputs', () => {
  const fixtureDirectory = new URL('../../oracles/fixtures/', import.meta.url);
  const rawCases = new Map(
    readdirSync(fixtureDirectory)
      .filter(name => name.startsWith('parity-wp5-solar-') && name.endsWith('.json'))
      .flatMap(
        name =>
          decodeOracleFixture<SolarFixture<RawInputCase>>(
            JSON.parse(readFileSync(new URL(name, fixtureDirectory), 'utf8'))
          ).cases
      )
      .map(fixture => [fixture.name, fixture])
  );
  for (const fixture of [
    ...IRRADIANCE_FIXTURES,
    ...SHADOW_FIXTURES,
    POSITION_FIXTURE,
    MEINEL_FIXTURE
  ]) {
    expect(fixture.versions['pvlib']).toBe('0.16.1');
    expect(fixture.versions['scipy']).toBeTruthy();
    expect(fixture.generatorSha256).toMatch(/^[a-f0-9]{64}$/);
  }
  expect(IRRADIANCE_CASES).toHaveLength(5);
  expect(SHADOW_CASES).toHaveLength(12);
  expect(POSITION_FIXTURE.cases).toHaveLength(8);
  for (const fixture of [...IRRADIANCE_CASES, ...SHADOW_CASES]) {
    const pixelCount = fixture.width * fixture.height;
    expect(fixture.horizon).toHaveLength(pixelCount * fixture.directionCount);
    for (const name of ['horizon', 'slope', 'aspect', 'skyViewFactor'] as const) {
      const rawInput = rawCases.get(fixture.name)![name]!;
      expect(rawInput.every(value => !Number.isFinite(value) || Math.fround(value) === value)).toBe(
        true
      );
      // JSON module transformation can alter a final f64 decimal digit. Verify
      // that the actual f32 upload reconstructs the exact original f32 inputs.
      expect(Array.from(Float32Array.from(fixture[name]))).toEqual(rawInput);
    }
    for (const values of Object.values(fixture.expected)) expect(values).toHaveLength(pixelCount);
  }
  for (const fixture of POSITION_FIXTURE.cases) {
    expect(fixture.positions).toHaveLength(548);
    for (const values of Object.values(fixture.expected)) expect(values).toHaveLength(274);
    compareSolarValues(
      'GPUSolarPosition parameter packer',
      fixture.name,
      'float32 split J2000 reference',
      'parameters',
      Array.from(getGPUSolarPositionParameterValues({timestamp: fixture.timestamp})),
      fixture.parameters,
      EXACT_SOLAR_TOLERANCE
    );
  }
});

it('WP5 irradiance references retain full finite masks and additive energy components', () => {
  for (const fixture of IRRADIANCE_CASES) {
    for (const [metric, values] of Object.entries(fixture.expected)) {
      compareSolarValues(
        'GPUSolarIrradiance reference integrity',
        fixture.name,
        'reference finite mask (not GPU execution)',
        `${metric}-validity`,
        values.map(value => (Number.isFinite(value) ? 1 : 0)),
        fixture.expectedValidity,
        EXACT_SOLAR_TOLERANCE
      );
    }
    compareSolarValues(
      'GPUSolarIrradiance reference integrity',
      fixture.name,
      'direct plus diffuse energy identity',
      'totalEnergy',
      fixture.expected.direct.map((value, index) => value + fixture.expected.diffuse[index]),
      fixture.expected.total,
      {profile: 'reference integrity', absolute: 1e-10, relative: 1e-12}
    );
  }
  const empty = IRRADIANCE_CASES.find(fixture => fixture.name === 'synthetic-empty-table')!;
  expect(empty.expected.total.filter(Number.isFinite).every(value => value === 0)).toBe(true);
  const real = IRRADIANCE_CASES.filter(fixture => fixture.name.startsWith('alps-'));
  for (const fixture of real) {
    expect(fixture.sampleCount).toBe(96);
    expect(fixture.expected.total.every(Number.isFinite)).toBe(true);
    expect(
      Math.max(...fixture.expected.total) - Math.min(...fixture.expected.total)
    ).toBeGreaterThan(1000);
  }
});

it('WP5 disk references preserve ties, wraparound, half-disks and output-specific missingness', () => {
  const hard = SHADOW_CASES.find(fixture => fixture.name === 'synthetic-mask-0')!;
  compareSolarValues(
    'GPUSolarShadowMask reference integrity',
    hard.name,
    'analytic hard tie and adjacent float32 horizons',
    'hardVisibility',
    hard.expected.sunVisibility.slice(0, 7),
    [0, 1, 0, 1, 0, 1, 0],
    EXACT_SOLAR_TOLERANCE
  );
  const half = SHADOW_CASES.find(fixture => fixture.name === 'synthetic-mask-6')!;
  compareSolarValues(
    'GPUSolarShadowMask reference integrity',
    half.name,
    'uniform half-disk area',
    'halfVisibility',
    [half.expected.sunVisibility[0]],
    [0.5],
    {profile: 'reference integrity', absolute: 1e-14, relative: 0}
  );
  const wrapped = SHADOW_CASES.filter(fixture =>
    ['synthetic-mask-2', 'synthetic-mask-3', 'synthetic-mask-4'].includes(fixture.name)
  );
  for (const fixture of wrapped.slice(1)) {
    for (const metric of ['sunVisibility', 'illumination'] as const) {
      compareSolarValues(
        'GPUSolarShadowMask reference integrity',
        fixture.name,
        'periodic compass azimuth identity',
        metric,
        fixture.expected[metric],
        wrapped[0].expected[metric],
        {profile: 'reference integrity', absolute: 1e-13, relative: 0}
      );
    }
  }
  expect(Number.isNaN(hard.expected.sunVisibility[7])).toBe(true);
  for (const index of [8, 9, 10]) {
    expect(Number.isFinite(hard.expected.sunVisibility[index])).toBe(true);
    expect(Number.isNaN(hard.expected.illumination[index])).toBe(true);
  }
});

it('WP5 Meinel CPU helper agrees with pvlib Kasten-Young air mass and published power law', () => {
  const fixture = MEINEL_FIXTURE.cases[0];
  compareSolarValues(
    'GPUSolarIrradiance Meinel helper',
    fixture.name,
    'composite: pvlib.atmosphere.get_relative_airmass(kastenyoung1989), Meinel 1353*0.7^(AM^0.678)',
    'directNormalWm2',
    fixture.altitudes.map(getGPUSolarDirectNormalIrradiance),
    fixture.directNormal,
    {profile: 'R', absolute: 1e-5, relative: 1e-4}
  );
});
