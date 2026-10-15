// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  compareCurvatureValues,
  CURVATURE_CASES,
  CURVATURE_FIXTURES,
  CURVATURE_KINDS
} from './parity-wp5-curvature-fixtures';

it('WP5 curvature references retain provenance, all cells and exact validity masks', () => {
  expect(CURVATURE_CASES).toHaveLength(67);
  for (const fixture of CURVATURE_FIXTURES) {
    expect(fixture.versions['xdem']).toBe('0.2.3');
    expect(fixture.versions['scipy']).toBeTruthy();
    expect(fixture.generatorSha256).toMatch(/^[a-f0-9]{64}$/);
    for (const testCase of fixture.cases) {
      const pixelCount = testCase.width * testCase.height;
      expect(testCase.elevation).toHaveLength(pixelCount);
      expect(testCase.inputValidity).toHaveLength(pixelCount);
      expect(testCase.expectedValidity).toHaveLength(pixelCount);
      for (const kind of CURVATURE_KINDS) {
        expect(testCase.expected[kind]).toHaveLength(pixelCount);
        compareCurvatureValues(
          testCase.name,
          'reference fixture structural check (not GPU execution)',
          `${kind}-finite-mask`,
          testCase.expected[kind].map(value => (Number.isFinite(value) ? 1 : 0)),
          testCase.expectedValidity,
          {profile: 'E', absolute: 0, relative: 0}
        );
      }
    }
  }
});

it('WP5 independent Hessians reproduce analytic mixed-quadratic derivatives and orientation', () => {
  const cases = CURVATURE_CASES.filter(testCase => testCase.name.startsWith('mixed-quadratic'));
  expect(cases).toHaveLength(12);
  for (const testCase of cases) {
    const partials = testCase.partials!;
    const radius = testCase.method === 'florinsky' ? 2 : 1;
    const rowSign = testCase.rowDirection === 'south' ? 1 : -1;
    for (const name of ['east', 'north', 'eastEast', 'eastNorth', 'northNorth'] as const) {
      const actual: number[] = [];
      const expected: number[] = [];
      for (let row = radius; row < testCase.height - radius; row++) {
        for (let column = radius; column < testCase.width - radius; column++) {
          const east = column - 7;
          const north = 6 - row;
          const derivatives = {
            east: ((2 - east / 2 + north / 8) / 2) * 1.25,
            north: ((1 + east / 8 + north) / 2) * 1.25 * rowSign,
            eastEast: -0.15625,
            eastNorth: 0.0390625 * rowSign,
            northNorth: 0.3125
          };
          actual.push(partials[name][row * testCase.width + column]);
          expected.push(derivatives[name]);
        }
      }
      compareCurvatureValues(
        testCase.name,
        'analytic derivatives of the generator polynomial (reference self-check, not GPU)',
        name,
        actual,
        expected,
        {profile: 'R', absolute: 1e-10, relative: 1e-10}
      );
    }
  }
});

it('WP5 real DEM windows preserve identical input cells at their shared seam', () => {
  for (const method of ['evans-young', 'zevenbergen-thorne', 'florinsky']) {
    const west = CURVATURE_CASES.find(testCase => testCase.name === `alps-adjacent-0-${method}`)!;
    const east = CURVATURE_CASES.find(testCase => testCase.name === `alps-adjacent-1-${method}`)!;
    expect(west.source.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(east.source.sha256).toBe(west.source.sha256);
    expect(east.source.coreOrigin![1] - west.source.coreOrigin![1]).toBe(96);
    const westOverlap: number[] = [];
    const eastOverlap: number[] = [];
    for (let row = 0; row < west.height; row++) {
      for (let column = 0; column < 4; column++) {
        westOverlap.push(west.elevation[row * west.width + 96 + column]);
        eastOverlap.push(east.elevation[row * east.width + column]);
      }
    }
    compareCurvatureValues(
      `alps-${method}`,
      'source PNG overlapping halo cells (reference self-check, not GPU)',
      'input-seam',
      eastOverlap,
      westOverlap,
      {profile: 'E', absolute: 0, relative: 0}
    );
  }
});

it('WP5 curvature references preserve flat-plane zeros and the dome flat-gradient convention', () => {
  for (const testCase of CURVATURE_CASES) {
    if (testCase.name.startsWith('flat-')) {
      for (const kind of CURVATURE_KINDS) {
        const finite = testCase.expected[kind].filter(Number.isFinite);
        compareCurvatureValues(
          testCase.name,
          'analytic constant surface (reference self-check, not GPU)',
          kind,
          finite,
          finite.map(() => 0),
          {profile: 'R', absolute: 1e-10, relative: 0}
        );
      }
    }
    if (testCase.name.startsWith('dome-')) {
      const center = 6 * testCase.width + 7;
      compareCurvatureValues(
        testCase.name,
        'analytic dome center: flat-gradient directional zero and negative Hessian trace',
        'profile/plan/tangential/laplacian',
        CURVATURE_KINDS.map(kind => testCase.expected[kind][center]),
        [0, 0, 0, -0.234375],
        {profile: 'R', absolute: 1e-10, relative: 0}
      );
      expect(testCase.expected.profile[center + 1]).toBeGreaterThan(0);
      expect(testCase.expected.plan[center + 1]).toBeGreaterThan(0);
      expect(testCase.expected.tangential[center + 1]).toBeGreaterThan(0);
    }
  }
});
