// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import alps from '../oracles/fixtures/parity-wp5-deep-alps-dem-geomorphons.json';
import canyon from '../oracles/fixtures/parity-wp5-deep-grand-canyon-dem-geomorphons.json';
import ridge from '../oracles/fixtures/parity-wp5-deep-ridge-geomorphons.json';
import plane from '../oracles/fixtures/parity-wp5-deep-plane-geomorphons.json';
import horizon from '../oracles/fixtures/parity-wp5-deep-horizon.json';
import {computeGeomorphons} from './geomorphons/geomorphons-oracle';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {getGPUGeomorphonsParameterValues} from '../../src/gpu-terrain/geomorphons';
import {getGPUPointHorizonDistanceLattice} from '../../src/gpu-terrain/point-horizon';
import {
  GEOMORPHON_FIXTURES,
  measureComparison,
  assertComparisons
} from './parity-wp5-round8-fixtures';

for (const fixture of GEOMORPHON_FIXTURES) {
  it(`exact geomorphon definition agrees with CPU convention: ${fixture.input.name}`, () => {
    const input = fixture.input;
    const errors: number[] = [];
    for (const settings of fixture.cases) {
      const actual = computeGeomorphons(
        input.elevation,
        undefined,
        input.width,
        input.height,
        getGPUGeomorphonsParameterValues(settings, makeGPURasterMetadata(input)),
        settings
      );
      for (const metric of ['forms', 'ternary'] as const) {
        errors.push(
          measureComparison(
            'REFERENCE AUDIT (not GPU)',
            `${input.name}/${settings.name}`,
            'GRASS with certified exact-ratio ties',
            metric,
            actual[metric],
            settings[metric]
          )
        );
      }
    }
    assertComparisons(errors);
  });
}

for (const fixture of [alps, canyon, ridge, plane]) {
  it(`every changed GRASS cell has an exact tie certificate: ${fixture.input.name}`, () => {
    for (const settings of fixture.cases) {
      for (const [certificates, forms, ternary] of [
        [settings.originalTieCertificates, settings.rawForms, settings.rawTernary],
        [settings.normalizedTieCertificates, settings.normalizedForms, settings.normalizedTernary]
      ] as const) {
        const changed = forms.flatMap((value, index) =>
          value !== settings.forms[index] || ternary[index] !== settings.ternary[index]
            ? [index]
            : []
        );
        expect(certificates.map(value => value.index)).toEqual(changed);
        for (const certificate of certificates) {
          expect(certificate.directions.length).toBeGreaterThan(0);
          for (const direction of certificate.directions) {
            // Fraction canonicalization makes this an exact integer-ratio identity, no epsilon.
            expect(direction.maximumRatio.replace('-', '')).toBe(
              direction.minimumRatio.replace('-', '')
            );
            expect(direction.grassDigit).not.toBe(direction.matchedDigit);
          }
        }
      }
    }
  });
}

it('small DEMs reproduce the cap, southwest sentinel, and sample-lattice differences', () => {
  const cap = horizon.cases.find(value => value.name === 'cap')!;
  const sentinel = horizon.cases.find(value => value.name === 'sentinel')!;
  const fractional = horizon.cases.find(value => value.name === 'point-fractional')!;
  const matched = horizon.cases.find(value => value.name === 'point-matched')!;
  const east = (4 * 9 + 4) * 4 + 1;
  const south = 7 * 9 * 4 + 2;
  const angle = (height: number, distance: number, curvature: number) =>
    (Math.atan(height / distance - curvature * distance) * 180) / Math.PI;
  expect(cap.horizon[east]).toBe(0);
  expect(cap.rawHorizon[east]).toBeCloseTo(
    angle(10, 4 * cap.cellSize, cap.curvatureCoefficient),
    4
  );
  expect(sentinel.rawHorizon[south]).toBe(0);
  expect(sentinel.horizon[south]).toBeCloseTo(angle(10, 2, sentinel.curvatureCoefficient), 4);
  const interpolatedHeight = (2 * fractional.cellSize - 1.25) / fractional.cellSize;
  expect(fractional.pointHorizon![1]).toBeCloseTo(
    angle(interpolatedHeight, 1.25, fractional.curvatureCoefficient),
    10
  );
  expect(fractional.pointDistance![1]).toBe(1.25);
  expect(Math.abs(fractional.pointHorizon![1] - fractional.horizon[east])).toBeGreaterThan(7);
  expect(matched.pointHorizon![1]).toBeCloseTo(matched.horizon[east], 4);
  expect(matched.pointDistance![1]).toBe(2);
  for (const settings of [fractional, matched]) {
    const lattice = getGPUPointHorizonDistanceLattice({
      ...settings.latticeOptions,
      cellSize: settings.cellSize,
      minimumDistance: settings.minimumDistance!,
      maximumDistance: settings.maximumDistance!
    });
    expect(
      Array.from({length: lattice.sampleCount}, (_, index) => lattice.getDistance(index))
    ).toEqual(settings.sampleDistances);
  }
  assertComparisons(
    horizon.cases.map(settings =>
      measureComparison(
        'REFERENCE AUDIT (not GPU)',
        settings.name,
        'GRASS matched cardinal samples',
        'horizon',
        settings.horizon,
        settings.cardinalDefinition,
        'D'
      )
    )
  );
});
