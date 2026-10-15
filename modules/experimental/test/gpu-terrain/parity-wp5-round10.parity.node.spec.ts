// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeGeomorphons} from './geomorphons/geomorphons-oracle';
import {computeWeissLandforms} from './topographic-position/weiss-landforms-oracle';
import {getGPUGeomorphonsParameterValues} from '../../src/gpu-terrain/geomorphons';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {
  GEOMORPHON_FIXTURES,
  HORIZON_FIXTURES,
  measureComparison,
  assertComparisons
} from './parity-wp5-round8-fixtures';
import {WEISS_FIXTURES} from './parity-wp5-round8-composite-fixtures';

it('Weiss reference uses the complete classification mask for population and outputs', () => {
  const failures: number[] = [];
  for (const fixture of WEISS_FIXTURES) {
    const input = fixture.input;
    for (const settings of fixture.cases) {
      if (settings.cellSizeMode !== 'uniform') continue;
      const actual = computeWeissLandforms(
        input.elevation,
        input.validity,
        input.width,
        input.height,
        {
          scales: [settings.smallScale!, settings.largeScale!],
          cellSize: input.cellSize,
          standardization: settings.standardization
        }
      );
      for (const metric of ['landforms', 'standardizedPosition'] as const) {
        failures.push(
          measureComparison(
            'CPU convention audit (not GPU)',
            settings.name,
            'corrected SciPy/NumPy Weiss reference',
            metric,
            actual[metric],
            settings.expected[metric],
            metric === 'landforms' ? 'E' : 'R'
          )
        );
      }
      expect(actual.validity).toEqual(settings.expected['outputValidity']);
    }
  }
  assertComparisons(failures);
});

it('nodata padding removes GRASS ridge reflection artifact without adding terrain', () => {
  const fixture = GEOMORPHON_FIXTURES.find(value => value.input.name === 'ridge')!;
  const input = fixture.input;
  const failures: number[] = [];
  for (const settings of fixture.cases) {
    const actual = computeGeomorphons(
      input.elevation,
      undefined,
      input.width,
      input.height,
      getGPUGeomorphonsParameterValues(settings, makeGPURasterMetadata(input)),
      settings
    );
    for (const metric of ['forms', 'ternary'] as const)
      failures.push(
        measureComparison(
          'CPU convention audit (not GPU)',
          `ridge/${settings.name}`,
          'GRASS with nodata region padding',
          metric,
          actual[metric],
          settings[metric]
        )
      );
    expect(settings.forms).toEqual(
      settings.forms.map(
        (_, index) =>
          settings.forms[
            (input.height - 1 - Math.floor(index / input.width)) * input.width +
              (index % input.width)
          ]
      )
    );
  }
  assertComparisons(failures);
});

it('adapted GRASS horizon obeys the sixteen-cell finite-domain cardinal definition', () => {
  const failures: number[] = [];
  // Definition audit only: atan of the maximum apparent slope over the specified cardinal cells.
  // This checks external reference conventions; GPU comparisons remain in the browser spec.
  for (const fixture of HORIZON_FIXTURES) {
    const {width, height, cellSize, elevation} = fixture.input;
    const expected: number[] = [];
    for (let row = 0; row < height; row++)
      for (let column = 0; column < width; column++) {
        for (const [rowStep, columnStep] of [
          [-1, 0],
          [0, 1],
          [1, 0],
          [0, -1]
        ]) {
          let tangent = 0;
          for (let offset = 1; offset <= 16; offset++) {
            const sampleRow = row + rowStep * offset;
            const sampleColumn = column + columnStep * offset;
            if (sampleRow < 0 || sampleRow >= height || sampleColumn < 0 || sampleColumn >= width)
              break;
            const distance = offset * cellSize;
            tangent = Math.max(
              tangent,
              (elevation[sampleRow * width + sampleColumn] - elevation[row * width + column]) /
                distance -
                distance / (2 * 6371000)
            );
          }
          expected.push((Math.atan(tangent) * 180) / Math.PI);
        }
      }
    failures.push(
      measureComparison(
        'REFERENCE AUDIT (not GPU)',
        fixture.input.name,
        'cardinal apparent-slope definition',
        'horizonDegrees',
        fixture.cases[0].horizon,
        expected,
        'D'
      )
    );
  }
  assertComparisons(failures);
});

import pointReference from '../oracles/fixtures/parity-wp5-round10-point-horizon.json';
import {getGPUPointHorizonDistanceLattice} from '../../src/gpu-terrain/point-horizon';
import {computePointHorizonProfile} from './point-horizon/point-horizon-oracle';

it('SciPy point profiles match the public lattice convention at unchanged D bounds', () => {
  const failures: number[] = [];
  for (const fixture of HORIZON_FIXTURES) {
    const input = fixture.input;
    const settings = pointReference.cases.find(value => value.name === fixture.cases[0].name)!;
    expect(settings.elevationSha256).toBe(input.elevationSha256);
    const result = computePointHorizonProfile(
      {
        width: input.width,
        height: input.height,
        values: input.elevation,
        projection: 'planar',
        cellSize: [input.cellSize, input.cellSize],
        azimuthCount: 4,
        firstAzimuth: 0,
        azimuthSpan: 4,
        curvature: settings.curvatureCoefficient,
        lattice: getGPUPointHorizonDistanceLattice(settings)
      },
      [[16, 16, 0]]
    );
    failures.push(
      measureComparison(
        'CPU convention audit (not GPU)',
        settings.name,
        'composite SciPy interp1d',
        'nonnegativeHorizonDegrees',
        Array.from(result.skylineAngle, value => Math.max(0, value)),
        settings.horizon,
        'D'
      )
    );
    failures.push(
      measureComparison(
        'CPU convention audit (not GPU)',
        settings.name,
        'composite SciPy interp1d',
        'positiveHorizonDistanceMeters',
        Array.from(result.distance, (value, index) => (result.skylineAngle[index] > 0 ? value : 0)),
        settings.distance,
        'D',
        {absolute: 1e-3, relative: 2e-6}
      )
    );
  }
  assertComparisons(failures);
});
