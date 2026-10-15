// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, readdirSync, statSync} from 'node:fs';
import {expect, it} from 'vitest';
import originalRidgeGeomorphons from '../oracles/fixtures/parity-wp5-round8-ridge-geomorphons.json';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {getGPUGeomorphonsParameterValues} from '../../src/gpu-terrain/geomorphons';
import {getGPUTerrainVectorRuggednessParameterValues} from '../../src/gpu-terrain/topographic-position';
import {computeGeomorphons} from './geomorphons/geomorphons-oracle';
import {computeVectorRuggedness} from './topographic-position/terrain-ruggedness-oracle';
import {computeTopographicPosition} from './topographic-position/topographic-position-oracle';
import {
  GEOMORPHON_FIXTURES,
  VISIBILITY_FIXTURES,
  HORIZON_FIXTURES,
  measureComparison,
  assertComparisons
} from './parity-wp5-round8-fixtures';
import {
  TPI_FIXTURES,
  VRM_FIXTURES,
  TEXTURE_FIXTURES,
  WEISS_FIXTURES
} from './parity-wp5-round8-composite-fixtures';

it('Round 8 reference provenance, input hashes, all output lengths and size budget', () => {
  const folder = new URL('../oracles/fixtures/', import.meta.url);
  for (const name of readdirSync(folder).filter(
    value => value.startsWith('parity-wp5-round8-') && value.endsWith('.json')
  )) {
    const path = new URL(name, folder);
    expect(statSync(path).size).toBeLessThan(1500000);
    const fixture = JSON.parse(readFileSync(path, 'utf8'));
    expect(fixture.generatorSha256).toMatch(/^[a-f0-9]{64}$/);
    const values = Float32Array.from(fixture.input.elevation);
    expect(createHash('sha256').update(new Uint8Array(values.buffer)).digest('hex')).toBe(
      fixture.input.elevationSha256
    );
    expect(fixture.versions.python).toBeTruthy();
    expect(fixture.cases.length).toBeGreaterThan(0);
  }
  for (const fixture of [
    ...TPI_FIXTURES,
    ...VRM_FIXTURES,
    ...WEISS_FIXTURES,
    ...TEXTURE_FIXTURES
  ]) {
    const count = fixture.input.width * fixture.input.height;
    for (const settings of fixture.cases)
      for (const values of Object.values(settings.expected)) expect(values.length % count).toBe(0);
  }
});
it('GRASS/GDAL retain every disagreement and exact cumulative reductions', () => {
  for (const fixture of VISIBILITY_FIXTURES)
    for (const settings of fixture.cases) {
      for (const [observerIndex, grass] of settings.grass.entries()) {
        const gdal = settings.gdal[observerIndex];
        expect(grass.flatMap((value, index) => (value !== gdal[index] ? [index] : []))).toEqual(
          settings.referenceDisagreements[observerIndex]
        );
        measureComparison(
          'REFERENCE AUDIT (not GPU)',
          settings.name,
          'GRASS r.viewshed versus GDAL gdal_viewshed',
          `observer-${observerIndex}`,
          grass,
          gdal
        );
      }
      for (const [reference, expected] of [
        [settings.grass, settings.grassCounts],
        [settings.gdal, settings.gdalCounts]
      ] as const) {
        expect(expected).toEqual(
          expected.map((_, index) =>
            reference.reduce((sum, view) => sum + Number(view[index] === 1), 0)
          )
        );
      }
    }
  expect(
    VISIBILITY_FIXTURES[2].cases.every(value =>
      value.referenceDisagreements.every(indices => indices.length === 0)
    )
  ).toBe(true);
});
it('CPU geomorphon convention matches adapted GRASS references exactly (not GPU parity)', () => {
  const failures: number[] = [];
  for (const fixture of GEOMORPHON_FIXTURES)
    for (const settings of fixture.cases) {
      const input = fixture.input;
      const grid = makeGPURasterMetadata({...input, cellSize: input.cellSize});
      const result = computeGeomorphons(
        input.elevation,
        undefined,
        input.width,
        input.height,
        getGPUGeomorphonsParameterValues(settings, grid),
        settings
      );
      if (input.name === 'ridge' && settings.name === 'geom_0')
        console.log(
          'GEOMORPHON_DIAGNOSTIC',
          JSON.stringify(
            result.forms.flatMap((value, index) =>
              value !== settings.forms[index]
                ? [
                    {
                      column: index % input.width,
                      row: Math.floor(index / input.width),
                      cpu: value,
                      grass: settings.forms[index]
                    }
                  ]
                : []
            )
          )
        );
      failures.push(
        measureComparison(
          'GPUGeomorphons CPU convention audit',
          `${input.name}/${settings.name}`,
          'GRASS + composite certified exact-tie adapter',
          'forms',
          result.forms,
          settings.forms
        )
      );
      failures.push(
        measureComparison(
          'GPUGeomorphons CPU convention audit',
          `${input.name}/${settings.name}`,
          'GRASS + composite certified exact-tie adapter',
          'ternary',
          result.ternary,
          settings.ternary
        )
      );
    }
  assertComparisons(failures);
});
it('retains the original Round 8 GRASS southern-border asymmetry diagnostic', () => {
  // Deep fixtures already adapt the finite-domain border and exact ties.
  // Keep this historical diagnostic anchored to the unadapted Round 8 output.
  const ridge = originalRidgeGeomorphons;
  const forms = ridge.cases[0].forms;
  const {width, height} = ridge.input;
  const reflected = forms.map(
    (_, index) => forms[(height - 1 - Math.floor(index / width)) * width + (index % width)]
  );
  // The input ridge is north/south invariant. This is an external-reference
  // self-consistency check, not permission to accept the GPU disagreements.
  expect(forms.some((value, index) => value !== reflected[index])).toBe(true);
});
it('SciPy TPI finite-window convention audit against existing CPU oracle (not GPU)', () => {
  const failures: number[] = [];
  // Fine preset only: the old CPU oracle loops over out-of-domain cells for radius 512.
  for (const fixture of TPI_FIXTURES.filter(value => value.cases[0].name.includes('-fine-'))) {
    const input = fixture.input;
    const settings = fixture.cases[0];
    const result = computeTopographicPosition(
      input.elevation,
      input.validity,
      input.width,
      input.height,
      settings.scales!,
      settings.quantum
    );
    for (const metric of [
      'topographicPositionIndex',
      'deviationFromMean',
      'maximumDeviation',
      'maximumDeviationRadius'
    ] as const)
      failures.push(
        measureComparison(
          'GPUTerrainTopographicPosition CPU convention audit',
          settings.name,
          'composite SciPy uniform_filter',
          metric,
          result[metric],
          settings.expected[metric],
          metric === 'maximumDeviationRadius' ? 'E' : 'R'
        )
      );
  }
  assertComparisons(failures);
});
it('SciPy Horn normals / VRM and row metric convention audit (not GPU)', () => {
  const failures: number[] = [];
  for (const fixture of VRM_FIXTURES)
    for (const settings of fixture.cases) {
      const input = fixture.input;
      const grid = makeGPURasterMetadata({...input, origin: [0, input.north]});
      const result = computeVectorRuggedness(
        input.elevation,
        input.validity,
        input.width,
        input.height,
        getGPUTerrainVectorRuggednessParameterValues({radius: settings.radius}, grid),
        settings
      );
      failures.push(
        measureComparison(
          'GPUTerrainVectorRuggedness CPU convention audit',
          settings.name,
          'composite SciPy sobel/uniform_filter',
          'vectorRuggedness',
          result.vectorRuggedness,
          settings.expected['vectorRuggedness'],
          'R'
        )
      );
      failures.push(
        measureComparison(
          'GPUTerrainVectorRuggedness CPU convention audit',
          settings.name,
          'composite SciPy minimum_filter',
          'validity',
          result.validity,
          settings.expected['outputValidity']
        )
      );
    }
  assertComparisons(failures);
});
it('GRASS horizon point output explicitly retains nonnegative cardinal restriction', () => {
  for (const fixture of HORIZON_FIXTURES) {
    const settings = fixture.cases[0];
    expect(settings.horizon.length).toBe(fixture.input.width * fixture.input.height * 4);
    expect(
      settings.point[0].horizons.map(value => value.azimuth).sort((first, second) => first - second)
    ).toEqual([0, 90, 180, 270]);
    expect(settings.horizon.every(value => value >= 0 && value <= 90)).toBe(true);
  }
});

it('GRASS unweighted VRM addon agrees with independent Horn normal reduction conventions (not GPU)', () => {
  const failures: number[] = [];
  for (const name of ['alps-dem', 'grand-canyon-dem', 'ridge']) {
    const raw = JSON.parse(
      readFileSync(
        new URL(
          `../oracles/fixtures/parity-wp5-round8-${name}-ruggedness-addon.json`,
          import.meta.url
        ),
        'utf8'
      )
    );
    const input = raw.input;
    const grid = makeGPURasterMetadata(input);
    for (const settings of raw.cases) {
      const result = computeVectorRuggedness(
        input.elevation,
        undefined,
        input.width,
        input.height,
        getGPUTerrainVectorRuggednessParameterValues({radius: settings.radius}, grid),
        {radius: settings.radius, borderMode: 'nodata'}
      );
      const expected = settings.expected.map((value: number | null) =>
        value === null ? NaN : value
      );
      failures.push(
        measureComparison(
          'GPUTerrainVectorRuggedness CPU convention audit',
          settings.name,
          'GRASS r.vector.ruggedness exponent=0',
          'vectorRuggedness',
          result.vectorRuggedness,
          expected,
          'R'
        )
      );
      failures.push(
        measureComparison(
          'GPUTerrainVectorRuggedness CPU convention audit',
          settings.name,
          'GRASS valid centre normal mask',
          'validity',
          result.validity,
          settings.validity
        )
      );
    }
  }
  assertComparisons(failures);
});
