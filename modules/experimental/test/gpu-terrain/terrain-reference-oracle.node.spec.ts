// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import derivativesFixture from '../oracles/fixtures/terrain-derivatives.json';
import ruggednessFixture from '../oracles/fixtures/terrain-ruggedness.json';
import {expect, it} from 'vitest';
import {computeTerrainDerivatives} from './terrain-analysis/terrain-analysis-oracle';
import {computeTerrainRuggedness} from './topographic-position/terrain-ruggedness-oracle';

/**
 * Cross-validates the float64 CPU oracles of the terrain contributors against `gdaldem` (real GDAL
 * library code) through the fixtures written by `test/oracles/terrain-derivatives.py`. The
 * fixtures record the exact convention mapping in `conventionMap`; the same fixtures drive the GPU
 * comparison in `terrain-reference-oracle.spec.ts`.
 */

type Nullable = (number | null)[];

type DerivativeTolerance = {absolute: number; minimumGradient?: number};

type DerivativeFixture = {
  cases: {
    name: string;
    width: number;
    height: number;
    cellSize: [number, number];
    elevation: Nullable;
    tolerances: Record<string, DerivativeTolerance>;
    variants: {
      name: string;
      output: 'slope' | 'aspect' | 'hillshade';
      ours: {
        slopeUnits?: 'degrees' | 'percent';
        zFactor: number;
        cellSizeOverride?: [number, number];
        azimuthDegrees?: number;
        altitudeDegrees?: number;
      };
      toleranceKey: string;
      values: Nullable;
    }[];
  }[];
};

type RuggednessFixture = {
  cases: {
    name: string;
    width: number;
    height: number;
    elevation: Nullable;
    tolerance: {absolute: number};
    variants: {
      name: string;
      output: 'tpi' | 'tri-riley' | 'tri-wilson' | 'roughness';
      edgeMode: 'nodata' | 'extrapolate';
      values: Nullable;
    }[];
  }[];
};

// Imported, not read with node:fs, so the browser-based headless project can load them.
const FIXTURES: Record<string, unknown> = {
  'terrain-derivatives.json': derivativesFixture,
  'terrain-ruggedness.json': ruggednessFixture
};

function loadFixture<T>(name: string): T {
  return FIXTURES[name] as T;
}

/** Fixture elevation as float64 with null (nodata) as NaN. */
function toFloat64(values: Nullable): Float64Array {
  return Float64Array.from(values, value => value ?? Number.NaN);
}

it('terrain derivative oracle agrees with gdaldem slope, aspect and hillshade', () => {
  const fixture = loadFixture<DerivativeFixture>('terrain-derivatives.json');
  expect(fixture.cases.length).toBeGreaterThan(0);
  for (const testCase of fixture.cases) {
    const elevation = toFloat64(testCase.elevation);
    for (const variant of testCase.variants) {
      const label = `${testCase.name}/${variant.name}`;
      const tolerance = testCase.tolerances[variant.toleranceKey];
      // GDAL aspect ignores cell size (see conventionMap.aspect), so the fixture overrides it.
      const cellSize = variant.ours.cellSizeOverride ?? testCase.cellSize;
      const settings = [
        cellSize[0],
        cellSize[1],
        variant.ours.zFactor,
        variant.ours.azimuthDegrees ?? 315,
        variant.ours.altitudeDegrees ?? 45,
        0,
        0,
        0
      ];
      const oracle = computeTerrainDerivatives(
        elevation,
        undefined,
        testCase.width,
        testCase.height,
        settings,
        {borderMode: 'nodata', slopeUnits: variant.ours.slopeUnits ?? 'degrees'}
      );
      // Gradient magnitude of the oracle at unit scale, used to skip ill-conditioned aspects.
      const slopeOracle = computeTerrainDerivatives(
        elevation,
        undefined,
        testCase.width,
        testCase.height,
        settings,
        {borderMode: 'nodata'}
      ).slope;
      let compared = 0;
      for (const [index, reference] of variant.values.entries()) {
        const where = `${label} pixel ${index}`;
        if (variant.output === 'aspect') {
          const flat = oracle.aspect[index] === -1;
          if (reference === null) {
            // GDAL: flat cells (and nodata) are nodata. Ours: -1 for flat, NaN for nodata.
            expect(flat || Number.isNaN(oracle.aspect[index]), where).toBe(true);
            continue;
          }
          expect(Number.isNaN(oracle.aspect[index]) || flat, `${where} should be defined`).toBe(
            false
          );
          if (Math.tan((slopeOracle[index] * Math.PI) / 180) < (tolerance.minimumGradient ?? 0)) {
            continue;
          }
          const difference = Math.abs(oracle.aspect[index] - reference);
          expect(Math.min(difference, 360 - difference), where).toBeLessThan(tolerance.absolute);
          compared++;
          continue;
        }
        const actual = oracle[variant.output][index];
        if (reference === null) {
          expect(Number.isNaN(actual), `${where} should be nodata`).toBe(true);
          continue;
        }
        expect(Number.isNaN(actual), `${where} should be defined`).toBe(false);
        expect(Math.abs(actual - reference), where).toBeLessThan(tolerance.absolute);
        compared++;
      }
      expect(compared, `${label} compared pixels`).toBeGreaterThan(30);
    }
  }
});

it('terrain ruggedness oracle agrees with gdaldem TPI, TRI and roughness', () => {
  const fixture = loadFixture<RuggednessFixture>('terrain-ruggedness.json');
  expect(fixture.cases.length).toBeGreaterThan(0);
  for (const testCase of fixture.cases) {
    const elevation = toFloat64(testCase.elevation);
    for (const edgeMode of ['nodata', 'extrapolate'] as const) {
      const oracle = computeTerrainRuggedness(
        elevation,
        undefined,
        testCase.width,
        testCase.height,
        {edgeMode}
      );
      const outputs = {
        tpi: oracle.topographicPositionIndex,
        'tri-riley': oracle.rileyRuggedness,
        'tri-wilson': oracle.wilsonRuggedness,
        roughness: oracle.roughness
      };
      for (const variant of testCase.variants.filter(entry => entry.edgeMode === edgeMode)) {
        const label = `${testCase.name}/${variant.name}`;
        let compared = 0;
        for (const [index, reference] of variant.values.entries()) {
          const actual = outputs[variant.output][index];
          if (reference === null) {
            expect(Number.isNaN(actual), `${label} pixel ${index} should be nodata`).toBe(true);
            continue;
          }
          expect(Number.isNaN(actual), `${label} pixel ${index} should be defined`).toBe(false);
          expect(Math.abs(actual - reference), `${label} pixel ${index}`).toBeLessThan(
            testCase.tolerance.absolute
          );
          compared++;
        }
        expect(compared, `${label} compared pixels`).toBeGreaterThan(30);
      }
    }
  }
});
