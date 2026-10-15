// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {getGPUTerrainLineOfSightParameterValues} from '../../src/gpu-terrain/terrain-analysis';
import {getGPUPointHorizonDistanceLattice} from '../../src/gpu-terrain/point-horizon';
import {computeTerrainSightLine} from './terrain-analysis/terrain-analysis-oracle';
import {marchOracleRay} from './point-horizon/point-horizon-oracle';
import {MATCHED_VISIBILITY_FIXTURES} from './parity-wp5-deep-visibility-fixtures';
import {SHADOW_FIXTURES, measureComparison, assertComparisons} from './parity-wp5-round8-fixtures';
import saddle from '../oracles/fixtures/parity-wp5-deep-saddle-visibility.json';

for (const fixture of MATCHED_VISIBILITY_FIXTURES) {
  it(`external radial reference agrees with coarse CPU convention: ${fixture.input.name}`, () => {
    const {width, height, cellSize, elevation} = fixture.input;
    const grid = makeGPURasterMetadata({width, height, cellSize});
    const errors: number[] = [];
    for (const settings of fixture.cases) {
      const parameters = getGPUTerrainLineOfSightParameterValues(settings, grid);
      for (const [observerIndex, observer] of settings.observers.entries()) {
        const actual = elevation.map(
          (_, index) =>
            computeTerrainSightLine(
              elevation,
              undefined,
              width,
              height,
              observer,
              [index % width, Math.floor(index / width)],
              parameters
            ).code
        );
        errors.push(
          measureComparison(
            'CPU convention audit (not GPU)',
            `${settings.name}/coarse/observer-${observerIndex}`,
            'SciPy + GDAL radial Wang recurrence',
            'visibility',
            actual,
            settings.coarse[observerIndex]
          )
        );
      }
      expect(settings.coarseCounts).toEqual(
        elevation.map((_, index) =>
          settings.coarse.reduce((count, codes) => count + Number(codes[index] === 1), 0)
        )
      );
    }
    assertComparisons(errors);
  });

  it(`external radial reference agrees with point CPU convention: ${fixture.input.name}`, () => {
    const {width, height, cellSize, elevation} = fixture.input;
    const lattice = getGPUPointHorizonDistanceLattice({
      cellSize,
      minimumDistance: 1,
      maximumDistance: Math.ceil(Math.hypot(width, height) * cellSize)
    });
    const errors: number[] = [];
    for (const settings of fixture.cases) {
      for (const [observerIndex, observer] of settings.observers.entries()) {
        const eye = elevation[observer[1] * width + observer[0]] + settings.observerHeight;
        const actual = elevation.map((targetGround, index) => {
          const deltaColumn = (index % width) - observer[0];
          const deltaRow = Math.floor(index / width) - observer[1];
          const distance = Math.hypot(deltaColumn, deltaRow) * cellSize;
          if (settings.maximumDistance > 0 && distance > settings.maximumDistance) return 2;
          if (distance === 0) return 1;
          const curvature = Math.fround(settings.curvatureCoefficient);
          const ray = marchOracleRay(
            {
              width,
              height,
              values: elevation,
              projection: 'planar',
              azimuthCount: 1,
              firstAzimuth: 0,
              azimuthSpan: 1,
              lattice,
              curvature,
              cellSize: [cellSize, cellSize],
              maximumDistance: settings.maximumDistance
            },
            observer[0],
            observer[1],
            eye,
            (deltaColumn * cellSize) / distance,
            (-deltaRow * cellSize) / distance,
            distance
          );
          const targetSlope =
            (targetGround + settings.targetHeight - eye) / distance - curvature * distance;
          return Number(targetSlope >= ray.tQ);
        });
        errors.push(
          measureComparison(
            'CPU convention audit (not GPU)',
            `${settings.name}/point/observer-${observerIndex}`,
            'SciPy + GDAL radial Wang recurrence',
            'visibility',
            actual,
            settings.point[observerIndex]
          )
        );
      }
    }
    assertComparisons(errors);
  }, 120000);
}

it('the minimal saddle distinguishes finite sampling from native reference conventions', () => {
  const settings = saddle.cases[0];
  // Adjacent diagonal: n=1, so coarse has no blocker. At d=4 m on the point
  // lattice, t=d/(8*sqrt(2)); z=16*t*(1-t)=3.65685424949 m > eye=1 m.
  const targetIndex = 5 * 9 + 5;
  const fraction = 4 / (8 * Math.SQRT2);
  const sampledHeight = 16 * fraction * (1 - fraction);
  expect(sampledHeight).toBeCloseTo(3.6568542494923806, 12);
  expect((sampledHeight - 1) / 4).toBeGreaterThan(-1 / (8 * Math.SQRT2));
  expect(settings.coarse[0][targetIndex]).toBe(1);
  expect(settings.point[0][targetIndex]).toBe(0);
  expect(settings.grass[0][targetIndex]).toBe(1);
  expect(settings.gdal[0][targetIndex]).toBe(1);
  // All k/6 samples along (2,4)->(8,1) have z=0. Wang's off-ray
  // propagated planes nevertheless hide it; the true bilinear ray is clear.
  expect(settings.coarse[1][17]).toBe(1);
  expect(settings.point[1][17]).toBe(1);
  expect(settings.gdal[1][17]).toBe(0);
  // GRASS's angular active-cell interpolation is also a distinct sample set.
  expect(settings.coarse[0][14]).toBe(1);
  expect(settings.point[0][14]).toBe(1);
  expect(settings.grass[0][14]).toBe(0);
  // GRASS interpolates the angular gradient between the active cell's entry
  // and centre events. For cell (4,3), heights at those events are 2 and 0.
  const eventWeight = (Math.PI / 2 - Math.atan(3)) / (Math.PI / 4);
  const interpolatedAngle =
    Math.atan(-1 / 8) + (Math.atan(1 / Math.sqrt(32)) - Math.atan(-1 / 8)) * eventWeight;
  expect((interpolatedAngle * 180) / Math.PI).toBeCloseTo(-0.09925079462917862, 10);
  expect(interpolatedAngle).toBeGreaterThan(Math.atan(-1 / Math.sqrt(640)));
  // The actual bilinear terrain minus LOS in that first patch is
  // -24*t^2 + 9*t - 1, whose maximum is -5/32: no intersection.
  expect(-24 * (3 / 16) ** 2 + 9 * (3 / 16) - 1).toBe(-5 / 32);
});

it('cardinal hard shadows match the independent r.sunmask finite-domain definition', () => {
  const errors: number[] = [];
  const fixtures = [...SHADOW_FIXTURES, {input: saddle.input, cases: saddle.shadows}];
  for (const fixture of fixtures) {
    const {width, height, cellSize, elevation} = fixture.input;
    for (const settings of fixture.cases) {
      const direction = [
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0]
      ][settings.azimuthDegrees / 90];
      const sunSlope = Math.tan((settings.altitudeDegrees * Math.PI) / 180);
      const actual = elevation.map((ground, index) => {
        for (let step = 1; step < Math.max(width, height); step++) {
          const column = (index % width) + direction[0] * step;
          const row = Math.floor(index / width) + direction[1] * step;
          if (column < 0 || column >= width || row < 0 || row >= height) break;
          if ((elevation[row * width + column] - ground) / (step * cellSize) > sunSlope) return 0;
        }
        return 1;
      });
      errors.push(
        measureComparison(
          'CPU definition audit (not GPU)',
          `${fixture.input.name}/${settings.azimuthDegrees}`,
          'GRASS r.sunmask -z',
          'hardSunVisibility',
          actual,
          settings.sunVisibility
        )
      );
    }
  }
  assertComparisons(errors);
});
