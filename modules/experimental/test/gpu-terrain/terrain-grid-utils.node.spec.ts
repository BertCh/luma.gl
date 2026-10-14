// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getTerrainGroundCellSize,
  getTerrainGroundCellSizeWGSL,
  getTerrainGroundScale,
  getTerrainPixelDirection
} from '../../src/gpu-terrain/terrain-grid-utils';
import {getRasterGroundCellSize} from '../../src/gpu-raster/cost-distance/raster-grid-utils';

// Expected values: python3 -I, closed-form WGS84 (pyproj not installed). a = 6378137,
// f = 1 / 298.257223563, e2 = f (2 - f), w = 1 - e2 sin^2(lat), N = a / sqrt(w),
// M = a (1 - e2) / w^1.5. Geographic meters per degree: x = N cos(lat) PI / 180, y = M PI / 180.
// Web Mercator ground meters per EPSG:3857 meter at lat = atan(sinh(PI (1 - 2y))):
// x = N cos(lat) / a, y = M cos(lat) / a.
const GEOGRAPHIC_METERS_PER_DEGREE: [number, [number, number]][] = [
  [0, [111319.49079327357, 110574.27582159435]],
  [45, [78846.83509397812, 111131.77741417562]],
  [60, [55800.001572436144, 111412.28745777943]]
];

it('getTerrainGroundScale uses WGS84 radii of curvature at the row latitude', () => {
  for (const [latitude, [x, y]] of GEOGRAPHIC_METERS_PER_DEGREE) {
    const scale = getTerrainGroundScale('geographic', latitude);
    expect(scale[0]).toBeCloseTo(x, 6);
    expect(scale[1]).toBeCloseTo(y, 6);
    // Southern rows mirror northern rows.
    expect(getTerrainGroundScale('geographic', -latitude)[1]).toBeCloseTo(y, 6);
  }
  // y = 0.5 is the equator (x = N(0) / a = 1, y = 1 - e2); y = 0.25 is latitude 66.51326044311186.
  const equator = getTerrainGroundScale('web-mercator', 0.5);
  expect(equator[0]).toBeCloseTo(1, 12);
  expect(equator[1]).toBeCloseTo(0.9933056200098586, 12);
  const north = getTerrainGroundScale('web-mercator', 0.25);
  expect(north[0]).toBeCloseTo(0.39966367720255397, 12);
  expect(north[1]).toBeCloseTo(0.39923631668162435, 12);
  expect(getTerrainGroundScale('uniform', 45)).toEqual([1, 1]);
});

it('getTerrainGroundCellSize evaluates the curved scale once at the row center', () => {
  // Two rows spanning latitude 46 -> 44: the second row center is at 44.5 degrees.
  const [x, y] = getTerrainGroundCellSize('geographic', [0.5, 1], 46, 44, 1, 2);
  const [scaleX, scaleY] = getTerrainGroundScale('geographic', 44.5);
  expect(x).toBeCloseTo(0.5 * scaleX, 9);
  expect(y).toBeCloseTo(scaleY, 9);
});

it('raster grid cell sizes share the terrain WGS84 model', () => {
  const settings = {cellSize: [0.001, 0.001] as const, northEdge: 61, southEdge: 59};
  for (const mode of ['geographic', 'web-mercator', 'uniform'] as const) {
    const modeSettings =
      mode === 'web-mercator'
        ? {cellSize: [10, 10] as const, northEdge: 0.2, southEdge: 0.3}
        : settings;
    const raster = getRasterGroundCellSize(mode, 3.5, 8, modeSettings);
    const terrain = getTerrainGroundCellSize(
      mode,
      modeSettings.cellSize,
      modeSettings.northEdge,
      modeSettings.southEdge,
      3,
      8
    );
    expect(raster[0]).toBeCloseTo(terrain[0], 12);
    expect(raster[1]).toBeCloseTo(terrain[1], 12);
  }
});

it('getTerrainGroundCellSizeWGSL emits the ellipsoidal row scale', () => {
  const slots = {cellSizeIndex: 0, northEdgeIndex: 2};
  const geographic = getTerrainGroundCellSizeWGSL('geographic', slots);
  const mercator = getTerrainGroundCellSizeWGSL('web-mercator', slots);
  for (const source of [geographic, mercator]) {
    expect(source).toContain('return cellSize * groundScale;');
    expect(source).toMatch(/0\.0066943\d*/);
  }
  expect(mercator).toContain('tanh(mercatorY)');
  expect(getTerrainGroundCellSizeWGSL('uniform', slots)).not.toContain('groundScale');
});

it('getTerrainPixelDirection keeps a 45 degree ground azimuth on a 60 N geographic grid', () => {
  // python3 -I with the WGS84 formulas above: 0.001 degree cells at 60 N are
  // [55.800001572436145, 111.41228745777943] m. The pixel step of the 45 degree (north-east)
  // ground direction (sin 45, -cos 45) on south-advancing rows is normalize(direction / cell) =
  // (0.8941256901499505, -0.44781608972196935); the pixel diagonal used before was a ground
  // azimuth of 26.60365263545291 degrees.
  const groundCell = getTerrainGroundCellSize('geographic', [0.001, 0.001], 60, 60, 0, 1);
  const groundDirection: [number, number] = [Math.SQRT1_2, -Math.SQRT1_2];
  const [x, y] = getTerrainPixelDirection(groundDirection, groundCell);
  expect(x).toBeCloseTo(0.8941256901499505, 12);
  expect(y).toBeCloseTo(-0.44781608972196935, 12);
  const azimuth = (Math.atan2(x * groundCell[0], -y * groundCell[1]) * 180) / Math.PI;
  expect(azimuth).toBeCloseTo(45, 10);
  const pixelDiagonalAzimuth =
    (Math.atan2(groundDirection[0] * groundCell[0], -groundDirection[1] * groundCell[1]) * 180) /
    Math.PI;
  expect(pixelDiagonalAzimuth).toBeCloseTo(26.60365263545291, 9);
  // Square cells return the direction unchanged (bit-identical isotropic rays).
  expect(getTerrainPixelDirection(groundDirection, [10, 10])).toEqual(groundDirection);
});
