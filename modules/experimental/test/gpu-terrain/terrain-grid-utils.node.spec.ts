// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize,
  getRasterGroundCellSizeWGSL,
  getRasterGroundScale,
  getRasterPixelDirection,
  writeRasterGridParameterValues
} from '../../src/analysis-core/raster-grid';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../src/gpu-raster';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {getOracleWebMercatorY} from './terrain-oracle-utils';

// Expected values: python3 -I, closed-form WGS84 (pyproj not installed). a = 6378137,
// f = 1 / 298.257223563, e2 = f (2 - f), w = 1 - e2 sin^2(lat), N = a / sqrt(w),
// M = a (1 - e2) / w^1.5. Geographic meters per degree: x = N cos(lat) PI / 180, y = M PI / 180.
// Web Mercator ground meters per EPSG:3857 meter at lat = atan(sinh(y / a)) for EPSG:3857 y in
// meters (normalized y n is the meter value (0.5 - n) 2 PI a): x = N cos(lat) / a, y = M cos(lat) / a.
const GEOGRAPHIC_METERS_PER_DEGREE: [number, [number, number]][] = [
  [0, [111319.49079327357, 110574.27582159435]],
  [45, [78846.83509397812, 111131.77741417562]],
  [60, [55800.001572436144, 111412.28745777943]]
];

it('getRasterGroundScale uses WGS84 radii of curvature at the row latitude', () => {
  for (const [latitude, [x, y]] of GEOGRAPHIC_METERS_PER_DEGREE) {
    const scale = getRasterGroundScale('geographic', latitude);
    expect(scale[0]).toBeCloseTo(x, 6);
    expect(scale[1]).toBeCloseTo(y, 6);
    // Southern rows mirror northern rows.
    expect(getRasterGroundScale('geographic', -latitude)[1]).toBeCloseTo(y, 6);
  }
  // Normalized y = 0.5 is the equator (x = N(0) / a = 1, y = 1 - e2); y = 0.25 is latitude
  // 66.51326044311186.
  const equator = getRasterGroundScale('web-mercator', getOracleWebMercatorY(0.5));
  expect(equator[0]).toBeCloseTo(1, 12);
  expect(equator[1]).toBeCloseTo(0.9933056200098586, 12);
  const north = getRasterGroundScale('web-mercator', getOracleWebMercatorY(0.25));
  expect(north[0]).toBeCloseTo(0.39966367720255397, 12);
  expect(north[1]).toBeCloseTo(0.39923631668162435, 12);
  expect(getRasterGroundScale('uniform', 45)).toEqual([1, 1]);
});

it('getRasterGroundCellSize evaluates the curved scale once at the row center', () => {
  // Two rows spanning latitude 46 -> 44: the second row center (coordinate 1.5) is at 44.5.
  const grid = makeGPURasterMetadata({width: 1, height: 2, cellSize: [0.5, 1], origin: [0, 46]});
  const [x, y] = getRasterGroundCellSize('geographic', getRasterGridGeoreference(grid), 1.5, 2);
  const [scaleX, scaleY] = getRasterGroundScale('geographic', 44.5);
  expect(x).toBeCloseTo(0.5 * scaleX, 9);
  expect(y).toBeCloseTo(scaleY, 9);
  // 'uniform' returns the grid cell size unchanged.
  expect(getRasterGroundCellSize('uniform', getRasterGridGeoreference(grid), 1.5, 2)).toEqual([
    0.5, 1
  ]);
});

it('the grid georeference is the cell size and the row-edge world y of the affine', () => {
  const area = makeGPURasterMetadata({width: 4, height: 3, cellSize: [2, 5], origin: [10, 100]});
  expect(area.affine).toEqual([2, 0, 10, 0, -5, 100]);
  expect(getRasterGridGeoreference(area)).toEqual({
    cellSize: [2, 5],
    northEdge: 100,
    southEdge: 85
  });
  // Point grids address pixel centers: the edges move by half a row.
  expect(getRasterGridGeoreference({...area, pixelInterpretation: 'point'})).toEqual({
    cellSize: [2, 5],
    northEdge: 102.5,
    southEdge: 87.5
  });
  // Rows advancing north (row 0 southernmost) give a south-to-north edge pair.
  const north = makeGPURasterMetadata({width: 1, height: 2, origin: [0, 0], rowDirection: 'north'});
  expect(getRasterGridGeoreference(north)).toEqual({cellSize: [1, 1], northEdge: 0, southEdge: 2});
  const target = new Float32Array(8).fill(-1);
  writeRasterGridParameterValues(target, area, {cellSizeIndex: 1, northEdgeIndex: 5});
  expect(Array.from(target)).toEqual([-1, 2, 5, -1, -1, 100, 85, -1]);
});

it('makeGPURasterBand wraps a per-pixel view on the grid', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 3, height: 2});
  const values = createTransientView(graph, 'directions', 'uint32', 6);
  const validity = createTransientView(graph, 'validity', 'uint32', 6);
  expect(makeGPURasterBand(values, grid, {id: 'flow', validity})).toEqual({
    id: 'flow',
    format: 'uint32',
    storage: {kind: 'buffer', values},
    validity
  });
  expect(makeGPURasterBand(values, grid).id).toBe('band');
  expect(() => makeGPURasterBand(createTransientView(graph, 'short', 'float32', 5), grid)).toThrow(
    /one value per grid pixel/
  );
  device.destroy();
});

it('getRasterGroundCellSizeWGSL emits the ellipsoidal row scale', () => {
  const slots = {cellSizeIndex: 0, northEdgeIndex: 2};
  const geographic = getRasterGroundCellSizeWGSL('geographic', slots);
  const mercator = getRasterGroundCellSizeWGSL('web-mercator', slots);
  for (const source of [geographic, mercator]) {
    expect(source).toContain('return cellSize * groundScale;');
    expect(source).toMatch(/0\.0066943\d*/);
  }
  expect(mercator).toContain('tanh(mercatorY)');
  expect(getRasterGroundCellSizeWGSL('uniform', slots)).not.toContain('groundScale');
});

it('getRasterPixelDirection keeps a 45 degree ground azimuth on a 60 N geographic grid', () => {
  // python3 -I with the WGS84 formulas above: 0.001 degree cells at 60 N are
  // [55.800001572436145, 111.41228745777943] m. The pixel step of the 45 degree (north-east)
  // ground direction (sin 45, -cos 45) on south-advancing rows is normalize(direction / cell) =
  // (0.8941256901499505, -0.44781608972196935); the pixel diagonal used before was a ground
  // azimuth of 26.60365263545291 degrees.
  const groundCell = getRasterGroundCellSize(
    'geographic',
    {cellSize: [0.001, 0.001], northEdge: 60, southEdge: 60},
    0.5,
    1
  );
  const groundDirection: [number, number] = [Math.SQRT1_2, -Math.SQRT1_2];
  const [x, y] = getRasterPixelDirection(groundDirection, groundCell);
  expect(x).toBeCloseTo(0.8941256901499505, 12);
  expect(y).toBeCloseTo(-0.44781608972196935, 12);
  const azimuth = (Math.atan2(x * groundCell[0], -y * groundCell[1]) * 180) / Math.PI;
  expect(azimuth).toBeCloseTo(45, 10);
  const pixelDiagonalAzimuth =
    (Math.atan2(groundDirection[0] * groundCell[0], -groundDirection[1] * groundCell[1]) * 180) /
    Math.PI;
  expect(pixelDiagonalAzimuth).toBeCloseTo(26.60365263545291, 9);
  // Square cells return the direction unchanged (bit-identical isotropic rays).
  expect(getRasterPixelDirection(groundDirection, [10, 10])).toEqual(groundDirection);
});
