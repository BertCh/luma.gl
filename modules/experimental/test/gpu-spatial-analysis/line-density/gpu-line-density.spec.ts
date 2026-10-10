// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPULineDensityParameterValues,
  GPULineDensity
} from '../../../src/gpu-spatial-analysis/line-density';
import {
  createGeometryFixture,
  createRandom,
  expectClose
} from '../outline-geometry/geometry-fixture';
import {computeLineLengthsPerCell, type Point} from './line-density-oracle';

function createPaths(seed: number, pathCount: number, extent: number): Point[][] {
  const random = createRandom(seed);
  return Array.from({length: pathCount}, (_, pathIndex) => {
    const length = pathIndex % 11 === 4 ? 1 : 2 + Math.floor(random() * 6);
    let x = (random() * 1.4 - 0.2) * extent;
    let y = (random() * 1.4 - 0.2) * extent;
    return Array.from({length}, () => {
      // Some axis-aligned and integer-aligned steps exercise exact grid-line crossings.
      const kind = random();
      if (kind < 0.15) {
        x = Math.round(x + (random() - 0.5) * extent * 0.4);
      } else if (kind < 0.3) {
        y = Math.round(y + (random() - 0.5) * extent * 0.4);
      } else {
        x += (random() - 0.5) * extent * 0.5;
        y += (random() - 0.5) * extent * 0.5;
      }
      return [x, y] as Point;
    });
  });
}

function flatten(paths: Point[][]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const path of paths) {
    for (const point of path) {
      positions.push(point[0], point[1]);
    }
    offsets.push(positions.length / 2);
  }
  return {positions: new Float32Array(positions), offsets: new Uint32Array(offsets)};
}

function createFixture(
  device: Awaited<ReturnType<typeof getWebGPUTestDevice>>,
  paths: Point[][],
  options: {columns: number; rows: number; spherical?: boolean; maximumRecords?: number}
) {
  const flat = flatten(paths);
  const cellCount = options.columns * options.rows;
  return createGeometryFixture(device as NonNullable<typeof device>, {
    inputs: {
      positions: {values: flat.positions, format: 'float32x2'},
      pathOffsets: {values: flat.offsets, format: 'uint32'}
    },
    outputs: {
      lengths: {format: 'float32', length: cellCount},
      densities: {format: 'float32', length: cellCount},
      overflow: {format: 'uint32', length: 1},
      totalRecords: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPULineDensity({
        positions: inputs['positions'] as never,
        pathOffsets: inputs['pathOffsets'] as never,
        columns: options.columns,
        rows: options.rows,
        spatialContext: options.spherical
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'}
          : {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        maximumRecords: options.maximumRecords,
        parameters,
        output: {
          lengths: outputs['lengths'] as never,
          densities: outputs['densities'] as never,
          overflow: outputs['overflow'] as never,
          totalRecords: outputs['totalRecords'] as never
        }
      })
  });
}

it('GPULineDensity sums exact segment-cell lengths like the f64 oracle and pans per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const columns = 12;
  const rows = 9;
  const paths = createPaths(3, 300, 100);
  const rounded = paths.map(path =>
    path.map(([x, y]) => [Math.fround(x), Math.fround(y)] as Point)
  );
  const fixture = createFixture(device, paths, {columns, rows});
  let maximumError = 0;
  for (const grid of [
    {minX: 0, minY: 0, width: 100 / columns, height: 100 / rows},
    {minX: -20, minY: 10, width: 7.5, height: 11}
  ]) {
    const result = await fixture.run(
      getGPULineDensityParameterValues({
        minX: grid.minX,
        minY: grid.minY,
        cellWidth: grid.width,
        cellHeight: grid.height
      })
    );
    const expected = computeLineLengthsPerCell(rounded, {...grid, columns, rows});
    expect(result['overflow'][0]).toBe(0);
    expect(result['totalRecords'][0]).toBeGreaterThanOrEqual(expected.pieces);
    expect(Math.max(...result['lengths'])).toBeGreaterThan(1);
    for (let cell = 0; cell < columns * rows; cell++) {
      expectClose(result['lengths'][cell], expected.lengths[cell], 2e-4, 2e-4, `cell ${cell}`);
      maximumError = Math.max(
        maximumError,
        Math.abs(result['lengths'][cell] - expected.lengths[cell])
      );
      expectClose(
        result['densities'][cell],
        expected.lengths[cell] / (grid.width * grid.height),
        3e-4,
        1e-6,
        `density ${cell}`
      );
    }
  }
  console.log(`GPULineDensity max absolute length error ${maximumError.toExponential(2)}`);
  expect(fixture.getCompileCount()).toBe(0);
  // Determinism: a second encoding is bitwise identical.
  const first = await fixture.run();
  const second = await fixture.run();
  expect(second['lengths']).toEqual(first['lengths']);
  fixture.destroy();
});

it('GPULineDensity conserves the clipped length of a diagonal and flags record overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // One diagonal across a 10 x 10 grid passes through 19 cells (through lattice corners it counts 10).
  const paths: Point[][] = [
    [
      [-5, -5],
      [15, 15]
    ],
    [
      [0.5, 0.5],
      [9.5, 0.5],
      [9.5, 9.5]
    ]
  ];
  const fixture = createFixture(device, paths, {columns: 10, rows: 10});
  const parameters = getGPULineDensityParameterValues({
    minX: 0,
    minY: 0,
    cellWidth: 1,
    cellHeight: 1
  });
  const result = await fixture.run(parameters);
  const total = result['lengths'].reduce((sum, value) => sum + value, 0);
  expect(total).toBeCloseTo(10 * Math.SQRT2 + 9 + 9, 3);
  expect(result['overflow'][0]).toBe(0);
  fixture.destroy();
  const small = createFixture(device, paths, {columns: 10, rows: 10, maximumRecords: 8});
  const clipped = await small.run(parameters);
  expect(clipped['overflow'][0]).toBe(1);
  expect(clipped['totalRecords'][0]).toBeGreaterThan(8);
  expect(clipped['lengths'].reduce((sum, value) => sum + value, 0)).toBeLessThan(total);
  small.destroy();
});

it('GPULineDensity measures spherical pieces as great-circle meters', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const columns = 6;
  const rows = 4;
  const paths: Point[][] = [
    [
      [10.1, 50.1],
      [11.9, 50.9],
      [11.2, 52.5]
    ],
    [
      [10, 51.5],
      [12, 51.5]
    ]
  ];
  const fixture = createFixture(device, paths, {columns, rows, spherical: true});
  const grid = {minX: 10, minY: 50, width: 0.5, height: 0.5};
  const result = await fixture.run(
    getGPULineDensityParameterValues({
      minX: grid.minX,
      minY: grid.minY,
      cellWidth: grid.width,
      cellHeight: grid.height
    })
  );
  const rounded = paths.map(path =>
    path.map(([x, y]) => [Math.fround(x), Math.fround(y)] as Point)
  );
  const expected = computeLineLengthsPerCell(rounded, {...grid, columns, rows}, true);
  for (let cell = 0; cell < columns * rows; cell++) {
    expectClose(result['lengths'][cell], expected.lengths[cell], 1e-3, 2, `cell ${cell}`);
  }
  expect(Math.max(...result['lengths'])).toBeGreaterThan(1000);
  // Spherical cell area: 0.5 degree by 0.5 degree at about 50 degrees north is about 1.8e9 m^2.
  const area =
    6371008.8 ** 2 *
    ((0.5 * Math.PI) / 180) *
    (Math.sin((50.5 * Math.PI) / 180) - Math.sin((50 * Math.PI) / 180));
  expectClose(result['densities'][0], result['lengths'][0] / area, 2e-3, 1e-12, 'density');
  fixture.destroy();
});
