// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUBoundsFilterParameterValues,
  GPUBoundsFilter,
  type GPUBoundsFilterMode
} from '../../../src/gpu-spatial-analysis/bounds-filter/index';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';

// Pinned from geopandas 1.2.0 / shapely 2.1.2 (scratchpad build/D/gen_bounds.py): 42 features
// (points, lines, rectangles on a 0.5 grid, an empty polygon at row 40 and box(0, 0, 16, 16) at
// row 41). Expected rows are GeoSeries.envelope.intersects(box) and .contains(box); intersects also
// equals GeoSeries.cx[xmin:xmax, ymin:ymax] for every box here. 'within' is the closed comparison of
// GeoSeries.bounds (withinClosed): OGC envelope.within(box) rejects features whose bounds only touch
// the box border, so it is a strict subset of the closed result.
const BOUNDS = [
  15.0, 10.0, 17.5, 13.5, 9.0, 12.0, 12.0, 12.5, 0.5, 4.5, 1.5, 7.5, 14.5, 0.0, 16.0, 3.0, 2.0,
  12.5, 2.0, 14.0, 13.0, 4.5, 14.0, 5.5, 11.5, 4.0, 15.0, 5.5, 7.5, 8.0, 9.5, 10.0, 8.0, 15.5, 11.0,
  18.5, 11.0, 9.5, 12.0, 13.0, 7.0, 3.0, 10.0, 3.5, 13.5, 9.5, 13.5, 9.5, 7.0, 0.5, 7.5, 2.5, 15.5,
  7.0, 18.5, 10.5, 13.0, 10.0, 14.5, 12.0, 4.0, 7.5, 5.5, 8.0, 15.5, 0.0, 15.5, 0.5, 15.5, 11.0,
  19.0, 11.5, 11.5, 5.5, 13.0, 5.5, 9.5, 13.0, 12.0, 13.5, 8.5, 4.0, 12.0, 7.5, 2.5, 8.0, 6.0, 11.0,
  11.0, 10.0, 11.0, 12.5, 7.5, 1.0, 8.0, 3.0, 11.5, 8.0, 13.5, 11.0, 10.5, 5.5, 13.0, 7.5, 1.5, 0.5,
  4.0, 2.0, 9.5, 5.0, 10.0, 5.5, 6.0, 13.0, 7.5, 14.5, 9.0, 15.5, 10.5, 17.5, 7.0, 9.5, 8.5, 12.0,
  8.5, 10.5, 12.0, 11.0, 8.5, 7.0, 9.5, 7.5, 0.5, 6.0, 3.5, 6.0, 6.0, 15.0, 9.5, 15.5, 0.0, 10.5,
  0.0, 11.5, 7.5, 13.5, 7.5, 16.0, 8.5, 2.0, 11.5, 5.0, 7.5, 15.0, 9.5, 18.5, 15.0, 9.0, 16.0, 9.5,
  0, 0, 0, 0, 0.0, 0.0, 16.0, 16.0
];
const CASES: {
  box: number[];
  intersects: number[];
  within: number[];
  withinClosed: number[];
  contains: number[];
}[] = [
  {
    box: [4, 4, 10, 10],
    intersects: [7, 15, 20, 21, 27, 30, 32, 37, 41],
    within: [7, 15, 27, 32],
    withinClosed: [7, 15, 27, 32],
    contains: [41]
  },
  {
    box: [0, 0, 16, 16],
    intersects: [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
      26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41
    ],
    within: [
      1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
      31, 32, 33, 34, 37, 39, 41
    ],
    withinClosed: [
      1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
      30, 31, 32, 33, 34, 35, 36, 37, 39, 41
    ],
    contains: [41]
  },
  {box: [6, 6, 8, 8], intersects: [7, 21, 41], within: [], withinClosed: [], contains: [41]},
  {box: [20, 20, 30, 30], intersects: [], within: [], withinClosed: [], contains: []}
];
const EMPTY_ROW = 40;

function createBoundsValues(): Float32Array {
  const values = new Float32Array(BOUNDS);
  // GeoPandas reports NaN bounds for empty geometries.
  values.fill(NaN, EMPTY_ROW * 4, EMPTY_ROW * 4 + 4);
  return values;
}

function createFixture(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  mode: GPUBoundsFilterMode,
  capacity: number
) {
  return createGeometryFixture(device, {
    inputs: {bounds: {values: createBoundsValues(), format: 'float32x4'}},
    outputs: {
      ids: {format: 'uint32', length: capacity},
      mask: {format: 'uint32', length: 42},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      requiredCount: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUBoundsFilter({
        bounds: inputs['bounds'] as never,
        box: parameters,
        mode,
        mask: outputs['mask'] as never,
        output: {
          ids: outputs['ids'] as never,
          count: outputs['count'] as never,
          overflow: outputs['overflow'] as never,
          requiredCount: outputs['requiredCount'] as never
        }
      })
  });
}

for (const mode of ['intersects', 'within', 'contains'] as const) {
  it(`GPUBoundsFilter ${mode} matches geopandas envelope.${mode}(box) and moves per frame`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const fixture = createFixture(device, mode, 64);
    for (const testCase of CASES) {
      const [minX, minY, maxX, maxY] = testCase.box;
      const result = await fixture.run(getGPUBoundsFilterParameterValues({minX, minY, maxX, maxY}));
      const expected = mode === 'within' ? testCase.withinClosed : testCase[mode];
      for (const row of testCase.within) {
        expect(testCase.withinClosed).toContain(row);
      }
      expect(result['overflow'][0]).toBe(0);
      expect(result['count'][0]).toBe(expected.length);
      expect(result['requiredCount'][0]).toBe(expected.length);
      expect(result['ids'].slice(0, expected.length)).toEqual(expected);
      expect(result['mask'].reduce((sum, value) => sum + value, 0)).toBe(expected.length);
      expect(result['mask'][EMPTY_ROW]).toBe(0);
    }
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}

it('GPUBoundsFilter clamps to a smaller capacity and reports overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, 'intersects', 5);
  const result = await fixture.run(
    getGPUBoundsFilterParameterValues({minX: 0, minY: 0, maxX: 16, maxY: 16})
  );
  expect(result['requiredCount'][0]).toBe(41);
  expect(result['count'][0]).toBe(5);
  expect(result['overflow'][0]).toBe(1);
  expect(result['ids']).toEqual(CASES[1].intersects.slice(0, 5));
  fixture.destroy();
});

it('getGPUBoundsFilterParameterValues rejects inverted and non-finite boxes', () => {
  expect(() => getGPUBoundsFilterParameterValues({minX: 2, minY: 0, maxX: 1, maxY: 1})).toThrow();
  expect(() =>
    getGPUBoundsFilterParameterValues({minX: 0, minY: 0, maxX: Infinity, maxY: 1})
  ).toThrow();
});
