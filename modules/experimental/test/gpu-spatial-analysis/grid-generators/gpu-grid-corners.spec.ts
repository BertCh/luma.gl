// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUGridCornerCount,
  getGPUGridGeneratorParameterValues,
  GPUGridGenerator
} from '../../../src/gpu-spatial-analysis/grid-generators';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {generateGridCorners} from './grid-generator-oracle';

for (const gridType of ['square', 'hex', 'hexFlat'] as const) {
  it(`GPUGridGenerator unique ${gridType} corners match the polygon vertex union across frames`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const publicType = gridType === 'square' ? 'square' : 'hex';
    for (const [columns, rows] of [
      [1, 1],
      [1, 4],
      [4, 1],
      [2, 2],
      [5, 4]
    ]) {
      const count = getGPUGridCornerCount(publicType, columns, rows);
      const fixture = createGeometryFixture(device, {
        inputs: {},
        outputs: {corners: {format: 'float32x2', length: count}},
        create: ({outputs, parameters}) =>
          new GPUGridGenerator({
            gridType: publicType,
            hexOrientation: gridType === 'hexFlat' ? 'flat' : undefined,
            columns,
            rows,
            parameters,
            output: {corners: outputs['corners'] as never}
          })
      });
      try {
        for (const [minX, minY, width, height] of [
          [-3, 7, 2, 3],
          [10.25, -2.5, 0.75, 1.25]
        ]) {
          const result = await fixture.run(
            getGPUGridGeneratorParameterValues({
              minX,
              minY,
              cellWidth: width,
              cellHeight: height
            })
          );
          const expected = generateGridCorners(
            publicType,
            columns,
            rows,
            minX,
            minY,
            width,
            height,
            gridType === 'hexFlat'
          );
          expect(expected.length).toBe(count);
          const matched = new Set<number>();
          for (let corner = 0; corner < count; corner++) {
            const match = expected.findIndex(
              point =>
                Math.hypot(
                  point[0] - result['corners'][2 * corner],
                  point[1] - result['corners'][2 * corner + 1]
                ) < 1e-5
            );
            expect(match, `corner ${corner}, ${columns} x ${rows}`).toBeGreaterThanOrEqual(0);
            expect(matched.has(match), 'no repeated shared vertices').toBe(false);
            matched.add(match);
          }
        }
        expect(fixture.getCompileCount()).toBe(0);
      } finally {
        fixture.destroy();
      }
    }
  });
}

it('GPUGridGenerator masks corner points independently of polygon intersection, including hole boundaries', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // GeoPandas 1.2 make_grid(feature_type='corners', intersect=True) tests each unique point,
  // not its incident cells: https://geopandas.org/en/latest/docs/reference/api/geopandas.GeoSeries.make_grid.html
  const fixture = createGeometryFixture(device, {
    inputs: {
      extent: {
        format: 'float32x2',
        values: Float32Array.from([0, 0, 3, 0, 3, 3, 0, 3, 1, 1, 2, 1, 2, 2, 1, 2])
      },
      ringOffsets: {format: 'uint32', values: Uint32Array.from([0, 4, 8])}
    },
    outputs: {
      positions: {format: 'float32x2', length: 64},
      corners: {format: 'float32x2', length: 25},
      intersects: {format: 'uint32', length: 16},
      cornerIntersects: {format: 'uint32', length: 25}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUGridGenerator({
        gridType: 'square',
        columns: 4,
        rows: 4,
        parameters,
        extent: {positions: inputs['extent'] as never, ringOffsets: inputs['ringOffsets'] as never},
        output: {
          positions: outputs['positions'] as never,
          corners: outputs['corners'] as never,
          intersects: outputs['intersects'] as never,
          cornerIntersects: outputs['cornerIntersects'] as never
        }
      })
  });
  try {
    for (const origin of [0, 0.5]) {
      const result = await fixture.run(
        getGPUGridGeneratorParameterValues({
          minX: origin,
          minY: origin,
          cellWidth: 1,
          cellHeight: 1
        })
      );
      const expected: number[] = [];
      for (let row = 0; row <= 4; row++) {
        for (let column = 0; column <= 4; column++) {
          const x = origin + column;
          const y = origin + row;
          expect(
            result['corners'].slice(2 * (row * 5 + column), 2 * (row * 5 + column) + 2)
          ).toEqual([x, y]);
          expected.push(Number(x <= 3 && y <= 3 && !(x > 1 && x < 2 && y > 1 && y < 2)));
        }
      }
      expect(result['cornerIntersects']).toEqual(expected);
      expect(result['intersects'].some(Boolean)).toBe(true);
      // Every cell touches the shell at origin zero, including cells adjacent to outside corners.
      if (origin === 0) {
        expect(result['intersects']).toEqual(new Array(16).fill(1));
      }
    }
  } finally {
    fixture.destroy();
  }
});
