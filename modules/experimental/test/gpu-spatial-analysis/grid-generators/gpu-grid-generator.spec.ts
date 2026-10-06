// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUGridCellCount,
  getGPUGridGeneratorParameterValues,
  getGPUGridVerticesPerCell,
  GPUGridGenerator,
  type GPUGridType
} from '../../../src/gpu-spatial-analysis/grid-generators';
import {createGeometryFixture, expectClose} from '../outline-geometry/geometry-fixture';
import {generateGrid} from './grid-generator-oracle';

const GRID_TYPES: GPUGridType[] = ['square', 'hex', 'triangle', 'point'];

for (const gridType of GRID_TYPES) {
  it(`GPUGridGenerator builds ${gridType} grids like the f64 oracle and moves per frame`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const columns = 7;
    const rows = 5;
    const cellCount = getGPUGridCellCount(gridType, columns, rows);
    const verticesPerCell = getGPUGridVerticesPerCell(gridType);
    const fixture = createGeometryFixture(device, {
      inputs: {},
      outputs: {
        ...(verticesPerCell > 0
          ? {positions: {format: 'float32x2' as const, length: cellCount * verticesPerCell}}
          : {}),
        centers: {format: 'float32x2' as const, length: cellCount}
      },
      create: ({outputs, parameters}) =>
        new GPUGridGenerator({
          gridType,
          columns,
          rows,
          parameters,
          output: {
            positions: outputs['positions'] as never,
            centers: outputs['centers'] as never
          }
        })
    });
    for (const [minX, minY, w, h] of [
      [-10, 20, 2, 3],
      [123.5, -48.25, 0.75, 0.6495]
    ]) {
      const result = await fixture.run(
        getGPUGridGeneratorParameterValues({minX, minY, cellWidth: w, cellHeight: h})
      );
      const expected = generateGrid(gridType, columns, rows, minX, minY, w, h);
      expect(expected.centers.length).toBe(cellCount);
      let maximumDifference = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        for (let axis = 0; axis < 2; axis++) {
          const actual = result['centers'][2 * cell + axis];
          expectClose(actual, expected.centers[cell][axis], 1e-6, 1e-5, `center ${cell}`);
          maximumDifference = Math.max(
            maximumDifference,
            Math.abs(actual - expected.centers[cell][axis])
          );
        }
      }
      for (let vertex = 0; vertex < expected.positions.length; vertex++) {
        for (let axis = 0; axis < 2; axis++) {
          expectClose(
            result['positions'][2 * vertex + axis],
            expected.positions[vertex][axis],
            1e-6,
            1e-5,
            `vertex ${vertex}`
          );
        }
      }
      // Nonzero, distinct output (a failed compile reads back zeros).
      expect(Math.max(...result['centers'].map(Math.abs))).toBeGreaterThan(1);
      expect(maximumDifference).toBeLessThan(1e-3);
      // Polygons are counter-clockwise with positive area.
      for (let cell = 0; cell < cellCount && verticesPerCell > 0; cell++) {
        let area = 0;
        for (let vertex = 0; vertex < verticesPerCell; vertex++) {
          const a = cell * verticesPerCell + vertex;
          const b = cell * verticesPerCell + ((vertex + 1) % verticesPerCell);
          area +=
            result['positions'][2 * a] * result['positions'][2 * b + 1] -
            result['positions'][2 * b] * result['positions'][2 * a + 1];
        }
        expect(area).toBeGreaterThan(0);
      }
    }
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}
