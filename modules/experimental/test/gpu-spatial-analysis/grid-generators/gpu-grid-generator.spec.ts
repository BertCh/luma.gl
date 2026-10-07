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
import {GEOPANDAS_FLAT_HEX, GEOPANDAS_INTERSECT_MASK} from './grid-generator-geopandas-fixture';

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

it('GPUGridGenerator hexOrientation flat matches GeoPandas make_grid flat_topped cells', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {origin, cellSize, columns, rows, cells} = GEOPANDAS_FLAT_HEX;
  const fixture = createGeometryFixture(device, {
    inputs: {},
    outputs: {
      positions: {format: 'float32x2', length: columns * rows * 6},
      centers: {format: 'float32x2', length: columns * rows}
    },
    create: ({outputs, parameters}) =>
      new GPUGridGenerator({
        gridType: 'hex',
        hexOrientation: 'flat',
        columns,
        rows,
        parameters,
        output: {positions: outputs['positions'] as never, centers: outputs['centers'] as never}
      })
  });
  const result = await fixture.run(
    getGPUGridGeneratorParameterValues({
      minX: origin[0],
      minY: origin[1],
      cellWidth: cellSize,
      cellHeight: cellSize
    })
  );
  for (let cell = 0; cell < cells.length; cell++) {
    expectClose(result['centers'][2 * cell], cells[cell].center[0], 1e-6, 1e-5, `center ${cell}`);
    expectClose(
      result['centers'][2 * cell + 1],
      cells[cell].center[1],
      1e-6,
      1e-5,
      `center ${cell}`
    );
    // GeoPandas starts the ring at the bottom-left corner, ours at angle 0: align by nearest vertex.
    const actual = Array.from({length: 6}, (_, vertex) => [
      result['positions'][2 * (cell * 6 + vertex)],
      result['positions'][2 * (cell * 6 + vertex) + 1]
    ]);
    const reference = cells[cell].ring;
    const shift = reference.findIndex(
      point => Math.hypot(point[0] - actual[0][0], point[1] - actual[0][1]) < 1e-4
    );
    expect(shift, `cell ${cell} corner 0`).toBeGreaterThanOrEqual(0);
    for (let vertex = 0; vertex < 6; vertex++) {
      const expected = reference[(shift + vertex) % 6];
      expectClose(actual[vertex][0], expected[0], 1e-6, 1e-5, `cell ${cell} x ${vertex}`);
      expectClose(actual[vertex][1], expected[1], 1e-6, 1e-5, `cell ${cell} y ${vertex}`);
    }
  }
  fixture.destroy();
});

for (const gridType of ['square', 'hex', 'hexFlat', 'triangle', 'point'] as const) {
  it(`GPUGridGenerator intersects mask matches shapely intersects for ${gridType} cells`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const {rings, origin, cellWidth, cellHeight, columns, rows, masks} = GEOPANDAS_INTERSECT_MASK;
    const isHex = gridType === 'hex' || gridType === 'hexFlat';
    const publicType: GPUGridType = isHex ? 'hex' : gridType;
    const cellCount = getGPUGridCellCount(publicType, columns, rows);
    const verticesPerCell = getGPUGridVerticesPerCell(publicType);
    const ringOffsets = [0];
    for (const ring of rings) {
      ringOffsets.push(ringOffsets[ringOffsets.length - 1] + ring.length);
    }
    const fixture = createGeometryFixture(device, {
      inputs: {
        extentPositions: {
          values: Float32Array.from(rings.flat().flat()),
          format: 'float32x2'
        },
        extentRingOffsets: {values: Uint32Array.from(ringOffsets), format: 'uint32'}
      },
      outputs: {
        ...(verticesPerCell > 0
          ? {positions: {format: 'float32x2' as const, length: cellCount * verticesPerCell}}
          : {centers: {format: 'float32x2' as const, length: cellCount}}),
        intersects: {format: 'uint32', length: cellCount}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUGridGenerator({
          gridType: publicType,
          hexOrientation: gridType === 'hexFlat' ? 'flat' : undefined,
          columns,
          rows,
          parameters,
          extent: {
            positions: inputs['extentPositions'] as never,
            ringOffsets: inputs['extentRingOffsets'] as never
          },
          output: {
            positions: outputs['positions'] as never,
            centers: outputs['centers'] as never,
            intersects: outputs['intersects'] as never
          }
        })
    });
    const result = await fixture.run(
      getGPUGridGeneratorParameterValues({
        minX: origin[0],
        minY: origin[1],
        cellWidth,
        cellHeight
      })
    );
    expect(Array.from(result['intersects'])).toEqual(masks[gridType]);
    // Both outcomes occur, so a failed compile (all zeros) cannot pass.
    expect(masks[gridType].includes(0) && masks[gridType].includes(1)).toBe(true);
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}
