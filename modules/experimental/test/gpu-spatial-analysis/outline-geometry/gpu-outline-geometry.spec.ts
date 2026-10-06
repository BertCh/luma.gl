// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUOutlineGeometryParameterValues,
  getGPUOutlineGeometryVerticesPerInput,
  GPUOutlineGeometry,
  type GPUOutlineGeometryType
} from '../../../src/gpu-spatial-analysis/outline-geometry';
import {createGeometryFixture, createRandom, expectClose} from './geometry-fixture';
import {
  buildOutlineTriangles,
  getPointSegmentDistance,
  isCoveredByTriangles,
  type Point
} from './outline-geometry-oracle';

const JOIN_SEGMENTS = 12;

const PATHS: Point[][] = [
  [
    [0, 0],
    [10, 0],
    [10, 8],
    [3, 9]
  ],
  [[20, 20]],
  [
    [-5, -5],
    [-5, -5],
    [-1, -9]
  ],
  [
    [30, 0],
    [36, 2],
    [33, 7]
  ]
];

function flatten(paths: Point[][]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const path of paths) {
    for (const [x, y] of path) {
      positions.push(x, y);
    }
    offsets.push(positions.length / 2);
  }
  return {vertices: paths.flat(), positions: new Float32Array(positions), offsets};
}

for (const geometryType of ['points', 'lines', 'rings'] as GPUOutlineGeometryType[]) {
  it(`GPUOutlineGeometry builds ${geometryType} like the f64 oracle and covers the offset region`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const {vertices, positions, offsets} = flatten(PATHS);
    const verticesPerInput = getGPUOutlineGeometryVerticesPerInput(JOIN_SEGMENTS);
    const fixture = createGeometryFixture(device, {
      inputs: {
        positions: {values: positions, format: 'float32x2'},
        ...(geometryType === 'points'
          ? {}
          : {pathOffsets: {values: new Uint32Array(offsets), format: 'uint32'}})
      },
      outputs: {
        output: {format: 'float32x2', length: vertices.length * verticesPerInput}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUOutlineGeometry({
          positions: inputs['positions'] as never,
          pathOffsets: inputs['pathOffsets'] as never,
          geometryType,
          joinSegments: JOIN_SEGMENTS,
          parameters,
          output: {positions: outputs['output'] as never}
        })
    });
    const random = createRandom(7);
    for (const distance of [1.5, 0.25, 4]) {
      const result = await fixture.run(getGPUOutlineGeometryParameterValues({distance}));
      const expected = buildOutlineTriangles(
        vertices,
        geometryType === 'points' ? undefined : offsets,
        geometryType,
        distance,
        JOIN_SEGMENTS
      );
      expect(expected.length).toBe(vertices.length * verticesPerInput);
      for (let vertex = 0; vertex < expected.length; vertex++) {
        expectClose(result['output'][2 * vertex], expected[vertex][0], 1e-5, 1e-4);
        expectClose(result['output'][2 * vertex + 1], expected[vertex][1], 1e-5, 1e-4);
      }
      expect(Math.max(...result['output'].map(Math.abs))).toBeGreaterThan(10);
      // Coverage: points well inside the offset are covered, points beyond it are not.
      const triangles: Point[] = [];
      for (let vertex = 0; vertex < expected.length; vertex++) {
        triangles.push([result['output'][2 * vertex], result['output'][2 * vertex + 1]]);
      }
      const inscribed = Math.cos(Math.PI / JOIN_SEGMENTS);
      let insideChecked = 0;
      let outsideChecked = 0;
      for (let sample = 0; sample < 400; sample++) {
        const point: Point = [random() * 60 - 12, random() * 40 - 14];
        let nearest = Infinity;
        PATHS.forEach((path, pathIndex) => {
          if (geometryType === 'points' || path.length === 1) {
            for (const vertex of path) {
              nearest = Math.min(nearest, Math.hypot(point[0] - vertex[0], point[1] - vertex[1]));
            }
          }
          const last = geometryType === 'rings' ? path.length : path.length - 1;
          for (let segment = 0; geometryType !== 'points' && segment < last; segment++) {
            nearest = Math.min(
              nearest,
              getPointSegmentDistance(point, path[segment], path[(segment + 1) % path.length])
            );
          }
          void pathIndex;
        });
        if (nearest <= distance * inscribed * 0.98) {
          insideChecked++;
          expect(isCoveredByTriangles(point, triangles), `inside ${point}`).toBe(true);
        } else if (nearest > distance * 1.02) {
          outsideChecked++;
          expect(isCoveredByTriangles(point, triangles), `outside ${point}`).toBe(false);
        }
      }
      expect(insideChecked).toBeGreaterThan(0);
      expect(outsideChecked).toBeGreaterThan(0);
    }
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}

it('GPUOutlineGeometry offsets spherical vertices by meters', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = new Float32Array([10, 60, 10.01, 60]);
  const verticesPerInput = getGPUOutlineGeometryVerticesPerInput(8);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: positions, format: 'float32x2'},
      pathOffsets: {values: new Uint32Array([0, 2]), format: 'uint32'}
    },
    outputs: {output: {format: 'float32x2', length: 2 * verticesPerInput}},
    create: ({inputs, outputs, parameters}) =>
      new GPUOutlineGeometry({
        positions: inputs['positions'] as never,
        pathOffsets: inputs['pathOffsets'] as never,
        geometryType: 'lines',
        coordinateSystem: 'spherical',
        joinSegments: 8,
        parameters,
        output: {positions: outputs['output'] as never}
      })
  });
  const meters = 500;
  const result = await fixture.run(getGPUOutlineGeometryParameterValues({distance: meters}));
  const degreesPerMeterLatitude = 180 / Math.PI / 6371008.8;
  // First disc triangle: center, then the point at angle 0 (due east), then 45 degrees.
  expectClose(
    result['output'][2],
    10 + (meters * degreesPerMeterLatitude) / Math.cos(Math.PI / 3),
    1e-4,
    1e-6
  );
  expectClose(result['output'][3], 60, 0, 1e-5);
  // Quad corner north of the first vertex: the segment runs east, its left normal points north.
  const quadStart = 3 * 8;
  expectClose(
    result['output'][2 * quadStart + 1],
    60 + meters * degreesPerMeterLatitude,
    1e-4,
    1e-6
  );
  fixture.destroy();
});
