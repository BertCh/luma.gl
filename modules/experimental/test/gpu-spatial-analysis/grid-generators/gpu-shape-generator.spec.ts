// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUShapeGeneratorParameterValues,
  getGPUShapeVertexCount,
  GPUShapeGenerator,
  type GPUShapeType
} from '../../../src/gpu-spatial-analysis/grid-generators/index';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {
  TURF_ELLIPSE_AXES,
  TURF_SECTOR_BEARINGS,
  TURF_SHAPE_CENTERS,
  TURF_SHAPE_RADII,
  TURF_SHAPE_VALUES
} from './turf-shape-values';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

function createShapeFixture(
  device: Device,
  shape: GPUShapeType,
  coordinateSystem: 'planar' | 'geodesic',
  centers: number[][],
  radii: number[][],
  maximumSegments: number
) {
  const featureCount = centers.length;
  const vertexCapacity = featureCount * getGPUShapeVertexCount(shape, maximumSegments);
  return createGeometryFixture(device, {
    inputs: {
      centers: {values: Float32Array.from(centers.flat()), format: 'float32x2'},
      radii:
        shape === 'ellipse'
          ? {values: Float32Array.from(radii.flat()), format: 'float32x2'}
          : {values: Float32Array.from(radii.map(row => row[0])), format: 'float32'},
      bearings: {
        values: Float32Array.from(TURF_SECTOR_BEARINGS.slice(0, featureCount).flat()),
        format: 'float32x2'
      },
      rotations: {
        values: Float32Array.from(TURF_ELLIPSE_AXES.slice(0, featureCount).map(axes => axes[2])),
        format: 'float32'
      }
    },
    outputs: {
      positions: {format: 'float32x2', length: vertexCapacity},
      offsets: {format: 'uint32', length: featureCount + 1},
      vertexCount: {format: 'uint32', length: 1}
    },
    parameterLength: 2,
    create: ({inputs, outputs, parameters}) =>
      new GPUShapeGenerator({
        shape,
        coordinateSystem,
        maximumSegments,
        centers: inputs['centers'] as never,
        radii: inputs['radii'] as never,
        bearings: shape === 'sector' ? (inputs['bearings'] as never) : undefined,
        rotations: shape === 'ellipse' ? (inputs['rotations'] as never) : undefined,
        parameters,
        output: {
          positions: outputs['positions'] as never,
          offsets: outputs['offsets'] as never,
          vertexCount: outputs['vertexCount'] as never
        }
      })
  });
}

for (const [shape, segments] of [
  ['circle', 12],
  ['sector', 8],
  ['ellipse', 10]
] as const) {
  it(`GPUShapeGenerator geodesic ${shape} matches turf and changes segments per frame`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const maximumSegments = 32;
    const radii =
      shape === 'ellipse'
        ? TURF_ELLIPSE_AXES.map(axes => [axes[0], axes[1]])
        : TURF_SHAPE_RADII.map(radius => [radius]);
    const fixture = createShapeFixture(
      device,
      shape,
      'geodesic',
      TURF_SHAPE_CENTERS,
      radii,
      maximumSegments
    );
    const result = await fixture.run(getGPUShapeGeneratorParameterValues({segmentCount: segments}));
    const vertices = getGPUShapeVertexCount(shape, segments);
    expect(result['offsets']).toEqual([0, vertices, 2 * vertices, 3 * vertices]);
    expect(result['vertexCount'][0]).toBe(3 * vertices);
    let maximumError = 0;
    for (let feature = 0; feature < 3; feature++) {
      const expected = TURF_SHAPE_VALUES[shape][feature];
      expect(expected.length).toBe(vertices);
      for (let vertex = 0; vertex < vertices; vertex++) {
        for (let axis = 0; axis < 2; axis++) {
          const actual = result['positions'][2 * (feature * vertices + vertex) + axis];
          maximumError = Math.max(maximumError, Math.abs(actual - expected[vertex][axis]));
        }
      }
    }
    // f32 coordinates near 140 degrees round to 1.5e-5.
    expect(maximumError).toBeLessThan(3e-5);
    expect(Math.max(...result['positions'].map(Math.abs))).toBeGreaterThan(10);
    // A different segment count and radius scale rewrite offsets and vertices without compiling.
    const coarse = await fixture.run(
      getGPUShapeGeneratorParameterValues({segmentCount: 5, radiusScale: 2})
    );
    const coarseVertices = getGPUShapeVertexCount(shape, 5);
    expect(coarse['offsets'][3]).toBe(3 * coarseVertices);
    expect(coarse['vertexCount'][0]).toBe(3 * coarseVertices);
    // Clamped to the compile-time maximum.
    const clamped = await fixture.run(getGPUShapeGeneratorParameterValues({segmentCount: 1000}));
    expect(clamped['vertexCount'][0]).toBe(3 * getGPUShapeVertexCount(shape, maximumSegments));
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}

it('GPUShapeGenerator planar rings are exact offsets by radius and bearing', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const centers = [
    [10, -5],
    [0.5, 200]
  ];
  const fixtureCircle = createShapeFixture(device, 'circle', 'planar', centers, [[3], [0.25]], 16);
  const circle = await fixtureCircle.run(getGPUShapeGeneratorParameterValues({segmentCount: 8}));
  // Counter-clockwise from north: first vertex is straight up, quarter turn is west.
  expect(circle['positions'][0]).toBeCloseTo(10, 5);
  expect(circle['positions'][1]).toBeCloseTo(-2, 5);
  expect(circle['positions'][2 * 2]).toBeCloseTo(7, 5);
  expect(circle['positions'][2 * 2 + 1]).toBeCloseTo(-5, 5);
  expect(circle['positions'][2 * 8]).toBe(circle['positions'][0]);
  const second = 9 + 8;
  expect(circle['positions'][2 * second]).toBeCloseTo(0.5, 5);
  expect(circle['positions'][2 * second + 1]).toBeCloseTo(200.25, 4);
  fixtureCircle.destroy();

  const fixtureSector = createShapeFixture(device, 'sector', 'planar', centers, [[2], [1]], 16);
  const sector = await fixtureSector.run(getGPUShapeGeneratorParameterValues({segmentCount: 4}));
  // Feature 0 bearings (30, 200) clockwise: center, arc from 30 to 200 degrees, center.
  const pointAt = (index: number) => [
    sector['positions'][2 * index],
    sector['positions'][2 * index + 1]
  ];
  expect(pointAt(0)).toEqual([10, -5]);
  expect(pointAt(6)).toEqual([10, -5]);
  const sine = Math.sin((30 * Math.PI) / 180);
  expect(pointAt(1)[0]).toBeCloseTo(10 + 2 * sine, 5);
  expect(pointAt(5)[0]).toBeCloseTo(10 + 2 * Math.sin((200 * Math.PI) / 180), 5);
  expect(pointAt(5)[1]).toBeCloseTo(-5 + 2 * Math.cos((200 * Math.PI) / 180), 5);
  fixtureSector.destroy();
});

it('GPUShapeGenerator validates its inputs', () => {
  expect(() => getGPUShapeGeneratorParameterValues({segmentCount: 0})).toThrow();
  expect(() => getGPUShapeGeneratorParameterValues({segmentCount: 8, radiusScale: -1})).toThrow();
  expect(getGPUShapeVertexCount('sector', 8)).toBe(11);
});
