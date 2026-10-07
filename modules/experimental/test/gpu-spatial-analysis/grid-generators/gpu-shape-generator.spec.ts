// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUShapeGeneratorParameterValues,
  getGPUShapeVertexCount,
  GPU_SHAPE_GENERATOR_EARTH_RADIUS,
  GPUShapeGenerator,
  type GPUShapeGeneratorProps,
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
  maximumSegments: number,
  options: Pick<GPUShapeGeneratorProps, 'ellipseSpacing'> & {rotations?: number[]} = {}
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
        values: Float32Array.from(
          centers.map(
            (_, index) => options.rotations?.[index] ?? TURF_ELLIPSE_AXES[index]?.[2] ?? 0
          )
        ),
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
        ellipseSpacing: options.ellipseSpacing,
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
  expect(
    () => new GPUShapeGenerator({shape: 'circle', ellipseSpacing: 'arc-length'} as never)
  ).toThrow('ellipseSpacing');
  expect(
    () => new GPUShapeGenerator({shape: 'ellipse', ellipseSpacing: 'unknown'} as never)
  ).toThrow('ellipseSpacing');
});

/** Double-precision Simpson integration of speed, independent of the GPU's chord table. */
function getEllipseArcOracle(axes: number[], segments: number): number[][] {
  const intervals = 16384;
  const step = (2 * Math.PI) / intervals;
  const speed = (phase: number) => Math.hypot(axes[0] * Math.sin(phase), axes[1] * Math.cos(phase));
  const lengths = [0];
  for (let index = 1; index <= intervals; index++) {
    lengths.push(
      lengths[index - 1] +
        (step / 6) *
          (speed((index - 1) * step) + 4 * speed((index - 0.5) * step) + speed(index * step))
    );
  }
  return Array.from({length: segments + 1}, (_, vertex) => {
    const target = (lengths[intervals] * (vertex % segments)) / segments;
    let lower = 0;
    let upper = intervals;
    while (upper - lower > 1) {
      const middle = Math.floor((lower + upper) / 2);
      if (lengths[middle] < target) lower = middle;
      else upper = middle;
    }
    const phase = step * (lower + (target - lengths[lower]) / (lengths[upper] - lengths[lower]));
    return [axes[0] * Math.cos(phase), axes[1] * Math.sin(phase)];
  });
}

function rotateEllipsePoint(point: number[], rotation: number): number[] {
  const tilt = (rotation * Math.PI) / 180;
  return [
    point[0] * Math.cos(tilt) + point[1] * Math.sin(tilt),
    point[1] * Math.cos(tilt) - point[0] * Math.sin(tilt)
  ];
}

it('GPUShapeGenerator arc-length ellipses match integrated arc positions across aspect ratios', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const axes = [
    [1, 1],
    [10, 2],
    [10000, 1],
    [1, 10000]
  ];
  const rotations = [0, 37, 0, -21];
  const fixture = createShapeFixture(
    device,
    'ellipse',
    'planar',
    axes.map(() => [0, 0]),
    axes,
    1024,
    {
      ellipseSpacing: 'arc-length',
      rotations
    }
  );
  try {
    // Odd counts cross quadrant boundaries; large counts exercise intervals near the tips.
    for (const segments of [7, 32, 127, 1024]) {
      const result = await fixture.run(
        getGPUShapeGeneratorParameterValues({segmentCount: segments})
      );
      const vertices = segments + 1;
      expect(result['offsets']).toEqual(
        axes.map((_, index) => index * vertices).concat(axes.length * vertices)
      );
      expect(result['vertexCount'][0]).toBe(axes.length * vertices);
      for (let feature = 0; feature < axes.length; feature++) {
        const oracle = getEllipseArcOracle(axes[feature], segments);
        for (let vertex = 0; vertex <= segments; vertex++) {
          const expected = rotateEllipsePoint(oracle[vertex], rotations[feature]);
          const offset = 2 * (feature * vertices + vertex);
          const error = Math.hypot(
            result['positions'][offset] - expected[0],
            result['positions'][offset + 1] - expected[1]
          );
          expect(
            error,
            `axes=${axes[feature]}, segments=${segments}, vertex=${vertex}`
          ).toBeLessThan(1e-5 * Math.max(...axes[feature]));
        }
        const start = 2 * feature * vertices;
        expect(result['positions'].slice(start + 2 * segments, start + 2 * vertices)).toEqual(
          result['positions'].slice(start, start + 2)
        );
      }
    }
    // An encoded graph recomputes the table from GPU input and parameter contents each frame.
    const updated = axes.map(() => [3, 9]);
    fixture.writeInput('radii', Float32Array.from(updated.flat()));
    const result = await fixture.run(
      getGPUShapeGeneratorParameterValues({segmentCount: 13, radiusScale: 2})
    );
    const oracle = getEllipseArcOracle([6, 18], 13);
    for (let vertex = 0; vertex <= 13; vertex++) {
      expect(
        Math.hypot(
          result['positions'][2 * vertex] - oracle[vertex][0],
          result['positions'][2 * vertex + 1] - oracle[vertex][1]
        )
      ).toBeLessThan(0.00018);
    }
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('GPUShapeGenerator arc-length ellipses preserve collapsed axes and zero radius scale', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const axes = [
    [5, 0],
    [0, 5],
    [0, 0],
    [-1, 5],
    [Number.NaN, 5],
    [5, Number.POSITIVE_INFINITY]
  ];
  const fixture = createShapeFixture(
    device,
    'ellipse',
    'planar',
    axes.map(() => [2, -3]),
    axes,
    32,
    {
      ellipseSpacing: 'arc-length',
      rotations: axes.map(() => 0)
    }
  );
  try {
    const result = await fixture.run(getGPUShapeGeneratorParameterValues({segmentCount: 19}));
    for (let feature = 0; feature < axes.length; feature++) {
      for (let vertex = 0; vertex <= 19; vertex++) {
        const fraction = (vertex % 19) / 19;
        // A degenerate ellipse traverses a line at constant speed, with a triangle waveform.
        const east = Math.abs(4 * fraction - 2) - 1;
        const north = 1 - Math.abs(((4 * fraction + 1) % 4) - 2);
        const expected =
          feature === 2
            ? [2, -3]
            : feature === 0 || feature === 5
              ? [2 + 5 * east, -3]
              : [2, -3 + 5 * north];
        const offset = 2 * (feature * 20 + vertex);
        expect(
          result['positions'][offset],
          `feature=${feature}, vertex=${vertex}, east`
        ).toBeCloseTo(expected[0], 4);
        expect(
          result['positions'][offset + 1],
          `feature=${feature}, vertex=${vertex}, north`
        ).toBeCloseTo(expected[1], 4);
      }
    }
    const collapsed = await fixture.run(
      getGPUShapeGeneratorParameterValues({segmentCount: 8, radiusScale: 0})
    );
    for (let vertex = 0; vertex < axes.length * 9; vertex++) {
      expect(collapsed['positions'].slice(2 * vertex, 2 * vertex + 2)).toEqual([2, -3]);
    }
  } finally {
    fixture.destroy();
  }
});

it('GPUShapeGenerator geodesic arc-length ellipses apply spherical destination after spacing', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const center = [-73, 41];
  for (const [axes, rotation] of [
    [[500000, 10000], 31],
    [[500000, 0], 0],
    [[0, 500000], 0]
  ] as const) {
    const fixture = createShapeFixture(device, 'ellipse', 'geodesic', [center], [[...axes]], 64, {
      ellipseSpacing: 'arc-length',
      rotations: [rotation]
    });
    try {
      const result = await fixture.run(getGPUShapeGeneratorParameterValues({segmentCount: 31}));
      const oracle = getEllipseArcOracle([...axes], 31);
      for (let vertex = 0; vertex <= 31; vertex++) {
        const [east, north] = rotateEllipsePoint(oracle[vertex], rotation);
        const distance = Math.hypot(east, north) / GPU_SHAPE_GENERATOR_EARTH_RADIUS;
        const bearing = Math.atan2(east, north);
        const latitude = (center[1] * Math.PI) / 180;
        const latitude2 = Math.asin(
          Math.sin(latitude) * Math.cos(distance) +
            Math.cos(latitude) * Math.sin(distance) * Math.cos(bearing)
        );
        const longitudeDelta = Math.atan2(
          Math.sin(bearing) * Math.sin(distance) * Math.cos(latitude),
          Math.cos(distance) - Math.sin(latitude) * Math.sin(latitude2)
        );
        expect(
          Math.abs(result['positions'][2 * vertex] - (center[0] + (longitudeDelta * 180) / Math.PI))
        ).toBeLessThan(0.00006);
        expect(
          Math.abs(result['positions'][2 * vertex + 1] - (latitude2 * 180) / Math.PI)
        ).toBeLessThan(0.00006);
      }
    } finally {
      fixture.destroy();
    }
  }
});
