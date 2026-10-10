// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';
import {
  GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS,
  GPUGeometryMeasures,
  type GPUGeometryMeasuresProps
} from '../../../src/gpu-spatial-analysis/geometry-measures';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createFlatGeometry,
  getSphericalBoxArea,
  getWgs84BoxArea,
  measureFeature,
  roundFeatures,
  type FeatureMeasures,
  type NestedFeatures
} from './geometry-measures-oracle';

type MeasuresReadback = {
  lengths: number[];
  areas: number[];
  signedAreas: number[];
  centroids: number[][];
  bounds: number[][];
  vertexCounts: number[];
  featureCounts?: number[];
};

type MeasuresResult = {
  features: MeasuresReadback;
  groups?: MeasuresReadback;
  groupBits?: number[];
};

type MeasuresOptions = Pick<GPUGeometryMeasuresProps, 'geometryType' | 'holeRule'> & {
  coordinateSystem: 'planar' | 'spherical' | 'wgs84' | 'geodesic';
  radius?: number;
  groupIds?: number[];
  groupCount?: number;
};

type MeasuresFixture = {
  run(): Promise<MeasuresResult>;
  /** Rewrites positions (same layout) for the next run. */
  writePositions(positions: Float32Array): void;
  getCompileCount(): number;
  destroy(): void;
};

function createMeasuresFixture(
  device: Device,
  features: NestedFeatures,
  options: MeasuresOptions
): MeasuresFixture {
  const flat = createFlatGeometry(features);
  const featureCount = features.length;
  const isPolygon = options.geometryType === 'polygons';
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'geometry-measures-fixture'});
  const positionsBuffer = track(createInputBuffer(device, flat.positions));
  const rowCount = flat.positions.length / 2;
  const createColumns = (prefix: string, rowLength: number, withFeatureCounts: boolean) => {
    const buffersByName = {
      lengths: track(createOutputBuffer(device, rowLength)),
      areas: track(createOutputBuffer(device, rowLength)),
      signedAreas: track(createOutputBuffer(device, rowLength)),
      centroids: track(createOutputBuffer(device, 2 * rowLength)),
      bounds: track(createOutputBuffer(device, 4 * rowLength)),
      vertexCounts: track(createOutputBuffer(device, rowLength)),
      featureCounts: track(createOutputBuffer(device, rowLength))
    };
    const views = {
      lengths: importGraphBuffer(
        graph,
        `${prefix}-lengths`,
        buffersByName.lengths,
        'float32',
        rowLength
      ),
      ...(isPolygon
        ? {
            areas: importGraphBuffer(
              graph,
              `${prefix}-areas`,
              buffersByName.areas,
              'float32',
              rowLength
            ),
            signedAreas: importGraphBuffer(
              graph,
              `${prefix}-signed-areas`,
              buffersByName.signedAreas,
              'float32',
              rowLength
            )
          }
        : {}),
      centroids: importGraphBuffer(
        graph,
        `${prefix}-centroids`,
        buffersByName.centroids,
        'float32x2',
        rowLength
      ),
      bounds: importGraphBuffer(
        graph,
        `${prefix}-bounds`,
        buffersByName.bounds,
        'float32x4',
        rowLength
      ),
      vertexCounts: importGraphBuffer(
        graph,
        `${prefix}-vertex-counts`,
        buffersByName.vertexCounts,
        'uint32',
        rowLength
      ),
      ...(withFeatureCounts
        ? {
            featureCounts: importGraphBuffer(
              graph,
              `${prefix}-feature-counts`,
              buffersByName.featureCounts,
              'uint32',
              rowLength
            )
          }
        : {})
    };
    const read = async (): Promise<MeasuresReadback> => {
      const centroids = await readFloat32(buffersByName.centroids, 2 * rowLength);
      const bounds = await readFloat32(buffersByName.bounds, 4 * rowLength);
      return {
        lengths: await readFloat32(buffersByName.lengths, rowLength),
        areas: await readFloat32(buffersByName.areas, rowLength),
        signedAreas: await readFloat32(buffersByName.signedAreas, rowLength),
        centroids: Array.from({length: rowLength}, (_, row) =>
          centroids.slice(2 * row, 2 * row + 2)
        ),
        bounds: Array.from({length: rowLength}, (_, row) => bounds.slice(4 * row, 4 * row + 4)),
        vertexCounts: await readUint32(buffersByName.vertexCounts, rowLength),
        featureCounts: withFeatureCounts
          ? await readUint32(buffersByName.featureCounts, rowLength)
          : undefined
      };
    };
    return {views, read, buffers: buffersByName};
  };
  const featureColumns = createColumns('feature', featureCount, false);
  const groupColumns = options.groupIds
    ? createColumns('group', options.groupCount ?? 1, true)
    : undefined;
  graph.add(
    new GPUGeometryMeasures({
      spatialContext: {
        coordinateSpace: options.coordinateSystem === 'planar' ? 'planar' : 'longitude-latitude',
        metric:
          options.coordinateSystem === 'planar'
            ? 'native'
            : options.coordinateSystem === 'spherical'
              ? 'great-circle'
              : 'ellipsoidal',
        units: options.coordinateSystem === 'planar' ? 'native' : 'meters',
        sphereRadius: options.coordinateSystem === 'spherical' ? options.radius : undefined
      },
      ellipsoidalEdgeModel: options.coordinateSystem === 'wgs84' ? 'coordinate-linear' : undefined,
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      ringOffsets: importGraphBuffer(
        graph,
        'ring-offsets',
        track(createInputBuffer(device, flat.ringOffsets)),
        'uint32',
        flat.ringOffsets.length
      ),
      featureRingOffsets: importGraphBuffer(
        graph,
        'feature-ring-offsets',
        track(createInputBuffer(device, flat.featureRingOffsets)),
        'uint32',
        flat.featureRingOffsets.length
      ),
      geometryType: options.geometryType,

      holeRule: options.holeRule,

      output: featureColumns.views,
      ...(options.groupIds && groupColumns
        ? {
            groupIds: importGraphBuffer(
              graph,
              'group-ids',
              track(createInputBuffer(device, new Uint32Array(options.groupIds))),
              'uint32',
              featureCount
            ),
            groupCount: options.groupCount,
            groupOutput: groupColumns.views
          }
        : {})
    })
  );
  const compiled = graph.compile();
  const compileSpy = vi.spyOn(graph, 'compile');
  return {
    async run() {
      submitGraph(device, compiled, undefined);
      const result: MeasuresResult = {features: await featureColumns.read()};
      if (groupColumns) {
        result.groups = await groupColumns.read();
        const groupCount = options.groupCount ?? 1;
        result.groupBits = [
          ...(await readUint32(groupColumns.buffers.lengths, groupCount)),
          ...(await readUint32(groupColumns.buffers.centroids, 2 * groupCount)),
          ...(await readUint32(groupColumns.buffers.areas, groupCount))
        ];
      }
      return result;
    },
    writePositions(positions) {
      positionsBuffer.write(positions);
    },
    getCompileCount() {
      return compileSpy.mock.calls.length;
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectClose(actual: number, expected: number, tolerance: number, label: string): void {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
    return;
  }
  // Allow one f32 ulp of the expected magnitude on top of the stated tolerance.
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    tolerance + 1.2e-7 * Math.abs(expected)
  );
}

function expectFeatureParity(
  actual: MeasuresReadback,
  expected: FeatureMeasures[],
  tolerances: {length: number; area: number; position: number},
  isPolygon: boolean
): void {
  expected.forEach((reference, feature) => {
    const label = `feature ${feature}`;
    expect(actual.vertexCounts[feature], label).toBe(reference.vertexCount);
    expectClose(
      actual.lengths[feature],
      reference.length,
      tolerances.length * Math.max(1, reference.length),
      `${label} length`
    );
    if (isPolygon) {
      expectClose(
        actual.signedAreas[feature],
        reference.signedArea,
        tolerances.area * Math.max(1, Math.abs(reference.signedArea)),
        `${label} signed area`
      );
      expectClose(
        actual.areas[feature],
        reference.area,
        tolerances.area * Math.max(1, reference.area),
        `${label} area`
      );
    }
    for (let axis = 0; axis < 2; axis++) {
      expectClose(
        actual.centroids[feature][axis],
        reference.centroid[axis],
        tolerances.position,
        `${label} centroid ${axis}`
      );
    }
    for (let corner = 0; corner < 4; corner++) {
      expectClose(
        actual.bounds[feature][corner],
        reference.bounds[corner],
        tolerances.position,
        `${label} bounds ${corner}`
      );
    }
  });
}

const square = (x: number, y: number, size: number, clockwise = false): number[][] => {
  const ring = [
    [x, y],
    [x + size, y],
    [x + size, y + size],
    [x, y + size]
  ];
  return clockwise ? ring.reverse() : ring;
};

const PLANAR_FEATURES: NestedFeatures = [
  // Unit square, CCW, no closing vertex.
  [square(0, 0, 1)],
  // Square with a CW hole (RFC 7946 winding).
  [square(0, 0, 10), square(2, 2, 3, true)],
  // Same with a CCW hole: only first-ring-exterior subtracts it.
  [square(0, 0, 10), square(2, 2, 3)],
  // Multipolygon: two CCW squares.
  [square(0, 0, 2), square(5, 5, 1)],
  // L-shape with an explicit closing vertex, clockwise.
  [
    [
      [0, 0],
      [0, 2],
      [1, 2],
      [1, 1],
      [2, 1],
      [2, 0],
      [0, 0]
    ]
  ],
  // Empty feature.
  [],
  // Degenerate (collinear) ring: zero area, vertex-mean centroid.
  [
    [
      [0, 0],
      [1, 1],
      [2, 2]
    ]
  ],
  // Far from the origin: a 1 m square at 1e6 m (Web Mercator scale).
  [square(1e6 + 0.25, 2e6 + 0.5, 1)]
];

it('GPUGeometryMeasures matches the planar oracle under both hole rules', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features = roundFeatures(PLANAR_FEATURES);
  for (const holeRule of ['winding', 'first-ring-exterior'] as const) {
    const fixture = createMeasuresFixture(device, features, {
      geometryType: 'polygons',
      coordinateSystem: 'planar',
      holeRule
    });
    const {features: actual} = await fixture.run();
    const expected = features.map(feature =>
      measureFeature(feature, {isPolygon: true, coordinateSystem: 'planar', holeRule})
    );
    expectFeatureParity(actual, expected, {length: 1e-6, area: 1e-6, position: 1e-5}, true);
    fixture.destroy();
  }
  // Spot checks of the semantics, independent of the oracle.
  const fixture = createMeasuresFixture(device, features, {
    geometryType: 'polygons',
    coordinateSystem: 'planar'
  });
  const {features: winding} = await fixture.run();
  expect(winding.areas.slice(0, 5)).toEqual([1, 91, 109, 5, 3]);
  expect(winding.signedAreas[4]).toBe(-3);
  expect(winding.areas[7]).toBe(1);
  expect(winding.centroids[7][0]).toBe(Math.fround(1e6 + 0.75));
  expect(winding.lengths[0]).toBe(4);
  expect(winding.vertexCounts[5]).toBe(0);
  expect(winding.centroids[5][0]).toBeNaN();
  fixture.destroy();
});

it('GPUGeometryMeasures measures lines with length-weighted centroids', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features = roundFeatures([
    [
      [
        [0, 0],
        [3, 4],
        [3, 10]
      ]
    ],
    // Multi-line feature.
    [
      [
        [0, 0],
        [1, 0]
      ],
      [
        [5, 5],
        [5, 7]
      ]
    ],
    [[[2, 2]]],
    [
      [
        [123456.5, 654321.25],
        [123457.5, 654321.25],
        [123457.5, 654323.25]
      ]
    ]
  ]);
  const fixture = createMeasuresFixture(device, features, {
    geometryType: 'lines',
    coordinateSystem: 'planar'
  });
  const {features: actual} = await fixture.run();
  const expected = features.map(feature =>
    measureFeature(feature, {isPolygon: false, coordinateSystem: 'planar'})
  );
  expectFeatureParity(actual, expected, {length: 1e-6, area: 0, position: 1e-5}, false);
  expect(actual.lengths[0]).toBe(11);
  expect(actual.centroids[1]).toEqual([3.5, 4]);
  fixture.destroy();
});

it('GPUGeometryMeasures spherical areas match closed forms and the f64 oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const radius = GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS;
  const features = roundFeatures([
    [square(10, 0, 1)],
    [square(-45, 60, 1)],
    // Antimeridian-crossing box 179E..179W, 10S..10N.
    [
      [
        [179, -10],
        [-179, -10],
        [-179, 10],
        [179, 10]
      ]
    ],
    // Country-sized irregular polygon (roughly France) with a hole.
    [
      [
        [-4.7, 48.4],
        [-1.2, 46.2],
        [-1.8, 43.4],
        [3.1, 42.4],
        [7.6, 43.8],
        [6.1, 46.3],
        [8.2, 49.0],
        [2.5, 51.1]
      ],
      [
        [1.5, 46.5],
        [2.5, 47.5],
        [3.5, 46.5]
      ]
    ],
    // Small city block, about 100 m.
    [square(2.35, 48.85, 0.001)]
  ]);
  const fixture = createMeasuresFixture(device, features, {
    geometryType: 'polygons',
    coordinateSystem: 'spherical',
    radius
  });
  const {features: actual} = await fixture.run();
  const expected = features.map(feature =>
    measureFeature(feature, {isPolygon: true, coordinateSystem: 'spherical', radius})
  );
  expectFeatureParity(actual, expected, {length: 2e-6, area: 2e-5, position: 2e-5}, true);
  const boxes = [
    getSphericalBoxArea(10, 0, 11, 1, radius),
    getSphericalBoxArea(-45, 60, -44, 61, radius),
    getSphericalBoxArea(179, -10, 181, 10, radius)
  ];
  let maximumRelativeError = 0;
  boxes.forEach((area, feature) => {
    const relativeError = Math.abs(actual.areas[feature] - area) / area;
    maximumRelativeError = Math.max(maximumRelativeError, relativeError);
    expect(relativeError, `box ${feature}`).toBeLessThan(1e-6);
  });
  expect(actual.bounds[2]).toEqual([179, -10, 181, 10]);
  expect(actual.centroids[2][0]).toBeCloseTo(180, 4);
  console.log(`GPUGeometryMeasures spherical box area max relative error ${maximumRelativeError}`);
  fixture.destroy();
});

it('GPUGeometryMeasures WGS84 lengths match f64 Vincenty', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const segments: number[][][] = [
    [
      [2.35, 48.85],
      [2.350013, 48.850001]
    ], // ~1 m
    [
      [-73.98, 40.75],
      [-73.97, 40.758]
    ], // ~1.2 km
    [
      [139.7, 35.7],
      [140.5, 36.4]
    ], // ~105 km
    [
      [-0.13, 51.5],
      [2.35, 48.85]
    ], // ~344 km
    [
      [-74, 40.7],
      [-0.13, 51.5]
    ], // ~5,570 km
    [
      [151.2, -33.9],
      [-118.2, 34]
    ], // ~12,000 km across the antimeridian
    [
      [0, 89.5],
      [180, 89.5]
    ] // over the pole
  ];
  const features = roundFeatures(segments.map(segment => [segment]));
  const fixture = createMeasuresFixture(device, features, {
    geometryType: 'lines',
    coordinateSystem: 'wgs84'
  });
  const {features: actual} = await fixture.run();
  const report: string[] = [];
  features.forEach((feature, index) => {
    const reference = measureFeature(feature, {isPolygon: false, coordinateSystem: 'wgs84'}).length;
    const error = Math.abs(actual.lengths[index] - reference);
    report.push(`${reference.toFixed(1)} m: ${error.toExponential(2)} m`);
    // f32 Vincenty: about 1e-7 relative.
    expect(error, `segment ${index}`).toBeLessThanOrEqual(1e-3 + 2e-7 * reference);
  });
  console.log(`GPUGeometryMeasures wgs84 length errors vs Vincenty: ${report.join(', ')}`);
  fixture.destroy();
});

it('GPUGeometryMeasures WGS84 areas match exact authalic box areas and the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const boxes = [
    [0, 0, 1],
    [-45, 60, 1],
    [100, -80, 5],
    [2.35, 48.85, 0.001],
    [-120, 30, 20]
  ];
  const features = roundFeatures(boxes.map(([x, y, size]) => [square(x, y, size)]));
  const fixture = createMeasuresFixture(device, features, {
    geometryType: 'polygons',
    coordinateSystem: 'wgs84'
  });
  const {features: actual} = await fixture.run();
  const report: string[] = [];
  features.forEach((feature, index) => {
    // The f32-rounded corners, exactly as the GPU sees them.
    const ring = feature[0];
    const exact = getWgs84BoxArea(ring[0][0], ring[0][1], ring[1][0], ring[2][1]);
    const relativeError = Math.abs(actual.areas[index] - exact) / exact;
    report.push(`${exact.toExponential(3)} m2: ${relativeError.toExponential(2)}`);
    expect(relativeError, `box ${index}`).toBeLessThan(1e-6);
    const oracle = measureFeature(feature, {isPolygon: true, coordinateSystem: 'wgs84'});
    // asin near the poles amplifies f32 rounding: about 2 m of latitude at 80 degrees.
    expect(
      Math.abs(actual.centroids[index][1] - oracle.centroid[1]),
      `box ${index} centroid latitude`
    ).toBeLessThan(5e-5);
  });
  console.log(`GPUGeometryMeasures wgs84 box area relative errors: ${report.join(', ')}`);
  fixture.destroy();
});

it('GPUGeometryMeasures reduces groups deterministically and re-encodes new positions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  let seed = 7;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0x100000000;
  };
  const featureCount = 600;
  const groupCount = 5;
  const features: NestedFeatures = Array.from({length: featureCount}, (_, feature) =>
    feature % 50 === 7 ? [] : [square(random() * 1000, random() * 1000, 1 + random() * 20)]
  );
  // Group 3 stays empty; some IDs are out of range and ignored.
  const groupIds = features.map((_, feature) =>
    feature % 97 === 0 ? 99 : [0, 1, 2, 4][feature % 4]
  );
  const rounded = roundFeatures(features);
  const fixture = createMeasuresFixture(device, rounded, {
    geometryType: 'polygons',
    coordinateSystem: 'planar',
    groupIds,
    groupCount
  });
  const first = await fixture.run();
  const second = await fixture.run();
  expect(second.groupBits).toEqual(first.groupBits);
  const expected = rounded.map(feature =>
    measureFeature(feature, {isPolygon: true, coordinateSystem: 'planar'})
  );
  for (let group = 0; group < groupCount; group++) {
    const members = expected.filter((_, feature) => groupIds[feature] === group);
    const groups = first.groups;
    if (!groups) {
      throw new Error('missing groups');
    }
    expect(groups.featureCounts?.[group]).toBe(members.length);
    const valid = members.filter(member => member.vertexCount > 0);
    expect(groups.vertexCounts[group]).toBe(
      valid.reduce((sum, member) => sum + member.vertexCount, 0)
    );
    const area = members.reduce((sum, member) => sum + member.area, 0);
    expectClose(groups.areas[group], area, 1e-5 * Math.max(1, area), `group ${group} area`);
    if (valid.length === 0) {
      expect(groups.centroids[group][0]).toBeNaN();
      expect(groups.bounds[group][0]).toBeNaN();
      continue;
    }
    const centroidX =
      valid.reduce((sum, member) => sum + member.area * member.centroid[0], 0) / area;
    expectClose(groups.centroids[group][0], centroidX, 1e-3, `group ${group} centroid`);
    expectClose(
      groups.bounds[group][0],
      Math.min(...valid.map(member => member.bounds[0])),
      1e-3,
      `group ${group} bounds`
    );
  }
  // New positions, same layout: no recompile.
  const shifted = roundFeatures(
    features.map(feature => feature.map(ring => ring.map(([x, y]) => [x + 10, y - 5])))
  );
  fixture.writePositions(createFlatGeometry(shifted).positions);
  const moved = await fixture.run();
  const movedExpected = shifted.map(feature =>
    measureFeature(feature, {isPolygon: true, coordinateSystem: 'planar'})
  );
  expectFeatureParity(
    moved.features,
    movedExpected,
    {length: 1e-6, area: 1e-5, position: 1e-3},
    true
  );
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});
