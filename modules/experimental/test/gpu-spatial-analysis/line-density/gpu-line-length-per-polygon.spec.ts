// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerPolygonConsumerGPUTests} from '../zonal-statistics/polygon-consumer-status-fixtures';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {
  createGeometryFixture,
  createRandom,
  expectClose
} from '../outline-geometry/geometry-fixture';
import type {Point} from './line-density-oracle';
import {
  computeLineLengthPerPolygon,
  type OraclePolygonFeature
} from './line-length-per-polygon-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

function flattenLines(lines: Point[][]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const line of lines) {
    for (const [x, y] of line) {
      positions.push(x, y);
    }
    offsets.push(positions.length / 2);
  }
  return {positions: new Float32Array(positions), offsets: new Uint32Array(offsets)};
}

function flattenPolygons(features: OraclePolygonFeature[]) {
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  for (const feature of features) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const [x, y] of ring) {
          positions.push(x, y);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: new Float32Array(positions),
    ringOffsets: new Uint32Array(ringOffsets),
    polygonOffsets: new Uint32Array(polygonOffsets),
    featureOffsets: new Uint32Array(featureOffsets)
  };
}

function box(x0: number, y0: number, x1: number, y1: number): Point[] {
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1]
  ];
}

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

function createFixture(
  device: Device,
  lines: Point[][],
  features: OraclePolygonFeature[],
  options: {
    weights?: number[];
    spherical?: boolean;
    units?: 'meters' | 'kilometers';
    pairCapacity?: number;
    maximumCrossingCount?: number;
  } = {}
) {
  const flat = flattenLines(lines);
  const polygons = flattenPolygons(features);
  return createGeometryFixture(device, {
    inputs: {
      positions: {values: flat.positions, format: 'float32x2'},
      lineOffsets: {values: flat.offsets, format: 'uint32'},
      ...(options.weights
        ? {lineWeights: {values: new Float32Array(options.weights), format: 'float32' as const}}
        : {}),
      polygonPositions: {values: polygons.positions, format: 'float32x2'},
      featureOffsets: {values: polygons.featureOffsets, format: 'uint32'},
      polygonOffsets: {values: polygons.polygonOffsets, format: 'uint32'},
      ringOffsets: {values: polygons.ringOffsets, format: 'uint32'}
    },
    outputs: {
      lengths: {format: 'float32', length: features.length},
      weightedLengths: {format: 'float32', length: features.length},
      segmentCounts: {format: 'uint32', length: features.length},
      overflow: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) =>
      new GPULineLengthPerPolygon({
        positions: inputs['positions'] as never,
        lineOffsets: inputs['lineOffsets'] as never,
        lineWeights: inputs['lineWeights'] as never,
        polygons: {
          kind: 'polygons',
          positions: inputs['polygonPositions'] as never,
          featureOffsets: inputs['featureOffsets'] as never,
          polygonOffsets: inputs['polygonOffsets'] as never,
          ringOffsets: inputs['ringOffsets'] as never
        },
        spatialContext: options.spherical
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: options.units}
          : undefined,
        pairCapacity: options.pairCapacity,
        maximumCrossingCount: options.maximumCrossingCount,
        output: {
          lengths: outputs['lengths'] as never,
          weightedLengths: options.weights ? (outputs['weightedLengths'] as never) : undefined,
          segmentCounts: outputs['segmentCounts'] as never,
          status: {
            overflow: outputs['overflow'] as never
          }
        }
      })
  });
}

it('GPULineLengthPerPolygon clips lines against shells, holes and multipolygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features: OraclePolygonFeature[] = [
    [[box(0, 0, 10, 10), box(4, 4, 6, 6).reverse()]],
    [[box(20, 0, 30, 10)], [box(40, 0, 50, 10)]],
    [[box(100, 100, 110, 110)]]
  ];
  const lines: Point[][] = [
    [
      [-5, 5],
      [15, 5]
    ],
    [
      [25, -5],
      [25, 15],
      [45, 15],
      [45, 5]
    ],
    [
      [2, 2],
      [3, 3],
      [3, 8]
    ]
  ];
  const fixture = createFixture(device, lines, features, {weights: [1, 2, 3]});
  const result = await fixture.run();
  expect(result['overflow'][0]).toBe(0);
  // Line 0 inside box 0: 10 minus the 2-wide hole; line 2 fully inside, outside the hole.
  expectClose(result['lengths'][0], 8 + Math.SQRT2 + 5, 1e-5, 1e-5);
  expectClose(result['lengths'][1], 10 + 5, 1e-5, 1e-5);
  expect(result['lengths'][2]).toBe(0);
  expectClose(result['weightedLengths'][0], 8 * 1 + (Math.SQRT2 + 5) * 3, 1e-5, 1e-5);
  expectClose(result['weightedLengths'][1], 10 * 2 + 5 * 2, 1e-5, 1e-5);
  expect(result['segmentCounts']).toEqual([3, 2, 0]);
  fixture.destroy();
});

it('GPULineLengthPerPolygon matches the f64 oracle on random lines and polygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(11);
  const extent = 100;
  const features: OraclePolygonFeature[] = Array.from({length: 40}, (_, index) => {
    const x = random() * extent * 0.8;
    const y = random() * extent * 0.8;
    const size = 5 + random() * 20;
    const outer = box(x, y, x + size, y + size);
    const rings = [outer];
    if (index % 3 === 0) {
      rings.push(box(x + size * 0.3, y + size * 0.3, x + size * 0.6, y + size * 0.7).reverse());
    }
    const feature: OraclePolygonFeature = [rings];
    if (index % 5 === 0) {
      feature.push([box(x + size + 2, y, x + size + 8, y + 6)]);
    }
    return feature;
  });
  const lines: Point[][] = Array.from({length: 150}, () => {
    let x = random() * extent;
    let y = random() * extent;
    return Array.from({length: 2 + Math.floor(random() * 5)}, () => {
      x += (random() - 0.5) * 40;
      y += (random() - 0.5) * 40;
      return [Math.fround(x), Math.fround(y)] as Point;
    });
  });
  const weights = lines.map(() => Math.fround(0.5 + random() * 3));
  const fixture = createFixture(device, lines, features, {weights});
  const gpuStart = performance.now();
  const result = await fixture.run();
  const gpuMs = performance.now() - gpuStart;
  const gpuStartSecond = performance.now();
  await fixture.run();
  const gpuSecondMs = performance.now() - gpuStartSecond;
  const cpuStart = performance.now();
  const rounded = features.map(feature =>
    feature.map(polygon => polygon.map(ring => ring.map(([x, y]) => [x, y] as Point)))
  );
  const expected = computeLineLengthPerPolygon(
    lines,
    weights.map(value => Math.fround(value)),
    rounded
  );
  const cpuMs = performance.now() - cpuStart;
  // eslint-disable-next-line no-console
  console.log(
    `GPULineLengthPerPolygon timing (150 lines, 40 polygons): gpu first ${gpuMs.toFixed(1)} ms, gpu second ${gpuSecondMs.toFixed(1)} ms, cpu oracle ${cpuMs.toFixed(1)} ms`
  );
  expect(result['overflow'][0]).toBe(0);
  expect(Math.max(...result['lengths'])).toBeGreaterThan(10);
  for (let feature = 0; feature < features.length; feature++) {
    expectClose(
      result['lengths'][feature],
      expected.lengths[feature],
      5e-4,
      5e-4,
      `length ${feature}`
    );
    expectClose(
      result['weightedLengths'][feature],
      expected.weightedLengths[feature],
      5e-4,
      5e-4,
      `weighted ${feature}`
    );
    expect(result['segmentCounts'][feature], `count ${feature}`).toBe(
      expected.segmentCounts[feature]
    );
  }
  // Reproducible bit for bit.
  const again = await fixture.run();
  expect(again['lengths']).toEqual(result['lengths']);
  fixture.destroy();
});

it('GPULineLengthPerPolygon flags crossing and candidate overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A comb polygon crossed by one long line has many crossings.
  const teeth: Point[] = [[0, 0]];
  for (let tooth = 0; tooth < 6; tooth++) {
    teeth.push([tooth * 2 + 1, 10], [tooth * 2 + 2, 0]);
  }
  teeth.push([12, -5], [0, -5]);
  const features: OraclePolygonFeature[] = [[[teeth]]];
  const lines: Point[][] = [
    [
      [-1, 5],
      [13, 5]
    ]
  ];
  const tight = createFixture(device, lines, features, {maximumCrossingCount: 2});
  expect((await tight.run())['overflow'][0]).toBe(1);
  tight.destroy();
  const roomy = createFixture(device, lines, features, {maximumCrossingCount: 32});
  const result = await roomy.run();
  expect(result['overflow'][0]).toBe(0);
  const expected = computeLineLengthPerPolygon(lines, undefined, features);
  expectClose(result['lengths'][0], expected.lengths[0], 1e-5, 1e-5);
  expect(result['lengths'][0]).toBeGreaterThan(1);
  roomy.destroy();
  const small = createFixture(
    device,
    [
      ...lines,
      [
        [0, 2],
        [12, 2]
      ]
    ],
    features,
    {pairCapacity: 1}
  );
  expect((await small.run())['overflow'][0]).toBe(1);
  small.destroy();
});

it('GPULineLengthPerPolygon measures spherical pieces as great-circle distance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features: OraclePolygonFeature[] = [[[box(0, 0, 10, 10)]]];
  const lines: Point[][] = [
    [
      [-5, 5],
      [15, 5]
    ]
  ];
  const fixture = createFixture(device, lines, features, {spherical: true});
  const result = await fixture.run();
  const expected = computeLineLengthPerPolygon(lines, undefined, features, true);
  expectClose(result['lengths'][0], expected.lengths[0], 1e-4, 1);
  expect(result['lengths'][0]).toBeGreaterThan(1e6);
  fixture.destroy();
});

it('GPULineLengthPerPolygon honours kilometers units on the default sphere', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features: OraclePolygonFeature[] = [[[box(0, 0, 10, 10)]]];
  const lines: Point[][] = [
    [
      [-5, 5],
      [15, 5]
    ]
  ];
  const fixture = createFixture(device, lines, features, {spherical: true, units: 'kilometers'});
  const result = await fixture.run();
  // The oracle measures meters on the same default sphere.
  const expected = computeLineLengthPerPolygon(lines, undefined, features, true);
  expectClose(result['lengths'][0], expected.lengths[0] / 1000, 1e-4, 1e-3);
  fixture.destroy();
});

registerPolygonConsumerGPUTests('GPULineLengthPerPolygon');
