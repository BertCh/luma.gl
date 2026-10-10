// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUGreatCircleArcsParameterValues,
  getGPULineSegmentizeParameterValues,
  GPUGreatCircleArcs,
  GPULineSegmentize
} from '../../../src/gpu-spatial-analysis/line-segmentize';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getCentralAngle} from '../geometry-measures/geodesic-oracle';
import {
  createFlatPaths,
  createRandom,
  segmentizePaths,
  tessellateGreatCircleArcs,
  type FlatPaths
} from './line-segmentize-oracle';
import {
  createLinePathFixture,
  expectLinePathParity,
  getSphericalErrorMeters
} from './line-path-test-utils';

const EARTH_RADIUS = 6371008.8;
// One f32 ulp of a longitude near 180 degrees is about 1.7 m on the equator; f32 unit vectors add
// about 1e-7 rad (0.6 m). Errors are great-circle distances to the f64 slerp.
const SPHERICAL_TOLERANCE_METERS = 5;
// f32 unit vectors fix the great circle to about 1e-7 / sin(angle) rad, so arcs that are far from
// antipodal (angle below 2.6 rad, about 16,500 km) stay within a few meters of the f64 slerp.
const GLOBAL_ARC_TOLERANCE_METERS = 10;
const MAXIMUM_TEST_ARC_ANGLE = 2.6;

function createSegmentizeFixture(
  device: Device,
  paths: FlatPaths,
  options: {capacity: number; spherical?: boolean; maximumPieces?: number}
) {
  return createLinePathFixture(device, {
    inputs: {
      positions: {values: paths.positions, format: 'float32x2'},
      pathOffsets: {values: paths.pathOffsets, format: 'uint32'}
    },
    capacity: options.capacity,
    pathCapacity: paths.pathOffsets.length - 1,
    createContributor: (inputs, parameters, output) =>
      new GPULineSegmentize({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        pathOffsets: inputs['pathOffsets'] as GraphDataView<'uint32'>,
        coordinateSystem: options.spherical ? 'spherical' : 'planar',
        maximumPiecesPerSegment: options.maximumPieces,
        parameters,
        output
      })
  });
}

function createRandomPlanarPaths(seed: number, pathCount: number): FlatPaths {
  const random = createRandom(seed);
  const paths = Array.from({length: pathCount}, (_, pathIndex) => {
    // Include empty and single-vertex paths.
    const length =
      pathIndex % 17 === 3 ? 0 : pathIndex % 13 === 5 ? 1 : 2 + Math.floor(random() * 8);
    let x = random() * 100;
    let y = random() * 100;
    return Array.from({length}, (__, vertexIndex) => {
      if (vertexIndex > 0 && random() > 0.1) {
        x += (random() - 0.5) * 20;
        y += (random() - 0.5) * 20;
      }
      return [x, y];
    });
  });
  return createFlatPaths(paths);
}

it('GPULineSegmentize densifies planar paths like the f64 oracle and changes length per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createRandomPlanarPaths(3, 400);
  const capacity = 40000;
  const fixture = createSegmentizeFixture(device, paths, {
    capacity,
    maximumPieces: 64
  });
  for (const maximumSegmentLength of [0, 2.718281828, 0.7071067, 13.37, 0.01]) {
    const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
    const expected = segmentizePaths(paths, {
      maximumSegmentLength,
      maximumPieces: 64
    });
    expectLinePathParity(actual, expected, capacity, 2e-5 * 100, 1e-5);
    expect(actual.pathCount).toBe(400);
    expect(actual.sourcePaths).toEqual(Array.from({length: 400}, (_, index) => index));
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineSegmentize clamps path offsets and reports overflow at capacity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createRandomPlanarPaths(5, 60);
  const capacity = 300;
  const fixture = createSegmentizeFixture(device, paths, {capacity});
  const maximumSegmentLength = 0.5;
  const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
  const expected = segmentizePaths(paths, {
    maximumSegmentLength,
    maximumPieces: 1024
  });
  expect(expected.positions.length).toBeGreaterThan(capacity);
  expectLinePathParity(actual, expected, capacity, 2e-3, 1e-5);
  expect(actual.overflow).toBe(1);
  const relaxed = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength: 0}));
  expect(relaxed.overflow).toBe(0);
  expect(relaxed.count).toBe(paths.positions.length / 2);
  fixture.destroy();
});

it('GPULineSegmentize slerps spherical paths with continuous longitudes across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createFlatPaths([
    // Pacific crossing, west to east across 180.
    [
      [170, 10],
      [-175, 20],
      [-160, 15],
      [175, 5]
    ],
    // High latitude.
    [
      [-30, 70],
      [60, 80]
    ],
    // Short city-scale path (meters).
    [
      [2.35, 48.85],
      [2.351, 48.8505],
      [2.3525, 48.851]
    ],
    [],
    [[10, 10]]
  ]);
  const capacity = 4096;
  const fixture = createSegmentizeFixture(device, paths, {
    capacity,
    spherical: true
  });
  let maximumError = 0;
  for (const maximumSegmentLength of [100000, 333333, 25]) {
    const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
    const expected = segmentizePaths(paths, {
      maximumSegmentLength,
      maximumPieces: 1024,
      spherical: true,
      radius: EARTH_RADIUS
    });
    maximumError = Math.max(
      maximumError,
      expectLinePathParity(
        actual,
        expected,
        capacity,
        SPHERICAL_TOLERANCE_METERS,
        1e-5,
        getSphericalErrorMeters
      )
    );
    // Continuity: no jump of more than 180 degrees inside a path.
    for (let path = 0; path < 5; path++) {
      for (let row = actual.pathOffsets[path] + 1; row < actual.pathOffsets[path + 1]; row++) {
        expect(Math.abs(actual.positions[row][0] - actual.positions[row - 1][0])).toBeLessThan(180);
      }
    }
  }
  console.log(`GPULineSegmentize spherical max error ${maximumError} m`);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPUGreatCircleArcs matches the f64 slerp for origin/destination pairs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(11);
  const sources: number[][] = [
    [139.69, 35.69], // Tokyo
    [-0.1276, 51.5072], // London
    [0, 0],
    [12, 41],
    [-170, -60]
  ];
  const targets: number[][] = [
    [-122.42, 37.77], // San Francisco: crosses the antimeridian
    [-74.006, 40.7128], // New York
    [0, 0], // zero length
    [12.0001, 41.0001], // ~14 m
    [170, -65]
  ];
  while (sources.length < 300) {
    const source = [random() * 360 - 180, random() * 160 - 80];
    const target = [random() * 360 - 180, random() * 160 - 80];
    if (getCentralAngle(source, target) < MAXIMUM_TEST_ARC_ANGLE) {
      sources.push(source);
      targets.push(target);
    }
  }
  const f32 = (values: number[][]) => values.map(([x, y]) => [Math.fround(x), Math.fround(y)]);
  const sourceValues = new Float32Array(sources.flat());
  const targetValues = new Float32Array(targets.flat());
  const capacity = 300 * 66;
  const maximumSegments = 64;
  const fixture = createLinePathFixture(device, {
    inputs: {
      sources: {values: sourceValues, format: 'float32x2'},
      targets: {values: targetValues, format: 'float32x2'}
    },
    capacity,
    pathCapacity: sources.length,
    createContributor: (inputs, parameters, output) =>
      new GPUGreatCircleArcs({
        sources: inputs['sources'] as GraphDataView<'float32x2'>,
        targets: inputs['targets'] as GraphDataView<'float32x2'>,
        maximumSegments,
        parameters,
        output
      })
  });
  let maximumError = 0;
  for (const [maximumSegmentLength, minimumSegments] of [
    [500000, 1],
    [0, 16],
    [1234567, 4]
  ]) {
    const actual = await fixture.run(
      getGPUGreatCircleArcsParameterValues({
        maximumSegmentLength,
        minimumSegments
      })
    );
    const expected = tessellateGreatCircleArcs(f32(sources), f32(targets), {
      maximumSegmentLength,
      minimumSegments,
      maximumSegments,
      radius: EARTH_RADIUS
    });
    maximumError = Math.max(
      maximumError,
      expectLinePathParity(
        actual,
        expected,
        capacity,
        GLOBAL_ARC_TOLERANCE_METERS,
        2e-5,
        getSphericalErrorMeters
      )
    );
    // Tokyo -> San Francisco ends at the unwrapped longitude.
    const tokyoEnd = actual.positions[actual.pathOffsets[1] - 1];
    expect(tokyoEnd[0]).toBeCloseTo(-122.42 + 360, 3);
  }
  console.log(`GPUGreatCircleArcs max error ${maximumError} m`);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPUGreatCircleArcs reports overflow and clamps offsets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const sources = new Float32Array([0, 0, 10, 10, 20, 20]);
  const targets = new Float32Array([90, 0, 10, 50, -160, 20]);
  const capacity = 20;
  const fixture = createLinePathFixture(device, {
    inputs: {
      sources: {values: sources, format: 'float32x2'},
      targets: {values: targets, format: 'float32x2'}
    },
    capacity,
    pathCapacity: 3,
    createContributor: (inputs, parameters, output) =>
      new GPUGreatCircleArcs({
        sources: inputs['sources'] as GraphDataView<'float32x2'>,
        targets: inputs['targets'] as GraphDataView<'float32x2'>,
        parameters,
        output
      })
  });
  const actual = await fixture.run(
    getGPUGreatCircleArcsParameterValues({
      maximumSegmentLength: 0,
      minimumSegments: 8
    })
  );
  expect(actual.requiredCount).toBe(27);
  expect(actual.count).toBe(20);
  expect(actual.overflow).toBe(1);
  expect(actual.pathOffsets).toEqual([0, 9, 18, 20]);
  fixture.destroy();
});

it('GPULineSegmentize matches the oracle for long paths and one segment cut into many pieces', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(31);
  const walk = (length: number) => {
    let x = random() * 50;
    let y = random() * 50;
    return Array.from({length}, () => {
      x += (random() - 0.5) * 8;
      y += (random() - 0.5) * 8;
      return [x, y];
    });
  };
  // Row counts around the 64-row prefix tile, one 2000-row path, and a 2-vertex path of length
  // 5000 that is cut into 1000 pieces by the segment length below.
  const paths = createFlatPaths([
    walk(63),
    walk(64),
    walk(65),
    walk(2000),
    [
      [0, 0],
      [5000, 0]
    ],
    walk(3)
  ]);
  const capacity = 60000;
  const fixture = createSegmentizeFixture(device, paths, {capacity, maximumPieces: 1024});
  for (const maximumSegmentLength of [5, 0, 41.3]) {
    const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
    const expected = segmentizePaths(paths, {maximumSegmentLength, maximumPieces: 1024});
    expectLinePathParity(actual, expected, capacity, 2e-3, 2e-5);
  }
  fixture.destroy();
});
