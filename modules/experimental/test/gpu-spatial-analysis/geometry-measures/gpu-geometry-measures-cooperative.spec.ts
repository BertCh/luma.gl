// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
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
import {createFlatGeometry, type NestedFeatures} from './geometry-measures-oracle';

type Readback = {
  lengths: number[];
  areas: number[];
  signedAreas: number[];
  centroids: number[];
  bounds: number[];
  vertexCounts: number[];
  extremes: number[];
};

/** Runs one measures graph and reads every column of the feature output. */
async function measure(
  device: Device,
  features: NestedFeatures,
  options: Pick<GPUGeometryMeasuresProps, 'geometryType' | 'coordinateSystem' | 'holeRule'>,
  cooperativeRingRows: number
): Promise<Readback> {
  const flat = createFlatGeometry(features);
  const count = features.length;
  const isPolygon = options.geometryType === 'polygons';
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'measures-cooperative'});
  const lengths = track(createOutputBuffer(device, count));
  const areas = track(createOutputBuffer(device, count));
  const signedAreas = track(createOutputBuffer(device, count));
  const centroids = track(createOutputBuffer(device, 2 * count));
  const bounds = track(createOutputBuffer(device, 4 * count));
  const vertexCounts = track(createOutputBuffer(device, count));
  const extremes = track(createOutputBuffer(device, 4 * count));
  graph.add(
    new GPUGeometryMeasures({
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, flat.positions)),
        'float32x2',
        flat.positions.length / 2
      ),
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
      ...options,
      cooperativeRingRows,
      output: {
        lengths: importGraphBuffer(graph, 'lengths', lengths, 'float32', count),
        ...(isPolygon
          ? {
              areas: importGraphBuffer(graph, 'areas', areas, 'float32', count),
              signedAreas: importGraphBuffer(graph, 'signed-areas', signedAreas, 'float32', count)
            }
          : {}),
        centroids: importGraphBuffer(graph, 'centroids', centroids, 'float32x2', count),
        bounds: importGraphBuffer(graph, 'bounds', bounds, 'float32x4', count),
        vertexCounts: importGraphBuffer(graph, 'vertex-counts', vertexCounts, 'uint32', count),
        extremeVertices: importGraphBuffer(graph, 'extremes', extremes, 'uint32x4', count)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    lengths: await readFloat32(lengths, count),
    areas: await readFloat32(areas, count),
    signedAreas: await readFloat32(signedAreas, count),
    centroids: await readFloat32(centroids, 2 * count),
    bounds: await readFloat32(bounds, 4 * count),
    vertexCounts: await readUint32(vertexCounts, count),
    extremes: await readUint32(extremes, 4 * count)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** Star-shaped ring of `count` vertices around a center, counter-clockwise, with radial jitter. */
function createRing(
  random: () => number,
  count: number,
  center: [number, number],
  radiusX: number,
  radiusY: number
): number[][] {
  return Array.from({length: count}, (_, index) => {
    const angle = (2 * Math.PI * index) / count;
    const radius = 0.7 + 0.3 * random();
    return [
      Math.fround(center[0] + radius * radiusX * Math.cos(angle)),
      Math.fround(center[1] + radius * radiusY * Math.sin(angle))
    ];
  });
}

function expectParity(
  actual: Readback,
  serial: Readback,
  label: string,
  isPolygon: boolean,
  /** Bounds are exact for planar input; unwrapped longitudes differ by f32 summation order. */
  boundsTolerance = 0
): void {
  const close = (a: number, b: number, scale: number, name: string) =>
    expect(Math.abs(a - b), `${label} ${name}: ${a} vs ${b}`).toBeLessThanOrEqual(
      1e-5 * Math.max(scale, Math.abs(b))
    );
  expect(actual.vertexCounts, label).toEqual(serial.vertexCounts);
  expect(actual.extremes, `${label} extreme vertices`).toEqual(serial.extremes);
  serial.bounds.forEach((bound, index) =>
    expect(Math.abs(actual.bounds[index] - bound), `${label} bounds ${index}`).toBeLessThanOrEqual(
      boundsTolerance
    )
  );
  serial.lengths.forEach((length, feature) => {
    close(actual.lengths[feature], length, 1e-3, `length ${feature}`);
    if (isPolygon) {
      close(actual.areas[feature], serial.areas[feature], 1e-3, `area ${feature}`);
      close(actual.signedAreas[feature], serial.signedAreas[feature], 1e-3, `signed ${feature}`);
    }
    const extent = Math.max(
      Math.abs(serial.bounds[4 * feature + 2] - serial.bounds[4 * feature]),
      Math.abs(serial.bounds[4 * feature + 3] - serial.bounds[4 * feature + 1]),
      1e-3
    );
    for (const axis of [0, 1]) {
      expect(
        Math.abs(actual.centroids[2 * feature + axis] - serial.centroids[2 * feature + axis]),
        `${label} centroid ${feature}.${axis}`
      ).toBeLessThanOrEqual(2e-5 * extent + 1e-6 * Math.abs(serial.centroids[2 * feature + axis]));
    }
  });
}

/** A mix of ring sizes around the 64-lane chunking, several rings per feature, and small features. */
function createPlanarFeatures(random: () => number): NestedFeatures {
  const features: NestedFeatures = [];
  for (const sizes of [
    [3],
    [7, 5],
    [64],
    [65],
    [127, 3, 129],
    [300],
    [4, 300, 5, 71],
    [9, 9, 9],
    [1024]
  ]) {
    features.push(
      sizes.map((size, ring) =>
        createRing(
          random,
          size,
          [1000 + 40 * features.length, -500 + 10 * ring],
          10 + 5 * ring,
          7 + 3 * ring
        )
      )
    );
  }
  return features;
}

it('GPUGeometryMeasures cooperative rings match the serial walk (planar polygons, lines, points)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features = createPlanarFeatures(createRandom(5));
  for (const options of [
    {geometryType: 'polygons' as const},
    {geometryType: 'polygons' as const, holeRule: 'first-ring-exterior' as const},
    {geometryType: 'lines' as const},
    {geometryType: 'points' as const}
  ]) {
    const serial = await measure(device, features, options, 0);
    // A threshold of 6 sends every feature with a ring above 6 vertices down the cooperative path.
    const cooperative = await measure(device, features, options, 6);
    expectParity(
      cooperative,
      serial,
      `${options.geometryType} ${options.holeRule ?? ''}`,
      options.geometryType === 'polygons'
    );
  }
  device.destroy?.();
});

it('GPUGeometryMeasures cooperative rings unwrap longitudes across chunk boundaries', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(11);
  // A 140-degree-wide ring centered on the antimeridian: longitudes unwrap from the first vertex
  // across the 180 degree seam, and chunks of the ring start on both sides of it.
  const features: NestedFeatures = [
    [createRing(random, 700, [180, 25], 70, 12).map(([x, y]) => [((x + 540) % 360) - 180, y])],
    [createRing(random, 200, [10, -30], 40, 15)],
    [
      createRing(random, 150, [-170, 60], 30, 8).map(([x, y]) => [((x + 540) % 360) - 180, y]),
      createRing(random, 90, [-175, 60], 8, 3).map(([x, y]) => [((x + 540) % 360) - 180, y])
    ]
  ];
  for (const coordinateSystem of ['spherical', 'wgs84', 'geodesic'] as const) {
    for (const geometryType of ['polygons', 'lines'] as const) {
      const options = {geometryType, coordinateSystem};
      const serial = await measure(device, features, options, 0);
      const cooperative = await measure(device, features, options, 6);
      expectParity(
        cooperative,
        serial,
        `${coordinateSystem} ${geometryType}`,
        geometryType === 'polygons',
        2e-4
      );
    }
  }
  device.destroy?.();
});

it('GPUGeometryMeasures measures a 6000-vertex ring cooperatively against a closed form', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const count = 6000;
  const radius = 250;
  const ring = Array.from({length: count}, (_, index) => [
    Math.fround(40000 + radius * Math.cos((2 * Math.PI * index) / count)),
    Math.fround(-9000 + radius * Math.sin((2 * Math.PI * index) / count))
  ]);
  // One huge ring among small features, as in the skewed case the cooperative path targets.
  const features: NestedFeatures = [[ring.slice(0, 4)], [ring], [ring.slice(10, 18)]];
  const result = await measure(device, features, {geometryType: 'polygons'}, 512);
  const expectedArea = 0.5 * count * radius * radius * Math.sin((2 * Math.PI) / count);
  const expectedPerimeter = 2 * count * radius * Math.sin(Math.PI / count);
  expect(Math.abs(result.areas[1] - expectedArea) / expectedArea).toBeLessThan(2e-5);
  expect(Math.abs(result.lengths[1] - expectedPerimeter) / expectedPerimeter).toBeLessThan(2e-5);
  expect(result.vertexCounts).toEqual([4, count, 8]);
  expect(Math.abs(result.centroids[2] - 40000)).toBeLessThan(0.05);
  expect(Math.abs(result.centroids[3] + 9000)).toBeLessThan(0.05);
  // The first row with the smallest x and the largest y wins ties: rows are in the feature's range.
  const serial = await measure(device, features, {geometryType: 'polygons'}, 0);
  expect(result.extremes).toEqual(serial.extremes);
  device.destroy?.();
});
