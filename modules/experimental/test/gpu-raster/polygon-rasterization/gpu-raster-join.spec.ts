// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUPolygonRasterizationExtentValues,
  GPURasterJoin
} from '../../../src/gpu-raster/polygon-rasterization';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  findContainingFeature,
  joinPointsOnCPU,
  NO_ZONE,
  type OracleRasterJoin,
  type OraclePolygons
} from './polygon-rasterization-oracle';
import {
  createRasterizationFixture,
  destroyRasterizationFixture,
  readRasterization,
  type RasterizationFixture
} from './polygon-rasterization-fixture';
import {createRandom, createRandomScene} from './polygon-rasterization-scenes';

type Extent = readonly [number, number, number, number];

type JoinBuffers = {
  points: Buffer;
  values: Buffer;
  counts: Buffer;
  sums: Buffer;
  boundaryCounts: Buffer;
  unassigned: Buffer;
  outside: Buffer;
  pointZones: Buffer;
  pointBoundaryMask: Buffer;
};

/** Adds a `GPURasterJoin` with every output to `graph` and returns its buffers. */
function addJoin(
  device: Device,
  graph: GPUCommandGraph,
  props: {
    width: number;
    height: number;
    extent: GraphDataView<'float32'>;
    zones: GraphDataView<'uint32'>;
    boundary: GraphDataView<'uint32'>;
    zoneCount: number;
    points: Float32Array;
    values: Float32Array;
  }
): JoinBuffers {
  const pointCount = props.points.length / 2;
  const buffers: JoinBuffers = {
    points: createInputBuffer(device, props.points),
    values: createInputBuffer(device, props.values),
    counts: createOutputBuffer(device, props.zoneCount),
    sums: createOutputBuffer(device, props.zoneCount),
    boundaryCounts: createOutputBuffer(device, props.zoneCount),
    unassigned: createOutputBuffer(device, 1),
    outside: createOutputBuffer(device, 1),
    pointZones: createOutputBuffer(device, pointCount),
    pointBoundaryMask: createOutputBuffer(device, pointCount)
  };
  const zoneColumn = (name: keyof JoinBuffers, format: 'uint32' | 'float32') =>
    importGraphBuffer(graph, `join-${name}`, buffers[name], format, props.zoneCount);
  graph.add(
    new GPURasterJoin({
      width: props.width,
      height: props.height,
      extent: props.extent,
      points: importGraphBuffer(graph, 'join-points', buffers.points, 'float32x2', pointCount),
      values: importGraphBuffer(graph, 'join-values', buffers.values, 'float32', pointCount),
      zones: props.zones,
      boundary: props.boundary,
      zoneCount: props.zoneCount,
      output: {
        counts: zoneColumn('counts', 'uint32') as GraphDataView<'uint32'>,
        sums: zoneColumn('sums', 'float32') as GraphDataView<'float32'>,
        boundaryCounts: zoneColumn('boundaryCounts', 'uint32') as GraphDataView<'uint32'>,
        unassignedBoundaryCount: importGraphBuffer(
          graph,
          'join-unassigned',
          buffers.unassigned,
          'uint32',
          1
        ),
        outsideCount: importGraphBuffer(graph, 'join-outside', buffers.outside, 'uint32', 1),
        pointZones: importGraphBuffer(
          graph,
          'join-point-zones',
          buffers.pointZones,
          'uint32',
          pointCount
        ),
        pointBoundaryMask: importGraphBuffer(
          graph,
          'join-point-boundary',
          buffers.pointBoundaryMask,
          'uint32',
          pointCount
        )
      }
    })
  );
  return buffers;
}

async function expectJoinMatches(
  buffers: JoinBuffers,
  expected: OracleRasterJoin,
  zoneCount: number
): Promise<number[]> {
  const pointCount = expected.pointZones.length;
  expect(await readUint32(buffers.pointZones, pointCount)).toEqual(Array.from(expected.pointZones));
  expect(await readUint32(buffers.pointBoundaryMask, pointCount)).toEqual(
    Array.from(expected.pointBoundaryMask)
  );
  expect(await readUint32(buffers.counts, zoneCount)).toEqual(expected.counts);
  expect(await readUint32(buffers.boundaryCounts, zoneCount)).toEqual(expected.boundaryCounts);
  expect((await readUint32(buffers.unassigned, 1))[0]).toBe(expected.unassignedBoundaryCount);
  expect((await readUint32(buffers.outside, 1))[0]).toBe(expected.outsideCount);
  const sums = await readFloat32(buffers.sums, zoneCount);
  for (const [zone, sum] of expected.sums.entries()) {
    expect(Math.abs(sums[zone] - sum)).toBeLessThanOrEqual(Math.max(1e-3, Math.abs(sum) * 1e-5));
  }
  return sums;
}

function destroyJoinBuffers(buffers: JoinBuffers): void {
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
}

/** Random points inside and around `extent`, plus non-finite rows, avoiding cell borders. */
function createPoints(
  seed: number,
  pointCount: number,
  width: number,
  height: number,
  extent: Extent
): {points: Float32Array; values: Float32Array} {
  const random = createRandom(seed);
  const points = new Float32Array(pointCount * 2);
  const values = new Float32Array(pointCount);
  const [originX, originY, cellWidth, cellHeight] = extent;
  for (let point = 0; point < pointCount; point++) {
    const column = Math.floor(random() * (width + 4)) - 2;
    const row = Math.floor(random() * (height + 4)) - 2;
    points[point * 2] = originX + (column + 0.05 + 0.9 * random()) * cellWidth;
    points[point * 2 + 1] = originY + (row + 0.05 + 0.9 * random()) * cellHeight;
    values[point] = random() < 0.05 ? Number.NaN : Math.round((random() * 20 - 5) * 8) / 8;
  }
  points[0] = Number.NaN;
  points[3] = Number.POSITIVE_INFINITY;
  return {points, values};
}

it('GPURasterJoin matches the CPU raster join on a given zone raster, deterministically', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 23;
  const height = 17;
  const zoneCount = 9;
  const extent: Extent = [-3, 2, 0.5, 0.25];
  const random = createRandom(5);
  const zones = Uint32Array.from({length: width * height}, () => {
    const roll = random();
    // Mostly zones 0..8 (zone 8 stays empty), some no-zone cells and out-of-range IDs.
    return roll < 0.1 ? NO_ZONE : roll < 0.15 ? 12 : Math.floor(random() * 8);
  });
  const boundary = Uint32Array.from({length: width * height}, () => (random() < 0.3 ? 1 : 0));
  const {points, values} = createPoints(9, 5000, width, height, extent);

  const graph = new GPUCommandGraph(device, {id: 'raster-join-test'});
  const zonesBuffer = createInputBuffer(device, zones);
  const boundaryBuffer = createInputBuffer(device, boundary);
  const extentBuffer = new GPUParameterBuffer(device, {
    id: 'extent',
    format: 'float32',
    length: 4,
    values: getGPUPolygonRasterizationExtentValues(...extent)
  });
  const buffers = addJoin(device, graph, {
    width,
    height,
    extent: extentBuffer.importToGraph(graph),
    zones: importGraphBuffer(graph, 'zones', zonesBuffer, 'uint32', width * height),
    boundary: importGraphBuffer(graph, 'boundary', boundaryBuffer, 'uint32', width * height),
    zoneCount,
    points,
    values
  });
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const expected = joinPointsOnCPU({
    raster: {width, height, extent},
    zones,
    boundary,
    zoneCount,
    points,
    values
  });
  expect(expected.counts[8]).toBe(0);
  expect(expected.outsideCount).toBeGreaterThan(100);
  expect(expected.unassignedBoundaryCount).toBeGreaterThan(0);
  const firstSums = await expectJoinMatches(buffers, expected, zoneCount);
  expect(firstSums[8]).toBe(0);

  // Same compiled graph: bitwise identical sums.
  submitGraph(device, compiled, undefined);
  const secondSums = await readFloat32(buffers.sums, zoneCount);
  expect(new Uint32Array(Float32Array.from(secondSums).buffer)).toEqual(
    new Uint32Array(Float32Array.from(firstSums).buffer)
  );
  compiled.destroy();
  destroyJoinBuffers(buffers);
  zonesBuffer.destroy();
  boundaryBuffer.destroy();
  extentBuffer.destroy();
});

/** Checks join outputs against the raster oracle and exact point-in-polygon. */
async function expectEndToEnd(
  fixture: RasterizationFixture,
  buffers: JoinBuffers,
  polygons: OraclePolygons,
  props: {points: Float32Array; values: Float32Array; extent: Extent; zoneCount: number}
): Promise<void> {
  const {width, height} = fixture;
  const rasterization = await readRasterization(fixture);
  expect(rasterization.overflow).toBe(0);
  const expected = joinPointsOnCPU({
    raster: {width, height, extent: props.extent},
    zones: Uint32Array.from(rasterization.zones),
    boundary: rasterization.boundary,
    zoneCount: props.zoneCount,
    points: props.points,
    values: props.values
  });
  await expectJoinMatches(buffers, expected, props.zoneCount);

  // Exact point-in-polygon agrees on every interior point, and the boundary counts bound the error.
  const exactCounts = new Array(props.zoneCount).fill(0);
  let interiorPointCount = 0;
  const pointCount = props.points.length / 2;
  for (let point = 0; point < pointCount; point++) {
    const x = props.points[point * 2];
    const y = props.points[point * 2 + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      continue;
    }
    const column = Math.floor(Math.fround(Math.fround(x - props.extent[0]) / props.extent[2]));
    const row = Math.floor(Math.fround(Math.fround(y - props.extent[1]) / props.extent[3]));
    const isInsideRaster = column >= 0 && row >= 0 && column < width && row < height;
    if (!isInsideRaster) {
      continue;
    }
    const exactZone = findContainingFeature(polygons, x, y);
    if (exactZone !== NO_ZONE) {
      exactCounts[exactZone]++;
    }
    if (!expected.pointBoundaryMask[point]) {
      interiorPointCount++;
      expect(expected.pointZones[point], `interior point ${point}`).toBe(exactZone);
    }
  }
  expect(interiorPointCount).toBeGreaterThan(pointCount / 4);
  const allBoundaryPoints =
    expected.boundaryCounts.reduce((sum, count) => sum + count, 0) +
    expected.unassignedBoundaryCount;
  for (let zone = 0; zone < props.zoneCount; zone++) {
    const lower = expected.counts[zone] - expected.boundaryCounts[zone];
    expect(exactCounts[zone]).toBeGreaterThanOrEqual(lower);
    expect(exactCounts[zone]).toBeLessThanOrEqual(lower + allBoundaryPoints);
  }
}

it('GPURasterJoin over GPUPolygonRasterization agrees with exact point-in-polygon on interior points across per-frame extents', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const polygons = createRandomScene(23, 30, 24, 24, 5);
  const zoneCount = polygons.featureOffsets.length - 1;
  const width = 48;
  const height = 48;
  const extents: Extent[] = [
    [0, 0, 0.5, 0.5],
    [-2, 4, 0.25, 0.375]
  ];
  const fixture = createRasterizationFixture(device, {
    polygons,
    width,
    height,
    extent: extents[0],
    crossingCapacity: 4096
  });
  const {points, values} = createPoints(31, 6000, width, height, extents[0]);
  const buffers = addJoin(device, fixture.graph, {
    width,
    height,
    extent: fixture.extentView,
    zones: fixture.zonesView,
    boundary: fixture.boundaryView,
    zoneCount,
    points,
    values
  });
  const compiled = fixture.graph.compile();
  for (const extent of extents) {
    fixture.extent.write(getGPUPolygonRasterizationExtentValues(...extent));
    const frame = createPoints(31 + extent[1], 6000, width, height, extent);
    buffers.points.write(frame.points);
    submitGraph(device, compiled, undefined);
    await expectEndToEnd(fixture, buffers, polygons, {
      points: frame.points,
      values,
      extent,
      zoneCount
    });
  }
  expect(points.length).toBe(12000);
  compiled.destroy();
  destroyJoinBuffers(buffers);
  destroyRasterizationFixture(fixture);
});
