// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUPolygonRasterization} from '../../../src/gpu-raster/polygon-rasterization';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {getGPUPolygonRasterizationParameterValues} from '../../../src/gpu-raster/polygon-rasterization';
import {expect, it} from 'vitest';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  createPolygons,
  createSplitFijiPolygons,
  NO_ZONE,
  rasterizePolygonsOnCPU,
  type OraclePolygons,
  type OracleRasterization
} from './polygon-rasterization-oracle';
import {
  createRasterizationFixture,
  destroyRasterizationFixture,
  getExtentParameterValues,
  readRasterization
} from './polygon-rasterization-fixture';
import {createExactScene, createRandomScene} from './polygon-rasterization-scenes';
import {requireTestDevice} from '../../utils/require-test-device';

type Extent = readonly [number, number, number, number];

/** Asserts zones equal the oracle outside ambiguous cells and boundary flags are conservative. */
function expectMatchesOracle(
  actual: {zones: number[]; boundary: number[]},
  expected: OracleRasterization,
  options: {allowAmbiguous: boolean; ignoreCell?: (cell: number) => boolean}
): {ambiguousMismatches: number} {
  let ambiguousMismatches = 0;
  const strictMismatches: number[] = [];
  const missedBoundary: number[] = [];
  const extraBoundary: number[] = [];
  for (let cell = 0; cell < expected.zones.length; cell++) {
    if (options.ignoreCell?.(cell)) {
      continue;
    }
    if (actual.zones[cell] !== expected.zones[cell]) {
      if (options.allowAmbiguous && expected.ambiguous[cell]) {
        ambiguousMismatches++;
      } else {
        strictMismatches.push(cell);
      }
    }
    if (expected.boundary[cell] && !actual.boundary[cell]) {
      missedBoundary.push(cell);
    }
    if (actual.boundary[cell] && !expected.looseBoundary[cell]) {
      extraBoundary.push(cell);
    }
  }
  expect(strictMismatches.slice(0, 10), 'zone mismatches').toEqual([]);
  expect(missedBoundary.slice(0, 10), 'missed boundary cells').toEqual([]);
  expect(extraBoundary.slice(0, 10), 'non-conservative boundary cells').toEqual([]);
  return {ambiguousMismatches};
}

it('GPUPolygonRasterization matches the oracle on holes, overlaps, sub-cell polygons, and center-aligned edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createExactScene();
  const raster = {width: 16, height: 12, extent: [0, 0, 1, 1] as Extent};
  const fixture = createRasterizationFixture(device, {...raster, polygons, crossingCapacity: 256});
  expect(fixture.contributor.singleSort).toBe(true);
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const actual = await readRasterization(fixture);
  const expected = rasterizePolygonsOnCPU(polygons, raster);
  expectMatchesOracle(actual, expected, {allowAmbiguous: false});
  expect(actual.overflow).toBe(0);
  expect(actual.crossingCount).toBe(expected.crossingCount);

  const zoneAt = (column: number, row: number) => actual.zones[row * 16 + column];
  // Hole of feature 0, overlap won by feature 0, and feature 1 outside the overlap.
  expect(zoneAt(3, 3)).toBe(NO_ZONE);
  expect(zoneAt(5, 5)).toBe(0);
  expect(zoneAt(8, 4)).toBe(1);
  // Sub-cell multipolygon: the triangle covers no center, the square covers (12.5, 3.5).
  expect(zoneAt(12, 1)).toBe(NO_ZONE);
  expect(zoneAt(12, 3)).toBe(2);
  expect(actual.boundary[1 * 16 + 12]).toBe(1);
  // Edges on centers: rows 8 and 9, columns 0..2 (feature 3) and 3..4 (feature 4), no gap.
  expect(actual.zones.slice(8 * 16, 8 * 16 + 6)).toEqual([3, 3, 3, 4, 4, NO_ZONE]);
  expect(zoneAt(0, 10)).toBe(NO_ZONE);
  // The duplicate feature 6 always loses to feature 1; feature 7 is clipped at the left edge.
  expect(actual.zones.includes(6)).toBe(false);
  expect(zoneAt(0, 7)).toBe(7);
  expect(actual.zones.filter(zone => zone === 5).length).toBeGreaterThan(5);

  // A second encoding of the same compiled graph is identical.
  submitGraph(device, compiled, undefined);
  expect((await readRasterization(fixture)).zones).toEqual(actual.zones);
  compiled.destroy();
  destroyRasterizationFixture(fixture);
});

it('GPUPolygonRasterization confines a non-finite vertex to its own polygon', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createExactScene();
  const broken = createPolygons([
    [
      [
        [
          [13, 8],
          [15, 8],
          [Number.NaN, 9.5],
          [15, 11],
          [13, 11]
        ]
      ]
    ]
  ]);
  // Append the broken polygon as feature 8.
  const vertexBase = scene.positions.length / 2;
  const ringBase = scene.ringOffsets.length - 1;
  const polygonBase = scene.polygonOffsets.length - 1;
  const polygons: OraclePolygons = {
    positions: Float32Array.from([...scene.positions, ...broken.positions]),
    ringOffsets: Uint32Array.from([
      ...scene.ringOffsets,
      ...Array.from(broken.ringOffsets.slice(1), offset => offset + vertexBase)
    ]),
    polygonOffsets: Uint32Array.from([
      ...scene.polygonOffsets,
      ...Array.from(broken.polygonOffsets.slice(1), offset => offset + ringBase)
    ]),
    featureOffsets: Uint32Array.from([
      ...scene.featureOffsets,
      ...Array.from(broken.featureOffsets.slice(1), offset => offset + polygonBase)
    ])
  };
  const raster = {width: 16, height: 12, extent: [0, 0, 1, 1] as Extent};
  const fixture = createRasterizationFixture(device, {...raster, polygons, crossingCapacity: 256});
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const actual = await readRasterization(fixture);
  const expected = rasterizePolygonsOnCPU(polygons, raster);
  // Columns 13..15 of rows 8..10 belong to the broken polygon; everything else is exact.
  expectMatchesOracle(actual, expected, {
    allowAmbiguous: false,
    ignoreCell: cell => cell % 16 >= 13 && Math.floor(cell / 16) >= 8
  });
  compiled.destroy();
  destroyRasterizationFixture(fixture);
});

it('GPUPolygonRasterization matches the oracle on random polygons with holes and multipolygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const seed of [3, 17]) {
    const polygons = createRandomScene(seed, 60, 34, 26, 6);
    // Origin offset so polygons extend past every raster edge.
    const raster = {width: 64, height: 48, extent: [1, 1, 0.5, 0.5] as Extent};
    const expected = rasterizePolygonsOnCPU(polygons, raster);
    const fixture = createRasterizationFixture(device, {
      ...raster,
      polygons,
      crossingCapacity: expected.crossingCount + 64
    });
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const actual = await readRasterization(fixture);
    const {ambiguousMismatches} = expectMatchesOracle(actual, expected, {allowAmbiguous: true});
    expect(ambiguousMismatches).toBeLessThanOrEqual(2);
    expect(actual.crossingCount).toBe(expected.crossingCount);
    expect(actual.overflow).toBe(0);
    expect(new Set(actual.zones).size).toBeGreaterThan(20);
    compiled.destroy();
    destroyRasterizationFixture(fixture);
  }
});

it('GPUPolygonRasterization follows per-frame extents and reports overflow without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createRandomScene(29, 40, 30, 30, 5);
  const width = 40;
  const height = 40;
  const extents: Extent[] = [
    [0, 0, 0.75, 0.75],
    [-1.5, 5, 0.75, 0.5],
    // Denser rows over the busy middle band: more crossings than the capacity.
    [0, 10, 0.75, 0.25],
    [0, 0, 0.75, 0.75]
  ];
  const first = rasterizePolygonsOnCPU(polygons, {width, height, extent: extents[0]});
  const second = rasterizePolygonsOnCPU(polygons, {width, height, extent: extents[1]});
  const crossingCapacity = Math.max(first.crossingCount, second.crossingCount) + 16;
  const dense = rasterizePolygonsOnCPU(polygons, {width, height, extent: extents[2]});
  expect(dense.crossingCount).toBeGreaterThan(crossingCapacity);
  const fixture = createRasterizationFixture(device, {
    width,
    height,
    extent: extents[0],
    polygons,
    crossingCapacity
  });
  const compiled = fixture.graph.compile();
  let overflowFrameCount = 0;
  for (const extent of extents) {
    fixture.extent.write(getExtentParameterValues(width, height, extent));
    submitGraph(device, compiled, undefined);
    const actual = await readRasterization(fixture);
    const expected = rasterizePolygonsOnCPU(polygons, {width, height, extent});
    expect(actual.crossingCount).toBe(expected.crossingCount);
    if (expected.crossingCount > crossingCapacity) {
      overflowFrameCount++;
      expect(actual.overflow).toBe(1);
      expect(actual.zones.every(zone => zone === NO_ZONE)).toBe(true);
      // Boundary flags do not depend on the crossing capacity.
      expectMatchesOracle(
        {zones: Array.from(expected.zones), boundary: actual.boundary},
        expected,
        {allowAmbiguous: false}
      );
    } else {
      expect(actual.overflow).toBe(0);
      const {ambiguousMismatches} = expectMatchesOracle(actual, expected, {allowAmbiguous: true});
      expect(ambiguousMismatches).toBeLessThanOrEqual(2);
    }
  }
  expect(overflowFrameCount).toBe(1);
  compiled.destroy();
  destroyRasterizationFixture(fixture);
});

it('GPUPolygonRasterization sorts with two stable passes when the packed key exceeds 32 bits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 65535;
  const height = 64;
  // About 1,300 polygons need 11 key bits; 64 * 65536 row-column keys need 22.
  const polygons = createRandomScene(41, 1100, width, height, 3);
  const raster = {width, height, extent: [0, 0, 1, 1] as Extent};
  // float32 crossings near x = 65535 are only good to about 2^-8 cells.
  const expected = rasterizePolygonsOnCPU(polygons, raster, {
    tieTolerance: 1e-2,
    loosePadding: 0.15
  });
  const fixture = createRasterizationFixture(device, {
    ...raster,
    polygons,
    crossingCapacity: expected.crossingCount + 64
  });
  expect(fixture.contributor.singleSort).toBe(false);
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const actual = await readRasterization(fixture);
  const {ambiguousMismatches} = expectMatchesOracle(actual, expected, {allowAmbiguous: true});
  // Only float32 ties within 0.01 cells of a crossing may differ.
  expect(ambiguousMismatches).toBeLessThanOrEqual(32);
  expect(actual.crossingCount).toBe(expected.crossingCount);
  expect(new Set(actual.zones).size).toBeGreaterThan(500);
  compiled.destroy();
  destroyRasterizationFixture(fixture);
});

it('GPUPolygonRasterization mirrors the zone and boundary rasters for a north-up grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createExactScene();
  const width = 16;
  const height = 12;
  const raster = {width, height, extent: [0, 0, 1, 1] as Extent};
  const fixture = createRasterizationFixture(device, {...raster, polygons, crossingCapacity: 256});
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const southUp = await readRasterization(fixture);
  // The same raster as a north-up image: row 0 is the northernmost row, the top edge is y = 12.
  fixture.extent.write(
    getGPUPolygonRasterizationParameterValues(
      makeGPURasterMetadata({width, height, cellSize: [1, 1], origin: [0, height]})
    )
  );
  submitGraph(device, compiled, undefined);
  const northUp = await readRasterization(fixture);
  const mirror = (values: number[]) =>
    values.map(
      (_, cell) => values[(height - 1 - Math.floor(cell / width)) * width + (cell % width)]
    );
  expect(northUp.zones).toEqual(mirror(southUp.zones));
  expect(northUp.boundary).toEqual(mirror(southUp.boundary));
  expect(northUp.zones.some(zone => zone !== NO_ZONE)).toBe(true);
  expect(northUp.crossingCount).toBe(southUp.crossingCount);
  compiled.destroy();
  destroyRasterizationFixture(fixture);
});

it('GPUPolygonRasterization shifts split Fiji per frame and ignores empty rings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createSplitFijiPolygons();
  const width = 80;
  const height = 16;
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'longitude-extent',
    format: 'float32',
    length: 4,
    values: new Float32Array([170, -19, 0.25, 0.25])
  });
  const positions = createInputBuffer(device, polygons.positions);
  const featureOffsets = createInputBuffer(device, polygons.featureOffsets);
  const polygonOffsets = createInputBuffer(device, polygons.polygonOffsets);
  const ringOffsets = createInputBuffer(device, polygons.ringOffsets);
  const zones = createOutputBuffer(device, width * height);
  const boundary = createOutputBuffer(device, width * height);
  const overflow = createOutputBuffer(device, 1);
  const crossingCount = createOutputBuffer(device, 1);
  graph.add(
    new GPUPolygonRasterization({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
      grid: makeGPURasterMetadata({width, height}),
      parameters: parameters.importToGraph(graph),
      positions: importGraphBuffer(
        graph,
        'positions',
        positions,
        'float32x2',
        polygons.positions.length / 2
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'features',
        featureOffsets,
        'uint32',
        polygons.featureOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygons',
        polygonOffsets,
        'uint32',
        polygons.polygonOffsets.length
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'rings',
        ringOffsets,
        'uint32',
        polygons.ringOffsets.length
      ),
      crossingCapacity: 256,
      output: {
        zones: importGraphBuffer(graph, 'zones', zones, 'uint32', width * height),
        boundary: importGraphBuffer(graph, 'boundary', boundary, 'uint32', width * height),
        status: {
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          crossingCount: importGraphBuffer(graph, 'crossing-count', crossingCount, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  for (const extent of [
    [170, -19, 0.25, 0.25],
    [-190, -19, 0.25, 0.25],
    [-180, -19, 0.25, 0.25],
    // Grid lies inside a polygon with both vertical edges outside its x range.
    [178, -17.5, 0.015625, 0.0625]
  ] as const) {
    parameters.write(new Float32Array(extent));
    submitGraph(device, compiled, undefined);
    const expected = rasterizePolygonsOnCPU(
      polygons,
      {width, height, extent},
      {longitudeLatitude: true, loosePadding: 0.05}
    );
    expectMatchesOracle(
      {
        zones: await readUint32(zones, width * height),
        boundary: await readUint32(boundary, width * height)
      },
      expected,
      {allowAmbiguous: false}
    );
    expect(await readUint32(overflow, 1)).toEqual([0]);
    expect(await readUint32(crossingCount, 1)).toEqual([expected.crossingCount]);
  }
  // All preparation copies empty: unused position capacity must not create edges.
  ringOffsets.write(new Uint32Array(polygons.ringOffsets.length));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(zones, width * height)).toEqual(new Array(width * height).fill(NO_ZONE));
  expect(await readUint32(boundary, width * height)).toEqual(new Array(width * height).fill(0));
  expect(await readUint32(crossingCount, 1)).toEqual([0]);
  compiled.destroy();
  parameters.destroy();
  for (const buffer of [
    positions,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    zones,
    boundary,
    overflow,
    crossingCount
  ])
    buffer.destroy();
});
