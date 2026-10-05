// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {getGPUPolygonRasterizationExtentValues} from '../../../src/gpu-raster/polygon-rasterization';
import {
  createPolygons,
  NO_ZONE,
  rasterizePolygonsOnCPU,
  type OraclePolygons,
  type OracleRasterization
} from './polygon-rasterization-oracle';
import {
  createRasterizationFixture,
  destroyRasterizationFixture,
  readRasterization
} from './polygon-rasterization-fixture';
import {createExactScene, createRandomScene} from './polygon-rasterization-scenes';

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
  if (!device) {
    return;
  }
  const polygons = createExactScene();
  const raster = {width: 16, height: 12, extent: [0, 0, 1, 1] as Extent};
  const fixture = createRasterizationFixture(device, {...raster, polygons, crossingCapacity: 256});
  expect(fixture.recipe.singleSort).toBe(true);
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
  if (!device) {
    return;
  }
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
  if (!device) {
    return;
  }
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
  if (!device) {
    return;
  }
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
    fixture.extent.write(getGPUPolygonRasterizationExtentValues(...extent));
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
  if (!device) {
    return;
  }
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
  expect(fixture.recipe.singleSort).toBe(false);
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
