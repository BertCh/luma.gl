// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUArealInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/index';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  getGPUPolygonRasterizationExtentValues,
  GPUPolygonRasterization
} from '../../../src/gpu-raster/polygon-rasterization';
import {
  computeArealOracle,
  computeTransferOracle,
  createBlockZones,
  NO_ZONE
} from './areal-interpolation-oracle';
import {createSeededRandom, expectClose, GraphRig} from './areal-interpolation-harness';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

async function runCase(
  device: Device,
  options: {
    denominator: 'zone' | 'overlap';
    mode: 'extensive' | 'intensive';
    masked: boolean;
    capacitySlack: number;
  }
) {
  const width = 24;
  const height = 18;
  const source = createBlockZones(width, height, 7, 5, {holeSeed: 4, holeRate: 0.05});
  const target = createBlockZones(width, height, 6, 8, {
    shiftX: 2,
    shiftY: 3,
    holeSeed: 9,
    holeRate: 0.08
  });
  const random = createSeededRandom(21);
  const cellWeights = options.masked
    ? Float32Array.from({length: width * height}, () => (random() < 0.15 ? 0 : 0.25 + random()))
    : undefined;
  const expected = computeArealOracle({
    sourceZones: source.zones,
    targetZones: target.zones,
    sourceCount: source.zoneCount,
    targetCount: target.zoneCount,
    cellWeights,
    denominator: options.denominator
  });
  const pairCount = expected.neighbors.length;
  expect(pairCount).toBeGreaterThan(10);
  const sourceValues = Float32Array.from({length: source.zoneCount}, (_, index) => 10 + index * 3);
  const categories = Uint32Array.from({length: source.zoneCount}, (_, index) => index % 3);

  const rig = new GraphRig(device);
  const capacity = pairCount + options.capacitySlack;
  const offsets = rig.output('uint32', target.zoneCount + 1);
  const neighbors = rig.output('uint32', capacity);
  const weights = rig.output('float32', capacity);
  const alternate = rig.output('float32', capacity);
  const areas = rig.output('float32', capacity);
  const overflow = rig.output('uint32', 1);
  const totalPairs = rig.output('uint32', 1);
  const extensiveTransfer = rig.output('float32', target.zoneCount);
  const intensiveTransfer = rig.output('float32', target.zoneCount);
  const shares = rig.output('float32', target.zoneCount * 3);
  const arealWeights = {offsets: offsets.view, neighbors: neighbors.view, weights: weights.view};
  const alternateWeights = {...arealWeights, weights: alternate.view};
  const extensiveWeights = options.mode === 'extensive' ? arealWeights : alternateWeights;
  const intensiveWeights = options.mode === 'extensive' ? alternateWeights : arealWeights;
  const sourceValuesView = rig.input(sourceValues, 'float32');
  // Two variables lagged at once (columnCount 2): the values and a constant 1.
  const twoColumns = rig.input(
    Float32Array.from({length: source.zoneCount * 2}, (_, index) =>
      index % 2 === 0 ? sourceValues[index >> 1] : 1
    ),
    'float32'
  );
  const twoColumnOutput = rig.output('float32', target.zoneCount * 2);
  rig.run(
    new GPUArealInterpolation({
      sourceZones: rig.input(source.zones, 'uint32'),
      targetZones: rig.input(target.zones, 'uint32'),
      sourceCount: source.zoneCount,
      targetCount: target.zoneCount,
      cellWeights: cellWeights ? rig.input(cellWeights, 'float32') : undefined,
      denominator: options.denominator,
      mode: options.mode,
      weights: arealWeights,
      alternateWeights: alternate.view,
      areas: areas.view,
      overflow: overflow.view,
      totalPairs: totalPairs.view,
      categories: {
        sourceCategories: rig.input(categories, 'uint32'),
        categoryCount: 3,
        output: shares.view
      }
    }),
    // The roadmap composition: GPUSpatialLag over the cross (target to source) area-share weights.
    new GPUSpatialLag({
      id: 'lag-extensive',
      weights: extensiveWeights,
      sourceCount: source.zoneCount,
      values: sourceValuesView,
      output: extensiveTransfer.view
    }),
    new GPUSpatialLag({
      id: 'lag-intensive',
      weights: intensiveWeights,
      sourceCount: source.zoneCount,
      values: sourceValuesView,
      output: intensiveTransfer.view
    }),
    new GPUSpatialLag({
      id: 'lag-columns',
      weights: extensiveWeights,
      sourceCount: source.zoneCount,
      columnCount: 2,
      values: twoColumns,
      output: twoColumnOutput.view
    })
  );
  const actualOffsets = await offsets.readUint32();
  const actual = {
    offsets: actualOffsets,
    neighbors: (await neighbors.readUint32()).slice(0, pairCount),
    weights: (await weights.readFloat32()).slice(0, pairCount),
    alternate: (await alternate.readFloat32()).slice(0, pairCount),
    areas: (await areas.readFloat32()).slice(0, pairCount),
    overflow: (await overflow.readUint32())[0],
    totalPairs: (await totalPairs.readUint32())[0],
    extensive: await extensiveTransfer.readFloat32(),
    intensive: await intensiveTransfer.readFloat32(),
    shares: await shares.readFloat32(),
    columns: await twoColumnOutput.readFloat32()
  };
  rig.destroy();
  return {expected, actual, sourceValues, categories, pairCount, targetCount: target.zoneCount};
}

for (const denominator of ['zone', 'overlap'] as const) {
  for (const masked of [false, true]) {
    it(`GPUArealInterpolation matches the oracle (${denominator}, ${masked ? 'dasymetric' : 'plain'})`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) return;
      const {expected, actual, sourceValues, categories, pairCount, targetCount} = await runCase(
        device,
        {denominator, mode: 'extensive', masked, capacitySlack: 5}
      );
      expect(actual.overflow).toBe(0);
      expect(actual.totalPairs).toBe(pairCount);
      expect(actual.offsets).toEqual(expected.offsets);
      expect(actual.neighbors).toEqual(expected.neighbors);
      expectClose(actual.areas, expected.areas, 'areas');
      expectClose(actual.weights, expected.extensive, 'extensive weights');
      expectClose(actual.alternate, expected.intensive, 'intensive weights');
      // Nonzero guard: a failed WGSL compile yields silent zeros.
      expect(actual.weights.some(weight => weight > 0)).toBe(true);
      expect(actual.extensive.some(value => value > 0)).toBe(true);
      expectClose(
        actual.extensive,
        computeTransferOracle(
          expected.offsets,
          expected.neighbors,
          expected.extensive,
          sourceValues
        ),
        'extensive transfer',
        1e-4
      );
      // columnCount 2 matches the single-column lag of the values, and lags a constant 1 into the
      // total extensive share received by each target.
      expectClose(
        actual.columns.filter((_, index) => index % 2 === 0),
        actual.extensive,
        'two-column lag, first column'
      );
      expect(actual.columns.some((value, index) => index % 2 === 1 && value > 0)).toBe(true);
      expectClose(
        actual.intensive,
        computeTransferOracle(
          expected.offsets,
          expected.neighbors,
          expected.intensive,
          sourceValues
        ),
        'intensive transfer',
        1e-4
      );
      // Categorical shares sum to 1 for every covered target and match the oracle.
      for (let target = 0; target < targetCount; target++) {
        const covered = expected.offsets[target + 1] > expected.offsets[target];
        const row = actual.shares.slice(target * 3, target * 3 + 3);
        expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(covered ? 1 : 0, 4);
        for (let category = 0; category < 3; category++) {
          let inCategory = 0;
          let total = 0;
          for (let slot = expected.offsets[target]; slot < expected.offsets[target + 1]; slot++) {
            total += expected.areas[slot];
            if (categories[expected.neighbors[slot]] === category)
              inCategory += expected.areas[slot];
          }
          expect(row[category]).toBeCloseTo(total > 0 ? inCategory / total : 0, 4);
        }
      }
    });
  }
}

it('GPUArealInterpolation writes intensive weights to weights.weights when mode is intensive', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {expected, actual} = await runCase(device, {
    denominator: 'zone',
    mode: 'intensive',
    masked: false,
    capacitySlack: 0
  });
  expectClose(actual.weights, expected.intensive, 'intensive weights');
  expectClose(actual.alternate, expected.extensive, 'extensive weights');
  expect(actual.weights.some(weight => weight > 0)).toBe(true);
});

it('GPUArealInterpolation conserves extensive mass over the shared extent with overlap denominators', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {actual, sourceValues, expected} = await runCase(device, {
    denominator: 'overlap',
    mode: 'extensive',
    masked: true,
    capacitySlack: 0
  });
  // Every source that overlaps a target gives away its whole value.
  const overlapping = new Set(expected.neighbors);
  let expectedTotal = 0;
  for (const source of overlapping) expectedTotal += sourceValues[source];
  const transferred = actual.extensive.reduce((sum, value) => sum + value, 0);
  expect(transferred).toBeCloseTo(expectedTotal, 2);
  expect(transferred).toBeGreaterThan(0);
});

it('GPUArealInterpolation flags capacity overflow and clamps the offsets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 12;
  const source = createBlockZones(width, 12, 4, 4);
  const target = createBlockZones(width, 12, 3, 3, {shiftX: 1});
  const expected = computeArealOracle({
    sourceZones: source.zones,
    targetZones: target.zones,
    sourceCount: source.zoneCount,
    targetCount: target.zoneCount,
    denominator: 'zone'
  });
  const capacity = 10;
  expect(expected.neighbors.length).toBeGreaterThan(capacity);
  const rig = new GraphRig(device);
  const offsets = rig.output('uint32', target.zoneCount + 1);
  const neighbors = rig.output('uint32', capacity);
  const weights = rig.output('float32', capacity);
  const overflow = rig.output('uint32', 1);
  const totalPairs = rig.output('uint32', 1);
  rig.run(
    new GPUArealInterpolation({
      sourceZones: rig.input(source.zones, 'uint32'),
      targetZones: rig.input(target.zones, 'uint32'),
      sourceCount: source.zoneCount,
      targetCount: target.zoneCount,
      weights: {offsets: offsets.view, neighbors: neighbors.view, weights: weights.view},
      overflow: overflow.view,
      totalPairs: totalPairs.view
    })
  );
  expect((await overflow.readUint32())[0]).toBe(1);
  expect((await totalPairs.readUint32())[0]).toBe(expected.neighbors.length);
  const actualOffsets = await offsets.readUint32();
  expect(actualOffsets).toEqual(expected.offsets.map(offset => Math.min(offset, capacity)));
  expect(actualOffsets[target.zoneCount]).toBe(capacity);
  const actualNeighbors = await neighbors.readUint32();
  expect(actualNeighbors).toEqual(expected.neighbors.slice(0, capacity));
  rig.destroy();
});

it('GPUArealInterpolation composes with GPUPolygonRasterization on a common grid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const rectangle = (x0: number, y0: number, x1: number, y1: number) => [
    x0,
    y0,
    x1,
    y0,
    x1,
    y1,
    x0,
    y1
  ];
  const toPolygons = (rectangles: number[][]) => ({
    positions: Float32Array.from(rectangles.flat()),
    featureOffsets: Uint32Array.from(rectangles.map((_, index) => index).concat(rectangles.length)),
    polygonOffsets: Uint32Array.from(rectangles.map((_, index) => index).concat(rectangles.length)),
    ringOffsets: Uint32Array.from(
      rectangles.map((_, index) => index * 4).concat(rectangles.length * 4)
    )
  });
  // Sources: two halves of [0,16]x[0,8]. Targets: three vertical strips of unequal width.
  const sources = toPolygons([rectangle(0, 0, 8, 8), rectangle(8, 0, 16, 8)]);
  const targets = toPolygons([
    rectangle(0, 0, 4, 8),
    rectangle(4, 0, 12, 8),
    rectangle(12, 0, 16, 8)
  ]);
  const width = 64;
  const height = 32;
  const rig = new GraphRig(device);
  const extent = rig.input(
    getGPUPolygonRasterizationExtentValues(0, 0, 16 / width, 8 / height),
    'float32'
  );
  const rasterize = (id: string, polygons: ReturnType<typeof toPolygons>) => {
    const zones = rig.output('uint32', width * height);
    const rasterOverflow = rig.output('uint32', 1);
    const producer = new GPUPolygonRasterization({
      id,
      width,
      height,
      extent,
      polygonPositions: rig.input(polygons.positions, 'float32x2', polygons.positions.length / 2),
      featureOffsets: rig.input(polygons.featureOffsets, 'uint32'),
      polygonOffsets: rig.input(polygons.polygonOffsets, 'uint32'),
      ringOffsets: rig.input(polygons.ringOffsets, 'uint32'),
      crossingCapacity: 1024,
      zones: zones.view,
      overflow: rasterOverflow.view
    });
    return {zones, producer};
  };
  const sourceRaster = rasterize('source-raster', sources);
  const targetRaster = rasterize('target-raster', targets);
  const offsets = rig.output('uint32', 4);
  const neighbors = rig.output('uint32', 8);
  const weights = rig.output('float32', 8);
  const intensive = rig.output('float32', 8);
  const overflow = rig.output('uint32', 1);
  const total = rig.output('float32', 3);
  rig.run(
    sourceRaster.producer,
    targetRaster.producer,
    new GPUArealInterpolation({
      sourceZones: sourceRaster.zones.view,
      targetZones: targetRaster.zones.view,
      sourceCount: 2,
      targetCount: 3,
      weights: {offsets: offsets.view, neighbors: neighbors.view, weights: weights.view},
      alternateWeights: intensive.view,
      overflow: overflow.view
    }),
    new GPUSpatialLag({
      weights: {offsets: offsets.view, neighbors: neighbors.view, weights: weights.view},
      sourceCount: 2,
      values: rig.input(Float32Array.of(80, 160), 'float32'),
      output: total.view
    })
  );
  expect((await sourceRaster.zones.readUint32()).includes(NO_ZONE)).toBe(false);
  expect(await offsets.readUint32()).toEqual([0, 1, 3, 4]);
  expect((await neighbors.readUint32()).slice(0, 4)).toEqual([0, 0, 1, 1]);
  // Source 0 (x in 0..8) is split 4/8 into strip 0 and 4/8 into strip 1: extensive shares 0.5.
  expectClose((await weights.readFloat32()).slice(0, 4), [0.5, 0.5, 0.5, 0.5], 'extensive shares');
  // Strip 1 is half source 0, half source 1: intensive shares 0.5 each; strips 0 and 2 are whole.
  expectClose((await intensive.readFloat32()).slice(0, 4), [1, 0.5, 0.5, 1], 'intensive shares');
  expectClose(await total.readFloat32(), [40, 40 + 80, 80], 'transferred totals');
  rig.destroy();
});

/**
 * Reference from tobler 0.12 `area_interpolate` (allocate_total, exact polygon intersection) on
 * unit-aligned rectangles: sources are 7x5 blocks and targets 6x8 blocks shifted by (2, 3) over a
 * 24x18 grid, extensive and intensive variable `10 + 3 * sourceIndex`. The edges are grid aligned,
 * so the raster result is exact.
 */
const TOBLER_EXTENSIVE = [
  5.714286, 9.857143, 12.000001, 17.761905, 12.666667, 24.228572, 38.400001, 41.828573, 56.076192,
  37.866669, 34.057144, 52.885715, 55.885716, 72.638097, 48.133335
];
const TOBLER_INTENSIVE = [
  10.0, 11.5, 14.0, 16.500001, 19.0, 26.5, 28.0, 30.500001, 33.0, 35.5, 41.200001, 42.700002,
  45.200003, 47.700001, 50.200002
];

it('GPUArealInterpolation plus GPUSpatialLag matches tobler area_interpolate on aligned rectangles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const source = createBlockZones(24, 18, 7, 5);
  const target = createBlockZones(24, 18, 6, 8, {shiftX: 2, shiftY: 3});
  expect(source.zoneCount).toBe(16);
  expect(target.zoneCount).toBe(15);
  const values = Float32Array.from({length: 16}, (_, index) => 10 + index * 3);
  const rig = new GraphRig(device);
  const offsets = rig.output('uint32', 16);
  const neighbors = rig.output('uint32', 128);
  const extensiveWeights = rig.output('float32', 128);
  const intensiveWeights = rig.output('float32', 128);
  const overflow = rig.output('uint32', 1);
  const extensive = rig.output('float32', 15);
  const intensive = rig.output('float32', 15);
  const valuesView = rig.input(values, 'float32');
  rig.run(
    new GPUArealInterpolation({
      sourceZones: rig.input(source.zones, 'uint32'),
      targetZones: rig.input(target.zones, 'uint32'),
      sourceCount: 16,
      targetCount: 15,
      weights: {offsets: offsets.view, neighbors: neighbors.view, weights: extensiveWeights.view},
      alternateWeights: intensiveWeights.view,
      overflow: overflow.view
    }),
    new GPUSpatialLag({
      id: 'tobler-extensive',
      weights: {offsets: offsets.view, neighbors: neighbors.view, weights: extensiveWeights.view},
      sourceCount: 16,
      values: valuesView,
      output: extensive.view
    }),
    new GPUSpatialLag({
      id: 'tobler-intensive',
      weights: {offsets: offsets.view, neighbors: neighbors.view, weights: intensiveWeights.view},
      sourceCount: 16,
      values: valuesView,
      output: intensive.view
    })
  );
  expect((await overflow.readUint32())[0]).toBe(0);
  const actualExtensive = await extensive.readFloat32();
  expect(actualExtensive.every(value => value > 1)).toBe(true);
  expectClose(actualExtensive, TOBLER_EXTENSIVE, 'tobler extensive', 1e-5, 1e-5);
  expectClose(await intensive.readFloat32(), TOBLER_INTENSIVE, 'tobler intensive', 1e-5, 1e-5);
  rig.destroy();
});
