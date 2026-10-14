// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {getGPUPolygonRasterizationParameterValues} from '../../../src/gpu-raster/polygon-rasterization/index';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {
  computeArealOracle,
  computeTransferOracle,
  NO_ZONE
} from '../areal-interpolation/areal-interpolation-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';

type Rectangle = [number, number, number, number];

// Oracle: <scratchpad>/fix/work/D/change-of-support/change_of_support_recipe.py (numpy 2.5.4):
// raster area of each source and the fraction of it inside the target system (cell-center rule).
const NUMPY_SOURCE_AREAS = [108, 108, 108, 108];
const NUMPY_TRANSFERRED_FRACTIONS = [
  1.0, 0.9444444444444444, 0.5555555555555556, 0.46296296296296297
];

const WIDTH = 24;
const HEIGHT = 18;
// Unit cells from the origin, row 0 at the smallest y.
const GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, rowDirection: 'north'});

/** Rectangles `[x0, y0, x1, y1]` with integer corners as a polygon feature set. */
function createRectangleSystem(rectangles: Rectangle[]) {
  const positions: number[] = [];
  const ringOffsets = [0];
  for (const [x0, y0, x1, y1] of rectangles) {
    positions.push(x0, y0, x1, y0, x1, y1, x0, y1);
    ringOffsets.push(positions.length / 2);
  }
  const indices = rectangles.map((_, index) => index);
  return {
    rectangles,
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.from([...indices, rectangles.length]),
    featureOffsets: Uint32Array.from([...indices, rectangles.length])
  };
}

/** CPU raster of the cell-center rule on unit cells with origin (0, 0). */
function rasterize(rectangles: Rectangle[]): Uint32Array {
  const zones = new Uint32Array(WIDTH * HEIGHT).fill(NO_ZONE);
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const zone = rectangles.findIndex(
        ([x0, y0, x1, y1]) => x + 0.5 > x0 && x + 0.5 < x1 && y + 0.5 > y0 && y + 0.5 < y1
      );
      if (zone >= 0) {
        zones[y * WIDTH + x] = zone;
      }
    }
  }
  return zones;
}

it('addChangeOfSupportRecipe rasterizes two zone systems and transfers values by area', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 2 x 2 source blocks and 3 x 2 target blocks that straddle them; the target system leaves a gap.
  const source = createRectangleSystem([
    [0, 0, 12, 9],
    [12, 0, 24, 9],
    [0, 9, 12, 18],
    [12, 9, 24, 18]
  ]);
  const target = createRectangleSystem([
    [0, 0, 8, 6],
    [8, 0, 16, 6],
    [16, 0, 24, 6],
    [0, 6, 10, 14],
    [10, 6, 22, 14]
  ]);
  const sourceZones = rasterize(source.rectangles);
  const targetZones = rasterize(target.rectangles);
  const sourceValues = Float32Array.from([100, 40, 70, 10]);
  const expected = computeArealOracle({
    sourceZones,
    targetZones,
    sourceCount: 4,
    targetCount: 5
  });
  const pairCount = expected.neighbors.length;
  expect(pairCount).toBeGreaterThan(5);

  const fixture = new RecipeTestFixture(device, 'change-of-support-recipe-test');
  try {
    const capacity = pairCount + 4;
    const offsets = fixture.output('offsets', 'uint32', 6);
    const neighbors = fixture.output('neighbors', 'uint32', capacity);
    const weights = fixture.output('weights', 'float32', capacity);
    const intensiveWeightValues = fixture.output('intensive-weights', 'float32', capacity);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const totalPairs = fixture.output('total-pairs', 'uint32', 1);
    const extensive = fixture.output('extensive', 'float32', 5);
    const intensive = fixture.output('intensive', 'float32', 5);
    const sourceAreas = fixture.output('source-areas', 'float32', 4);
    const fractions = fixture.output('source-fractions', 'float32', 4);
    const invalidCount = fixture.output('invalid-count', 'uint32', 1);
    const toZones = (system: ReturnType<typeof createRectangleSystem>, name: string) => ({
      kind: 'polygons' as const,
      positions: fixture.input(
        `${name}-positions`,
        system.positions,
        'float32x2',
        system.positions.length / 2
      ),
      featureOffsets: fixture.input(
        `${name}-features`,
        system.featureOffsets,
        'uint32',
        system.featureOffsets.length
      ),
      polygonOffsets: fixture.input(
        `${name}-polygons`,
        system.polygonOffsets,
        'uint32',
        system.polygonOffsets.length
      ),
      ringOffsets: fixture.input(
        `${name}-rings`,
        system.ringOffsets,
        'uint32',
        system.ringOffsets.length
      ),
      crossingCapacity: 1024
    });
    const recipe = addChangeOfSupportRecipe(fixture.graph, {
      grid: GRID,
      parameters: fixture.input(
        'parameters',
        getGPUPolygonRasterizationParameterValues(GRID),
        'float32',
        4
      ),
      source: toZones(source, 'source'),
      target: toZones(target, 'target'),
      pairCapacity: capacity,
      sourceValues: fixture.input('source-values', sourceValues, 'float32', 4),
      outputs: {
        extensiveWeights: {
          offsets: offsets.view,
          neighbors: neighbors.view,
          weights: weights.view
        },
        intensiveWeightValues: intensiveWeightValues.view,
        extensiveValues: extensive.view,
        intensiveValues: intensive.view,
        sourceAreas: sourceAreas.view,
        sourceTransferredFractions: fractions.view,
        status: {
          overflow: overflow.view,
          requiredCount: totalPairs.view,
          invalidCount: invalidCount.view
        }
      }
    });
    expect(recipe.contributors.length).toBe(5);
    fixture.run();

    expect((await fixture.readUint32(overflow, 1))[0]).toBe(0);
    expect((await fixture.readUint32(totalPairs, 1))[0]).toBe(pairCount);
    expect(await fixture.readUint32(offsets, 6)).toEqual(expected.offsets);
    expect(await fixture.readUint32(neighbors, pairCount)).toEqual(expected.neighbors);
    const gpuExtensive = await fixture.readFloat32(extensive, 5);
    const gpuIntensive = await fixture.readFloat32(intensive, 5);
    const oracleExtensive = computeTransferOracle(
      expected.offsets,
      expected.neighbors,
      expected.extensive,
      sourceValues
    );
    const oracleIntensive = computeTransferOracle(
      expected.offsets,
      expected.neighbors,
      expected.intensive,
      sourceValues
    );
    for (let zone = 0; zone < 5; zone++) {
      expect(
        isClose(gpuExtensive[zone], oracleExtensive[zone], 1e-3, 1e-4),
        `extensive ${zone}`
      ).toBe(true);
      expect(
        isClose(gpuIntensive[zone], oracleIntensive[zone], 1e-3, 1e-4),
        `intensive ${zone}`
      ).toBe(true);
    }
    // Conservation diagnostics: mass lost outside the target system is reported per source.
    const gpuAreas = await fixture.readFloat32(sourceAreas, 4);
    const gpuFractions = await fixture.readFloat32(fractions, 4);
    for (let zone = 0; zone < 4; zone++) {
      expect(gpuAreas[zone]).toBeCloseTo(NUMPY_SOURCE_AREAS[zone], 3);
      expect(gpuFractions[zone]).toBeCloseTo(NUMPY_TRANSFERRED_FRACTIONS[zone], 5);
    }
    expect((await fixture.readUint32(invalidCount, 1))[0]).toBe(0);
    // The extensive mass transferred equals the mass of each source times its fraction.
    const transferredMass = gpuExtensive.reduce((sum, value) => sum + value, 0);
    const sourceMass = sourceValues.reduce(
      (sum, value, zone) => sum + value * NUMPY_TRANSFERRED_FRACTIONS[zone],
      0
    );
    expect(transferredMass).toBeCloseTo(sourceMass, 2);
    // Nonzero guards: a failed shader gives silent zeros.
    expect(Math.max(...gpuExtensive)).toBeGreaterThan(1);
    // Intensive values are means of source values.
    for (const value of gpuIntensive) {
      expect(value).toBeGreaterThanOrEqual(10 - 1e-3);
      expect(value).toBeLessThanOrEqual(100 + 1e-3);
    }
    // Target 0 = [0,8]x[0,6] lies inside source 0 only.
    expect(gpuIntensive[0]).toBeCloseTo(100, 3);
    expect(gpuExtensive[0]).toBeCloseTo((48 / 108) * 100, 2);
  } finally {
    fixture.destroy();
  }
}, 120000);

it.each([
  'meters',
  'kilometers'
] as const)('change-of-support uses authalic cell areas in square %s', async units => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `change-of-support-authalic-${units}`);
  try {
    const grid = makeGPURasterMetadata({width: 1, height: 2, rowDirection: 'north'});
    const source = createRectangleSystem([[0, 0, 1, 2]]);
    const target = createRectangleSystem([
      [0, 0, 1, 1],
      [0, 1, 1, 2]
    ]);
    const zones = (system: ReturnType<typeof createRectangleSystem>, name: string) => ({
      kind: 'polygons' as const,
      positions: fixture.input(
        `${name}-positions`,
        system.positions,
        'float32x2',
        system.positions.length / 2
      ),
      featureOffsets: fixture.input(
        `${name}-features`,
        system.featureOffsets,
        'uint32',
        system.featureOffsets.length
      ),
      polygonOffsets: fixture.input(
        `${name}-polygons`,
        system.polygonOffsets,
        'uint32',
        system.polygonOffsets.length
      ),
      ringOffsets: fixture.input(
        `${name}-rings`,
        system.ringOffsets,
        'uint32',
        system.ringOffsets.length
      ),
      crossingCapacity: 16
    });
    const sourceAreas = fixture.output('source-areas', 'float32', 1);
    const extensiveValues = fixture.output('extensive-values', 'float32', 2);
    addChangeOfSupportRecipe(fixture.graph, {
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units},
      grid,
      parameters: fixture.input(
        'parameters',
        getGPUPolygonRasterizationParameterValues(grid),
        'float32',
        4
      ),
      source: zones(source, 'source'),
      target: zones(target, 'target'),
      pairCapacity: 2,
      sourceValues: fixture.input('source-values', Float32Array.of(100), 'float32', 1),
      outputs: {sourceAreas: sourceAreas.view, extensiveValues: extensiveValues.view}
    });
    fixture.run();
    // Binary64 oracle: the WGS84 authalic radius and latitude-strip formula.
    const radius = 6371007.180918474 / (units === 'kilometers' ? 1000 : 1);
    const radians = Math.PI / 180;
    const area = radius * radius * radians * Math.sin(2 * radians);
    expect(isClose((await fixture.readFloat32(sourceAreas, 1))[0], area, 0, 1e-6)).toBe(true);
    const transferred = await fixture.readFloat32(extensiveValues, 2);
    const firstShare = Math.sin(radians) / Math.sin(2 * radians);
    expect(isClose(transferred[0], 100 * firstShare, 1e-4, 1e-5)).toBe(true);
    expect(isClose(transferred[1], 100 * (1 - firstShare), 1e-4, 1e-5)).toBe(true);
  } finally {
    fixture.destroy();
  }
});
