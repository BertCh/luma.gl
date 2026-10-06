// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {getGPUDistanceFieldParameterValues} from '../../../src/gpu-raster/distance-field/index';
import {addStraightLineCatchmentsRecipe} from '../../../src/gpu-spatial-analysis/recipes/straight-line-catchments-recipe';
import {createSeededRandom} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {RecipeTestFixture} from './recipe-harness';

const NONE = 0xffffffff;
const WIDTH = 40;
const HEIGHT = 30;
const CELL = 2;

it('addStraightLineCatchmentsRecipe allocates cells to the nearest facility and summarises values', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createSeededRandom(4);
  const seeds = Float32Array.from([
    10.5, 12.5, 61.5, 15.5, 33.5, 50.5, 70.5, 8.5, 20.5, 25.5, 500, 500 /* off the grid */
  ]);
  const seedCount = seeds.length / 2;
  const values = new Float32Array(WIDTH * HEIGHT);
  for (let cell = 0; cell < values.length; cell++) {
    values[cell] = Math.floor(random() * 50);
  }
  values[7] = NaN;

  // CPU oracle: each seed snaps to its cell; nearest seed cell per cell center, smallest id on ties.
  const seedCells = Array.from({length: seedCount}, (_, id) => {
    const x = Math.floor(seeds[2 * id] / CELL);
    const y = Math.floor(seeds[2 * id + 1] / CELL);
    return x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT ? {id, x, y} : null;
  }).filter(cell => cell !== null);
  const allocation = new Uint32Array(WIDTH * HEIGHT);
  const distances = new Float64Array(WIDTH * HEIGHT);
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      let best = NONE;
      let bestDistance = Infinity;
      for (const seed of seedCells) {
        const distance = Math.hypot((x - seed.x) * CELL, (y - seed.y) * CELL);
        if (distance < bestDistance || (distance === bestDistance && seed.id < best)) {
          bestDistance = distance;
          best = seed.id;
        }
      }
      allocation[y * WIDTH + x] = best;
      distances[y * WIDTH + x] = bestDistance;
    }
  }
  const expected = Array.from({length: seedCount}, () => ({
    cells: 0,
    valid: 0,
    sum: 0,
    min: Infinity,
    max: -Infinity
  }));
  for (let cell = 0; cell < allocation.length; cell++) {
    const zone = allocation[cell];
    expected[zone].cells++;
    if (Number.isFinite(values[cell])) {
      expected[zone].valid++;
      expected[zone].sum += values[cell];
      expected[zone].min = Math.min(expected[zone].min, values[cell]);
      expected[zone].max = Math.max(expected[zone].max, values[cell]);
    }
  }

  const fixture = new RecipeTestFixture(device, 'straight-line-catchments-test');
  try {
    const outputs = {
      allocation: fixture.output('allocation', 'uint32', WIDTH * HEIGHT),
      distances: fixture.output('distances', 'float32', WIDTH * HEIGHT),
      cellCounts: fixture.output('cell-counts', 'uint32', seedCount),
      valueCounts: fixture.output('value-counts', 'uint32', seedCount),
      sums: fixture.output('sums', 'float32', seedCount),
      means: fixture.output('means', 'float32', seedCount),
      minimums: fixture.output('minimums', 'float32', seedCount),
      maximums: fixture.output('maximums', 'float32', seedCount)
    };
    const recipe = addStraightLineCatchmentsRecipe(fixture.graph, {
      width: WIDTH,
      height: HEIGHT,
      settings: fixture.parameters(
        'distance-settings',
        'float32',
        getGPUDistanceFieldParameterValues({origin: [0, 0], cellSize: [CELL, CELL]})
      ),
      seedPositions: fixture.input('seeds', seeds, 'float32x2', seedCount),
      values: fixture.input('values', values, 'float32', values.length),
      sumOrder: 'sorted',
      allocation: outputs.allocation.view,
      distances: outputs.distances.view,
      statistics: {
        cellCounts: outputs.cellCounts.view,
        valueCounts: outputs.valueCounts.view,
        sums: outputs.sums.view,
        means: outputs.means.view,
        minimums: outputs.minimums.view,
        maximums: outputs.maximums.view
      }
    });
    expect(recipe.contributors.length).toBe(2);
    fixture.run();

    const gpuAllocation = await fixture.readUint32(outputs.allocation, allocation.length);
    expect(gpuAllocation).toEqual(Array.from(allocation));
    expect(new Set(gpuAllocation).size).toBe(seedCells.length);
    const gpuDistances = await fixture.readFloat32(outputs.distances, allocation.length);
    for (let cell = 0; cell < allocation.length; cell += 7) {
      expect(gpuDistances[cell]).toBeCloseTo(distances[cell], 3);
    }
    const cellCounts = await fixture.readUint32(outputs.cellCounts, seedCount);
    const valueCounts = await fixture.readUint32(outputs.valueCounts, seedCount);
    const sums = await fixture.readFloat32(outputs.sums, seedCount);
    const means = await fixture.readFloat32(outputs.means, seedCount);
    const minimums = await fixture.readFloat32(outputs.minimums, seedCount);
    const maximums = await fixture.readFloat32(outputs.maximums, seedCount);
    for (const [zone, want] of expected.entries()) {
      expect(cellCounts[zone], `zone ${zone} cells`).toBe(want.cells);
      expect(valueCounts[zone], `zone ${zone} valid`).toBe(want.valid);
      if (want.valid > 0) {
        expect(sums[zone]).toBeCloseTo(want.sum, 2);
        expect(means[zone]).toBeCloseTo(want.sum / want.valid, 3);
        expect(minimums[zone]).toBe(want.min);
        expect(maximums[zone]).toBe(want.max);
      }
    }
    expect(cellCounts.reduce((total, value) => total + value, 0)).toBe(WIDTH * HEIGHT);
    expect(Math.min(...expected.slice(0, 5).map(zone => zone.cells))).toBeGreaterThan(50);
  } finally {
    fixture.destroy();
  }
}, 120000);
