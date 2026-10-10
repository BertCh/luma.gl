// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUCalendarBucketsParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '../../../src/gpu-dataframe/index';
import {
  getGPUEmergingHotSpotParameterValues,
  GPU_EMERGING_HOT_SPOT_CATEGORIES
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {
  computeEmergingHotSpotCells,
  computeSpaceTimeGiStar,
  createSeededRandom
} from '../emerging-hot-spots/emerging-hot-spots-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';

const WIDTH = 6;
const HEIGHT = 5;
const SLICES = 12;
const CELLS = WIDTH * HEIGHT;

/** Events on a 6x5 lattice over twelve months of 2023; two cells have designed trends. */
function createEvents() {
  const random = createSeededRandom(31);
  const positions: number[] = [];
  const words: number[] = [];
  const cellIds: number[] = [];
  const counts = new Uint32Array(CELLS * SLICES);
  for (let cell = 0; cell < CELLS; cell++) {
    for (let month = 0; month < SLICES; month++) {
      let eventCount = 2 + Math.floor(random() * 3);
      if (cell === 1 * WIDTH + 1) {
        eventCount += 3 * month;
      } else if (cell === 3 * WIDTH + 4) {
        eventCount += 25;
      }
      for (let event = 0; event < eventCount; event++) {
        const x = (cell % WIDTH) + 0.05 + 0.9 * random();
        const y = Math.floor(cell / WIDTH) + 0.05 + 0.9 * random();
        const milliseconds = Date.UTC(2023, month, 1 + Math.floor(random() * 28), 12);
        positions.push(x, y);
        words.push(milliseconds % 2 ** 32, Math.floor(milliseconds / 2 ** 32));
        cellIds.push(cell);
        counts[cell * SLICES + month]++;
      }
    }
  }
  return {
    positions: Float32Array.from(positions),
    timestamps: Uint32Array.from(words),
    cellIds: Uint32Array.from(cellIds),
    counts
  };
}

it('addSpaceTimeHotSpotsRecipe counts events into a cube and classifies emerging hot spots', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const events = createEvents();
  const eventCount = events.cellIds.length;
  const cube = {gridWidth: WIDTH, gridHeight: HEIGHT, sliceCount: SLICES, values: events.counts};
  const frame = {radius: 1.5, temporalWindow: 1};
  const packed = getGPUEmergingHotSpotParameterValues(frame);
  const palette = Array.from({length: 17}, (_, category) =>
    packGPUColor(category * 14, 255 - category * 14, 128)
  );

  const fixture = new RecipeTestFixture(device, 'space-time-hot-spots-recipe-test');
  try {
    const cubeOutput = fixture.output('cube', 'uint32', CELLS * SLICES);
    const giZScores = fixture.output('gi-z', 'float32', CELLS * SLICES);
    const category = fixture.output('category', 'uint32', CELLS);
    const hotSlices = fixture.output('hot-slices', 'uint32', CELLS);
    const trendS = fixture.output('trend-s', 'sint32', CELLS);
    const colors = fixture.output('colors', 'uint32', CELLS);
    const recipe = addSpaceTimeHotSpotsRecipe(fixture.graph, {
      timestamps: fixture.input('timestamps', events.timestamps, 'uint32x2', eventCount),
      calendarParameters: fixture.input(
        'calendar-parameters',
        getGPUCalendarBucketsParameterValues(0),
        'sint32',
        2
      ),
      slices: {field: 'month', firstValue: 1, count: SLICES},
      cells: {
        kind: 'lattice',
        positions: fixture.input('positions', events.positions, 'float32x2', eventCount),
        width: WIDTH,
        height: HEIGHT,
        bounds: [0, 0, WIDTH, HEIGHT]
      },
      parameters: fixture.parameters('emerging-parameters', 'float32', packed),
      outputs: {
        cube: cubeOutput.view,
        giZScores: giZScores.view,
        category: category.view,
        hotSliceCount: hotSlices.view,
        trendS: trendS.view,
        colors: colors.view
      },
      color: {
        palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 17),
        parameters: fixture.parameters(
          'color-parameters',
          'float32',
          getGPUColorScaleParameterValues({scale: 'ordinal', domainCount: 0, paletteCount: 17})
        )
      }
    });
    expect(recipe.contributors.length).toBe(4);
    fixture.run();

    // The dense cube equals the CPU histogram exactly.
    const gpuCube = await fixture.readUint32(cubeOutput, CELLS * SLICES);
    expect(gpuCube).toEqual([...events.counts]);
    expect(gpuCube.reduce((sum, value) => sum + value, 0)).toBe(eventCount);

    const oracleZ = computeSpaceTimeGiStar(cube, packed, 4);
    const z = await fixture.readFloat32(giZScores, CELLS * SLICES);
    for (let bin = 0; bin < z.length; bin++) {
      expect(isClose(z[bin], oracleZ[bin], 2e-3, 1e-3), `z bin ${bin}`).toBe(true);
    }
    const oracle = computeEmergingHotSpotCells(z, cube, packed);
    const gpuCategory = await fixture.readUint32(category, CELLS);
    const level = packed[3];
    let compared = 0;
    for (let cell = 0; cell < CELLS; cell++) {
      if (Math.abs(oracle.trendP[cell] - level) > 0.02 * level) {
        expect(gpuCategory[cell], `category ${cell}`).toBe(oracle.category[cell]);
        compared++;
      }
    }
    expect(compared).toBeGreaterThan(CELLS / 2);
    expect(await fixture.readUint32(hotSlices, CELLS)).toEqual(oracle.hotSliceCount);
    expect(await fixture.readInt32(trendS, CELLS)).toEqual(oracle.trendS);

    // The designed clusters produce hot categories (1-8), and the quiet corner has no pattern.
    const hotCategories = gpuCategory.filter(
      value =>
        value >= GPU_EMERGING_HOT_SPOT_CATEGORIES.NEW_HOT &&
        value <= GPU_EMERGING_HOT_SPOT_CATEGORIES.HISTORICAL_HOT
    );
    expect(hotCategories.length).toBeGreaterThanOrEqual(3);
    expect(gpuCategory[WIDTH * 4]).toBe(GPU_EMERGING_HOT_SPOT_CATEGORIES.NO_PATTERN);
    expect(Math.max(...(await fixture.readUint32(hotSlices, CELLS)))).toBeGreaterThan(0);

    const gpuColors = await fixture.readUint32(colors, CELLS);
    for (let cell = 0; cell < CELLS; cell++) {
      expect(gpuColors[cell]).toBe(palette[gpuCategory[cell]]);
    }
  } finally {
    fixture.destroy();
  }
}, 120000);

it('addSpaceTimeHotSpotsRecipe accepts cell ids with weights and skips out-of-range slices', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const events = createEvents();
  const eventCount = events.cellIds.length;
  // Restrict to the first six months: later events fall outside the slices and are dropped.
  const sliceCount = 6;
  const expected = new Uint32Array(CELLS * sliceCount);
  for (let cell = 0; cell < CELLS; cell++) {
    for (let slice = 0; slice < sliceCount; slice++) {
      expected[cell * sliceCount + slice] = events.counts[cell * SLICES + slice];
    }
  }
  // Rook weights of the lattice, binary.
  const offsets = [0];
  const neighbors: number[] = [];
  for (let cell = 0; cell < CELLS; cell++) {
    const x = cell % WIDTH;
    const y = Math.floor(cell / WIDTH);
    if (y > 0) neighbors.push(cell - WIDTH);
    if (x > 0) neighbors.push(cell - 1);
    if (x < WIDTH - 1) neighbors.push(cell + 1);
    if (y < HEIGHT - 1) neighbors.push(cell + WIDTH);
    offsets.push(neighbors.length);
  }
  const fixture = new RecipeTestFixture(device, 'space-time-hot-spots-ids-test');
  try {
    const cubeOutput = fixture.output('cube', 'uint32', CELLS * sliceCount);
    const giZScores = fixture.output('gi-z', 'float32', CELLS * sliceCount);
    addSpaceTimeHotSpotsRecipe(fixture.graph, {
      timestamps: fixture.input('timestamps', events.timestamps, 'uint32x2', eventCount),
      calendarParameters: fixture.input(
        'calendar-parameters',
        getGPUCalendarBucketsParameterValues(0),
        'sint32',
        2
      ),
      slices: {field: 'month', firstValue: 1, count: sliceCount},
      cells: {
        kind: 'ids',
        cellIds: fixture.input('cell-ids', events.cellIds, 'uint32', eventCount),
        cellCount: CELLS,
        weights: {
          offsets: fixture.input('offsets', Uint32Array.from(offsets), 'uint32', CELLS + 1),
          neighbors: fixture.input(
            'neighbors',
            Uint32Array.from(neighbors),
            'uint32',
            neighbors.length
          ),
          weights: fixture.input(
            'weights',
            new Float32Array(neighbors.length).fill(1),
            'float32',
            neighbors.length
          )
        }
      },
      parameters: fixture.parameters(
        'emerging-parameters',
        'float32',
        getGPUEmergingHotSpotParameterValues({temporalWindow: 1})
      ),
      outputs: {cube: cubeOutput.view, giZScores: giZScores.view}
    });
    fixture.run();
    expect(await fixture.readUint32(cubeOutput, CELLS * sliceCount)).toEqual([...expected]);
    const z = await fixture.readFloat32(giZScores, CELLS * sliceCount);
    expect(z.filter(Number.isFinite).length).toBe(CELLS * sliceCount);
    expect(Math.max(...z)).toBeGreaterThan(1.5);
  } finally {
    fixture.destroy();
  }
}, 120000);
