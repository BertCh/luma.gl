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
  getGPUEmergingHotSpotsParameterValues,
  GPU_EMERGING_HOT_SPOTS_CATEGORIES,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {
  computeEmergingHotSpotCells,
  computeSpaceTimeGiStar,
  createSeededRandom,
  SLICE_CASES,
  SLICE_EVENTS
} from '../emerging-hot-spots/emerging-hot-spots-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';

const WIDTH = 6;
const HEIGHT = 5;
const SLICES = 12;
const CELLS = WIDTH * HEIGHT;

/** Events on a 6x5 lattice over twelve months of 2023: a hot block with growing counts, a cold block. */
function createEvents() {
  const random = createSeededRandom(31);
  const positions: number[] = [];
  const words: number[] = [];
  const cellIds: number[] = [];
  const counts = new Uint32Array(CELLS * SLICES);
  for (let cell = 0; cell < CELLS; cell++) {
    for (let month = 0; month < SLICES; month++) {
      const column = cell % WIDTH;
      const row = Math.floor(cell / WIDTH);
      let eventCount = 8 + Math.floor(random() * 5);
      if (column >= 1 && column <= 3 && row >= 1 && row <= 2) {
        // A hot block whose counts grow every month.
        eventCount += 6 * month;
      } else if (column >= 4 && row >= 3) {
        // A cold block with almost no events.
        eventCount = Math.floor(random() * 2);
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
  requireTestDevice(device);
  const events = createEvents();
  const eventCount = events.cellIds.length;
  const cube = {gridWidth: WIDTH, gridHeight: HEIGHT, sliceCount: SLICES, values: events.counts};
  const frame = {radius: 1.5, temporalWindow: 1};
  // The oracle reads the same packed layout the recipe writes from `frame`.
  const packed = getGPUEmergingHotSpotsParameterValues(frame);
  // 18 codes: the 17 ArcGIS categories and UNDETERMINED.
  const palette = Array.from({length: 18}, (_, category) =>
    packGPUColor(category * 14, 255 - category * 14, 128)
  );

  const fixture = new RecipeTestFixture(device, 'space-time-hot-spots-recipe-test');
  try {
    const cubeOutput = fixture.output('cube', 'uint32', CELLS * SLICES);
    const giZScores = fixture.output('gi-z', 'float32', CELLS * SLICES);
    const adjustedPValues = fixture.output('adjusted-p', 'float32', CELLS * SLICES);
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
      slices: {step: 'month', origin: {year: 2023, month: 1}, count: SLICES},
      cells: {
        kind: 'lattice',
        positions: fixture.input('positions', events.positions, 'float32x2', eventCount),
        width: WIDTH,
        height: HEIGHT,
        bounds: [0, 0, WIDTH, HEIGHT]
      },
      settings: frame,
      parameters: fixture.parameterBuffer(
        'emerging-parameters',
        'float32',
        GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
      ),
      outputs: {
        cube: cubeOutput.view,
        giZScores: giZScores.view,
        adjustedPValues: adjustedPValues.view,
        category: category.view,
        hotSliceCount: hotSlices.view,
        trendS: trendS.view,
        colors: colors.view
      },
      color: {
        palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 18),
        parameters: fixture.parameters(
          'color-parameters',
          'float32',
          getGPUColorScaleParameterValues({scale: 'ordinal', domainCount: 0, paletteCount: 18})
        )
      }
    });
    expect(recipe.inference).toEqual({
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: 'benjamini-hochberg',
      // Derived from the packed settings: 1 - the default 0.95 confidence level.
      significanceLevel: expect.closeTo(0.05, 12),
      testedRowsDefinition: 'valid space-time bins'
    });
    expect(Object.getPrototypeOf(recipe.inference)).toBe(Object.prototype);
    // Calendar, emerging hot spots and color scale: the atomic count path adds no contributor.
    expect(recipe.contributors.length).toBe(3);
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
    // Default correction: Benjamini-Hochberg. Classification uses the GPU's adjusted p-values.
    const oracle = computeEmergingHotSpotCells(z, cube, packed, {
      adjustedPValues: await fixture.readFloat32(adjustedPValues, CELLS * SLICES)
    });
    const gpuCategory = await fixture.readUint32(category, CELLS);
    const level = packed[3];
    let compared = 0;
    for (let cell = 0; cell < CELLS; cell++) {
      if (!(Math.abs(oracle.trendP[cell] - level) <= 0.02 * level)) {
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
        value >= GPU_EMERGING_HOT_SPOTS_CATEGORIES.NEW_HOT &&
        value <= GPU_EMERGING_HOT_SPOTS_CATEGORIES.HISTORICAL_HOT
    );
    expect(hotCategories.length).toBeGreaterThanOrEqual(3);
    expect(gpuCategory[WIDTH * 4]).toBe(GPU_EMERGING_HOT_SPOTS_CATEGORIES.NO_PATTERN);
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
  requireTestDevice(device);
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
      slices: {step: 'month', origin: {year: 2023, month: 1}, count: sliceCount},
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
      settings: {temporalWindow: 1},
      parameters: fixture.parameterBuffer(
        'emerging-parameters',
        'float32',
        GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
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

it('addSpaceTimeHotSpotsRecipe atomic counts match the sorted GPUGroupStatistics path bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A few hundred events: a dense hot bin (atomic contention), sparse noise, positions outside the
  // lattice, cell ids out of range, months past the last slice, and masked events.
  const sliceCount = 9;
  const random = createSeededRandom(7);
  const positions: number[] = [];
  const words: number[] = [];
  const cellIds: number[] = [];
  const mask: number[] = [];
  const expected = new Uint32Array(CELLS * sliceCount);
  for (let event = 0; event < 480; event++) {
    const isHot = event < 160;
    const column = isHot ? 2 : Math.floor(random() * (WIDTH + 2)) - 1;
    const row = isHot ? 3 : Math.floor(random() * HEIGHT);
    const month = isHot ? 4 : Math.floor(random() * 12);
    const isMasked = !isHot && random() < 0.1;
    const milliseconds = Date.UTC(2023, month, 1 + Math.floor(random() * 28), 6);
    positions.push(column + 0.1 + 0.8 * random(), row + 0.1 + 0.8 * random());
    words.push(milliseconds % 2 ** 32, Math.floor(milliseconds / 2 ** 32));
    const isInside = column >= 0 && column < WIDTH;
    const cell = row * WIDTH + column;
    cellIds.push(isInside ? cell : CELLS + event);
    mask.push(isMasked ? 0 : 1);
    if (isInside && month < sliceCount && !isMasked) {
      expected[cell * sliceCount + month]++;
    }
  }
  const eventCount = mask.length;
  const binCount = CELLS * sliceCount;
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

  const fixture = new RecipeTestFixture(device, 'space-time-hot-spots-paths-test');
  try {
    const timestamps = fixture.input('timestamps', Uint32Array.from(words), 'uint32x2', eventCount);
    const eventMask = fixture.input('mask', Uint32Array.from(mask), 'uint32', eventCount);
    const calendarParameters = fixture.input(
      'calendar-parameters',
      getGPUCalendarBucketsParameterValues(0),
      'sint32',
      2
    );
    const settings = {radius: 1.5, temporalWindow: 1};
    let parameterSerial = 0;
    const createParameters = () =>
      fixture.parameterBuffer(
        `emerging-parameters-${parameterSerial++}`,
        'float32',
        GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
      );
    const lattice = {
      kind: 'lattice',
      positions: fixture.input('positions', Float32Array.from(positions), 'float32x2', eventCount),
      width: WIDTH,
      height: HEIGHT,
      bounds: [0, 0, WIDTH, HEIGHT]
    } as const;
    const ids = {
      kind: 'ids',
      cellIds: fixture.input('cell-ids', Uint32Array.from(cellIds), 'uint32', eventCount),
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
    } as const;

    const runs = [];
    for (const [cellsName, cells] of [
      ['lattice', lattice],
      ['ids', ids]
    ] as const) {
      for (const path of ['atomic', 'sorted'] as const) {
        const name = `${cellsName}-${path}`;
        const cube = fixture.output(`${name}-cube`, 'uint32', binCount);
        const giZScores = fixture.output(`${name}-gi-z`, 'float32', binCount);
        const category = fixture.output(`${name}-category`, 'uint32', CELLS);
        const groupOverflow = fixture.output(`${name}-overflow`, 'uint32', 1);
        const recipe = addSpaceTimeHotSpotsRecipe(fixture.graph, {
          id: name,
          timestamps,
          mask: eventMask,
          calendarParameters,
          slices: {step: 'month', origin: {year: 2023, month: 1}, count: sliceCount},
          cells,
          settings,
          parameters: createParameters(),
          outputs: {
            cube: cube.view,
            giZScores: giZScores.view,
            category: category.view,
            // Asking for the overflow flag selects the sorted GPUGroupStatistics path.
            ...(path === 'sorted' ? {status: {groupOverflow: groupOverflow.view}} : {})
          }
        });
        expect(recipe.contributors.length, name).toBe(path === 'sorted' ? 3 : 2);
        // The slice-range stage is always present; the sorted path adds cube-counts.
        expect(recipe.status.stages.length, name).toBe(path === 'sorted' ? 2 : 1);
        runs.push({name, path, cube, giZScores, category, groupOverflow});
      }
    }
    fixture.run();

    for (let index = 0; index < runs.length; index += 2) {
      const atomic = runs[index];
      const sorted = runs[index + 1];
      const atomicCube = await fixture.readUint32(atomic.cube, binCount);
      expect(atomicCube, atomic.name).toEqual([...expected]);
      expect(await fixture.readUint32(sorted.cube, binCount), sorted.name).toEqual(atomicCube);
      expect(await fixture.readUint32(sorted.groupOverflow, 1), sorted.name).toEqual([0]);
      // Same cube, same downstream kernels: the float32 z-scores agree in every bit.
      expect(await fixture.readUint32(sorted.giZScores, binCount), sorted.name).toEqual(
        await fixture.readUint32(atomic.giZScores, binCount)
      );
      expect(await fixture.readUint32(sorted.category, CELLS), sorted.name).toEqual(
        await fixture.readUint32(atomic.category, CELLS)
      );
    }
    expect(expected[(3 * WIDTH + 2) * sliceCount + 4]).toBeGreaterThanOrEqual(160);
  } finally {
    fixture.destroy();
  }
}, 120000);

it('addSpaceTimeHotSpotsRecipe bins absolute time steps across year boundaries (python calendar reference)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One cell, one event per reference timestamp. Expected slices come from python datetime
  // arithmetic (scratchpad/fix/work/B/b5_emerging_ref.py), independent of the shader formulas.
  const eventCount = SLICE_EVENTS.length;
  const words: number[] = [];
  for (const [year, month, day, hour] of SLICE_EVENTS) {
    const milliseconds = Date.UTC(year, month - 1, day, hour);
    words.push(milliseconds % 2 ** 32, Math.floor(milliseconds / 2 ** 32));
  }
  const fixture = new RecipeTestFixture(device, 'space-time-hot-spots-steps-test');
  try {
    const timestamps = fixture.input('timestamps', Uint32Array.from(words), 'uint32x2', eventCount);
    const positions = fixture.input(
      'positions',
      new Float32Array(eventCount * 2).fill(0.5),
      'float32x2',
      eventCount
    );
    const calendarParameters = fixture.input(
      'calendar-parameters',
      getGPUCalendarBucketsParameterValues(0),
      'sint32',
      2
    );
    const settings = {radius: 0, temporalWindow: 0};
    let parameterSerial = 0;
    const createParameters = () =>
      fixture.parameterBuffer(
        `emerging-parameters-${parameterSerial++}`,
        'float32',
        GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
      );
    const runs = SLICE_CASES.map((testCase, index) => {
      const [year, month, day, hour] = testCase.origin;
      const name = `${testCase.step}-${testCase.interval}-${index}`;
      const cube = fixture.output(`${name}-cube`, 'uint32', testCase.count);
      const skipped = fixture.output(`${name}-skipped`, 'uint32', 1);
      const recipe = addSpaceTimeHotSpotsRecipe(fixture.graph, {
        id: name,
        timestamps,
        calendarParameters,
        slices: {
          step: testCase.step,
          interval: testCase.interval,
          origin: {year, month, day, hour},
          count: testCase.count
        },
        cells: {
          kind: 'lattice',
          positions,
          width: 1,
          height: 1,
          bounds: [0, 0, 1, 1]
        },
        settings,
        parameters: createParameters(),
        outputs: {cube: cube.view, status: {skippedEventCount: skipped.view}}
      });
      // The slice-range stage exposes the same count as invalidCount.
      expect(recipe.status.stages[0].stage).toBe('slice-range');
      return {testCase, name, cube, skipped};
    });
    fixture.run();
    for (const {testCase, name, cube, skipped} of runs) {
      const expected = new Array<number>(testCase.count).fill(0);
      let expectedSkipped = 0;
      for (const slice of testCase.expected) {
        if (slice >= 0) {
          expected[slice]++;
        } else {
          expectedSkipped++;
        }
      }
      expect(await fixture.readUint32(cube, testCase.count), name).toEqual(expected);
      expect(await fixture.readUint32(skipped, 1), name).toEqual([expectedSkipped]);
    }
  } finally {
    fixture.destroy();
  }
}, 120000);
