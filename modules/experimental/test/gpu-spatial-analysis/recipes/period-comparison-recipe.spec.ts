// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '../../../src/gpu-dataframe/column-classification/index';
import {addPeriodComparisonRecipe} from '../../../src/gpu-spatial-analysis/recipes/period-comparison-recipe';
import {
  aggregateCellsOnCPU,
  getQuadbinPointKeys,
  joinCellKey
} from '../cell-aggregation/cell-aggregation-oracle';
import {compareCellTablesOnCPU} from '../cell-table-compare/cell-table-compare-oracle';
import {createSeededRandom} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';

const RESOLUTION = 8;
const TABLE_CAPACITY = 96;
const UNION_CAPACITY = 160;
const PALETTE = [
  packGPUColor(33, 102, 172),
  packGPUColor(146, 197, 222),
  packGPUColor(247, 247, 247),
  packGPUColor(244, 165, 130),
  packGPUColor(178, 24, 43)
];
const EDGES = [-1e6, -8, -2, 2, 8, 1e6];

/** Two periods over the same area: the cluster moves from the south-west to the north-east. */
function createPeriods() {
  const random = createSeededRandom(77);
  const scatter = (count: number, west: number, south: number, size: number) => {
    const points: number[] = [];
    for (let index = 0; index < count; index++) {
      points.push(west + random() * size, south + random() * size);
    }
    return points;
  };
  const before = Float32Array.from([...scatter(300, 10, 10, 8), ...scatter(250, 10.2, 10.2, 2.5)]);
  const after = Float32Array.from([...scatter(300, 10, 10, 8), ...scatter(250, 15, 15, 2.5)]);
  return {before, after};
}

function toOracleCells(positions: Float32Array) {
  return aggregateCellsOnCPU({
    family: 'quadbin',
    resolution: RESOLUTION,
    keys: getQuadbinPointKeys(positions, RESOLUTION),
    sumScale: 65536
  });
}

it('addPeriodComparisonRecipe aggregates two periods, compares them and colors the change', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const periods = createPeriods();
  const oracle = compareCellTablesOnCPU(
    toOracleCells(periods.before),
    toOracleCells(periods.after),
    {measure: 'count', sumScale: 65536, zScore: 'poisson', capacity: UNION_CAPACITY}
  );
  expect(oracle.rows.length).toBeGreaterThan(30);
  expect(oracle.total).toBeLessThanOrEqual(UNION_CAPACITY);

  const fixture = new RecipeTestFixture(device, 'period-comparison-recipe-test');
  try {
    const cells = fixture.output('union-cells', 'uint32x2', UNION_CAPACITY);
    const delta = fixture.output('delta', 'float32', UNION_CAPACITY);
    const zScore = fixture.output('z-score', 'float32', UNION_CAPACITY);
    const count = fixture.output('union-count', 'uint32', 1);
    const overflow = fixture.output('union-overflow', 'uint32', 1);
    const breaks = fixture.output('breaks', 'float32', 6);
    const classCount = fixture.output('class-count', 'uint32', 1);
    const colors = fixture.output('colors', 'uint32', UNION_CAPACITY);
    const classIndices = fixture.output('class-indices', 'uint32', UNION_CAPACITY);
    const recipe = addPeriodComparisonRecipe(fixture.graph, {
      family: 'quadbin',
      resolution: RESOLUTION,
      tableCapacity: TABLE_CAPACITY,
      unionCapacity: UNION_CAPACITY,
      before: {
        positions: fixture.input(
          'before-positions',
          periods.before,
          'float32x2',
          periods.before.length / 2
        )
      },
      after: {
        positions: fixture.input(
          'after-positions',
          periods.after,
          'float32x2',
          periods.after.length / 2
        )
      },
      output: {
        cells: cells.view,
        delta: delta.view,
        zScore: zScore.view,
        count: count.view,
        overflow: overflow.view
      },
      classify: 'delta',
      classBreaksParameters: fixture.parameters(
        'class-breaks-parameters',
        'float32',
        getGPUClassBreaksParameterValues({method: 'custom', customEdges: EDGES}, 5)
      ),
      maximumClassCount: 5,
      methods: ['custom'],
      palette: fixture.input('palette', Uint32Array.from(PALETTE), 'uint32', 5),
      colorScaleParameters: fixture.parameters(
        'color-scale-parameters',
        'float32',
        getGPUColorScaleParameterValues({
          scale: 'threshold',
          domainCount: 6,
          paletteCount: 5,
          noDataColor: packGPUColor(0, 0, 0, 0)
        })
      ),
      maximumPaletteCount: 5,
      breaks: breaks.view,
      classCount: classCount.view,
      colors: colors.view,
      classIndices: classIndices.view
    });
    expect(recipe.contributors.length).toBe(5);
    fixture.run();

    expect((await fixture.readUint32(overflow, 1))[0]).toBe(0);
    const [unionCount] = await fixture.readUint32(count, 1);
    expect(unionCount).toBe(oracle.rows.length);
    const words = await fixture.readUint32(cells, 2 * UNION_CAPACITY);
    const gpuDelta = await fixture.readFloat32(delta, UNION_CAPACITY);
    const gpuZ = await fixture.readFloat32(zScore, UNION_CAPACITY);
    let gained = 0;
    let lost = 0;
    for (const [row, expected] of oracle.rows.entries()) {
      expect(joinCellKey(words[2 * row], words[2 * row + 1]), `key ${row}`).toBe(expected.key);
      expect(gpuDelta[row], `delta ${row}`).toBe(expected.delta);
      expect(isClose(gpuZ[row], expected.zScore, 1e-4, 1e-4), `z ${row}`).toBe(true);
      gained += expected.delta > 8 ? 1 : 0;
      lost += expected.delta < -8 ? 1 : 0;
    }
    // The moving cluster produces both gains and losses.
    expect(gained).toBeGreaterThan(0);
    expect(lost).toBeGreaterThan(0);

    // Custom diverging edges are used as given; classes follow the inner edges.
    expect((await fixture.readUint32(classCount, 1))[0]).toBe(5);
    expect(await fixture.readFloat32(breaks, 6)).toEqual(EDGES);
    const classes = await fixture.readUint32(classIndices, UNION_CAPACITY);
    const gpuColors = await fixture.readUint32(colors, UNION_CAPACITY);
    const used = new Set<number>();
    for (const [row, expected] of oracle.rows.entries()) {
      const expectedClass = EDGES.slice(1, 5).filter(edge => edge <= expected.delta).length;
      expect(classes[row], `class ${row}`).toBe(expectedClass);
      expect(gpuColors[row]).toBe(PALETTE[expectedClass]);
      used.add(expectedClass);
    }
    expect(used.has(0)).toBe(true);
    expect(used.has(4)).toBe(true);
    // The unused tail is no-data.
    for (let row = unionCount; row < UNION_CAPACITY; row++) {
      expect(classes[row]).toBe(0xffffffff);
    }
  } finally {
    fixture.destroy();
  }
}, 120000);
