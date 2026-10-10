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
import {getGPUNeighborSearchParameterValues} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/index';
import {
  aggregateCellsOnCPU,
  getQuadbinPointKeys,
  quadbinCellToTile
} from '../cell-aggregation/cell-aggregation-oracle';
import {webMercatorTileCenter} from '../cell-indexing/cell-indexing-oracle';
import {
  computeHotSpotOracle,
  createDistanceBandWeights,
  createSeededRandom
} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {computeLocalPermutationOracle} from '../permutation-inference/permutation-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';

const RESOLUTION = 8;
const TABLE_CAPACITY = 128;
const NEIGHBOR_CAPACITY = TABLE_CAPACITY * 12;
/** Cell centers are 1.40625 degrees apart in longitude; the radius picks the 4-neighborhood. */
const RADIUS = 1.7;

/** Points on lng 10..20, lat 10..20 with a dense cluster that forms a hot spot. */
function createScene(): Float32Array {
  const random = createSeededRandom(5);
  const points: number[] = [];
  for (let index = 0; index < 500; index++) {
    points.push(10 + random() * 10, 10 + random() * 10);
  }
  for (let index = 0; index < 400; index++) {
    points.push(14 + random() * 1.5, 14 + random() * 1.5);
  }
  return Float32Array.from(points);
}

/** CPU twin of the points source: keys, table rows, f32 cell centers and counts. */
function getCellTableOracle(positions: Float32Array) {
  const cells = aggregateCellsOnCPU({
    family: 'quadbin',
    resolution: RESOLUTION,
    keys: getQuadbinPointKeys(positions, RESOLUTION),
    sumScale: 65536
  });
  const centers = new Float32Array(cells.length * 2);
  for (const [row, cell] of cells.entries()) {
    const {x, y, z} = quadbinCellToTile(cell.key);
    const [longitude, latitude] = webMercatorTileCenter(x, y, z);
    centers[2 * row] = longitude;
    centers[2 * row + 1] = latitude;
  }
  return {cells, centers, counts: Float32Array.from(cells.map(cell => cell.count))};
}

it('addHotSpotAnalysisRecipe bins points, links cell centers and matches the Gi* oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createScene();
  const pointCount = positions.length / 2;
  const table = getCellTableOracle(positions);
  const rows = table.cells.length;
  const weights = createDistanceBandWeights(table.centers, RADIUS);
  const oracle = computeHotSpotOracle({weights, values: table.counts});
  const permutationParameters = {seed: 7, permutations: 199, significanceLevel: 0.05};
  const permutationOracle = computeLocalPermutationOracle({
    weights,
    values: table.counts,
    statistic: 'localGStar',
    maximumNeighbors: 32,
    ...permutationParameters
  });

  const fixture = new RecipeTestFixture(device, 'hot-spot-recipe-test');
  try {
    const {graph} = fixture;
    const tableCount = fixture.output('table-count', 'uint32', 1);
    const zScores = fixture.output('z-scores', 'float32', TABLE_CAPACITY);
    const bins = fixture.output('bins', 'sint32', TABLE_CAPACITY);
    const neighborCounts = fixture.output('neighbor-counts', 'uint32', TABLE_CAPACITY);
    const exceedances = fixture.output('exceedances', 'uint32', TABLE_CAPACITY);
    const pseudoPValues = fixture.output('pseudo-p-values', 'float32', TABLE_CAPACITY);
    const significant = fixture.output('significant', 'uint32', TABLE_CAPACITY);
    const breaks = fixture.output('breaks', 'float32', 6);
    const classCount = fixture.output('class-count', 'uint32', 1);
    const colors = fixture.output('colors', 'uint32', TABLE_CAPACITY);
    const classIndices = fixture.output('class-indices', 'uint32', TABLE_CAPACITY);
    const palette = [
      packGPUColor(10, 20, 200),
      packGPUColor(90, 120, 230),
      packGPUColor(230, 230, 230),
      packGPUColor(240, 140, 90),
      packGPUColor(200, 20, 10)
    ];
    const recipe = addHotSpotAnalysisRecipe(graph, {
      source: {
        kind: 'points',
        positions: fixture.input('positions', positions, 'float32x2', pointCount),
        family: 'quadbin',
        resolution: RESOLUTION,
        tableCapacity: TABLE_CAPACITY,
        neighborCapacity: NEIGHBOR_CAPACITY,
        gridSize: [16, 16],
        neighborSearchParameters: fixture.parameters(
          'neighbor-parameters',
          'float32',
          getGPUNeighborSearchParameterValues({bounds: [5, 5, 25, 25], radius: RADIUS})
        )
      },
      parameters: fixture.parameters(
        'autocorrelation-parameters',
        'float32',
        getGPUSpatialAutocorrelationParameterValues({})
      ),
      scratch: {table: {count: tableCount.view}},
      outputs: {
        zScores: zScores.view,
        bins: bins.view,
        neighborCounts: neighborCounts.view,
        permutation: {
          exceedances: exceedances.view,
          pseudoPValues: pseudoPValues.view,
          significant: significant.view
        },
        color: {
          breaks: breaks.view,
          classCount: classCount.view,
          colors: colors.view,
          classIndices: classIndices.view
        }
      },
      permutation: {
        parameters: fixture.parameters(
          'permutation-parameters',
          'uint32',
          getGPUPermutationParameterValues(permutationParameters)
        ),
        maximumPermutations: 199
      },
      color: {
        classBreaksParameters: fixture.parameters(
          'class-breaks-parameters',
          'float32',
          getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 5}, 5)
        ),
        maximumClassCount: 5,
        methods: ['equal-interval'],
        colorScaleParameters: fixture.parameters(
          'color-scale-parameters',
          'float32',
          getGPUColorScaleParameterValues({scale: 'quantile', domainCount: 6, paletteCount: 5})
        ),
        palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 5),
        maximumPaletteCount: 5
      }
    });
    expect(recipe.contributors.length).toBe(7);
    fixture.run();

    const [count] = await fixture.readUint32(tableCount, 1);
    expect(count).toBe(rows);
    expect(count).toBeGreaterThan(20);
    const z = await fixture.readFloat32(zScores, TABLE_CAPACITY);
    const counts = await fixture.readUint32(neighborCounts, TABLE_CAPACITY);
    let hot = 0;
    for (let row = 0; row < rows; row++) {
      expect(isClose(z[row], oracle.zScores[row], 2e-3, 1e-3), `z row ${row}`).toBe(true);
      expect(counts[row]).toBe(oracle.neighborCounts[row]);
      hot += oracle.zScores[row] > 1.96 ? 1 : 0;
    }
    expect(hot).toBeGreaterThan(0);
    for (let row = rows; row < TABLE_CAPACITY; row++) {
      expect(z[row], `empty row ${row}`).toBeNaN();
    }
    const gpuBins = await fixture.readInt32(bins, rows);
    let comparedBins = 0;
    for (let row = 0; row < rows; row++) {
      const critical = [1.6448536, 1.959964, 2.5758293].some(
        value => Math.abs(Math.abs(oracle.zScores[row]) - value) < 1e-2
      );
      if (!critical) {
        expect(gpuBins[row], `bin row ${row}`).toBe(oracle.bins[row]);
        comparedBins++;
      }
    }
    expect(comparedBins).toBeGreaterThan(rows / 2);
    expect(Math.max(...gpuBins)).toBeGreaterThan(0);

    // Permutation confirmation of the same chain.
    const gpuExceedances = await fixture.readUint32(exceedances, rows);
    let mismatched = 0;
    for (let row = 0; row < rows; row++) {
      if (gpuExceedances[row] !== permutationOracle.exceedances[row]) {
        mismatched++;
        expect(
          Math.abs(gpuExceedances[row] - permutationOracle.exceedances[row])
        ).toBeLessThanOrEqual(1);
      }
    }
    expect(mismatched).toBeLessThanOrEqual(Math.max(1, rows * 0.02));
    const hotRow = oracle.zScores.indexOf(Math.max(...oracle.zScores));
    const gpuSignificant = await fixture.readUint32(significant, rows);
    expect(permutationOracle.significant[hotRow]).toBe(1);
    expect(gpuSignificant[hotRow]).toBe(1);

    // Class breaks of the z-scores and the threshold color scale of the same rows.
    const finite = oracle.zScores.filter(Number.isFinite);
    const minimum = Math.min(...finite);
    const maximum = Math.max(...finite);
    const gpuBreaks = await fixture.readFloat32(breaks, 6);
    expect((await fixture.readUint32(classCount, 1))[0]).toBe(5);
    for (let edge = 0; edge <= 5; edge++) {
      const expected = minimum + ((maximum - minimum) / 5) * edge;
      expect(isClose(gpuBreaks[edge], expected, 5e-3, 2e-3), `edge ${edge}`).toBe(true);
    }
    const gpuClasses = await fixture.readUint32(classIndices, rows);
    const gpuColors = await fixture.readUint32(colors, rows);
    for (let row = 0; row < rows; row++) {
      let expectedClass = 0;
      for (let edge = 1; edge < 5; edge++) {
        if (gpuBreaks[edge] <= z[row]) {
          expectedClass++;
        }
      }
      expect(gpuClasses[row], `class row ${row}`).toBe(expectedClass);
      expect(gpuColors[row]).toBe(palette[expectedClass]);
    }
    expect(Math.max(...gpuClasses)).toBe(4);
  } finally {
    fixture.destroy();
  }
}, 120000);

it('addHotSpotAnalysisRecipe analyses a dense lattice with queen weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 12;
  const height = 10;
  const random = createSeededRandom(9);
  const values = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      values[y * width + x] = 10 + random() + (x < 4 && y < 4 ? 8 : 0);
    }
  }
  // Queen weights of the lattice, brute force.
  const offsets = [0];
  const neighbors: number[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if ((dx || dy) && nx >= 0 && nx < width && ny >= 0 && ny < height) {
            neighbors.push(ny * width + nx);
          }
        }
      }
      offsets.push(neighbors.length);
    }
  }
  const weights = {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: new Float32Array(neighbors.length).fill(1)
  };
  const oracle = computeHotSpotOracle({weights, values});

  const fixture = new RecipeTestFixture(device, 'hot-spot-recipe-lattice-test');
  try {
    const zScores = fixture.output('z-scores', 'float32', width * height);
    const overflow = fixture.output('weights-overflow', 'uint32', 1);
    addHotSpotAnalysisRecipe(fixture.graph, {
      source: {
        kind: 'lattice',
        values: fixture.input('values', values, 'float32', values.length),
        width,
        height,
        neighborCapacity: width * height * 8
      },
      parameters: fixture.parameters(
        'autocorrelation-parameters',
        'float32',
        getGPUSpatialAutocorrelationParameterValues({})
      ),
      outputs: {weightsOverflow: overflow.view, zScores: zScores.view}
    });
    fixture.run();
    expect((await fixture.readUint32(overflow, 1))[0]).toBe(0);
    const z = await fixture.readFloat32(zScores, width * height);
    for (let row = 0; row < z.length; row++) {
      expect(isClose(z[row], oracle.zScores[row], 2e-3, 1e-3), `z row ${row}`).toBe(true);
    }
    expect(Math.max(...z)).toBeGreaterThan(2);
  } finally {
    fixture.destroy();
  }
}, 120000);
