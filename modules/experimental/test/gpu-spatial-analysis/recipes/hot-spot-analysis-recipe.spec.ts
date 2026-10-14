// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellArea, gridDisk, latLngToCell, polygonToCells} from 'h3-js';
import {expect, it} from 'vitest';
import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '../../../src/gpu-dataframe/column-classification/index';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/index';
import {
  aggregateCellsOnCPU,
  bigIntToH3,
  getQuadbinPointKeys,
  getQuadbinTileX,
  getQuadbinTileY,
  quadbinCellToTile,
  quadbinTileToCell
} from '../cell-aggregation/cell-aggregation-oracle';
import {webMercatorTileBounds} from '../cell-indexing/cell-indexing-oracle';
import {
  computeHotSpotOracle,
  createSeededRandom
} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {computeLocalPermutationOracle} from '../permutation-inference/permutation-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';

const RESOLUTION = 8;
const CELL_CAPACITY = 256;
/** Authalic radius in kilometers, the sphere of H3 `cellAreaKm2` and of the recipe's quadbin areas. */
const AUTHALIC_RADIUS_KILOMETERS = 6371.007180918475;

/**
 * Points over the whole study area (lng 6..24, lat 6..24, about 12 per cell) with an empty hole at
 * lng 16..22, lat 7..13 (a block of zero cells that forms a cold spot), a dense cluster that forms
 * a hot spot, and 60 points west of the study area. The hot cluster is kept moderate: a heavier one
 * inflates the global standard deviation until no zero block can be significant. The CPU oracle
 * gives 22 empty cells, 12 of them significant cold spots with 8 neighbors (min z -3.2), and 9 hot
 * cells.
 */
function createScene(): Float32Array {
  const random = createSeededRandom(5);
  const points: number[] = [];
  while (points.length < 2 * 2000) {
    const x = 6 + random() * 18;
    const y = 6 + random() * 18;
    if (x >= 16 && x <= 22 && y >= 7 && y <= 13) {
      continue;
    }
    points.push(x, y);
  }
  for (let index = 0; index < 150; index++) {
    points.push(14 + random() * 1.5, 14 + random() * 1.5);
  }
  for (let index = 0; index < 60; index++) {
    points.push(random() * 3, 10 + random() * 10);
  }
  return Float32Array.from(points);
}

/** Tile range of the study area: whole tiles covering lng 6..24, lat 6..24. */
function getStudyTiles() {
  return {
    x0: getQuadbinTileX(6, RESOLUTION),
    x1: getQuadbinTileX(24, RESOLUTION),
    y0: getQuadbinTileY(24, RESOLUTION),
    y1: getQuadbinTileY(6, RESOLUTION)
  };
}

/** CPU twin of the study-area points source for quadbin, double precision. */
function getQuadbinStudyOracle(positions: Float32Array, pointValues?: Float32Array) {
  const {x0, x1, y0, y1} = getStudyTiles();
  const studyKeys: bigint[] = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      studyKeys.push(quadbinTileToCell(x, y, RESOLUTION));
    }
  }
  studyKeys.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const pointCells = aggregateCellsOnCPU({
    family: 'quadbin',
    resolution: RESOLUTION,
    keys: getQuadbinPointKeys(positions, RESOLUTION),
    values: pointValues,
    sumScale: 65536
  });
  const pointByKey = new Map(pointCells.map(cell => [cell.key, cell]));
  const index = new Map(studyKeys.map((key, row) => [key, row]));
  const tiles = studyKeys.map(key => quadbinCellToTile(key));
  const areas = tiles.map(({x, y, z}) => {
    const [west, south, east, north] = webMercatorTileBounds(x, y, z);
    const radians = Math.PI / 180;
    return (
      AUTHALIC_RADIUS_KILOMETERS ** 2 *
      (east - west) *
      radians *
      (Math.sin(north * radians) - Math.sin(south * radians))
    );
  });
  const counts = studyKeys.map(key => pointByKey.get(key)?.count ?? 0);
  const sums = studyKeys.map(key => Number(pointByKey.get(key)?.sum ?? 0n) / 65536);
  const offsets = [0];
  const neighbors: number[] = [];
  for (const {x, y} of tiles) {
    const row: number[] = [];
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const neighbor = index.get(quadbinTileToCell(x + dx, y + dy, RESOLUTION));
        if ((dx || dy) && neighbor !== undefined) {
          row.push(neighbor);
        }
      }
    }
    neighbors.push(...row.sort((a, b) => a - b));
    offsets.push(neighbors.length);
  }
  const pointsInStudy = counts.reduce((total, count) => total + count, 0);
  const totalPoints = pointCells.reduce((total, cell) => total + cell.count, 0);
  return {
    studyKeys,
    counts,
    sums,
    areas,
    offsets,
    neighbors,
    outside: totalPoints - pointsInStudy,
    weights: {
      offsets: Uint32Array.from(offsets),
      neighbors: Uint32Array.from(neighbors),
      weights: new Float32Array(neighbors.length).fill(1)
    }
  };
}

type SceneRun = Awaited<ReturnType<typeof runQuadbinScene>>;

/** Runs the points source on the quadbin scene and reads every output back. */
async function runQuadbinScene(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  options: {
    multipleTesting: 'none' | 'benjamini-hochberg';
    minimumNeighbors?: number;
    extras?: boolean;
    /** Neighbor slots of the grid-disk CSR; defaults to 8 per study row (no truncation). */
    neighborCapacity?: number;
  }
) {
  const positions = createScene();
  const pointCount = positions.length / 2;
  const pointValues = new Float32Array(pointCount).fill(2);
  const oracle = getQuadbinStudyOracle(positions, pointValues);
  const rows = oracle.studyKeys.length;
  expect(rows).toBeLessThanOrEqual(CELL_CAPACITY);
  const polygon = getStudyTiles();
  const [west, , , north] = webMercatorTileBounds(polygon.x0, polygon.y0, RESOLUTION);
  const [, south, east] = webMercatorTileBounds(polygon.x1, polygon.y1, RESOLUTION);
  const permutationParameters = {seed: 7, permutationCount: 199, significanceLevel: 0.05};

  const fixture = new RecipeTestFixture(device, 'hot-spot-recipe-test');
  try {
    const {graph} = fixture;
    const cap = CELL_CAPACITY;
    const slots = options.neighborCapacity ?? cap * 8;
    const outputs = {
      cells: fixture.output('cells', 'uint32x2', cap),
      tableCount: fixture.output('table-count', 'uint32', 1),
      tableOverflow: fixture.output('table-overflow', 'uint32', 1),
      values: fixture.output('values', 'float32', cap),
      offsets: fixture.output('weights-offsets', 'uint32', cap + 1),
      neighbors: fixture.output('weights-neighbors', 'uint32', slots),
      weightValues: fixture.output('weights-values', 'float32', slots),
      weightsOverflow: fixture.output('weights-overflow', 'uint32', 1),
      weightsRequiredCount: fixture.output('weights-required-count', 'uint32', 1),
      zScores: fixture.output('z-scores', 'float32', cap),
      bins: fixture.output('bins', 'sint32', cap),
      neighborCounts: fixture.output('neighbor-counts', 'uint32', cap),
      invalidCount: fixture.output('invalid-count', 'uint32', 1),
      outside: fixture.output('outside', 'uint32', 1),
      exceedances: fixture.output('exceedances', 'uint32', cap),
      pseudoPValues: fixture.output('pseudo-p-values', 'float32', cap),
      significant: fixture.output('significant', 'uint32', cap),
      breaks: fixture.output('breaks', 'float32', 6),
      classCount: fixture.output('class-count', 'uint32', 1),
      colors: fixture.output('colors', 'uint32', cap),
      classIndices: fixture.output('class-indices', 'uint32', cap)
    };
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
        values: fixture.input('point-values', pointValues, 'float32', pointCount),
        family: 'quadbin',
        resolution: RESOLUTION,
        studyArea: {
          kind: 'polygons',
          positions: fixture.input(
            'study-polygon',
            Float32Array.from([west, south, east, south, east, north, west, north]),
            'float32x2',
            4
          ),
          featureOffsets: fixture.input('study-features', Uint32Array.from([0, 1]), 'uint32', 2),
          polygonOffsets: fixture.input('study-polygons', Uint32Array.from([0, 1]), 'uint32', 2),
          ringOffsets: fixture.input('study-rings', Uint32Array.from([0, 4]), 'uint32', 2),
          candidateCapacity: 1024
        },
        cellCapacity: cap,
        ...(options.neighborCapacity === undefined
          ? {}
          : {neighborCapacity: options.neighborCapacity})
      },
      minimumNeighbors: options.minimumNeighbors,
      multipleTesting: options.multipleTesting,
      settings: options.extras ? permutationParameters : {},
      parameters: fixture.parameterBuffer(
        'autocorrelation-parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      scratch: {
        table: {
          cells: outputs.cells.view,
          status: {count: outputs.tableCount.view, overflow: outputs.tableOverflow.view}
        },
        values: outputs.values.view,
        weights: {
          offsets: outputs.offsets.view,
          neighbors: outputs.neighbors.view,
          weights: outputs.weightValues.view
        }
      },
      outputs: {
        zScores: outputs.zScores.view,
        bins: outputs.bins.view,
        neighborCounts: outputs.neighborCounts.view,
        ...(options.extras
          ? {
              permutation: {
                exceedances: outputs.exceedances.view,
                pseudoPValues: outputs.pseudoPValues.view,
                significant: outputs.significant.view
              },
              color: {
                breaks: outputs.breaks.view,
                classCount: outputs.classCount.view,
                colors: outputs.colors.view,
                classIndices: outputs.classIndices.view
              }
            }
          : {}),
        status: {
          weightsOverflow: outputs.weightsOverflow.view,
          weightsRequiredCount: outputs.weightsRequiredCount.view,
          invalidCount: outputs.invalidCount.view,
          pointsOutsideStudyArea: outputs.outside.view
        }
      },
      ...(options.extras
        ? {
            permutation: {
              parameters: fixture.parameterBuffer(
                'permutation-parameters',
                'uint32',
                GPU_PERMUTATION_PARAMETER_LENGTH
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
              methods: ['equal-interval' as const],
              colorScaleParameters: fixture.parameters(
                'color-scale-parameters',
                'float32',
                getGPUColorScaleParameterValues({
                  scale: 'quantile',
                  domainCount: 6,
                  paletteCount: 5
                })
              ),
              palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 5),
              maximumPaletteCount: 5
            }
          }
        : {})
    });
    fixture.run();
    const cells = await fixture.readUint32(outputs.cells, cap * 2);
    const gpuKeys = oracle.studyKeys.map(
      (_, row) => (BigInt(cells[2 * row + 1]) << 32n) | BigInt(cells[2 * row])
    );
    return {
      recipe,
      oracle,
      rows,
      palette,
      permutationParameters,
      gpuKeys,
      tableCount: (await fixture.readUint32(outputs.tableCount, 1))[0],
      tableOverflow: (await fixture.readUint32(outputs.tableOverflow, 1))[0],
      values: await fixture.readFloat32(outputs.values, cap),
      offsets: await fixture.readUint32(outputs.offsets, cap + 1),
      neighbors: await fixture.readUint32(outputs.neighbors, slots),
      weightValues: await fixture.readFloat32(outputs.weightValues, slots),
      weightsOverflow: (await fixture.readUint32(outputs.weightsOverflow, 1))[0],
      weightsRequiredCount: (await fixture.readUint32(outputs.weightsRequiredCount, 1))[0],
      zScores: await fixture.readFloat32(outputs.zScores, cap),
      bins: await fixture.readInt32(outputs.bins, cap),
      neighborCounts: await fixture.readUint32(outputs.neighborCounts, cap),
      invalidCount: (await fixture.readUint32(outputs.invalidCount, 1))[0],
      outside: (await fixture.readUint32(outputs.outside, 1))[0],
      exceedances: await fixture.readUint32(outputs.exceedances, cap),
      significant: await fixture.readUint32(outputs.significant, cap),
      breaks: await fixture.readFloat32(outputs.breaks, 6),
      classCount: (await fixture.readUint32(outputs.classCount, 1))[0],
      colors: await fixture.readUint32(outputs.colors, cap),
      classIndices: await fixture.readUint32(outputs.classIndices, cap)
    };
  } finally {
    fixture.destroy();
  }
}

it('addHotSpotAnalysisRecipe analyses every study-area cell, zeros included, with grid-disk neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = await runQuadbinScene(device, {multipleTesting: 'none'});
  const {oracle, rows} = run;
  // The study table is the deduplicated, sorted cover; the points are joined onto it.
  expect(run.tableCount).toBe(rows);
  expect(run.tableOverflow).toBe(0);
  expect(run.gpuKeys).toEqual(oracle.studyKeys);
  expect(run.weightsOverflow).toBe(0);
  expect(run.weightsRequiredCount).toBe(oracle.neighbors.length);
  // Empty study cells (the hole) are rows with value 0, not masked out.
  const emptyCells = oracle.counts.filter(count => count === 0).length;
  expect(emptyCells).toBeGreaterThanOrEqual(20);
  expect(run.outside).toBe(oracle.outside);
  expect(run.outside).toBeGreaterThanOrEqual(50);

  // Values are sum / area (density); empty cells are analysed as 0.
  for (let row = 0; row < rows; row++) {
    const expected = (2 * oracle.counts[row]) / oracle.areas[row];
    expect(isClose(run.values[row], expected, 1e-6, 2e-4), `value row ${row}`).toBe(true);
    expect(oracle.sums[row]).toBe(2 * oracle.counts[row]);
  }
  expect(run.recipe.rowCount).toBe(CELL_CAPACITY);

  // Neighbor CSR equals the CPU grid-disk oracle: self excluded, binary weights, ascending.
  const total = oracle.neighbors.length;
  for (let row = 0; row <= rows; row++) {
    expect(run.offsets[row], `offset ${row}`).toBe(oracle.offsets[row]);
  }
  for (let slot = 0; slot < total; slot++) {
    expect(run.neighbors[slot], `neighbor slot ${slot}`).toBe(oracle.neighbors[slot]);
    expect(run.weightValues[slot]).toBe(1);
  }
  expect(Math.max(...oracle.offsets.slice(1).map((end, row) => end - oracle.offsets[row]))).toBe(8);

  // Gi* on the density of all study cells (no occupied-only masking) matches the oracle.
  const density = oracle.counts.map((count, row) => (2 * count) / oracle.areas[row]);
  const expected = computeHotSpotOracle({
    weights: oracle.weights,
    // The oracle reads plain numbers; the type only names the GPU column format.
    values: density as unknown as Float32Array
  });
  const neighborTotals = oracle.offsets.slice(1).map((end, row) => end - oracle.offsets[row]);
  let invalid = 0;
  let coldSpots = 0;
  let hotSpots = 0;
  for (let row = 0; row < rows; row++) {
    if (neighborTotals[row] < 8) {
      invalid++;
      expect(run.zScores[row], `guarded z row ${row}`).toBeNaN();
      expect(run.bins[row]).toBe(0);
      continue;
    }
    expect(isClose(run.zScores[row], expected.zScores[row], 2e-3, 1e-3), `z row ${row}`).toBe(true);
    // Neighbors other than the row itself.
    expect(run.neighborCounts[row]).toBe(neighborTotals[row]);
    const critical = [1.6448536, 1.959964, 2.5758293].some(
      value => Math.abs(Math.abs(expected.zScores[row]) - value) < 1e-2
    );
    if (!critical) {
      expect(run.bins[row], `bin row ${row}`).toBe(expected.bins[row]);
    }
    if (oracle.counts[row] === 0 && run.bins[row] < 0) {
      coldSpots++;
    }
    hotSpots += run.bins[row] > 0 ? 1 : 0;
  }
  expect(run.invalidCount).toBe(invalid);
  expect(invalid).toBeGreaterThan(0);
  // A cluster of zeros is a cold spot, and the cluster of points is a hot spot.
  expect(coldSpots).toBeGreaterThan(0);
  expect(hotSpots).toBeGreaterThan(0);
  for (let row = rows; row < CELL_CAPACITY; row++) {
    expect(run.zScores[row], `unused row ${row}`).toBeNaN();
  }
  expect(run.recipe.inference.multipleTesting).toBe('none');
}, 120000);

it('addHotSpotAnalysisRecipe reports a truncated grid-disk CSR in the weights stage', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 100 slots hold the first rows of the CSR only (the full table has several hundred).
  const capacity = 100;
  const run = await runQuadbinScene(device, {multipleTesting: 'none', neighborCapacity: capacity});
  const {oracle, rows} = run;
  const total = oracle.neighbors.length;
  expect(total).toBeGreaterThan(capacity);
  // Rule 9: the overflow flag and the unclamped total for the capacity grow loop.
  expect(run.weightsOverflow).toBe(1);
  expect(run.weightsRequiredCount).toBe(total);
  const weightsStage = run.recipe.status.stages.find(entry => entry.stage === 'weights');
  expect(weightsStage?.status.overflow).toBe(run.recipe.status.weightsOverflow);
  expect(weightsStage?.status.requiredCount).toBe(run.recipe.status.weightsRequiredCount);
  // Offsets are clamped to the capacity; rows that fit keep the exact neighbors.
  for (let row = 0; row <= rows; row++) {
    expect(run.offsets[row], `offset ${row}`).toBe(Math.min(oracle.offsets[row], capacity));
  }
  for (let row = 0; row < rows && oracle.offsets[row + 1] <= capacity; row++) {
    for (let slot = oracle.offsets[row]; slot < oracle.offsets[row + 1]; slot++) {
      expect(run.neighbors[slot], `neighbor slot ${slot}`).toBe(oracle.neighbors[slot]);
    }
  }
}, 120000);

it('addHotSpotAnalysisRecipe defaults to Benjamini-Hochberg and is never looser than uncorrected', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const uncorrected = await runQuadbinScene(device, {multipleTesting: 'none'});
  const corrected = await runQuadbinScene(device, {multipleTesting: 'benjamini-hochberg'});
  expect(corrected.recipe.inference.multipleTesting).toBe('benjamini-hochberg');
  let significant = 0;
  for (let row = 0; row < corrected.rows; row++) {
    expect(Math.abs(corrected.bins[row])).toBeLessThanOrEqual(Math.abs(uncorrected.bins[row]));
    if (corrected.bins[row] !== 0) {
      expect(Math.sign(corrected.bins[row])).toBe(Math.sign(uncorrected.bins[row]));
      significant++;
    }
  }
  expect(significant).toBeGreaterThan(0);
  expect(corrected.invalidCount).toBe(uncorrected.invalidCount);
}, 120000);

it('addHotSpotAnalysisRecipe confirms with permutations and colors the z-scores (no guard)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run: SceneRun = await runQuadbinScene(device, {
    multipleTesting: 'none',
    minimumNeighbors: 0,
    extras: true
  });
  const {oracle, rows} = run;
  expect(run.invalidCount).toBe(0);
  const density = oracle.counts.map((count, row) => (2 * count) / oracle.areas[row]);
  const expected = computeHotSpotOracle({
    weights: oracle.weights,
    // The oracle reads plain numbers; the type only names the GPU column format.
    values: density as unknown as Float32Array
  });
  for (let row = 0; row < rows; row++) {
    expect(isClose(run.zScores[row], expected.zScores[row], 2e-3, 1e-3), `z row ${row}`).toBe(true);
  }

  // Permutation confirmation of the same chain, on exactly the f32 values the GPU analysed.
  const gpuValues = Array.from(run.values.slice(0, rows));
  const permutationOracle = computeLocalPermutationOracle({
    weights: oracle.weights,
    values: Float32Array.from(gpuValues),
    statistic: 'localGStar',
    maximumNeighbors: 32,
    ...run.permutationParameters
  });
  let mismatched = 0;
  for (let row = 0; row < rows; row++) {
    if (run.exceedances[row] !== permutationOracle.exceedances[row]) {
      mismatched++;
      expect(
        Math.abs(run.exceedances[row] - permutationOracle.exceedances[row])
      ).toBeLessThanOrEqual(1);
    }
  }
  expect(mismatched).toBeLessThanOrEqual(Math.max(1, rows * 0.02));
  const hotRow = expected.zScores.indexOf(Math.max(...expected.zScores));
  expect(permutationOracle.significant[hotRow]).toBe(1);
  expect(run.significant[hotRow]).toBe(1);
  expect(run.recipe.permutationInference?.permutationCount).toBe(199);

  // Class breaks of the z-scores and the threshold color scale of the same rows.
  const finite = expected.zScores.filter(Number.isFinite);
  const minimum = Math.min(...finite);
  const maximum = Math.max(...finite);
  expect(run.classCount).toBe(5);
  for (let edge = 0; edge <= 5; edge++) {
    const edgeValue = minimum + ((maximum - minimum) / 5) * edge;
    expect(isClose(run.breaks[edge], edgeValue, 5e-3, 2e-3), `edge ${edge}`).toBe(true);
  }
  for (let row = 0; row < rows; row++) {
    let expectedClass = 0;
    for (let edge = 1; edge < 5; edge++) {
      if (run.breaks[edge] <= run.zScores[row]) {
        expectedClass++;
      }
    }
    expect(run.classIndices[row], `class row ${row}`).toBe(expectedClass);
    expect(run.colors[row]).toBe(run.palette[expectedClass]);
  }
  expect(Math.max(...run.classIndices.slice(0, rows))).toBe(4);
}, 120000);

const H3_RESOLUTION = 4;

function cellAreaKm2(name: string): number {
  return cellArea(name, 'km2');
}

it('addHotSpotAnalysisRecipe on H3 uses gridDisk(2) neighbors and cell-area density', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(21);
  const points: number[] = [];
  for (let index = 0; index < 600; index++) {
    points.push(-102 + random() * 8, 36 + random() * 6);
  }
  const positions = Float32Array.from(points);
  const pointCount = positions.length / 2;
  const cap = 512;
  const fixture = new RecipeTestFixture(device, 'hot-spot-recipe-h3');
  try {
    const cells = fixture.output('cells', 'uint32x2', cap);
    const tableCount = fixture.output('table-count', 'uint32', 1);
    const values = fixture.output('values', 'float32', cap);
    const offsets = fixture.output('offsets', 'uint32', cap + 1);
    const neighbors = fixture.output('neighbors', 'uint32', cap * 18);
    const neighborCounts = fixture.output('neighbor-counts', 'uint32', cap);
    const zScores = fixture.output('z-scores', 'float32', cap);
    const invalidCount = fixture.output('invalid', 'uint32', 1);
    const recipe = addHotSpotAnalysisRecipe(fixture.graph, {
      source: {
        kind: 'points',
        positions: fixture.input('positions', positions, 'float32x2', pointCount),
        family: 'h3',
        resolution: H3_RESOLUTION,
        studyArea: {
          kind: 'polygons',
          positions: fixture.input(
            'study-polygon',
            Float32Array.from([-100, 36, -93, 36, -93, 42, -100, 42]),
            'float32x2',
            4
          ),
          featureOffsets: fixture.input('study-features', Uint32Array.from([0, 1]), 'uint32', 2),
          polygonOffsets: fixture.input('study-polygons', Uint32Array.from([0, 1]), 'uint32', 2),
          ringOffsets: fixture.input('study-rings', Uint32Array.from([0, 4]), 'uint32', 2),
          // The H3 cover lattice spaces longitude by the equatorward latitude (denser than the
          // former poleward spacing), so this polygon needs more than 8192 candidates; a short
          // capacity silently drops the northern rows, which the h3-js check below catches.
          candidateCapacity: 32768
        },
        cellCapacity: cap
      },
      analyze: 'density',
      parameters: fixture.parameterBuffer(
        'autocorrelation-parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      multipleTesting: 'none',
      scratch: {
        table: {cells: cells.view, status: {count: tableCount.view}},
        values: values.view,
        weights: {offsets: offsets.view, neighbors: neighbors.view}
      },
      outputs: {
        zScores: zScores.view,
        neighborCounts: neighborCounts.view,
        status: {invalidCount: invalidCount.view}
      }
    });
    expect(recipe.table).toBeDefined();
    fixture.run();
    const rows = (await fixture.readUint32(tableCount, 1))[0];
    expect(rows).toBeGreaterThan(100);
    const words = await fixture.readUint32(cells, rows * 2);
    const keys = Array.from(
      {length: rows},
      (_, row) => (BigInt(words[2 * row + 1]) << 32n) | BigInt(words[2 * row])
    );
    const sorted = [...keys].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    expect(keys).toEqual(sorted);
    const names = keys.map(bigIntToH3);
    // The cover agrees with h3-js polygonToCells up to f32 centers within 1e-5 degrees of an edge.
    const reference = new Set(
      polygonToCells(
        [
          [36, -100],
          [36, -93],
          [42, -93],
          [42, -100]
        ],
        H3_RESOLUTION
      )
    );
    const shared = names.filter(name => reference.has(name)).length;
    expect(shared).toBeGreaterThanOrEqual(0.98 * Math.max(reference.size, rows));

    // CSR = gridDisk(2) restricted to study cells, self excluded, ascending.
    const gpuOffsets = await fixture.readUint32(offsets, rows + 1);
    const gpuNeighbors = await fixture.readUint32(neighbors, gpuOffsets[rows]);
    const index = new Map(names.map((name, row) => [name, row]));
    for (let row = 0; row < rows; row++) {
      const expectedRow = gridDisk(names[row], 2)
        .map(name => index.get(name))
        .filter((neighbor): neighbor is number => neighbor !== undefined && neighbor !== row)
        .sort((a, b) => a - b);
      const actualRow = Array.from(gpuNeighbors.slice(gpuOffsets[row], gpuOffsets[row + 1]));
      expect(actualRow, `neighbors of ${names[row]}`).toEqual(expectedRow);
    }

    // Density = count / cellAreaKm2 (km^2), counts from the forward index within f32 tolerance.
    const gpuValues = await fixture.readFloat32(values, rows);
    const binned = new Map<string, number>();
    for (let point = 0; point < pointCount; point++) {
      const name = latLngToCell(positions[2 * point + 1], positions[2 * point], H3_RESOLUTION);
      binned.set(name, (binned.get(name) ?? 0) + 1);
    }
    let mismatched = 0;
    for (let row = 0; row < rows; row++) {
      const expectedDensity = (binned.get(names[row]) ?? 0) / cellAreaKm2(names[row]);
      if (!isClose(gpuValues[row], expectedDensity, 1e-6, 2e-3)) {
        mismatched++;
      }
    }
    expect(mismatched).toBeLessThanOrEqual(Math.ceil(0.02 * rows));

    // Interior hexagons (18 neighbors) are tested; study-area edge cells are guarded.
    const z = await fixture.readFloat32(zScores, rows);
    const gpuNeighborCounts = await fixture.readUint32(neighborCounts, rows);
    let guarded = 0;
    for (let row = 0; row < rows; row++) {
      const available = gpuOffsets[row + 1] - gpuOffsets[row];
      if (available < 8) {
        guarded++;
        expect(z[row], `guarded ${row}`).toBeNaN();
      } else {
        expect(Number.isFinite(z[row]), `z ${row}`).toBe(true);
        expect(gpuNeighborCounts[row]).toBe(available);
      }
    }
    expect((await fixture.readUint32(invalidCount, 1))[0]).toBe(guarded);
    expect(guarded).toBeGreaterThan(0);
  } finally {
    fixture.destroy();
  }
}, 120000);

it('addHotSpotAnalysisRecipe analyses a dense lattice with queen weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
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

  // The default guard (8 neighbors) leaves only the interior of a queen lattice; 0 disables it.
  for (const minimumNeighbors of [undefined, 0]) {
    const fixture = new RecipeTestFixture(device, 'hot-spot-recipe-lattice-test');
    try {
      const zScores = fixture.output('z-scores', 'float32', width * height);
      const overflow = fixture.output('weights-overflow', 'uint32', 1);
      const invalid = fixture.output('invalid-count', 'uint32', 1);
      const recipe = addHotSpotAnalysisRecipe(fixture.graph, {
        source: {
          kind: 'lattice',
          values: fixture.input('values', values, 'float32', values.length),
          width,
          height,
          neighborCapacity: width * height * 8
        },
        parameters: fixture.parameterBuffer(
          'autocorrelation-parameters',
          'float32',
          GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
        ),
        minimumNeighbors,
        outputs: {
          zScores: zScores.view,
          status: {weightsOverflow: overflow.view, invalidCount: invalid.view}
        }
      });
      expect(recipe.inference.testedRowsDefinition).toBe(
        minimumNeighbors === 0 ? 'lattice cells' : 'lattice cells with at least 8 neighbors'
      );
      fixture.run();
      expect((await fixture.readUint32(overflow, 1))[0]).toBe(0);
      const z = await fixture.readFloat32(zScores, width * height);
      let guarded = 0;
      for (let row = 0; row < z.length; row++) {
        if (minimumNeighbors !== 0 && offsets[row + 1] - offsets[row] < 8) {
          guarded++;
          expect(z[row], `guarded z row ${row}`).toBeNaN();
        } else {
          expect(isClose(z[row], oracle.zScores[row], 2e-3, 1e-3), `z row ${row}`).toBe(true);
        }
      }
      expect((await fixture.readUint32(invalid, 1))[0]).toBe(guarded);
      expect(guarded > 0).toBe(minimumNeighbors !== 0);
      expect(Math.max(...z.filter(Number.isFinite))).toBeGreaterThan(2);
    } finally {
      fixture.destroy();
    }
  }
}, 120000);
