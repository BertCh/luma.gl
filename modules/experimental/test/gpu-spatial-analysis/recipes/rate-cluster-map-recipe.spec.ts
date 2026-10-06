// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {packGPUColor} from '../../../src/gpu-dataframe/column-classification/index';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {addRateClusterMapRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {computeLocalPermutationOracle} from '../permutation-inference/permutation-oracle';
import {computeEmpiricalBayesOracle} from '../rate-smoothing/empirical-bayes-oracle';
import {
  computeLocalMoranOracle,
  createSeededRandom,
  getQuadrant
} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {
  computeContiguityOracle,
  computeTransformOracle,
  flattenPolygons,
  type OraclePolygons
} from '../spatial-weights/spatial-weights-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';

const COLUMNS = 7;
const ROWS = 6;
const PERMUTATIONS = 199;

/** Unit squares with a high-rate corner, a low-rate corner and one empty-population polygon. */
function createScene() {
  const random = createSeededRandom(21);
  const polygons: OraclePolygons = [];
  const events: number[] = [];
  const populations: number[] = [];
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLUMNS; x++) {
      polygons.push([
        [
          [x, y],
          [x + 1, y],
          [x + 1, y + 1],
          [x, y + 1]
        ]
      ]);
      const population = 2000 + Math.floor(random() * 8000);
      const rate = x < 3 && y < 3 ? 0.08 : x >= 5 && y >= 4 ? 0.005 : 0.03;
      populations.push(population);
      events.push(Math.round(population * rate * (0.9 + 0.2 * random())));
    }
  }
  populations[COLUMNS * 4 + 2] = 0;
  return {polygons, events: Float32Array.from(events), populations: Float32Array.from(populations)};
}

it('addRateClusterMapRecipe smooths rates, builds contiguity and colors significant LISA quadrants', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {polygons, events, populations} = createScene();
  const rowCount = polygons.length;
  const layout = flattenPolygons(polygons);
  const csr = computeContiguityOracle(polygons, 'rook');
  const weights = {
    offsets: Uint32Array.from(csr.offsets),
    neighbors: Uint32Array.from(csr.neighbors),
    weights: Float32Array.from(computeTransformOracle(csr, 'row'))
  };
  const empiricalBayes = computeEmpiricalBayesOracle({events, populations});
  const palette = [
    packGPUColor(200, 200, 200),
    packGPUColor(215, 25, 28),
    packGPUColor(171, 217, 233),
    packGPUColor(44, 123, 182),
    packGPUColor(253, 174, 97)
  ];
  const permutationParameters = {seed: 11, permutations: PERMUTATIONS, significanceLevel: 0.05};

  const fixture = new RecipeTestFixture(device, 'rate-cluster-map-recipe-test');
  try {
    const standardized = fixture.output('standardized', 'float32', rowCount);
    const zScores = fixture.output('z-scores', 'float32', rowCount);
    const lag = fixture.output('lag', 'float32', rowCount);
    const quadrants = fixture.output('quadrants', 'uint32', rowCount);
    const colors = fixture.output('colors', 'uint32', rowCount);
    const significant = fixture.output('significant', 'uint32', rowCount);
    const pseudoPValues = fixture.output('pseudo-p-values', 'float32', rowCount);
    const weightsOffsets = fixture.output('weights-offsets', 'uint32', rowCount + 1);
    const weightsNeighbors = fixture.output('weights-neighbors', 'uint32', rowCount * 4);
    const weightsValues = fixture.output('weights-values', 'float32', rowCount * 4);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const recipe = addRateClusterMapRecipe(fixture.graph, {
      events: fixture.input('events', events, 'float32', rowCount),
      populations: fixture.input('populations', populations, 'float32', rowCount),
      positions: fixture.input(
        'positions',
        layout.positions,
        'float32x2',
        layout.positions.length / 2
      ),
      ringOffsets: fixture.input(
        'ring-offsets',
        layout.ringOffsets,
        'uint32',
        layout.ringOffsets.length
      ),
      polygonOffsets: fixture.input(
        'polygon-offsets',
        layout.polygonOffsets,
        'uint32',
        layout.polygonOffsets.length
      ),
      criterion: 'rook',
      neighborCapacity: rowCount * 4,
      parameters: fixture.parameters(
        'autocorrelation-parameters',
        'float32',
        getGPUSpatialAutocorrelationParameterValues({})
      ),
      permutation: {
        parameters: fixture.parameters(
          'permutation-parameters',
          'uint32',
          getGPUPermutationParameterValues(permutationParameters)
        ),
        maximumPermutations: PERMUTATIONS,
        alternative: 'two-sided',
        significant: significant.view,
        pseudoPValues: pseudoPValues.view
      },
      palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 5),
      standardizedRates: standardized.view,
      zScores: zScores.view,
      spatialLag: lag.view,
      quadrants: quadrants.view,
      colors: colors.view,
      weights: {
        offsets: weightsOffsets.view,
        neighbors: weightsNeighbors.view,
        weights: weightsValues.view
      },
      weightsOverflow: overflow.view
    });
    expect(recipe.contributors.length).toBe(5);
    fixture.run();

    expect((await fixture.readUint32(overflow, 1))[0]).toBe(0);
    expect(await fixture.readUint32(weightsOffsets, rowCount + 1)).toEqual([...weights.offsets]);
    const used = weights.offsets[rowCount];
    expect(await fixture.readUint32(weightsNeighbors, used)).toEqual([...weights.neighbors]);
    const gpuWeights = await fixture.readFloat32(weightsValues, used);
    for (let slot = 0; slot < used; slot++) {
      expect(Math.abs(gpuWeights[slot] - weights.weights[slot])).toBeLessThan(1e-6);
    }

    // Empirical Bayes standardization (the excluded polygon is NaN).
    const rates = await fixture.readFloat32(standardized, rowCount);
    for (let row = 0; row < rowCount; row++) {
      expect(
        isClose(rates[row], empiricalBayes.standardized[row], 5e-3, 5e-3),
        `rate row ${row}`
      ).toBe(true);
    }
    expect(rates[COLUMNS * 4 + 2]).toBeNaN();

    // Local Moran on the GPU rates with the oracle weights.
    const values = Float32Array.from(rates);
    const moran = computeLocalMoranOracle({weights, values});
    const z = await fixture.readFloat32(zScores, rowCount);
    const gpuLag = await fixture.readFloat32(lag, rowCount);
    for (let row = 0; row < rowCount; row++) {
      expect(isClose(z[row], moran.zScores[row], 2e-3, 1e-3), `z row ${row}`).toBe(true);
      expect(isClose(gpuLag[row], moran.spatialLag[row], 1e-4, 1e-3), `lag row ${row}`).toBe(true);
    }

    // Permutation-gated quadrants.
    const finiteMask = Uint32Array.from(values, value => (Number.isFinite(value) ? 1 : 0));
    const permutation = computeLocalPermutationOracle({
      weights,
      values,
      mask: finiteMask,
      statistic: 'localMoran',
      alternative: 'two-sided',
      maximumNeighbors: 32,
      ...permutationParameters
    });
    const gpuQuadrants = await fixture.readUint32(quadrants, rowCount);
    const gpuColors = await fixture.readUint32(colors, rowCount);
    const gpuSignificant = await fixture.readUint32(significant, rowCount);
    let compared = 0;
    let colored = 0;
    for (let row = 0; row < rowCount; row++) {
      expect(gpuColors[row]).toBe(palette[gpuQuadrants[row]]);
      if (gpuQuadrants[row] !== 0) {
        colored++;
      }
      if (gpuSignificant[row] === permutation.significant[row]) {
        const expected = permutation.significant[row]
          ? getQuadrant(values[row] - moran.moments.mean, moran.spatialLag[row])
          : 0;
        expect(gpuQuadrants[row], `quadrant row ${row}`).toBe(expected);
        compared++;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(rowCount - 2);
    expect(colored).toBeGreaterThan(0);
    expect(gpuQuadrants).toContain(1);
    expect(gpuQuadrants[COLUMNS * 4 + 2]).toBe(0);
  } finally {
    fixture.destroy();
  }
}, 120000);
