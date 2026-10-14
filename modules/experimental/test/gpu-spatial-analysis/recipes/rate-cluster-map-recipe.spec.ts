// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {packGPUColor} from '../../../src/gpu-dataframe/column-classification/index';
import {getMultipleTestingAdjustedPValues} from '../../../src/gpu-spatial-analysis/multiple-testing/index';
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
/**
 * esda's default. With the two-sided default alternative the smallest pseudo p-value of 199
 * permutations is 0.01, and Benjamini-Hochberg over the 41 tested polygons then leaves no row
 * significant (CPU oracle: min adjusted p 0.082); 999 permutations give min adjusted p 0.041 and
 * two significant polygons, so the gating is still exercised.
 */
const PERMUTATIONS = 999;

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
    const adjustedPValues = fixture.output('adjusted-p-values', 'float32', rowCount);
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
        seed: permutationParameters.seed
      },
      significanceLevel: permutationParameters.significanceLevel,
      palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 5),
      outputs: {
        standardizedRates: standardized.view,
        zScores: zScores.view,
        quadrants: quadrants.view,
        colors: colors.view,
        weightsOverflow: overflow.view,
        permutation: {
          significant: significant.view,
          pseudoPValues: pseudoPValues.view,
          adjustedPValues: adjustedPValues.view
        }
      },
      scratch: {
        spatialLag: lag.view,
        weights: {
          offsets: weightsOffsets.view,
          neighbors: weightsNeighbors.view,
          weights: weightsValues.view
        }
      }
    });
    expect(recipe.contributors.length).toBe(5);
    // Default correction is Benjamini-Hochberg; the metadata states what the legend means.
    expect(recipe.inference).toEqual({
      alternative: 'two-sided',
      permutationCount: PERMUTATIONS,
      seed: permutationParameters.seed,
      multipleTesting: 'benjamini-hochberg',
      significanceLevel: 0.05,
      // The recipe asks the permutation node it built (see rate-cluster-map-recipe.node.spec.ts).
      testedRowsDefinition: expect.stringContaining('positive-weight neighbors')
    });
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
    const gpuAdjusted = await fixture.readFloat32(adjustedPValues, rowCount);
    // The family is the tested rows only: the NaN-rate polygon and islands have NaN p-values and
    // are not counted in m. Benjamini-Hochberg over the oracle pseudo p-values (tested rows only).
    const oracleAdjusted = getMultipleTestingAdjustedPValues(
      permutation.pseudoPValues,
      'benjamini-hochberg'
    );
    expect(gpuAdjusted[COLUMNS * 4 + 2]).toBeNaN();
    let compared = 0;
    let colored = 0;
    for (let row = 0; row < rowCount; row++) {
      expect(gpuColors[row]).toBe(palette[gpuQuadrants[row]]);
      if (gpuQuadrants[row] !== 0) {
        colored++;
      }
      // Significance is the BH-adjusted p-value at the level, over all tested rows.
      const oracleSignificant = oracleAdjusted[row] <= 0.05 ? 1 : 0;
      if (Number.isNaN(oracleAdjusted[row])) {
        expect(gpuAdjusted[row], `untested row ${row}`).toBeNaN();
        expect(gpuQuadrants[row], `untested quadrant ${row}`).toBe(0);
      }
      if (gpuSignificant[row] === oracleSignificant) {
        const expected = oracleSignificant
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

// Oracle: <scratchpad>/fix/work/B/b6_moran_local_rate.py (esda 2.10.0, libpysal 4.15.0):
// `Moran_Local_Rate(e, b, lat2W(4, 5, rook=True) row-standardized)` on this 5 x 4 scene.
const PARITY_COLUMNS = 5;
const PARITY_ROWS = 4;
const PARITY_EVENTS = [4, 9, 12, 3, 14, 6, 11, 20, 15, 8, 3, 7, 16, 18, 5, 2, 5, 9, 10, 4];
const PARITY_POPULATIONS = [
  800, 1200, 900, 1000, 1500, 700, 1100, 1600, 1300, 900, 900, 1000, 1400, 1500, 1100, 1200, 900,
  1000, 1300, 800
];
const ESDA_ASSUNCAO_RATE = [
  -0.848400240573539, -0.21070092747783314, 1.4264569680970565, -1.4926115372650133,
  0.37368325456132756, 0.09655848634443782, 0.5373347301986594, 1.4351091440031807,
  1.048398299202036, 0.1938196662419266, -1.3469769610769853, -0.3422465861383595,
  1.0371259594020437, 1.2452821913217844, -1.0819967521727547, -1.9918721559379855,
  -0.7306583101494206, 0.2329358894249671, -0.15584299042192398, -0.848400240573539
];
const ESDA_LOCAL_I = [
  -0.010614419475975854, -0.05995861505144364, -0.026512565498333345, -1.4072932534086802,
  -0.2495037260127498, -0.07833331150499916, 0.18641055600884177, 1.5830237829723002,
  0.4523526734244279, 0.047422826272235576, 0.8349717051856403, 0.014369110371909698,
  0.7674760657481344, 0.3614609987740427, -0.262826594116496, 1.802842831133667, 0.4025377229665517,
  0.035795890419974656, -0.02309184048571893, 0.41297716649074584
];
const ESDA_QUADRANTS = [2, 2, 4, 2, 4, 4, 1, 1, 1, 1, 3, 3, 1, 1, 2, 3, 3, 1, 2, 3];

it('addRateClusterMapRecipe analytic path matches esda Moran_Local_Rate and corrects the family', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rowCount = PARITY_COLUMNS * PARITY_ROWS;
  const polygons: OraclePolygons = [];
  for (let y = 0; y < PARITY_ROWS; y++) {
    for (let x = 0; x < PARITY_COLUMNS; x++) {
      polygons.push([
        [
          [x, y],
          [x + 1, y],
          [x + 1, y + 1],
          [x, y + 1]
        ]
      ]);
    }
  }
  const layout = flattenPolygons(polygons);
  const palette = [
    packGPUColor(200, 200, 200),
    packGPUColor(215, 25, 28),
    packGPUColor(171, 217, 233),
    packGPUColor(44, 123, 182),
    packGPUColor(253, 174, 97)
  ];
  const fixture = new RecipeTestFixture(device, 'rate-cluster-map-parity-test');
  try {
    const standardized = fixture.output('standardized', 'float32', rowCount);
    const localI = fixture.output('local-i', 'float32', rowCount);
    const quadrants = fixture.output('quadrants', 'uint32', rowCount);
    const pValues = fixture.output('p-values', 'float32', rowCount);
    const adjusted = fixture.output('adjusted', 'float32', rowCount);
    const recipe = addRateClusterMapRecipe(fixture.graph, {
      events: fixture.input('events', Float32Array.from(PARITY_EVENTS), 'float32', rowCount),
      populations: fixture.input(
        'populations',
        Float32Array.from(PARITY_POPULATIONS),
        'float32',
        rowCount
      ),
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
      palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 5),
      outputs: {
        standardizedRates: standardized.view,
        localI: localI.view,
        quadrants: quadrants.view,
        analytic: {pValues: pValues.view, adjustedPValues: adjusted.view}
      }
    });
    // Analytic inference is a preview: no permutations, but the default correction still applies.
    expect(recipe.inference.permutationCount).toBe(0);
    expect(recipe.inference.multipleTesting).toBe('benjamini-hochberg');
    expect(recipe.inference.alternative).toBe('two-sided');
    expect(recipe.inference.seed).toBeUndefined();
    expect(recipe.permutation).toBeUndefined();
    fixture.run();

    const rates = await fixture.readFloat32(standardized, rowCount);
    const gpuLocalI = await fixture.readFloat32(localI, rowCount);
    for (let row = 0; row < rowCount; row++) {
      expect(isClose(rates[row], ESDA_ASSUNCAO_RATE[row], 2e-4, 2e-4), `rate ${row}`).toBe(true);
      expect(isClose(gpuLocalI[row], ESDA_LOCAL_I[row], 2e-3, 2e-3), `local I ${row}`).toBe(true);
    }
    const gpuP = await fixture.readFloat32(pValues, rowCount);
    const gpuAdjusted = await fixture.readFloat32(adjusted, rowCount);
    const expectedAdjusted = getMultipleTestingAdjustedPValues(gpuP, 'benjamini-hochberg');
    const gpuQuadrants = await fixture.readUint32(quadrants, rowCount);
    for (let row = 0; row < rowCount; row++) {
      expect(isClose(gpuAdjusted[row], expectedAdjusted[row], 1e-5, 1e-5), `adjusted ${row}`).toBe(
        true
      );
      // A quadrant is reported exactly where the corrected p-value is significant, and it is the
      // esda quadrant: the palette meaning does not depend on the correction.
      const significant = gpuAdjusted[row] <= 0.05;
      expect(gpuQuadrants[row] !== 0, `gate ${row}`).toBe(significant);
      if (significant) {
        expect(gpuQuadrants[row], `quadrant ${row}`).toBe(ESDA_QUADRANTS[row]);
      }
    }
  } finally {
    fixture.destroy();
  }
}, 120000);

// Oracle: <scratchpad>/fix/work/B/b6_multipletests.py (statsmodels 0.15.0 `multipletests`,
// method 'fdr_bh'), with NaN (untested) rows interleaved: they must not enter the family.
it('Benjamini-Hochberg over the tested family ignores NaN rows (statsmodels fdr_bh)', () => {
  const finite = [
    0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222, 0.251, 0.269, 0.275,
    0.34, 0.341, 0.384, 0.569, 0.594, 0.696, 0.762, 0.94, 0.942, 0.975, 0.986
  ];
  const expected = [
    0.025, 0.1, 0.21, 0.21, 0.21, 0.25, 0.26428571428571423, 0.49107142857142855,
    0.49107142857142855, 0.49107142857142855, 0.49107142857142855, 0.49107142857142855,
    0.49107142857142855, 0.49107142857142855, 0.5328125, 0.5328125, 0.5647058823529412,
    0.781578947368421, 0.781578947368421, 0.8699999999999999, 0.9071428571428571, 0.986, 0.986,
    0.986, 0.986
  ];
  const family: number[] = [];
  for (const [index, value] of finite.entries()) {
    family.push(value);
    if (index % 5 === 2) {
      family.push(NaN);
    }
  }
  const adjusted = getMultipleTestingAdjustedPValues(family, 'benjamini-hochberg');
  let next = 0;
  for (const [index, value] of family.entries()) {
    if (Number.isNaN(value)) {
      expect(adjusted[index]).toBeNaN();
    } else {
      expect(adjusted[index]).toBeCloseTo(expected[next++], 12);
    }
  }
  expect(next).toBe(expected.length);
});
