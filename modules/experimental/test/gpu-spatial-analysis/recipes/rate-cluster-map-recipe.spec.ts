// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {packGPUColor} from '../../../src/gpu-dataframe/column-classification/index';
import {getMultipleTestingAdjustedPValues} from '../../../src/gpu-spatial-analysis/multiple-testing/index';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
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
import {requireTestDevice} from '../../utils/require-test-device';
import {RATE_CLUSTER_REFERENCE} from './rate-cluster-esda-reference';

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
  requireTestDevice(device);
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
  const permutationParameters = {seed: 11, permutationCount: PERMUTATIONS, significanceLevel: 0.05};

  const fixture = new RecipeTestFixture(device, 'rate-cluster-map-recipe-test');
  const parameters = new GPUParameterBuffer(device, {
    id: 'rate-cluster-map-autocorrelation-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  const permutationParameterBuffer = new GPUParameterBuffer(device, {
    id: 'rate-cluster-map-permutation-parameters',
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH
  });
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
      polygons: {
        kind: 'polygons',
        // One polygon per feature: the identity feature offsets.
        featureOffsets: fixture.input(
          'feature-offsets',
          Uint32Array.from({length: layout.polygonOffsets.length}, (_, index) => index),
          'uint32',
          layout.polygonOffsets.length
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
        )
      },
      criterion: 'rook',
      neighborCapacity: rowCount * 4,
      parameters,
      settings: permutationParameters,
      permutation: {
        parameters: permutationParameterBuffer,
        maximumPermutations: PERMUTATIONS,
        alternative: 'two-sided'
      },
      palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 5),
      outputs: {
        standardizedRates: standardized.view,
        zScores: zScores.view,
        quadrants: quadrants.view,
        colors: colors.view,
        permutation: {
          significant: significant.view,
          pseudoPValues: pseudoPValues.view,
          adjustedPValues: adjustedPValues.view
        },
        status: {weightsOverflow: overflow.view}
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
    parameters.destroy();
    permutationParameterBuffer.destroy();
    fixture.destroy();
  }
}, 120000);

// Oracle: test/oracles/statistics-rate-cluster-recipe.py (esda, libpysal):
// `Moran_Local_Rate(e, b, lat2W(4, 5, rook=True) row-standardized)` on this 5 x 4 scene.
const {parity} = RATE_CLUSTER_REFERENCE;
const PARITY_COLUMNS = parity.columns;
const PARITY_ROWS = parity.rows;
const PARITY_EVENTS = parity.events;
const PARITY_POPULATIONS = parity.populations;
const ESDA_ASSUNCAO_RATE = parity.assuncaoRate;
const ESDA_LOCAL_I = parity.localI;
const ESDA_QUADRANTS = parity.quadrants;

it('addRateClusterMapRecipe analytic path matches esda Moran_Local_Rate and corrects the family', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
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
  const parameters = new GPUParameterBuffer(device, {
    id: 'rate-cluster-map-parity-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
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
      polygons: {
        kind: 'polygons',
        // One polygon per feature: the identity feature offsets.
        featureOffsets: fixture.input(
          'feature-offsets',
          Uint32Array.from({length: layout.polygonOffsets.length}, (_, index) => index),
          'uint32',
          layout.polygonOffsets.length
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
        )
      },
      criterion: 'rook',
      neighborCapacity: rowCount * 4,
      parameters,
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
    parameters.destroy();
    fixture.destroy();
  }
}, 120000);

// Oracle: test/oracles/statistics-rate-cluster-recipe.py (statsmodels `multipletests`,
// method 'fdr_bh'), with NaN (untested) rows interleaved: they must not enter the family.
it('Benjamini-Hochberg over the tested family ignores NaN rows (statsmodels fdr_bh)', () => {
  const finite = RATE_CLUSTER_REFERENCE.benjaminiHochberg.pValues;
  const expected = RATE_CLUSTER_REFERENCE.benjaminiHochberg.adjusted;
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
