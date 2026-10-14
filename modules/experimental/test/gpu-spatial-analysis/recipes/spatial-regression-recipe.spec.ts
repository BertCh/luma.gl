// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  addSpatialRegressionRecipe,
  getSpatialRegressionSpecification
} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {
  getGPUGeographicallyWeightedRegressionParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_REFINEMENT_PASS_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_EXCLUDED_ROW_COUNT,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-regression/index';
import {computeGeographicallyWeightedRegressionOnCPU} from '../spatial-regression/geographically-weighted-regression-oracle';
import {fitOrdinaryLeastSquaresOnCPU} from '../spatial-regression/ordinary-least-squares-oracle';
import {
  computeSpatialRegressionDiagnosticsOnCPU,
  createDiagnosticsExclusionMask,
  createDiagnosticsScene
} from '../spatial-regression/spatial-regression-diagnostics-oracle';
import {computeLocalMoranOracle} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';

const SIDE = 8;
const LADDER = [3, 4, 6];
const LADDER_CAPACITY = 4;

it.each([
  false,
  true
])('addSpatialRegressionRecipe chains OLS, diagnostics, residual Moran and local fits (geographic: %s)', async geographic => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createDiagnosticsScene(SIDE, 11, 0.6, true);
  const rowCount = SIDE * SIDE;
  const k = scene.predictorCount;
  const spatialContext = geographic
    ? ({
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'kilometers',
        sphereRadius: 6371
      } as const)
    : undefined;
  const positions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) {
    positions[2 * row] = geographic ? -75 + (row % SIDE) * 0.01 : row % SIDE;
    positions[2 * row + 1] = geographic
      ? 40 + Math.floor(row / SIDE) * 0.01
      : Math.floor(row / SIDE);
  }
  const settings = {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: LADDER} as const;
  const ols = fitOrdinaryLeastSquaresOnCPU({
    predictors: scene.predictors,
    response: scene.response,
    predictorCount: k
  });
  const diagnostics = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    k
  );
  const moran = computeLocalMoranOracle({
    weights: scene.weights,
    values: Float32Array.from(ols.residuals)
  });
  const local = computeGeographicallyWeightedRegressionOnCPU({
    positions,
    sphereRadius: spatialContext?.sphereRadius,
    predictors: scene.predictors,
    predictorCount: k,
    response: scene.response,
    settings,
    // The recipe keeps the contributor default: two refinement passes over a capacity-4 ladder.
    maximumBandwidthCount: LADDER_CAPACITY,
    refinementPassCount: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_REFINEMENT_PASS_COUNT
  });

  const fixture = new RecipeTestFixture(device, 'spatial-regression-recipe-test');
  try {
    const coefficients = fixture.output('coefficients', 'float32', k + 1);
    const olsStatus = fixture.output('ols-status', 'uint32', 1);
    const residuals = fixture.output('residuals', 'float32', rowCount);
    const tests = fixture.output(
      'tests',
      'float32',
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
    );
    const diagnosticsSummary = fixture.output(
      'diagnostics-summary',
      'float32',
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH
    );
    const diagnosticsStatus = fixture.output('diagnostics-status', 'uint32', 1);
    const decision = fixture.output('decision', 'uint32', 1);
    const moranZ = fixture.output('moran-z', 'float32', rowCount);
    const localCoefficients = fixture.output('local-coefficients', 'float32', rowCount * (k + 1));
    const selectedBandwidth = fixture.output('selected-bandwidth', 'float32', 2);
    const localSummary = fixture.output(
      'local-summary',
      'float32',
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
    );
    const recipe = addSpatialRegressionRecipe(fixture.graph, {
      predictors: fixture.input('predictors', scene.predictors, 'float32', rowCount * k),
      response: fixture.input('response', scene.response, 'float32', rowCount),
      predictorCount: k,
      weights: {
        offsets: fixture.input('offsets', scene.weights.offsets, 'uint32', rowCount + 1),
        neighbors: fixture.input(
          'neighbors',
          scene.weights.neighbors,
          'uint32',
          scene.weights.neighbors.length
        ),
        weights: fixture.input(
          'weights',
          scene.weights.weights,
          'float32',
          scene.weights.weights.length
        )
      },
      parameters: fixture.parameterBuffer(
        'autocorrelation-parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      outputs: {
        ols: {
          coefficients: coefficients.view,
          status: {code: olsStatus.view},
          residuals: residuals.view
        },
        diagnostics: {
          tests: tests.view,
          summary: diagnosticsSummary.view,
          status: {code: diagnosticsStatus.view}
        },
        specification: {decision: decision.view},
        residualMoran: {zScores: moranZ.view},
        localFits: {
          coefficients: localCoefficients.view,
          selectedBandwidth: selectedBandwidth.view,
          summary: localSummary.view
        }
      },
      localFits: {
        spatialContext,
        positions: fixture.input('positions', positions, 'float32x2', rowCount),
        parameters: fixture.parameters(
          'gwr-parameters',
          'float32',
          getGPUGeographicallyWeightedRegressionParameterValues(settings, LADDER_CAPACITY)
        ),
        maximumBandwidthCount: LADDER_CAPACITY
      }
    });
    // OLS, diagnostics, the specification decision kernel, residual local Moran, GWR.
    expect(recipe.contributors.length).toBe(5);
    fixture.run();

    expect((await fixture.readUint32(olsStatus, 1))[0]).toBe(0);
    expect((await fixture.readUint32(diagnosticsStatus, 1))[0]).toBe(0);
    const gpuCoefficients = await fixture.readFloat32(coefficients, k + 1);
    for (let index = 0; index <= k; index++) {
      expect(
        isClose(gpuCoefficients[index], ols.coefficients[index], 5e-3, 5e-3),
        `coefficient ${index}`
      ).toBe(true);
    }

    // Diagnostics: six rows of [statistic, df, p] against the dense-algebra oracle.
    const gpuTests = await fixture.readFloat32(
      tests,
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
    );
    const expectedTests = [
      diagnostics.lmLag,
      diagnostics.lmError,
      diagnostics.robustLmLag,
      diagnostics.robustLmError,
      diagnostics.lmSarma
    ];
    for (const [row, expected] of expectedTests.entries()) {
      expect(isClose(gpuTests[3 * row], expected.statistic, 1e-3, 5e-3), `test ${row}`).toBe(true);
      expect(gpuTests[3 * row + 1]).toBe(expected.degreesOfFreedom);
    }
    // Error dependence is strong in this scene, so the LM-error statistic is clearly nonzero.
    expect(gpuTests[3]).toBeGreaterThan(5);
    expect(isClose(gpuTests[15], diagnostics.moranZ, 1e-2, 2e-2)).toBe(true);

    // Specification decision: strong error dependence only (spreg LM-lag p 0.14, LM-error p 0.0015)
    // gives the spatial error model, and the GPU decision equals the CPU rule on the GPU p-values.
    const gpuDecision = (await fixture.readUint32(decision, 1))[0];
    expect(gpuDecision).toBe(2);
    expect(gpuDecision).toBe(getSpatialRegressionSpecification(gpuTests, 0));
    expect(recipe.specification.decision).toBeDefined();

    // Residual local Moran on the GPU residuals.
    const gpuResiduals = await fixture.readFloat32(residuals, rowCount);
    const z = await fixture.readFloat32(moranZ, rowCount);
    let finite = 0;
    for (let row = 0; row < rowCount; row++) {
      expect(Math.abs(gpuResiduals[row] - ols.residuals[row])).toBeLessThan(5e-3);
      if (Number.isFinite(moran.zScores[row])) {
        finite++;
        expect(isClose(z[row], moran.zScores[row], 5e-2, 2e-2), `moran z row ${row}`).toBe(true);
      }
    }
    expect(finite).toBeGreaterThan(rowCount / 2);

    // Local fits at the selected bandwidth.
    const selected = await fixture.readFloat32(selectedBandwidth, 2);
    expect(selected[0]).toBe(local.selectedIndex);
    expect(selected[1]).toBeCloseTo(local.selectedValue, 4);
    const gpuLocal = await fixture.readFloat32(localCoefficients, rowCount * (k + 1));
    let comparedLocal = 0;
    for (let index = 0; index < gpuLocal.length; index++) {
      if (Number.isFinite(local.coefficients[index])) {
        comparedLocal++;
        expect(
          isClose(gpuLocal[index], local.coefficients[index], 2e-2, 1e-2),
          `local coefficient ${index}`
        ).toBe(true);
      }
    }
    expect(comparedLocal).toBeGreaterThan(rowCount);
    expect(Math.max(...gpuLocal.map(Math.abs))).toBeGreaterThan(0.5);
    expect(GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH).toBeGreaterThan(0);
  } finally {
    fixture.destroy();
  }
}, 120000);

it('addSpatialRegressionRecipe runs diagnostics after a masked OLS fit and writes the decision', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createDiagnosticsScene(SIDE, 11, 0.6, true);
  const rowCount = SIDE * SIDE;
  const k = scene.predictorCount;
  const mask = createDiagnosticsExclusionMask(rowCount, 7, 3);
  const expected = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    k,
    {mask, restandardizeRows: true}
  );
  const fixture = new RecipeTestFixture(device, 'spatial-regression-recipe-masked-test');
  try {
    const tests = fixture.output(
      'tests',
      'float32',
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
    );
    const summary = fixture.output(
      'diagnostics-summary',
      'float32',
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH
    );
    const status = fixture.output('diagnostics-status', 'uint32', 1);
    const decision = fixture.output('decision', 'uint32', 1);
    addSpatialRegressionRecipe(fixture.graph, {
      predictors: fixture.input('predictors', scene.predictors, 'float32', rowCount * k),
      response: fixture.input('response', scene.response, 'float32', rowCount),
      mask: fixture.input('mask', mask, 'uint32', rowCount),
      restandardizeRows: true,
      // One packed level drives the specification search and the residual Moran.
      settings: {significanceLevel: 0.01},
      predictorCount: k,
      weights: {
        offsets: fixture.input('offsets', scene.weights.offsets, 'uint32', rowCount + 1),
        neighbors: fixture.input(
          'neighbors',
          scene.weights.neighbors,
          'uint32',
          scene.weights.neighbors.length
        ),
        weights: fixture.input(
          'weights',
          scene.weights.weights,
          'float32',
          scene.weights.weights.length
        )
      },
      parameters: fixture.parameterBuffer(
        'autocorrelation-parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      outputs: {
        diagnostics: {tests: tests.view, summary: summary.view, status: {code: status.view}},
        specification: {decision: decision.view}
      }
    });
    fixture.run();
    expect((await fixture.readUint32(status, 1))[0]).toBe(0);
    const gpuTests = await fixture.readFloat32(
      tests,
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
    );
    const gpuSummary = await fixture.readFloat32(
      summary,
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH
    );
    expect(gpuSummary[0]).toBe(expected.rowCount);
    expect(gpuSummary[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_EXCLUDED_ROW_COUNT]).toBe(
      expected.excludedRowCount
    );
    expect(isClose(gpuTests[3], expected.lmError.statistic, 1e-3, 5e-3)).toBe(true);
    expect(isClose(gpuTests[0], expected.lmLag.statistic, 1e-3, 5e-3)).toBe(true);
    expect((await fixture.readUint32(decision, 1))[0]).toBe(
      getSpatialRegressionSpecification(gpuTests, 0, 0.01)
    );
  } finally {
    fixture.destroy();
  }
}, 120000);
