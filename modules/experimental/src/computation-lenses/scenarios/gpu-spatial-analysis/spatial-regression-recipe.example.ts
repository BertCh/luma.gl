// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  addSpatialRegressionRecipe,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH,
  GPUContiguityWeights,
  GPUParameterBuffer,
  GPUSpatialWeightsTransform,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the Chicago North Side tracts. */
export type SpatialRegressionRecipeExampleInput = {
  /** Tract polygons as four-level views. */
  tracts: GPUContiguityWeightsProps['polygons'];
  /** Observations per tract. */
  observations: GraphDataView<'float32'>;
  /** Row-major predictors per tract: income in $10,000 and no-vehicle households in hundreds. */
  predictors: GraphDataView<'float32'>;
};

/** Observations per tract regressed on income and car-free households, with spatial diagnostics. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {tracts, observations, predictors}: SpatialRegressionRecipeExampleInput
) {
  const count = observations.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'autocorrelation-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: {
      offsets: output('weightsRowOffsets', 'uint32', count + 1),
      neighbors: output('weightsNeighbors', 'uint32', 4096),
      weights: output('weightsValues', 'float32', 4096),
      status: {overflow: output('weightsOverflow', 'uint32', 1)}
    }
  });
  graph.add(queen);
  const rowWeights = new GPUSpatialWeightsTransform({
    weights: queen.outputWeights,
    operation: 'row'
  });
  graph.add(rowWeights);
  const coefficients = output('coefficients', 'float32', 3);
  const residuals = output('residuals', 'float32', count);
  const fitted = output('fitted', 'float32', count);
  const tests = output('tests', 'float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH);
  const decision = output('decision', 'uint32', 1);
  const moranZScores = output('moranZScores', 'float32', count);
  // #region example
  const regression = addSpatialRegressionRecipe(graph, {
    response: observations,
    predictors,
    predictorCount: 2,
    weights: rowWeights.outputWeights,
    parameters,
    settings: {significanceLevel: 0.05},
    restandardizeRows: true,
    outputs: {
      ols: {coefficients, residuals, fitted},
      diagnostics: {tests},
      specification: {decision},
      residualMoran: {zScores: moranZScores}
    }
  });
  // residuals map where the fit misses; decision is Anselin's 0 OLS, 1 lag, 2 error, 3 both, 4 undetermined
  // #endregion example
  return {regression, coefficients, residuals, fitted, tests, decision, moranZScores, parameters};
}
