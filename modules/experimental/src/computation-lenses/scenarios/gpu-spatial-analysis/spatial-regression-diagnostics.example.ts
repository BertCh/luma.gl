// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH,
  GPUContiguityWeights,
  GPUOrdinaryLeastSquares,
  GPUSpatialRegressionDiagnostics,
  GPUSpatialWeightsTransform,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract design and polygons; the caller owns the recording graph. */
export type SpatialRegressionDiagnosticsInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  /** Row-major shares of poverty, car-free households and uninsured residents, three per tract. */
  predictors: GraphDataView<'float32'>;
  /** Diabetes prevalence (%) per tract. */
  response: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the diagnostics result figure. */
export async function runExample(device: Device, input: SpatialRegressionDiagnosticsInput) {
  const {graph, tracts, predictors, response} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = response.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  const weightOutput = {
    offsets: createInteger('weightsRowOffsets', count + 1),
    neighbors: createInteger('weightsNeighbors', 8192),
    weights: createFloat('weightsValues', 8192),
    status: {overflow: createInteger('weightsOverflow', 1)}
  };
  const olsOutput = {
    coefficients: createFloat('olsCoefficients', 4),
    standardErrors: createFloat('olsStandardErrors', 4),
    tStatistics: createFloat('olsTStatistics', 4),
    summary: createFloat('olsSummary', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
    status: {code: createInteger('olsStatus', 1)},
    residuals: createFloat('residuals')
  };
  const output = {
    tests: createFloat('tests', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH),
    summary: createFloat('summary', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH),
    status: {code: createInteger('status', 1)}
  };
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  const row = new GPUSpatialWeightsTransform({weights: queen.outputWeights, operation: 'row'});
  graph.add(row);
  graph.add(
    new GPUOrdinaryLeastSquares({predictors, response, predictorCount: 3, output: olsOutput})
  );
  const diagnostics = new GPUSpatialRegressionDiagnostics({
    weights: row.outputWeights,
    predictors,
    response,
    residuals: olsOutput.residuals, // the OLS fit being diagnosed
    predictorCount: 3,
    output
  });
  graph.add(diagnostics);
  // six tests, three values each: statistic, degrees of freedom, p-value
  const {tests} = output;
  // #endregion example
  return {
    graph,
    output,
    tests,
    residuals: olsOutput.residuals,
    weightsOverflow: weightOutput.status.overflow
  };
}
