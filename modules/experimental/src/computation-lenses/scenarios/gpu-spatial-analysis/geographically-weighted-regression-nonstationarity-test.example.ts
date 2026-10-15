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
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPU_GWR_NONSTATIONARITY_SUMMARY,
  GPU_GWR_NONSTATIONARITY_TABLE_STRIDE,
  GPUGeographicallyWeightedRegression,
  GPUGeographicallyWeightedRegressionNonstationarityTest
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Golden-section iterations of the regression under test. */
const SEARCH_ITERATIONS = 12;
/** Permutations the test runs; the pseudo p-value grid is k / 100. */
export const NONSTATIONARITY_PERMUTATIONS = 99;

/** Uploaded tract design and parameters; the caller owns the recording graph. */
export type NonstationarityTestInput = {
  graph: GPUCommandGraph;
  /** Tract centroids in local-frame metres. */
  positions: GraphDataView<'float32x2'>;
  /** Row-major shares of poverty, car-free households and uninsured residents, three per tract. */
  predictors: GraphDataView<'float32'>;
  /** Diabetes prevalence (%) per tract. */
  response: GraphDataView<'float32'>;
  /** Bisquare kernel, adaptive bandwidth, default search bounds. */
  parameters: GraphDataView<'float32'>;
  /** Seed and count written with `getGPUPermutationParameterValues`. */
  permutationParameters: GraphDataView<'uint32'>;
};

/** Runs the exact public-API example shown below the non-stationarity result figure. */
export async function runExample(device: Device, input: NonstationarityTestInput) {
  const {graph, positions, predictors, response, parameters, permutationParameters} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = response.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const fit = {
    coefficients: createFloat('coefficients', count * 4),
    selectedBandwidth: createFloat('selectedBandwidth', 2),
    summary: createFloat('fitSummary', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
  };
  const output = {
    table: createFloat('table', 4 * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE),
    summary: createFloat('summary', GPU_GWR_NONSTATIONARITY_SUMMARY.length),
    standardDeviations: createFloat('standardDeviations', (NONSTATIONARITY_PERMUTATIONS + 1) * 4)
  };
  const gwr = new GPUGeographicallyWeightedRegression({
    positions,
    predictors,
    predictorCount: 3,
    response,
    parameters,
    maximumBandwidthCount: 2,
    bandwidthSearch: 'golden-section',
    maximumSearchIterations: SEARCH_ITERATIONS,
    output: fit
  });
  graph.add(gwr);
  // #region example
  // fit: the GWR above, with golden-section bandwidth search, supplies the coefficients
  const test = new GPUGeographicallyWeightedRegressionNonstationarityTest({
    positions,
    predictors,
    predictorCount: 3,
    response,
    bandwidthParameters: parameters,
    selectedBandwidth: fit.selectedBandwidth, // reused by every permuted refit
    coefficients: fit.coefficients,
    parameters: permutationParameters, // seed and permutation count
    maximumPermutations: NONSTATIONARITY_PERMUTATIONS,
    maximumBandwidthCount: 2,
    output
  });
  graph.add(test);
  // one row per coefficient: observed spread, pseudo p-value, permuted mean and z
  const {table} = output;
  // #endregion example
  return {graph, output, table, fit};
}
