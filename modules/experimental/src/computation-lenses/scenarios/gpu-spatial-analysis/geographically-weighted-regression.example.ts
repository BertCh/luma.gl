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
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPUGeographicallyWeightedRegression
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Number of golden-section iterations built into the graph. */
const SEARCH_ITERATIONS = 12;

/** Uploaded tract design and ladder parameters; the caller owns the recording graph. */
export type GeographicallyWeightedRegressionInput = {
  graph: GPUCommandGraph;
  /** Tract centroids in local-frame metres. */
  positions: GraphDataView<'float32x2'>;
  /** Row-major shares of poverty, car-free households and uninsured residents, three per tract. */
  predictors: GraphDataView<'float32'>;
  /** Diabetes prevalence (%) per tract. */
  response: GraphDataView<'float32'>;
  /** Bisquare kernel, adaptive bandwidth, default search bounds (`getGPUGeographicallyWeightedRegressionParameterValues`). */
  parameters: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the GWR result figure. */
export async function runExample(device: Device, input: GeographicallyWeightedRegressionInput) {
  const {graph, positions, predictors, response, parameters} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = response.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const output = {
    coefficients: createFloat('coefficients', count * 4),
    standardErrors: createFloat('standardErrors', count * 4),
    tValues: createFloat('tValues', count * 4),
    localR2: createFloat('localR2'),
    selectedBandwidth: createFloat('selectedBandwidth', 2),
    candidateTable: createFloat(
      'candidateTable',
      (SEARCH_ITERATIONS + 1) * GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE
    ),
    summary: createFloat('summary', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
  };
  // #region example
  const gwr = new GPUGeographicallyWeightedRegression({
    positions, // tract centroids in local-frame metres
    predictors,
    predictorCount: 3,
    response,
    parameters, // bisquare kernel, adaptive bandwidth, default search bounds
    maximumBandwidthCount: 2,
    bandwidthSearch: 'golden-section',
    maximumSearchIterations: SEARCH_ITERATIONS,
    output
  });
  graph.add(gwr);
  // per tract: 4 local coefficients (intercept first) and the bandwidth k the search chose
  const {coefficients, selectedBandwidth} = output;
  // #endregion example
  return {graph, output, coefficients, selectedBandwidth};
}
