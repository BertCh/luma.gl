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
  GPUOrdinaryLeastSquares
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract design; the caller owns the recording graph. */
export type OrdinaryLeastSquaresInput = {
  graph: GPUCommandGraph;
  /** Row-major shares of poverty, car-free households and uninsured residents, three per tract. */
  predictors: GraphDataView<'float32'>;
  /** Diabetes prevalence (%) per tract. */
  response: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the ordinary least squares result figure. */
export async function runExample(device: Device, input: OrdinaryLeastSquaresInput) {
  const {graph, predictors, response} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = response.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const output = {
    coefficients: createFloat('coefficients', 4),
    standardErrors: createFloat('standardErrors', 4),
    tStatistics: createFloat('tStatistics', 4),
    summary: createFloat('summary', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
    status: {code: createTransientView(graph, 'status', 'uint32', 1, usage)},
    residuals: createFloat('residuals'),
    fitted: createFloat('fitted')
  };
  // #region example
  const ols = new GPUOrdinaryLeastSquares({
    predictors, // poverty, car-free and uninsured shares per tract
    response, // diabetes prevalence (%)
    predictorCount: 3,
    output
  });
  graph.add(ols);
  // intercept first, then one row per predictor
  const {coefficients, standardErrors, tStatistics} = output;
  // #endregion example
  return {
    graph,
    views: {
      coefficients,
      standardErrors,
      tStatistics,
      summary: output.summary,
      status: output.status.code,
      residuals: output.residuals,
      fitted: output.fitted
    }
  };
}
