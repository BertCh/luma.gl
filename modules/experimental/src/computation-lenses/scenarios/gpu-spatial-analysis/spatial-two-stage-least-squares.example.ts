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
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE,
  GPUSpatialTwoStageLeastSquares,
  GPUContiguityWeights,
  GPUSpatialWeightsTransform,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract design and polygons; the caller owns the recording graph. */
export type SpatialTwoStageLeastSquaresInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  /** Row-major shares of poverty, car-free households and uninsured residents, three per tract. */
  predictors: GraphDataView<'float32'>;
  /** Diabetes prevalence (%) per tract. */
  response: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the two-stage least squares result figure. */
export async function runExample(device: Device, input: SpatialTwoStageLeastSquaresInput) {
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
  const output = {
    table: createFloat('table', 5 * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE),
    summary: createFloat('summary', GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH),
    status: {code: createInteger('status', 1)},
    residuals: createFloat('residuals')
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
  const weights = row.outputWeights;
  const twoStage = new GPUSpatialTwoStageLeastSquares({
    weights,
    predictors,
    response,
    predictorCount: 3,
    instrumentOrder: 1, // instruments: the predictors and their spatial lags
    output
  });
  graph.add(twoStage);
  // intercept, three predictors, then the spatial lag rho: estimate, se, z, p per row
  const {table} = output;
  // #endregion example
  return {
    graph,
    output,
    table,
    weightsOverflow: weightOutput.status.overflow
  };
}
