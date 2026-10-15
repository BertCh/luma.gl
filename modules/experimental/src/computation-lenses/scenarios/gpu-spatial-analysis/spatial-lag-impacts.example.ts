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
  GPU_SPATIAL_LAG_IMPACTS_STRIDE,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE,
  GPUContiguityWeights,
  GPUSpatialLagImpacts,
  GPUSpatialLagMaximumLikelihood,
  GPUSpatialWeightsTransform,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Series length of the log-determinant and impact series. */
const MAXIMUM_POWER = 128;

/** Uploaded tract design and polygons; the caller owns the recording graph. */
export type SpatialLagImpactsInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  /** Row-major shares of poverty, car-free households and uninsured residents, three per tract. */
  predictors: GraphDataView<'float32'>;
  /** Diabetes prevalence (%) per tract. */
  response: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the lag impacts result figure. */
export async function runExample(device: Device, input: SpatialLagImpactsInput) {
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
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  const row = new GPUSpatialWeightsTransform({weights: queen.outputWeights, operation: 'row'});
  graph.add(row);
  const weights = row.outputWeights;
  const fitOutput = {
    table: createFloat('table', 5 * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE),
    summary: createFloat('summary', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH),
    status: {code: createInteger('status', 1)},
    powerTraces: createFloat('powerTraces', MAXIMUM_POWER),
    powerSums: createFloat('powerSums', MAXIMUM_POWER),
    infinityNorm: createFloat('infinityNorm', 1)
  };
  const output = {impacts: createFloat('impacts', 3 * GPU_SPATIAL_LAG_IMPACTS_STRIDE)};
  // #region example
  // weights: row-standardised queen contiguity of the tracts, built above
  const fit = new GPUSpatialLagMaximumLikelihood({
    weights,
    predictors,
    response,
    predictorCount: 3,
    maximumPower: MAXIMUM_POWER,
    output: fitOutput // table, status and the power traces the impacts reuse
  });
  graph.add(fit);
  const impacts = new GPUSpatialLagImpacts({
    coefficients: fitOutput.table,
    coefficientStride: 4,
    rhoIndex: 4, // intercept, three predictors, then rho
    predictorCount: 3,
    rowCount: count,
    maximumPower: MAXIMUM_POWER,
    traces: fitOutput,
    status: fitOutput.status.code,
    output
  });
  graph.add(impacts);
  // per predictor: direct, indirect and total effect on diabetes prevalence
  const {impacts: effects} = output;
  // #endregion example
  return {
    graph,
    output,
    fitOutput,
    effects,
    weightsOverflow: weightOutput.status.overflow
  };
}
