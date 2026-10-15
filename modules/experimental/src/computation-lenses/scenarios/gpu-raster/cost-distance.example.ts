// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPUCostDistance,
  GPU_COST_DISTANCE_PARAMETER_LENGTH,
  makeGPURasterBand,
  getGPUCostDistanceParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for cost-distance. */
export type CostDistanceExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  friction: GraphDataView<'float32'>;
  sourceMask: GraphDataView<'uint32'>;
  costs: GraphDataView<'float32'>;
  settledAt: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  iterationCount: GraphDataView<'uint32'>;
};

/** Adds the recorded cost-distance computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: CostDistanceExampleInput) {
  const {graph, grid, friction, sourceMask, costs, settledAt, converged, iterationCount} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'cost-distance-parameters',
    format: 'float32',
    length: GPU_COST_DISTANCE_PARAMETER_LENGTH,
    values: getGPUCostDistanceParameterValues({}, grid)
  });
  graph.add(
    new GPUCostDistance({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      friction: makeGPURasterBand(friction, grid), // cost per metre; NaN is impassable
      sourceMask, // 1 at the source cell
      cellSizeMode: 'uniform',
      maximumIterations: 128,
      parameters: parameters.importToGraph(graph),
      output: {costs, settledAt, status: {converged, iterationCount}} // costs in friction × m
    })
  );
  // #endregion example
  return {parameters};
}
