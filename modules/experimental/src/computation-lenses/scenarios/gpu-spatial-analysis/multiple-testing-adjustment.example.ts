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
  GPUContiguityWeights,
  GPUHotSpotAnalysis,
  GPUMultipleTestingAdjustment,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract geometry and poverty percentages; the caller owns the recording graph. */
export type MultipleTestingAdjustmentInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the Multiple Testing Adjustment result figure. */
export async function runExample(device: Device, input: MultipleTestingAdjustmentInput) {
  const {graph, tracts, values, parameters} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = values.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  // 16K slots comfortably cover the 791-tract queen topology.
  const weightOutput = {
    offsets: createInteger('weightsRowOffsets', count + 1),
    neighbors: createInteger('weightsNeighbors', 16384),
    weights: createFloat('weightsValues', 16384),
    status: {overflow: createInteger('weightsOverflow', 1)}
  };
  const zScores = createFloat('zScores');
  const pValues = createFloat('pValues');
  const adjustedPValues = createFloat('adjustedPValues');
  const testedCount = createInteger('testedCount', 1);
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  // Hot spot z-scores supply one unadjusted p-value per tract; the family is every tested tract.
  graph.add(
    new GPUHotSpotAnalysis({
      weights: queen.outputWeights,
      values,
      parameters,
      minimumNeighbors: 4,
      output: {zScores, pValues}
    })
  );
  graph.add(
    new GPUMultipleTestingAdjustment({
      pValues,
      method: 'benjamini-hochberg',
      output: {adjustedPValues, testedCount}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    weights: queen.outputWeights,
    views: {
      values,
      zScores,
      pValues,
      adjustedPValues,
      testedCount,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsValues: weightOutput.weights,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
