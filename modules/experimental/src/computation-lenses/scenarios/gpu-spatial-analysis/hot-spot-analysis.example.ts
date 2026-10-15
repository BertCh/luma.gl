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
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract geometry and poverty percentages; the caller owns the recording graph. */
export type HotSpotAnalysisInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the Hot Spot Analysis result figure. */
export async function runExample(device: Device, input: HotSpotAnalysisInput) {
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
  const output = {
    zScores: createFloat('zScores'),
    bins: createTransientView(graph, 'bins', 'sint32', count, usage),
    pValues: createFloat('pValues'),
    adjustedPValues: createFloat('adjustedPValues'),
    neighborCounts: createInteger('neighborCounts'),
    globalStatistics: createFloat('globalStatistics', 5)
  };
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  // Binary weights with the focal tract included (selfWeight 1) are the classic Gi*.
  const hotSpots = new GPUHotSpotAnalysis({
    weights: queen.outputWeights,
    values,
    parameters,
    selfWeight: 1,
    islands: 'exclude',
    minimumNeighbors: 4,
    multipleTesting: 'benjamini-hochberg',
    output
  });
  graph.add(hotSpots);
  const {zScores, bins} = output;
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    weights: queen.outputWeights,
    views: {
      values,
      ...output,
      zScores,
      bins,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsValues: weightOutput.weights,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
