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
  GPUSpatialWeightsSummary,
  GPUSpatialWeightsValidation,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Runs the public-API example used by this weights card. */
export async function runExample(
  device: Device,
  input: {
    graph: GPUCommandGraph;
    tracts: GPUContiguityWeightsProps['polygons'] & {positions: GraphDataView<'float32x2'>};
    values: GraphDataView<'float32'>;
    searchParameters: GraphDataView<'float32'>;
    colorParameters: GraphDataView<'uint32'>;
  }
) {
  const {graph, tracts} = input;
  // Inputs and outputs must belong to this device.
  assert(graph.device === device);
  const count = tracts.featureOffsets.length - 1;
  const capacity = 32768;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  const createWeights = (prefix: string) => ({
    offsets: createInteger(`${prefix}RowOffsets`, count + 1),
    neighbors: createInteger(`${prefix}Neighbors`, capacity),
    weights: createFloat(`${prefix}Values`, capacity),
    status: {overflow: createInteger(`${prefix}Overflow`, 1)}
  });
  const sourceOutput = createWeights('weights');
  const flags = createInteger('flags');
  const invalidCount = createInteger('invalidCount', 1);
  const statistics = createFloat('statistics', 3);
  const counts = createInteger('counts', 6);
  const neighborCounts = createInteger('neighborCounts');
  const summaryOutput = {statistics, counts, cardinality: neighborCounts};
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: sourceOutput
  });
  graph.add(queen);
  const validation = new GPUSpatialWeightsValidation({
    weights: queen.outputWeights,
    output: {flags, status: {invalidCount}}
  });
  graph.add(validation);
  const weights = validation.outputWeights;
  const result = {flags, invalidCount};
  // #endregion example
  graph.add(new GPUSpatialWeightsSummary({weights, output: summaryOutput}));
  return {
    graph,
    weights,
    result,
    views: {
      weightsRowOffsets: weights.offsets,
      weightsNeighbors: weights.neighbors,
      weightsValues: weights.weights,
      weightsOverflow: weights.status!.overflow!,
      statistics: statistics,
      counts: counts,
      neighborCounts: neighborCounts,
      flags: flags,
      invalidCount: invalidCount
    }
  };
}
