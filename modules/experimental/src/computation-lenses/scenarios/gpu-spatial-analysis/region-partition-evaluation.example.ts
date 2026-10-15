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
  GPURegionPartitionEvaluation,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tracts, one poverty column and the region label of each tract. */
export type RegionPartitionEvaluationInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  labels: GraphDataView<'uint32'>;
};

/** Runs the exact public-API example shown below the partition evaluation result figure. */
export async function runExample(device: Device, input: RegionPartitionEvaluationInput) {
  const {graph, tracts, values, labels} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = values.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  // 2K slots cover the queen topology of the 136 study tracts.
  const weightOutput = {
    offsets: createInteger('weightsRowOffsets', count + 1),
    neighbors: createInteger('weightsNeighbors', 2048),
    weights: createFloat('weightsValues', 2048),
    status: {overflow: createInteger('weightsOverflow', 1)}
  };
  const output = {
    summary: createFloat('summary', 9),
    regionSizes: createInteger('regionSizes', 8),
    regionWithinSumOfSquares: createFloat('regionWithinSumOfSquares', 8)
  };
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  // Score any label column; here the eight community areas, with the queen graph for boundaries.
  graph.add(
    new GPURegionPartitionEvaluation({
      values,
      labels,
      labelCapacity: 8,
      weights: queen.outputWeights,
      output
    })
  );
  const {summary, regionSizes} = output;
  // #endregion example
  return {
    graph,
    weights: queen.outputWeights,
    views: {
      values,
      labels,
      ...output,
      summary,
      regionSizes,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
