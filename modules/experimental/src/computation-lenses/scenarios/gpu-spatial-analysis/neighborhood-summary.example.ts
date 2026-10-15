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
  GPUNeighborhoodSummary,
  GPUSpatialWeightsTransform,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract geometry, poverty percentages and community-area labels; the caller owns the graph. */
export type NeighborhoodSummaryInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  categories: GraphDataView<'uint32'>;
};

/** Runs the exact public-API example shown below the neighbourhood summary result figure. */
export async function runExample(device: Device, input: NeighborhoodSummaryInput) {
  const {graph, tracts, values, categories} = input;
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
  // One statistic, so the table has one column per tract.
  const table = createFloat('table');
  const entropy = createFloat('entropy');
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  const rowWeights = new GPUSpatialWeightsTransform({
    weights: queen.outputWeights,
    operation: 'row'
  });
  graph.add(rowWeights);
  // The row-standardised mean of the neighbours (the spatial lag), and how mixed their labels are.
  graph.add(
    new GPUNeighborhoodSummary({
      weights: rowWeights.outputWeights,
      values,
      categories,
      statistics: ['mean'],
      output: {table, entropy}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    weights: rowWeights.outputWeights,
    views: {
      values,
      categories,
      table,
      entropy,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsValues: weightOutput.weights,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
