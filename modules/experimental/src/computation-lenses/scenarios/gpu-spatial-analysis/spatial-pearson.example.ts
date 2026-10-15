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
  GPUSpatialPearson,
  GPUSpatialWeightsTransform,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract geometry and two tract columns; the caller owns the recording graph. */
export type SpatialPearsonInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  secondValues: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the Spatial Pearson result figure. */
export async function runExample(device: Device, input: SpatialPearsonInput) {
  const {graph, tracts, values, secondValues} = input;
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
  const statistic = createFloat('statistic', 1);
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
  // x at the tract, y at its neighbours; the result is one correlation in [-1, 1].
  graph.add(
    new GPUSpatialPearson({
      weights: rowWeights.outputWeights,
      values,
      secondValues,
      output: {statistic}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    weights: rowWeights.outputWeights,
    views: {
      values,
      secondValues,
      statistic,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsValues: weightOutput.weights,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
