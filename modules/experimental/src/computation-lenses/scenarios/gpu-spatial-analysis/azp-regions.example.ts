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
  GPUAZPRegions,
  GPUContiguityWeights,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tracts, one poverty column, starting labels and the seed; the caller owns the graph. */
export type AZPRegionsInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  initialLabels: GraphDataView<'uint32'>;
  parameters: GraphDataView<'uint32'>;
};

/** Runs the exact public-API example shown below the AZP result figure. */
export async function runExample(device: Device, input: AZPRegionsInput) {
  const {graph, tracts, values, initialLabels, parameters} = input;
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
    labels: createInteger('labels'),
    objective: createFloat('objective', 1),
    status: createInteger('status', 4)
  };
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  // The eight community areas are the contiguous starting regions; AZP moves border tracts.
  graph.add(
    new GPUAZPRegions({
      weights: queen.outputWeights,
      values,
      initialLabels,
      regionCapacity: 8,
      maximumIterations: 64,
      parameters,
      output
    })
  );
  const {labels, objective} = output;
  // #endregion example
  return {
    graph,
    weights: queen.outputWeights,
    views: {
      values,
      initialLabels,
      ...output,
      labels,
      objective,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
