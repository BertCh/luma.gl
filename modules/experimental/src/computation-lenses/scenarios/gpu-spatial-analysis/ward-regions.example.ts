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
  GPUWardRegions,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tracts, one poverty column and the seed; the caller owns the recording graph. */
export type WardRegionsInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'uint32'>;
};

/** Runs the exact public-API example shown below the Ward result figure. */
export async function runExample(device: Device, input: WardRegionsInput) {
  const {graph, tracts, values, parameters} = input;
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
  // The fifth status word receives the number of merge rounds.
  const output = {
    labels: createInteger('labels'),
    objective: createFloat('objective', 1),
    status: createInteger('status', 5)
  };
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  // Merge the cheapest mutually nearest adjacent regions each round until five remain.
  graph.add(
    new GPUWardRegions({
      weights: queen.outputWeights,
      values,
      targetRegionCount: 5,
      maximumIterations: 32,
      mergeSchedule: 'mutual-nearest',
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
      ...output,
      labels,
      objective,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
