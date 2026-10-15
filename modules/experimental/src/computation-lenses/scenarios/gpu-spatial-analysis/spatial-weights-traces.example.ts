// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUSpatialWeightsTraces,
  GPUContiguityWeights,
  GPUSpatialWeightsTransform,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract design and polygons; the caller owns the recording graph. */
export type SpatialWeightsTracesInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
};

/** Runs the exact public-API example shown below the weights traces result figure. */
export async function runExample(device: Device, input: SpatialWeightsTracesInput) {
  const {graph, tracts} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = tracts.featureOffsets.length - 1;
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
  const output = {
    powerTraces: createFloat('powerTraces', 12),
    powerSums: createFloat('powerSums', 12),
    infinityNorm: createFloat('infinityNorm', 1)
  };
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  const row = new GPUSpatialWeightsTransform({weights: queen.outputWeights, operation: 'row'});
  graph.add(row);
  const weights = row.outputWeights;
  const traces = new GPUSpatialWeightsTraces({
    weights,
    maximumPower: 12,
    mode: 'exact', // one unit-vector chain per tract
    output
  });
  graph.add(traces);
  // tr((W/s)^q) and 1'(W/s)^q 1 for q = 1..12, with s = the largest absolute row sum
  const {powerTraces, powerSums} = output;
  // #endregion example
  return {
    graph,
    output,
    powerTraces,
    powerSums,
    weightsOverflow: weightOutput.status.overflow
  };
}
