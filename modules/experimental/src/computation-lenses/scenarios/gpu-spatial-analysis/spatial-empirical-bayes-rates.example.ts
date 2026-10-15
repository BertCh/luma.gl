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
  GPUSpatialEmpiricalBayesRates,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract geometry, event counts and populations; the caller owns the recording graph. */
export type SpatialEmpiricalBayesRatesInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  events: GraphDataView<'float32'>;
  populations: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the spatial empirical-Bayes result figure. */
export async function runExample(device: Device, input: SpatialEmpiricalBayesRatesInput) {
  const {graph, tracts, events, populations} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = events.length;
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
  const spatialRates = createFloat('spatialRates');
  const smoothedRates = createFloat('smoothedRates');
  const priorMeans = createFloat('priorMeans');
  const priorVariances = createFloat('priorVariances');
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  // Only the neighbour structure counts: each tract pools itself and its queen neighbours.
  graph.add(
    new GPUSpatialEmpiricalBayesRates({
      events,
      populations,
      weights: queen.outputWeights,
      output: {spatialRates, smoothedRates, priorMeans, priorVariances}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    weights: queen.outputWeights,
    views: {
      events,
      populations,
      spatialRates,
      smoothedRates,
      priorMeans,
      priorVariances,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsValues: weightOutput.weights,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
