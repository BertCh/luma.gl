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
  GPUSpatialWeightsTransform,
  GPULocalMoran,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract geometry and poverty percentages; the caller owns the recording graph. */
export type LocalMoranInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the Local Moran result figure. */
export async function runExample(device: Device, input: LocalMoranInput) {
  const {graph, tracts, values, parameters} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = values.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  // 16K slots and 64K candidate pairs comfortably cover this 791-tract topology.
  const weightOutput = {
    offsets: createInteger('weightsRowOffsets', count + 1),
    neighbors: createInteger('weightsNeighbors', 16384),
    weights: createFloat('weightsValues', 16384),
    status: {overflow: createInteger('weightsOverflow', 1)}
  };
  const output = {
    localI: createFloat('localI'),
    zScores: createFloat('zScores'),
    spatialLag: createFloat('spatialLag'),
    quadrants: createInteger('quadrants'),
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
  const rowWeights = new GPUSpatialWeightsTransform({
    weights: queen.outputWeights,
    operation: 'row'
  });
  graph.add(rowWeights);
  const moran = new GPULocalMoran({
    values,
    weights: rowWeights.outputWeights,
    parameters,
    islands: 'exclude',
    minimumNeighbors: 4,
    multipleTesting: 'benjamini-hochberg',
    quadrantGating: 'analytic',
    output
  });
  graph.add(moran);
  const {localI, adjustedPValues} = output;
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    weights: rowWeights.outputWeights,
    views: {
      values,
      ...output,
      localI,
      adjustedPValues,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsValues: weightOutput.weights,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
