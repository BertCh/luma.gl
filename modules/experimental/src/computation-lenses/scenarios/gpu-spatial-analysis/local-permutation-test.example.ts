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
  GPULocalPermutationTest,
  GPUSpatialWeightsTransform,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tract geometry, poverty percentages and packed permutation settings. */
export type LocalPermutationTestInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'uint32'>;
};

/** Runs the exact public-API example shown below the Local Permutation Test result figure. */
export async function runExample(device: Device, input: LocalPermutationTestInput) {
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
    exceedances: createInteger('exceedances'),
    pseudoPValues: createFloat('pValues'),
    observed: createFloat('observed'),
    adjustedPValues: createFloat('adjustedPValues'),
    status: {overflow: createInteger('overflow', 1)}
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
  // 99 permutations give p-values on the grid k/100; two-sided doubles the smaller tail.
  const test = new GPULocalPermutationTest({
    weights: rowWeights.outputWeights,
    values,
    parameters,
    statistic: 'localMoran',
    alternative: 'two-sided',
    islands: 'exclude',
    maximumPermutations: 99,
    multipleTesting: 'benjamini-hochberg',
    output
  });
  graph.add(test);
  const {pseudoPValues, adjustedPValues} = output;
  // #endregion example
  return {
    graph,
    test,
    encodeOptions: {parameters: undefined},
    weights: rowWeights.outputWeights,
    views: {
      values,
      exceedances: output.exceedances,
      pValues: pseudoPValues,
      observed: output.observed,
      adjustedPValues,
      overflow: output.status.overflow,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsValues: weightOutput.weights,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
