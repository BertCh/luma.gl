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
  GPUMapColoring,
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
  const {graph, tracts, colorParameters} = input;
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
  const colors = createInteger('colors');
  const colorCount = createInteger('colorCount', 1);
  const conflictCount = createInteger('conflictCount', 1);
  const converged = createInteger('converged', 1);
  const iterationCount = createInteger('iterationCount', 1);
  const output = {colors, colorCount, conflictCount, status: {converged, iterationCount}};
  const statistics = createFloat('statistics', 3);
  const counts = createInteger('counts', 6);
  const neighborCounts = createInteger('neighborCounts');
  const summaryOutput = {statistics, counts, cardinality: neighborCounts};
  // #region example
  const rook = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'rook',
    output: sourceOutput
  });
  graph.add(rook);
  const weights = rook.outputWeights;
  graph.add(
    new GPUMapColoring({
      weights,
      parameters: colorParameters,
      priority: 'largest-degree-first',
      maximumIterations: 64,
      output
    })
  );
  const result = {colors, colorCount, conflictCount};
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
      colors: colors,
      colorCount: colorCount,
      conflictCount: conflictCount,
      converged: converged,
      iterationCount: iterationCount
    }
  };
}
