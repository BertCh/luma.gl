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
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree,
  type GPUContiguityWeightsProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded tracts, one poverty column and the SKATER target; the caller owns the graph. */
export type SkaterRegionsInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'uint32'>;
};

/** Runs the exact public-API example shown below the SKATER result figure. */
export async function runExample(device: Device, input: SkaterRegionsInput) {
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
  const treeOutput = {
    treeEdgeFlags: createInteger('treeEdgeFlags', 2048),
    componentLabels: createInteger('componentLabels'),
    standardizedValues: createFloat('standardizedValues')
  };
  // Seven greedy steps are compiled: maximumRegionCount - 1.
  const output = {
    labels: createInteger('labels'),
    regionCount: createInteger('regionCount', 1),
    cutEdges: createInteger('cutEdges', 7),
    cutGains: createFloat('cutGains', 7)
  };
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  const tree = new GPUSpatialWeightsMinimumSpanningTree({
    weights: queen.outputWeights,
    values,
    output: treeOutput
  });
  graph.add(tree);
  // Parameters request five regions, with at least five tracts on each side of every cut.
  graph.add(
    new GPUSkaterRegions({
      weights: queen.outputWeights,
      treeEdgeFlags: treeOutput.treeEdgeFlags,
      componentLabels: treeOutput.componentLabels,
      values: treeOutput.standardizedValues,
      maximumRegionCount: 8,
      parameters,
      output
    })
  );
  const {labels, cutGains} = output;
  // #endregion example
  return {
    graph,
    weights: queen.outputWeights,
    views: {
      values,
      ...treeOutput,
      ...output,
      labels,
      cutGains,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
