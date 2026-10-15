// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkCostWeights} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs and the facility-by-node cost matrix produced by snapping and a cost matrix. */
export type NetworkCostWeightsExampleInput = {
  graph: GPUCommandGraph;
  matrixCosts: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  facilityCount: number;
  nodeCount: number;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkCostWeightsExampleInput) {
  const {graph, matrixCosts, parameters, facilityCount, nodeCount} = input;
  const slotCount = facilityCount * nodeCount;
  // #region example
  const costWeights = new GPUNetworkCostWeights({
    costs: matrixCosts,
    rowCount: facilityCount,
    orientation: 'rows',
    parameters,
    output: {
      offsets: createTransientView(graph, 'weightOffsets', 'uint32', facilityCount + 1),
      neighbors: createTransientView(graph, 'weightNeighbors', 'uint32', slotCount),
      weights: createTransientView(graph, 'weightValues', 'float32', slotCount),
      distances: createTransientView(graph, 'weightDistances', 'float32', slotCount),
      status: {overflow: createTransientView(graph, 'weightOverflow', 'uint32', 1)}
    }
  });
  graph.add(costWeights);
  const {offsets, neighbors, weights} = costWeights.props.output;
  // #endregion example
  const {distances, status} = costWeights.props.output;
  return {costWeights, offsets, neighbors, weights, distances: distances!, status: status!};
}
