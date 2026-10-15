// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNetworkCostMatrix,
  GPUNetworkSnapping,
  type GPUNetworkTopology
} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: roads with node positions and the facilities' longitude/latitude. */
export type NetworkCostMatrixExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  facilityPositions: GraphDataView<'float32x2'>;
};

/** Builds the contributors shown on the card; the lens compiles, executes and reads them back. */
export async function runExample(_device: Device, input: NetworkCostMatrixExampleInput) {
  const {graph, roads, facilityPositions} = input;
  const facilityCount = facilityPositions.length;
  const nodeCount = roads.offsets.length - 1;
  const converged = createTransientView(graph, 'converged', 'uint32', 1);
  const snappedEdges = createTransientView(graph, 'snappedEdges', 'uint32', facilityCount);
  // #region example
  const startNodes = createTransientView(graph, 'startNodes', 'uint32', 2 * facilityCount);
  const startCosts = createTransientView(graph, 'startCosts', 'float32', 2 * facilityCount);
  const snapping = new GPUNetworkSnapping({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    positions: facilityPositions,
    topology: roads,
    startDirection: 'forward',
    output: {snappedEdges, startNodes, startCosts}
  });
  const matrix = new GPUNetworkCostMatrix({
    topology: roads,
    startNodes,
    startCosts,
    startsPerRow: 2,
    maximumIterations: 24,
    output: {
      costs: createTransientView(graph, 'matrixCosts', 'float32', facilityCount * nodeCount),
      status: {converged}
    }
  });
  graph.add([snapping, matrix]);
  const {costs} = matrix.props.output;
  // #endregion example
  return {snapping, matrix, costs, converged, startNodes, startCosts, snappedEdges};
}
