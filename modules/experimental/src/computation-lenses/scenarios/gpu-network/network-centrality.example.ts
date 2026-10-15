// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkCentrality, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: the roads, their exact transpose and the sampled pivot intersections. */
export type NetworkCentralityExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  reverseRoads: GPUNetworkTopology;
  pivotNodes: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkCentralityExampleInput) {
  const {graph, roads, reverseRoads, pivotNodes} = input;
  const pivotCount = pivotNodes.length;
  // #region example
  const centrality = new GPUNetworkCentrality({
    topology: roads,
    reverseTopology: reverseRoads,
    pivotNodes,
    laneCount: pivotCount,
    maximumIterations: 32,
    output: {
      closeness: createTransientView(graph, 'closeness', 'float32', pivotCount),
      reachedCounts: createTransientView(graph, 'reachedCounts', 'uint32', pivotCount),
      status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
    }
  });
  graph.add(centrality);
  const {closeness, reachedCounts} = centrality.props.output;
  // #endregion example
  return {
    centrality,
    closeness: closeness!,
    reachedCounts: reachedCounts!,
    converged: centrality.props.output.status!.converged!
  };
}
