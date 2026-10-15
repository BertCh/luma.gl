// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkSubgraphFilter, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: the road network and the packed predicate ranges (one edge column). */
export type NetworkSubgraphFilterExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  parameters: GraphDataView<'float32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkSubgraphFilterExampleInput) {
  const {graph, roads, parameters} = input;
  const nodeCount = roads.offsets.length - 1;
  const edgeCount = roads.neighbors.length;
  // #region example
  const filter = new GPUNetworkSubgraphFilter({
    topology: roads,
    directed: true,
    edgeColumns: [roads.weights],
    parameters,
    dropIsolated: true,
    output: {
      nodeMask: createTransientView(graph, 'nodeMask', 'uint32', nodeCount),
      edgeMask: createTransientView(graph, 'edgeMask', 'uint32', edgeCount),
      counts: createTransientView(graph, 'liveCounts', 'uint32', 4)
    }
  });
  graph.add(filter);
  const {nodeMask, edgeMask, counts} = filter.props.output;
  // #endregion example
  return {filter, nodeMask, edgeMask, counts: counts!};
}
