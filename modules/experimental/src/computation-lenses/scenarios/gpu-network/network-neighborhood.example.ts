// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkNeighborhood, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: the road network, one start intersection and the hop radius. */
export type NetworkNeighborhoodExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  startNodes: GraphDataView<'uint32'>;
  hops: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkNeighborhoodExampleInput) {
  const {graph, roads, startNodes, hops} = input;
  const nodeCount = roads.offsets.length - 1;
  // #region example
  const neighborhood = new GPUNetworkNeighborhood({
    topology: {offsets: roads.offsets, neighbors: roads.neighbors},
    startNodes,
    hops,
    maximumHops: 8,
    output: {hopDistances: createTransientView(graph, 'hopDistances', 'uint32', nodeCount)}
  });
  graph.add(neighborhood);
  const {hopDistances} = neighborhood.props.output;
  // #endregion example
  return {neighborhood, hopDistances};
}
