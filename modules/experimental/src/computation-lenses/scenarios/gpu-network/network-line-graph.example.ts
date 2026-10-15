// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkLineGraph, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: roads with node positions, packed turn costs and the arc capacity bound. */
export type NetworkLineGraphExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  parameters: GraphDataView<'float32'>;
  arcCapacity: number;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkLineGraphExampleInput) {
  const {graph, roads, parameters, arcCapacity} = input;
  const edgeCount = roads.neighbors.length;
  // #region example
  const lineGraph = new GPUNetworkLineGraph({
    topology: roads,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    parameters,
    output: {
      topology: {
        offsets: createTransientView(graph, 'lineOffsets', 'uint32', edgeCount + 1),
        neighbors: createTransientView(graph, 'lineNeighbors', 'uint32', arcCapacity),
        weights: createTransientView(graph, 'lineWeights', 'float32', arcCapacity)
      },
      arcCount: createTransientView(graph, 'arcCount', 'uint32', 1),
      status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
    }
  });
  graph.add(lineGraph);
  const {topology: lineTopology, arcCount} = lineGraph.props.output;
  // #endregion example
  return {
    lineGraph,
    lineOffsets: lineTopology.offsets,
    lineNeighbors: lineTopology.neighbors,
    lineWeights: lineTopology.weights,
    arcCount: arcCount!,
    overflow: lineGraph.props.output.status.overflow
  };
}
