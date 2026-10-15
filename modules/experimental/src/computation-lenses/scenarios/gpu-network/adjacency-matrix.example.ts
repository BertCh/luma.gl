// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUAdjacencyMatrix, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: the road network and the matrix position of every intersection. */
export type AdjacencyMatrixExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  order: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: AdjacencyMatrixExampleInput) {
  const {graph, roads, order} = input;
  // #region example
  const matrix = new GPUAdjacencyMatrix({
    topology: roads,
    directed: true,
    order,
    resolution: 8,
    output: {
      counts: createTransientView(graph, 'matrixCounts', 'uint32', 64),
      maximumCount: createTransientView(graph, 'maximumCount', 'uint32', 1)
    }
  });
  graph.add(matrix);
  const {counts, maximumCount} = matrix.props.output;
  // #endregion example
  return {matrix, counts, maximumCount: maximumCount!};
}
