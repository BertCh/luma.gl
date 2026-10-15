// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUAdjacencyMatrixOrder} from '@luma.gl/experimental/gpu-network';

/** Uploaded input: one group label per intersection. */
export type AdjacencyMatrixOrderExampleInput = {
  graph: GPUCommandGraph;
  groups: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: AdjacencyMatrixOrderExampleInput) {
  const {graph, groups} = input;
  // #region example
  const ordering = new GPUAdjacencyMatrixOrder({
    groups,
    output: {order: createTransientView(graph, 'order', 'uint32', groups.length)}
  });
  graph.add(ordering);
  const {order} = ordering.props.output;
  // #endregion example
  return {ordering, order};
}
