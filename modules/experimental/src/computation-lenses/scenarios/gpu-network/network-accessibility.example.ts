// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkAccessibility} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs and the facility-by-node cost matrix, searched over the reversed roads. */
export type NetworkAccessibilityExampleInput = {
  graph: GPUCommandGraph;
  matrixCosts: GraphDataView<'float32'>;
  opportunityWeights: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  nodeCount: number;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkAccessibilityExampleInput) {
  const {graph, matrixCosts, opportunityWeights, parameters, nodeCount} = input;
  // #region example
  const accessibility = new GPUNetworkAccessibility({
    costs: matrixCosts,
    orientation: 'opportunity-rows',
    opportunityWeights,
    parameters,
    gravity: createTransientView(graph, 'gravity', 'float32', nodeCount)
  });
  graph.add(accessibility);
  const {gravity} = accessibility.props;
  // #endregion example
  return {accessibility, gravity: gravity!};
}
