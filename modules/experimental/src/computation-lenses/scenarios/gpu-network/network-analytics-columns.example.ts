// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNetworkAnalyticsColumns,
  type GPUNetworkTopology
} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: the road network. */
export type NetworkAnalyticsColumnsExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkAnalyticsColumnsExampleInput) {
  const {graph, roads} = input;
  const nodeCount = roads.offsets.length - 1;
  // #region example
  const analytics = new GPUNetworkAnalyticsColumns({
    topology: roads,
    degree: {
      output: createTransientView(graph, 'degree', 'uint32', nodeCount),
      normalized: createTransientView(graph, 'normalized', 'float32', nodeCount),
      extent: createTransientView(graph, 'extent', 'uint32', 2)
    }
  });
  graph.add(analytics);
  const {output: degree, normalized, extent} = analytics.props.degree!;
  // #endregion example
  return {analytics, degree, normalized: normalized!, extent: extent!};
}
