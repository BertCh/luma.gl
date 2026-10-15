// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNetworkStatistics,
  getGPUNetworkStatisticsLength,
  type GPUNetworkTopology
} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: the road network. */
export type NetworkStatisticsExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkStatisticsExampleInput) {
  const {graph, roads} = input;
  // #region example
  const statistics = new GPUNetworkStatistics({
    topology: roads,
    directed: true,
    degreeBinCount: 8,
    degreeBinning: 'linear',
    maximumComponentIterations: 32,
    output: createTransientView(graph, 'summary', 'uint32', getGPUNetworkStatisticsLength(8))
  });
  graph.add(statistics);
  const {output: summary} = statistics.props;
  // #endregion example
  return {statistics, summary};
}
