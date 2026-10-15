// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkCostTable, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: the road network with travel seconds, and the one-row cost limit. */
export type NetworkCostTableExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  costLimit: GraphDataView<'float32'>;
  tableCapacity: number;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkCostTableExampleInput) {
  const {graph, roads, costLimit, tableCapacity} = input;
  const nodeCount = roads.offsets.length - 1;
  // #region example
  const table = new GPUNetworkCostTable({
    topology: roads,
    costLimit,
    laneCount: 600,
    maximumIterations: 12,
    output: {
      offsets: createTransientView(graph, 'tableOffsets', 'uint32', nodeCount + 1),
      targets: createTransientView(graph, 'tableTargets', 'uint32', tableCapacity),
      costs: createTransientView(graph, 'tableCosts', 'float32', tableCapacity),
      status: {
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        requiredCount: createTransientView(graph, 'requiredCount', 'uint32', 1),
        converged: createTransientView(graph, 'converged', 'uint32', 1)
      }
    }
  });
  graph.add(table);
  const {offsets, targets, costs} = table.props.output;
  // #endregion example
  const {status} = table.props.output;
  return {table, offsets, targets, costs, status: status!};
}
