// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: one origin zone, destination zone and trip count per record. */
export type FlowAggregationExampleInput = {
  graph: GPUCommandGraph;
  originZoneIds: GraphDataView<'uint32'>;
  destinationZoneIds: GraphDataView<'uint32'>;
  tripCounts: GraphDataView<'float32'>;
  zoneCount: number;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: FlowAggregationExampleInput) {
  const {graph, originZoneIds, destinationZoneIds, tripCounts, zoneCount} = input;
  // #region example
  const flows = new GPUFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds,
    destinationZoneIds,
    weights: tripCounts,
    sumOrder: 'sorted',
    pairCapacity: 512,
    output: {
      ids: createTransientView(graph, 'flowKeys', 'uint32', 32),
      flowOriginZoneIds: createTransientView(graph, 'flowOriginZones', 'uint32', 32),
      flowDestinationZoneIds: createTransientView(graph, 'flowDestinationZones', 'uint32', 32),
      flowCounts: createTransientView(graph, 'flowCounts', 'uint32', 32),
      flowWeights: createTransientView(graph, 'flowWeights', 'float32', 32),
      status: {
        count: createTransientView(graph, 'flowCount', 'uint32', 1),
        overflow: createTransientView(graph, 'flowOverflow', 'uint32', 1),
        requiredCount: createTransientView(graph, 'flowRequiredCount', 'uint32', 1),
        pairOverflow: createTransientView(graph, 'flowPairOverflow', 'uint32', 1)
      }
    }
  });
  graph.add(flows);
  const {flowOriginZoneIds, flowDestinationZoneIds, flowWeights} = flows.props.output;
  // #endregion example
  const {flowCounts, status} = flows.props.output;
  return {
    flows,
    flowOriginZoneIds: flowOriginZoneIds!,
    flowDestinationZoneIds: flowDestinationZoneIds!,
    flowWeights: flowWeights!,
    flowCounts: flowCounts!,
    status
  };
}
