// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkSnapping, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: roads with node positions and the facilities' longitude/latitude. */
export type NetworkSnappingExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  facilityPositions: GraphDataView<'float32x2'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkSnappingExampleInput) {
  const {graph, roads, facilityPositions} = input;
  const facilityCount = facilityPositions.length;
  // #region example
  const snapping = new GPUNetworkSnapping({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    positions: facilityPositions,
    topology: roads,
    output: {
      snappedEdges: createTransientView(graph, 'snappedEdges', 'uint32', facilityCount),
      snapDistances: createTransientView(graph, 'snapDistances', 'float32', facilityCount),
      snappedPositions: createTransientView(graph, 'snappedPositions', 'float32x2', facilityCount)
    }
  });
  graph.add(snapping);
  const {snappedEdges, snapDistances, snappedPositions} = snapping.props.output;
  // #endregion example
  return {
    snapping,
    snappedEdges,
    snapDistances: snapDistances!,
    snappedPositions: snappedPositions!
  };
}
