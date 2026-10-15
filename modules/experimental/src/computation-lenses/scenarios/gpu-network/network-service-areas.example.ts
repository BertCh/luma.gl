// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkServiceAreas, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: the road network and the facilities' prepared nearest intersections. */
export type NetworkServiceAreasExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  facilityNodes: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkServiceAreasExampleInput) {
  const {graph, roads, facilityNodes} = input;
  const nodeCount = roads.offsets.length - 1;
  // #region example
  const serviceAreas = new GPUNetworkServiceAreas({
    topology: roads,
    facilities: facilityNodes,
    maximumIterations: 24,
    output: {
      assignments: createTransientView(graph, 'assignments', 'uint32', nodeCount),
      costs: createTransientView(graph, 'costs', 'float32', nodeCount),
      status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
    }
  });
  graph.add(serviceAreas);
  const {assignments, costs} = serviceAreas.props.output;
  // #endregion example
  return {
    serviceAreas,
    assignments,
    costs: costs!,
    converged: serviceAreas.props.output.status!.converged!
  };
}
