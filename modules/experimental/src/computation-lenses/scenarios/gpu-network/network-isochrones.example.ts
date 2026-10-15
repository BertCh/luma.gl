// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkIsochrones, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: planar roads in meters, one start, cost breaks and packed raster parameters. */
export type NetworkIsochronesExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  startNodes: GraphDataView<'uint32'>;
  costBreaks: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  triangleCapacity: number;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkIsochronesExampleInput) {
  const {graph, roads, startNodes, costBreaks, parameters, triangleCapacity} = input;
  const nodeCount = roads.offsets.length - 1;
  const costs = createTransientView(graph, 'costs', 'float32', nodeCount);
  const converged = createTransientView(graph, 'converged', 'uint32', 1);
  // #region example
  const isochrones = new GPUNetworkIsochrones({
    topology: roads,
    startNodes,
    costBreaks,
    parameters,
    costs,
    maximumIterations: 24,
    output: {status: {converged}},
    raster: {
      width: 48,
      height: 48,
      mode: 'min',
      output: {
        triangles: createTransientView(graph, 'triangles', 'float32x2', 3 * triangleCapacity),
        triangleBands: createTransientView(graph, 'triangleBands', 'uint32', triangleCapacity),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    }
  });
  graph.add(isochrones);
  const {triangles, triangleBands} = isochrones.props.raster!.output;
  // #endregion example
  const {status} = isochrones.props.raster!.output;
  return {
    isochrones,
    costs,
    converged,
    triangles: triangles!,
    triangleBands: triangleBands!,
    status: status!
  };
}
