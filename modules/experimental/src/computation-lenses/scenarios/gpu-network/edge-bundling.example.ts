// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUEdgeBundling} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: zone centroids (longitude/latitude) and one origin and destination zone per flow. */
export type EdgeBundlingExampleInput = {
  graph: GPUCommandGraph;
  zonePositions: GraphDataView<'float32x2'>;
  originZones: GraphDataView<'uint32'>;
  destinationZones: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: EdgeBundlingExampleInput) {
  const {graph, zonePositions, originZones, destinationZones} = input;
  const flowCount = originZones.length;
  // #region example
  const bundling = new GPUEdgeBundling({
    nodePositions: zonePositions,
    sourceNodes: originZones,
    targetNodes: destinationZones,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
    pointsPerEdge: 16,
    maximumIterations: 8,
    densityResolution: 128,
    output: {
      paths: createTransientView(graph, 'bundledPaths', 'float32x2', flowCount * 16),
      lineOffsets: createTransientView(graph, 'bundleOffsets', 'uint32', flowCount + 1)
    }
  });
  graph.add(bundling);
  const {paths, lineOffsets} = bundling.props.output;
  // #endregion example
  return {bundling, paths, lineOffsets: lineOffsets!};
}
