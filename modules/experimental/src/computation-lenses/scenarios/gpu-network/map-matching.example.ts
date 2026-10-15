// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUMapMatching, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: planar roads in metres, GPS fixes in the same frame and the track offsets. */
export type MapMatchingExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  fixes: GraphDataView<'float32x2'>;
  trackOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  bounds: {minimum: readonly [number, number]; maximum: readonly [number, number]};
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: MapMatchingExampleInput) {
  const {graph, roads, fixes, trackOffsets, parameters, bounds} = input;
  const fixCount = fixes.length;
  // #region example
  const matching = new GPUMapMatching({
    topology: roads,
    positions: fixes,
    trackOffsets,
    parameters,
    candidateCount: 6,
    cellSize: 100,
    bounds,
    routeNodeBudget: 128,
    output: {
      matchedEdges: createTransientView(graph, 'matchedEdges', 'uint32', fixCount),
      snappedPositions: createTransientView(graph, 'snappedPositions', 'float32x2', fixCount),
      snapDistances: createTransientView(graph, 'matchSnapDistances', 'float32', fixCount),
      breaks: createTransientView(graph, 'matchBreaks', 'uint32', fixCount),
      matchedCount: createTransientView(graph, 'matchedCount', 'uint32', 1),
      status: {overflow: createTransientView(graph, 'matchOverflow', 'uint32', 1)}
    }
  });
  graph.add(matching);
  const {matchedEdges, snappedPositions, breaks} = matching.props.output;
  // #endregion example
  const {snapDistances, matchedCount, status} = matching.props.output;
  return {
    matching,
    matchedEdges,
    snappedPositions: snappedPositions!,
    snapDistances: snapDistances!,
    breaks: breaks!,
    matchedCount: matchedCount!,
    overflow: status!.overflow!
  };
}
