// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkKFunction, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: a symmetric road network in metres, crash events and the scalar settings. */
export type NetworkKFunctionExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  crashPositions: GraphDataView<'float32x2'>;
  maximumDistance: GraphDataView<'float32'>;
  maximumSnapDistance: GraphDataView<'float32'>;
  networkLength: GraphDataView<'float32'>;
  parameters: GraphDataView<'uint32'>;
  /** Number of seeded random patterns used for the pointwise envelope. */
  simulationCount: number;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkKFunctionExampleInput) {
  const {graph, roads, crashPositions, maximumDistance, maximumSnapDistance} = input;
  // A larger batch limits repeated graph structure while keeping scratch storage docs-sized.
  const patternBandCount = (1 + input.simulationCount) * 8;
  // #region example
  const kFunction = new GPUNetworkKFunction({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    topology: roads,
    positions: crashPositions,
    maximumDistance,
    maximumSnapDistance,
    networkLength: input.networkLength,
    parameters: input.parameters,
    bandCount: 8,
    simulationCount: input.simulationCount,
    rowsPerBlock: 256,
    maximumIterations: 24,
    output: {
      kValues: createTransientView(graph, 'kValues', 'float32', patternBandCount),
      pairCounts: createTransientView(graph, 'pairCounts', 'uint32', patternBandCount),
      envelope: createTransientView(graph, 'envelope', 'float32', 24),
      snappedEventCount: createTransientView(graph, 'snappedEventCount', 'uint32', 1),
      status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
    }
  });
  graph.add(kFunction);
  const {envelope} = kFunction.props.output;
  const {kValues, pairCounts, snappedEventCount} = kFunction.props.output;
  // #endregion example
  return {
    kFunction,
    kValues,
    envelope: envelope!,
    pairCounts: pairCounts!,
    snappedEventCount: snappedEventCount!,
    converged: kFunction.props.output.status!.converged!
  };
}
