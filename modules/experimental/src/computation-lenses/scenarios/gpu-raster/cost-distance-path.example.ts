// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView, GPUHistogram} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPUCostDistance,
  GPU_COST_DISTANCE_PARAMETER_LENGTH,
  GPUCostDistancePath,
  type GPUCostDistancePathOutput,
  makeGPURasterBand,
  getGPUCostDistanceParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for cost-distance-path. */
export type CostDistancePathExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  friction: GraphDataView<'float32'>;
  sourceMask: GraphDataView<'uint32'>;
  costs: GraphDataView<'float32'>;
  backLinks: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  iterationCount: GraphDataView<'uint32'>;
  targets: {
    target: GraphDataView<'uint32'>;
    path: GPUCostDistancePathOutput['path'];
    mask: GraphDataView<'uint32'>;
  }[];
};

/** Adds the recorded cost-distance-path computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: CostDistancePathExampleInput) {
  const {graph, grid, friction, sourceMask, costs, backLinks, converged, iterationCount, targets} =
    input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'cost-path-parameters',
    format: 'float32',
    length: GPU_COST_DISTANCE_PARAMETER_LENGTH,
    values: getGPUCostDistanceParameterValues({}, grid)
  });
  graph.add(
    new GPUCostDistance({
      grid,
      friction: makeGPURasterBand(friction, grid),
      sourceMask,
      maximumIterations: 128,
      parameters: parameters.importToGraph(graph),
      output: {costs, backLinks, status: {converged, iterationCount}}
    })
  );
  for (const [index, {target, path, mask}] of targets.entries()) {
    graph.add(
      new GPUCostDistancePath({id: `walk-${index}`, grid, backLinks, target, output: {path}})
    );
    // The histogram counts each traced cell ID once.
    const domain: [number, number] = [0, mask.length];
    graph.add(new GPUHistogram({id: `mask-${index}`, input: path.ids, domain, output: mask}));
  }
  // #endregion example
  return {parameters};
}
