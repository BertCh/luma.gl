// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPURasterConnectedComponents,
  GPURasterDenseComponents,
  GPURasterPatchMetrics,
  makeGPURasterBand
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-patch-metrics. */
export type RasterPatchMetricsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  burnMask: GraphDataView<'uint32'>;
  sparseLabels: GraphDataView<'uint32'>;
  sparseValidity: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  labels: GraphDataView<'uint32'>;
  labelValidity: GraphDataView<'uint32'>;
  componentCount: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  pixelCounts: GraphDataView<'uint32'>;
  areas: GraphDataView<'float32'>;
  perimeters: GraphDataView<'float32'>;
  perimeterFaceCounts: GraphDataView<'uint32'>;
};

/** Adds the recorded raster-patch-metrics computation; it owns no extra resources. */
export async function runExample(device: Device, input: RasterPatchMetricsExampleInput) {
  const {graph, grid, burnMask, sparseLabels, sparseValidity, converged} = input;
  const {labels, labelValidity, componentCount, overflow} = input;
  const {pixelCounts, areas, perimeters, perimeterFaceCounts} = input;
  // #region example
  new GPURasterConnectedComponents({
    ...grid, // width and height
    input: makeGPURasterBand(burnMask, grid), // 1 where dNBR > 0.27
    output: sparseLabels,
    outputValidity: sparseValidity,
    converged
  }).addToGraph(graph);
  new GPURasterDenseComponents({
    ...grid, // width and height
    input: sparseLabels,
    inputValidity: sparseValidity,
    converged,
    output: labels, // patch 1, 2, ... in row-major order of their first cell
    outputValidity: labelValidity,
    componentCount,
    overflow
  }).addToGraph(graph);
  graph.add(
    new GPURasterPatchMetrics({
      grid,
      labels,
      status: {converged, componentCount, overflow},
      output: {pixelCounts, areas, perimeters, perimeterFaceCounts} // one row per patch
    })
  );
  // #endregion example
  return {};
}
