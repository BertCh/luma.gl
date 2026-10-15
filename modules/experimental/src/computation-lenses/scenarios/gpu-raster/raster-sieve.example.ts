// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPURasterConnectedComponents,
  GPURasterDenseComponents,
  GPURasterSieve,
  makeGPURasterBand,
  GPU_RASTER_SIEVE_PARAMETER_LENGTH,
  getGPURasterSieveParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-sieve. */
export type RasterSieveExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  burnMask: GraphDataView<'uint32'>;
  sparseLabels: GraphDataView<'uint32'>;
  sparseValidity: GraphDataView<'uint32'>;
  labelValidity: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  labels: GraphDataView<'uint32'>;
  componentCount: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  sievedLabels: GraphDataView<'uint32'>;
  patchTargets: GraphDataView<'uint32'>;
  sievedCount: GraphDataView<'uint32'>;
};

/** Labels the burn-mask patches, then adds the recorded raster sieve; returns its parameter buffer. */
export async function runExample(device: Device, input: RasterSieveExampleInput) {
  const {graph, grid, converged, labels, componentCount, overflow} = input;
  const {sievedLabels, patchTargets, sievedCount} = input;
  // Prerequisite: 4-connected patches of the burn mask, densely labelled.
  new GPURasterConnectedComponents({
    ...grid,
    input: makeGPURasterBand(input.burnMask, grid),
    output: input.sparseLabels,
    outputValidity: input.sparseValidity,
    converged
  }).addToGraph(graph);
  new GPURasterDenseComponents({
    ...grid,
    input: input.sparseLabels,
    inputValidity: input.sparseValidity,
    converged,
    output: labels,
    outputValidity: input.labelValidity,
    componentCount,
    overflow
  }).addToGraph(graph);
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'sieve-parameters',
    format: 'uint32',
    length: GPU_RASTER_SIEVE_PARAMETER_LENGTH,
    values: getGPURasterSieveParameterValues({minimumPixels: 12}) // keep patches of 12+ cells
  });
  graph.add(
    new GPURasterSieve({
      grid,
      labels, // dense patch labels of the burn mask, from GPURasterDenseComponents
      status: {converged, componentCount, overflow}, // guards: unfinished labels are never sieved
      patchCapacity: 48,
      parameters: parameters.importToGraph(graph),
      output: {labels: sievedLabels, patchTargets, sievedCount} // removed patches become 0
    })
  );
  // #endregion example
  return {parameters};
}
