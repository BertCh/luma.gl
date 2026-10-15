// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPURasterReclassify,
  GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH,
  getGPURasterReclassifyParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-reclassify. */
export type RasterReclassifyExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  landCover: GraphDataView<'float32'>;
  classBreaks: GraphDataView<'float32'>;
  suitabilityValues: GraphDataView<'float32'>;
  suitability: GraphDataView<'float32'>;
};

/** Adds the recorded raster-reclassify computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: RasterReclassifyExampleInput) {
  const {graph, landCover, classBreaks, suitabilityValues, suitability} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'reclassify-parameters',
    format: 'float32',
    length: GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH,
    values: getGPURasterReclassifyParameterValues({breakCount: 4, closed: 'left'})
  });
  graph.add(
    new GPURasterReclassify({
      values: landCover, // WorldCover class codes 10, 30, 40, 50, 80
      breaks: classBreaks, // 20, 35, 45, 65 split the five classes
      classValues: suitabilityValues, // suitability of each interval
      noDataValue: 0,
      parameters: parameters.importToGraph(graph),
      output: {reclassified: suitability} // rank 1 (low) to 5 (high)
    })
  );
  // #endregion example
  return {parameters};
}
