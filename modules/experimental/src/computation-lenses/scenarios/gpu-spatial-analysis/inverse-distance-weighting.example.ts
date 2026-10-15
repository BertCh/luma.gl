// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUInverseDistanceWeighting,
  getGPUInverseDistanceWeightingParameterValues,
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

export type InverseDistanceWeightingExampleInput = {
  stations: GraphDataView<'float32x2'>;
  precipitation: GraphDataView<'float32'>;
  spatialContext: GPUSpatialContext;
  extent: readonly [number, number, number, number];
};

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: InverseDistanceWeightingExampleInput
) {
  const {stations, precipitation, spatialContext, extent} = input;
  const width = Math.round((extent[2] - extent[0]) / 10000);
  const height = Math.round((extent[3] - extent[1]) / 10000);
  const grid = makeGPURasterMetadata({
    width,
    height,
    cellSize: 10000,
    origin: [extent[0], extent[3]]
  });
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const rainfall = createTransientView(graph, 'rainfall', 'float32', width * height, usage);
  const counts = createTransientView(graph, 'counts', 'uint32', width * height, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'inverse-distance-weighting-parameters',
    format: 'float32',
    length: 9,
    values: getGPUInverseDistanceWeightingParameterValues(
      {searchRadius: 150000, power: 2, neighborCount: 8, minimumNeighborCount: 3, taper: 'none'},
      grid
    )
  });
  // #region example
  // Parameters: power 2, no taper, eight nearest stations within 150 km; minimum three.
  graph.add(
    new GPUInverseDistanceWeighting({
      positions: stations,
      values: precipitation,
      spatialContext,
      parameters: parameters.importToGraph(graph),
      grid,
      indexGridSize: [32, 32],
      indexBounds: extent,
      maximumNeighborCount: 8,
      output: {values: rainfall, counts}
    })
  );
  // rainfall in mm; counts reports contributing stations per cell
  // #endregion example
  return {rainfall, counts, parameters};
}
