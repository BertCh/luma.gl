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
  GPUKriging,
  getGPUKrigingParameterValues,
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

export type KrigingExampleInput = {
  stations: GraphDataView<'float32x2'>;
  precipitation: GraphDataView<'float32'>;
  spatialContext: GPUSpatialContext;
  extent: readonly [number, number, number, number];
};

export async function runExample(graph: GPUCommandGraph<undefined>, input: KrigingExampleInput) {
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
  const variance = createTransientView(graph, 'variance', 'float32', width * height, usage);
  const invalidCount = createTransientView(graph, 'invalid-count', 'uint32', 1, usage);
  const uncertainCount = createTransientView(graph, 'uncertain-count', 'uint32', 1, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'kriging-parameters',
    format: 'float32',
    length: 12,
    values: getGPUKrigingParameterValues(
      {
        searchRadius: 150000,
        neighborCount: 8,
        minimumNeighborCount: 3,
        variogram: {model: 'spherical', nugget: 100, sill: 10000, range: 150000}
      },
      grid
    )
  });
  // #region example
  // Fixed spherical model: nugget 100 mm², partial sill 10000 mm², range 150 km.
  // Search: eight nearest stations within 150 km; minimum three distinct positions.
  graph.add(
    new GPUKriging({
      positions: stations,
      values: precipitation,
      spatialContext,
      parameters: parameters.importToGraph(graph),
      grid,
      indexGridSize: [32, 32],
      indexBounds: extent,
      maximumNeighborCount: 8,
      output: {values: rainfall, variance, status: {invalidCount, uncertainCount}}
    })
  );
  // rainfall in mm; variance in mm²; invalidCount and uncertainCount diagnose nodata
  // #endregion example
  return {rainfall, variance, invalidCount, uncertainCount, parameters};
}
