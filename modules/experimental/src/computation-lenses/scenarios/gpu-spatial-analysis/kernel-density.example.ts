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
  getGPUKernelDensityParameterValues,
  GPUKernelDensity,
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type KernelDensityExampleInput = {
  /** iNaturalist observations, projected to equal-area local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Study extent `[minX, minY, maxX, maxY]` in frame metres, a whole number of cells. */
  extent: readonly [number, number, number, number];
};

/** Kernel density of observations on a 125 m grid with a 400 m gaussian bandwidth. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, spatialContext, extent}: KernelDensityExampleInput
) {
  // #region example
  const [minX, minY, maxX, maxY] = extent;
  const cellSize = 125; // metres
  const [width, height] = [(maxX - minX) / cellSize, (maxY - minY) / cellSize].map(Math.round);
  const grid = makeGPURasterMetadata({width, height, cellSize, origin: [minX, maxY]});
  const values = getGPUKernelDensityParameterValues({bandwidth: 400}, grid);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'kde-parameters',
    format: 'float32',
    length: values.length,
    values
  });
  const usage = Buffer.STORAGE | Buffer.COPY_SRC; // COPY_SRC: read the result back
  const density = createTransientView(graph, 'density', 'float32', width * height, usage);
  graph.add(
    new GPUKernelDensity({
      spatialContext,
      positions: observations,
      parameters: parameters.importToGraph(graph),
      grid,
      indexGridSize: [32, 32],
      indexBounds: extent,
      output: {values: density}
    })
  );
  // density: observations per m², edge-corrected to the extent
  // #endregion example
  return {density, grid, parameters};
}
