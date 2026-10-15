// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUPointDensity, type GPUSpatialContext} from '@luma.gl/experimental/gpu-spatial-analysis';

export type PointDensityExampleInput = {
  observations: GraphDataView<'float32x2'>;
  spatialContext: GPUSpatialContext;
  extent: readonly [number, number, number, number];
};

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: PointDensityExampleInput
) {
  // #region example
  const {observations, spatialContext, extent} = input;
  const width = Math.round((extent[2] - extent[0]) / 125);
  const height = Math.round((extent[3] - extent[1]) / 125);
  const counts = createTransientView(
    graph,
    'counts',
    'float32',
    width * height,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  graph.add(
    new GPUPointDensity({
      positions: observations,
      spatialContext,
      bounds: extent,
      gridSize: [width, height],
      statistic: 'count',
      output: {values: counts}
    })
  );
  // counts: observations per 125 m cell, without smoothing or area normalization
  // #endregion example
  return {counts};
}
