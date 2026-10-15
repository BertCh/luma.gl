// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUPointToCell} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Overture places uploaded by the lens scenario, as longitude/latitude degrees. */
export type PointToCellExampleInput = {places: GraphDataView<'float32x2'>};

/** Keys every place to its H3 cell at resolution 9. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {places}: PointToCellExampleInput
) {
  // #region example
  const cells = createTransientView(
    graph,
    'cells',
    'uint32x2',
    places.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  graph.add(
    new GPUPointToCell({
      family: 'h3',
      resolution: 9, // cells of about 0.1 km²
      positions: places, // longitude/latitude degrees
      output: {cells} // 64-bit keys as (low, high) uint32 words
    })
  );
  // #endregion example
  return {cells};
}
