// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULineDensity,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Roads in UTM 16N metres (planar) and the grid uploaded by the lens scenario. */
export type LineDensityExampleInput = {
  roads: GPULineGeometryFields;
  /** `[minX, minY, cellWidth, cellHeight]` from `getGPULineDensityParameterValues`, in metres. */
  parameters: GraphDataView<'float32'>;
  columns: number;
  rows: number;
};

/** Road length and density in every cell of a 100 m grid over Wicker Park. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {roads, parameters, columns, rows}: LineDensityExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = {
    lengths: createTransientView(graph, 'lengths', 'float32', columns * rows, usage),
    densities: createTransientView(graph, 'densities', 'float32', columns * rows, usage),
    status: {
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage),
      requiredCount: createTransientView(graph, 'pieceCount', 'uint32', 1, usage)
    }
  };
  graph.add(new GPULineDensity({...roads, columns, rows, parameters, output}));
  // lengths: m of road per cell; densities: m of road per m² of cell, rows from the south
  const {lengths, densities} = output;
  // #endregion example
  return {...output, lengths, densities};
}
