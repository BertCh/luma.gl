// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellGeometry} from '@luma.gl/experimental/gpu-spatial-analysis';

/** H3 cell keys (resolution 9) uploaded or produced by the lens scenario. */
export type CellGeometryExampleInput = {cells: GraphDataView<'uint32x2'>};

/** Decodes H3 cells to centres and boundary polygons. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {cells}: CellGeometryExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = {
    centers: createTransientView(graph, 'centers', 'float32x2', cells.length, usage),
    boundaries: createTransientView(graph, 'boundaries', 'float32x2', cells.length * 6, usage),
    vertexCounts: createTransientView(graph, 'vertexCounts', 'uint32', cells.length, usage)
  };
  graph.add(
    new GPUCellGeometry({
      family: 'h3',
      cells,
      maximumVertexCount: 6, // hexagons; pentagons and face-crossing cells need up to 10
      output
    })
  );
  const {centers, boundaries} = output; // longitude/latitude degrees
  // #endregion example
  return {...output, centers, boundaries};
}
