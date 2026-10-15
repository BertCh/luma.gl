// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellGridPath} from '@luma.gl/experimental/gpu-spatial-analysis';

/** H3 cell keys (resolution 9) of each origin place and its destination, one pair per row. */
export type CellGridPathExampleInput = {
  origins: GraphDataView<'uint32x2'>;
  destinations: GraphDataView<'uint32x2'>;
};

/** Finds the H3 grid distance and the cells of the grid path between each origin and destination. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {origins, destinations}: CellGridPathExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const maximumPathLength = 16; // cells stored per pair
  const output = {
    distances: createTransientView(graph, 'distances', 'uint32', origins.length, usage),
    counts: createTransientView(graph, 'counts', 'uint32', origins.length, usage),
    cells: createTransientView(
      graph,
      'pathCells',
      'uint32x2',
      origins.length * maximumPathLength,
      usage
    )
  };
  graph.add(new GPUCellGridPath({family: 'h3', origins, destinations, maximumPathLength, output}));
  // Pair i owns path rows [i * 16, i * 16 + counts[i]): origin first, destination last.
  const {distances, cells: pathCells} = output;
  // #endregion example
  return {...output, distances, pathCells};
}
