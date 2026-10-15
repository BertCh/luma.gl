// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellTopology} from '@luma.gl/experimental/gpu-spatial-analysis';

/** H3 cell keys (resolution 9) of the places whose neighbourhoods are expanded. */
export type CellTopologyExampleInput = {cells: GraphDataView<'uint32x2'>};

/** Expands each H3 cell to the disk of cells within 4 grid steps. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {cells}: CellTopologyExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const diskSize = 61; // 3k(k+1)+1 cells per input cell for k = 4
  const output = {
    cells: createTransientView(graph, 'diskCells', 'uint32x2', cells.length * diskSize, usage),
    distances: createTransientView(graph, 'distances', 'uint32', cells.length * diskSize, usage),
    counts: createTransientView(graph, 'counts', 'uint32', cells.length, usage)
  };
  graph.add(new GPUCellTopology({family: 'h3', operation: {type: 'disk', k: 4}, cells, output}));
  // Row i owns entries [i * 61, (i + 1) * 61), ordered by (distance, key).
  const {cells: diskCells, distances} = output;
  // #endregion example
  return {...output, diskCells, distances};
}
