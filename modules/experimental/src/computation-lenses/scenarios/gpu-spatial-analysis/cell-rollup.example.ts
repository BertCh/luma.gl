// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellRollup, type GPUCellTable} from '@luma.gl/experimental/gpu-spatial-analysis';

/** The zoom-18 cell table of the places and the parent table capacity. */
export type CellRollupExampleInput = {source: GPUCellTable; capacity: number};

/** Rolls the zoom-18 place counts up to zoom-16 parent cells without re-reading the places. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {source, capacity}: CellRollupExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const parents = {
    cells: createTransientView(graph, 'cells', 'uint32x2', capacity, usage),
    counts: createTransientView(graph, 'counts', 'uint32', capacity, usage),
    status: {
      count: createTransientView(graph, 'cellCount', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage)
    }
  };
  graph.add(
    new GPUCellRollup({
      family: 'quadbin',
      sourceResolution: 18,
      resolution: 16,
      source,
      output: parents
    })
  );
  // Parent counts are integer sums of their child counts, so they equal a direct zoom-16 aggregation.
  const {cells, counts} = parents;
  // #endregion example
  return {...parents, cells, counts};
}
