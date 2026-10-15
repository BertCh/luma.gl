// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellPyramid} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Overture places (longitude/latitude degrees) and the table capacity of each level. */
export type CellPyramidExampleInput = {
  places: GraphDataView<'float32x2'>;
  capacities: readonly number[];
};

/** Builds place counts at Quadbin zooms 18, 17 and 16 in one pass over the places. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {places, capacities}: CellPyramidExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const levels = capacities.map((capacity, level) => ({
    cells: createTransientView(graph, `level${level}Cells`, 'uint32x2', capacity, usage),
    counts: createTransientView(graph, `level${level}Counts`, 'uint32', capacity, usage),
    status: {
      count: createTransientView(graph, `level${level}Count`, 'uint32', 1, usage),
      overflow: createTransientView(graph, `level${level}Overflow`, 'uint32', 1, usage)
    }
  }));
  const levelCounts = createTransientView(graph, 'levelCounts', 'uint32', 3, usage);
  graph.add(
    new GPUCellPyramid({
      family: 'quadbin',
      positions: places,
      resolutions: [18, 17, 16], // finest first
      output: {levels, levelCounts}
    })
  );
  // levels[0] is aggregated from the places; each coarser level rolls up the one before it.
  const [finest, , coarsest] = levels;
  // #endregion example
  return {levels, levelCounts, finest, coarsest};
}
