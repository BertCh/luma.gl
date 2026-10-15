// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellCompaction} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Resolution-9 H3 cells, a mask that drops some of them, and the output capacity. */
export type CellCompactionExampleInput = {
  cells: GraphDataView<'uint32x2'>;
  mask: GraphDataView<'uint32'>;
  capacity: number;
};

/** Merges complete sibling sets of resolution-9 cells into their parents, down to resolution 7. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {cells, mask, capacity}: CellCompactionExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const compacted = {
    cells: createTransientView(graph, 'compactedCells', 'uint32x2', capacity, usage),
    status: {
      count: createTransientView(graph, 'compactedCount', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage),
      droppedCount: createTransientView(graph, 'droppedCount', 'uint32', 1, usage)
    }
  };
  graph.add(
    new GPUCellCompaction({
      family: 'h3',
      operation: {type: 'compact', resolution: 9, minimumResolution: 7},
      cells,
      mask, // masked rows are dropped before merging
      output: compacted
    })
  );
  // Rows [0, status.count) are the merged cells, mixed resolutions, ascending key order.
  const {cells: compactedCells} = compacted;
  // #endregion example
  return {...compacted, compactedCells};
}
