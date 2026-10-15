// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellAggregation} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Overture places (longitude/latitude degrees) and the cell table capacity. */
export type CellAggregationExampleInput = {
  places: GraphDataView<'float32x2'>;
  capacity: number;
};

/** Counts places per Quadbin cell at zoom 18 (cells of about 0.01 km²). */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {places, capacity}: CellAggregationExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const table = {
    cells: createTransientView(graph, 'cells', 'uint32x2', capacity, usage),
    counts: createTransientView(graph, 'counts', 'uint32', capacity, usage),
    status: {
      count: createTransientView(graph, 'cellCount', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage)
    }
  };
  graph.add(
    new GPUCellAggregation({family: 'quadbin', resolution: 18, positions: places, output: table})
  );
  // Rows [0, status.count) are occupied cells in ascending key order; capacity bounds the table.
  const {cells, counts} = table;
  // #endregion example
  return {...table, cells, counts};
}
