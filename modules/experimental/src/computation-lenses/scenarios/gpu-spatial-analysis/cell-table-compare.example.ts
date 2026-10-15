// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellTableCompare, type GPUCellTable} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Two Quadbin zoom-17 count tables (restaurants and shops) and the union capacity. */
export type CellTableCompareExampleInput = {
  restaurants: GPUCellTable;
  shops: GPUCellTable;
  capacity: number;
};

/** Outer-joins the restaurant and shop counts per cell and differences them. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {restaurants, shops, capacity}: CellTableCompareExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const column = <Format extends 'uint32x2' | 'uint32' | 'float32'>(id: string, format: Format) =>
    createTransientView(graph, id, format, capacity, usage);
  const output = {
    cells: column('cells', 'uint32x2'), // union keys, ascending
    presence: column('presence', 'uint32'), // bit 1: in restaurants, bit 2: in shops
    before: column('before', 'float32'), // restaurants per cell
    after: column('after', 'float32'), // shops per cell
    delta: column('delta', 'float32'), // after - before
    zScore: column('zScore', 'float32'),
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage)
    }
  };
  graph.add(new GPUCellTableCompare({before: restaurants, after: shops, measure: 'count', output}));
  const {cells, delta} = output; // rows past status.count hold the empty key
  // #endregion example
  return {...output, cells, delta};
}
