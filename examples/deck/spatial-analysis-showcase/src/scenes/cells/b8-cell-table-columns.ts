// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCellTable} from '@luma.gl/experimental/gpu-spatial-analysis';
import {addKernelPass} from '../../engine/mode-kernels';

/**
 * Adapter that turns a cell table into the per-row columns the column statistics read, with the
 * same semantics as the library's recipe adapter `addCellTableColumnsNode`: `values[i]` is
 * `sumValues[i]` when the table has it, else the row count as f32, and `mask[i]` is 1 for occupied
 * rows and 0 for the empty tail of the capacity-bounded table.
 *
 * `addCellTableColumnsNode` is internal to the library recipes and not exported from the package
 * barrel, so this scene carries a local copy built on the showcase kernel helper. The one addition
 * is the optional `rowCount`: a rolled-up table is rewritten at every resolution, so rows past its
 * GPU row count may still hold the previous resolution's counts and must be masked out.
 */
export function addCellTableColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  table: Pick<GPUCellTable, 'counts' | 'sumValues'>,
  output: {values: GraphDataView<'float32'>; mask: GraphDataView<'uint32'>},
  rowCount?: GraphDataView<'uint32'>
): void {
  const rows = table.counts.length;
  addKernelPass(graph, {
    id: `${id}-cell-table-columns`,
    invocationCount: rows,
    bindings: [
      {name: 'counts', view: table.counts, type: 'u32', access: 'read'},
      ...(table.sumValues
        ? [{name: 'sumValues', view: table.sumValues, type: 'f32', access: 'read'} as const]
        : []),
      ...(rowCount
        ? [{name: 'rowCount', view: rowCount, type: 'u32', access: 'read'} as const]
        : []),
      {name: 'columnValues', view: output.values, type: 'f32', access: 'read_write'},
      {name: 'columnMask', view: output.mask, type: 'u32', access: 'read_write'}
    ],
    body: `let occupied = counts[countsOffset + index];
  let inRange = ${rowCount ? 'index < rowCount[rowCountOffset]' : 'true'};
  columnValues[columnValuesOffset + index] = ${
    table.sumValues ? 'sumValues[sumValuesOffset + index]' : 'f32(occupied)'
  };
  columnMask[columnMaskOffset + index] = select(0u, 1u, inRange && occupied > 0u);`
  });
}
