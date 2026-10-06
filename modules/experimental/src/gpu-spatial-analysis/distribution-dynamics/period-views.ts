// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';

/**
 * Returns the packed view of period `period` of a period-major column (`rows * periods` entries,
 * period `t` at `[t * rows, (t + 1) * rows)`), sharing the column's buffer.
 *
 * @internal
 */
export function getPeriodView<Parameters, Format extends GPUVectorFormat>(
  graph: GPUCommandGraph<Parameters>,
  column: GraphDataView<Format>,
  period: number,
  rows: number
): GraphDataView<Format> {
  return graph.createDataView(column.buffer, {
    format: column.format,
    length: rows,
    byteOffset: column.byteOffset + period * rows * column.rowByteLength
  });
}
