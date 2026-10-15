// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUHilbertKeys} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Places uploaded by the lens scenario, in UTM 16N metres (plain coordinates, no metric). */
export type HilbertKeysExampleInput = {places: GraphDataView<'float32x2'>};

/** Hilbert curve key of every place, and the places in curve order. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {places}: HilbertKeysExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = {
    keys: createTransientView(graph, 'keys', 'uint32', places.length, usage),
    sortedRows: createTransientView(graph, 'sortedRows', 'uint32', places.length, usage),
    sortedKeys: createTransientView(graph, 'sortedKeys', 'uint32', places.length, usage)
  };
  // Order 6: a 64 x 64 curve over the places' extent, reduced on the GPU
  graph.add(new GPUHilbertKeys({positions: places, order: 6, output}));
  // keys: curve index in [0, 4096) per place; sortedRows: place rows in curve order
  const {keys, sortedRows} = output;
  // #endregion example
  return {...output, keys, sortedRows};
}
