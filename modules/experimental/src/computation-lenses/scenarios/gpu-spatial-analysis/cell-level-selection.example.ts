// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellLevelSelection} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Per-level cell counts from the pyramid and the active level index (one uint32). */
export type CellLevelSelectionExampleInput = {
  levelCounts: GraphDataView<'uint32'>;
  activeLevel: GraphDataView<'uint32'>;
};

/** Publishes the cell count and first row of the pyramid level chosen by `activeLevel`. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {levelCounts, activeLevel}: CellLevelSelectionExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const selection = {
    count: createTransientView(graph, 'selectedCount', 'uint32', 1, usage),
    firstRow: createTransientView(graph, 'selectedFirstRow', 'uint32', 1, usage)
  };
  graph.add(
    new GPUCellLevelSelection({
      levelCounts,
      activeLevel, // rewrite this one uint32 to switch levels without rebuilding the pyramid
      output: {status: {count: selection.count}, firstRow: selection.firstRow}
    })
  );
  const {count, firstRow} = selection;
  // #endregion example
  return {count, firstRow};
}
