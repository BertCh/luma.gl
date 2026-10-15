// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCellMeasures} from '@luma.gl/experimental/gpu-spatial-analysis';

/** H3 cell keys (resolution 9) of the places to measure. */
export type CellMeasuresExampleInput = {cells: GraphDataView<'uint32x2'>};

/** Measures the spherical area and perimeter of each H3 cell. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {cells}: CellMeasuresExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = {
    areas: createTransientView(graph, 'areas', 'float32', cells.length, usage),
    perimeters: createTransientView(graph, 'perimeters', 'float32', cells.length, usage),
    pentagons: createTransientView(graph, 'pentagons', 'uint32', cells.length, usage)
  };
  graph.add(new GPUCellMeasures({family: 'h3', cells, units: 'meters', output}));
  const {areas, perimeters} = output; // m² and m on the WGS84 authalic sphere
  // #endregion example
  return {...output, areas, perimeters};
}
