// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCellCover,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Community areas in longitude/latitude degrees and the cell column to write into. */
export type CellCoverExampleInput = {
  areas: GPUPolygonGeometryFields;
  /** Output capacity rows, uploaded as zero (empty) keys so rows the cover never writes stay empty. */
  cellSlots: GraphDataView<'uint32x2'>;
};

/** Polyfills the community areas with H3 resolution-8 cells whose centres fall inside. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {areas, cellSlots}: CellCoverExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const scalar = (id: string) => createTransientView(graph, id, 'uint32', 1, usage);
  const output = {
    featureIds: createTransientView(graph, 'featureIds', 'uint32', cellSlots.length, usage),
    cells: cellSlots, // (low, high) uint32 words; rows past status.count stay empty
    status: {
      count: scalar('count'),
      requiredCount: scalar('requiredCount'),
      overflow: scalar('overflow'),
      candidateOverflow: scalar('candidateOverflow')
    }
  };
  graph.add(
    new GPUCellCover({
      ...areas, // longitude/latitude degrees, planar edges
      family: 'h3',
      resolution: 8, // cells of about 0.74 km²
      containment: 'center',
      candidateCapacity: 1 << 15, // lattice points tested over all areas
      output
    })
  );
  const {featureIds, cells} = output; // one (area, cell) pair per row, in area order
  // #endregion example
  return {...output, featureIds, cells};
}
