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
  GPULineNoding,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Street centrelines in UTM 16N metres and the noded-segment slots to write into. */
export type LineNodingExampleInput = {
  roads: GPULineGeometryFields;
  /** Segment capacity rows, uploaded as NaN so unwritten slots are not drawn. */
  segmentSlots: GraphDataView<'float32x4'>;
};

/** Splits the street centrelines at every crossing, touch and shared vertex. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {roads, segmentSlots}: LineNodingExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const capacity = segmentSlots.length;
  const column = <Format extends 'uint32' | 'float32'>(id: string, format: Format) =>
    createTransientView(graph, id, format, capacity, usage);
  const scalar = (id: string) => createTransientView(graph, id, 'uint32', 1, usage);
  // #region example
  const output = {
    endpoints: segmentSlots, // (x0, y0, x1, y1) in metres
    sourceFeatureIds: column('sourceFeatureIds', 'uint32'),
    sourceRingIds: column('sourceRingIds', 'uint32'),
    sourceEdgeIds: column('sourceEdgeIds', 'uint32'),
    sourceStartParameters: column('startParameters', 'float32'),
    sourceEndParameters: column('endParameters', 'float32'),
    status: {
      count: scalar('count'),
      requiredCount: scalar('requiredCount'),
      overflow: scalar('overflow'),
      candidateOverflow: scalar('candidateOverflow')
    }
  };
  graph.add(
    new GPULineNoding({lines: {kind: 'lines', ...roads}, intersectionCapacity: 1024, output})
  );
  const {endpoints, sourceFeatureIds, sourceEdgeIds} = output; // one row per atomic segment
  // #endregion example
  return {...output, endpoints, sourceFeatureIds, sourceEdgeIds};
}
