// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURegionStatistics} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Number of histogram bins in the summary. */
export const REGION_STATISTICS_BIN_COUNT = 8;

/** Planar tract centroids, counts and a lasso polygon; the caller owns the recording graph. */
export type RegionStatisticsInput = {
  graph: GPUCommandGraph;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  lassoVertices: GraphDataView<'float32x2'>;
  lassoVertexCount: GraphDataView<'uint32'>;
};

/** Runs the exact public-API example shown below the region statistics result figure. */
export async function runExample(device: Device, input: RegionStatisticsInput) {
  const {graph, positions, values, lassoVertices, lassoVertexCount} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const mask = createTransientView(graph, 'mask', 'uint32', positions.length, usage);
  const summary = createTransientView(
    graph,
    'summary',
    'uint32',
    8 + REGION_STATISTICS_BIN_COUNT,
    usage
  );
  // #region example
  // The lasso selects tract centroids; the summary counts them and reduces their values.
  graph.add(
    new GPURegionStatistics({
      positions,
      values,
      region: {kind: 'polygon', vertices: lassoVertices, vertexCount: lassoVertexCount},
      histogram: {binCount: REGION_STATISTICS_BIN_COUNT},
      output: {mask, summary}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    views: {positions, values, lassoVertices, lassoVertexCount, mask, summary}
  };
}
