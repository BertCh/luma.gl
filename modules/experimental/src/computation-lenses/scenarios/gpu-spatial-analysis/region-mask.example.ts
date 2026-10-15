// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURegionMask} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Planar tract centroids and rectangle bounds `[minX, minY, maxX, maxY]`; the caller owns the graph. */
export type RegionMaskInput = {
  graph: GPUCommandGraph;
  positions: GraphDataView<'float32x2'>;
  bounds: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the region mask result figure. */
export async function runExample(device: Device, input: RegionMaskInput) {
  const {graph, positions, bounds} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const mask = createTransientView(graph, 'mask', 'uint32', positions.length, usage);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1, usage);
  // #region example
  // One 0/1 value per tract centroid: 1 inside the inclusive rectangle.
  graph.add(
    new GPURegionMask({
      positions,
      region: {kind: 'rectangle', bounds},
      output: {mask, status: {overflow}}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    views: {positions, bounds, mask, overflow}
  };
}
