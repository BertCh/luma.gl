// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {GPUFocalStatistics} from '@luma.gl/experimental/gpu-spatial-analysis';

/** An elevation raster, its shape and packed window settings; the caller owns the recording graph. */
export type FocalStatisticsInput = {
  graph: GPUCommandGraph;
  elevation: GraphDataView<'float32'>;
  /** `[radius, shape, minimumCount, propagateCenterNoData]` from `getGPUFocalStatisticsParameterValues`. */
  parameters: GraphDataView<'float32'>;
  width: number;
  height: number;
  cellSize: number;
  origin: readonly [number, number];
};

/** Largest window radius, in cells, the compiled kernel accepts. */
export const FOCAL_MAXIMUM_RADIUS = 4;

/** Runs the exact public-API example shown below the focal statistics result figure. */
export async function runExample(device: Device, input: FocalStatisticsInput) {
  const {graph, elevation, parameters, width, height, cellSize, origin} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const mean = createTransientView(graph, 'mean', 'float32', width * height, usage);
  const standardDeviation = createTransientView(
    graph,
    'standardDeviation',
    'float32',
    width * height,
    usage
  );
  const windowCounts = createTransientView(graph, 'windowCounts', 'uint32', width * height, usage);
  const grid = makeGPURasterMetadata({width, height, cellSize, origin: [...origin]});
  // #region example
  // Every cell summarises the valid cells of the window around it; edge windows are clipped.
  graph.add(
    new GPUFocalStatistics({
      values: elevation,
      grid,
      maximumRadius: FOCAL_MAXIMUM_RADIUS,
      parameters,
      output: {mean, standardDeviation, counts: windowCounts}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    views: {elevation, mean, standardDeviation, windowCounts}
  };
}
