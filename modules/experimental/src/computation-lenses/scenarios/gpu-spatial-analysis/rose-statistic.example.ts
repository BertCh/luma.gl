// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURoseStatistic, type GPUSpatialContext} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Street segment ends in local metres, the segment lengths and the planar context of the frame. */
export type RoseStatisticExampleInput = {
  /** Segment start points, projected to local-frame metres. */
  starts: GraphDataView<'float32x2'>;
  /** Segment end points, projected to local-frame metres. */
  ends: GraphDataView<'float32x2'>;
  /** Segment lengths in metres; longer streets count more. */
  weights: GraphDataView<'float32'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** Number of rose sectors: 18 bins of 10 degrees over the half circle of axial directions. */
export const ROSE_BIN_COUNT = 18;

/** Axial rose of street directions: opposite directions of one street count as the same. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {starts, ends, weights, spatialContext}: RoseStatisticExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const summary = createTransientView(graph, 'summary', 'float32', 5, usage);
  const bins = createTransientView(graph, 'bins', 'float32', ROSE_BIN_COUNT, usage);
  // #region example
  graph.add(
    new GPURoseStatistic({
      spatialContext,
      starts,
      ends,
      weights,
      circular: {
        mode: 'axial',
        angleUnit: 'radians',
        origin: 'positive-x-counter-clockwise',
        binCount: ROSE_BIN_COUNT
      },
      output: {summary, bins}
    })
  );
  // summary: [count, mean angle, resultant length, circular variance, Rayleigh z]
  // bins: summed segment length per 10 degree sector
  // #endregion example
  return {summary, bins};
}
