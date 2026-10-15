// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryResample,
  type GPUTrajectoryGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded AIS tracks for GPUTrajectoryResample. */
export type TrajectoryResampleExampleInput = {
  tracks: GPUTrajectoryGeometryFields;
  sampleCount: number;
};

/** Runs GPUTrajectoryResample and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: TrajectoryResampleExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {tracks, sampleCount} = input;
  const trackCount = tracks.trackOffsets.length - 1;
  const output = {
    samples: createTransientView(graph, 'samples', 'float32x2', trackCount * sampleCount, usage),
    sampleTimes: createTransientView(
      graph,
      'sampleTimes',
      'float32',
      trackCount * sampleCount,
      usage
    )
  };
  // #region example
  // UTM 18N metres: 16 samples per track, evenly spaced along its path length.
  graph.add(new GPUTrajectoryResample({...tracks, sampleCount, spacing: 'arc-length', output}));
  const {samples} = output;
  // #endregion example
  return {...output, samples};
}
