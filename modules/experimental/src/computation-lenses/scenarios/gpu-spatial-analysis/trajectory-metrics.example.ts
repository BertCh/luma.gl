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
  GPUTrajectoryMetrics,
  type GPUTrajectoryGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded AIS tracks and stop parameters for GPUTrajectoryMetrics. */
export type TrajectoryMetricsExampleInput = {
  tracks: GPUTrajectoryGeometryFields;
  parameters: GraphDataView<'float32'>;
  stopCapacity: number;
};

/** Runs GPUTrajectoryMetrics and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: TrajectoryMetricsExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {tracks, parameters, stopCapacity} = input;
  const trackCount = tracks.trackOffsets.length - 1;
  const output = {
    trackLengths: createTransientView(graph, 'trackLengths', 'float32', trackCount, usage),
    trackDurations: createTransientView(graph, 'trackDurations', 'float32', trackCount, usage),
    averageSpeeds: createTransientView(graph, 'averageSpeeds', 'float32', trackCount, usage),
    maximumSpeeds: createTransientView(graph, 'maximumSpeeds', 'float32', trackCount, usage),
    trackStopCounts: createTransientView(graph, 'trackStopCounts', 'uint32', trackCount, usage),
    stops: {
      ids: createTransientView(graph, 'stopTracks', 'uint32', stopCapacity, usage),
      status: {
        count: createTransientView(graph, 'stopCount', 'uint32', 1, usage),
        overflow: createTransientView(graph, 'stopOverflow', 'uint32', 1, usage)
      },
      centroids: createTransientView(graph, 'stopCentroids', 'float32x2', stopCapacity, usage),
      durations: createTransientView(graph, 'stopDurations', 'float32', stopCapacity, usage)
    }
  };
  // #region example
  // UTM 18N metres, seconds: a step under 0.5 m/s is slow, a stop lasts at least 300 s.
  graph.add(new GPUTrajectoryMetrics({...tracks, parameters, output}));
  // Draw the same track paths coloured by average speed in m/s.
  const measuredTracks = tracks.positions;
  // #endregion example
  return {...output, measuredTracks};
}
