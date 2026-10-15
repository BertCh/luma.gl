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
  GPUTrajectoryPlayhead,
  type GPUTrajectoryGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded AIS tracks and the playhead parameters for GPUTrajectoryPlayhead. */
export type TrajectoryPlayheadExampleInput = {
  tracks: GPUTrajectoryGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** Runs GPUTrajectoryPlayhead and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: TrajectoryPlayheadExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {tracks, parameters} = input;
  const trackCount = tracks.trackOffsets.length - 1;
  const output = {
    currentPositions: createTransientView(
      graph,
      'currentPositions',
      'float32x2',
      trackCount,
      usage
    ),
    headings: createTransientView(graph, 'headings', 'float32', trackCount, usage),
    speeds: createTransientView(graph, 'speeds', 'float32', trackCount, usage),
    trackStatus: createTransientView(graph, 'trackStatus', 'uint32', trackCount, usage),
    segmentRows: createTransientView(graph, 'segmentRows', 'uint32', trackCount, usage),
    segmentFractions: createTransientView(graph, 'segmentFractions', 'float32', trackCount, usage),
    activeTracks: {
      ids: createTransientView(graph, 'activeTrackIds', 'uint32', trackCount, usage),
      status: {
        count: createTransientView(graph, 'activeTrackCount', 'uint32', 1, usage),
        overflow: createTransientView(graph, 'activeTrackOverflow', 'uint32', 1, usage)
      }
    }
  };
  // #region example
  // UTM 18N metres, seconds: playhead at 3600 s (15:00Z); a longer interval than 300 s is a gap.
  graph.add(new GPUTrajectoryPlayhead({...tracks, parameters, output}));
  // Draw each track's position at the playhead, coloured by its status.
  const {currentPositions, trackStatus} = output;
  // #endregion example
  return {...output, currentPositions, trackStatus};
}
