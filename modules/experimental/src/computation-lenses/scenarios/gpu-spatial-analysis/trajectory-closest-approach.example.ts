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
  GPUTrajectoryClosestApproach,
  type GPUTrajectoryGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded AIS tracks and distance for GPUTrajectoryClosestApproach. */
export type TrajectoryClosestApproachExampleInput = {
  tracks: GPUTrajectoryGeometryFields;
  /** One row: encounter distance in metres. */
  distance: GraphDataView<'float32'>;
  pairCapacity: number;
};

/** Runs GPUTrajectoryClosestApproach and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: TrajectoryClosestApproachExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {tracks, distance, pairCapacity} = input;
  const output = {
    pairs: {
      ids: createTransientView(graph, 'pairTracks', 'uint32', pairCapacity, usage),
      status: {
        count: createTransientView(graph, 'pairCount', 'uint32', 1, usage),
        overflow: createTransientView(graph, 'pairOverflow', 'uint32', 1, usage)
      },
      partners: createTransientView(graph, 'pairPartners', 'uint32', pairCapacity, usage),
      minimumDistances: createTransientView(
        graph,
        'minimumDistances',
        'float32',
        pairCapacity,
        usage
      ),
      closestTimes: createTransientView(graph, 'closestTimes', 'float32', pairCapacity, usage),
      firstTimes: createTransientView(graph, 'firstTimes', 'float32', pairCapacity, usage),
      durations: createTransientView(graph, 'durations', 'float32', pairCapacity, usage)
    },
    status: {
      entryOverflow: createTransientView(graph, 'entryOverflow', 'uint32', 1, usage),
      hitOverflow: createTransientView(graph, 'hitOverflow', 'uint32', 1, usage)
    }
  };
  // #region example
  // UTM 18N metres, seconds: tracks within 200 m at the same instant, from the raw fixes.
  graph.add(
    new GPUTrajectoryClosestApproach({
      ...tracks,
      distance,
      cellSize: 250,
      timeCellSize: 120,
      entryCapacity: 1 << 17,
      hitCapacity: 1 << 16,
      output
    })
  );
  const {pairs} = output;
  // #endregion example
  return {...output, pairs};
}
