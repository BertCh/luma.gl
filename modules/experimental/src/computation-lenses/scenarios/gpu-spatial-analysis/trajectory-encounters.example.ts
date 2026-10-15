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
  GPUTrajectoryEncounters,
  GPUTrajectoryResample,
  type GPUTrajectoryGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded AIS tracks, shared clock and distance for GPUTrajectoryEncounters. */
export type TrajectoryEncountersExampleInput = {
  tracks: GPUTrajectoryGeometryFields;
  /** `[start, step]` of the shared clock in seconds. */
  clock: GraphDataView<'float32'>;
  /** One row: encounter distance in metres. */
  distance: GraphDataView<'float32'>;
  bucketCount: number;
  bounds: [number, number, number, number];
  pairCapacity: number;
};

/** Runs GPUTrajectoryEncounters and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: TrajectoryEncountersExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {tracks, clock, distance, bucketCount, bounds, pairCapacity} = input;
  const trackCount = tracks.trackOffsets.length - 1;
  const samples = createTransientView(
    graph,
    'clockSamples',
    'float32x2',
    trackCount * bucketCount,
    usage
  );
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
      firstBuckets: createTransientView(graph, 'firstBuckets', 'uint32', pairCapacity, usage),
      bucketCounts: createTransientView(graph, 'bucketCounts', 'uint32', pairCapacity, usage)
    },
    status: {candidateOverflow: createTransientView(graph, 'candidateOverflow', 'uint32', 1, usage)}
  };
  // #region example
  // UTM 18N metres: resample every track to a 60 s clock, then find pairs within 200 m.
  graph.add(
    new GPUTrajectoryResample({
      ...tracks,
      sampleCount: bucketCount,
      spacing: 'clock',
      clock,
      output: {samples}
    })
  );
  graph.add(
    new GPUTrajectoryEncounters({
      samples,
      trackCount,
      bucketCount,
      distance,
      cellSize: 250,
      bounds,
      hitCapacity: 1 << 16,
      output
    })
  );
  const {pairs} = output;
  // #endregion example
  return {...output, samples, pairs};
}
