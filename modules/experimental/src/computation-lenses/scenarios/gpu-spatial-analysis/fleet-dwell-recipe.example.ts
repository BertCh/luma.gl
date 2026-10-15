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
  addFleetDwellRecipe,
  getGPUTrajectoryMetricsParameterValues,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  GPUParameterBuffer,
  type GPUFleetDwellPolygonZones,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the New York harbor AIS tracks and anchorage zones. */
export type FleetDwellRecipeExampleInput = {
  /** Vessel fixes projected to local-frame metres, sorted by track then time. */
  positions: GraphDataView<'float32x2'>;
  /** Seconds since the start of the recording, one per fix. */
  timestamps: GraphDataView<'float32'>;
  /** Start of each track in `positions`, plus the final end. */
  trackOffsets: GraphDataView<'uint32'>;
  /** Anchorage areas as four-level polygon views in the same frame. */
  anchorages: Omit<GPUFleetDwellPolygonZones, 'candidateCapacity'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** Where vessels stop (under 0.5 m/s for five minutes) and for how long, per anchorage. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {positions, timestamps, trackOffsets, anchorages, spatialContext}: FleetDwellRecipeExampleInput
) {
  const stopCapacity = 160;
  const zoneCount = anchorages.featureOffsets.length - 1;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'metrics-parameters',
    format: 'float32',
    length: GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
    values: getGPUTrajectoryMetricsParameterValues({
      stopSpeedThreshold: 0.5,
      stopMinimumDuration: 300
    })
  });
  const stops = {
    ids: output('stopTracks', 'uint32', stopCapacity),
    centroids: output('stopCentroids', 'float32x2', stopCapacity),
    durations: output('stopDurations', 'float32', stopCapacity),
    status: {count: output('stopCount', 'uint32', 1), overflow: output('stopOverflow', 'uint32', 1)}
  };
  const table = {
    counts: output('zoneStopCounts', 'uint32', zoneCount),
    sumValues: output('zoneDwell', 'float32', zoneCount),
    maximums: output('zoneLongestDwell', 'float32', zoneCount)
  };
  const stopZones = output('stopZones', 'uint32', stopCapacity);
  // #region example
  const dwell = addFleetDwellRecipe(graph, {
    spatialContext,
    trajectories: {kind: 'trajectories', positions, timestamps, trackOffsets},
    parameters: parameters.importToGraph(graph),
    stopCapacity,
    zones: {...anchorages, candidateCapacity: 4096},
    outputs: {stops, table},
    scratch: {stopZones}
  });
  // table.counts and table.sumValues give stops and total dwell seconds per anchorage
  // #endregion example
  return {dwell, stops, table, stopZones, parameters};
}
