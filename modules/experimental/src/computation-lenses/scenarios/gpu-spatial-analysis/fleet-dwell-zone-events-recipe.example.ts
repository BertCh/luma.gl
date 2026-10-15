// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  addFleetDwellZoneEventsRecipe,
  type GPUPolygonGeometryFields,
  type GPUTrajectoryGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded AIS tracks and harbour zones for the zone events fleet dwell recipe. */
export type FleetDwellZoneEventsRecipeExampleInput = {
  tracks: GPUTrajectoryGeometryFields;
  zones: GPUPolygonGeometryFields;
  trackEventCapacity: number;
};

/** Runs the recipe and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: FleetDwellZoneEventsRecipeExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {tracks, zones, trackEventCapacity} = input;
  const zoneCount = zones.featureOffsets.length - 1;
  const zoneVisits = createTransientView(graph, 'zoneVisits', 'uint32', zoneCount, usage);
  const zoneDwell = createTransientView(graph, 'zoneDwell', 'float32', zoneCount, usage);
  // #region example
  // Planar UTM 18N metres, seconds: dwell inside each of 13 anchorage polygons.
  const dwell = addFleetDwellZoneEventsRecipe(graph, {
    trajectories: {kind: 'trajectories', ...tracks},
    zones: {kind: 'polygons', ...zones},
    candidateCapacity: 1 << 17,
    trackEventCapacity,
    outputs: {table: {counts: zoneVisits, sumValues: zoneDwell}}
  });
  // Draw the anchorages coloured by total dwell seconds; zoneVisits counts visiting tracks.
  const {sumValues} = dwell.table;
  // #endregion example
  return {dwell, zoneVisits, zoneDwell: sumValues};
}
