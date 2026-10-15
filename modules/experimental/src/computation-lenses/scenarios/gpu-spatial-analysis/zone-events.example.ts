// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUZoneEvents,
  type GPUPolygonGeometryFields,
  type GPUTrajectoryGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded AIS tracks and harbour zones for GPUZoneEvents. */
export type ZoneEventsExampleInput = {
  tracks: GPUTrajectoryGeometryFields;
  zones: GPUPolygonGeometryFields;
  trackEventCapacity: number;
};

/** Runs GPUZoneEvents and returns the views drawn or inspected by its card. */
export async function runExample(graph: GPUCommandGraph<undefined>, input: ZoneEventsExampleInput) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {tracks, zones, trackEventCapacity} = input;
  const trackCount = tracks.trackOffsets.length - 1;
  const zoneCount = zones.featureOffsets.length - 1;
  const eventCapacity = trackCount * trackEventCapacity;
  const output = {
    events: {
      ids: createTransientView(graph, 'eventTracks', 'uint32', eventCapacity, usage),
      status: {
        count: createTransientView(graph, 'eventCount', 'uint32', 1, usage),
        overflow: createTransientView(graph, 'eventOverflow', 'uint32', 1, usage)
      },
      eventZones: createTransientView(graph, 'eventZones', 'uint32', eventCapacity, usage),
      eventTypes: createTransientView(graph, 'eventTypes', 'uint32', eventCapacity, usage),
      eventTimes: createTransientView(graph, 'eventTimes', 'float32', eventCapacity, usage)
    },
    dwellTimes: createTransientView(graph, 'dwellTimes', 'float32', trackCount * zoneCount, usage),
    visitCounts: createTransientView(graph, 'visitCounts', 'uint32', trackCount * zoneCount, usage),
    trackEventCounts: createTransientView(graph, 'trackEventCounts', 'uint32', trackCount, usage)
  };
  // #region example
  // Planar UTM 18N metres, seconds: tracks crossing 13 anchorage polygons.
  graph.add(
    new GPUZoneEvents({
      tracks: {kind: 'trajectories', ...tracks},
      zones: {kind: 'polygons', ...zones},
      candidateCapacity: 1 << 17,
      trackEventCapacity,
      output
    })
  );
  // Draw the track paths coloured by their number of zone entries and exits.
  const {trackEventCounts} = output;
  // #endregion example
  return {...output, trackEventCounts};
}
