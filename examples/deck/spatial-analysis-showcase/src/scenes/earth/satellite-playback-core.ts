// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SatelliteTrackSet} from './satellite-tracks';

/** Readback of the playhead graph: who is active, where and how high, and how many trail segments are live. */
export type SatellitePlaybackSnapshot = {
  activeCount: number;
  activeOverflow: number;
  trailCount: number;
  status: Uint32Array;
  /** Longitude/latitude per track. */
  positions: Float32Array;
  /** Altitude in meters per track. */
  elevations: Float32Array;
};

export type SatellitePlaybackCore = {
  tracks: SatelliteTrackSet;
  positionsBuffer: Buffer;
  altitudesBuffer: Buffer;
  segmentEndsBuffer: Buffer;
  segmentTracksBuffer: Buffer;
  /** `uint32` palette row per track, rewritable. */
  colorsBuffer: Buffer;
  /** `uint32` group per track. */
  groupsBuffer: Buffer;
  /** `uint32` satellite index per track. */
  satellitesBuffer: Buffer;
  currentPositions: Buffer;
  currentElevations: Buffer;
  activeIds: Buffer;
  markerDraw: DrawCommandBuffer;
  trailIds: Buffer;
  fadeWeights: Buffer;
  clipFractions: Buffer;
  trailDraw: DrawCommandBuffer;
  playheadCompiled: CompiledGPUCommandGraph<void>;
  trailCompiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  /** Latest snapshot, or null before the first readback. */
  getSnapshot: () => SatellitePlaybackSnapshot | null;
  /** Writes the playhead parameters and encodes the playhead graph. */
  encodePlayhead: (
    commandEncoder: CommandEncoder,
    playhead: number,
    maxGapSeconds?: number
  ) => void;
  /** Writes the window `[end - length, end]` and encodes the trail graph. */
  encodeTrails: (
    commandEncoder: CommandEncoder,
    end: number,
    lengthSeconds: number,
    fadeSeconds: number
  ) => void;
  /** Restricts the trail window to tracks of one group (`null` for all). */
  setGroupFilter: (group: number | null) => void;
  /** Restricts the trail window to the tracks for which `allowed` is true. */
  setTrackFilter: (allowed: (track: number) => boolean) => void;
  /** Stops the readback; call before destroying the resources. */
  stop: () => void;
};

/**
 * Playhead and trail graphs shared by the satellite scenes: `GPUTrajectoryPlayhead` interpolates
 * every track (with the altitude column as its elevations) at the clock, and `GPUTimeWindowFilter`
 * selects the trail segments inside a sliding window. Both only read parameter buffers per frame.
 */
export function createSatellitePlaybackCore(
  resources: SpatialAnalysisResources,
  tracks: SatelliteTrackSet,
  colorRows: Uint32Array,
  onSnapshot: (snapshot: SatellitePlaybackSnapshot) => void
): SatellitePlaybackCore {
  const {device} = resources;
  const {trackCount, vertexCount, segmentCount} = tracks;
  const positionsBuffer = resources.createBuffer('positions', tracks.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', tracks.timestamps);
  const altitudesBuffer = resources.createBuffer('altitudes', tracks.altitudes);
  const offsetsBuffer = resources.createBuffer('track-offsets', tracks.offsets);
  const colorsBuffer = resources.createBuffer('track-colors', colorRows);
  const groupsBuffer = resources.createBuffer('track-groups', tracks.group);
  const satellitesBuffer = resources.createBuffer('track-satellites', tracks.satelliteIndex);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', tracks.segmentEndVertices);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', tracks.segmentTracks);
  const segmentStartTimesBuffer = resources.createBuffer(
    'segment-start-times',
    tracks.segmentStartTimes
  );
  const segmentEndTimesBuffer = resources.createBuffer('segment-end-times', tracks.segmentEndTimes);
  const segmentMaskBuffer = resources.createBuffer(
    'segment-mask',
    new Uint32Array(segmentCount).fill(1)
  );

  // ---- Playhead graph ---------------------------------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const currentElevations = resources.createBuffer('current-elevations', trackCount * 4);
  const headings = resources.createBuffer('headings', trackCount * 4);
  const speeds = resources.createBuffer('speeds', trackCount * 4);
  const status = resources.createBuffer('status', trackCount * 4);
  const activeIds = resources.createBuffer('active-ids', trackCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const markerDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'satellite-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'satellite-playhead'});
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      id: 'playhead',
      positions: importGraphBuffer(
        playheadGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        vertexCount
      ),
      timestamps: importGraphBuffer(
        playheadGraph,
        'timestamps',
        timestampsBuffer,
        'float32',
        vertexCount
      ),
      trackOffsets: importGraphBuffer(
        playheadGraph,
        'offsets',
        offsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      elevations: importGraphBuffer(
        playheadGraph,
        'altitudes',
        altitudesBuffer,
        'float32',
        vertexCount
      ),
      parameters: playheadParameters.importToGraph(playheadGraph),
      currentPositions: importGraphBuffer(
        playheadGraph,
        'current-positions',
        currentPositions,
        'float32x2',
        trackCount
      ),
      currentElevations: importGraphBuffer(
        playheadGraph,
        'current-elevations',
        currentElevations,
        'float32',
        trackCount
      ),
      headings: importGraphBuffer(playheadGraph, 'headings', headings, 'float32', trackCount),
      speeds: importGraphBuffer(playheadGraph, 'speeds', speeds, 'float32', trackCount),
      status: importGraphBuffer(playheadGraph, 'status', status, 'uint32', trackCount),
      activeTracks: {
        ids: importGraphBuffer(playheadGraph, 'active-ids', activeIds, 'uint32', trackCount),
        count: importGraphBuffer(playheadGraph, 'active-count', activeCount, 'uint32', 1),
        overflow: importGraphBuffer(playheadGraph, 'active-overflow', activeOverflow, 'uint32', 1)
      },
      drawInstanceCount: playheadGraph.importGPUData(
        'marker-draw-count',
        markerDraw.getInstanceCountData(0)
      )
    })
  );
  const playheadCompiled = resources.track(playheadGraph.compile());

  // ---- Trail (time window) graph ----------------------------------------------------------------
  const trailIds = resources.createBuffer('trail-ids', segmentCount * 4);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailOverflow = resources.createBuffer('trail-overflow', 4);
  const fadeWeights = resources.createBuffer('fade-weights', segmentCount * 4);
  const clipFractions = resources.createBuffer('clip-fractions', segmentCount * 8);
  const trackVisibleCounts = resources.createBuffer('track-visible-counts', trackCount * 4);
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'satellite-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'satellite-trails'});
  trailGraph.add(
    new GPUTimeWindowFilter({
      id: 'trail-window',
      timestamps: importGraphBuffer(
        trailGraph,
        'segment-start-times',
        segmentStartTimesBuffer,
        'float32',
        segmentCount
      ),
      endTimestamps: importGraphBuffer(
        trailGraph,
        'segment-end-times',
        segmentEndTimesBuffer,
        'float32',
        segmentCount
      ),
      window: windowParameters.importToGraph(trailGraph),
      additionalPredicates: [
        {
          kind: 'selection',
          mask: importGraphBuffer(
            trailGraph,
            'segment-mask',
            segmentMaskBuffer,
            'uint32',
            segmentCount
          )
        }
      ],
      output: {
        ids: importGraphBuffer(trailGraph, 'trail-ids', trailIds, 'uint32', segmentCount),
        count: importGraphBuffer(trailGraph, 'trail-count', trailCount, 'uint32', 1),
        overflow: importGraphBuffer(trailGraph, 'trail-overflow', trailOverflow, 'uint32', 1)
      },
      fadeWeights: importGraphBuffer(
        trailGraph,
        'fade-weights',
        fadeWeights,
        'float32',
        segmentCount
      ),
      clipFractions: importGraphBuffer(
        trailGraph,
        'clip-fractions',
        clipFractions,
        'float32x2',
        segmentCount
      ),
      trackIds: importGraphBuffer(
        trailGraph,
        'segment-tracks',
        segmentTracksBuffer,
        'uint32',
        segmentCount
      ),
      trackVisibleCounts: importGraphBuffer(
        trailGraph,
        'track-visible-counts',
        trackVisibleCounts,
        'uint32',
        trackCount
      ),
      drawInstanceCount: trailGraph.importGPUData(
        'trail-draw-count',
        trailDraw.getInstanceCountData(0)
      )
    })
  );
  const trailCompiled = resources.track(trailGraph.compile());

  // ---- Readback ---------------------------------------------------------------------------------
  let snapshot: SatellitePlaybackSnapshot | null = null;
  let destroyed = false;
  const reader = new SummaryReader(
    resources,
    'satellite-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: activeOverflow, size: 4},
      {buffer: trailCount, size: 4},
      {buffer: status, size: trackCount * 4},
      {buffer: currentPositions, size: trackCount * 8},
      {buffer: currentElevations, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const statusStart = 3;
      const positionStart = statusStart + trackCount;
      const elevationStart = positionStart + trackCount * 2;
      snapshot = {
        activeCount: words[0],
        activeOverflow: words[1],
        trailCount: words[2],
        status: words.slice(statusStart, positionStart),
        positions: floats.slice(positionStart, elevationStart),
        elevations: floats.slice(elevationStart, elevationStart + trackCount)
      };
      onSnapshot(snapshot);
    }
  );

  return {
    tracks,
    positionsBuffer,
    altitudesBuffer,
    segmentEndsBuffer,
    segmentTracksBuffer,
    colorsBuffer,
    groupsBuffer,
    satellitesBuffer,
    currentPositions,
    currentElevations,
    activeIds,
    markerDraw,
    trailIds,
    fadeWeights,
    clipFractions,
    trailDraw,
    playheadCompiled,
    trailCompiled,
    reader,
    getSnapshot: () => snapshot,
    stop() {
      destroyed = true;
      reader.stop();
    },
    encodePlayhead(commandEncoder, playhead, maxGapSeconds = 0) {
      playheadParameters.write(
        getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: maxGapSeconds})
      );
      playheadCompiled.encode(commandEncoder, {parameters: undefined});
    },
    encodeTrails(commandEncoder, end, lengthSeconds, fadeSeconds) {
      windowParameters.write(
        getGPUTimeWindowParameterValues({
          start: end - lengthSeconds,
          end,
          startFadeDuration: fadeSeconds
        })
      );
      trailCompiled.encode(commandEncoder, {parameters: undefined});
    },
    setGroupFilter(group) {
      this.setTrackFilter(track => group === null || tracks.group[track] === group);
    },
    setTrackFilter(allowed) {
      const trackAllowed = new Uint8Array(trackCount);
      for (let track = 0; track < trackCount; track++) trackAllowed[track] = allowed(track) ? 1 : 0;
      const mask = new Uint32Array(segmentCount);
      for (let segment = 0; segment < segmentCount; segment++) {
        mask[segment] = trackAllowed[tracks.segmentTracks[segment]];
      }
      segmentMaskBuffer.write(mask);
    }
  };
}

export {GPU_TRAJECTORY_PLAYHEAD_STATUS};
