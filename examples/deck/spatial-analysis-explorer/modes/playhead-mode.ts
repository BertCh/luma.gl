// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {
  ResampledTrailLayer,
  VehicleMarkerLayer,
  VEHICLE_MARKER_VERTEX_COUNT
} from './playhead-layers';

/** Samples per resampled track (fixed at compile time: it sizes the dense output buffer). */
const RESAMPLE_SAMPLE_COUNT = 16;
/** Frames between status summary readbacks. */
const READBACK_INTERVAL_FRAMES = 12;
/** Speed in meters per second at which the marker colormap ends. */
const SPEED_FOR_FULL_COLOR = 14;
const VIRIDIS_COLORS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37]
] as const;

function formatClock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  return `${minutes}:${String(total % 60).padStart(2, '0')}`;
}

/**
 * Trip playback. One compiled `GPUTrajectoryPlayhead` interpolates every taxi trip at a per-frame
 * playhead and compacts the trips that are moving into `activeTracks`, whose clamped count is
 * written straight into an indirect draw record. A custom layer draws one arrow per active trip,
 * gathered by trip index from the current position, heading and speed columns, colored by speed.
 * The playhead, the play/pause state, the playback speed and the maximum gap are all written to
 * one parameter buffer or a JavaScript variable each frame, so nothing recompiles. A second graph,
 * `GPUTrajectoryResample`, runs once and resamples every trip to a fixed number of samples
 * (arc-length spacing) that are drawn as faint polylines. Per-status counts come from a small,
 * throttled readback of the status column.
 *
 * Timestamps are float32 seconds since the dataset's first sample.
 */
export const playheadMode: SpatialAnalysisModeDefinition = {
  id: 'playhead',
  title: 'Playhead',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTrajectoryResample'],
  description:
    'Taxi trips played back on the GPU. Each moving trip is one arrow at its interpolated ' +
    'position, rotated to its heading and colored by speed. Play, scrub, change playback speed ' +
    'or the maximum gap; every control is a buffer write. Faint lines show each trip resampled ' +
    'to 16 equal-arc-length points.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.4},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'playhead');

    const trackCount = trips.vendors.length;
    const rowCount = trips.vertexTimestamps.length;
    const epoch = trips.timeRange[0];
    const duration = Math.max(1, trips.timeRange[1] - epoch);
    const rebasedTimestamps = new Float32Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      rebasedTimestamps[row] = trips.vertexTimestamps[row] - epoch;
    }

    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const timestampsBuffer = resources.createBuffer('timestamps', rebasedTimestamps);
    const offsetsBuffer = resources.createBuffer('track-offsets', trips.tripOffsets);
    const currentPositionsBuffer = resources.createBuffer('current-positions', trackCount * 8);
    const headingsBuffer = resources.createBuffer('headings', trackCount * 4);
    const speedsBuffer = resources.createBuffer('speeds', trackCount * 4);
    const statusBuffer = resources.createBuffer('status', trackCount * 4);
    const activeIdsBuffer = resources.createBuffer('active-ids', trackCount * 4);
    const activeCountBuffer = resources.createBuffer('active-count', 4);
    const activeOverflowBuffer = resources.createBuffer('active-overflow', 4);
    const samplesBuffer = resources.createBuffer(
      'resampled-samples',
      trackCount * RESAMPLE_SAMPLE_COUNT * 8
    );
    const parameterBuffer = resources.createParameterBuffer(
      'parameters',
      'float32',
      GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
    );
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'playhead-vehicle-draw',
        type: 'draw',
        commands: [{vertexCount: VEHICLE_MARKER_VERTEX_COUNT, instanceCount: 0}]
      })
    );

    // Summary ring layout: count, overflow, 2 padding words, then the status column.
    const summaryByteLength = 16 + trackCount * 4;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'playhead-summary', byteLength: summaryByteLength})
    );

    const playheadGraph = new GPUCommandGraph<void>(device, {id: 'playhead'});
    const positionsView = importGraphBuffer(
      playheadGraph,
      'positions',
      positionsBuffer,
      'float32x2',
      rowCount
    );
    const timestampsView = importGraphBuffer(
      playheadGraph,
      'timestamps',
      timestampsBuffer,
      'float32',
      rowCount
    );
    const offsetsView = importGraphBuffer(
      playheadGraph,
      'track-offsets',
      offsetsBuffer,
      'uint32',
      trackCount + 1
    );
    playheadGraph.add(
      new GPUTrajectoryPlayhead({
        id: 'playhead',
        positions: positionsView,
        timestamps: timestampsView,
        trackOffsets: offsetsView,
        parameters: parameterBuffer.importToGraph(playheadGraph),
        currentPositions: importGraphBuffer(
          playheadGraph,
          'current-positions',
          currentPositionsBuffer,
          'float32x2',
          trackCount
        ),
        headings: importGraphBuffer(
          playheadGraph,
          'headings',
          headingsBuffer,
          'float32',
          trackCount
        ),
        speeds: importGraphBuffer(playheadGraph, 'speeds', speedsBuffer, 'float32', trackCount),
        status: importGraphBuffer(playheadGraph, 'status', statusBuffer, 'uint32', trackCount),
        activeTracks: {
          ids: importGraphBuffer(
            playheadGraph,
            'active-ids',
            activeIdsBuffer,
            'uint32',
            trackCount
          ),
          count: importGraphBuffer(playheadGraph, 'active-count', activeCountBuffer, 'uint32', 1),
          overflow: importGraphBuffer(
            playheadGraph,
            'active-overflow',
            activeOverflowBuffer,
            'uint32',
            1
          )
        },
        drawInstanceCount: playheadGraph.importGPUData(
          'vehicle-draw-count',
          drawCommands.getInstanceCountData(0)
        )
      })
    );
    const compiledPlayhead: CompiledGPUCommandGraph<void> = resources.track(
      playheadGraph.compile()
    );

    // Static inputs: resample once on the first frame, not every frame.
    const resampleGraph = new GPUCommandGraph<void>(device, {id: 'playhead-resample'});
    resampleGraph.add(
      new GPUTrajectoryResample({
        id: 'resample',
        positions: importGraphBuffer(
          resampleGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          rowCount
        ),
        timestamps: importGraphBuffer(
          resampleGraph,
          'timestamps',
          timestampsBuffer,
          'float32',
          rowCount
        ),
        trackOffsets: importGraphBuffer(
          resampleGraph,
          'track-offsets',
          offsetsBuffer,
          'uint32',
          trackCount + 1
        ),
        sampleCount: RESAMPLE_SAMPLE_COUNT,
        spacing: 'arc-length',
        samples: importGraphBuffer(
          resampleGraph,
          'resampled-samples',
          samplesBuffer,
          'float32x2',
          trackCount * RESAMPLE_SAMPLE_COUNT
        )
      })
    );
    const compiledResample: CompiledGPUCommandGraph<void> = resources.track(
      resampleGraph.compile()
    );

    let playhead = 0;
    let playing = true;
    let playbackSpeed = 60;
    let maximumGap = 0;
    let showTrails = true;
    let resampled = false;
    let readbackPending = false;
    let destroyed = false;
    let lastScrubSecond = -1;
    const parameterValues = new Float32Array(GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH);

    context.controls.addToggle({
      label: 'Playing',
      value: playing,
      onChange: value => {
        playing = value;
      }
    });
    const scrubSlider = context.controls.addSlider({
      label: 'Playhead (per-frame)',
      min: 0,
      max: Math.ceil(duration),
      step: 1,
      value: 0,
      format: value => formatClock(value),
      onChange: value => {
        playhead = value;
        lastScrubSecond = Math.floor(value);
      }
    });
    context.controls.addSlider({
      label: 'Playback speed (per-frame)',
      min: 1,
      max: 600,
      step: 1,
      value: playbackSpeed,
      format: value => `${value}x`,
      onChange: value => {
        playbackSpeed = value;
      }
    });
    context.controls.addSlider({
      label: 'Maximum gap (per-frame, 0 = off)',
      min: 0,
      max: 180,
      step: 1,
      value: maximumGap,
      format: value => (value === 0 ? 'off' : `${value} s`),
      onChange: value => {
        maximumGap = value;
      }
    });
    context.controls.addToggle({
      label: 'Show resampled trips (16 points each)',
      value: showTrails,
      onChange: value => {
        showTrails = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Arrow color: trip speed (slow to fast)',
      gradient: {colors: VIRIDIS_COLORS, minimumLabel: '0', maximumLabel: '14 m/s'}
    });
    context.controls.addNote(
      'Arrows point along the interpolated segment. Raising the maximum gap hides trips whose ' +
        'bracketing samples are further apart than the limit (status gap).'
    );
    context.controls.addReadout('Trips', formatCount(trackCount));
    context.controls.addReadout(
      'Resampled output',
      `${formatCount(trackCount)} x ${RESAMPLE_SAMPLE_COUNT} (one-time)`
    );
    const clockReadout = context.controls.addReadout('Playhead');
    const activeReadout = context.controls.addReadout('Active (drawn)');
    const beforeReadout = context.controls.addReadout('Before start');
    const afterReadout = context.controls.addReadout('After end');
    const gapReadout = context.controls.addReadout('In gap');
    const overflowReadout = context.controls.addReadout('Active list overflow');
    context.controls.addReadout('Data', trips.attribution);

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: activeCountBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: activeOverflowBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 4,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: statusBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 16,
        size: trackCount * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: summaryByteLength});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        let before = 0;
        let after = 0;
        let gap = 0;
        for (let track = 0; track < trackCount; track++) {
          const status = words[4 + track];
          if (status === GPU_TRAJECTORY_PLAYHEAD_STATUS.beforeStart) before++;
          else if (status === GPU_TRAJECTORY_PLAYHEAD_STATUS.afterEnd) after++;
          else if (status === GPU_TRAJECTORY_PLAYHEAD_STATUS.gap) gap++;
        }
        activeReadout.setValue(formatCount(words[0]));
        beforeReadout.setValue(formatCount(before));
        afterReadout.setValue(formatCount(after));
        gapReadout.setValue(formatCount(gap));
        overflowReadout.setValue(words[1] ? 'yes' : 'no');
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiledPlayhead, compiledResample],
      encode(commandEncoder, frame) {
        if (playing) {
          playhead += frame.deltaSeconds * playbackSpeed;
          if (playhead > duration) playhead -= duration;
          const second = Math.floor(playhead);
          if (second !== lastScrubSecond) {
            lastScrubSecond = second;
            scrubSlider.setValue(second);
          }
        }
        clockReadout.setValue(`${formatClock(playhead)} of ${formatClock(duration)}`);
        parameterBuffer.write(
          getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: maximumGap}, parameterValues)
        );
        if (!resampled) {
          compiledResample.encode(commandEncoder, {parameters: undefined});
          resampled = true;
        }
        compiledPlayhead.encode(commandEncoder, {parameters: undefined});
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) {
          void readSummary(commandEncoder);
        }
      },
      getLayers(): Layer[] {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        return [
          new ResampledTrailLayer({
            id: 'playhead-resampled',
            coordinateOrigin,
            samples: samplesBuffer,
            trackCount,
            sampleCount: RESAMPLE_SAMPLE_COUNT,
            color: [160, 190, 255, 55],
            widthPixels: 1,
            visible: showTrails
          }),
          new VehicleMarkerLayer({
            id: 'playhead-vehicles',
            coordinateOrigin,
            ids: activeIdsBuffer,
            positions: currentPositionsBuffer,
            headings: headingsBuffer,
            speeds: speedsBuffer,
            drawCommands,
            speedForFullColor: SPEED_FOR_FULL_COLOR
          })
        ];
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
