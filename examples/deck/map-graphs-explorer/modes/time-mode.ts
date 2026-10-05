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
  getGPUTimeWindowParameterValues,
  GPUTimeWindowFilter,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  importGraphBuffer
} from '@luma.gl/experimental/map-graphs';
import {MapGraphsSegmentLayer} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';

/** Frames between scrubber updates and summary readbacks. */
const SCRUBBER_INTERVAL_FRAMES = 10;
const READBACK_INTERVAL_FRAMES = 15;
/** deck.gl trips colors for vendor 0 and 1. */
const VENDOR_PALETTE = [
  [253, 128, 93],
  [23, 184, 190]
] as const;

/**
 * Animated trip trails: every consecutive vertex pair of every trip is one timed segment row. A
 * single compiled `GPUTimeWindowFilter` (interval mode) selects the segments inside a moving
 * window, writes fade weights and trail clip fractions, compacts the ids, and writes the visible
 * count straight into an indirect draw record. The window is the only per-frame input and nothing
 * is read back to draw.
 */
export const timeMode: MapGraphsModeDefinition = {
  id: 'time',
  title: 'Time',
  recipes: ['GPUTimeWindowFilter'],
  description:
    'Animated taxi trails: a GPU time window selects the trip segments that are live, fades them ' +
    'toward the tail and clips the leading segment. The window moves through a parameter buffer; ' +
    'the visible count feeds an indirect draw without CPU readback.',
  initialViewState: {longitude: -73.99, latitude: 40.73, zoom: 12.8},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new MapGraphsResources(device, 'time');

    // One row per consecutive vertex pair, built once on the CPU.
    const tripCount = trips.vendors.length;
    let segmentCount = 0;
    for (let trip = 0; trip < tripCount; trip++) {
      segmentCount += Math.max(0, trips.tripOffsets[trip + 1] - trips.tripOffsets[trip] - 1);
    }
    const segments = new Float32Array(segmentCount * 4);
    const startTimes = new Float32Array(segmentCount);
    const endTimes = new Float32Array(segmentCount);
    const trackIds = new Uint32Array(segmentCount);
    const vendors = new Uint32Array(segmentCount);
    let row = 0;
    for (let trip = 0; trip < tripCount; trip++) {
      const first = trips.tripOffsets[trip];
      const last = trips.tripOffsets[trip + 1] - 1;
      for (let vertex = first; vertex < last; vertex++, row++) {
        segments.set(trips.vertexPositions.subarray(vertex * 2, vertex * 2 + 4), row * 4);
        startTimes[row] = trips.vertexTimestamps[vertex];
        endTimes[row] = trips.vertexTimestamps[vertex + 1];
        trackIds[row] = trip;
        vendors[row] = trips.vendors[trip];
      }
    }

    const segmentsBuffer = resources.createBuffer('segments', segments);
    const startTimesBuffer = resources.createBuffer('start-times', startTimes);
    const endTimesBuffer = resources.createBuffer('end-times', endTimes);
    const trackIdsBuffer = resources.createBuffer('track-ids', trackIds);
    const vendorsBuffer = resources.createBuffer('vendors', vendors);
    const idsBuffer = resources.createBuffer('ids', segmentCount * 4);
    const countBuffer = resources.createBuffer('count', 4);
    const overflowBuffer = resources.createBuffer('overflow', 4);
    const weightsBuffer = resources.createBuffer('fade-weights', segmentCount * 4);
    const clipBuffer = resources.createBuffer('clip-fractions', segmentCount * 8);
    const trackCountsBuffer = resources.createBuffer('track-visible-counts', tripCount * 4);
    const windowBuffer = resources.createParameterBuffer(
      'window',
      'float32',
      GPU_TIME_WINDOW_PARAMETER_LENGTH
    );
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'time-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'time-summary', byteLength: 8 + tripCount * 4})
    );

    const graph = new GPUCommandGraph<void>(device, {id: 'time'});
    graph.add(
      new GPUTimeWindowFilter({
        id: 'time-window',
        timestamps: importGraphBuffer(
          graph,
          'start-times',
          startTimesBuffer,
          'float32',
          segmentCount
        ),
        endTimestamps: importGraphBuffer(
          graph,
          'end-times',
          endTimesBuffer,
          'float32',
          segmentCount
        ),
        window: windowBuffer.importToGraph(graph),
        output: {
          ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', segmentCount),
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
        },
        fadeWeights: importGraphBuffer(
          graph,
          'fade-weights',
          weightsBuffer,
          'float32',
          segmentCount
        ),
        clipFractions: importGraphBuffer(
          graph,
          'clip-fractions',
          clipBuffer,
          'float32x2',
          segmentCount
        ),
        trackIds: importGraphBuffer(graph, 'track-ids', trackIdsBuffer, 'uint32', segmentCount),
        trackVisibleCounts: importGraphBuffer(
          graph,
          'track-counts',
          trackCountsBuffer,
          'uint32',
          tripCount
        ),
        drawInstanceCount: graph.importGPUData(
          'draw-instance-count',
          drawCommands.getInstanceCountData(0)
        )
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    const [timeStart, timeEnd] = trips.timeRange;
    let trailSeconds = 180;
    let speed = 60;
    let playing = true;
    let currentTime = timeStart + trailSeconds + (timeEnd - timeStart) * 0.25;
    let readbackPending = false;
    let destroyed = false;

    const loopStart = () => timeStart;
    const loopEnd = () => timeEnd + trailSeconds;
    const formatClock = (seconds: number) => {
      const whole = Math.max(0, Math.floor(seconds - timeStart));
      return `${String(Math.floor(whole / 60)).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`;
    };

    const playButton = context.controls.addButton({
      label: 'Pause',
      onClick: () => {
        playing = !playing;
        playButton.setValue(playing ? 'Pause' : 'Play');
      }
    });
    context.controls.addSlider({
      label: 'Speed (simulated seconds per second)',
      min: 10,
      max: 300,
      step: 10,
      value: speed,
      format: value => `${value}×`,
      onChange: value => {
        speed = value;
      }
    });
    context.controls.addSlider({
      label: 'Trail length (per-frame window)',
      min: 20,
      max: 600,
      step: 10,
      value: trailSeconds,
      format: value => `${value} s`,
      onChange: value => {
        trailSeconds = value;
        scrubber.setValue(currentTime);
      }
    });
    const scrubber = context.controls.addSlider({
      label: 'Time',
      min: Math.floor(timeStart),
      max: Math.ceil(timeEnd + 600),
      step: 1,
      value: currentTime,
      format: formatClock,
      onChange: value => {
        currentTime = value;
      }
    });
    context.controls.addLegend({
      title: 'Vendor (trail fades toward its tail)',
      entries: [
        {color: VENDOR_PALETTE[0], label: 'Vendor 0'},
        {color: VENDOR_PALETTE[1], label: 'Vendor 1'}
      ]
    });
    const clockReadout = context.controls.addReadout('Simulated time');
    const windowReadout = context.controls.addReadout('Window');
    const visibleReadout = context.controls.addReadout('Visible segments');
    const tripsReadout = context.controls.addReadout('Active trips');
    context.controls.addReadout('Segments in dataset', formatCount(segmentCount));
    context.controls.addReadout('Data', trips.attribution);

    const readSummary = async (commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: countBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: overflowBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 4,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: trackCountsBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 8,
        size: tripCount * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: 8 + tripCount * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        let activeTrips = 0;
        for (let trip = 0; trip < tripCount; trip++) {
          if (words[2 + trip] > 0) activeTrips++;
        }
        visibleReadout.setValue(`${formatCount(words[0])}${words[1] ? ' (overflow)' : ''}`);
        tripsReadout.setValue(formatCount(activeTrips));
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        if (playing) {
          currentTime += frame.deltaSeconds * speed;
        }
        const span = loopEnd() - loopStart();
        if (currentTime > loopEnd() || currentTime < loopStart()) {
          currentTime = loopStart() + ((((currentTime - loopStart()) % span) + span) % span);
        }
        windowBuffer.write(
          getGPUTimeWindowParameterValues({
            start: currentTime - trailSeconds,
            end: currentTime,
            startFadeDuration: trailSeconds
          })
        );
        compiled.encode(commandEncoder, {parameters: undefined});
        if (frame.frameIndex % SCRUBBER_INTERVAL_FRAMES === 0) {
          scrubber.setValue(currentTime);
          clockReadout.setValue(formatClock(currentTime));
          windowReadout.setValue(
            `${formatClock(currentTime - trailSeconds)} – ${formatClock(currentTime)}`
          );
        }
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0 && !readbackPending) {
          void readSummary(commandEncoder);
        }
      },
      getLayers(): Layer[] {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        return [
          new MapGraphsSegmentLayer({
            id: 'time-backdrop',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            color: [150, 165, 190, 22]
          }),
          new MapGraphsSegmentLayer({
            id: 'time-trails',
            coordinateOrigin,
            segments: segmentsBuffer,
            ids: idsBuffer,
            drawCommands,
            weights: weightsBuffer,
            clipFractions: clipBuffer,
            values: vendorsBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [...VENDOR_PALETTE[0], 255],
              [...VENDOR_PALETTE[1], 255]
            ],
            widthPixels: 2.5
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
