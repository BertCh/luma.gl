// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock, formatPlaybackTime} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

/** Option state of the nyc-taxi-trails scene. */
export type NycTaxiTrailsOptions = {
  play: boolean;
  time: number;
  playSpeed: number;
  loop: boolean;
  trailMinutes: number;
  tailFade: number;
  trailWidth: number;
  colorBy: 'fare' | 'distance' | 'none';
  hideBelow: number;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  markerSize: number;
  showTrails: boolean;
  showBackdrop: boolean;
};

/** Value ranges of the colour attributes (USD and miles). */
export const TRAIL_COLOR_RANGES = {fare: [4, 40], distance: [0.5, 10]} as const;
const STATUS_INTERVAL_FRAMES = 10;

/**
 * Routed taxi trails over a 30-minute Friday morning window. A `GPUTrajectoryPlayhead` graph
 * interpolates every taxi at the clock, a `GPUTimeWindowFilter` graph selects the route segments of
 * the last few minutes with fade weights and clip fractions, and both are drawn straight from GPU
 * storage with indirect counts. The clock, the trail length and the fade are parameter buffers.
 */
export async function createNycTaxiTrails(
  ctx: SceneContext<NycTaxiTrailsOptions>
): Promise<SceneInstance<NycTaxiTrailsOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get('poopdeck-nyc-taxi-paths');
  const origin = dataset.defaultOrigin;
  const offsets = dataset.column<Uint32Array>('pathOffsets');
  const timestamps = Float32Array.from(dataset.column<Uint32Array>('timestamp'));
  const positions = dataset.projectColumn('vertices', origin);
  const fares = dataset.column<Float32Array>('fare_amount');
  const distances = dataset.column<Float32Array>('trip_distance');
  const startTimes = dataset.column<Uint32Array>('startTime');
  const endTimes = dataset.column<Uint32Array>('endTime');
  const trackCount = offsets.length - 1;
  const vertexCount = timestamps.length;
  const timeOriginSeconds =
    (dataset.manifest.properties as {timeOriginMs: number}).timeOriginMs / 1000;
  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
  const resources = new SpatialAnalysisResources(device, 'taxi-trails');

  // ---- Segment tables ----------------------------------------------------------------------------
  const segmentCount = vertexCount - trackCount;
  const segments = new Float32Array(segmentCount * 4);
  const segmentTracks = new Uint32Array(segmentCount);
  const segmentStartTimes = new Float32Array(segmentCount);
  const segmentEndTimes = new Float32Array(segmentCount);
  let timeEnd = 0;
  {
    let row = 0;
    for (let track = 0; track < trackCount; track++) {
      const first = offsets[track];
      const last = offsets[track + 1] - 1;
      for (let vertex = first; vertex < last; vertex++, row++) {
        segments.set(positions.subarray(vertex * 2, vertex * 2 + 4), row * 4);
        segmentTracks[row] = track;
        segmentStartTimes[row] = timestamps[vertex];
        segmentEndTimes[row] = timestamps[vertex + 1];
      }
      timeEnd = Math.max(timeEnd, timestamps[last]);
    }
  }

  const positionsBuffer = resources.createBuffer('positions', positions);
  const timestampsBuffer = resources.createBuffer('timestamps', timestamps);
  const offsetsBuffer = resources.createBuffer('track-offsets', offsets);
  const segmentsBuffer = resources.createBuffer('segments', segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', segmentTracks);
  const segmentStartTimesBuffer = resources.createBuffer('segment-start-times', segmentStartTimes);
  const segmentEndTimesBuffer = resources.createBuffer('segment-end-times', segmentEndTimes);
  const fareBuffer = resources.createBuffer('fare', fares);
  const distanceBuffer = resources.createBuffer('distance', distances);

  // ---- Playhead graph: every taxi at the clock -----------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
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
      id: 'taxi-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'taxi-playhead'});
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
      parameters: playheadParameters.importToGraph(playheadGraph),
      currentPositions: importGraphBuffer(
        playheadGraph,
        'current-positions',
        currentPositions,
        'float32x2',
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

  // ---- Trail graph: route segments of the last few minutes -----------------------------------------
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
      id: 'taxi-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'taxi-trails'});
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

  // ---- Context chart: taxis on the road per half minute (CPU, from the trip start and end times) ----
  const BIN_SECONDS = 30;
  const binCount = Math.ceil(timeEnd / BIN_SECONDS) + 1;
  const onRoad = new Float64Array(binCount);
  for (let track = 0; track < trackCount; track++) {
    const firstBin = Math.floor(startTimes[track] / BIN_SECONDS);
    const lastBin = Math.min(binCount - 1, Math.floor(endTimes[track] / BIN_SECONDS));
    for (let bin = firstBin; bin <= lastBin; bin++) onRoad[bin]++;
  }
  const binTimes = Array.from(onRoad, (_, index) => index * BIN_SECONDS);

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, timeEnd], rate: 1, step: 5}
  );

  let destroyed = false;
  let lastChartStep = -1;
  const statusReader = new SummaryReader(
    resources,
    'taxi-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: activeOverflow, size: 4},
      {buffer: trailCount, size: 4},
      {buffer: trailOverflow, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const [active, activeOver, trail, trailOver] = new Uint32Array(bytes);
      ctx.setReadout('active', active);
      ctx.setReadout('segments', trail);
      ctx.setReadout(
        'overflow',
        activeOver || trailOver ? 'yes: some taxis or segments are not drawn' : 'no'
      );
    }
  );

  function formatClock(seconds: number): string {
    const total = Math.round(timeOriginSeconds + seconds);
    const hours = Math.floor((total % 86400) / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function updateChart(playhead: number): void {
    const step = Math.floor(playhead / 10);
    if (step === lastChartStep) return;
    lastChartStep = step;
    ctx.setChart('roadChart', {
      kind: 'line',
      series: [{label: 'taxis on the road', x: binTimes, y: Array.from(onRoad), area: true}],
      xLabel: 'Seconds after 07:59',
      yLabel: 'trips active',
      height: 110,
      formatX: value => formatClock(value).slice(0, 5),
      markers: [{x: playhead, label: formatClock(playhead).slice(0, 5)}],
      description: 'Trips on the road per half minute, with the clock marked.'
    });
  }

  ctx.setReadout('trips', trackCount);
  ctx.setReadout('vertices', vertexCount);
  ctx.setReadout('segmentsTotal', segmentCount);
  updateChart(ctx.options.time);

  return {
    getCompiledGraphs: () => [playheadCompiled, trailCompiled] as never[],

    setOption(id) {
      if (id === 'time' || id === 'play' || id === 'playSpeed' || id === 'loop') return;
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      const playhead = clock.advance(frame);
      ctx.setReadout(
        'clock',
        `${formatClock(playhead)} (${formatPlaybackTime.duration(playhead)} in)`
      );
      updateChart(playhead);
      playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead}));
      playheadCompiled.encode(commandEncoder, {parameters: undefined});
      if (options.showTrails) {
        const trailSeconds = options.trailMinutes * 60;
        windowParameters.write(
          getGPUTimeWindowParameterValues({
            start: playhead - trailSeconds,
            end: playhead,
            startFadeDuration: trailSeconds * options.tailFade
          })
        );
        trailCompiled.encode(commandEncoder, {parameters: undefined});
      }
      if (frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) statusReader.markStale();
      statusReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const colored = options.colorBy !== 'none';
      const range =
        options.colorBy === 'distance' ? TRAIL_COLOR_RANGES.distance : TRAIL_COLOR_RANGES.fare;
      const style = colored
        ? {
            values: options.colorBy === 'distance' ? distanceBuffer : fareBuffer,
            valueFormat: 'float32' as const,
            colormap: options.ramp,
            valueRange: range as unknown as [number, number],
            ...(options.hideBelow > 0 ? {discardAtOrBelow: options.hideBelow} : {})
          }
        : {
            color: (dark ? [255, 214, 120, 255] : [200, 90, 10, 255]) as [
              number,
              number,
              number,
              number
            ]
          };
      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-backdrop',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            color: dark ? [190, 200, 220, 18] : [60, 70, 90, 22]
          })
        );
      }
      if (options.showTrails) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-trails',
            coordinateOrigin,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            valueIndices: segmentTracksBuffer,
            widthPixels: options.trailWidth,
            ...style
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'taxi-heads',
          coordinateOrigin,
          positions: currentPositions,
          ids: activeIds,
          drawCommands: markerDraw,
          radiusPixels: options.markerSize,
          opacity: 0.95,
          ...(colored
            ? {...style, color: [255, 255, 255, 255] as [number, number, number, number]}
            : {color: dark ? [255, 255, 255, 255] : [30, 30, 30, 255]})
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      resources.destroy();
    }
  };
}
