// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUTrajectoryMetricsParameterValues,
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryMetrics,
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {createPlaybackClock} from '../../engine/playback';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {binValues, histogramChart, lineChart} from './f-chart-helpers';
import {StopMarkerLayer, VesselMarkerLayer} from './b12-layers';
import {
  formatDuration,
  formatEasternClock,
  formatUtcClock,
  KNOTS_PER_METER_SECOND,
  loadVesselTracks,
  METERS_PER_KNOT_SECOND,
  VESSEL_CATEGORIES,
  VESSEL_CATEGORY_COLORS,
  VESSEL_CATEGORY_LABELS
} from './b12-tracks';

/** Option state of the harbor playback scene. */
export type HarborPlaybackOptions = {
  playing: boolean;
  time: number;
  playbackSpeed: number;
  loop: boolean;
  maxGapMinutes: number;
  vesselFilter: string;
  markerColor: 'category' | 'speed';
  markerSize: number;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  showBackdrop: boolean;
  showTrails: boolean;
  trailMinutes: number;
  tailFade: number;
  trailColor: 'category' | 'speed';
  showRoutes: boolean;
  routeSamples: '16' | '32' | '64';
  routeSpacing: 'arc-length' | 'time';
  showStops: boolean;
  stopSpeedKnots: number;
  stopMinutes: number;
};

const STOP_CAPACITY = 4096;
/** Frames between status readbacks (a readback is also requested when a threshold changes). */
const STATUS_INTERVAL_FRAMES = 12;
const SECONDS_PER_DAY = 86400;
/** Speed in knots at which speed ramps end. */
const SPEED_RAMP_KNOTS = 25;
const NO_TRACK = 0xffffffff;
const NAUTICAL_MILE_METERS = 1852;

type ResampleGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  samples: Buffer;
  routeSegments: Buffer;
  sampleCount: number;
  encoded: boolean;
};

/**
 * Harbor playback: one playhead graph interpolates every AIS track at the clock, a time-window
 * graph selects the trail segments that are live, a metrics graph finds stops and per-step speeds
 * once per threshold change, and a resample graph rebuilds each track as evenly spaced samples.
 * All of them take their per-frame inputs as parameter buffers; the only compile-time choices are
 * the number and spacing of resampled route points.
 */
export async function createHarborPlayback(
  ctx: SceneContext<HarborPlaybackOptions>
): Promise<SceneInstance<HarborPlaybackOptions>> {
  const vessels = loadVesselTracks(ctx.datasets.get('ais-vessels'));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = vessels;
  const resources = new SpatialAnalysisResources(device, 'harbor');
  const coordinateOrigin: [number, number, number] = [vessels.origin[0], vessels.origin[1], 0];

  // ---- Static inputs ----------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', vessels.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', vessels.timestamps);
  const offsetsBuffer = resources.createBuffer('track-offsets', vessels.offsets);
  const categoryBuffer = resources.createBuffer('category', vessels.category);
  const segmentsBuffer = resources.createBuffer('segments', vessels.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', vessels.segmentTracks);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', vessels.segmentEndVertices);
  const segmentStartTimesBuffer = resources.createBuffer(
    'segment-start-times',
    vessels.segmentStartTimes
  );
  const segmentEndTimesBuffer = resources.createBuffer(
    'segment-end-times',
    vessels.segmentEndTimes
  );
  const segmentMaskBuffer = resources.createBuffer(
    'segment-mask',
    new Uint32Array(segmentCount).fill(1)
  );

  // ---- Playhead graph ---------------------------------------------------------------------------
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
      id: 'harbor-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'harbor-playhead'});
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
      id: 'harbor-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'harbor-trails'});
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

  // ---- Metrics graph (stops, per-track and per-step columns) -----------------------------------
  const trackLengths = resources.createBuffer('track-lengths', trackCount * 4);
  const trackDurations = resources.createBuffer('track-durations', trackCount * 4);
  const averageSpeeds = resources.createBuffer('average-speeds', trackCount * 4);
  const maximumSpeeds = resources.createBuffer('maximum-speeds', trackCount * 4);
  const stepSpeeds = resources.createBuffer('step-speeds', vertexCount * 4);
  const trackStopCounts = resources.createBuffer('track-stop-counts', trackCount * 4);
  const stopIds = resources.createBuffer('stop-ids', STOP_CAPACITY * 4);
  const stopCount = resources.createBuffer('stop-count', 4);
  const stopOverflow = resources.createBuffer('stop-overflow', 4);
  const stopTotal = resources.createBuffer('stop-total', 4);
  const stopCentroids = resources.createBuffer('stop-centroids', STOP_CAPACITY * 8);
  const stopDurations = resources.createBuffer('stop-durations', STOP_CAPACITY * 4);
  const metricsParameters = resources.createParameterBuffer(
    'metrics',
    'float32',
    GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
  );
  const stopDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'harbor-stop-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const metricsGraph = new GPUCommandGraph<void>(device, {id: 'harbor-metrics'});
  metricsGraph.add(
    new GPUTrajectoryMetrics({
      id: 'metrics',
      positions: importGraphBuffer(
        metricsGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        vertexCount
      ),
      timestamps: importGraphBuffer(
        metricsGraph,
        'timestamps',
        timestampsBuffer,
        'float32',
        vertexCount
      ),
      trackOffsets: importGraphBuffer(
        metricsGraph,
        'offsets',
        offsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      parameters: metricsParameters.importToGraph(metricsGraph),
      trackLengths: importGraphBuffer(
        metricsGraph,
        'track-lengths',
        trackLengths,
        'float32',
        trackCount
      ),
      trackDurations: importGraphBuffer(
        metricsGraph,
        'track-durations',
        trackDurations,
        'float32',
        trackCount
      ),
      averageSpeeds: importGraphBuffer(
        metricsGraph,
        'average-speeds',
        averageSpeeds,
        'float32',
        trackCount
      ),
      maximumSpeeds: importGraphBuffer(
        metricsGraph,
        'maximum-speeds',
        maximumSpeeds,
        'float32',
        trackCount
      ),
      stepSpeeds: importGraphBuffer(
        metricsGraph,
        'step-speeds',
        stepSpeeds,
        'float32',
        vertexCount
      ),
      trackStopCounts: importGraphBuffer(
        metricsGraph,
        'track-stop-counts',
        trackStopCounts,
        'uint32',
        trackCount
      ),
      stops: {
        output: {
          ids: importGraphBuffer(metricsGraph, 'stop-ids', stopIds, 'uint32', STOP_CAPACITY),
          count: importGraphBuffer(metricsGraph, 'stop-count', stopCount, 'uint32', 1),
          overflow: importGraphBuffer(metricsGraph, 'stop-overflow', stopOverflow, 'uint32', 1),
          totalCount: importGraphBuffer(metricsGraph, 'stop-total', stopTotal, 'uint32', 1)
        },
        drawInstanceCount: metricsGraph.importGPUData(
          'stop-draw-count',
          stopDraw.getInstanceCountData(0)
        ),
        centroids: importGraphBuffer(
          metricsGraph,
          'stop-centroids',
          stopCentroids,
          'float32x2',
          STOP_CAPACITY
        ),
        durations: importGraphBuffer(
          metricsGraph,
          'stop-durations',
          stopDurations,
          'float32',
          STOP_CAPACITY
        )
      }
    })
  );
  const metricsCompiled = resources.track(metricsGraph.compile());

  // ---- Resample graph (compile-time sample count and spacing) -----------------------------------
  let resample: ResampleGraph | null = null;

  function buildResample(sampleCount: number, spacing: 'arc-length' | 'time'): ResampleGraph {
    if (resample) {
      resources.release(resample.compiled);
      resources.release(resample.samples);
      resources.release(resample.routeSegments);
    }
    const segmentsPerRoute = sampleCount - 1;
    const samples = resources.createBuffer(`samples-${sampleCount}`, trackCount * sampleCount * 8);
    const routeSegments = resources.createBuffer(
      `route-segments-${sampleCount}`,
      trackCount * segmentsPerRoute * 16
    );
    const graph = new GPUCommandGraph<void>(device, {id: `harbor-resample-${spacing}`});
    const offsetsView = importGraphBuffer(
      graph,
      'offsets',
      offsetsBuffer,
      'uint32',
      trackCount + 1
    );
    const samplesView = importGraphBuffer(
      graph,
      'samples',
      samples,
      'float32x2',
      trackCount * sampleCount
    );
    graph.add(
      new GPUTrajectoryResample({
        id: 'resample',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
        timestamps: importGraphBuffer(
          graph,
          'timestamps',
          timestampsBuffer,
          'float32',
          vertexCount
        ),
        trackOffsets: offsetsView,
        sampleCount,
        spacing,
        samples: samplesView
      })
    );
    // Polyline segments between consecutive samples of one track, for drawing.
    addKernelPass(graph, {
      id: 'route-segments',
      invocationCount: trackCount * segmentsPerRoute,
      declarations: `const SAMPLES: u32 = ${sampleCount}u;`,
      bindings: [
        {
          name: 'samples',
          view: samplesView,
          type: 'f32',
          access: 'read'
        },
        {
          name: 'offsets',
          view: offsetsView,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'segments',
          view: importGraphBuffer(
            graph,
            'route-segments',
            routeSegments,
            'float32',
            trackCount * segmentsPerRoute * 4
          ),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let track = index / (SAMPLES - 1u);
  let k = index % (SAMPLES - 1u);
  let row = (track * SAMPLES + k) * 2u;
  var valid = offsets[offsetsOffset + track + 1u] - offsets[offsetsOffset + track] >= 2u;
  let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  segments[segmentsOffset + index * 4u] = select(nan, samples[samplesOffset + row], valid);
  segments[segmentsOffset + index * 4u + 1u] = select(nan, samples[samplesOffset + row + 1u], valid);
  segments[segmentsOffset + index * 4u + 2u] = select(nan, samples[samplesOffset + row + 2u], valid);
  segments[segmentsOffset + index * 4u + 3u] = select(nan, samples[samplesOffset + row + 3u], valid);`
    });
    return {
      compiled: resources.track(graph.compile()),
      samples,
      routeSegments,
      sampleCount,
      encoded: false
    };
  }

  // ---- State ------------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'playing', speed: 'playbackSpeed', loop: 'loop'},
    {range: [0, SECONDS_PER_DAY], rate: 1, step: 60}
  );
  let playhead = ctx.options.time;
  let chartHour = -1;

  // Tracks with a position, per hour of the UTC day (static: from each track's first and last fix).
  const hourlyTracks = new Float64Array(24);
  for (let track = 0; track < trackCount; track++) {
    const first = vessels.timestamps[vessels.offsets[track]];
    const last = vessels.timestamps[vessels.offsets[track + 1] - 1];
    const firstHour = Math.max(0, Math.floor(first / 3600));
    const lastHour = Math.min(23, Math.floor(last / 3600));
    for (let hour = firstHour; hour <= lastHour; hour++) hourlyTracks[hour]++;
  }
  const hourCenters = Array.from({length: 24}, (_, hour) => hour + 0.5);
  function updateHourlyChart(): void {
    const marker = Math.round((playhead / 3600) * 4) / 4;
    if (marker === chartHour) return;
    chartHour = marker;
    ctx.setChart(
      'hourlyChart',
      lineChart(hourCenters, hourlyTracks, {
        label: 'tracks',
        xLabel: 'hour of day (UTC)',
        yLabel: 'tracks reporting',
        xDomain: [0, 24],
        markers: [{x: marker, label: 'now'}],
        formatX: value => `${Math.round(value)}`,
        formatY: value => `${Math.round(value)}`,
        description:
          'Number of AIS tracks with a fix in each hour of the day; a line marks the playhead.'
      })
    );
  }
  updateHourlyChart();
  let selectedTrack = NO_TRACK;
  let destroyed = false;
  let metricsDirty = true;
  let statusStale = true;
  let statusSnapshot: {
    status: Uint32Array;
    positions: Float32Array;
    speeds: Float32Array;
  } | null = null;
  let metricsSnapshot: {
    lengths: Float32Array;
    averages: Float32Array;
    maxima: Float32Array;
    stopCounts: Uint32Array;
    durations: Float32Array;
  } | null = null;

  ctx.setReadout(
    'tracks',
    `${formatCount(trackCount)} tracks / ${formatCount(vessels.vesselCount)} vessels`
  );
  ctx.setReadout('vertices', `${formatCount(vertexCount)} AIS fixes`);

  function writeMetricsParameters(): void {
    metricsParameters.write(
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: ctx.options.stopSpeedKnots * METERS_PER_KNOT_SECOND,
        stopMinimumDuration: ctx.options.stopMinutes * 60
      })
    );
    metricsDirty = true;
  }

  function writeSegmentMask(): void {
    const filter = ctx.options.vesselFilter;
    const mask = new Uint32Array(segmentCount);
    if (filter === 'all') {
      mask.fill(1);
    } else {
      const wanted = VESSEL_CATEGORIES.indexOf(filter as (typeof VESSEL_CATEGORIES)[number]);
      for (let segment = 0; segment < segmentCount; segment++) {
        mask[segment] = vessels.category[vessels.segmentTracks[segment]] === wanted ? 1 : 0;
      }
    }
    segmentMaskBuffer.write(mask);
  }

  writeMetricsParameters();
  writeSegmentMask();
  resample = buildResample(Number(ctx.options.routeSamples), ctx.options.routeSpacing);
  ctx.setReadout(
    'resampled',
    `${formatCount(trackCount)} routes x ${ctx.options.routeSamples} samples`
  );

  // ---- Readbacks --------------------------------------------------------------------------------
  const statusReader = new SummaryReader(
    resources,
    'harbor-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: activeOverflow, size: 4},
      {buffer: status, size: trackCount * 4},
      {buffer: currentPositions, size: trackCount * 8},
      {buffer: speeds, size: trackCount * 4},
      {buffer: trailCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const statusStart = 2;
      const positionStart = statusStart + trackCount;
      const speedStart = positionStart + trackCount * 2;
      const trailStart = speedStart + trackCount;
      statusSnapshot = {
        status: words.slice(statusStart, statusStart + trackCount),
        positions: floats.slice(positionStart, positionStart + trackCount * 2),
        speeds: floats.slice(speedStart, speedStart + trackCount)
      };
      let before = 0;
      let after = 0;
      let gap = 0;
      for (let track = 0; track < trackCount; track++) {
        const value = statusSnapshot.status[track];
        if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.beforeStart) before++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.afterEnd) after++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.gap) gap++;
      }
      ctx.setReadout('active', words[0]);
      ctx.setReadout('beforeStart', before);
      ctx.setReadout('afterEnd', after);
      ctx.setReadout('inGap', gap);
      ctx.setReadout('trailSegments', words[trailStart]);
      updateHourlyChart();
      describeSelection();
    }
  );

  const metricsReader = new SummaryReader(
    resources,
    'harbor-metrics',
    [
      {buffer: stopCount, size: 4},
      {buffer: stopOverflow, size: 4},
      {buffer: stopTotal, size: 4},
      {buffer: trackLengths, size: trackCount * 4},
      {buffer: averageSpeeds, size: trackCount * 4},
      {buffer: maximumSpeeds, size: trackCount * 4},
      {buffer: trackStopCounts, size: trackCount * 4},
      {buffer: stopDurations, size: STOP_CAPACITY * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const lengthStart = 3;
      const averageStart = lengthStart + trackCount;
      const maximumStart = averageStart + trackCount;
      const stopCountStart = maximumStart + trackCount;
      const durationStart = stopCountStart + trackCount;
      metricsSnapshot = {
        lengths: floats.slice(lengthStart, averageStart),
        averages: floats.slice(averageStart, maximumStart),
        maxima: floats.slice(maximumStart, stopCountStart),
        stopCounts: words.slice(stopCountStart, durationStart),
        durations: floats.slice(durationStart, durationStart + STOP_CAPACITY)
      };
      let totalLength = 0;
      let fastest = 0;
      let tracksWithStops = 0;
      for (let track = 0; track < trackCount; track++) {
        totalLength += metricsSnapshot.lengths[track];
        fastest = Math.max(fastest, metricsSnapshot.maxima[track]);
        if (metricsSnapshot.stopCounts[track] > 0) tracksWithStops++;
      }
      let longest = 0;
      for (let stop = 0; stop < Math.min(words[0], STOP_CAPACITY); stop++) {
        longest = Math.max(longest, metricsSnapshot.durations[stop]);
      }
      ctx.setReadout('stops', `${formatCount(words[2])}${words[1] ? ' (list truncated)' : ''}`);
      ctx.setReadout(
        'stopTracks',
        `${formatCount(tracksWithStops)} of ${formatCount(trackCount)} tracks`
      );
      ctx.setReadout('longestStop', formatDuration(longest));
      const averageKnots = Float32Array.from(
        metricsSnapshot.averages,
        speed => speed * KNOTS_PER_METER_SECOND
      );
      ctx.setChart(
        'speedChart',
        histogramChart(binValues(averageKnots, 0, 20, 20), 0, 20, {
          xLabel: 'mean speed of a track (kn)',
          yLabel: 'tracks',
          formatX: value => value.toFixed(0),
          description:
            'Histogram of the mean speed of every AIS track. The spike near zero is moored vessels.'
        })
      );
      const stopCountValue = Math.min(words[0], STOP_CAPACITY);
      const stopMinutes = Float32Array.from(
        metricsSnapshot.durations.subarray(0, stopCountValue),
        seconds => seconds / 60
      );
      ctx.setChart(
        'dwellChart',
        stopCountValue
          ? histogramChart(binValues(stopMinutes, 0, 360, 24), 0, 360, {
              xLabel: 'stop duration (min, 6 h and over in the last bin)',
              yLabel: 'stops',
              color: 3,
              formatX: value => `${Math.round(value)}`,
              description: 'Histogram of detected stop durations at the current thresholds.'
            })
          : null
      );
      ctx.setReadout('distance', `${formatCount(totalLength / NAUTICAL_MILE_METERS)} nm`);
      ctx.setReadout('fastest', `${(fastest * KNOTS_PER_METER_SECOND).toFixed(1)} kn`);
      describeSelection();
    }
  );

  function describeVessel(track: number): string {
    const category = VESSEL_CATEGORIES[vessels.category[track]];
    const length = vessels.length[track];
    return `${VESSEL_CATEGORY_LABELS[category]} (MMSI ${vessels.mmsi[track]}${length > 0 ? `, ${length.toFixed(0)} m` : ''})`;
  }

  function describeSelection(): void {
    if (selectedTrack === NO_TRACK) {
      ctx.setReadout('selected', 'click a vessel');
      return;
    }
    let text = describeVessel(selectedTrack);
    if (metricsSnapshot) {
      const distance = metricsSnapshot.lengths[selectedTrack] / NAUTICAL_MILE_METERS;
      const average = metricsSnapshot.averages[selectedTrack] * KNOTS_PER_METER_SECOND;
      const maximum = metricsSnapshot.maxima[selectedTrack] * KNOTS_PER_METER_SECOND;
      text += `: ${distance.toFixed(1)} nm, mean ${average.toFixed(1)} kn, max ${maximum.toFixed(1)} kn, ${metricsSnapshot.stopCounts[selectedTrack]} stops`;
    }
    ctx.setReadout('selected', text);
  }
  describeSelection();

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      playheadCompiled,
      trailCompiled,
      metricsCompiled,
      ...(resample ? [resample.compiled] : [])
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'vesselFilter':
          writeSegmentMask();
          ctx.requestLayers();
          break;
        case 'stopSpeedKnots':
        case 'stopMinutes':
          writeMetricsParameters();
          statusStale = true;
          break;
        case 'routeSamples':
        case 'routeSpacing':
          resample = buildResample(Number(state.routeSamples), state.routeSpacing);
          ctx.setReadout(
            'resampled',
            `${formatCount(trackCount)} routes x ${state.routeSamples} samples`
          );
          ctx.requestLayers();
          break;
        case 'time':
        case 'playing':
        case 'playbackSpeed':
        case 'loop':
        case 'maxGapMinutes':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      // Paused, the clock sits on the slider, so every story step and deep link is deterministic.
      playhead = clock.advance(frame);
      ctx.setReadout('clock', `${formatUtcClock(playhead)} UTC (${formatEasternClock(playhead)})`);

      if (metricsDirty) {
        metricsCompiled.encode(commandEncoder, {parameters: undefined});
        metricsDirty = false;
        metricsReader.request(commandEncoder);
      } else {
        metricsReader.flush(commandEncoder);
      }
      if (resample && !resample.encoded) {
        resample.compiled.encode(commandEncoder, {parameters: undefined});
        resample.encoded = true;
      }

      playheadParameters.write(
        getGPUTrajectoryPlayheadParameterValues({
          playhead,
          maxGap: options.maxGapMinutes * 60
        })
      );
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

      if (statusStale || frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) {
        statusReader.markStale();
        statusStale = false;
      }
      statusReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'harbor-backdrop',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            color: dark ? [190, 200, 220, 26] : [60, 70, 90, 30]
          })
        );
      }
      if (options.showRoutes && resample) {
        const routeSegmentCount = trackCount * (resample.sampleCount - 1);
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `harbor-routes-${resample.sampleCount}`,
            coordinateOrigin,
            segments: resample.routeSegments,
            instanceCount: routeSegmentCount,
            values: categoryBuffer,
            valueFormat: 'uint32',
            valueDivisor: resample.sampleCount - 1,
            colormap: 'category',
            palette: VESSEL_CATEGORY_COLORS,
            widthPixels: 1.6,
            opacity: 0.55
          }),
          new SpatialAnalysisPointLayer({
            id: `harbor-route-samples-${resample.sampleCount}`,
            coordinateOrigin,
            positions: resample.samples,
            instanceCount: trackCount * resample.sampleCount,
            values: categoryBuffer,
            valueFormat: 'uint32',
            valueDivisor: resample.sampleCount,
            colormap: 'category',
            palette: VESSEL_CATEGORY_COLORS,
            radiusPixels: 2.4,
            opacity: 0.95
          })
        );
      }
      if (options.showTrails) {
        const bySpeed = options.trailColor === 'speed';
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'harbor-trails',
            coordinateOrigin,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            values: bySpeed ? stepSpeeds : categoryBuffer,
            valueFormat: bySpeed ? 'float32' : 'uint32',
            valueIndices: bySpeed ? segmentEndsBuffer : segmentTracksBuffer,
            colormap: bySpeed ? options.ramp : 'category',
            valueScale: KNOTS_PER_METER_SECOND,
            valueRange: [0, SPEED_RAMP_KNOTS],
            palette: VESSEL_CATEGORY_COLORS,
            widthPixels: 2.6
          })
        );
      }
      if (options.showStops) {
        layers.push(
          new StopMarkerLayer({
            id: 'harbor-stops',
            coordinateOrigin,
            centroids: stopCentroids,
            durations: stopDurations,
            drawCommands: stopDraw,
            baseRadiusPixels: 3,
            radiusPerSqrtSecond: 0.12,
            maximumRadiusPixels: 16,
            durationForFullColor: 6 * 3600,
            opacity: 0.85
          })
        );
      }
      layers.push(
        new VesselMarkerLayer({
          id: 'harbor-vessels',
          coordinateOrigin,
          ids: activeIds,
          positions: currentPositions,
          headings,
          speeds,
          categories: categoryBuffer,
          drawCommands: markerDraw,
          sizePixels: options.markerSize,
          colorMode: options.markerColor,
          ramp: options.ramp,
          speedForFullColor: SPEED_RAMP_KNOTS * METERS_PER_KNOT_SECOND,
          palette: VESSEL_CATEGORY_COLORS,
          categoryFilter:
            options.vesselFilter === 'all'
              ? null
              : VESSEL_CATEGORIES.indexOf(
                  options.vesselFilter as (typeof VESSEL_CATEGORIES)[number]
                ),
          selectedTrack: selectedTrack === NO_TRACK ? null : selectedTrack,
          outlineColor: dark ? [8, 10, 16, 235] : [20, 24, 32, 215]
        })
      );
      return layers;
    },

    getTooltip(event) {
      const track = pickVessel(event.pixel);
      if (track < 0 || !statusSnapshot) return null;
      const speed = statusSnapshot.speeds[track] * KNOTS_PER_METER_SECOND;
      const stops = metricsSnapshot?.stopCounts[track];
      return `${describeVessel(track)} - ${speed.toFixed(1)} kn${stops === undefined ? '' : `, ${stops} stops today`}`;
    },

    onClick(event) {
      const track = pickVessel(event.pixel);
      selectedTrack = track < 0 || track === selectedTrack ? NO_TRACK : track;
      describeSelection();
      ctx.requestLayers();
      return track >= 0;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      metricsReader.stop();
      resources.destroy();
    }
  };

  /** Nearest active vessel within 14 CSS pixels of the pointer, or -1. */
  function pickVessel(pixel: readonly [number, number]): number {
    const viewport = ctx.getViewport();
    const snapshot = statusSnapshot;
    if (!viewport || !snapshot) return -1;
    const filter = ctx.options.vesselFilter;
    let best = -1;
    let bestDistance = 14 * 14;
    for (let track = 0; track < trackCount; track++) {
      if (snapshot.status[track] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active) continue;
      if (filter !== 'all' && VESSEL_CATEGORIES[vessels.category[track]] !== filter) continue;
      const [longitude, latitude] = vessels.unproject(
        snapshot.positions[track * 2],
        snapshot.positions[track * 2 + 1]
      );
      const [x, y] = viewport.project([longitude, latitude]);
      const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
      if (squared < bestDistance) {
        bestDistance = squared;
        best = track;
      }
    }
    return best;
  }
}
