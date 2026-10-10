// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
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
import {NYC, nearestPlaceLabel} from '../../cartography/gazetteer';
import {formatCount, formatDistance, formatPercent, liveText} from '../../cartography/live-text';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {MapAnnotation, SceneContext, SceneInstance, TooltipContent} from '../scene';
import {StopMarkerLayer, VesselMarkerLayer} from './b12-layers';
import {
  KNOTS_PER_METER_SECOND,
  loadVesselTracks,
  METERS_PER_KNOT_SECOND,
  VESSEL_CATEGORIES,
  VESSEL_CATEGORY_LABELS
} from './b12-tracks';
import {buildZoneSet} from './b12-zones';
import {binValues, histogramChart, lineChart} from './f-chart-helpers';
import {
  buildAnchorages,
  findTrackBracket,
  getFerryTracks,
  getReportingProfile,
  getSegmentRowStarts,
  getTimeSamplePile,
  getTrackBounds,
  getTrackPathLength,
  getVesselTracks,
  pickMovingFerry,
  pickTransitTrack,
  summariseStops,
  type StopSummary,
  type TrackBracket
} from './harbor-playback-focus';
import {
  getFocusInks,
  getStopClassColors,
  STOP_BASE_RADIUS_PIXELS,
  STOP_BREAKS_MINUTES,
  STOP_BREAKS_SECONDS,
  STOP_MAXIMUM_RADIUS_PIXELS,
  STOP_RADIUS_PER_SQRT_SECOND
} from './harbor-playback-style';
import {
  CONTEXT_TRACK_INK,
  formatDwell,
  formatKnots,
  formatZonedClock,
  getCategoryPaletteFromGroups,
  getShipSpeedClasses,
  HARBOR_CATEGORY_TO_GROUP,
  HEAD_HALO_INK,
  SHIP_SPEED_BREAKS_KNOTS,
  SUBJECT_INK,
  VESSEL_GROUP_LABELS,
  VESSEL_GROUPS
} from './movement-style';

/** Option state of the harbor playback scene. */
export type HarborPlaybackOptions = {
  playing: boolean;
  time: number;
  playbackSpeed: number;
  loop: boolean;
  markerColor: 'uniform' | 'group' | 'speed';
  stoppedSquares: boolean;
  showBackdrop: boolean;
  vesselFilter: string;
  markerSize: number;
  labelFerry: boolean;
  maxGapMinutes: number;
  focus: 'none' | 'ferry' | 'transit';
  showTrails: boolean;
  trailMinutes: number;
  tailFade: number;
  routeSpacing: 'arc-length' | 'time';
  showRoutes: boolean;
  routeSamples: '16' | '32' | '64';
  showStops: boolean;
  stopSpeedKnots: number;
  stopMinutes: number;
};

const STOP_CAPACITY = 4096;
/** Frames between status readbacks while playing (a readback is also requested when anything moves by hand). */
const STATUS_INTERVAL_FRAMES = 12;
const SECONDS_PER_DAY = 86400;
const NO_TRACK = 0xffffffff;
const NAUTICAL_MILE_METERS = 1852;
const DAY_ORIGIN = '2024-06-12T00:00:00Z';
const NEW_YORK = 'America/New_York';
/** Zoom at which arrows grow from base size - 1 to base size + 1 pixels. */
const MARKER_ZOOM_BREAK = 11.5;
/** The harbour overview the fingerprint flies to. */
const HARBOR_OVERVIEW = {longitude: -74.05, latitude: 40.655, zoom: 10.6};
/** Stop speed thresholds of the sweep, in knots: 0.2 to 3.0. */
const SWEEP_THRESHOLDS_KNOTS = Array.from(
  {length: 15},
  (_, index) => Math.round((0.2 + index * 0.2) * 10) / 10
);
/** Frames the stop minimum duration must rest before the sweep is rerun. */
const SWEEP_SETTLE_FRAMES = 20;
/** Fixes drawn on each side of the bracketing pair of the ferry. */
const FERRY_FIX_WINDOW = 7;
/** A ferry slower than this (m/s, about 3 kn) is not "mid-crossing". */
const FERRY_MINIMUM_SPEED = 3 * METERS_PER_KNOT_SECOND;
/** Radius in metres within which resampled points count as a pile. */
const PILE_RADIUS_METERS = 150;

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
 * once per threshold change (and, on the stops step, once per swept threshold), and a resample
 * graph rebuilds each track as evenly spaced samples. All of them take their per-frame inputs as
 * parameter buffers; the only compile-time choices are the number and spacing of resampled points.
 *
 * Two CPU mirrors serve the story: {@link findTrackBracket} repeats the playhead's binary search
 * for one vessel so the fixes, the chord and the fraction can be drawn, and the stop list is joined
 * to vessel groups and anchorages when it is read back.
 */
export async function createHarborPlayback(
  ctx: SceneContext<HarborPlaybackOptions>
): Promise<SceneInstance<HarborPlaybackOptions>> {
  const vessels = loadVesselTracks(ctx.datasets.get('ais-vessels'));
  const zonesDataset = ctx.datasets.get('ais-zones');
  if (!zonesDataset.geojson) throw new Error('ais-zones has no geometry');
  const zones = buildZoneSet(zonesDataset.geojson, vessels.project);
  const anchorages = buildAnchorages(zones, vessels.unproject);
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = vessels;
  const resources = new SpatialAnalysisResources(device, 'harbor');
  const coordinateOrigin: [number, number, number] = [vessels.origin[0], vessels.origin[1], 0];
  const segmentRowStarts = getSegmentRowStarts(vessels);

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

  // The anchorage outlines (hairlines of the stops step): the edges of the anchorage polygons only.
  const anchorageEdges: number[] = [];
  for (let edge = 0; edge < zones.edgeZones.length; edge++) {
    const zone = zones.edgeZones[edge];
    // ZONE_KINDS[0] is the anchorage kind.
    if (zones.kinds[zone] === 0) {
      anchorageEdges.push(...zones.outlineSegments.subarray(edge * 4, edge * 4 + 4));
    }
  }
  const anchorageEdgeCount = anchorageEdges.length / 4;
  const anchorageOutlineBuffer = resources.createBuffer(
    'anchorage-outlines',
    Float32Array.from(anchorageEdges)
  );

  // Small id lists the focus-vessel layers gather through (rewritten when the focus changes).
  const focusCapacity = Math.max(vessels.longestTrack, 64);
  const focusFixIds = resources.createBuffer('focus-fix-ids', focusCapacity * 4);
  const focusSegmentIds = resources.createBuffer('focus-segment-ids', focusCapacity * 4);
  const bracketFixIds = resources.createBuffer('bracket-fix-ids', 2 * 4);
  const bracketSegmentId = resources.createBuffer('bracket-segment-id', 4);
  const focusTrackId = resources.createBuffer('focus-track-id', 4);
  const sampleIds = resources.createBuffer('sample-ids', 64 * 4);
  const routeIds = resources.createBuffer('route-ids', 64 * 4);
  const focusArrowDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'harbor-focus-arrow-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 1}]
    })
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
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
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
          requiredCount: importGraphBuffer(metricsGraph, 'stop-total', stopTotal, 'uint32', 1)
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
    {range: [0, SECONDS_PER_DAY], rate: 1, step: 10}
  );
  let playhead = ctx.options.time;
  let frameCounter = 0;
  let activityChartBin = -1;
  let selectedTrack = NO_TRACK;
  let destroyed = false;
  let metricsDirty = true;
  let statusStale = true;
  let statusSnapshot: {
    status: Uint32Array;
    positions: Float32Array;
    speeds: Float32Array;
    headings: Float32Array;
  } | null = null;
  let metricsSnapshot: {
    lengths: Float32Array;
    averages: Float32Array;
    maxima: Float32Array;
    stopCounts: Uint32Array;
    durations: Float32Array;
  } | null = null;
  let averageKnots: Float32Array | null = null;
  let stopMinutesList: Float32Array | null = null;
  let stopSummary: StopSummary | null = null;
  let lastGroupCounts = '';

  // ---- The day, from the data -------------------------------------------------------------------
  const stoppedSpeedDefault = 0.5 * METERS_PER_KNOT_SECOND;
  const profile = getReportingProfile(vessels, stoppedSpeedDefault);
  const ferryTracks = getFerryTracks(
    vessels,
    NYC.places['st-george-terminal'].lngLat,
    NYC.places['whitehall-terminal'].lngLat
  );
  const transitTrack = pickTransitTrack(vessels, stoppedSpeedDefault);

  const activityHours = Array.from(
    {length: profile.quarterHourTracks.length},
    (_, bin) => (bin + 0.5) / 4
  );
  function updateActivityChart(): void {
    const bin = Math.min(
      profile.quarterHourTracks.length - 1,
      Math.max(0, Math.floor(playhead / 900))
    );
    if (bin === activityChartBin) return;
    activityChartBin = bin;
    ctx.setChart(
      'activityChart',
      lineChart(activityHours, profile.quarterHourTracks, {
        label: 'reporting tracks',
        xLabel: 'time of day (New York)',
        yLabel: 'tracks reporting',
        xDomain: [0, 24],
        yDomain: [0, Math.max(...profile.quarterHourTracks) * 1.08],
        markers: [{x: (bin + 0.5) / 4, label: 'now'}],
        formatX: value =>
          formatZonedClock(value * 3600, DAY_ORIGIN, NEW_YORK).replace(/:\d\d /, ' '),
        formatY: value => formatCount(value),
        description:
          'Tracks with at least one AIS fix in each 15-minute interval of the UTC day, labelled in New York local time. The vertical rule is the shared playhead; an afternoon break records the feed-wide reporting silence.'
      })
    );
  }
  updateActivityChart();

  ctx.setLegendData('ground', ctx.ground());
  ctx.setTimelineData({
    domain: [0, SECONDS_PER_DAY],
    histogram: Array.from(profile.quarterHourTracks),
    events: profile.silence ? [{at: profile.silence.start, label: 'No reports'}] : []
  });
  const cadenceText =
    profile.fixInterval >= 45 && profile.fixInterval <= 90
      ? 'about one fix a minute'
      : `a fix every ${formatDwell(profile.fixInterval)} (median)`;
  ctx.setFurniture({
    title: {
      sample: `${formatCount(vessels.vesselCount)} vessels · ${formatCount(trackCount)} tracks · ${cadenceText}`
    }
  });
  ctx.setReadout(
    'tracks',
    `${formatCount(trackCount)} tracks / ${formatCount(vessels.vesselCount)} vessels`
  );
  ctx.setReadout('vertices', `${formatCount(vertexCount)} AIS fixes`);
  ctx.setReadout(
    'numerics',
    'Positions are linear interpolations in planar metres between two fixes, found by one binary search per track (the GPU runs all tracks at once). Timestamps are float32 seconds, exact below 2^24. Speeds are planar step speeds, not the reported speed over ground.'
  );
  ctx.setReadout('fixCadence', formatDwell(profile.fixInterval));
  ctx.setReadout('mooredCadence', formatDwell(profile.mooredInterval));
  ctx.setReadout(
    'busiestHour',
    formatZonedClock(profile.busiestHourStart + 1800, DAY_ORIGIN, NEW_YORK).replace(/:\d\d /, ' ')
  );
  ctx.setReadout(
    'silence',
    profile.silence
      ? `${formatDwell(profile.silence.length)}, from ${formatZonedClock(profile.silence.start, DAY_ORIGIN, NEW_YORK)}`
      : 'none: some vessel reports at all times'
  );
  if (ctx.reducedMotion() && ctx.options.playing) ctx.setOptions({playing: false});

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
  updateResampledReadout();

  function updateResampledReadout(): void {
    ctx.setReadout(
      'resampled',
      `${formatCount(trackCount)} routes x ${ctx.options.routeSamples} samples`
    );
  }

  function updateCost(): void {
    const options = ctx.options;
    ctx.setCost({
      records: vertexCount,
      passes: 1 + (options.showTrails ? 1 : 0) + (options.showStops ? 1 : 0)
    });
  }
  updateCost();

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
      {buffer: headings, size: trackCount * 4},
      {buffer: trailCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const statusStart = 2;
      const positionStart = statusStart + trackCount;
      const speedStart = positionStart + trackCount * 2;
      const headingStart = speedStart + trackCount;
      const trailStart = headingStart + trackCount;
      statusSnapshot = {
        status: words.slice(statusStart, statusStart + trackCount),
        positions: floats.slice(positionStart, positionStart + trackCount * 2),
        speeds: floats.slice(speedStart, speedStart + trackCount),
        headings: floats.slice(headingStart, headingStart + trackCount)
      };
      let before = 0;
      let after = 0;
      let gap = 0;
      let stopped = 0;
      const stoppedSpeed = ctx.options.stopSpeedKnots * METERS_PER_KNOT_SECOND;
      const groupCounts = new Array<number>(VESSEL_GROUPS.length).fill(0);
      for (let track = 0; track < trackCount; track++) {
        const value = statusSnapshot.status[track];
        if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.beforeStart) before++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.afterEnd) after++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.gap) gap++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.active) {
          groupCounts[HARBOR_CATEGORY_TO_GROUP[vessels.category[track]]]++;
          if (statusSnapshot.speeds[track] < stoppedSpeed) stopped++;
        }
      }
      const active = words[0];
      ctx.setReadout('active', active);
      ctx.setReadout('beforeStart', before);
      ctx.setReadout('afterEnd', after);
      ctx.setReadout('inGap', gap);
      ctx.setReadout('trailSegments', words[trailStart]);
      ctx.setReadout('keptShare', formatPercent(words[trailStart] / Math.max(1, segmentCount), 1));
      ctx.setReadout('stoppedNow', stopped);
      ctx.setReadout('stoppedShare', active > 0 ? formatPercent(stopped / active) : '–');
      if (ctx.options.markerColor === 'group') {
        const signature = groupCounts.join(',');
        if (signature !== lastGroupCounts) {
          lastGroupCounts = signature;
          ctx.setLegendData('groupCounts', groupCounts);
        }
      }
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
      {buffer: stopDurations, size: STOP_CAPACITY * 4},
      {buffer: stopIds, size: STOP_CAPACITY * 4},
      {buffer: stopCentroids, size: STOP_CAPACITY * 8}
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
      const idStart = durationStart + STOP_CAPACITY;
      const centroidStart = idStart + STOP_CAPACITY;
      metricsSnapshot = {
        lengths: floats.slice(lengthStart, averageStart),
        averages: floats.slice(averageStart, maximumStart),
        maxima: floats.slice(maximumStart, stopCountStart),
        stopCounts: words.slice(stopCountStart, durationStart),
        durations: floats.slice(durationStart, durationStart + STOP_CAPACITY)
      };
      let totalLength = 0;
      let fastest = 0;
      for (let track = 0; track < trackCount; track++) {
        totalLength += metricsSnapshot.lengths[track];
        fastest = Math.max(fastest, metricsSnapshot.maxima[track]);
      }
      const listed = Math.min(words[0], STOP_CAPACITY);
      let longest = 0;
      for (let stop = 0; stop < listed; stop++) {
        longest = Math.max(longest, metricsSnapshot.durations[stop]);
      }
      ctx.setReadout('stops', `${formatCount(words[2])}${words[1] ? ' (list truncated)' : ''}`);
      ctx.setReadout('longestStop', formatDwell(longest));
      averageKnots = Float32Array.from(
        metricsSnapshot.averages,
        speed => speed * KNOTS_PER_METER_SECOND
      );
      publishSpeedChart();
      stopMinutesList = Float32Array.from(
        metricsSnapshot.durations.subarray(0, listed),
        seconds => seconds / 60
      );
      publishDwellChart();
      stopSummary = summariseStops({
        vessels,
        anchorages,
        count: listed,
        trackIds: words.subarray(idStart, idStart + listed),
        centroids: floats.subarray(centroidStart, centroidStart + listed * 2),
        durations: metricsSnapshot.durations,
        classBreaksSeconds: STOP_BREAKS_SECONDS
      });
      ctx.setReadout('anchoredShare', formatPercent(stopSummary.anchoredShare));
      ctx.setReadout('stopShare', formatPercent(stopSummary.tugAndFerryShare));
      ctx.setLegendData('stopClassCounts', stopSummary.classCounts);
      publishStopNotes();
      ctx.setReadout('distance', `${formatCount(totalLength / NAUTICAL_MILE_METERS)} nm`);
      ctx.setReadout('fastest', `${(fastest * KNOTS_PER_METER_SECOND).toFixed(1)} kn`);
      describeSelection();
    }
  );

  /** The histogram of track mean speeds, in the same five classes as the map. */
  function publishSpeedChart(): void {
    if (!averageKnots) return;
    ctx.setChart(
      'speedChart',
      histogramChart(binValues(averageKnots, 0, 20, 40), 0, 20, {
        xLabel: 'mean speed of a track (kn)',
        yLabel: 'tracks',
        breaks: [...SHIP_SPEED_BREAKS_KNOTS],
        classColors: getShipSpeedClasses(ctx.ground()),
        formatX: value => value.toFixed(0),
        description:
          'Histogram of the mean speed of every AIS track over the day, coloured by the five speed classes of the map. The tall first bar is vessels tied up all day.'
      })
    );
  }

  /** The histogram of stop durations, in the four dwell classes of the stop discs. */
  function publishDwellChart(): void {
    ctx.setChart(
      'dwellChart',
      stopMinutesList?.length
        ? histogramChart(binValues(stopMinutesList, 0, 480, 32), 0, 480, {
            xLabel: 'stop duration (min; 8 h and over in the last bar)',
            yLabel: 'stops',
            breaks: [...STOP_BREAKS_MINUTES],
            classColors: getStopClassColors(ctx.ground()),
            formatX: value => `${Math.round(value)}`,
            description:
              'Histogram of detected stop durations at the current thresholds, coloured by the four dwell classes of the map.'
          })
        : null
    );
  }

  /** Finding notes of the stops step: the densest stops and the busiest official anchorage. */
  function publishStopNotes(): void {
    if (!ctx.options.showStops || !stopSummary) {
      ctx.setAnnotations('stop-notes', null);
      return;
    }
    const notes: MapAnnotation[] = [];
    const cluster = stopSummary.busiestCluster;
    if (cluster) {
      notes.push({
        kind: 'note',
        id: 'busiest-cluster',
        coordinate: cluster.lngLat,
        title: `${formatCount(cluster.count)} stops here`,
        text: nearestPlaceLabel(NYC, cluster.lngLat) ?? undefined
      });
    }
    const anchorage = stopSummary.busiestAnchorage;
    if (anchorage) {
      notes.push({
        kind: 'note',
        id: 'busiest-anchorage',
        coordinate: anchorage.lngLat,
        title: `${anchorage.name}: ${formatCount(anchorage.count)} stops`,
        text: anchorage.region
          ? `the busiest official anchorage, ${anchorage.region}`
          : 'the busiest official anchorage'
      });
    }
    ctx.setAnnotations('stop-notes', notes.length ? notes : null);
  }

  // ---- The stop-speed sweep: one compiled graph, one parameter write per run --------------------
  const sweep = {active: false, index: 0, awaiting: false, counts: [] as number[]};
  // A deep link that opens on the stops step has no option change to wake the sweep.
  let sweepStale = ctx.options.showStops;
  let sweepChangedAt = 0;

  const sweepReader = new SummaryReader(
    resources,
    'harbor-sweep',
    [{buffer: stopTotal, size: 4}],
    bytes => {
      if (destroyed || !sweep.active) return;
      sweep.counts.push(new Uint32Array(bytes)[0]);
      sweep.index++;
      sweep.awaiting = false;
      if (sweep.index >= SWEEP_THRESHOLDS_KNOTS.length) finishSweep();
    }
  );

  function startSweep(): void {
    sweep.active = true;
    sweep.index = 0;
    sweep.awaiting = false;
    sweep.counts = [];
    sweepStale = false;
    // The sweep overwrites the stop list, so the discs are hidden until the reader's own run.
    ctx.requestLayers();
  }

  function runSweepStep(commandEncoder: CommandEncoder): void {
    if (sweep.awaiting) {
      sweepReader.flush(commandEncoder);
      return;
    }
    metricsParameters.write(
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: SWEEP_THRESHOLDS_KNOTS[sweep.index] * METERS_PER_KNOT_SECOND,
        stopMinimumDuration: ctx.options.stopMinutes * 60
      })
    );
    metricsCompiled.encode(commandEncoder, {parameters: undefined});
    sweep.awaiting = true;
    sweepReader.request(commandEncoder);
  }

  function finishSweep(): void {
    sweep.active = false;
    const counts = sweep.counts;
    const high = Math.max(...counts);
    const low = Math.min(...counts);
    ctx.setReadout('sweepSpread', high > 0 ? formatPercent((high - low) / high) : '–');
    ctx.setChart(
      'stopSweep',
      lineChart(SWEEP_THRESHOLDS_KNOTS, counts, {
        label: 'stops',
        xLabel: 'stop speed threshold (kn)',
        yLabel: 'stops found',
        title: `Compiled once, ${SWEEP_THRESHOLDS_KNOTS.length} runs`,
        yDomain: [0, Math.max(1, high * 1.1)],
        formatX: value => value.toFixed(1),
        formatY: value => formatCount(value),
        link: {option: 'stopSpeedKnots', label: value => `${value.toFixed(1)} kn`},
        description:
          'Number of stops found when the same compiled graph is run at each stop speed threshold from 0.2 to 3 knots, at the current minimum duration. The axis starts at zero.'
      })
    );
    // Back to the reader's own thresholds: the next frame reruns the graph for the map.
    writeMetricsParameters();
    ctx.requestLayers();
  }

  // ---- Between the fixes: the vessel the steps follow --------------------------------------------
  type FocusFrame = {
    track: number;
    bracket: TrackBracket | null;
    /** The arrow is drawn: a bracket exists and the gap test passes. */
    arrowVisible: boolean;
  };
  let focus: FocusFrame = {track: NO_TRACK, bracket: null, arrowVisible: false};
  let focusVesselTracks: number[] = [];
  let focusSignature = '';
  let focusInitialised = false;
  let focusFixCount = 0;
  let focusSegmentCount = 0;
  let lastDiagramFrame = -100;
  let lastFraction = -1;
  let transitIdsSampleCount = 0;

  /** Rewrites the id lists of the focus layers for the current track and bracket. */
  function writeFocusIds(track: number, bracket: TrackBracket | null): void {
    const first = vessels.offsets[track];
    const last = vessels.offsets[track + 1] - 1;
    let fixFrom = first;
    let fixTo = last;
    if (ctx.options.focus === 'ferry') {
      // Only the fixes around the playhead: a day-long ferry track would be a wall of rings.
      if (bracket) {
        fixFrom = Math.max(first, bracket.startRow - FERRY_FIX_WINDOW);
        fixTo = Math.min(last, bracket.endRow + FERRY_FIX_WINDOW);
      } else {
        fixTo = fixFrom - 1;
      }
    }
    const fixes = new Uint32Array(Math.max(0, fixTo - fixFrom + 1));
    for (let row = fixFrom; row <= fixTo; row++) fixes[row - fixFrom] = row;
    focusFixCount = fixes.length;
    if (fixes.length) focusFixIds.write(fixes);
    // Segment `row` of a track joins fix `row` and `row + 1`.
    const segmentFrom = segmentRowStarts[track] + (fixFrom - first);
    const segmentCountHere = Math.max(0, fixTo - fixFrom);
    const segmentsList = new Uint32Array(segmentCountHere);
    for (let index = 0; index < segmentCountHere; index++)
      segmentsList[index] = segmentFrom + index;
    focusSegmentCount = segmentsList.length;
    if (segmentsList.length) focusSegmentIds.write(segmentsList);
    if (bracket) {
      bracketFixIds.write(Uint32Array.of(bracket.startRow, bracket.endRow));
      bracketSegmentId.write(Uint32Array.of(segmentRowStarts[track] + (bracket.startRow - first)));
    }
    focusTrackId.write(Uint32Array.of(track));
  }

  /** Rewrites the sample and route id lists of the transit track for the current sample count. */
  function writeTransitSampleIds(): void {
    if (transitTrack < 0 || !resample) return;
    const count = resample.sampleCount;
    const samples = new Uint32Array(count);
    for (let index = 0; index < count; index++) samples[index] = transitTrack * count + index;
    sampleIds.write(samples);
    const routes = new Uint32Array(count - 1);
    for (let index = 0; index < count - 1; index++)
      routes[index] = transitTrack * (count - 1) + index;
    routeIds.write(routes);
    transitIdsSampleCount = count;
  }

  /** The note and readout that say what the spacing of the transit samples did. */
  function updateTransitNote(): void {
    if (ctx.options.focus !== 'transit' || transitTrack < 0 || !resample) {
      ctx.setAnnotations('route-note', null);
      ctx.setReadout('sampleSpacing', null);
      return;
    }
    const count = resample.sampleCount;
    if (ctx.options.routeSpacing === 'arc-length') {
      const spacing = getTrackPathLength(vessels, transitTrack) / (count - 1);
      const middle = Math.floor(
        (vessels.offsets[transitTrack] + vessels.offsets[transitTrack + 1] - 1) / 2
      );
      const [longitude, latitude] = vessels.unproject(
        vessels.positions[middle * 2],
        vessels.positions[middle * 2 + 1]
      );
      ctx.setReadout('sampleSpacing', `a sample every ${formatDistance(spacing)} along the route`);
      ctx.setAnnotations('route-note', [
        {
          kind: 'note',
          id: 'route-note',
          coordinate: [longitude, latitude],
          title: liveText('A sample every {spacing:distance}', {spacing}),
          text: 'equal steps along the path'
        }
      ]);
    } else {
      const pile = getTimeSamplePile(vessels, transitTrack, count, PILE_RADIUS_METERS);
      const [longitude, latitude] = vessels.unproject(pile.x, pile.y);
      ctx.setReadout('sampleSpacing', `${pile.pileSize} of ${count} samples at one spot`);
      ctx.setAnnotations('route-note', [
        {
          kind: 'note',
          id: 'route-note',
          coordinate: [longitude, latitude],
          title: liveText('{pile:integer} of {count:integer} samples', {
            pile: pile.pileSize,
            count
          }),
          text: `pile up where it waited, ${nearestPlaceLabel(NYC, [longitude, latitude]) ?? 'in the harbour'}`
        }
      ]);
    }
  }

  function fitTransit(): void {
    if (transitTrack < 0) return;
    ctx.fitBounds(getTrackBounds(vessels, transitTrack), {
      transitionMs: 1400,
      maxZoom: 13.5,
      padding: 60
    });
  }

  /** Called when the `focus` option changes (and once after the first frame for a deep link). */
  function onFocusChanged(state: HarborPlaybackOptions): void {
    focusVesselTracks = [];
    focusSignature = '';
    focus = {track: NO_TRACK, bracket: null, arrowVisible: false};
    focusFixCount = 0;
    focusSegmentCount = 0;
    ctx.setAnnotations('focus-notes', null);
    ctx.setChart('fixDiagram', null);
    ctx.setReadout('fraction', null);
    if (state.focus === 'transit') {
      writeTransitSampleIds();
      fitTransit();
    }
    updateTransitNote();
    ctx.requestLayers();
  }

  /** Per frame: which track is in focus at the playhead, its fixes, chord and fraction. */
  function updateFocus(time: number, frameIndex: number): void {
    const options = ctx.options;
    if (options.focus === 'none') return;
    let track = NO_TRACK;
    let bracket: TrackBracket | null = null;
    if (options.focus === 'ferry') {
      if (!focusVesselTracks.length) {
        // The fastest ferry at the playhead the step opened on: caught mid-crossing.
        const pick = pickMovingFerry(vessels, ferryTracks, time, FERRY_MINIMUM_SPEED);
        const start = pick?.track ?? ferryTracks[0];
        if (start !== undefined) focusVesselTracks = getVesselTracks(vessels, start);
      }
      for (const candidate of focusVesselTracks) {
        const found = findTrackBracket(vessels, candidate, time);
        if (found) {
          track = candidate;
          bracket = found;
          break;
        }
      }
    } else if (transitTrack >= 0) {
      track = transitTrack;
      bracket = findTrackBracket(vessels, track, time);
    }
    const maxGapSeconds = options.maxGapMinutes * 60;
    const gapped =
      bracket !== null && maxGapSeconds > 0 && bracket.endTime - bracket.startTime > maxGapSeconds;
    const arrowVisible = bracket !== null && !gapped;
    // The ferry's id lists follow its bracket; the transit track's cover the whole track.
    const signature = `${track}:${options.focus === 'ferry' ? (bracket?.startRow ?? -1) : 0}:${options.focus}`;
    if (signature !== focusSignature || arrowVisible !== focus.arrowVisible) {
      focusSignature = signature;
      if (track !== NO_TRACK) writeFocusIds(track, bracket);
      ctx.requestLayers();
    }
    focus = {track, bracket, arrowVisible};
    if (options.focus !== 'ferry') return;
    if (!bracket) {
      ctx.setReadout('fraction', null);
      ctx.setChart('fixDiagram', null);
      ctx.setAnnotations('focus-notes', null);
      lastFraction = -1;
      return;
    }
    if (bracket.fraction !== lastFraction) {
      lastFraction = bracket.fraction;
      ctx.setReadout('fraction', bracket.fraction.toFixed(2));
      const [longitudeA, latitudeA] = vessels.unproject(
        vessels.positions[bracket.startRow * 2],
        vessels.positions[bracket.startRow * 2 + 1]
      );
      const [longitudeB, latitudeB] = vessels.unproject(
        vessels.positions[bracket.endRow * 2],
        vessels.positions[bracket.endRow * 2 + 1]
      );
      const [longitudeNow, latitudeNow] = vessels.unproject(bracket.x, bracket.y);
      ctx.setAnnotations('focus-notes', [
        {
          kind: 'point',
          id: 'fix-a',
          coordinate: [longitudeA, latitudeA],
          text: 'fix A',
          marker: 'none',
          anchor: 'auto',
          priority: 3
        },
        {
          kind: 'point',
          id: 'fix-b',
          coordinate: [longitudeB, latitudeB],
          text: 'fix B',
          marker: 'none',
          anchor: 'auto',
          priority: 3
        },
        {
          kind: 'note',
          id: 'estimate-note',
          coordinate: [longitudeNow, latitudeNow],
          title: 'Estimated, not measured',
          text: liveText('{kn:fixed:0} kn between the two fixes', {
            kn: bracket.speed * KNOTS_PER_METER_SECOND
          })
        }
      ]);
      // The diagram changes with the fraction: at most ten times a second while the clock runs.
      if (frameIndex - lastDiagramFrame >= 6 || options.playing === false) {
        lastDiagramFrame = frameIndex;
        ctx.setChart('fixDiagram', buildFixDiagram(bracket));
      }
    }
  }

  // ---- A live label on one ferry (steps that play) ----------------------------------------------
  let ferryLabelTrack = NO_TRACK;
  let ferryLabelShown = false;

  function updateFerryLabel(time: number): void {
    if (!ctx.options.labelFerry) {
      if (ferryLabelShown) {
        ferryLabelShown = false;
        ctx.setAnnotations('ferry-live', null);
      }
      return;
    }
    let bracket =
      ferryLabelTrack === NO_TRACK ? null : findTrackBracket(vessels, ferryLabelTrack, time);
    if (!bracket || bracket.speed < FERRY_MINIMUM_SPEED) {
      const pick = pickMovingFerry(vessels, ferryTracks, time, FERRY_MINIMUM_SPEED);
      if (pick) {
        ferryLabelTrack = pick.track;
        bracket = pick.bracket;
      } else {
        bracket = null;
      }
    }
    if (!bracket) {
      if (ferryLabelShown) {
        ferryLabelShown = false;
        ctx.setAnnotations('ferry-live', null);
      }
      return;
    }
    ferryLabelShown = true;
    ctx.setAnnotations('ferry-live', [
      {
        kind: 'note',
        id: 'ferry-live',
        coordinate: vessels.unproject(bracket.x, bracket.y),
        title: liveText('Staten Island Ferry, {kn:fixed:0} kn', {
          kn: bracket.speed * KNOTS_PER_METER_SECOND
        }),
        text: 'on the St. George to Whitehall run'
      }
    ]);
  }

  // ---- Selection and tooltips -------------------------------------------------------------------
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
  let zoomBand = false;

  return {
    getCompiledGraphs: () => [
      playheadCompiled,
      trailCompiled,
      metricsCompiled,
      ...(resample ? [resample.compiled] : [])
    ],

    setOption(id, value, state) {
      switch (id) {
        case 'vesselFilter':
          writeSegmentMask();
          ctx.requestLayers();
          break;
        case 'stopSpeedKnots':
          // The same threshold decides which vessels are squares.
          writeMetricsParameters();
          statusStale = true;
          ctx.requestLayers();
          break;
        case 'stopMinutes':
          writeMetricsParameters();
          statusStale = true;
          sweepStale = true;
          sweepChangedAt = frameCounter;
          break;
        case 'showStops':
          if (state.showStops) {
            sweepStale = true;
            sweepChangedAt = frameCounter;
          }
          publishStopNotes();
          ctx.requestLayers();
          break;
        case 'routeSamples':
        case 'routeSpacing':
          resample = buildResample(Number(state.routeSamples), state.routeSpacing);
          updateResampledReadout();
          if (transitTrack >= 0) writeTransitSampleIds();
          updateTransitNote();
          ctx.requestLayers();
          break;
        case 'focus':
          onFocusChanged(state);
          break;
        case 'showRoutes':
          if (value) {
            ctx.flyTo(HARBOR_OVERVIEW, {transitionMs: 1200});
          } else if (state.focus === 'transit') {
            fitTransit();
          }
          ctx.requestLayers();
          break;
        case 'time':
        case 'playing':
        case 'playbackSpeed':
        case 'loop':
        case 'maxGapMinutes':
          break;
        case 'labelFerry':
          if (!state.labelFerry) {
            ferryLabelShown = false;
            ctx.setAnnotations('ferry-live', null);
          }
          break;
        default:
          ctx.requestLayers();
      }
      updateCost();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onGroundChange(ground) {
      ctx.setLegendData('ground', ground);
      publishSpeedChart();
      publishDwellChart();
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      frameCounter = frame.frameIndex;
      // Paused, the clock sits on the slider, so every story step and deep link is deterministic.
      playhead = clock.advance(frame);
      updateActivityChart();

      if (!focusInitialised && frame.frameIndex >= 1) {
        focusInitialised = true;
        if (options.focus !== 'none') onFocusChanged(options);
      }
      // Arrows are one pixel smaller below the break zoom and one larger above.
      const band = (ctx.getViewport()?.zoom ?? ctx.getViewState().zoom) >= MARKER_ZOOM_BREAK;
      if (band !== zoomBand) {
        zoomBand = band;
        ctx.requestLayers();
      }

      if (sweep.active) {
        runSweepStep(commandEncoder);
      } else {
        if (
          sweepStale &&
          options.showStops &&
          frame.frameIndex - sweepChangedAt > SWEEP_SETTLE_FRAMES
        ) {
          startSweep();
        }
        if (metricsDirty && !sweep.active) {
          metricsCompiled.encode(commandEncoder, {parameters: undefined});
          metricsDirty = false;
          metricsReader.request(commandEncoder);
        } else {
          metricsReader.flush(commandEncoder);
        }
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

      // Playing, the summary refreshes every few frames; moved by hand, on every frame.
      if (
        statusStale ||
        frame.frameIndex % STATUS_INTERVAL_FRAMES === 0 ||
        (clock.moved && !options.playing)
      ) {
        statusReader.markStale();
        statusStale = false;
      }
      statusReader.flush(commandEncoder);

      updateFocus(playhead, frame.frameIndex);
      updateFerryLabel(playhead);
    },

    getLayers() {
      const options = ctx.options;
      const ground = ctx.ground();
      const inks = getFocusInks(ground);
      const layers: Layer[] = [];
      const focusing = options.focus !== 'none';
      // The demotion ladder: the subject is 1, everything else steps down.
      const contextOpacity = focusing || options.showStops ? 0.5 : 1;
      const markerOpacity = options.showStops ? 0.5 : focusing ? 0.3 : 1;
      const dark = ground === 'dark';

      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'harbor-backdrop',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 0.8,
            color: CONTEXT_TRACK_INK[ground],
            blending: dark ? 'additive' : 'normal',
            opacity: contextOpacity
          })
        );
      }
      if (options.showStops && anchorageEdgeCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'harbor-anchorages',
            coordinateOrigin,
            segments: anchorageOutlineBuffer,
            instanceCount: anchorageEdgeCount,
            widthPixels: 0.7,
            color: dark ? [200, 210, 235, 110] : [40, 50, 70, 130]
          })
        );
      }
      if (options.showRoutes && resample) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `harbor-fingerprint-${resample.sampleCount}`,
            coordinateOrigin,
            segments: resample.routeSegments,
            instanceCount: trackCount * (resample.sampleCount - 1),
            widthPixels: 1,
            color: inks.fingerprint,
            blending: dark ? 'additive' : 'normal'
          })
        );
      }
      if (options.showTrails) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'harbor-trails',
            coordinateOrigin,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            values: stepSpeeds,
            valueFormat: 'float32',
            valueIndices: segmentEndsBuffer,
            valueScale: KNOTS_PER_METER_SECOND,
            classBreaks: [...SHIP_SPEED_BREAKS_KNOTS],
            classColors: getShipSpeedClasses(ground, 235),
            widthPixels: 2
          })
        );
      }
      if (focusing && focus.track !== NO_TRACK) {
        layers.push(...getFocusLayers(inks));
      }
      if (options.showStops && !sweep.active) {
        layers.push(
          new StopMarkerLayer({
            id: 'harbor-stops',
            coordinateOrigin,
            centroids: stopCentroids,
            durations: stopDurations,
            drawCommands: stopDraw,
            baseRadiusPixels: STOP_BASE_RADIUS_PIXELS,
            radiusPerSqrtSecond: STOP_RADIUS_PER_SQRT_SECOND,
            maximumRadiusPixels: STOP_MAXIMUM_RADIUS_PIXELS,
            classBreaks: STOP_BREAKS_SECONDS,
            classColors: getStopClassColors(ground),
            ringColor: HEAD_HALO_INK[ground],
            ringWidthPixels: 1.2,
            opacity: 0.95
          })
        );
      }
      layers.push(getMarkerLayer('harbor-vessels', markerOpacity, ground));
      if (focusing && focus.arrowVisible && focus.track !== NO_TRACK) {
        layers.push(
          getMarkerLayer('harbor-focus-vessel', 1, ground, {
            ids: focusTrackId,
            drawCommands: focusArrowDraw,
            track: focus.track
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const track = pickVessel(event.pixel);
      if (track < 0 || !statusSnapshot) return null;
      return getVesselTooltip(track);
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
      sweepReader.stop();
      resources.destroy();
    }
  };

  /** The arrows (squares for the stopped), coloured by the step's choice. */
  function getMarkerLayer(
    id: string,
    opacity: number,
    ground: 'light' | 'dark',
    only?: {ids: Buffer; drawCommands: DrawCommandBuffer; track: number}
  ): VesselMarkerLayer {
    const options = ctx.options;
    const speedClasses = options.markerColor === 'speed';
    const palette =
      options.markerColor === 'group'
        ? getCategoryPaletteFromGroups(HARBOR_CATEGORY_TO_GROUP, ground)
        : speedClasses
          ? getShipSpeedClasses(ground)
          : [SUBJECT_INK[ground]];
    const baseSize = options.markerSize + (zoomBand ? 1 : -1);
    return new VesselMarkerLayer({
      id,
      coordinateOrigin,
      ids: only?.ids ?? activeIds,
      positions: currentPositions,
      headings,
      speeds,
      categories: categoryBuffer,
      drawCommands: only?.drawCommands ?? markerDraw,
      sizePixels: only ? baseSize + 2 : baseSize,
      colorMode: speedClasses ? 'speedClasses' : 'category',
      speedClassBreaks: SHIP_SPEED_BREAKS_KNOTS.map(knots => knots * METERS_PER_KNOT_SECOND),
      stoppedSpeed: options.stoppedSquares ? options.stopSpeedKnots * METERS_PER_KNOT_SECOND : 0,
      palette,
      categoryFilter:
        options.vesselFilter === 'all'
          ? null
          : VESSEL_CATEGORIES.indexOf(options.vesselFilter as (typeof VESSEL_CATEGORIES)[number]),
      selectedTrack: only
        ? only.track
        : selectedTrack === NO_TRACK || options.focus !== 'none'
          ? null
          : selectedTrack,
      outlineColor: HEAD_HALO_INK[ground],
      opacity
    });
  }

  /** Fixes, chords and resampled points of the vessel in focus (steps 3 and 5). */
  function getFocusLayers(inks: ReturnType<typeof getFocusInks>): Layer[] {
    const options = ctx.options;
    const layers: Layer[] = [];
    if (focusSegmentCount > 0) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: `harbor-focus-track-${options.focus}`,
          coordinateOrigin,
          segments: segmentsBuffer,
          ids: focusSegmentIds,
          instanceCount: focusSegmentCount,
          widthPixels: 1.2,
          color: inks.rawTrack
        })
      );
    }
    if (options.focus === 'transit' && resample && transitIdsSampleCount > 1) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: `harbor-focus-route-${resample.sampleCount}`,
          coordinateOrigin,
          segments: resample.routeSegments,
          ids: routeIds,
          instanceCount: transitIdsSampleCount - 1,
          widthPixels: 1.6,
          color: inks.sampleRoute
        })
      );
    }
    if (options.focus === 'ferry' && focus.bracket) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'harbor-focus-chord',
          coordinateOrigin,
          segments: segmentsBuffer,
          ids: bracketSegmentId,
          instanceCount: 1,
          widthPixels: 2.2,
          color: inks.chord,
          outlineColor: inks.halo,
          outlineWidthPixels: 1
        })
      );
    }
    if (focusFixCount > 0) {
      layers.push(
        new SpatialAnalysisPointLayer({
          id: `harbor-focus-fixes-${options.focus}`,
          coordinateOrigin,
          positions: positionsBuffer,
          ids: focusFixIds,
          instanceCount: focusFixCount,
          shape: 'ring',
          radiusPixels: 3.5,
          outlineWidthPixels: 1.5,
          color: inks.fix
        })
      );
    }
    if (options.focus === 'ferry' && focus.bracket) {
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'harbor-focus-bracket',
          coordinateOrigin,
          positions: positionsBuffer,
          ids: bracketFixIds,
          instanceCount: 2,
          shape: 'circle',
          radiusPixels: 4.5,
          outlineColor: inks.halo,
          outlineWidthPixels: 1.2,
          color: inks.bracketFix
        })
      );
    }
    if (options.focus === 'transit' && resample && transitIdsSampleCount > 0) {
      layers.push(
        new SpatialAnalysisPointLayer({
          id: `harbor-focus-samples-${resample.sampleCount}`,
          coordinateOrigin,
          positions: resample.samples,
          ids: sampleIds,
          instanceCount: transitIdsSampleCount,
          shape: 'circle',
          radiusPixels: 3.5,
          outlineColor: inks.halo,
          outlineWidthPixels: 1.2,
          color: inks.sample
        })
      );
    }
    return layers;
  }

  /** The hover card of a vessel: group, kind, speed with its class swatch, heading, length, stops. */
  function getVesselTooltip(track: number): TooltipContent {
    const snapshot = statusSnapshot;
    const category = vessels.category[track];
    const group = HARBOR_CATEGORY_TO_GROUP[category];
    const speed = (snapshot?.speeds[track] ?? 0) * KNOTS_PER_METER_SECOND;
    const underway = speed >= ctx.options.stopSpeedKnots;
    const classIndex = SHIP_SPEED_BREAKS_KNOTS.filter(limit => speed >= limit).length;
    const length = vessels.length[track];
    const stops = metricsSnapshot?.stopCounts[track];
    const [longitude, latitude] = snapshot
      ? vessels.unproject(snapshot.positions[track * 2], snapshot.positions[track * 2 + 1])
      : [0, 0];
    return {
      title: VESSEL_GROUP_LABELS[VESSEL_GROUPS[group]],
      subtitle: `${VESSEL_CATEGORY_LABELS[VESSEL_CATEGORIES[category]]} · MMSI ${vessels.mmsi[track]}`,
      rows: [
        {
          label: 'Speed',
          value: formatKnots(speed),
          swatch: getShipSpeedClasses(ctx.ground())[classIndex],
          emphasis: true
        },
        {
          label: 'Heading',
          value: underway && snapshot ? formatBearing(snapshot.headings[track]) : 'at rest'
        },
        {label: 'Length', value: length > 0 ? `${length.toFixed(0)} m` : 'not reported'},
        ...(stops === undefined ? [] : [{label: 'Stops today', value: stops}])
      ],
      highlight: {kind: 'point', coordinate: [longitude, latitude], radiusPixels: 9}
    };
  }

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

/** `072° ENE` from a heading in radians (counterclockwise from +x, the contributor's convention). */
function formatBearing(heading: number): string {
  const bearing = (((90 - (heading * 180) / Math.PI) % 360) + 360) % 360;
  const winds = [
    'N',
    'NNE',
    'NE',
    'ENE',
    'E',
    'ESE',
    'SE',
    'SSE',
    'S',
    'SSW',
    'SW',
    'WSW',
    'W',
    'WNW',
    'NW',
    'NNW'
  ];
  return `${String(Math.round(bearing) % 360).padStart(3, '0')}° ${winds[Math.round(bearing / 22.5) % 16]}`;
}

/**
 * The "two fixes and a fraction" diagram: fix A, fix B, the chord between them and the playhead at
 * the fraction the GPU's binary search and lerp return.
 */
function buildFixDiagram(bracket: TrackBracket) {
  const left = 36;
  const right = 284;
  const baseline = 44;
  const position = left + (right - left) * bracket.fraction;
  const clock = (seconds: number) => {
    const total = Math.floor(seconds);
    const pad = (value: number) => String(value).padStart(2, '0');
    return `${pad(Math.floor(total / 3600) % 24)}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
  };
  return {
    kind: 'diagram' as const,
    width: 320,
    height: 96,
    description: `Fix A at ${clock(bracket.startTime)} UTC and fix B at ${clock(bracket.endTime)} UTC with the playhead ${bracket.fraction.toFixed(2)} of the way between them.`,
    svg: [
      `<line class="diagram-muted" x1="${left}" x2="${right}" y1="${baseline}" y2="${baseline}" stroke-width="1.5" stroke-dasharray="4 3"/>`,
      `<line class="diagram-signal" x1="${left}" x2="${position.toFixed(1)}" y1="${baseline}" y2="${baseline}" stroke-width="2.5"/>`,
      `<circle class="diagram-ink" cx="${left}" cy="${baseline}" r="5"/>`,
      `<circle class="diagram-ink" cx="${right}" cy="${baseline}" r="5"/>`,
      `<polygon class="diagram-accent" points="${position.toFixed(1)},${baseline - 3} ${(position - 6).toFixed(1)},${baseline - 14} ${(position + 6).toFixed(1)},${baseline - 14}"/>`,
      `<text class="diagram-ink" x="${left}" y="${baseline + 22}" text-anchor="middle" font-size="11">fix A</text>`,
      `<text class="diagram-muted" x="${left}" y="${baseline + 35}" text-anchor="middle" font-size="10">${clock(bracket.startTime)}</text>`,
      `<text class="diagram-ink" x="${right}" y="${baseline + 22}" text-anchor="middle" font-size="11">fix B</text>`,
      `<text class="diagram-muted" x="${right}" y="${baseline + 35}" text-anchor="middle" font-size="10">${clock(bracket.endTime)}</text>`,
      `<text class="diagram-signal" x="${position.toFixed(1)}" y="${baseline - 19}" text-anchor="middle" font-size="12">${bracket.fraction.toFixed(2)}</text>`
    ].join('')
  };
}
