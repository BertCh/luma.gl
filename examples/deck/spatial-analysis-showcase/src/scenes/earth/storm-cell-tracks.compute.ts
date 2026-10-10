// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  createGPUPointDensityGaussianKernel,
  getGPUOutlineGeometryParameterValues,
  getGPUOutlineGeometryVerticesPerInput,
  getGPUTrajectoryMetricsParameterValues,
  getGPUTrajectoryPlayheadParameterValues,
  GPUOutlineGeometry,
  GPUPointDensity,
  GPUTrajectoryMetrics,
  GPUTrajectoryPlayhead,
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
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {TriangleListLayer} from '../geometry/b3-layers';
import type {SceneContext, SceneInstance} from '../scene';
import {StopMarkerLayer, VesselMarkerLayer} from '../movement/b12-layers';
import {
  binValues,
  COMPASS_COLORS,
  COMPASS_SECTORS,
  formatInteger,
  formatStormClock,
  formatStormHourMinute,
  headingToCompass,
  histogramChart,
  loadStormFlashes,
  loadStormTracks,
  quantile,
  seriesChart,
  STORM_EVENT_SECONDS,
  STORM_SPEED_RAMP_KMH,
  type StormFlashes,
  type StormTrackSet
} from './storm-data';

/** Option state of the storm-cell-tracks scene. */
export type StormCellTracksOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  colorBy: 'speed' | 'heading';
  showBackdrop: boolean;
  showTrails: boolean;
  trailMinutes: number;
  tailFade: number;
  markerSize: number;
  showArrows: boolean;
  arrowMinutes: number;
  showStalls: boolean;
  stallSpeedKmh: number;
  stallMinutes: number;
  showSwath: boolean;
  swathKm: number;
  swathMode: 'so-far' | 'event';
  swathJoin: '8' | '16' | '32';
  swathOpacity: number;
  showLightning: boolean;
  lightningMode: 'window' | 'cumulative' | 'event';
  lightningMinutes: number;
  statistic: 'count' | 'energy';
  resolution: 'coarse' | 'medium' | 'fine';
  sigma: number;
  lightningOpacity: number;
  showHotSpots: boolean;
  hotPercentile: number;
  showFlashes: boolean;
};

const KMH_PER_METER_SECOND = 3.6;
const STOP_CAPACITY = 512;
const STATUS_INTERVAL_FRAMES = 10;
const SPEED_CLASS_COLORS = [
  [255, 237, 160, 255],
  [254, 178, 76, 255],
  [240, 59, 32, 255],
  [189, 0, 38, 255],
  [103, 0, 31, 255]
] as const;
const GRID_SIZES = {coarse: [80, 52], medium: [128, 84], fine: [192, 126]} as const;
const KERNEL_RADIUS = 8;
const KERNEL_WIDTH = KERNEL_RADIUS * 2 + 1;
const SETTLE_MILLISECONDS = 400;
const NO_TRACK = 0xffffffff;
const RATE_BIN_SECONDS = 900;

type DensityGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  values: Buffer;
  extent: Buffer;
  reader: SummaryReader;
  gridSize: readonly [number, number];
};

type SwathGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  triangles: Buffer;
  triangleCount: number;
};

/**
 * Storm-cell tracks on 21-22 May 2024. Five graphs share the track buffers:
 * - metrics (`GPUTrajectoryMetrics`): per-step speed and heading in each track's own tangent
 *   plane, stalled-cell detection; run again only when the stall thresholds change;
 * - playhead (`GPUTrajectoryPlayhead`): every cell interpolated at the clock, plus a small kernel
 *   that looks up the speed and heading of the segment each cell is on and writes a motion vector;
 * - trails (`GPUTimeWindowFilter`): the track segments inside a sliding window;
 * - swath (`GPUOutlineGeometry`): the tracks buffered in meters on the sphere; the compile-time
 *   join smoothness selects between cached graphs;
 * - lightning (`GPUTimeWindowFilter` mask feeding `GPUPointDensity`): flash density in a window,
 *   following the camera; statistic and resolution select between cached graphs.
 */
export async function createStormCellTracks(
  ctx: SceneContext<StormCellTracksOptions>
): Promise<SceneInstance<StormCellTracksOptions>> {
  const {device} = ctx;
  const tracks: StormTrackSet = loadStormTracks(ctx.datasets.get('poopdeck-mrms-precip-tracks'));
  const flashes: StormFlashes = loadStormFlashes(ctx.datasets.get('poopdeck-goes-glm-lightning'));
  const {trackCount, vertexCount, segmentCount} = tracks;
  const flashCount = flashes.count;
  const resources = new SpatialAnalysisResources(device, 'storm-cells');
  const lngLatDraw = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;
  const flashOrigin: [number, number, number] = [flashes.origin[0], flashes.origin[1], 0];
  const densityProjection = ctx.datasets
    .get('poopdeck-goes-glm-lightning')
    .getProjection(flashes.origin);

  // ---- Static inputs ----------------------------------------------------------------------------
  const localBuffer = resources.createBuffer('local-positions', tracks.positions);
  const lngLatBuffer = resources.createBuffer('lng-lat', tracks.lngLat);
  const timestampsBuffer = resources.createBuffer('timestamps', tracks.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', tracks.offsets);
  const segmentsBuffer = resources.createBuffer('segments', tracks.segments);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', tracks.segmentEndVertices);
  const segmentStartTimesBuffer = resources.createBuffer(
    'segment-start-times',
    tracks.segmentStartTimes
  );
  const segmentEndTimesBuffer = resources.createBuffer('segment-end-times', tracks.segmentEndTimes);
  const flashPositionsBuffer = resources.createBuffer('flash-local', flashes.local);
  const flashLngLatBuffer = resources.createBuffer('flash-lng-lat', flashes.lngLat);
  const flashTimesBuffer = resources.createBuffer('flash-times', flashes.times);
  const flashEnergyBuffer = resources.createBuffer('flash-energy', flashes.energy);

  // ---- Metrics graph ----------------------------------------------------------------------------
  const trackLengths = resources.createBuffer('track-lengths', trackCount * 4);
  const trackDurations = resources.createBuffer('track-durations', trackCount * 4);
  const averageSpeeds = resources.createBuffer('average-speeds', trackCount * 4);
  const maximumSpeeds = resources.createBuffer('maximum-speeds', trackCount * 4);
  const stepSpeeds = resources.createBuffer('step-speeds', vertexCount * 4);
  const stepSpeedClasses = resources.createBuffer('step-speed-classes', vertexCount * 4);
  const stepHeadings = resources.createBuffer('step-headings', vertexCount * 4);
  const stepSectors = resources.createBuffer('step-sectors', vertexCount * 4);
  const trackStopCounts = resources.createBuffer('track-stop-counts', trackCount * 4);
  const stopIds = resources.createBuffer('stop-ids', STOP_CAPACITY * 4);
  const stopCount = resources.createBuffer('stop-count', 4);
  const stopOverflow = resources.createBuffer('stop-overflow', 4);
  const stopTotal = resources.createBuffer('stop-total', 4);
  const stopStartRows = resources.createBuffer('stop-start-rows', STOP_CAPACITY * 4);
  const stopEndRows = resources.createBuffer('stop-end-rows', STOP_CAPACITY * 4);
  const stopCentroidsLocal = resources.createBuffer('stop-centroids-local', STOP_CAPACITY * 8);
  const stopDurations = resources.createBuffer('stop-durations', STOP_CAPACITY * 4);
  const stopLngLat = resources.createBuffer('stop-lng-lat', STOP_CAPACITY * 8);
  const metricsParameters = resources.createParameterBuffer(
    'metrics',
    'float32',
    GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
  );
  const stopDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'storm-stop-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const metricsGraph = new GPUCommandGraph<void>(device, {id: 'storm-metrics'});
  const metricsLngLat = importGraphBuffer(
    metricsGraph,
    'lng-lat',
    lngLatBuffer,
    'float32x2',
    vertexCount
  );
  const stepSpeedsView = importGraphBuffer(
    metricsGraph,
    'step-speeds',
    stepSpeeds,
    'float32',
    vertexCount
  );
  const stepHeadingsView = importGraphBuffer(
    metricsGraph,
    'step-headings',
    stepHeadings,
    'float32',
    vertexCount
  );
  const stepSectorsView = importGraphBuffer(
    metricsGraph,
    'step-sectors',
    stepSectors,
    'uint32',
    vertexCount
  );
  const stepSpeedClassesView = importGraphBuffer(
    metricsGraph,
    'step-speed-classes',
    stepSpeedClasses,
    'uint32',
    vertexCount
  );
  const stopStartView = importGraphBuffer(
    metricsGraph,
    'stop-start-rows',
    stopStartRows,
    'uint32',
    STOP_CAPACITY
  );
  const stopEndView = importGraphBuffer(
    metricsGraph,
    'stop-end-rows',
    stopEndRows,
    'uint32',
    STOP_CAPACITY
  );
  metricsGraph.add(
    new GPUTrajectoryMetrics({
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
      id: 'metrics',
      positions: importGraphBuffer(
        metricsGraph,
        'local-positions',
        localBuffer,
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
      stepSpeeds: stepSpeedsView,
      stepHeadings: stepHeadingsView,
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
        startRows: stopStartView,
        endRows: stopEndView,
        centroids: importGraphBuffer(
          metricsGraph,
          'stop-centroids-local',
          stopCentroidsLocal,
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
  // Compass sector (0 north, clockwise, 8 sectors) of every step, for categorical coloring.
  addKernelPass(metricsGraph, {
    id: 'step-sectors',
    invocationCount: vertexCount,
    bindings: [
      {name: 'headings', view: stepHeadingsView, type: 'f32', access: 'read'},
      {name: 'sectors', view: stepSectorsView, type: 'u32', access: 'read_write'}
    ],
    body: `var compass = 90.0 - degrees(headings[headingsOffset + index]);
  compass = compass - 360.0 * floor(compass / 360.0);
  sectors[sectorsOffset + index] = u32(floor((compass + 22.5) / 45.0)) % 8u;`
  });
  addKernelPass(metricsGraph, {
    id: 'step-speed-classes',
    invocationCount: vertexCount,
    bindings: [
      {
        name: 'speeds',
        view: stepSpeedsView,
        type: 'f32',
        access: 'read'
      },
      {name: 'classes', view: stepSpeedClassesView, type: 'u32', access: 'read_write'}
    ],
    body: `let kmh = speeds[speedsOffset + index] * ${KMH_PER_METER_SECOND};
  classes[classesOffset + index] = select(4u, select(3u, select(2u, select(1u, 0u, kmh < 20.0), kmh < 40.0), kmh < 70.0), kmh < 100.0);`
  });
  // A stalled cell is drawn at the mean longitude and latitude of the rows it stalled on.
  addKernelPass(metricsGraph, {
    id: 'stall-positions',
    invocationCount: STOP_CAPACITY,
    bindings: [
      {name: 'lngLat', view: metricsLngLat, type: 'f32', access: 'read'},
      {name: 'startRows', view: stopStartView, type: 'u32', access: 'read'},
      {name: 'endRows', view: stopEndView, type: 'u32', access: 'read'},
      {
        name: 'stallPositions',
        view: importGraphBuffer(
          metricsGraph,
          'stop-lng-lat',
          stopLngLat,
          'float32x2',
          STOP_CAPACITY
        ),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let first = startRows[startRowsOffset + index];
  let last = endRows[endRowsOffset + index];
  var sum = vec2<f32>(0.0, 0.0);
  var rows = 0.0;
  if (first != 0xffffffffu && last >= first) {
    for (var row = first; row <= last; row = row + 1u) {
      sum = sum + vec2<f32>(lngLat[lngLatOffset + row * 2u], lngLat[lngLatOffset + row * 2u + 1u]);
      rows = rows + 1.0;
    }
  }
  let mean = sum / max(rows, 1.0);
  stallPositions[stallPositionsOffset + index * 2u] = mean.x;
  stallPositions[stallPositionsOffset + index * 2u + 1u] = mean.y;`
  });
  const metricsCompiled = resources.track(metricsGraph.compile());

  // ---- Playhead graph ---------------------------------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const status = resources.createBuffer('status', trackCount * 4);
  const segmentRows = resources.createBuffer('segment-rows', trackCount * 4);
  const activeIds = resources.createBuffer('active-ids', trackCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const markerSpeeds = resources.createBuffer('marker-speeds', trackCount * 4);
  const markerSpeedClasses = resources.createBuffer('marker-speed-classes', trackCount * 4);
  const selectedMotionId = resources.createBuffer('selected-motion-id', 4);
  const markerHeadings = resources.createBuffer('marker-headings', trackCount * 4);
  const markerSectors = resources.createBuffer('marker-sectors', trackCount * 4);
  const arrowSegments = resources.createBuffer('arrow-segments', trackCount * 16);
  const arrivalPositions = resources.createBuffer('motion-arrival-positions', trackCount * 8);
  const swathDimensionSegments = resources.createBuffer(
    'swath-dimension-segments',
    trackCount * 16
  );
  const swathDimensionHalfWidth = resources.createParameterBuffer(
    'swath-dimension-half-width',
    'float32',
    4
  );
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const arrowParameters = resources.createParameterBuffer('arrow', 'float32', 4);
  const markerDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'storm-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'storm-playhead'});
  const playStatusView = importGraphBuffer(playheadGraph, 'status', status, 'uint32', trackCount);
  const playSegmentRowsView = importGraphBuffer(
    playheadGraph,
    'segment-rows',
    segmentRows,
    'uint32',
    trackCount
  );
  const playCurrentView = importGraphBuffer(
    playheadGraph,
    'current-positions',
    currentPositions,
    'float32x2',
    trackCount
  );
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      id: 'playhead',
      positions: importGraphBuffer(
        playheadGraph,
        'lng-lat',
        lngLatBuffer,
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
      currentPositions: playCurrentView,
      status: playStatusView,
      segmentRows: playSegmentRowsView,
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
  // The playhead gives the segment each cell is on; its speed and heading are the step columns of
  // `GPUTrajectoryMetrics` at the segment's end row. (Two small kernels keep each under eight
  // storage buffers, the WebGPU default limit.)
  const markerSpeedsView = importGraphBuffer(
    playheadGraph,
    'marker-speeds',
    markerSpeeds,
    'float32',
    trackCount
  );
  const markerHeadingsView = importGraphBuffer(
    playheadGraph,
    'marker-headings',
    markerHeadings,
    'float32',
    trackCount
  );
  const markerSpeedClassesView = importGraphBuffer(
    playheadGraph,
    'marker-speed-classes',
    markerSpeedClasses,
    'uint32',
    trackCount
  );
  addKernelPass(playheadGraph, {
    id: 'cell-motion',
    invocationCount: trackCount,
    bindings: [
      {name: 'status', view: playStatusView, type: 'u32', access: 'read'},
      {name: 'segmentRows', view: playSegmentRowsView, type: 'u32', access: 'read'},
      {
        name: 'stepSpeeds',
        view: importGraphBuffer(playheadGraph, 'step-speeds', stepSpeeds, 'float32', vertexCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'stepHeadings',
        view: importGraphBuffer(
          playheadGraph,
          'step-headings',
          stepHeadings,
          'float32',
          vertexCount
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'stepSectors',
        view: importGraphBuffer(playheadGraph, 'step-sectors', stepSectors, 'uint32', vertexCount),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'stepSpeedClasses',
        view: importGraphBuffer(
          playheadGraph,
          'step-speed-classes',
          stepSpeedClasses,
          'uint32',
          vertexCount
        ),
        type: 'u32',
        access: 'read'
      },
      {name: 'markerSpeeds', view: markerSpeedsView, type: 'f32', access: 'read_write'},
      {name: 'markerHeadings', view: markerHeadingsView, type: 'f32', access: 'read_write'},
      {name: 'markerSpeedClasses', view: markerSpeedClassesView, type: 'u32', access: 'read_write'},
      {
        name: 'markerSectors',
        view: importGraphBuffer(
          playheadGraph,
          'marker-sectors',
          markerSectors,
          'uint32',
          trackCount
        ),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: `var speed = 0.0;
  var heading = 0.0;
  var sector = 0u;
  var speedClass = 0u;
  let row = segmentRows[segmentRowsOffset + index];
  if (status[statusOffset + index] == ${GPU_TRAJECTORY_PLAYHEAD_STATUS.active}u && row != 0xffffffffu) {
    let endRow = row + 1u;
    speed = stepSpeeds[stepSpeedsOffset + endRow];
    heading = stepHeadings[stepHeadingsOffset + endRow];
    sector = stepSectors[stepSectorsOffset + endRow];
    speedClass = stepSpeedClasses[stepSpeedClassesOffset + endRow];
  }
  markerSpeeds[markerSpeedsOffset + index] = speed;
  markerHeadings[markerHeadingsOffset + index] = heading;
  markerSectors[markerSectorsOffset + index] = sector;
  markerSpeedClasses[markerSpeedClassesOffset + index] = speedClass;`
  });
  // A cross-track bracket is generated beside the selected live cell. `swathKm` remains the
  // documented half-width, so this segment is twice that distance and stays coupled to the ribbon.
  addKernelPass(playheadGraph, {
    id: 'swath-width-dimension',
    invocationCount: trackCount,
    bindings: [
      {name: 'status', view: playStatusView, type: 'u32', access: 'read'},
      {name: 'current', view: playCurrentView, type: 'f32', access: 'read'},
      {name: 'headings', view: markerHeadingsView, type: 'f32', access: 'read'},
      {
        name: 'halfWidth',
        view: swathDimensionHalfWidth.importToGraph(playheadGraph),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'dimensions',
        view: importGraphBuffer(
          playheadGraph,
          'swath-dimension-segments',
          swathDimensionSegments,
          'float32x4',
          trackCount
        ),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var segment = vec4<f32>(nan, nan, nan, nan);
  if (status[statusOffset + index] == ${GPU_TRAJECTORY_PLAYHEAD_STATUS.active}u) {
    let lng = current[currentOffset + index * 2u];
    let lat = current[currentOffset + index * 2u + 1u];
    let half = halfWidth[halfWidthOffset];
    let heading = headings[headingsOffset + index];
    let cosLat = max(cos(radians(lat)), 0.05);
    let east = -sin(heading) * half;
    let north = cos(heading) * half;
    segment = vec4<f32>(lng - east / (111194.9 * cosLat), lat - north / 111194.9, lng + east / (111194.9 * cosLat), lat + north / 111194.9);
  }
  dimensions[dimensionsOffset + index * 4u] = segment.x;
  dimensions[dimensionsOffset + index * 4u + 1u] = segment.y;
  dimensions[dimensionsOffset + index * 4u + 2u] = segment.z;
  dimensions[dimensionsOffset + index * 4u + 3u] = segment.w;`
  });
  // A motion vector: from the cell along its heading, as far as it travels in the vector time.
  addKernelPass(playheadGraph, {
    id: 'motion-vectors',
    invocationCount: trackCount,
    bindings: [
      {name: 'status', view: playStatusView, type: 'u32', access: 'read'},
      {name: 'current', view: playCurrentView, type: 'f32', access: 'read'},
      {name: 'speeds', view: markerSpeedsView, type: 'f32', access: 'read'},
      {name: 'headings', view: markerHeadingsView, type: 'f32', access: 'read'},
      {
        name: 'arrow',
        view: arrowParameters.importToGraph(playheadGraph),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'arrows',
        view: importGraphBuffer(
          playheadGraph,
          'arrow-segments',
          arrowSegments,
          'float32',
          trackCount * 4
        ),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'arrivals',
        view: importGraphBuffer(
          playheadGraph,
          'motion-arrival-positions',
          arrivalPositions,
          'float32x2',
          trackCount
        ),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var vector = vec4<f32>(nan, nan, nan, nan);
  if (status[statusOffset + index] == ${GPU_TRAJECTORY_PLAYHEAD_STATUS.active}u) {
    let lng = current[currentOffset + index * 2u];
    let lat = current[currentOffset + index * 2u + 1u];
    let travel = speeds[speedsOffset + index] * arrow[arrowOffset];
    let heading = headings[headingsOffset + index];
    let cosLat = max(cos(radians(lat)), 0.05);
    vector = vec4<f32>(
      lng, lat,
      lng + travel * cos(heading) / (111194.9 * cosLat),
      lat + travel * sin(heading) / 111194.9
    );
  }
  arrows[arrowsOffset + index * 4u] = vector.x;
  arrows[arrowsOffset + index * 4u + 1u] = vector.y;
  arrows[arrowsOffset + index * 4u + 2u] = vector.z;
  arrows[arrowsOffset + index * 4u + 3u] = vector.w;
  arrivals[arrivalsOffset + index * 2u] = vector.z;
  arrivals[arrivalsOffset + index * 2u + 1u] = vector.w;`
  });
  const playheadCompiled = resources.track(playheadGraph.compile());

  // ---- Trail graph ------------------------------------------------------------------------------
  const trailIds = resources.createBuffer('trail-ids', segmentCount * 4);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailOverflow = resources.createBuffer('trail-overflow', 4);
  const fadeWeights = resources.createBuffer('fade-weights', segmentCount * 4);
  const clipFractions = resources.createBuffer('clip-fractions', segmentCount * 8);
  const trailParameters = resources.createParameterBuffer(
    'trail-window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'storm-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'storm-trails'});
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
      window: trailParameters.importToGraph(trailGraph),
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
      drawInstanceCount: trailGraph.importGPUData(
        'trail-draw-count',
        trailDraw.getInstanceCountData(0)
      )
    })
  );
  const trailCompiled = resources.track(trailGraph.compile());

  // ---- Swath graphs (join smoothness is compile-time) -------------------------------------------
  const swathInput = resources.createBuffer('swath-input', vertexCount * 8);
  const swathParameters = resources.createParameterBuffer('swath', 'float32', 4);
  const swathDistance = resources.createParameterBuffer('swath-distance', 'float32', 4);
  const swathGraphs = new Map<string, SwathGraph>();

  function buildSwath(join: number): SwathGraph {
    const existing = swathGraphs.get(String(join));
    if (existing) return existing;
    const verticesPerInput = getGPUOutlineGeometryVerticesPerInput(join);
    const outputCount = vertexCount * verticesPerInput;
    const triangles = resources.createBuffer(`swath-triangles-${join}`, outputCount * 8);
    const graph = new GPUCommandGraph<void>(device, {id: `storm-swath-${join}`});
    const inputView = importGraphBuffer(graph, 'swath-input', swathInput, 'float32x2', vertexCount);
    addKernelPass(graph, {
      id: 'swath-input',
      invocationCount: vertexCount,
      bindings: [
        {
          name: 'lngLat',
          view: importGraphBuffer(graph, 'lng-lat', lngLatBuffer, 'float32x2', vertexCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'times',
          view: importGraphBuffer(graph, 'timestamps', timestampsBuffer, 'float32', vertexCount),
          type: 'f32',
          access: 'read'
        },
        {name: 'limit', view: swathParameters.importToGraph(graph), type: 'f32', access: 'read'},
        {name: 'clipped', view: inputView, type: 'f32', access: 'read_write'}
      ],
      body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let visible = times[timesOffset + index] <= limit[limitOffset];
  clipped[clippedOffset + index * 2u] = select(nan, lngLat[lngLatOffset + index * 2u], visible);
  clipped[clippedOffset + index * 2u + 1u] = select(nan, lngLat[lngLatOffset + index * 2u + 1u], visible);`
    });
    graph.add(
      new GPUOutlineGeometry({
        id: 'swath',
        positions: inputView,
        geometryType: 'lines',
        pathOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
        coordinateSystem: 'spherical',
        joinSegments: join,
        parameters: swathDistance.importToGraph(graph),
        output: {
          positions: importGraphBuffer(graph, 'triangles', triangles, 'float32x2', outputCount)
        }
      })
    );
    const built = {
      compiled: resources.track(graph.compile()),
      triangles,
      triangleCount: outputCount / 3
    };
    swathGraphs.set(String(join), built);
    return built;
  }

  // ---- Flash window and density graphs ----------------------------------------------------------
  const flashIds = resources.createBuffer('flash-ids', flashCount * 4);
  const flashSelected = resources.createBuffer('flash-selected', 4);
  const flashOverflow = resources.createBuffer('flash-overflow', 4);
  const flashMask = resources.createBuffer('flash-mask', flashCount * 4);
  const flashParameters = resources.createParameterBuffer(
    'flash-window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const flashDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'storm-flash-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const flashGraph = new GPUCommandGraph<void>(device, {id: 'storm-flash-window'});
  flashGraph.add(
    new GPUTimeWindowFilter({
      id: 'flash-window',
      timestamps: importGraphBuffer(
        flashGraph,
        'flash-times',
        flashTimesBuffer,
        'float32',
        flashCount
      ),
      window: flashParameters.importToGraph(flashGraph),
      output: {
        ids: importGraphBuffer(flashGraph, 'flash-ids', flashIds, 'uint32', flashCount),
        count: importGraphBuffer(flashGraph, 'flash-selected', flashSelected, 'uint32', 1),
        overflow: importGraphBuffer(flashGraph, 'flash-overflow', flashOverflow, 'uint32', 1)
      },
      outputMask: importGraphBuffer(flashGraph, 'flash-mask', flashMask, 'uint32', flashCount),
      drawInstanceCount: flashGraph.importGPUData(
        'flash-draw-count',
        flashDraw.getInstanceCountData(0)
      )
    })
  );
  const flashCompiled = resources.track(flashGraph.compile());

  const densityBounds = resources.createParameterBuffer('density-bounds', 'float32', 4);
  const kernel = resources.createParameterBuffer(
    'density-kernel',
    'float32',
    KERNEL_WIDTH * KERNEL_WIDTH
  );
  const densitySets = new Map<string, DensityGraph>();
  let currentDensity: DensityGraph | null = null;
  let currentDensityKey = '';

  let destroyed = false;
  let statusStale = true;
  let metricsDirty = true;
  let lastChangeTime = performance.now();
  let settleStale = true;
  let lastReadTime = 0;
  let lastBounds: Float32Array | null = null;
  let hotThreshold = Number.POSITIVE_INFINITY;
  let lastDensityValues: Float32Array | null = null;
  let selectedTrack = NO_TRACK;
  let statusSnapshot: {
    count: number;
    status: Uint32Array;
    positions: Float32Array;
    speeds: Float32Array;
    sectors: Uint32Array;
    headings: Float32Array;
  } | null = null;
  let metricsSnapshot: {
    averages: Float32Array;
    maxima: Float32Array;
    durations: Float32Array;
    stopCounts: Uint32Array;
  } | null = null;

  const markChanged = () => {
    settleStale = true;
    lastChangeTime = performance.now();
  };

  function writeKernel(sigma: number): void {
    const radius = sigma > 0 ? Math.min(KERNEL_RADIUS, Math.ceil(3 * sigma)) : 0;
    const small = createGPUPointDensityGaussianKernel(radius, sigma > 0 ? sigma : undefined);
    const size = radius * 2 + 1;
    const offset = KERNEL_RADIUS - radius;
    const weights = new Float32Array(KERNEL_WIDTH * KERNEL_WIDTH);
    for (let row = 0; row < size; row++) {
      for (let column = 0; column < size; column++) {
        weights[(row + offset) * KERNEL_WIDTH + column + offset] = small[row * size + column];
      }
    }
    kernel.write(weights);
    markChanged();
  }

  function applyHotThreshold(): void {
    const values = lastDensityValues;
    if (!values) return;
    const positive: number[] = [];
    for (const value of values) if (value > 0) positive.push(value);
    if (positive.length < 8) {
      hotThreshold = Number.POSITIVE_INFINITY;
      ctx.setReadout('hotCells', 'none yet');
      ctx.requestLayers();
      return;
    }
    positive.sort((a, b) => a - b);
    const fraction = ctx.options.hotPercentile / 100;
    hotThreshold = positive[Math.min(positive.length - 1, Math.floor(fraction * positive.length))];
    let hot = 0;
    for (const value of positive) if (value > hotThreshold) hot++;
    ctx.setReadout(
      'hotCells',
      `${formatInteger(hot)} of ${formatInteger(positive.length)} active cells`
    );
    ctx.requestLayers();
  }

  function buildDensity(options: StormCellTracksOptions): DensityGraph {
    const key = `${options.statistic}:${options.resolution}`;
    const existing = densitySets.get(key);
    if (existing) return existing;
    const gridSize = GRID_SIZES[options.resolution];
    const cellCount = gridSize[0] * gridSize[1];
    const weighted = options.statistic === 'energy';
    const values = resources.createBuffer(`density-values-${key}`, cellCount * 4);
    const extent = resources.createBuffer(`density-extent-${key}`, 8);
    const histogram = resources.createBuffer(`density-histogram-${key}`, 16 * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `storm-density-${key}`});
    graph.add(
      new GPUPointDensity({
        id: 'flash-density',
        positions: importGraphBuffer(
          graph,
          'flash-local',
          flashPositionsBuffer,
          'float32x2',
          flashCount
        ),
        mask: importGraphBuffer(graph, 'flash-mask', flashMask, 'uint32', flashCount),
        ...(weighted
          ? {
              weights: importGraphBuffer(
                graph,
                'flash-energy',
                flashEnergyBuffer,
                'float32',
                flashCount
              )
            }
          : {}),
        bounds: densityBounds.importToGraph(graph),
        gridSize,
        binning: 'grid',
        statistic: weighted ? 'sum' : 'count',
        smoothing: {
          kernel: kernel.importToGraph(graph),
          kernelWidth: KERNEL_WIDTH,
          kernelHeight: KERNEL_WIDTH,
          strategy: 'direct'
        },
        output: {
          values: importGraphBuffer(graph, 'values', values, 'float32', cellCount),
          extent: importGraphBuffer(graph, 'extent', extent, 'float32', 2),
          histogram: importGraphBuffer(graph, 'histogram', histogram, 'uint32', 16)
        }
      })
    );
    const reader = new SummaryReader(
      resources,
      `density-${key}`,
      [
        {buffer: extent, size: 8},
        {buffer: values, size: cellCount * 4}
      ],
      bytes => {
        if (destroyed || currentDensity?.values !== values) return;
        const [low, high] = new Float32Array(bytes, 0, 2);
        ctx.setLegendExtent('flashes', [low, high]);
        ctx.setReadout('peakDensity', high);
        lastDensityValues = new Float32Array(bytes, 8, cellCount).slice();
        applyHotThreshold();
      }
    );
    const built: DensityGraph = {
      compiled: resources.track(graph.compile()),
      values,
      extent,
      reader,
      gridSize
    };
    densitySets.set(key, built);
    return built;
  }

  function selectDensity(): void {
    const key = `${ctx.options.statistic}:${ctx.options.resolution}`;
    if (key === currentDensityKey && currentDensity) return;
    currentDensity = buildDensity(ctx.options);
    currentDensityKey = key;
    lastDensityValues = null;
    ctx.setReadout('grid', `${currentDensity.gridSize[0]} x ${currentDensity.gridSize[1]} cells`);
    markChanged();
  }

  // ---- Initial writes ---------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, STORM_EVENT_SECONDS], rate: 60, step: 300}
  );
  let playhead = ctx.options.time;

  function writeMetricsParameters(): void {
    metricsParameters.write(
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: ctx.options.stallSpeedKmh / KMH_PER_METER_SECOND,
        stopMinimumDuration: ctx.options.stallMinutes * 60
      })
    );
    metricsDirty = true;
  }
  function writeArrowParameters(): void {
    arrowParameters.write(Float32Array.of(ctx.options.arrowMinutes * 60, 0, 0, 0));
  }
  function writeSwathDistance(): void {
    swathDistance.write(
      getGPUOutlineGeometryParameterValues({distance: ctx.options.swathKm * 1000})
    );
    swathDimensionHalfWidth.write(Float32Array.of(ctx.options.swathKm * 1000, 0, 0, 0));
    ctx.setReadout(
      'swathWidth',
      `${(ctx.options.swathKm * 2).toLocaleString('en-US')} km full width (±${ctx.options.swathKm.toLocaleString('en-US')} km)`
    );
  }
  writeMetricsParameters();
  writeArrowParameters();
  writeSwathDistance();
  writeKernel(ctx.options.sigma);
  selectDensity();
  buildSwath(Number(ctx.options.swathJoin));

  ctx.setReadout(
    'tracks',
    `${formatInteger(trackCount)} cells, ${formatInteger(vertexCount)} fixes`
  );
  ctx.setReadout(
    'flashes',
    `${formatCount(flashCount)} of ${formatCount(flashes.flashesInWindow)} flashes (${(flashes.sampleFraction * 100).toFixed(0)}% sample)`
  );

  // Flash rate per 15 minutes, scaled up from the sample, and tracks alive per bin (static).
  const rateBins = Math.ceil(STORM_EVENT_SECONDS / RATE_BIN_SECONDS);
  const flashRate = new Float64Array(rateBins);
  for (let flash = 0; flash < flashCount; flash++) {
    flashRate[Math.min(rateBins - 1, Math.floor(flashes.times[flash] / RATE_BIN_SECONDS))]++;
  }
  const rateScale = 1 / flashes.sampleFraction / (RATE_BIN_SECONDS / 60);
  const rateX = Array.from({length: rateBins}, (_, bin) => ((bin + 0.5) * RATE_BIN_SECONDS) / 3600);
  const rateY = Array.from(flashRate, count => count * rateScale);
  let rateMarker = -1;
  function updateRateChart(force = false): void {
    const marker = Math.round(playhead / 300) * 300;
    if (!force && marker === rateMarker) return;
    rateMarker = marker;
    ctx.setChart(
      'rateChart',
      seriesChart([{label: 'flashes per minute', x: rateX, y: rateY, area: true}], {
        xLabel: 'hours after 12:00 UTC on 21 May',
        yLabel: 'flashes / min',
        xDomain: [0, STORM_EVENT_SECONDS / 3600],
        markers: [{x: marker / 3600, label: 'now'}],
        formatX: value => `${Math.round(value)}`,
        formatY: value => formatInteger(value),
        description:
          'Estimated GOES-16 lightning flashes per minute over the US in 15-minute bins; a line marks the playhead.'
      })
    );
  }
  updateRateChart(true);

  // ---- Readbacks --------------------------------------------------------------------------------
  const statusReader = new SummaryReader(
    resources,
    'storm-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: flashSelected, size: 4},
      {buffer: trailCount, size: 4},
      {buffer: status, size: trackCount * 4},
      {buffer: currentPositions, size: trackCount * 8},
      {buffer: markerSpeeds, size: trackCount * 4},
      {buffer: markerSectors, size: trackCount * 4},
      {buffer: markerHeadings, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const statusStart = 3;
      const positionStart = statusStart + trackCount;
      const speedStart = positionStart + trackCount * 2;
      const sectorStart = speedStart + trackCount;
      const headingStart = sectorStart + trackCount;
      statusSnapshot = {
        count: words[0],
        status: words.slice(statusStart, statusStart + trackCount),
        positions: floats.slice(positionStart, speedStart),
        speeds: floats.slice(speedStart, sectorStart),
        sectors: words.slice(sectorStart, headingStart),
        headings: floats.slice(headingStart, headingStart + trackCount)
      };
      const motionTrack =
        selectedTrack === NO_TRACK
          ? statusSnapshot.status.findIndex(
              value => value === GPU_TRAJECTORY_PLAYHEAD_STATUS.active
            )
          : selectedTrack;
      selectedMotionId.write(Uint32Array.of(motionTrack >= 0 ? motionTrack : 0));
      ctx.setReadout('activeCells', words[0]);
      ctx.setReadout('flashesNow', words[1]);
      ctx.setReadout('trailSegments', words[2]);
      let speedSum = 0;
      let active = 0;
      for (let track = 0; track < trackCount; track++) {
        if (statusSnapshot.status[track] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active) continue;
        speedSum += statusSnapshot.speeds[track] * KMH_PER_METER_SECOND;
        active++;
      }
      ctx.setReadout('activeSpeed', active ? `${(speedSum / active).toFixed(0)} km/h` : 'no cells');
      updateRateChart();
      describeSelection();
    }
  );

  const metricsReader = new SummaryReader(
    resources,
    'storm-metrics',
    [
      {buffer: stopCount, size: 4},
      {buffer: stopOverflow, size: 4},
      {buffer: stopTotal, size: 4},
      {buffer: stepSpeeds, size: vertexCount * 4},
      {buffer: stepHeadings, size: vertexCount * 4},
      {buffer: averageSpeeds, size: trackCount * 4},
      {buffer: maximumSpeeds, size: trackCount * 4},
      {buffer: trackStopCounts, size: trackCount * 4},
      {buffer: stopDurations, size: STOP_CAPACITY * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const speedStart = 3;
      const headingStart = speedStart + vertexCount;
      const averageStart = headingStart + vertexCount;
      const maximumStart = averageStart + trackCount;
      const stopCountStart = maximumStart + trackCount;
      const durationStart = stopCountStart + trackCount;
      metricsSnapshot = {
        averages: floats.slice(averageStart, maximumStart),
        maxima: floats.slice(maximumStart, stopCountStart),
        stopCounts: words.slice(stopCountStart, durationStart),
        durations: floats.slice(durationStart, durationStart + STOP_CAPACITY)
      };
      // Every step except the first row of each track.
      const stepKmh: number[] = [];
      const sectorCounts = new Float64Array(8);
      for (let track = 0; track < trackCount; track++) {
        for (let row = tracks.offsets[track] + 1; row < tracks.offsets[track + 1]; row++) {
          const kmh = floats[speedStart + row] * KMH_PER_METER_SECOND;
          stepKmh.push(kmh);
          if (kmh >= 1) {
            const compass = headingToCompass(floats[headingStart + row]);
            sectorCounts[Math.floor((compass + 22.5) / 45) % 8]++;
          }
        }
      }
      ctx.setReadout(
        'medianSpeed',
        `${quantile(stepKmh, 0.5).toFixed(0)} km/h (90% of steps under ${quantile(stepKmh, 0.9).toFixed(0)})`
      );
      ctx.setReadout('fastest', `${Math.max(...stepKmh).toFixed(0)} km/h`);
      ctx.setChart(
        'speedChart',
        histogramChart(binValues(stepKmh, 0, 120, 24), 0, 120, {
          xLabel: 'speed of a 10-minute step (km/h)',
          yLabel: 'steps',
          formatX: value => value.toFixed(0),
          description: 'Histogram of the speed of every step of every storm-cell track.'
        })
      );
      ctx.setChart('roseChart', {
        kind: 'bars',
        values: sectorCounts,
        labels: [...COMPASS_SECTORS],
        height: 110,
        yLabel: 'steps',
        highlight: [sectorCounts.indexOf(Math.max(...sectorCounts))],
        description:
          'Number of steps heading toward each compass sector, from GPUTrajectoryMetrics step headings.'
      });
      const stalls = Math.min(words[0], STOP_CAPACITY);
      let longest = 0;
      for (let stop = 0; stop < stalls; stop++) {
        longest = Math.max(longest, metricsSnapshot.durations[stop]);
      }
      let stalledTracks = 0;
      for (let track = 0; track < trackCount; track++) {
        if (metricsSnapshot.stopCounts[track] > 0) stalledTracks++;
      }
      ctx.setReadout('stalls', `${formatInteger(words[2])}${words[1] ? ' (list truncated)' : ''}`);
      ctx.setReadout(
        'stalledCells',
        `${formatInteger(stalledTracks)} of ${formatInteger(trackCount)} cells`
      );
      ctx.setReadout('longestStall', longest ? `${(longest / 60).toFixed(0)} min` : 'none');
      describeSelection();
    }
  );

  function describeCell(track: number): string {
    const parts = [`Cell ${track + 1}`];
    const first = tracks.offsets[track];
    const last = tracks.offsets[track + 1] - 1;
    parts.push(
      `${formatStormHourMinute(tracks.timestamps[first])} to ${formatStormHourMinute(tracks.timestamps[last])} UTC`
    );
    parts.push(`peak ${tracks.peakDbz[track].toFixed(0)} dBZ`);
    return parts.join(', ');
  }

  function describeSelection(): void {
    if (selectedTrack === NO_TRACK) {
      ctx.setReadout('selected', 'click a cell');
      return;
    }
    let text = describeCell(selectedTrack);
    if (metricsSnapshot) {
      text += `; mean ${(metricsSnapshot.averages[selectedTrack] * KMH_PER_METER_SECOND).toFixed(0)} km/h, max ${(metricsSnapshot.maxima[selectedTrack] * KMH_PER_METER_SECOND).toFixed(0)} km/h, ${metricsSnapshot.stopCounts[selectedTrack]} stalls`;
    }
    ctx.setReadout('selected', text);
  }
  describeSelection();

  function pickCell(pixel: readonly [number, number]): number {
    const viewport = ctx.getViewport();
    const snapshot = statusSnapshot;
    if (!viewport || !snapshot) return -1;
    let best = -1;
    let bestDistance = 16 * 16;
    for (let track = 0; track < trackCount; track++) {
      if (snapshot.status[track] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active) continue;
      const [x, y] = viewport.project([
        snapshot.positions[track * 2],
        snapshot.positions[track * 2 + 1]
      ]);
      const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
      if (squared < bestDistance) {
        bestDistance = squared;
        best = track;
      }
    }
    return best;
  }

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => {
      const swath = swathGraphs.get(ctx.options.swathJoin);
      return [
        metricsCompiled,
        playheadCompiled,
        trailCompiled,
        ...(swath ? [swath.compiled] : []),
        flashCompiled,
        ...(currentDensity ? [currentDensity.compiled] : [])
      ] as unknown as CompiledGPUCommandGraph<never>[];
    },

    setOption(id, _value, state) {
      switch (id) {
        case 'stallSpeedKmh':
        case 'stallMinutes':
          writeMetricsParameters();
          break;
        case 'arrowMinutes':
          writeArrowParameters();
          break;
        case 'swathKm':
          writeSwathDistance();
          break;
        case 'swathJoin':
          buildSwath(Number(state.swathJoin));
          ctx.requestLayers();
          break;
        case 'sigma':
          writeKernel(state.sigma);
          break;
        case 'statistic':
        case 'resolution':
          selectDensity();
          ctx.requestLayers();
          break;
        case 'lightningMode':
        case 'lightningMinutes':
          markChanged();
          break;
        case 'hotPercentile':
          applyHotThreshold();
          break;
        case 'time':
        case 'play':
        case 'speed':
        case 'loop':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onGroundChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      playhead = clock.advance(frame);
      ctx.setReadout('clock', formatStormClock(playhead));

      if (metricsDirty) {
        metricsCompiled.encode(commandEncoder, {parameters: undefined});
        metricsDirty = false;
        metricsReader.request(commandEncoder);
      } else {
        metricsReader.flush(commandEncoder);
      }

      playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: 0}));
      playheadCompiled.encode(commandEncoder, {parameters: undefined});

      if (options.showTrails) {
        const trailSeconds = options.trailMinutes * 60;
        trailParameters.write(
          getGPUTimeWindowParameterValues({
            start: playhead - trailSeconds,
            end: playhead,
            startFadeDuration: trailSeconds * options.tailFade
          })
        );
        trailCompiled.encode(commandEncoder, {parameters: undefined});
      }

      if (options.showSwath) {
        const swath = buildSwath(Number(options.swathJoin));
        swathParameters.write(
          Float32Array.of(
            options.swathMode === 'so-far' ? playhead : STORM_EVENT_SECONDS + 1,
            0,
            0,
            0
          )
        );
        swath.compiled.encode(commandEncoder, {parameters: undefined});
      }

      if (options.showLightning || options.showFlashes) {
        const windowSeconds = options.lightningMinutes * 60;
        const start = options.lightningMode === 'window' ? playhead - windowSeconds : 0;
        const end = options.lightningMode === 'event' ? STORM_EVENT_SECONDS : playhead;
        flashParameters.write(getGPUTimeWindowParameterValues({start, end}));
        flashCompiled.encode(commandEncoder, {parameters: undefined});
      }

      if (options.showLightning && currentDensity) {
        const viewBounds = getViewportMetricBounds(frame.viewport, densityProjection);
        const boundsData = Float32Array.from(viewBounds);
        if (!lastBounds || boundsData.some((value, index) => value !== lastBounds![index])) {
          markChanged();
        }
        if (clock.moved || options.play) markChanged();
        lastBounds = boundsData;
        densityBounds.write(boundsData);
        ctx.setReadout(
          'cellSize',
          `${((viewBounds[2] - viewBounds[0]) / currentDensity.gridSize[0] / 1000).toFixed(1)} km`
        );
        currentDensity.compiled.encode(commandEncoder, {parameters: undefined});
        // Read the cells when the picture has settled, or every 600 ms while it keeps changing.
        const now = performance.now();
        const settled = now - lastChangeTime > SETTLE_MILLISECONDS || now - lastReadTime > 600;
        if (settleStale && settled && !currentDensity.reader.isPending) {
          currentDensity.reader.request(commandEncoder);
          lastReadTime = now;
          settleStale = false;
        } else {
          currentDensity.reader.flush(commandEncoder);
        }
      }

      if (statusStale || frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) {
        statusReader.markStale();
        statusStale = false;
      }
      statusReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.ground() === 'dark';
      const layers: Layer[] = [];
      const bySpeed = options.colorBy === 'speed';

      if (options.showSwath) {
        const swath = swathGraphs.get(options.swathJoin);
        if (swath) {
          layers.push(
            new TriangleListLayer({
              id: `storm-swath-${options.swathJoin}`,
              coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
              positions: swath.triangles,
              triangleCount: swath.triangleCount,
              color: dark ? [205, 174, 255, 255] : [94, 66, 140, 255],
              opacity: options.swathOpacity
            })
          );
        }
        const motionTrack =
          selectedTrack === NO_TRACK
            ? (statusSnapshot?.status.findIndex(
                value => value === GPU_TRAJECTORY_PLAYHEAD_STATUS.active
              ) ?? -1)
            : selectedTrack;
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'storm-swath-width-bracket',
            ...lngLatDraw,
            segments: swathDimensionSegments,
            ids: selectedMotionId,
            instanceCount: motionTrack >= 0 ? 1 : 0,
            widthPixels: 2.2,
            cap: 'square',
            color: dark ? [255, 255, 255, 240] : [25, 28, 42, 240]
          })
        );
      }

      if (options.showLightning && currentDensity) {
        const smoothed = options.sigma > 0;
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `storm-flash-density-${currentDensityKey}`,
            coordinateOrigin: flashOrigin,
            gridSize: currentDensity.gridSize,
            bounds: densityBounds.buffer,
            binning: 'grid',
            values: currentDensity.values,
            valueFormat: 'float32',
            extent: currentDensity.extent,
            colormap: 'inferno',
            classBreaks: [1, 5, 15, 40],
            classColors: [
              [255, 237, 160, 190],
              [254, 178, 76, 205],
              [240, 59, 32, 220],
              [189, 0, 38, 235],
              [103, 0, 31, 245]
            ],
            sqrtScale: true,
            discardAtOrBelow: smoothed ? 0.02 : 0,
            tessellation: 48,
            color: [255, 255, 255, Math.round(options.lightningOpacity * 255)]
          })
        );
        if (options.showHotSpots && Number.isFinite(hotThreshold)) {
          layers.push(
            new SpatialAnalysisRasterLayer({
              id: `storm-hot-spots-${currentDensityKey}`,
              coordinateOrigin: flashOrigin,
              gridSize: currentDensity.gridSize,
              bounds: densityBounds.buffer,
              binning: 'grid',
              values: currentDensity.values,
              valueFormat: 'float32',
              colormap: 'uniform',
              discardAtOrBelow: hotThreshold,
              tessellation: 48,
              color: dark ? [90, 235, 255, 150] : [0, 140, 190, 140]
            })
          );
        }
      }

      if (options.showFlashes) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'storm-flash-points',
            ...lngLatDraw,
            positions: flashLngLatBuffer,
            ids: flashIds,
            drawCommands: flashDraw,
            radiusPixels: 1.2,
            color: dark ? [255, 240, 150, 150] : [150, 90, 0, 150]
          })
        );
      }

      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'storm-backdrop',
            ...lngLatDraw,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            color: dark ? [200, 210, 230, 38] : [50, 60, 80, 46]
          })
        );
      }

      if (options.showTrails) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `storm-trails-${options.colorBy}`,
            ...lngLatDraw,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            values: bySpeed ? stepSpeedClasses : stepSectors,
            valueFormat: 'uint32',
            valueIndices: segmentEndsBuffer,
            colormap: 'category',
            palette: bySpeed ? SPEED_CLASS_COLORS : COMPASS_COLORS,
            widthPixels: 2.6
          })
        );
      }

      if (options.showArrows) {
        const motionTrack =
          selectedTrack === NO_TRACK
            ? (statusSnapshot?.status.findIndex(
                value => value === GPU_TRAJECTORY_PLAYHEAD_STATUS.active
              ) ?? -1)
            : selectedTrack;
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'storm-selected-motion-vector',
            ...lngLatDraw,
            segments: arrowSegments,
            ids: selectedMotionId,
            instanceCount: motionTrack >= 0 ? 1 : 0,
            widthPixels: 1.8,
            dashArray: [6, 4],
            color: dark ? [255, 255, 255, 190] : [20, 24, 32, 200]
          }),
          new SpatialAnalysisPointLayer({
            id: 'storm-vector-arrival-ring',
            ...lngLatDraw,
            positions: arrivalPositions,
            ids: selectedMotionId,
            instanceCount: motionTrack >= 0 ? 1 : 0,
            radiusPixels: options.markerSize + 5,
            shape: 'ring',
            color: dark ? [255, 255, 255, 235] : [20, 24, 32, 235],
            outlineWidthPixels: 1.6
          })
        );
      }

      if (options.showStalls) {
        layers.push(
          new StopMarkerLayer({
            id: 'storm-stalls',
            ...lngLatDraw,
            centroids: stopLngLat,
            durations: stopDurations,
            drawCommands: stopDraw,
            baseRadiusPixels: 4,
            radiusPerSqrtSecond: 0.1,
            maximumRadiusPixels: 16,
            durationForFullColor: 3 * 3600,
            opacity: 0.9
          })
        );
      }

      layers.push(
        new VesselMarkerLayer({
          id: 'storm-cells',
          ...lngLatDraw,
          ids: activeIds,
          positions: currentPositions,
          headings: markerHeadings,
          speeds: markerSpeeds,
          categories: bySpeed ? markerSpeedClasses : markerSectors,
          drawCommands: markerDraw,
          sizePixels: options.markerSize,
          colorMode: 'category',
          ramp: 'inferno',
          speedForFullColor: STORM_SPEED_RAMP_KMH / KMH_PER_METER_SECOND,
          palette: bySpeed ? SPEED_CLASS_COLORS : COMPASS_COLORS,
          categoryFilter: null,
          selectedTrack: selectedTrack === NO_TRACK ? null : selectedTrack,
          outlineColor: dark ? [8, 10, 16, 235] : [20, 24, 32, 215]
        })
      );
      return layers;
    },

    getTooltip(event) {
      const track = pickCell(event.pixel);
      if (track < 0 || !statusSnapshot) return null;
      const speed = statusSnapshot.speeds[track] * KMH_PER_METER_SECOND;
      const compass = headingToCompass(statusSnapshot.headings[track]);
      return `${describeCell(track)}: moving ${speed.toFixed(0)} km/h toward ${COMPASS_SECTORS[statusSnapshot.sectors[track] % 8]} (${compass.toFixed(0)} degrees)`;
    },

    onClick(event) {
      const track = pickCell(event.pixel);
      selectedTrack = track < 0 || track === selectedTrack ? NO_TRACK : track;
      describeSelection();
      ctx.requestLayers();
      return track >= 0;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      metricsReader.stop();
      for (const set of densitySets.values()) set.reader.stop();
      resources.destroy();
    }
  };
}
