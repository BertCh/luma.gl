// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer, type Viewport} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPULineDensityParameterValues,
  getGPUSpatialClusteringParameterValues,
  getGPUTrajectoryMetricsParameterValues,
  getGPUTrajectoryPlayheadParameterValues,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPU_SPATIAL_CLUSTERING_NOISE,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS,
  GPULineDensity,
  GPUSpatialClustering,
  GPUTrajectoryMetrics,
  GPUTrajectoryPlayhead,
  GPUZoneEvents
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {createPlaybackClock} from '../../engine/playback';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {StopMarkerLayer, VesselMarkerLayer, ZoneEventMarkerLayer} from './b12-layers';
import {KNOTS_PER_METER_SECOND, METERS_PER_KNOT_SECOND} from './b12-tracks';
import {
  buildSpeedHistograms,
  countGateCrossings,
  formatHours,
  getHistogramMedian,
  rankAnchorages,
  SPEED_BIN_COUNT,
  type Anchorage,
  type GateCrossings,
  type SpeedHistograms
} from './us-shipping.stats';
import {
  AZIMUTHAL_EARTH_RADIUS_METERS,
  buildGateEdges,
  formatShippingClock,
  loadShippingTracks,
  SHIPPING_GATES,
  SHIPPING_PROJECTION_CENTER,
  SHIPPING_TYPE_COLORS,
  SHIPPING_TYPE_LABELS,
  SHIPPING_TYPES,
  type GateEdges
} from './us-shipping.tracks';

/** Option state of the `us-shipping-day` scene. */
export type UsShippingOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  vesselFilter: string;
  showVessels: boolean;
  markerColor: 'category' | 'speed';
  markerSize: number;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  showBackdrop: boolean;
  showTrails: boolean;
  trailMinutes: number;
  tailFade: number;
  trailColor: 'category' | 'speed';
  maxGapMinutes: number;
  showDensity: boolean;
  densityType: string;
  densityGrid: '512' | '1024' | '2048';
  densityExtent: 'us' | 'view';
  densityValue: 'length' | 'density';
  densityCeiling: number;
  densityOpacity: number;
  showGates: boolean;
  gateHalfWidth: number;
  eventsPerTrack: '8' | '16' | '32';
  showCrossings: boolean;
  pulseMinutes: number;
  showRestingCrossings: boolean;
  gateFocus: string;
  showStops: boolean;
  stopSpeedKnots: number;
  stopMinutes: number;
  showAnchorages: boolean;
  clusterRadiusKm: number;
  clusterMinStops: number;
  rankBy: 'dwell' | 'stops';
  speedSource: 'derived' | 'reported';
};

const STOP_CAPACITY = 16384;
const CLUSTER_CAPACITY = 1024;
const EVENT_CAPACITY = 32768;
const CANDIDATE_CAPACITY = 1 << 18;
const CLUSTER_GRID: readonly [number, number] = [512, 512];
const STATUS_INTERVAL_FRAMES = 12;
const NO_TRACK = 0xffffffff;
const SPEED_RAMP_KNOTS = 25;
const NAUTICAL_MILE_METERS = 1852;
/** Longitude and latitude extent of the whole-country density grid. */
const US_DENSITY_BOUNDS: readonly [number, number, number, number] = [-127.5, 24, -65.5, 49.6];
const GRID_ROWS: Record<UsShippingOptions['densityGrid'], [number, number]> = {
  '512': [512, 256],
  '1024': [1024, 512],
  '2048': [2048, 1024]
};
const VIEW_SETTLE_FRAMES = 10;
const RANK_LIST_LENGTH = 8;

type DensityBuffers = {
  lengths: Buffer;
  densities: Buffer;
  overflow: Buffer;
  totalRecords: Buffer;
  reader: SummaryReader;
  cells: number;
};

type DensityGraph = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  buffers: DensityBuffers;
  columns: number;
  rows: number;
};

type ZoneGraph = {
  eventsPerTrack: number;
  compiled: CompiledGPUCommandGraph<void>;
};

type DensityStats = {
  maximumLength: number;
  maximumDensity: number;
  percentiles: Float32Array;
  nonEmpty: number;
  cells: number;
  pieces: number;
  overflow: boolean;
  totalKilometers: number;
};

/**
 * One US day of AIS in one scene. Four independent graphs read the same track buffers: the playhead
 * and time-window graphs animate the fleet, a metrics graph finds stops and a clustering graph groups
 * them into anchorages, a zone-event graph counts gate crossings and a line-density graph paints the
 * traffic corridors. Analysis runs in azimuthal-equidistant meters, the playhead in Web Mercator
 * meters (so headings stay true on the map), and everything is drawn from longitude and latitude.
 */
export async function createUsShippingDay(
  ctx: SceneContext<UsShippingOptions>
): Promise<SceneInstance<UsShippingOptions>> {
  const tracks = loadShippingTracks(ctx.datasets.get('poopdeck-ais-us'));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = tracks;
  const resources = new SpatialAnalysisResources(device, 'us-shipping');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;
  const gateCount = SHIPPING_GATES.length;
  let destroyed = false;

  ctx.setStatus('Uploading tracks');
  // ---- Static inputs ----------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', tracks.positions);
  const mercatorBuffer = resources.createBuffer('mercator', tracks.mercator);
  const lngLatBuffer = resources.createBuffer('lng-lat', tracks.lngLat);
  const timestampsBuffer = resources.createBuffer('timestamps', tracks.timestamps);
  const offsetsBuffer = resources.createBuffer('track-offsets', tracks.offsets);
  const categoryBuffer = resources.createBuffer('category', tracks.category);
  const trackStartBuffer = resources.createBuffer('track-starts', tracks.trackStartTimes);
  const segmentsBuffer = resources.createBuffer('segments', tracks.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', tracks.segmentTracks);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', tracks.segmentEndVertices);
  const segmentStartTimesBuffer = resources.createBuffer(
    'segment-start-times',
    tracks.segmentStartTimes
  );
  const segmentEndTimesBuffer = resources.createBuffer('segment-end-times', tracks.segmentEndTimes);
  const segmentMaskBuffer = resources.createBuffer(
    'segment-mask',
    new Uint32Array(segmentCount).fill(1)
  );

  // ---- Playhead graph (Web Mercator meters, then longitude and latitude for drawing) ------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const headings = resources.createBuffer('headings', trackCount * 4);
  const rawSpeeds = resources.createBuffer('raw-speeds', trackCount * 4);
  const status = resources.createBuffer('status', trackCount * 4);
  const activeIds = resources.createBuffer('active-ids', trackCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const markerLngLat = resources.createBuffer('marker-lng-lat', trackCount * 8);
  const markerSpeeds = resources.createBuffer('marker-speeds', trackCount * 4);
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const markerDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'us-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'us-playhead'});
  const currentPositionsView = importGraphBuffer(
    playheadGraph,
    'current-positions',
    currentPositions,
    'float32x2',
    trackCount
  );
  const rawSpeedsView = importGraphBuffer(
    playheadGraph,
    'raw-speeds',
    rawSpeeds,
    'float32',
    trackCount
  );
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      id: 'playhead',
      positions: importGraphBuffer(
        playheadGraph,
        'mercator',
        mercatorBuffer,
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
      currentPositions: currentPositionsView,
      headings: importGraphBuffer(playheadGraph, 'headings', headings, 'float32', trackCount),
      speeds: rawSpeedsView,
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
  // Mercator meters (relative to the projection origin) to longitude and latitude, and the
  // Mercator speed to a true ground speed (Mercator stretches distances by 1 / cos(latitude)).
  addKernelPass(playheadGraph, {
    id: 'marker-lng-lat',
    invocationCount: trackCount,
    declarations: `const MERCATOR_ORIGIN: vec2<f32> = vec2<f32>(${tracks.mercatorOrigin[0].toFixed(3)}, ${tracks.mercatorOrigin[1].toFixed(3)});
const MERCATOR_RADIUS: f32 = 6378137.0;
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;`,
    bindings: [
      {name: 'mercator', view: currentPositionsView, type: 'f32', access: 'read'},
      {name: 'rawSpeeds', view: rawSpeedsView, type: 'f32', access: 'read'},
      {
        name: 'lngLat',
        view: importGraphBuffer(
          playheadGraph,
          'marker-lng-lat',
          markerLngLat,
          'float32x2',
          trackCount
        ),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'trueSpeeds',
        view: importGraphBuffer(
          playheadGraph,
          'marker-speeds',
          markerSpeeds,
          'float32',
          trackCount
        ),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let mx = mercator[mercatorOffset + index * 2u] + MERCATOR_ORIGIN.x;
  let my = mercator[mercatorOffset + index * 2u + 1u] + MERCATOR_ORIGIN.y;
  let longitude = mx / MERCATOR_RADIUS * RADIANS_TO_DEGREES;
  let latitudeRadians = 2.0 * atan(exp(my / MERCATOR_RADIUS)) - 1.5707963267948966;
  lngLat[lngLatOffset + index * 2u] = longitude;
  lngLat[lngLatOffset + index * 2u + 1u] = latitudeRadians * RADIANS_TO_DEGREES;
  trueSpeeds[trueSpeedsOffset + index] = rawSpeeds[rawSpeedsOffset + index] * cos(latitudeRadians);`
  });
  const playheadCompiled = resources.track(playheadGraph.compile());

  // ---- Trail (time window) graph ----------------------------------------------------------------
  ctx.setStatus('Compiling the playback graphs');
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
      id: 'us-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'us-trails'});
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

  // ---- Metrics graph: step speeds, stops, stop positions for drawing ----------------------------
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
  const stopStartRows = resources.createBuffer('stop-start-rows', STOP_CAPACITY * 4);
  const stopEndRows = resources.createBuffer('stop-end-rows', STOP_CAPACITY * 4);
  const stopLngLat = resources.createBuffer('stop-lng-lat', STOP_CAPACITY * 8);
  const clusterPoints = resources.createBuffer('cluster-points', STOP_CAPACITY * 8);
  const metricsParameters = resources.createParameterBuffer(
    'metrics',
    'float32',
    GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
  );
  const stopDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'us-stop-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const metricsGraph = new GPUCommandGraph<void>(device, {id: 'us-metrics'});
  const stopCountView = importGraphBuffer(metricsGraph, 'stop-count', stopCount, 'uint32', 1);
  const startRowsView = importGraphBuffer(
    metricsGraph,
    'stop-start-rows',
    stopStartRows,
    'uint32',
    STOP_CAPACITY
  );
  const endRowsView = importGraphBuffer(
    metricsGraph,
    'stop-end-rows',
    stopEndRows,
    'uint32',
    STOP_CAPACITY
  );
  const stopCentroidsView = importGraphBuffer(
    metricsGraph,
    'stop-centroids',
    stopCentroids,
    'float32x2',
    STOP_CAPACITY
  );
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
          count: stopCountView,
          overflow: importGraphBuffer(metricsGraph, 'stop-overflow', stopOverflow, 'uint32', 1),
          totalCount: importGraphBuffer(metricsGraph, 'stop-total', stopTotal, 'uint32', 1)
        },
        drawInstanceCount: metricsGraph.importGPUData(
          'stop-draw-count',
          stopDraw.getInstanceCountData(0)
        ),
        startRows: startRowsView,
        endRows: endRowsView,
        centroids: stopCentroidsView,
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
  // Stop centroids are analysis meters; drawing needs degrees. Average the longitude and latitude
  // of the rows each stop covers.
  addKernelPass(metricsGraph, {
    id: 'stop-lng-lat',
    invocationCount: STOP_CAPACITY,
    bindings: [
      {name: 'stopCount', view: stopCountView, type: 'u32', access: 'read'},
      {name: 'startRows', view: startRowsView, type: 'u32', access: 'read'},
      {name: 'endRows', view: endRowsView, type: 'u32', access: 'read'},
      {
        name: 'lngLat',
        view: importGraphBuffer(metricsGraph, 'lng-lat', lngLatBuffer, 'float32x2', vertexCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'output',
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
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var lng = nan;
  var lat = nan;
  if (index < stopCount[stopCountOffset]) {
    let first = startRows[startRowsOffset + index];
    let last = endRows[endRowsOffset + index];
    var sumLng = 0.0;
    var sumLat = 0.0;
    for (var row = first; row <= last; row++) {
      sumLng += lngLat[lngLatOffset + row * 2u];
      sumLat += lngLat[lngLatOffset + row * 2u + 1u];
    }
    let n = f32(last - first + 1u);
    lng = sumLng / n;
    lat = sumLat / n;
  }
  output[outputOffset + index * 2u] = lng;
  output[outputOffset + index * 2u + 1u] = lat;`
  });
  // Rows past the stop count are NaN so the clustering treats them as noise.
  addKernelPass(metricsGraph, {
    id: 'cluster-points',
    invocationCount: STOP_CAPACITY,
    bindings: [
      {name: 'stopCount', view: stopCountView, type: 'u32', access: 'read'},
      {name: 'centroids', view: stopCentroidsView, type: 'f32', access: 'read'},
      {
        name: 'points',
        view: importGraphBuffer(
          metricsGraph,
          'cluster-points',
          clusterPoints,
          'float32x2',
          STOP_CAPACITY
        ),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let valid = index < stopCount[stopCountOffset];
  points[pointsOffset + index * 2u] = select(nan, centroids[centroidsOffset + index * 2u], valid);
  points[pointsOffset + index * 2u + 1u] = select(nan, centroids[centroidsOffset + index * 2u + 1u], valid);`
  });
  const metricsCompiled = resources.track(metricsGraph.compile());

  // ---- Clustering graph: stops to anchorages -----------------------------------------------------
  ctx.setStatus('Compiling the analysis graphs');
  const clusterLabels = resources.createBuffer('cluster-labels', STOP_CAPACITY * 4);
  const clusterTotal = resources.createBuffer('cluster-total', 4);
  const clusterIds = resources.createBuffer('cluster-ids', CLUSTER_CAPACITY * 4);
  const clusterStored = resources.createBuffer('cluster-stored', 4);
  const clusterOverflow = resources.createBuffer('cluster-overflow', 4);
  const clusterSizes = resources.createBuffer('cluster-sizes', CLUSTER_CAPACITY * 4);
  const clusterCentroids = resources.createBuffer('cluster-centroids', CLUSTER_CAPACITY * 8);
  const clusterParameters = resources.createParameterBuffer(
    'clustering',
    'float32',
    GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
  );
  const clusterGraph = new GPUCommandGraph<void>(device, {id: 'us-anchorages'});
  clusterGraph.add(
    new GPUSpatialClustering({
      id: 'anchorages',
      positions: importGraphBuffer(
        clusterGraph,
        'cluster-points',
        clusterPoints,
        'float32x2',
        STOP_CAPACITY
      ),
      parameters: clusterParameters.importToGraph(clusterGraph),
      gridSize: CLUSTER_GRID,
      labels: importGraphBuffer(
        clusterGraph,
        'cluster-labels',
        clusterLabels,
        'uint32',
        STOP_CAPACITY
      ),
      clusterCount: importGraphBuffer(clusterGraph, 'cluster-total', clusterTotal, 'uint32', 1),
      clusters: {
        ids: importGraphBuffer(clusterGraph, 'cluster-ids', clusterIds, 'uint32', CLUSTER_CAPACITY),
        count: importGraphBuffer(clusterGraph, 'cluster-stored', clusterStored, 'uint32', 1),
        overflow: importGraphBuffer(clusterGraph, 'cluster-overflow', clusterOverflow, 'uint32', 1)
      },
      clusterSizes: importGraphBuffer(
        clusterGraph,
        'cluster-sizes',
        clusterSizes,
        'uint32',
        CLUSTER_CAPACITY
      ),
      clusterCentroids: importGraphBuffer(
        clusterGraph,
        'cluster-centroids',
        clusterCentroids,
        'float32x2',
        CLUSTER_CAPACITY
      )
    })
  );
  const clusterCompiled = resources.track(clusterGraph.compile());
  // Cluster markers are drawn from CPU-ranked rows (longitude, latitude and summed dwell).
  const anchorageLngLat = resources.createBuffer('anchorage-lng-lat', CLUSTER_CAPACITY * 8);
  const anchorageDwell = resources.createBuffer('anchorage-dwell', CLUSTER_CAPACITY * 4);
  const anchorageDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'us-anchorage-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  // ---- Zone-event graph: gate crossings ---------------------------------------------------------
  const projection = {project: tracks.project, unproject: tracks.unproject};
  let gateEdges: GateEdges = buildGateEdges(projection.project, ctx.options.gateHalfWidth);
  const edgeCount = gateEdges.zones.length;
  const edgeStartsBuffer = resources.createBuffer('edge-starts', gateEdges.starts);
  const edgeEndsBuffer = resources.createBuffer('edge-ends', gateEdges.ends);
  const edgeZonesBuffer = resources.createBuffer('edge-zones', gateEdges.zones);
  const gateLinesBuffer = resources.createBuffer('gate-lines', gateEdges.lines);
  const selectedGateBuffer = resources.createBuffer('selected-gate', 16);
  const eventTracks = resources.createBuffer('event-tracks', EVENT_CAPACITY * 4);
  const eventCount = resources.createBuffer('event-count', 4);
  const eventOverflow = resources.createBuffer('event-overflow', 4);
  const eventZones = resources.createBuffer('event-zones', EVENT_CAPACITY * 4);
  const eventTypes = resources.createBuffer('event-types', EVENT_CAPACITY * 4);
  const eventTimes = resources.createBuffer('event-times', EVENT_CAPACITY * 4);
  const eventPositions = resources.createBuffer('event-positions', EVENT_CAPACITY * 8);
  const eventLngLat = resources.createBuffer('event-lng-lat', EVENT_CAPACITY * 8);
  const candidateCount = resources.createBuffer('candidate-count', 4);
  const candidateOverflow = resources.createBuffer('candidate-overflow', 4);
  const trackOverflow = resources.createBuffer('track-overflow', 4);
  const eventListOverflow = resources.createBuffer('event-list-overflow', 4);
  const eventClock = resources.createParameterBuffer('event-clock', 'float32', 4);
  const eventDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'us-event-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  function buildZoneGraph(eventsPerTrack: number): ZoneGraph {
    const graph = new GPUCommandGraph<void>(device, {id: `us-zone-events-${eventsPerTrack}`});
    const eventPositionsView = importGraphBuffer(
      graph,
      'event-positions',
      eventPositions,
      'float32x2',
      EVENT_CAPACITY
    );
    graph.add(
      new GPUZoneEvents({
        id: 'gates',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
        timestamps: importGraphBuffer(
          graph,
          'timestamps',
          timestampsBuffer,
          'float32',
          vertexCount
        ),
        trackOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
        edgeStarts: importGraphBuffer(
          graph,
          'edge-starts',
          edgeStartsBuffer,
          'float32x2',
          edgeCount
        ),
        edgeEnds: importGraphBuffer(graph, 'edge-ends', edgeEndsBuffer, 'float32x2', edgeCount),
        edgeZones: importGraphBuffer(graph, 'edge-zones', edgeZonesBuffer, 'uint32', edgeCount),
        zoneCount: gateCount,
        candidateCapacity: CANDIDATE_CAPACITY,
        maxEventsPerTrack: eventsPerTrack,
        events: {
          output: {
            ids: importGraphBuffer(graph, 'event-tracks', eventTracks, 'uint32', EVENT_CAPACITY),
            count: importGraphBuffer(graph, 'event-count', eventCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'event-overflow', eventOverflow, 'uint32', 1)
          },
          eventZones: importGraphBuffer(graph, 'event-zones', eventZones, 'uint32', EVENT_CAPACITY),
          eventTypes: importGraphBuffer(graph, 'event-types', eventTypes, 'uint32', EVENT_CAPACITY),
          eventTimes: importGraphBuffer(
            graph,
            'event-times',
            eventTimes,
            'float32',
            EVENT_CAPACITY
          ),
          eventPositions: eventPositionsView
        },
        diagnostics: {
          candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1),
          candidateOverflow: importGraphBuffer(
            graph,
            'candidate-overflow',
            candidateOverflow,
            'uint32',
            1
          ),
          trackOverflow: importGraphBuffer(graph, 'track-overflow', trackOverflow, 'uint32', 1),
          eventOverflow: importGraphBuffer(
            graph,
            'event-list-overflow',
            eventListOverflow,
            'uint32',
            1
          )
        }
      })
    );
    // Crossing positions are analysis meters: invert the azimuthal-equidistant projection to
    // longitude and latitude so the event layer can draw them.
    const centerLongitude = (SHIPPING_PROJECTION_CENTER[0] * Math.PI) / 180;
    const centerLatitude = (SHIPPING_PROJECTION_CENTER[1] * Math.PI) / 180;
    addKernelPass(graph, {
      id: 'event-lng-lat',
      invocationCount: EVENT_CAPACITY,
      declarations: `const EARTH_RADIUS: f32 = ${AZIMUTHAL_EARTH_RADIUS_METERS.toFixed(1)};
const CENTER_LONGITUDE: f32 = ${centerLongitude.toFixed(9)};
const SIN_CENTER_LATITUDE: f32 = ${Math.sin(centerLatitude).toFixed(9)};
const COS_CENTER_LATITUDE: f32 = ${Math.cos(centerLatitude).toFixed(9)};
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;`,
      bindings: [
        {name: 'meters', view: eventPositionsView, type: 'f32', access: 'read'},
        {
          name: 'lngLat',
          view: importGraphBuffer(graph, 'event-lng-lat', eventLngLat, 'float32x2', EVENT_CAPACITY),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let x = meters[metersOffset + index * 2u];
  let y = meters[metersOffset + index * 2u + 1u];
  let rho = sqrt(x * x + y * y);
  var longitude = CENTER_LONGITUDE;
  var latitude = asin(SIN_CENTER_LATITUDE);
  if (rho > 1.0) {
    let c = rho / EARTH_RADIUS;
    latitude = asin(cos(c) * SIN_CENTER_LATITUDE + y * sin(c) / rho * COS_CENTER_LATITUDE);
    longitude = CENTER_LONGITUDE + atan2(
      x * sin(c),
      rho * COS_CENTER_LATITUDE * cos(c) - y * SIN_CENTER_LATITUDE * sin(c)
    );
  }
  lngLat[lngLatOffset + index * 2u] = longitude * RADIANS_TO_DEGREES;
  lngLat[lngLatOffset + index * 2u + 1u] = latitude * RADIANS_TO_DEGREES;`
    });
    return {eventsPerTrack, compiled: resources.track(graph.compile())};
  }
  ctx.setStatus('Compiling the gate-crossing graph');
  let zoneGraph = buildZoneGraph(Number(ctx.options.eventsPerTrack));

  // ---- Line-density graphs (lazy per type and grid, buffers cached per grid) ----------------------
  const densityParameters = resources.createParameterBuffer(
    'density',
    'float32',
    GPU_LINE_DENSITY_PARAMETER_LENGTH
  );
  const typeSets = new Map<
    string,
    {positions: Buffer; offsets: Buffer; vertexCount: number; trackCount: number}
  >();
  const densityBuffers = new Map<string, DensityBuffers>();
  let density: DensityGraph | null = null;
  let densityStats: DensityStats | null = null;
  let densityBounds: [number, number, number, number] = [...US_DENSITY_BOUNDS];

  function getTypeSet(type: string): {
    positions: Buffer;
    offsets: Buffer;
    vertexCount: number;
    trackCount: number;
  } {
    const cached = typeSets.get(type);
    if (cached) return cached;
    if (type === 'all') {
      const all = {positions: lngLatBuffer, offsets: offsetsBuffer, vertexCount, trackCount};
      typeSets.set(type, all);
      return all;
    }
    const wanted = SHIPPING_TYPES.indexOf(type as (typeof SHIPPING_TYPES)[number]);
    const selected: number[] = [];
    let total = 0;
    for (let track = 0; track < trackCount; track++) {
      if (tracks.category[track] === wanted) {
        selected.push(track);
        total += tracks.offsets[track + 1] - tracks.offsets[track];
      }
    }
    const positions = new Float32Array(Math.max(4, total * 2));
    const offsets = new Uint32Array(selected.length + 1);
    let row = 0;
    selected.forEach((track, index) => {
      offsets[index] = row;
      const first = tracks.offsets[track];
      const last = tracks.offsets[track + 1];
      positions.set(tracks.lngLat.subarray(first * 2, last * 2), row * 2);
      row += last - first;
    });
    offsets[selected.length] = row;
    const set = {
      positions: resources.createBuffer(`density-positions-${type}`, positions),
      offsets: resources.createBuffer(`density-offsets-${type}`, offsets),
      vertexCount: Math.max(2, total),
      trackCount: selected.length
    };
    typeSets.set(type, set);
    return set;
  }

  function getDensityBuffers(grid: UsShippingOptions['densityGrid']): DensityBuffers {
    const cached = densityBuffers.get(grid);
    if (cached) return cached;
    const [columns, rows] = GRID_ROWS[grid];
    const cells = columns * rows;
    const lengths = resources.createBuffer(`density-lengths-${grid}`, cells * 4);
    const densities = resources.createBuffer(`density-densities-${grid}`, cells * 4);
    const overflow = resources.createBuffer(`density-overflow-${grid}`, 4);
    const totalRecords = resources.createBuffer(`density-records-${grid}`, 4);
    const reader = new SummaryReader(
      resources,
      `density-${grid}`,
      [
        {buffer: overflow, size: 4},
        {buffer: totalRecords, size: 4},
        {buffer: lengths, size: cells * 4},
        {buffer: densities, size: cells * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes, 0, 2);
        const lengthValues = new Float32Array(bytes, 8, cells);
        const densityValues = new Float32Array(bytes, 8 + cells * 4, cells);
        const positive: number[] = [];
        let maximumLength = 0;
        let maximumDensity = 0;
        let totalMeters = 0;
        for (let cell = 0; cell < cells; cell++) {
          const value = lengthValues[cell];
          if (value > 0) {
            totalMeters += value;
            maximumLength = Math.max(maximumLength, value);
            maximumDensity = Math.max(maximumDensity, densityValues[cell]);
            positive.push(cell);
          }
        }
        // Percentile table of the non-empty cells, for the ceiling slider.
        const percentiles = new Float32Array(101 * 2);
        if (positive.length > 0) {
          for (let which = 0; which < 2; which++) {
            const source = which === 0 ? lengthValues : densityValues;
            const sorted = Float32Array.from(positive, cell => source[cell]).sort();
            for (let percent = 0; percent <= 100; percent++) {
              percentiles[which * 101 + percent] =
                sorted[
                  Math.min(sorted.length - 1, Math.floor((percent / 100) * (sorted.length - 1)))
                ];
            }
          }
        }
        densityStats = {
          maximumLength,
          maximumDensity,
          percentiles,
          nonEmpty: positive.length,
          cells,
          pieces: words[1],
          overflow: words[0] !== 0,
          totalKilometers: totalMeters / 1000
        };
        updateDensityReadouts();
        ctx.requestLayers();
      }
    );
    const buffers = {lengths, densities, overflow, totalRecords, reader, cells};
    densityBuffers.set(grid, buffers);
    return buffers;
  }

  function buildDensityGraph(
    type: string,
    grid: UsShippingOptions['densityGrid']
  ): DensityGraph | null {
    const set = getTypeSet(type);
    if (set.trackCount === 0 || set.vertexCount < 2) return null;
    const [columns, rows] = GRID_ROWS[grid];
    const buffers = getDensityBuffers(grid);
    const graph = new GPUCommandGraph<void>(device, {id: `us-density-${type}-${grid}`});
    graph.add(
      new GPULineDensity({
        id: 'corridors',
        positions: importGraphBuffer(
          graph,
          'positions',
          set.positions,
          'float32x2',
          set.vertexCount
        ),
        pathOffsets: importGraphBuffer(
          graph,
          'path-offsets',
          set.offsets,
          'uint32',
          set.trackCount + 1
        ),
        columns,
        rows,
        coordinateSystem: 'spherical',
        maximumRecords: Math.max(1024, 6 * set.vertexCount),
        parameters: densityParameters.importToGraph(graph),
        output: {
          lengths: importGraphBuffer(graph, 'lengths', buffers.lengths, 'float32', buffers.cells),
          densities: importGraphBuffer(
            graph,
            'densities',
            buffers.densities,
            'float32',
            buffers.cells
          ),
          overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1),
          totalRecords: importGraphBuffer(graph, 'records', buffers.totalRecords, 'uint32', 1)
        }
      })
    );
    return {
      key: `${type}:${grid}`,
      compiled: resources.track(graph.compile()),
      buffers,
      columns,
      rows
    };
  }

  function rebuildDensity(): void {
    if (density) {
      resources.release(density.compiled);
      density = null;
    }
    densityStats = null;
    density = buildDensityGraph(ctx.options.densityType, ctx.options.densityGrid);
    densityDirty = true;
    if (!density) {
      ctx.setReadout('density', 'no vessels of this type');
    }
  }

  // ---- State ------------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, 86340], rate: 1, step: 60}
  );
  let playhead = ctx.options.time;
  let selectedTrack = NO_TRACK;
  let metricsDirty = true;
  let clusterDirty = true;
  let zonesDirty = true;
  let densityDirty = true;
  let statusStale = true;
  let stepReaderRequested = false;
  let lastHour = -1;
  let statusSnapshot: {status: Uint32Array; positions: Float32Array; speeds: Float32Array} | null =
    null;
  let crossings: GateCrossings | null = null;
  let anchorages: Anchorage[] = [];
  let clusterSummary = {noise: 0, clustered: 0, count: 0, overflow: false, stopCount: 0};
  let histograms: Record<'derived' | 'reported', SpeedHistograms | null> = {
    derived: null,
    reported: buildSpeedHistograms(tracks, tracks.reportedKnots)
  };
  let viewPending: [number, number, number, number] | null = null;
  let viewPendingFrames = 0;
  let trackSnapshot: {lengths: Float32Array; maxima: Float32Array; stopCounts: Uint32Array} | null =
    null;
  const typeCounts = new Uint32Array(SHIPPING_TYPES.length);
  for (const type of tracks.category) typeCounts[type]++;

  ctx.setReadout(
    'tracks',
    `${formatCount(trackCount)} tracks / ${formatCount(tracks.vesselCount)} vessels`
  );
  ctx.setReadout('fixes', `${formatCount(vertexCount)} AIS fixes`);

  function writeMetricsParameters(): void {
    metricsParameters.write(
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: ctx.options.stopSpeedKnots * METERS_PER_KNOT_SECOND,
        stopMinimumDuration: ctx.options.stopMinutes * 60
      })
    );
    metricsDirty = true;
    clusterDirty = true;
  }

  function writeClusterParameters(): void {
    const [minX, minY, maxX, maxY] = tracks.bounds;
    clusterParameters.write(
      getGPUSpatialClusteringParameterValues({
        bounds: [minX - 1000, minY - 1000, maxX + 1000, maxY + 1000],
        epsilon: ctx.options.clusterRadiusKm * 1000,
        minimumPoints: Math.round(ctx.options.clusterMinStops)
      })
    );
    clusterDirty = true;
  }

  function writeGateEdges(): void {
    gateEdges = buildGateEdges(projection.project, ctx.options.gateHalfWidth);
    edgeStartsBuffer.write(gateEdges.starts);
    edgeEndsBuffer.write(gateEdges.ends);
    zonesDirty = true;
  }

  function writeSegmentMask(): void {
    const filter = ctx.options.vesselFilter;
    const mask = new Uint32Array(segmentCount);
    if (filter === 'all') {
      mask.fill(1);
    } else {
      const wanted = SHIPPING_TYPES.indexOf(filter as (typeof SHIPPING_TYPES)[number]);
      for (let segment = 0; segment < segmentCount; segment++) {
        mask[segment] = tracks.category[tracks.segmentTracks[segment]] === wanted ? 1 : 0;
      }
    }
    segmentMaskBuffer.write(mask);
  }

  function writeSelectedGate(): void {
    const rows = new Float32Array(4).fill(Number.NaN);
    const index = SHIPPING_GATES.findIndex(gate => gate.id === ctx.options.gateFocus);
    if (index >= 0) rows.set(gateEdges.lines.subarray(index * 4, index * 4 + 4));
    selectedGateBuffer.write(rows);
  }

  writeMetricsParameters();
  writeClusterParameters();
  writeSegmentMask();
  writeSelectedGate();
  rebuildDensity();
  updateSpeedCharts();
  ctx.setStatus('Computing the day');

  // ---- Readbacks --------------------------------------------------------------------------------
  const statusReader = new SummaryReader(
    resources,
    'us-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: activeOverflow, size: 4},
      {buffer: status, size: trackCount * 4},
      {buffer: markerLngLat, size: trackCount * 8},
      {buffer: markerSpeeds, size: trackCount * 4},
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
      let gap = 0;
      for (let track = 0; track < trackCount; track++) {
        if (statusSnapshot.status[track] === GPU_TRAJECTORY_PLAYHEAD_STATUS.gap) gap++;
      }
      ctx.setReadout('active', words[0]);
      ctx.setReadout('inGap', gap);
      ctx.setReadout('trailSegments', words[trailStart]);
      describeSelection();
    }
  );

  const analysisReader = new SummaryReader(
    resources,
    'us-analysis',
    [
      {buffer: stopCount, size: 4},
      {buffer: stopOverflow, size: 4},
      {buffer: stopTotal, size: 4},
      {buffer: clusterStored, size: 4},
      {buffer: clusterOverflow, size: 4},
      {buffer: clusterTotal, size: 4},
      {buffer: stopIds, size: STOP_CAPACITY * 4},
      {buffer: stopDurations, size: STOP_CAPACITY * 4},
      {buffer: clusterLabels, size: STOP_CAPACITY * 4},
      {buffer: clusterCentroids, size: CLUSTER_CAPACITY * 8},
      {buffer: trackLengths, size: trackCount * 4},
      {buffer: maximumSpeeds, size: trackCount * 4},
      {buffer: trackStopCounts, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const stops = Math.min(words[0], STOP_CAPACITY);
      const clusters = Math.min(words[3], CLUSTER_CAPACITY);
      let offset = 6;
      const ids = words.subarray(offset, offset + STOP_CAPACITY);
      offset += STOP_CAPACITY;
      const durations = floats.subarray(offset, offset + STOP_CAPACITY);
      offset += STOP_CAPACITY;
      const labels = words.subarray(offset, offset + STOP_CAPACITY);
      offset += STOP_CAPACITY;
      const centroids = floats.subarray(offset, offset + CLUSTER_CAPACITY * 2);
      offset += CLUSTER_CAPACITY * 2;
      trackSnapshot = {
        lengths: floats.slice(offset, offset + trackCount),
        maxima: floats.slice(offset + trackCount, offset + trackCount * 2),
        stopCounts: words.slice(offset + trackCount * 2, offset + trackCount * 3)
      };
      const result = rankAnchorages(
        tracks,
        {
          stopCount: stops,
          clusterCount: clusters,
          labels,
          trackOfStop: ids,
          durations,
          centroids,
          noise: GPU_SPATIAL_CLUSTERING_NOISE
        },
        ctx.options.rankBy
      );
      anchorages = result.ranked;
      clusterSummary = {
        noise: result.noiseStops,
        clustered: result.clusteredStops,
        count: words[5],
        overflow: words[1] !== 0 || words[4] !== 0,
        stopCount: words[2]
      };
      const lngLat = new Float32Array(CLUSTER_CAPACITY * 2).fill(Number.NaN);
      const dwell = new Float32Array(CLUSTER_CAPACITY);
      anchorages.forEach((anchorage, row) => {
        lngLat[row * 2] = anchorage.longitude;
        lngLat[row * 2 + 1] = anchorage.latitude;
        dwell[row] = anchorage.dwellSeconds;
      });
      anchorageLngLat.write(lngLat);
      anchorageDwell.write(dwell);
      anchorageDraw.buffer.write(Uint32Array.of(6, clusters, 0, 0));
      updateAnalysisReadouts();
      ctx.requestLayers();
    }
  );

  const stepReader = new SummaryReader(
    resources,
    'us-steps',
    [{buffer: stepSpeeds, size: vertexCount * 4}],
    bytes => {
      if (destroyed) return;
      const metersPerSecond = new Float32Array(bytes);
      const knots = new Float32Array(vertexCount);
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        knots[vertex] = metersPerSecond[vertex] * KNOTS_PER_METER_SECOND;
      }
      histograms = {...histograms, derived: buildSpeedHistograms(tracks, knots)};
      updateSpeedCharts();
    }
  );

  const zoneReader = new SummaryReader(
    resources,
    'us-events',
    [
      {buffer: eventCount, size: 4},
      {buffer: eventOverflow, size: 4},
      {buffer: candidateCount, size: 4},
      {buffer: candidateOverflow, size: 4},
      {buffer: trackOverflow, size: 4},
      {buffer: eventListOverflow, size: 4},
      {buffer: eventTracks, size: EVENT_CAPACITY * 4},
      {buffer: eventZones, size: EVENT_CAPACITY * 4},
      {buffer: eventTypes, size: EVENT_CAPACITY * 4},
      {buffer: eventTimes, size: EVENT_CAPACITY * 4},
      {buffer: eventPositions, size: EVENT_CAPACITY * 8}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const count = Math.min(words[0], EVENT_CAPACITY);
      let offset = 6;
      const read = <T extends Uint32Array | Float32Array>(source: T, length: number): T => {
        const slice = source.subarray(offset, offset + length) as T;
        offset += length;
        return slice;
      };
      const eventTrackRows = read(words, EVENT_CAPACITY);
      const eventZoneRows = read(words, EVENT_CAPACITY);
      const eventTypeRows = read(words, EVENT_CAPACITY);
      const eventTimeRows = read(floats, EVENT_CAPACITY);
      const eventPositionRows = read(floats, EVENT_CAPACITY * 2);
      crossings = countGateCrossings(tracks, gateEdges, {
        count,
        tracks: eventTrackRows,
        zones: eventZoneRows,
        types: eventTypeRows,
        times: eventTimeRows,
        positions: eventPositionRows
      });
      const flags = [
        words[3] !== 0 ? 'candidate scratch full' : '',
        words[4] !== 0 ? 'events per track capped' : '',
        words[5] !== 0 ? 'event list full' : ''
      ].filter(Boolean);
      ctx.setReadout(
        'zoneHealth',
        flags.length > 0
          ? `Overflow: ${flags.join(', ')}`
          : `${formatCount(words[0])} events; candidates ${formatCount(words[2])} of ${formatCount(CANDIDATE_CAPACITY)}`
      );
      updateCrossingCharts();
      ctx.requestLayers();
    }
  );

  // ---- Summaries --------------------------------------------------------------------------------
  function describeVessel(track: number): string {
    const type = SHIPPING_TYPES[tracks.category[track]];
    const length = tracks.length[track];
    return `${SHIPPING_TYPE_LABELS[type]} (MMSI ${tracks.mmsi[track]}${length > 0 ? `, ${length.toFixed(0)} m` : ''})`;
  }

  function describeSelection(): void {
    if (selectedTrack === NO_TRACK) {
      ctx.setReadout('selected', 'click a vessel');
      return;
    }
    let text = describeVessel(selectedTrack);
    if (trackSnapshot) {
      const distance = trackSnapshot.lengths[selectedTrack] / NAUTICAL_MILE_METERS;
      const maximum = trackSnapshot.maxima[selectedTrack] * KNOTS_PER_METER_SECOND;
      text += `: ${distance.toFixed(0)} nm in this track, top step ${maximum.toFixed(1)} kn, ${trackSnapshot.stopCounts[selectedTrack]} stops`;
    }
    ctx.setReadout('selected', text);
  }
  describeSelection();

  function updateDensityReadouts(): void {
    const stats = densityStats;
    if (!stats || !density) return;
    const unit = ctx.options.densityValue === 'length' ? 'km per cell' : 'km per km2';
    const ceiling = getDensityCeiling();
    ctx.setLegendExtent('density', [0, ceiling.value]);
    ctx.setReadout(
      'density',
      `${formatCount(stats.nonEmpty)} of ${formatCount(stats.cells)} cells have traffic (${density.columns} x ${density.rows} grid)`
    );
    ctx.setReadout(
      'densityPieces',
      `${formatCount(stats.pieces)} segment-cell pieces${stats.overflow ? ' (capacity exceeded, low cells)' : ''}`
    );
    ctx.setReadout(
      'densityTotal',
      `${formatCount(stats.totalKilometers)} km of track; busiest cell ${formatCount(ceiling.maximum)} ${unit}`
    );
  }

  function getDensityCeiling(): {value: number; maximum: number} {
    const stats = densityStats;
    if (!stats) return {value: 1, maximum: 0};
    const byLength = ctx.options.densityValue === 'length';
    const scale = byLength ? 0.001 : 1000;
    const row = byLength ? 0 : 101;
    const percent = Math.round(Math.min(100, Math.max(50, ctx.options.densityCeiling)));
    const value = Math.max(1e-9, stats.percentiles[row + percent] * scale);
    return {
      value,
      maximum: (byLength ? stats.maximumLength : stats.maximumDensity) * scale
    };
  }

  function updateAnalysisReadouts(): void {
    const top = anchorages.slice(0, RANK_LIST_LENGTH);
    ctx.setReadout(
      'stops',
      `${formatCount(clusterSummary.stopCount)} stops (${formatCount(clusterSummary.clustered)} in clusters, ${formatCount(clusterSummary.noise)} isolated)`
    );
    ctx.setReadout(
      'anchorages',
      `${formatCount(Math.min(clusterSummary.count, CLUSTER_CAPACITY))} stopping places${clusterSummary.overflow ? ' (capacity reached)' : ''}`
    );
    ctx.setReadout(
      'anchorageList',
      top.length === 0
        ? 'no clusters at these settings'
        : top
            .map(
              (anchorage, index) =>
                `${index + 1}. ${anchorage.place}: ${formatHours(anchorage.dwellSeconds)}, ${anchorage.vessels} vessels`
            )
            .join('\n')
    );
    ctx.setChart('anchorRank', {
      kind: 'bars',
      values: top.map(anchorage =>
        ctx.options.rankBy === 'dwell' ? anchorage.dwellSeconds / 3600 : anchorage.stops
      ),
      labels: top.map((_, index) => `${index + 1}`),
      highlight: top.length > 0 ? [0] : [],
      yLabel: ctx.options.rankBy === 'dwell' ? 'vessel-hours stopped' : 'stops',
      xLabel: 'rank (see the list above)',
      height: 120,
      description: 'Stopping places ranked by total dwell'
    });
    if (trackSnapshot) {
      let longest = 0;
      for (let track = 0; track < trackCount; track++) {
        longest = Math.max(longest, trackSnapshot.maxima[track]);
      }
      ctx.setReadout('fastest', `${(longest * KNOTS_PER_METER_SECOND).toFixed(1)} kn`);
    }
    describeSelection();
  }

  function updateSpeedCharts(): void {
    const source = histograms[ctx.options.speedSource] ?? histograms.reported;
    if (!source) return;
    const typeCount = SHIPPING_TYPES.length;
    const filter = ctx.options.vesselFilter;
    const row = filter === 'all' ? typeCount : Math.max(0, SHIPPING_TYPES.indexOf(filter as never));
    const values = Array.from(source.hours[row]);
    ctx.setChart('speedHist', {
      kind: 'histogram',
      values,
      xDomain: [0, SPEED_BIN_COUNT],
      xLabel: 'speed (kn)',
      yLabel: 'vessel-hours',
      color: row === typeCount ? 0 : row % 6,
      height: 120,
      markers: Number.isFinite(getHistogramMedian(source.hours[row]))
        ? [{x: getHistogramMedian(source.hours[row]), label: 'median'}]
        : [],
      description: `Time-weighted speed distribution of ${row === typeCount ? 'all vessels' : SHIPPING_TYPE_LABELS[SHIPPING_TYPES[row]]}`
    });
    const medians = SHIPPING_TYPES.map((_, type) => getHistogramMedian(source.hours[type]));
    ctx.setChart('typeSpeeds', {
      kind: 'bars',
      values: medians.map(value => (Number.isFinite(value) ? value : 0)),
      labels: SHIPPING_TYPES.map(type => type),
      highlight: filter === 'all' ? [] : [row],
      yLabel: 'median moving speed (kn)',
      height: 120,
      description: 'Median speed of moving vessels by type'
    });
    const parts = SHIPPING_TYPES.map((type, index) => {
      const moving = source.hours[index].reduce((total, value) => total + value, 0);
      const parked = source.stationaryHours[index];
      const share = moving + parked > 0 ? (parked / (moving + parked)) * 100 : 0;
      return `${SHIPPING_TYPE_LABELS[type].split(' (')[0].padEnd(11)} ${medians[index].toFixed(1).padStart(5)} kn   ${share.toFixed(0).padStart(3)}% stationary   ${formatCount(typeCounts[index])} tracks`;
    });
    ctx.setReadout('typeTable', parts.join('\n'));
  }

  function updateCrossingCharts(): void {
    const data = crossings;
    if (!data) return;
    const focus = ctx.options.gateFocus;
    const index = SHIPPING_GATES.findIndex(gate => gate.id === focus);
    const hours = Array.from({length: 24}, (_, hour) => hour + 0.5);
    const total = new Array<number>(24).fill(0);
    for (let gate = 0; gate < gateCount; gate++) {
      for (let hour = 0; hour < 24; hour++) {
        total[hour] += data.hourly[gate][0][hour] + data.hourly[gate][1][hour];
      }
    }
    const marker = [{x: Math.min(23.99, playhead / 3600), label: 'now'}];
    if (index < 0) {
      ctx.setChart('hourly', {
        kind: 'line',
        series: [{label: 'all gates', x: hours, y: total, area: true, color: 0}],
        xDomain: [0, 24],
        xLabel: 'hour of day (UTC)',
        yLabel: 'crossings per hour',
        markers: marker,
        height: 130,
        description: 'Gate crossings per hour, all gates'
      });
    } else {
      const gate = SHIPPING_GATES[index];
      ctx.setChart('hourly', {
        kind: 'line',
        series: [
          {label: gate.directions[0], x: hours, y: Array.from(data.hourly[index][0]), color: 0},
          {label: gate.directions[1], x: hours, y: Array.from(data.hourly[index][1]), color: 1},
          {label: 'all gates', x: hours, y: total, color: 5, dashed: true}
        ],
        xDomain: [0, 24],
        xLabel: 'hour of day (UTC)',
        yLabel: `crossings per hour at ${gate.short}`,
        markers: marker,
        height: 130,
        description: `Gate crossings per hour at ${gate.name}`
      });
    }
    const order = SHIPPING_GATES.map((_, gate) => gate).sort(
      (a, b) => data.totals[b] - data.totals[a]
    );
    ctx.setChart('gateTotals', {
      kind: 'bars',
      values: order.map(gate => data.totals[gate]),
      labels: order.map(gate => SHIPPING_GATES[gate].short),
      highlight: index < 0 ? [] : [order.indexOf(index)],
      yLabel: 'crossings today',
      height: 140,
      description: 'Total crossings per gate'
    });
    let busiest = 0;
    for (let hour = 1; hour < 24; hour++) if (total[hour] > total[busiest]) busiest = hour;
    const sum = total.reduce((a, b) => a + b, 0);
    ctx.setReadout(
      'crossings',
      `${formatCount(sum)} crossings today; busiest hour ${String(busiest).padStart(2, '0')}:00 UTC (${total[busiest]})`
    );
    if (index >= 0) {
      const gate = SHIPPING_GATES[index];
      ctx.setReadout(
        'gate',
        `${gate.short}: ${data.totals[index]} crossings, ${data.hourly[index][0].reduce((a, b) => a + b, 0)} ${gate.directions[0]}, ${data.hourly[index][1].reduce((a, b) => a + b, 0)} ${gate.directions[1]}`
      );
    } else {
      const top = order[0];
      ctx.setReadout('gate', `busiest: ${SHIPPING_GATES[top].short} (${data.totals[top]})`);
    }
  }

  function viewBounds(viewport: Viewport): [number, number, number, number] {
    const corners = [
      viewport.unproject([0, 0]),
      viewport.unproject([viewport.width, 0]),
      viewport.unproject([0, viewport.height]),
      viewport.unproject([viewport.width, viewport.height])
    ];
    return [
      Math.max(-180, Math.min(...corners.map(c => c[0]))),
      Math.max(-85, Math.min(...corners.map(c => c[1]))),
      Math.min(180, Math.max(...corners.map(c => c[0]))),
      Math.min(85, Math.max(...corners.map(c => c[1])))
    ];
  }

  function writeDensityParameters(): void {
    if (!density) return;
    const [west, south, east, north] = densityBounds;
    densityParameters.write(
      getGPULineDensityParameterValues({
        minX: west,
        minY: south,
        cellWidth: (east - west) / density.columns,
        cellHeight: (north - south) / density.rows
      })
    );
  }

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      playheadCompiled,
      trailCompiled,
      metricsCompiled,
      clusterCompiled,
      zoneGraph.compiled,
      ...(density ? [density.compiled] : [])
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'vesselFilter':
          writeSegmentMask();
          updateSpeedCharts();
          ctx.requestLayers();
          break;
        case 'stopSpeedKnots':
        case 'stopMinutes':
          writeMetricsParameters();
          break;
        case 'clusterRadiusKm':
        case 'clusterMinStops':
          writeClusterParameters();
          break;
        case 'rankBy':
          // Re-rank from a fresh readback of the same GPU results.
          analysisReader.markStale();
          break;
        case 'gateHalfWidth':
          writeGateEdges();
          break;
        case 'eventsPerTrack':
          resources.release(zoneGraph.compiled);
          zoneGraph = buildZoneGraph(Number(state.eventsPerTrack));
          zonesDirty = true;
          ctx.requestLayers();
          break;
        case 'gateFocus':
          writeSelectedGate();
          updateCrossingCharts();
          ctx.requestLayers();
          break;
        case 'speedSource':
          updateSpeedCharts();
          break;
        case 'densityType':
        case 'densityGrid':
          rebuildDensity();
          ctx.requestLayers();
          break;
        case 'densityExtent':
          densityBounds = [...US_DENSITY_BOUNDS];
          viewPending = null;
          densityDirty = true;
          ctx.requestLayers();
          break;
        case 'densityValue':
        case 'densityCeiling':
          updateDensityReadouts();
          ctx.requestLayers();
          break;
        case 'showDensity':
          if (state.showDensity) densityDirty = true;
          ctx.requestLayers();
          break;
        case 'play':
        case 'time':
        case 'speed':
        case 'loop':
        case 'maxGapMinutes':
        case 'pulseMinutes':
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
      playhead = clock.advance(frame);
      ctx.setReadout('clock', formatShippingClock(playhead));
      const hour = Math.floor(playhead / 3600);
      if (hour !== lastHour) {
        lastHour = hour;
        updateCrossingCharts();
      }

      if (metricsDirty || clusterDirty) {
        if (metricsDirty) metricsCompiled.encode(commandEncoder, {parameters: undefined});
        clusterCompiled.encode(commandEncoder, {parameters: undefined});
        metricsDirty = false;
        clusterDirty = false;
        analysisReader.request(commandEncoder);
        if (!stepReaderRequested) {
          stepReaderRequested = true;
          stepReader.request(commandEncoder);
        }
      } else {
        analysisReader.flush(commandEncoder);
        stepReader.flush(commandEncoder);
      }

      if (zonesDirty) {
        zoneGraph.compiled.encode(commandEncoder, {parameters: undefined});
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: eventCount,
          destinationBuffer: eventDraw.buffer,
          destinationOffset: 4,
          size: 4
        });
        zonesDirty = false;
        zoneReader.request(commandEncoder);
      } else {
        zoneReader.flush(commandEncoder);
      }
      eventClock.write(
        Float32Array.of(
          playhead,
          options.pulseMinutes * 60,
          options.showRestingCrossings ? 1 : 0,
          0
        )
      );

      if (options.showDensity && density) {
        if (options.densityExtent === 'view') {
          const bounds = viewBounds(frame.viewport);
          const same =
            viewPending && bounds.every((value, index) => value === viewPending?.[index]);
          if (same) viewPendingFrames++;
          else {
            viewPending = bounds;
            viewPendingFrames = 0;
          }
          if (
            viewPendingFrames === VIEW_SETTLE_FRAMES &&
            !bounds.every((value, index) => value === densityBounds[index])
          ) {
            densityBounds = bounds;
            densityDirty = true;
          }
        }
        if (densityDirty) {
          writeDensityParameters();
          density.compiled.encode(commandEncoder, {parameters: undefined});
          densityDirty = false;
          density.buffers.reader.request(commandEncoder);
        } else {
          density.buffers.reader.flush(commandEncoder);
        }
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
      if (options.showDensity && density && densityStats) {
        const byLength = options.densityValue === 'length';
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `us-density-${density.key}`,
            ...drawProps,
            gridSize: [density.columns, density.rows],
            bounds: densityBounds,
            rowOrigin: 'south',
            tessellation: 64,
            values: byLength ? density.buffers.lengths : density.buffers.densities,
            valueFormat: 'float32',
            colormap: options.ramp,
            valueRange: [0, getDensityCeiling().value],
            valueScale: byLength ? 0.001 : 1000,
            sqrtScale: true,
            discardAtOrBelow: 0,
            opacity: options.densityOpacity,
            color: [255, 255, 255, 255]
          })
        );
      }
      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'us-backdrop',
            ...drawProps,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 0.8,
            color: dark ? [190, 200, 220, 34] : [50, 60, 80, 40]
          })
        );
      }
      if (options.showTrails) {
        const bySpeed = options.trailColor === 'speed';
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'us-trails',
            ...drawProps,
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
            palette: SHIPPING_TYPE_COLORS,
            widthPixels: 2.2
          })
        );
      }
      if (options.showGates) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'us-gates',
            ...drawProps,
            segments: gateLinesBuffer,
            instanceCount: gateCount,
            widthPixels: 3,
            color: dark ? [255, 255, 255, 230] : [20, 24, 32, 230]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'us-gate-selected',
            ...drawProps,
            segments: selectedGateBuffer,
            instanceCount: 1,
            widthPixels: 7,
            color: [255, 200, 40, 255]
          })
        );
      }
      if (options.showStops) {
        layers.push(
          new StopMarkerLayer({
            id: 'us-stops',
            ...drawProps,
            centroids: stopLngLat,
            durations: stopDurations,
            drawCommands: stopDraw,
            baseRadiusPixels: 2,
            radiusPerSqrtSecond: 0.012,
            maximumRadiusPixels: 9,
            durationForFullColor: 12 * 3600,
            opacity: 0.8
          })
        );
      }
      if (options.showAnchorages) {
        layers.push(
          new StopMarkerLayer({
            id: 'us-anchorages',
            ...drawProps,
            centroids: anchorageLngLat,
            durations: anchorageDwell,
            drawCommands: anchorageDraw,
            baseRadiusPixels: 7,
            radiusPerSqrtSecond: 0.012,
            maximumRadiusPixels: 30,
            durationForFullColor: 400 * 3600,
            opacity: 0.9
          })
        );
      }
      if (options.showCrossings) {
        layers.push(
          new ZoneEventMarkerLayer({
            id: 'us-crossings',
            ...drawProps,
            positions: eventLngLat,
            eventTracks,
            eventTimes,
            eventTypes,
            trackStartTimes: trackStartBuffer,
            clock: eventClock.buffer,
            drawCommands: eventDraw,
            sizePixels: 7
          })
        );
      }
      if (options.showVessels) {
        layers.push(
          new VesselMarkerLayer({
            id: 'us-vessels',
            ...drawProps,
            ids: activeIds,
            positions: markerLngLat,
            headings,
            speeds: markerSpeeds,
            categories: categoryBuffer,
            drawCommands: markerDraw,
            sizePixels: options.markerSize,
            colorMode: options.markerColor,
            ramp: options.ramp,
            speedForFullColor: SPEED_RAMP_KNOTS * METERS_PER_KNOT_SECOND,
            palette: SHIPPING_TYPE_COLORS,
            categoryFilter:
              options.vesselFilter === 'all'
                ? null
                : SHIPPING_TYPES.indexOf(options.vesselFilter as (typeof SHIPPING_TYPES)[number]),
            selectedTrack: selectedTrack === NO_TRACK ? null : selectedTrack,
            outlineColor: dark ? [8, 10, 16, 235] : [20, 24, 32, 215]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const viewport = ctx.getViewport();
      if (!viewport) return null;
      const options = ctx.options;
      if (options.showGates && crossings) {
        for (let gate = 0; gate < gateCount; gate++) {
          const [x0, y0] = viewport.project([
            gateEdges.lines[gate * 4],
            gateEdges.lines[gate * 4 + 1]
          ]);
          const [x1, y1] = viewport.project([
            gateEdges.lines[gate * 4 + 2],
            gateEdges.lines[gate * 4 + 3]
          ]);
          if (distanceToSegment(event.pixel, [x0, y0], [x1, y1]) < 10) {
            const info = SHIPPING_GATES[gate];
            return `${info.name}: ${crossings.totals[gate]} crossings today (${crossings.hourly[gate][0].reduce((a, b) => a + b, 0)} ${info.directions[0]}, ${crossings.hourly[gate][1].reduce((a, b) => a + b, 0)} ${info.directions[1]})`;
          }
        }
      }
      if (options.showAnchorages) {
        const count = Math.min(anchorages.length, CLUSTER_CAPACITY);
        for (let rank = 0; rank < count; rank++) {
          const anchorage = anchorages[rank];
          const [x, y] = viewport.project([anchorage.longitude, anchorage.latitude]);
          if (Math.hypot(x - event.pixel[0], y - event.pixel[1]) < 14) {
            return `#${rank + 1} ${anchorage.place}: ${formatHours(anchorage.dwellSeconds)} stopped, ${anchorage.stops} stops, ${anchorage.vessels} vessels`;
          }
        }
      }
      const track = pickVessel(event.pixel);
      if (track < 0 || !statusSnapshot) return null;
      const speed = statusSnapshot.speeds[track] * KNOTS_PER_METER_SECOND;
      const stops = trackSnapshot?.stopCounts[track];
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
      analysisReader.stop();
      stepReader.stop();
      zoneReader.stop();
      for (const buffers of densityBuffers.values()) buffers.reader.stop();
      resources.destroy();
    }
  };

  /** Nearest active vessel within 14 CSS pixels of the pointer, or -1. */
  function pickVessel(pixel: readonly [number, number]): number {
    const viewport = ctx.getViewport();
    const snapshot = statusSnapshot;
    if (!viewport || !snapshot || !ctx.options.showVessels) return -1;
    const filter = ctx.options.vesselFilter;
    let best = -1;
    let bestDistance = 14 * 14;
    for (let track = 0; track < trackCount; track++) {
      if (snapshot.status[track] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active) continue;
      if (filter !== 'all' && SHIPPING_TYPES[tracks.category[track]] !== filter) continue;
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
}

/** Distance in pixels from a point to a segment. */
function distanceToSegment(
  point: readonly [number, number],
  start: readonly [number, number],
  end: readonly [number, number]
): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  const t =
    lengthSquared === 0
      ? 0
      : Math.min(
          1,
          Math.max(0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared)
        );
  return Math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy));
}
