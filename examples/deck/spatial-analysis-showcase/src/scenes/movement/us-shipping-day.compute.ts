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
  getGPUTrajectoryPlayheadParameterValues,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS,
  GPULineDensity,
  GPUTrajectoryPlayhead,
  GPUZoneEvents
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getClassCounts} from '../../cartography/breaks';
import {
  getClassIndexOf,
  getClassTableLayerProps,
  makeClassTable
} from '../../cartography/class-table';
import {US, nearestPlaceLabel} from '../../cartography/gazetteer';
import {formatCount, formatPercent, liveText} from '../../cartography/live-text';
import {evaluateZoomStops} from '../../cartography/zoom';
import type {ClassTable, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import type {PaletteColor} from '../../engine/ramps';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent} from '../scene';
import {VesselMarkerLayer, ZoneEventMarkerLayer} from './b12-layers';
import {KNOTS_PER_METER_SECOND, METERS_PER_KNOT_SECOND} from './b12-tracks';
import {
  CONTEXT_TRACK_INK,
  getCategoryPaletteFromGroups,
  getShipSpeedClasses,
  getTrafficDensityClasses,
  getVesselGroupPalette,
  HEAD_HALO_INK,
  inkFor,
  SHIP_SPEED_BREAKS_KNOTS,
  SUBJECT_INK,
  toHex,
  VESSEL_GROUP_LABELS,
  VESSEL_GROUPS,
  ZONE_EVENT_INK
} from './movement-style';
import {getClipWalkDiagram, getHourlyCrossingsDiagram} from './us-shipping-charts';
import {
  clipAndWalk,
  formatDensityValue,
  getBasisScale,
  getBusiestCell,
  getCellAreaSquareKilometers,
  getCellBounds,
  getCellCenter,
  getCellIndexAt,
  getCellSizeKilometers,
  getCellValue,
  getLogHistogram,
  getShareAtOrBelow,
  getSortedMedian,
  getSortedValues,
  getStretchBreaks,
  pickClipWalkSegment,
  STRETCH_METHODS,
  summarizeDensityField,
  type ClipWalkSegment,
  type DensityBasis,
  type DensityField,
  type DensityGrid,
  type Stretch
} from './us-shipping-density';
import {US_SHIPPING_PLACES} from './us-shipping-places';
import {
  buildSegmentTable,
  countGateCrossings,
  getActivityHistogram,
  getCrossingDirection,
  getFixIntervals,
  getGroupKilometers,
  getMedianChordSeconds,
  getReceiverEdge,
  getShareAbove,
  summarizeTracksNear,
  type GateCrossings
} from './us-shipping.stats';
import {
  AZIMUTHAL_EARTH_RADIUS_METERS,
  buildGateEdges,
  loadShippingTracks,
  SHIPPING_FRAMES,
  SHIPPING_GATES,
  SHIPPING_PROJECTION_CENTER,
  SHIPPING_TYPE_GROUP,
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
  maxGapMinutes: number;
  showVessels: boolean;
  showTrails: boolean;
  showBackdrop: boolean;
  showDensity: boolean;
  showGates: boolean;
  showReceiverEdge: boolean;
  markerColor: 'uniform' | 'group' | 'speed';
  trailMinutes: number;
  stretch: Stretch;
  densityGrid: '512' | '1024' | '2048';
  densityValue: DensityBasis;
  gateFocus: string;
  vesselFilter: string;
  markerSize: number;
  densityOpacity: number;
  gateHalfWidth: number;
  eventsPerTrack: '8' | '16' | '32';
  pulseMinutes: number;
};

const EVENT_CAPACITY = 32768;
const CANDIDATE_CAPACITY = 1 << 18;
const STATUS_INTERVAL_FRAMES = 12;
const NO_TRACK = 0xffffffff;
/** A crossing time no clock reaches: events hidden from the ring layer carry it. */
const HIDDEN_SECONDS = 1e9;
/** Longitude and latitude extent of the whole-country density grid. */
const US_DENSITY_BOUNDS: readonly [number, number, number, number] = [-127.5, 24, -65.5, 49.6];
const GRID_SIZES: Record<UsShippingOptions['densityGrid'], [number, number]> = {
  '512': [512, 256],
  '1024': [1024, 512],
  '2048': [2048, 1024]
};
const CLASS_COUNT = 5;
/** Alpha of the lowest density class (the lane floor); empty cells are not drawn at all. */
const LOWEST_CLASS_ALPHA = 140;
const STOPPED_KNOTS = SHIP_SPEED_BREAKS_KNOTS[0];
const SPEED_CLASS_BREAKS = SHIP_SPEED_BREAKS_KNOTS.map(knots => knots * METERS_PER_KNOT_SECOND);
/** Arrow half-length by zoom: 5 px over the whole country, 7 px at harbour scale. */
const ARROW_SIZE_STOPS = [
  [4, 5],
  [8, 7]
] as const;
const TRAIL_WIDTH_STOPS = [
  [4, 1.6],
  [8, 2.2]
] as const;
const SHORT_GROUP_LABELS = ['Passenger', 'Cargo and tanker', 'Tug and tow', 'Other'];
const INLAND_RADIUS_KILOMETERS = 25;
const ACTIVITY_BINS = 48;

type DensityBuffers = {
  lengths: Buffer;
  densities: Buffer;
  overflow: Buffer;
  totalRecords: Buffer;
  reader: SummaryReader;
  cells: number;
};

type DensityGraph = {
  grid: UsShippingOptions['densityGrid'];
  compiled: CompiledGPUCommandGraph<void>;
  buffers: DensityBuffers;
  columns: number;
  rows: number;
};

type ZoneGraph = {
  eventsPerTrack: number;
  compiled: CompiledGPUCommandGraph<void>;
};

type EventSnapshot = {
  count: number;
  tracks: Uint32Array;
  zones: Uint32Array;
  types: Uint32Array;
  times: Float32Array;
  positions: Float32Array;
};

/**
 * One US day of AIS. Three independent graphs read the same track buffers: the playhead and
 * time-window graphs animate the fleet, a zone-event graph counts gate crossings and a
 * line-density graph paints the day as km of track per km2. Analysis runs in azimuthal-equidistant
 * meters, the playhead in Web Mercator meters (so headings stay true on the map), and everything
 * is drawn from longitude and latitude.
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

  // ---- Zone-event graph: gate crossings ---------------------------------------------------------
  ctx.setStatus('Compiling the gate-crossing graph');
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
  // What the ring layer reads: one ring per crossing, coloured by direction. Enter events carry
  // the direction (0 first, 1 second), exit events and other gates are hidden with a time no
  // clock reaches. Written from the readback of the zone events.
  const eventDrawTypes = resources.createBuffer('event-draw-types', EVENT_CAPACITY * 4);
  const eventDrawTimes = resources.createBuffer(
    'event-draw-times',
    new Float32Array(EVENT_CAPACITY).fill(HIDDEN_SECONDS)
  );
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
  let zoneGraph = buildZoneGraph(Number(ctx.options.eventsPerTrack));

  // ---- Line-density graph (one grid at a time; buffers kept per grid) ----------------------------
  const densityParameters = resources.createParameterBuffer(
    'density',
    'float32',
    GPU_LINE_DENSITY_PARAMETER_LENGTH
  );
  const densityBuffers = new Map<string, DensityBuffers>();
  let density: DensityGraph | null = null;
  let densityField: DensityField | null = null;
  const densityBounds = [...US_DENSITY_BOUNDS] as [number, number, number, number];

  function getDensityGrid(): DensityGrid | null {
    return density ? {columns: density.columns, rows: density.rows, bounds: densityBounds} : null;
  }

  function getDensityBuffers(grid: UsShippingOptions['densityGrid']): DensityBuffers {
    const existing = densityBuffers.get(grid);
    if (existing) return existing;
    const [columns, rows] = GRID_SIZES[grid];
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
        // A readback of a grid the reader has since switched away from is stale.
        if (destroyed || ctx.options.densityGrid !== grid) return;
        const words = new Uint32Array(bytes, 0, 2);
        densityField = summarizeDensityField(
          {columns, rows, bounds: densityBounds},
          new Float32Array(bytes.slice(8, 8 + cells * 4)),
          new Float32Array(bytes.slice(8 + cells * 4, 8 + cells * 8)),
          words[1],
          words[0] !== 0
        );
        onDensityField();
      }
    );
    const buffers = {lengths, densities, overflow, totalRecords, reader, cells};
    densityBuffers.set(grid, buffers);
    return buffers;
  }

  function buildDensityGraph(grid: UsShippingOptions['densityGrid']): DensityGraph {
    const [columns, rows] = GRID_SIZES[grid];
    const buffers = getDensityBuffers(grid);
    const graph = new GPUCommandGraph<void>(device, {id: `us-density-${grid}`});
    graph.add(
      new GPULineDensity({
        id: 'corridors',
        positions: importGraphBuffer(graph, 'positions', lngLatBuffer, 'float32x2', vertexCount),
        pathOffsets: importGraphBuffer(
          graph,
          'path-offsets',
          offsetsBuffer,
          'uint32',
          trackCount + 1
        ),
        columns,
        rows,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        },
        maximumRecords: Math.max(1024, 6 * vertexCount),
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
    return {grid, compiled: resources.track(graph.compile()), buffers, columns, rows};
  }

  function rebuildDensity(): void {
    if (density) resources.release(density.compiled);
    densityField = null;
    density = buildDensityGraph(ctx.options.densityGrid);
    densityDirty = true;
    densityTable = null;
    ctx.setLegendData('densityTable', null);
    updateDensityReadouts();
    updateClipWalk();
  }

  // ---- Derived summaries (CPU, once) -------------------------------------------------------------
  const segmentTable = buildSegmentTable(tracks);
  const fixIntervals = getFixIntervals(tracks);
  const clipWalkSegment: ClipWalkSegment | null = pickClipWalkSegment(tracks);
  const midAtlanticBand: readonly [number, number] = [
    SHIPPING_FRAMES.midAtlantic.latitude - 2,
    SHIPPING_FRAMES.midAtlantic.latitude + 2
  ];
  const receiverEdgeLongitude = getReceiverEdge(tracks, midAtlanticBand, -77);
  const riverReach = US_SHIPPING_PLACES.places['mississippi-river'];
  const riverSummary = summarizeTracksNear(tracks, riverReach.lngLat, INLAND_RADIUS_KILOMETERS);
  ctx.setTimelineData({
    domain: [0, 86400],
    histogram: getActivityHistogram(tracks, ACTIVITY_BINS)
  });

  // ---- State ------------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, 86340], rate: 1, step: 60}
  );
  let playhead = ctx.options.time;
  let selectedTrack = NO_TRACK;
  let zonesDirty = true;
  let densityDirty = true;
  let statusStale = true;
  let lastHour = -1;
  let statusSnapshot: {status: Uint32Array; positions: Float32Array; speeds: Float32Array} | null =
    null;
  let crossings: GateCrossings | null = null;
  let lastEvents: EventSnapshot | null = null;
  let densityTable: ClassTable | null = null;
  let legendFilter: readonly number[] | null = null;
  let lastViewKey = '';
  let lastSizeBucket = -1;
  let lastTickMeters: number | null | undefined;
  let lastGroundKey = '';

  const vesselCount = tracks.vesselCount;
  const sampleLine = `${formatCount(vesselCount)} vessels · ${formatCount(vertexCount)} positions · terrestrial receivers only`;
  const runtimeFurniture: {
    title: {sample: string; chips: string[]};
    scaleBar: {ticks?: number[]};
  } = {
    title: {sample: sampleLine, chips: ['Receivers hear what they hear']},
    scaleBar: {}
  };
  ctx.setFurniture(runtimeFurniture);
  ctx.setReadout('medianFixGap', roundTo(fixIntervals.medianSeconds / 60, 1));
  ctx.setAnnotations('inland', [
    {
      kind: 'note',
      id: 'inland-river',
      coordinate: riverReach.lngLat,
      title: `${formatCount(riverSummary.tracks)} tracks within ${INLAND_RADIUS_KILOMETERS} km`,
      text: `${formatPercent(riverSummary.towShare)} of them tugs and tows`,
      maxZoom: 4.5
    }
  ]);
  if (ctx.reducedMotion()) ctx.setOptions({play: false});

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

  function getFocusedGate(): number {
    return SHIPPING_GATES.findIndex(gate => gate.id === ctx.options.gateFocus);
  }

  function writeSelectedGate(): void {
    const rows = new Float32Array(4).fill(Number.NaN);
    const index = getFocusedGate();
    if (index >= 0) rows.set(gateEdges.lines.subarray(index * 4, index * 4 + 4));
    selectedGateBuffer.write(rows);
  }

  /** Writes the ring layer's direction and time buffers from the last zone-event readback. */
  function writeCrossingRings(): void {
    const events = lastEvents;
    if (!events) return;
    const focus = getFocusedGate();
    const types = new Uint32Array(EVENT_CAPACITY);
    const times = new Float32Array(EVENT_CAPACITY).fill(HIDDEN_SECONDS);
    for (let event = 0; event < events.count; event++) {
      if (events.types[event] !== 0) continue;
      const gate = events.zones[event];
      if (gate >= gateCount || (focus >= 0 && gate !== focus)) continue;
      types[event] = getCrossingDirection(
        gateEdges,
        gate,
        events.positions[event * 2],
        events.positions[event * 2 + 1]
      );
      times[event] = events.times[event];
    }
    eventDrawTypes.write(types);
    eventDrawTimes.write(times);
  }

  // ---- Readbacks --------------------------------------------------------------------------------
  const statusReader = new SummaryReader(
    resources,
    'us-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: activeOverflow, size: 4},
      {buffer: status, size: trackCount * 4},
      {buffer: markerLngLat, size: trackCount * 8},
      {buffer: markerSpeeds, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const statusStart = 2;
      const positionStart = statusStart + trackCount;
      const speedStart = positionStart + trackCount * 2;
      statusSnapshot = {
        status: words.slice(statusStart, statusStart + trackCount),
        positions: floats.slice(positionStart, positionStart + trackCount * 2),
        speeds: floats.slice(speedStart, speedStart + trackCount)
      };
      let gap = 0;
      let moving = 0;
      for (let track = 0; track < trackCount; track++) {
        const state = statusSnapshot.status[track];
        if (state === GPU_TRAJECTORY_PLAYHEAD_STATUS.gap) gap++;
        if (
          state === GPU_TRAJECTORY_PLAYHEAD_STATUS.active &&
          statusSnapshot.speeds[track] * KNOTS_PER_METER_SECOND >= STOPPED_KNOTS
        ) {
          moving++;
        }
      }
      ctx.setReadout('active', words[0]);
      ctx.setReadout('movingShare', words[0] > 0 ? moving / words[0] : null);
      ctx.setReadout('inGap', gap);
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
        const slice = source.slice(offset, offset + length) as T;
        offset += length;
        return slice;
      };
      lastEvents = {
        count,
        tracks: read(words, EVENT_CAPACITY),
        zones: read(words, EVENT_CAPACITY),
        types: read(words, EVENT_CAPACITY),
        times: read(floats, EVENT_CAPACITY),
        positions: read(floats, EVENT_CAPACITY * 2)
      };
      crossings = countGateCrossings(tracks, gateEdges, lastEvents);
      writeCrossingRings();
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
      updateCrossingReadouts();
      ctx.requestLayers();
    }
  );

  // ---- Colour -----------------------------------------------------------------------------------
  function getMarkerPalette(): {palette: PaletteColor[]; mode: 'category' | 'speedClasses'} {
    const ground = ctx.ground();
    switch (ctx.options.markerColor) {
      case 'group':
        return {
          palette: getCategoryPaletteFromGroups(SHIPPING_TYPE_GROUP, ground),
          mode: 'category'
        };
      case 'speed':
        return {palette: getShipSpeedClasses(ground), mode: 'speedClasses'};
      default:
        return {palette: [inkFor(SUBJECT_INK, ground)], mode: 'category'};
    }
  }

  function getTrailPalette(): PaletteColor[] {
    const ground = ctx.ground();
    if (ctx.options.markerColor === 'group') {
      return getCategoryPaletteFromGroups(SHIPPING_TYPE_GROUP, ground, 235);
    }
    const [red, green, blue] = inkFor(SUBJECT_INK, ground);
    return [[red, green, blue, 235]];
  }

  function getDensityColors(): PaletteColor[] {
    return getTrafficDensityClasses(ctx.ground(), CLASS_COUNT).map(
      (color, index): PaletteColor => [
        color[0],
        color[1],
        color[2],
        index === 0 ? LOWEST_CLASS_ALPHA : 255
      ]
    );
  }

  // ---- Density classes and readouts -------------------------------------------------------------
  /** Rebuilds the class table from the readback: breaks come from the occupied cells, once per stretch. */
  function updateDensityTable(): void {
    const field = densityField;
    if (!field) {
      densityTable = null;
      ctx.setLegendData('densityTable', null);
      return;
    }
    const basis = ctx.options.densityValue;
    const sorted = getSortedValues(field, basis);
    if (sorted.length === 0) {
      densityTable = null;
      ctx.setLegendData('densityTable', null);
      return;
    }
    const breaks = getStretchBreaks(sorted, ctx.options.stretch, CLASS_COUNT);
    densityTable = makeClassTable({
      breaks,
      colors: getDensityColors(),
      unit: basis === 'length' ? 'km per cell' : 'km per km²',
      extent: [sorted[0], sorted[sorted.length - 1]],
      method: STRETCH_METHODS[ctx.options.stretch],
      noData: {label: 'No track in the cell'},
      format: formatDensityValue
    });
    ctx.setLegendData('densityTable', {
      table: densityTable,
      counts: getClassCounts(sorted, breaks),
      basis
    });
  }

  function updateDensityReadouts(): void {
    const field = densityField;
    const grid = getDensityGrid();
    if (grid) {
      const size = getCellSizeKilometers(grid, ctx.getViewState().latitude);
      ctx.setReadout('cellSize', `${size.width.toFixed(1)} × ${size.height.toFixed(1)} km`);
    }
    if (!field) {
      for (const id of [
        'pieces',
        'overflow',
        'portVsLane',
        'busiestCell',
        'medianCell',
        'occupiedCells'
      ]) {
        ctx.setReadout(id, null);
      }
      return;
    }
    const basis = ctx.options.densityValue;
    const sorted = getSortedValues(field, basis);
    const unit = basis === 'length' ? 'km per cell' : 'km per km²';
    const median = getSortedMedian(sorted);
    const busiest = sorted.length > 0 ? sorted[sorted.length - 1] : Number.NaN;
    ctx.setReadout(
      'pieces',
      `${formatCount(field.pieces)} pieces from ${formatCount(segmentCount)} segments`
    );
    ctx.setReadout(
      'overflow',
      field.overflow
        ? 'Piece capacity exceeded: the busiest cells are low'
        : 'None: every piece was counted'
    );
    ctx.setReadout('portVsLane', `${formatCount(busiest / median)}×`);
    ctx.setReadout('busiestCell', `${formatDensityValue(busiest)} ${unit}`);
    ctx.setReadout('medianCell', `${formatDensityValue(median)} ${unit}`);
    ctx.setReadout(
      'occupiedCells',
      `${formatCount(field.occupiedCount)} of ${formatCount(field.lengths.length)}`
    );
  }

  function updateCellHistogram(): void {
    const field = densityField;
    if (!field || !densityTable) {
      ctx.setChart('cellHistogram', null);
      return;
    }
    const sorted = getSortedValues(field, ctx.options.densityValue);
    const {counts, domain} = getLogHistogram(sorted, 36);
    ctx.setChart('cellHistogram', {
      kind: 'histogram',
      values: counts,
      xDomain: domain,
      breaks: densityTable.breaks.map(value => Math.log10(value)),
      classColors: densityTable.colors.map(color => [color[0], color[1], color[2], 255] as const),
      formatX: value => formatDensityValue(10 ** value),
      xLabel: `${ctx.options.densityValue === 'length' ? 'km per cell' : 'km per km²'} (log axis)`,
      yLabel: 'cells',
      height: 120,
      description:
        'Histogram of the occupied cells by track density on a logarithmic axis, coloured by class, with the class breaks marked.'
    });
  }

  /** The note at the busiest cell: how many times busier it is than the median occupied cell. */
  function updateDensityAnnotations(): void {
    const field = densityField;
    if (!ctx.options.showDensity || !field) {
      ctx.setAnnotations('busiest', null);
      return;
    }
    const basis = ctx.options.densityValue;
    const cell = getBusiestCell(field, basis);
    if (cell < 0) {
      ctx.setAnnotations('busiest', null);
      return;
    }
    const sorted = getSortedValues(field, basis);
    const center = getCellCenter(field.grid, cell);
    const ratio = sorted[sorted.length - 1] / getSortedMedian(sorted);
    const note: MapAnnotation = {
      kind: 'note',
      id: 'busiest-cell',
      coordinate: center,
      title: liveText('{ratio:integer}× the median cell', {ratio}),
      text: nearestPlaceLabel(US, center, {kinds: ['port']}) ?? undefined,
      priority: 3,
      // The busiest cell is in the national frame; the Gulf frame would only point off the map.
      maxZoom: 5
    };
    ctx.setAnnotations('busiest', [note]);
  }

  /** The clip-and-walk of the story's one real segment, on the current grid: map overlay and diagram. */
  function updateClipWalk(): void {
    const grid = getDensityGrid();
    if (!grid || !clipWalkSegment) {
      ctx.setChart('clipWalk', null);
      ctx.setAnnotations('clip-walk', null);
      return;
    }
    const pieces = clipAndWalk(grid, clipWalkSegment.from, clipWalkSegment.to);
    ctx.setChart('clipWalk', getClipWalkDiagram(grid, clipWalkSegment, pieces));
    const cells = new Set(pieces.map(piece => piece.row * grid.columns + piece.column));
    const overlay: MapAnnotation[] = [
      {
        kind: 'line',
        id: 'clip-walk-segment',
        coordinates: [clipWalkSegment.from, clipWalkSegment.to],
        tone: 'signal',
        text: ctx.options.showDensity ? 'The diagram segment' : undefined,
        minZoom: 4
      }
    ];
    for (const cell of cells) {
      const column = cell % grid.columns;
      const row = Math.floor(cell / grid.columns);
      const [west, south, east, north] = grid.bounds;
      const width = (east - west) / grid.columns;
      const height = (north - south) / grid.rows;
      overlay.push({
        kind: 'frame',
        id: `clip-walk-cell-${cell}`,
        bounds: [
          west + column * width,
          south + row * height,
          west + (column + 1) * width,
          south + (row + 1) * height
        ],
        tone: 'signal',
        minZoom: 8
      });
    }
    ctx.setAnnotations('clip-walk', ctx.options.showDensity ? overlay : null);
  }

  function onDensityField(): void {
    updateDensityTable();
    updateDensityReadouts();
    updateCellHistogram();
    updateDensityAnnotations();
    updateClipWalk();
    ctx.requestLayers();
  }

  // ---- Gate readouts ----------------------------------------------------------------------------
  function updateCrossingReadouts(): void {
    const data = crossings;
    if (!data) return;
    const focus = getFocusedGate();
    const first = new Array<number>(24).fill(0);
    const second = new Array<number>(24).fill(0);
    for (let gate = 0; gate < gateCount; gate++) {
      if (focus >= 0 && gate !== focus) continue;
      for (let hour = 0; hour < 24; hour++) {
        first[hour] += data.hourly[gate][0][hour];
        second[hour] += data.hourly[gate][1][hour];
      }
    }
    const theme = ctx.theme();
    const gate = focus >= 0 ? SHIPPING_GATES[focus] : null;
    ctx.setChart(
      'hourly',
      getHourlyCrossingsDiagram({
        first,
        second,
        firstLabel: gate ? gate.directions[0] : 'First direction',
        secondLabel: gate ? gate.directions[1] : 'Second direction',
        firstColor: toHex(inkFor(ZONE_EVENT_INK.enter, theme)),
        secondColor: toHex(inkFor(ZONE_EVENT_INK.exit, theme)),
        playheadHours: playhead / 3600,
        timeZone: gate ? gate.timeZone : null
      })
    );
    const total =
      focus >= 0 ? data.totals[focus] : data.totals.reduce((sum, value) => sum + value, 0);
    ctx.setReadout('crossings', total);
    ctx.setReadout('chord', roundTo(getMedianChordSeconds(data, focus) / 60, 1));
  }

  // ---- Vessel groups in view ---------------------------------------------------------------------
  function updateGroupShares(viewport: Viewport): void {
    const [west, south, east, north] = viewport.getBounds();
    const totals = getGroupKilometers(segmentTable, [west, south, east, north]);
    const sum = totals.reduce((a, b) => a + b, 0);
    ctx.setLegendData('groupKilometers', totals);
    ctx.setReadout('towShare', sum > 0 ? totals[2] / sum : null);
    ctx.setChart('groupShares', {
      kind: 'bars',
      values: totals.map(value => Math.round(value)),
      labels: SHORT_GROUP_LABELS,
      colors: getVesselGroupPalette(ctx.theme()),
      horizontal: true,
      xLabel: 'km of track in view over the day',
      height: 110,
      description: `Kilometres of track in the map view by vessel group: ${VESSEL_GROUPS.map(
        (group, index) => `${VESSEL_GROUP_LABELS[group]} ${Math.round(totals[index])}`
      ).join(', ')}.`
    });
  }

  function updateGapReadouts(): void {
    const minutes = ctx.options.maxGapMinutes;
    ctx.setReadout('gapShare', minutes > 0 ? getShareAbove(fixIntervals, minutes * 60) : 0);
  }

  function updateCoverageAnnotations(): void {
    if (!ctx.options.showReceiverEdge || !Number.isFinite(receiverEdgeLongitude)) {
      ctx.setAnnotations('coverage', null);
      return;
    }
    const middle = (midAtlanticBand[0] + midAtlanticBand[1]) / 2;
    ctx.setAnnotations('coverage', [
      {
        kind: 'line',
        id: 'receiver-edge-line',
        coordinates: [
          [receiverEdgeLongitude, midAtlanticBand[0]],
          [receiverEdgeLongitude, midAtlanticBand[1]]
        ],
        dashed: true,
        tone: 'signal'
      },
      {
        kind: 'note',
        id: 'receiver-edge',
        coordinate: [receiverEdgeLongitude, middle],
        title: 'Receiver edge',
        text: liveText('{share:percent} of fixes in this latitude band lie west of it', {
          share: 0.99
        }),
        priority: 3
      },
      {
        kind: 'area',
        id: 'open-ocean',
        coordinate: [receiverEdgeLongitude + 3.4, middle],
        text: 'Open ocean: no reports',
        tone: 'muted',
        size: 'medium'
      }
    ]);
  }

  /** Scale-bar tick at the cell size while the density is the subject. */
  function updateScaleBarTick(): void {
    const grid = getDensityGrid();
    const teaching = ctx.options.showDensity && grid !== null;
    const meters =
      teaching && grid
        ? Math.round(
            (getCellSizeKilometers(grid, ctx.getViewState().latitude).width * 1000) / 100
          ) * 100
        : null;
    if (meters === lastTickMeters) return;
    lastTickMeters = meters;
    runtimeFurniture.scaleBar = {ticks: meters ? [meters] : undefined};
    ctx.setFurniture(runtimeFurniture);
  }

  function updateCost(): void {
    const options = ctx.options;
    const passes =
      playheadCompiled.stats.nodeOrder.length +
      (options.showTrails ? trailCompiled.stats.nodeOrder.length : 0) +
      (options.showDensity && density ? density.compiled.stats.nodeOrder.length : 0) +
      (options.showGates ? zoneGraph.compiled.stats.nodeOrder.length : 0);
    ctx.setCost({records: vertexCount, passes});
  }

  rebuildDensity();
  writeSegmentMask();
  writeSelectedGate();
  updateGapReadouts();
  updateCoverageAnnotations();
  updateCost();
  ctx.setStatus('Computing the day');

  // ---- Tooltips ---------------------------------------------------------------------------------
  function describeVessel(track: number): string {
    const type = SHIPPING_TYPES[tracks.category[track]];
    const length = tracks.length[track];
    return `${SHIPPING_TYPE_LABELS[type]}${length > 0 ? `, ${length.toFixed(0)} m` : ''}`;
  }

  function getVesselTooltip(track: number): TooltipContent | null {
    if (!statusSnapshot) return null;
    const knots = statusSnapshot.speeds[track] * KNOTS_PER_METER_SECOND;
    const group = SHIPPING_TYPE_GROUP[tracks.category[track]];
    const moving = knots >= STOPPED_KNOTS;
    const speedClass = SHIP_SPEED_BREAKS_KNOTS.filter(value => knots >= value).length;
    const options = ctx.options;
    return {
      title: VESSEL_GROUP_LABELS[VESSEL_GROUPS[group]],
      subtitle: describeVessel(track),
      rows: [
        {
          label: 'Speed',
          value: knots.toFixed(1),
          unit: 'kn',
          emphasis: true,
          swatch:
            options.markerColor === 'speed'
              ? getShipSpeedClasses(ctx.ground())[speedClass]
              : options.markerColor === 'group'
                ? getVesselGroupPalette(ctx.ground())[group]
                : undefined
        },
        {label: 'State', value: moving ? 'Moving (arrow)' : 'Stopped (square)'}
      ]
    };
  }

  function getDensityTooltip(coordinate: readonly [number, number]): TooltipContent | null {
    const field = densityField;
    if (!field || !densityTable || !ctx.options.showDensity) return null;
    const cell = getCellIndexAt(field.grid, coordinate[0], coordinate[1]);
    if (cell < 0) return null;
    const basis = ctx.options.densityValue;
    const value = getCellValue(field, cell, basis);
    if (!(value > 0)) return null;
    const sorted = getSortedValues(field, basis);
    const row = Math.floor(cell / field.grid.columns);
    const unit = basis === 'length' ? 'km per cell' : 'km per km²';
    const classIndex = getClassIndexOf(densityTable, value);
    const area = getCellAreaSquareKilometers(field.grid, row);
    const [west, south, east, north] = getCellBounds(field.grid, cell);
    return {
      title:
        nearestPlaceLabel(US, getCellCenter(field.grid, cell), {kinds: ['port']}) ?? 'Open water',
      subtitle: `Cell of ${formatDensityValue(area)} km²`,
      rows: [
        {
          label: 'Track',
          value: formatDensityValue(value),
          unit,
          swatch: densityTable.colors[classIndex],
          emphasis: true
        },
        {
          label: 'Rank',
          value: `Denser than ${formatPercent(getShareAtOrBelow(sorted, value))} of occupied cells`
        }
      ],
      highlight: {kind: 'box', bounds: [west, south, east, north]}
    };
  }

  function getGateTooltip(gate: number): TooltipContent | null {
    if (!crossings) return null;
    const info = SHIPPING_GATES[gate];
    return {
      title: info.name,
      subtitle: 'Gate (approximate line)',
      rows: [
        {label: 'Crossings today', value: crossings.totals[gate], emphasis: true},
        {
          label: info.directions[0],
          value: crossings.hourly[gate][0].reduce((a, b) => a + b, 0)
        },
        {
          label: info.directions[1],
          value: crossings.hourly[gate][1].reduce((a, b) => a + b, 0)
        }
      ]
    };
  }

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      playheadCompiled,
      trailCompiled,
      zoneGraph.compiled,
      ...(density ? [density.compiled] : [])
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'vesselFilter':
          writeSegmentMask();
          ctx.requestLayers();
          break;
        case 'gateHalfWidth':
          writeGateEdges();
          break;
        case 'eventsPerTrack':
          resources.release(zoneGraph.compiled);
          zoneGraph = buildZoneGraph(Number(state.eventsPerTrack));
          zonesDirty = true;
          updateCost();
          ctx.requestLayers();
          break;
        case 'gateFocus':
          writeSelectedGate();
          writeCrossingRings();
          updateCrossingReadouts();
          ctx.requestLayers();
          break;
        case 'densityGrid':
          rebuildDensity();
          updateCost();
          ctx.requestLayers();
          break;
        case 'densityValue':
        case 'stretch':
          updateDensityTable();
          updateDensityReadouts();
          updateCellHistogram();
          updateDensityAnnotations();
          ctx.requestLayers();
          break;
        case 'showDensity':
          if (state.showDensity && !densityField) densityDirty = true;
          updateDensityAnnotations();
          updateClipWalk();
          updateScaleBarTick();
          updateCost();
          ctx.requestLayers();
          break;
        case 'showGates':
        case 'showTrails':
          updateCost();
          ctx.requestLayers();
          break;
        case 'showReceiverEdge':
          updateCoverageAnnotations();
          break;
        case 'maxGapMinutes':
          updateGapReadouts();
          break;
        case 'markerColor':
          lastViewKey = '';
          ctx.requestLayers();
          break;
        case 'play':
        case 'time':
        case 'speed':
        case 'loop':
        case 'pulseMinutes':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      updateCrossingReadouts();
      lastViewKey = '';
      ctx.requestLayers();
    },

    onGroundChange() {
      updateDensityTable();
      updateCellHistogram();
      ctx.requestLayers();
    },

    onLegendFilter(_legendId, classes) {
      legendFilter = classes;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      playhead = clock.advance(frame);
      const hour = Math.floor(playhead / 3600);
      if (hour !== lastHour) {
        lastHour = hour;
        updateCrossingReadouts();
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
      eventClock.write(Float32Array.of(playhead, options.pulseMinutes * 60, 0, 0));

      if (density) {
        if (densityDirty && options.showDensity) {
          densityParameters.write(
            getGPULineDensityParameterValues({
              minX: densityBounds[0],
              minY: densityBounds[1],
              cellWidth: (densityBounds[2] - densityBounds[0]) / density.columns,
              cellHeight: (densityBounds[3] - densityBounds[1]) / density.rows
            })
          );
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
            startFadeDuration: trailSeconds
          })
        );
        trailCompiled.encode(commandEncoder, {parameters: undefined});
      }

      if (statusStale || frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) {
        statusReader.markStale();
        statusStale = false;
      }
      statusReader.flush(commandEncoder);

      // Camera-driven work, once the view has settled on a new place or zoom band.
      const viewport = frame.viewport;
      const bucket = Math.round(viewport.zoom * 4);
      if (bucket !== lastSizeBucket) {
        lastSizeBucket = bucket;
        ctx.requestLayers();
      }
      if (frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) {
        const view = ctx.getViewState();
        const key = `${view.longitude.toFixed(2)}:${view.latitude.toFixed(2)}:${view.zoom.toFixed(2)}`;
        if (key !== lastViewKey) {
          lastViewKey = key;
          if (options.markerColor === 'group') updateGroupShares(viewport);
          updateDensityReadouts();
          updateScaleBarTick();
        }
        const groundKey = ctx.ground();
        if (groundKey !== lastGroundKey) {
          lastGroundKey = groundKey;
          ctx.requestLayers();
        }
      }
    },

    getLayers() {
      const options = ctx.options;
      const ground = ctx.ground();
      const zoom = ctx.getViewport()?.zoom ?? ctx.getViewState().zoom;
      const halo = inkFor(HEAD_HALO_INK, ground);
      const layers: Layer[] = [];
      const field = densityField;
      if (options.showDensity && density && field && densityTable) {
        const byLength = options.densityValue === 'length';
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `us-density-${density.grid}`,
            ...drawProps,
            gridSize: [density.columns, density.rows],
            bounds: densityBounds,
            rowOrigin: 'south',
            tessellation: 64,
            values: byLength ? density.buffers.lengths : density.buffers.densities,
            valueFormat: 'float32',
            colormap: 'uniform',
            valueScale: getBasisScale(options.densityValue),
            discardAtOrBelow: 0,
            ...getClassTableLayerProps(densityTable),
            highlightClasses: legendFilter,
            opacity: options.densityOpacity
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
            widthPixels: 0.5,
            blending: 'additive',
            opacity: options.showDensity ? 0.25 : 1,
            color: inkFor(CONTEXT_TRACK_INK, ground)
          })
        );
      }
      if (options.showTrails) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'us-trails',
            ...drawProps,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            values: categoryBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'category',
            palette: getTrailPalette(),
            widthPixels: TRAIL_WIDTH_STOPS
          })
        );
      }
      if (options.showGates) {
        const focused = getFocusedGate() >= 0;
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'us-gates',
            ...drawProps,
            segments: gateLinesBuffer,
            instanceCount: gateCount,
            widthPixels: focused ? 1.2 : 2.5,
            outlineColor: halo,
            outlineWidthPixels: focused ? 0.8 : 1.25,
            color: focused ? [232, 237, 242, 120] : [232, 237, 242, 255]
          })
        );
        if (focused) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'us-gate-selected',
              ...drawProps,
              segments: selectedGateBuffer,
              instanceCount: 1,
              widthPixels: 2.5,
              outlineColor: halo,
              outlineWidthPixels: 1.25,
              color: [255, 255, 255, 255]
            })
          );
        }
        layers.push(
          new ZoneEventMarkerLayer({
            id: 'us-crossings',
            ...drawProps,
            positions: eventLngLat,
            eventTracks,
            eventTimes: eventDrawTimes,
            eventTypes: eventDrawTypes,
            trackStartTimes: trackStartBuffer,
            clock: eventClock.buffer,
            drawCommands: eventDraw,
            enterColor: inkFor(ZONE_EVENT_INK.enter, ground),
            exitColor: inkFor(ZONE_EVENT_INK.exit, ground),
            sizePixels: 4.4
          })
        );
      }
      if (options.showVessels) {
        const {palette, mode} = getMarkerPalette();
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
            sizePixels:
              options.markerSize > 0
                ? options.markerSize
                : evaluateZoomStops(ARROW_SIZE_STOPS, zoom),
            colorMode: mode,
            speedClassBreaks: SPEED_CLASS_BREAKS,
            stoppedSpeed: STOPPED_KNOTS * METERS_PER_KNOT_SECOND,
            palette,
            categoryFilter:
              options.vesselFilter === 'all'
                ? null
                : SHIPPING_TYPES.indexOf(options.vesselFilter as (typeof SHIPPING_TYPES)[number]),
            selectedTrack: selectedTrack === NO_TRACK ? null : selectedTrack,
            outlineColor: halo
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
          if (distanceToSegment(event.pixel, [x0, y0], [x1, y1]) < 10) return getGateTooltip(gate);
        }
      }
      const track = pickVessel(event.pixel);
      if (track >= 0) return getVesselTooltip(track);
      return event.coordinate ? getDensityTooltip(event.coordinate) : null;
    },

    onClick(event) {
      const track = pickVessel(event.pixel);
      selectedTrack = track < 0 || track === selectedTrack ? NO_TRACK : track;
      ctx.requestLayers();
      return track >= 0;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
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

/** Rounds to `digits` decimals. */
function roundTo(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
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
