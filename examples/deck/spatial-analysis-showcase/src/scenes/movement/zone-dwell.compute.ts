// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe,
  getGPUTrajectoryMetricsParameterValues,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getClassCounts, getExtent} from '../../cartography/breaks';
import {getClassTableLayerProps, getClassIndexOf} from '../../cartography/class-table';
import {formatArea, formatCount, formatPercent} from '../../cartography/live-text';
import type {ClassTable, LngLat, MapAnnotation, MapHighlight} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {StopMarkerLayer, ZoneEventMarkerLayer} from './b12-layers';
import {
  formatEasternClock,
  formatUtcClock,
  loadVesselTracks,
  METERS_PER_KNOT_SECOND,
  VESSEL_CATEGORIES,
  VESSEL_CATEGORY_LABELS
} from './b12-tracks';
import {findZone, ZONE_KIND_LABELS} from './b12-zones';
import {formatDwell, getWaitingClasses} from './movement-style';
import {buildZoneDwellGeometry, OUTLINE_STYLE_NAMES} from './zone-dwell-geometry';
import {
  isApproximateZoneKind,
  matchesZoneKindFilter,
  type ZoneKindFilter
} from './zone-dwell-names';
import {
  getEventInks,
  getOutlineStyles,
  getStopRing,
  getTrackInk,
  makeZoneClassTable,
  STOP_BREAKS_SECONDS
} from './zone-dwell-style';
import {
  computeClassBreaks,
  formatQuantity,
  formatZoneValue,
  getBreaksKey,
  getGroupValues,
  getUnitBreakFactor,
  getZoneValueLabels,
  reduceToGroups,
  SECONDS_PER_DAY,
  type GroupStatistics,
  type ZoneMetric,
  type ZoneStatistics,
  type ZoneUnit
} from './zone-dwell-values';

/** Option state of the zone dwell scene. */
export type ZoneDwellOptions = {
  variant: 'inside' | 'stopped';
  compareVariants: boolean;
  metric: ZoneMetric;
  unit: ZoneUnit;
  zoneKind: ZoneKindFilter;
  showTracks: boolean;
  showEvents: boolean;
  showStops: boolean;
  showVessel: boolean;
  notes: 'none' | 'size' | 'approximate';
  fillOpacity: number;
  stopSpeedKnots: number;
  stopMinutes: number;
  eventsPerVessel: '16' | '32' | '64';
  playing: boolean;
  time: number;
  playbackSpeed: string;
  loop: boolean;
};

/** Playhead the story starts from, so a deep link and the first frame agree. */
export const DEFAULT_PLAYHEAD = 43200;

const STOP_CAPACITY = 4096;
const VISIT_CAPACITY = 8192;
const MAX_EVENTS_PER_TRACK = 64;
const EVENT_CANDIDATE_CAPACITY = 1 << 20;
const STOP_CANDIDATE_CAPACITY = 1 << 18;
const PULSE_SECONDS = 20 * 60;
const SETTLE_MILLISECONDS = 200;
const RANKED_ZONES = 10;
const NO_GROUP = -1;
const ENTER = 0;

type ZoneTable = {
  counts: Buffer;
  sums: Buffer;
  means: Buffer;
  maximums: Buffer;
};

type SummaryFlags = {
  events: number;
  eventOverflow: number;
  candidates: number;
  candidateOverflow: number;
  trackOverflow: number;
  listOverflow: number;
  stops: number;
  stopOverflow: number;
  joinOverflow: number;
};

const EMPTY_FLAGS: SummaryFlags = {
  events: 0,
  eventOverflow: 0,
  candidates: 0,
  candidateOverflow: 0,
  trackOverflow: 0,
  listOverflow: 0,
  stops: 0,
  stopOverflow: 0,
  joinOverflow: 0
};

/** Zone events read back from the GPU: one row per enter or exit. */
type EventTable = {
  /** Increases with every readback, so work keyed on it reruns for new data only. */
  version: number;
  count: number;
  tracks: Uint32Array;
  zones: Uint32Array;
  types: Uint32Array;
  /** Seconds after the first fix of the track. */
  times: Float32Array;
  /** Planar metres. */
  positions: Float32Array;
};

/** The visits (inside) or stops (stopped) of the zones, one row each. */
type StayTable = {
  /** Increases with every readback. */
  version: number;
  count: number;
  tracks: Uint32Array;
  zones: Uint32Array;
  seconds: Float32Array;
  /** Planar metres, stops only. */
  centroids?: Float32Array;
};

type VariantState = {
  variant: 'inside' | 'stopped';
  compiled: CompiledGPUCommandGraph<void>;
  table: ZoneTable;
  displayValues: Buffer;
  tableReader: SummaryReader;
  listReader: SummaryReader;
  eventReader: SummaryReader | null;
  needsEncode: boolean;
  stats: ZoneStatistics | null;
  flags: SummaryFlags;
  stays: StayTable | null;
  eventsPerVessel?: number;
};

/** The vessel the third step follows and what it did at the gate. */
type FollowedVessel = {
  track: number;
  category: string;
  /** Crossings of the gate zone, in time order: absolute seconds, type, position in metres. */
  crossings: {time: number; type: number; x: number; y: number}[];
  /** Visits to named zones for the Gantt chart. */
  visits: {group: number; start: number; end: number}[];
};

type FrozenBreaks = {
  breaks: number[];
  /** Zones with time that the breaks were computed from. */
  zonesWithTime: number;
  /** `[min, max]` of the reference values, in the unit the breaks were computed in. */
  extent: [number, number];
  method: string;
};

/**
 * Zone dwell: how long vessels spend in each harbor zone, two ways at once. The `inside` graph
 * runs `addFleetDwellZoneEventsRecipe` (`GPUZoneEvents` enter and exit events, then a dense
 * per-zone `GPUGroupStatistics`): any time inside a zone counts. The `stopped` graph runs
 * `addFleetDwellRecipe` (`GPUTrajectoryMetrics` stops, `GPUPointInPolygonJoin` to the zones, the
 * same statistics): only time spent stopped counts. Both graphs are compiled once and stay live,
 * so a swipe can draw them side by side on one frozen class table. Both write one dense row per
 * zone; the scene reads the small table back, adds the pieces of one named zone up, turns it into
 * the unit of the map and classes it once per unit.
 */
export async function createZoneDwell(
  ctx: SceneContext<ZoneDwellOptions>
): Promise<SceneInstance<ZoneDwellOptions>> {
  const vessels = loadVesselTracks(ctx.datasets.get('ais-vessels'));
  const zonesDataset = ctx.datasets.get('ais-zones');
  if (!zonesDataset.geojson) throw new Error('ais-zones has no geometry');
  const geometry = buildZoneDwellGeometry(zonesDataset.geojson, vessels.project);
  const {zones, groups} = geometry;
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = vessels;
  const zoneCount = zones.zoneCount;
  const edgeCount = zones.edgeZones.length;
  const resources = new SpatialAnalysisResources(device, 'zone-dwell');
  const coordinateOrigin: [number, number, number] = [vessels.origin[0], vessels.origin[1], 0];
  const gateGroup = groups.kinds.indexOf('gate');

  // ---- Static inputs --------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', vessels.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', vessels.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', vessels.offsets);
  const segmentsBuffer = resources.createBuffer('segments', vessels.segments);
  const trackStartTimes = new Float32Array(trackCount);
  for (let track = 0; track < trackCount; track++) {
    trackStartTimes[track] = vessels.timestamps[vessels.offsets[track]];
  }
  const trackStartBuffer = resources.createBuffer('track-start-times', trackStartTimes);
  const edgeZonesBuffer = resources.createBuffer('edge-zones', zones.edgeZones);
  const edgeStartsBuffer = resources.createBuffer('edge-starts', zones.edgeStarts);
  const edgeEndsBuffer = resources.createBuffer('edge-ends', zones.edgeEnds);
  const polygonBuffers = {
    positions: resources.createBuffer('polygon-positions', zones.polygonPositions),
    featureOffsets: resources.createBuffer('feature-offsets', zones.featureOffsets),
    polygonOffsets: resources.createBuffer('polygon-offsets', zones.polygonOffsets),
    ringOffsets: resources.createBuffer('ring-offsets', zones.ringOffsets)
  };

  // Fill triangles (largest zone first) and the outlines of the three line styles.
  const fillTriangles = resources.createBuffer('fill-triangles', geometry.triangles);
  const fillFeatures = resources.createBuffer('fill-features', geometry.triangleFeatures);
  const fillVertexCount = geometry.triangleFeatures.length;
  const outlineBuffers = Object.fromEntries(
    OUTLINE_STYLE_NAMES.map(style => [
      style,
      {
        buffer: resources.createBuffer(`outline-${style}`, geometry.outlines[style]),
        count: geometry.outlines[style].length / 4
      }
    ])
  ) as Record<(typeof OUTLINE_STYLE_NAMES)[number], {buffer: Buffer; count: number}>;

  // ---- Shared draw records and parameters -----------------------------------------------------
  const eventCapacity = trackCount * MAX_EVENTS_PER_TRACK;
  const eventTracks = resources.createBuffer('event-tracks', eventCapacity * 4);
  const eventCount = resources.createBuffer('event-count', 4);
  const eventOverflow = resources.createBuffer('event-overflow', 4);
  const eventZones = resources.createBuffer('event-zones', eventCapacity * 4);
  const eventTypes = resources.createBuffer('event-types', eventCapacity * 4);
  const eventTimes = resources.createBuffer('event-times', eventCapacity * 4);
  const eventPositions = resources.createBuffer('event-positions', eventCapacity * 8);
  const candidateCount = resources.createBuffer('candidate-count', 4);
  const candidateOverflow = resources.createBuffer('candidate-overflow', 4);
  const trackOverflow = resources.createBuffer('track-overflow', 4);
  const eventListOverflow = resources.createBuffer('event-list-overflow', 4);
  const visitTracks = resources.createBuffer('visit-tracks', VISIT_CAPACITY * 4);
  const visitZones = resources.createBuffer('visit-zones', VISIT_CAPACITY * 4);
  const visitDwell = resources.createBuffer('visit-dwell', VISIT_CAPACITY * 4);
  const visitCount = resources.createBuffer('visit-count', 4);
  const visitOverflow = resources.createBuffer('visit-overflow', 4);
  const eventClock = resources.createParameterBuffer('event-clock', 'float32', 4);

  // The pulses of the zones in view: the events read back, filtered on the CPU, uploaded compact.
  const pulseTracks = resources.createBuffer('pulse-tracks', eventCapacity * 4);
  const pulseTimes = resources.createBuffer('pulse-times', eventCapacity * 4);
  const pulseTypes = resources.createBuffer('pulse-types', eventCapacity * 4);
  const pulsePositions = resources.createBuffer('pulse-positions', eventCapacity * 8);
  const pulseDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'zone-pulse-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  const stopIds = resources.createBuffer('stop-ids', STOP_CAPACITY * 4);
  const stopCount = resources.createBuffer('stop-count', 4);
  const stopOverflow = resources.createBuffer('stop-overflow', 4);
  const stopCentroids = resources.createBuffer('stop-centroids', STOP_CAPACITY * 8);
  const stopDurations = resources.createBuffer('stop-durations', STOP_CAPACITY * 4);
  const stopZones = resources.createBuffer('stop-zones', STOP_CAPACITY * 4);
  const joinOverflow = resources.createBuffer('join-overflow', 4);
  const stopParameters = resources.createParameterBuffer(
    'stop-parameters',
    'float32',
    GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
  );
  // The stops inside the zones in view, compact, for the discs.
  const shownStopCentroids = resources.createBuffer('shown-stop-centroids', STOP_CAPACITY * 8);
  const shownStopDurations = resources.createBuffer('shown-stop-durations', STOP_CAPACITY * 4);
  const stopDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'zone-stop-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  // ---- State -----------------------------------------------------------------------------------
  let destroyed = false;
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'playing', speed: 'playbackSpeed', loop: 'loop'},
    {range: [0, SECONDS_PER_DAY], rate: 1, step: 60}
  );
  let playhead = ctx.options.time;
  let settleStale = true;
  let lastChange = performance.now();
  let selectedGroup = NO_GROUP;
  /** Reduced motion starts paused: the first request to play (a step opening) is declined. */
  let declineFirstPlay = ctx.reducedMotion();
  let followed: FollowedVessel | null = null;
  let eventTable: EventTable | null = null;
  let lastAnnotationTime = Number.NaN;
  let vesselAnnotated = false;
  let readVersion = 0;
  let highlightKey = '';
  let notesKey = '';
  let markersKey = '';
  let pulseKey = '';
  let stopKey = '';
  let legendHighlight: readonly number[] | null = null;
  let table: ClassTable | null = null;
  let insideGroups: GroupStatistics | null = null;
  let stoppedGroups: GroupStatistics | null = null;
  let insideValues: Float64Array | null = null;
  let stoppedValues: Float64Array | null = null;
  const frozenBreaks = new Map<string, FrozenBreaks>();
  let inside: VariantState;
  let stopped: VariantState;

  ctx.setReadout(
    'tracks',
    `${formatCount(trackCount)} tracks / ${formatCount(vessels.vesselCount)} vessels`
  );
  ctx.setReadout(
    'zones',
    `${zoneCount} polygons in ${groups.groupCount} named zones, ${formatCount(edgeCount)} edges`
  );
  ctx.setReadout(
    'approxZones',
    `${groups.kinds.filter(kind => isApproximateZoneKind(kind)).length} hand-drawn zones`
  );
  ctx.setCost({records: vertexCount});
  ctx.setFurniture({
    title: {
      sample: `${formatCount(trackCount)} AIS tracks, ${formatCount(groups.groupCount)} zones`
    }
  });

  const isShown = (group: number) =>
    matchesZoneKindFilter(ctx.options.zoneKind, groups.kinds[group]);
  const getGround = () => ctx.ground();

  function writeStopParameters(): void {
    stopParameters.write(
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: ctx.options.stopSpeedKnots * METERS_PER_KNOT_SECOND,
        stopMinimumDuration: ctx.options.stopMinutes * 60
      })
    );
  }

  function createTable(id: string): ZoneTable {
    return {
      counts: resources.createBuffer(`${id}-zone-counts`, zoneCount * 4),
      sums: resources.createBuffer(`${id}-zone-sums`, zoneCount * 4),
      means: resources.createBuffer(`${id}-zone-means`, zoneCount * 4),
      maximums: resources.createBuffer(`${id}-zone-maximums`, zoneCount * 4)
    };
  }

  function parseFlags(words: Uint32Array, offset: number): SummaryFlags {
    return {
      events: words[offset],
      eventOverflow: words[offset + 1],
      candidates: words[offset + 2],
      candidateOverflow: words[offset + 3],
      trackOverflow: words[offset + 4],
      listOverflow: words[offset + 5],
      stops: words[offset + 6],
      stopOverflow: words[offset + 7],
      joinOverflow: words[offset + 8]
    };
  }

  /** The dense per-zone table and the contributor flags, read back together. */
  function createTableReader(id: string, table: ZoneTable, state: () => VariantState) {
    const bytes = zoneCount * 4;
    return new SummaryReader(
      resources,
      `${id}-table`,
      [
        {buffer: table.counts, size: bytes},
        {buffer: table.sums, size: bytes},
        {buffer: table.maximums, size: bytes},
        {buffer: eventCount, size: 4},
        {buffer: eventOverflow, size: 4},
        {buffer: candidateCount, size: 4},
        {buffer: candidateOverflow, size: 4},
        {buffer: trackOverflow, size: 4},
        {buffer: eventListOverflow, size: 4},
        {buffer: stopCount, size: 4},
        {buffer: stopOverflow, size: 4},
        {buffer: joinOverflow, size: 4}
      ],
      raw => {
        if (destroyed) return;
        const words = new Uint32Array(raw);
        const floats = new Float32Array(raw);
        const target = state();
        target.stats = {
          counts: words.slice(0, zoneCount),
          sums: floats.slice(zoneCount, zoneCount * 2),
          maximums: floats.slice(zoneCount * 2, zoneCount * 3)
        };
        target.flags = parseFlags(words, zoneCount * 3);
        refresh();
      }
    );
  }

  function createVisitReader(id: string, state: () => VariantState) {
    return new SummaryReader(
      resources,
      `${id}-visits`,
      [
        {buffer: visitCount, size: 4},
        {buffer: visitTracks, size: VISIT_CAPACITY * 4},
        {buffer: visitZones, size: VISIT_CAPACITY * 4},
        {buffer: visitDwell, size: VISIT_CAPACITY * 4}
      ],
      raw => {
        if (destroyed) return;
        const words = new Uint32Array(raw);
        const floats = new Float32Array(raw);
        const count = Math.min(words[0], VISIT_CAPACITY);
        state().stays = {
          version: ++readVersion,
          count,
          tracks: words.slice(1, 1 + count),
          zones: words.slice(1 + VISIT_CAPACITY, 1 + VISIT_CAPACITY + count),
          seconds: floats.slice(1 + VISIT_CAPACITY * 2, 1 + VISIT_CAPACITY * 2 + count)
        };
        refresh();
      }
    );
  }

  function createStopReader(id: string, state: () => VariantState) {
    return new SummaryReader(
      resources,
      `${id}-stops`,
      [
        {buffer: stopCount, size: 4},
        {buffer: stopIds, size: STOP_CAPACITY * 4},
        {buffer: stopZones, size: STOP_CAPACITY * 4},
        {buffer: stopDurations, size: STOP_CAPACITY * 4},
        {buffer: stopCentroids, size: STOP_CAPACITY * 8}
      ],
      raw => {
        if (destroyed) return;
        const words = new Uint32Array(raw);
        const floats = new Float32Array(raw);
        const count = Math.min(words[0], STOP_CAPACITY);
        state().stays = {
          version: ++readVersion,
          count,
          tracks: words.slice(1, 1 + count),
          zones: words.slice(1 + STOP_CAPACITY, 1 + STOP_CAPACITY + count),
          seconds: floats.slice(1 + STOP_CAPACITY * 2, 1 + STOP_CAPACITY * 2 + count),
          centroids: floats.slice(1 + STOP_CAPACITY * 3, 1 + STOP_CAPACITY * 3 + count * 2)
        };
        refresh();
      }
    );
  }

  function createEventReader(id: string, capacity: number) {
    return new SummaryReader(
      resources,
      `${id}-events`,
      [
        {buffer: eventCount, size: 4},
        {buffer: eventTracks, size: capacity * 4},
        {buffer: eventZones, size: capacity * 4},
        {buffer: eventTypes, size: capacity * 4},
        {buffer: eventTimes, size: capacity * 4},
        {buffer: eventPositions, size: capacity * 8}
      ],
      raw => {
        if (destroyed) return;
        const words = new Uint32Array(raw);
        const floats = new Float32Array(raw);
        const count = Math.min(words[0], capacity);
        const base = 1;
        eventTable = {
          version: ++readVersion,
          count,
          tracks: words.slice(base, base + count),
          zones: words.slice(base + capacity, base + capacity + count),
          types: words.slice(base + capacity * 2, base + capacity * 2 + count),
          times: floats.slice(base + capacity * 3, base + capacity * 3 + count),
          positions: floats.slice(base + capacity * 4, base + capacity * 4 + count * 2)
        };
        followed = pickFollowedVessel();
        startAtFollowedVessel();
        refresh();
      }
    );
  }

  // ---- Variant graphs ------------------------------------------------------------------------
  function buildInside(eventsPerVessel: number): VariantState {
    const id = `inside-${eventsPerVessel}`;
    const zoneTable = createTable(id);
    const displayValues = resources.createBuffer(
      `${id}-display`,
      new Float32Array(zoneCount).fill(Number.NaN)
    );
    const graph = new GPUCommandGraph<void>(device, {id: `zone-events-${eventsPerVessel}`});
    const capacity = trackCount * eventsPerVessel;
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    addFleetDwellZoneEventsRecipe(graph, {
      id: 'zones',
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
      timestamps: view('timestamps', timestampsBuffer, 'float32', vertexCount),
      trackOffsets: view('offsets', offsetsBuffer, 'uint32', trackCount + 1),
      edgeStarts: importGraphBuffer(graph, 'edge-starts', edgeStartsBuffer, 'float32x2', edgeCount),
      edgeEnds: importGraphBuffer(graph, 'edge-ends', edgeEndsBuffer, 'float32x2', edgeCount),
      edgeZones: view('edge-zones', edgeZonesBuffer, 'uint32', edgeCount),
      zoneCount,
      candidateCapacity: EVENT_CANDIDATE_CAPACITY,
      maxEventsPerTrack: eventsPerVessel,
      outputs: {
        events: {
          output: {
            ids: view('event-tracks', eventTracks, 'uint32', capacity),
            count: view('event-count', eventCount, 'uint32', 1),
            overflow: view('event-overflow', eventOverflow, 'uint32', 1)
          },
          eventZones: view('event-zones', eventZones, 'uint32', capacity),
          eventTypes: view('event-types', eventTypes, 'uint32', capacity),
          eventTimes: view('event-times', eventTimes, 'float32', capacity),
          eventPositions: importGraphBuffer(
            graph,
            'event-positions',
            eventPositions,
            'float32x2',
            capacity
          )
        },
        diagnostics: {
          candidateCount: view('candidate-count', candidateCount, 'uint32', 1),
          candidateOverflow: view('candidate-overflow', candidateOverflow, 'uint32', 1),
          trackOverflow: view('track-overflow', trackOverflow, 'uint32', 1),
          eventOverflow: view('event-list-overflow', eventListOverflow, 'uint32', 1)
        },
        table: {
          counts: view('zone-counts', zoneTable.counts, 'uint32', zoneCount),
          sumValues: view('zone-sums', zoneTable.sums, 'float32', zoneCount),
          means: view('zone-means', zoneTable.means, 'float32', zoneCount),
          maximums: view('zone-maximums', zoneTable.maximums, 'float32', zoneCount)
        }
      },
      scratch: {
        visitTable: {
          output: {
            ids: view('visit-tracks', visitTracks, 'uint32', VISIT_CAPACITY),
            count: view('visit-count', visitCount, 'uint32', 1),
            overflow: view('visit-overflow', visitOverflow, 'uint32', 1)
          },
          zones: view('visit-zones', visitZones, 'uint32', VISIT_CAPACITY),
          dwellTimes: view('visit-dwell', visitDwell, 'float32', VISIT_CAPACITY)
        }
      }
    });
    const compiled = resources.track(graph.compile());
    const state: VariantState = {
      variant: 'inside',
      compiled,
      table: zoneTable,
      displayValues,
      tableReader: createTableReader(id, zoneTable, () => state),
      listReader: createVisitReader(id, () => state),
      eventReader: createEventReader(id, capacity),
      needsEncode: true,
      stats: null,
      flags: EMPTY_FLAGS,
      stays: null,
      eventsPerVessel
    };
    return state;
  }

  function buildStopped(): VariantState {
    const zoneTable = createTable('stopped');
    const displayValues = resources.createBuffer(
      'stopped-display',
      new Float32Array(zoneCount).fill(Number.NaN)
    );
    const graph = new GPUCommandGraph<void>(device, {id: 'zone-stops'});
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    addFleetDwellRecipe(graph, {
      id: 'zone-stops',
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
      timestamps: view('timestamps', timestampsBuffer, 'float32', vertexCount),
      trackOffsets: view('offsets', offsetsBuffer, 'uint32', trackCount + 1),
      parameters: stopParameters.importToGraph(graph),
      stopCapacity: STOP_CAPACITY,
      zones: {
        polygonPositions: importGraphBuffer(
          graph,
          'polygon-positions',
          polygonBuffers.positions,
          'float32x2',
          zones.polygonPositions.length / 2
        ),
        featureOffsets: view(
          'feature-offsets',
          polygonBuffers.featureOffsets,
          'uint32',
          zones.featureOffsets.length
        ),
        polygonOffsets: view(
          'polygon-offsets',
          polygonBuffers.polygonOffsets,
          'uint32',
          zones.polygonOffsets.length
        ),
        ringOffsets: view(
          'ring-offsets',
          polygonBuffers.ringOffsets,
          'uint32',
          zones.ringOffsets.length
        ),
        candidateCapacity: STOP_CANDIDATE_CAPACITY
      },
      outputs: {
        stops: {
          ids: view('stop-ids', stopIds, 'uint32', STOP_CAPACITY),
          count: view('stop-count', stopCount, 'uint32', 1),
          overflow: view('stop-overflow', stopOverflow, 'uint32', 1),
          centroids: importGraphBuffer(
            graph,
            'stop-centroids',
            stopCentroids,
            'float32x2',
            STOP_CAPACITY
          ),
          durations: view('stop-durations', stopDurations, 'float32', STOP_CAPACITY)
        },
        joinOverflow: view('join-overflow', joinOverflow, 'uint32', 1),
        table: {
          counts: view('zone-counts', zoneTable.counts, 'uint32', zoneCount),
          sumValues: view('zone-sums', zoneTable.sums, 'float32', zoneCount),
          means: view('zone-means', zoneTable.means, 'float32', zoneCount),
          maximums: view('zone-maximums', zoneTable.maximums, 'float32', zoneCount)
        }
      },
      scratch: {stopZones: view('stop-zones', stopZones, 'uint32', STOP_CAPACITY)}
    });
    const compiled = resources.track(graph.compile());
    const state: VariantState = {
      variant: 'stopped',
      compiled,
      table: zoneTable,
      displayValues,
      tableReader: createTableReader('stopped', zoneTable, () => state),
      listReader: createStopReader('stopped', () => state),
      eventReader: null,
      needsEncode: true,
      stats: null,
      flags: EMPTY_FLAGS,
      stays: null
    };
    return state;
  }

  const markChanged = () => {
    lastChange = performance.now();
    settleStale = true;
  };

  // ---- Class table, frozen once per unit --------------------------------------------------------
  /**
   * The breaks of a metric and unit, computed once from the inside statistics and kept: toggling
   * the filter, the variant or the thresholds recolours nothing but the zones, so a colour change
   * is a data change.
   */
  function getFrozenBreaks(metric: ZoneMetric, unit: ZoneUnit): FrozenBreaks | null {
    const key = getBreaksKey(metric, unit);
    const known = frozenBreaks.get(key);
    if (known) return known;
    if (!insideGroups) return null;
    const referenceUnit: ZoneUnit = unit === 'density' ? 'density' : 'total';
    const values = getGroupValues(insideGroups, metric, referenceUnit, geometry.groupAreasKm2);
    const finite = Array.from(values).filter(Number.isFinite);
    if (finite.length === 0) return null;
    const manual = metric === 'mean' || metric === 'longest';
    const frozen: FrozenBreaks = {
      breaks: computeClassBreaks(finite, metric),
      zonesWithTime: finite.length,
      extent: getExtent(finite),
      method: manual
        ? 'Manual breaks chosen for waiting times'
        : `Quantiles of the ${finite.length} zones with time`
    };
    frozenBreaks.set(key, frozen);
    return frozen;
  }

  function getGroupValuesFor(stats: GroupStatistics | null): Float64Array | null {
    if (!stats) return null;
    return getGroupValues(stats, ctx.options.metric, ctx.options.unit, geometry.groupAreasKm2);
  }

  function writeDisplay(state: VariantState, values: Float64Array | null): void {
    const out = new Float32Array(zoneCount).fill(Number.NaN);
    if (values) {
      for (let zone = 0; zone < zoneCount; zone++) {
        const group = groups.groupOfZone[zone];
        const value = values[group];
        if (Number.isFinite(value) && isShown(group)) out[zone] = value;
      }
    }
    state.displayValues.write(out);
  }

  /** The variant the map shows when it is not a swipe. */
  const getPrimary = () => (ctx.options.variant === 'stopped' ? stopped : inside);
  const getPrimaryValues = () => (ctx.options.variant === 'stopped' ? stoppedValues : insideValues);
  const getDefinition = () =>
    ctx.options.compareVariants ? 'both' : ctx.options.variant === 'stopped' ? 'stopped' : 'inside';

  // ---- Readouts, charts, notes ------------------------------------------------------------------
  function getRanked(values: Float64Array | null): number[] {
    if (!values) return [];
    const ranked: number[] = [];
    for (let group = 0; group < groups.groupCount; group++) {
      if (Number.isFinite(values[group]) && isShown(group)) ranked.push(group);
    }
    return ranked.sort((a, b) => values[b] - values[a]);
  }

  function getMedian(values: number[]): number {
    if (values.length === 0) return Number.NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function getTableColor(value: number) {
    if (!table) return undefined;
    const index = getClassIndexOf(table, value);
    return index >= 0 ? table.colors[index] : undefined;
  }

  function updateRanking(values: Float64Array | null): void {
    const {metric, unit} = ctx.options;
    const ranked = getRanked(values);
    const top = ranked.slice(0, RANKED_ZONES);
    const labels = getZoneValueLabels(metric, unit, getDefinition());
    if (!values || top.length === 0) {
      ctx.setChart('rankChart', null);
      ctx.setReadout('topZone', 'no zone has time');
      ctx.setReadout('busiestArea', '–');
      ctx.setReadout('typicalArea', '–');
      return;
    }
    ctx.setChart('rankChart', {
      kind: 'bars',
      horizontal: true,
      values: top.map(group => values[group]),
      labels: top.map(group => groups.names[group]),
      colors: top.map(group => getTableColor(values[group]) ?? ([128, 128, 128, 255] as const)),
      xLabel: `${labels.unit}${labels.basis ? ` ${labels.basis}` : ''}`,
      formatX: value => formatQuantity(value),
      onBarClick: index => selectGroup(top[index] === selectedGroup ? NO_GROUP : top[index]),
      description: `The ${top.length} zones with the highest ${labels.title.toLowerCase()}, coloured by class.`
    });
    ctx.setReadout(
      'topZone',
      `${groups.names[top[0]]}: ${formatZoneValue(values[top[0]], metric, unit)}`
    );
    const withTime = Array.from({length: groups.groupCount}, (_, group) => group).filter(group =>
      Number.isFinite(values[group])
    );
    ctx.setReadout(
      'busiestArea',
      formatArea(getMedian(top.map(group => geometry.groupAreasKm2[group])) * 1e6)
    );
    ctx.setReadout(
      'typicalArea',
      formatArea(getMedian(withTime.map(group => geometry.groupAreasKm2[group])) * 1e6)
    );
  }

  function getKindHours(stats: GroupStatistics | null) {
    const hours: Record<'anchorage' | 'channel' | 'approximate' | 'all', number> = {
      anchorage: 0,
      channel: 0,
      approximate: 0,
      all: 0
    };
    if (!stats) return hours;
    for (let group = 0; group < groups.groupCount; group++) {
      const value = stats.sums[group] / 3600;
      hours.all += value;
      if (groups.kinds[group] === 'anchorage') hours.anchorage += value;
      else if (groups.kinds[group] === 'channel') hours.channel += value;
      else hours.approximate += value;
    }
    return hours;
  }

  function updateDefinitionReadouts(): void {
    const stops = stopped.stays;
    if (stops) {
      let inZones = 0;
      for (let row = 0; row < stops.count; row++) if (stops.zones[row] < zoneCount) inZones++;
      ctx.setReadout(
        'stopCount',
        `${formatCount(inZones)} stops in the zones, ${formatCount(stops.count - inZones)} elsewhere`
      );
    } else {
      ctx.setReadout('stopCount', null);
    }
    const insideHours = getKindHours(insideGroups);
    const stoppedHours = getKindHours(stoppedGroups);
    ctx.setReadout(
      'insideHours',
      insideGroups ? `${formatCount(insideHours.all)} vessel-hours` : null
    );
    ctx.setReadout(
      'stoppedHours',
      stoppedGroups ? `${formatCount(stoppedHours.all)} vessel-hours` : null
    );
    const keep = (kind: 'anchorage' | 'channel') =>
      insideGroups && stoppedGroups && insideHours[kind] > 0
        ? formatPercent(stoppedHours[kind] / insideHours[kind])
        : null;
    ctx.setReadout('anchorageKeep', keep('anchorage'));
    ctx.setReadout('channelKeep', keep('channel'));
    ctx.setReadout(
      'approxShare',
      insideGroups && insideHours.all > 0
        ? formatPercent(insideHours.approximate / insideHours.all)
        : null
    );
    if (insideGroups && stoppedGroups) {
      const kindRows = [
        {label: 'Anchorages', kind: 'anchorage' as const},
        {label: 'Channels', kind: 'channel' as const},
        {label: 'Terminals', kind: 'terminal' as const},
        {label: 'Narrows gate', kind: 'gate' as const},
        {label: 'Tour area', kind: 'tourist' as const},
        {label: 'Ferry lane', kind: 'ferry' as const}
      ];
      const perKind = (stats: GroupStatistics, kind: string) => {
        let total = 0;
        for (let group = 0; group < groups.groupCount; group++) {
          if (groups.kinds[group] === kind) total += stats.sums[group] / 3600;
        }
        return total;
      };
      ctx.setChart('kindChart', {
        kind: 'dumbbell',
        aLabel: 'Inside',
        bLabel: 'Stopped',
        xLabel: 'vessel-hours in the zones of each kind',
        formatX: value => formatQuantity(value),
        rows: kindRows.map(row => ({
          label: row.label,
          a: perKind(insideGroups as GroupStatistics, row.kind),
          b: perKind(stoppedGroups as GroupStatistics, row.kind),
          highlight: row.kind === 'channel' || row.kind === 'anchorage'
        })),
        description:
          'Vessel-hours per kind of zone, counted as any time inside and as time spent stopped.'
      });
    } else {
      ctx.setChart('kindChart', null);
    }
  }

  function describeStay(row: {track: number; group: number; seconds: number}): string {
    const category = VESSEL_CATEGORIES[vessels.category[row.track]];
    const noun = VESSEL_CATEGORY_LABELS[category].split(' (')[0].toLowerCase();
    return `${formatDwell(row.seconds)}: a ${noun} in ${groups.names[row.group]}`;
  }

  function updateStayReadouts(): void {
    const primary = getPrimary();
    const stays = primary.stays;
    const minimumStop = ctx.options.stopMinutes * 60;
    if (!stays) {
      ctx.setReadout('longestVisit', null);
      ctx.setReadout('passageShare', null);
      ctx.setReadout('meanVisit', null);
      return;
    }
    let longest: {track: number; group: number; seconds: number} | null = null;
    let total = 0;
    let shown = 0;
    let passages = 0;
    for (let row = 0; row < stays.count; row++) {
      const zone = stays.zones[row];
      if (zone >= zoneCount) continue;
      const group = groups.groupOfZone[zone];
      if (!isShown(group)) continue;
      const seconds = stays.seconds[row];
      shown++;
      total += seconds;
      if (seconds < minimumStop) passages++;
      if (!longest || seconds > longest.seconds)
        longest = {track: stays.tracks[row], group, seconds};
    }
    ctx.setReadout('longestVisit', longest ? describeStay(longest) : 'none');
    ctx.setReadout('passageShare', shown > 0 ? formatPercent(passages / shown) : null);
    ctx.setReadout('meanVisit', shown > 0 ? formatDwell(total / shown) : null);
  }

  function updateEventReadouts(): void {
    const flags = inside.flags;
    const events = eventTable;
    let shown = 0;
    if (events) {
      for (let row = 0; row < events.count; row++) {
        if (isShown(groups.groupOfZone[events.zones[row]])) shown++;
      }
    }
    ctx.setReadout(
      'events',
      events ? `${formatCount(shown)} crossings in the zones in view` : null
    );
    ctx.setReadout(
      'eventsKept',
      `${formatCount(flags.events)} kept${flags.eventOverflow ? ' (OVERFLOW)' : ''}`
    );
    ctx.setReadout(
      'candidates',
      `${formatCount(flags.candidates)} of ${formatCount(EVENT_CANDIDATE_CAPACITY)}${flags.candidateOverflow ? ' (OVERFLOW)' : ''}`
    );
    const stoppedFlags = stopped.flags;
    ctx.setReadout(
      'overflow',
      [
        flags.candidateOverflow ? 'candidate scratch' : '',
        flags.trackOverflow
          ? 'a vessel exceeded the per-vessel event cap (statistics stay exact)'
          : '',
        flags.listOverflow ? 'event list' : '',
        stoppedFlags.joinOverflow || stoppedFlags.stopOverflow ? 'stop list or join' : ''
      ]
        .filter(Boolean)
        .join(', ') || 'none'
    );
  }

  /** The note annotations of the step: sizes and live values, or the hand-drawn zones. */
  function updateNotes(values: Float64Array | null): void {
    const {notes, metric, unit} = ctx.options;
    if (notes === 'none' || !values) {
      if (notesKey) ctx.setAnnotations('zone-notes', null);
      notesKey = '';
      return;
    }
    const annotations: MapAnnotation[] = [];
    if (notes === 'size') {
      let largest = -1;
      for (let group = 0; group < groups.groupCount; group++) {
        if (!isShown(group)) continue;
        if (largest < 0 || geometry.groupAreasKm2[group] > geometry.groupAreasKm2[largest]) {
          largest = group;
        }
      }
      const top = getRanked(values)[0];
      const describe = (group: number, lead: string): MapAnnotation => ({
        kind: 'note',
        id: `note:${lead.toLowerCase()}`,
        coordinate: geometry.groupLabelPoints[group],
        title: `${lead}: ${groups.names[group]}`,
        text: `${formatArea(geometry.groupAreasKm2[group] * 1e6)}, ${formatZoneValue(values[group], metric, unit)}`,
        priority: 5
      });
      if (largest >= 0) annotations.push(describe(largest, 'Largest'));
      if (top !== undefined && top !== largest) annotations.push(describe(top, 'Highest'));
    } else {
      let biggest = -1;
      for (let group = 0; group < groups.groupCount; group++) {
        if (!isApproximateZoneKind(groups.kinds[group])) continue;
        annotations.push({
          kind: 'outline',
          rings: geometry.groupOuterRings[group],
          dashed: true,
          tone: 'accent',
          id: `approx:${group}`
        });
        if (biggest < 0 || geometry.groupAreasKm2[group] > geometry.groupAreasKm2[biggest]) {
          biggest = group;
        }
      }
      if (insideGroups && biggest >= 0) {
        const hours = getKindHours(insideGroups);
        annotations.push({
          kind: 'note',
          id: 'note:hand-drawn',
          coordinate: geometry.groupLabelPoints[biggest],
          title: `${formatPercent(hours.approximate / Math.max(hours.all, 1e-9))} of the hours`,
          text: 'fall in zones drawn by hand',
          priority: 5
        });
      }
    }
    const key = JSON.stringify(annotations);
    if (key === notesKey) return;
    notesKey = key;
    ctx.setAnnotations('zone-notes', annotations);
  }

  // ---- The followed vessel (step 3) ----------------------------------------------------------------
  function pickFollowedVessel(): FollowedVessel | null {
    const events = eventTable;
    if (!events || gateGroup < 0 || events.count === 0) return null;
    const gateZones = new Set(groups.members[gateGroup]);
    const perTrack = new Map<number, number[]>();
    for (let row = 0; row < events.count; row++) {
      const list = perTrack.get(events.tracks[row]);
      if (list) list.push(row);
      else perTrack.set(events.tracks[row], [row]);
    }
    const preference = [3, 0, 1, 2, 6, 5, 4];
    let best: {track: number; rows: number[]; rank: number} | null = null;
    for (const [track, rows] of perTrack) {
      rows.sort((a, b) => events.times[a] - events.times[b]);
      const gateRows = rows.filter(row => gateZones.has(events.zones[row]));
      // A round trip through the Narrows: two transits, four crossings, entering first, and no
      // event lost to the per-vessel cap.
      if (gateRows.length !== 4 || events.types[gateRows[0]] !== ENTER) continue;
      if (rows.length >= (inside.eventsPerVessel ?? MAX_EVENTS_PER_TRACK)) continue;
      const rank = preference.indexOf(vessels.category[track]) * 1000 + rows.length;
      if (!best || rank < best.rank) best = {track, rows, rank};
    }
    if (!best) return null;
    const start = trackStartTimes[best.track];
    const end = vessels.timestamps[vessels.offsets[best.track + 1] - 1];
    const crossings = best.rows
      .filter(row => gateZones.has(events.zones[row]))
      .map(row => ({
        time: start + events.times[row],
        type: events.types[row],
        x: events.positions[row * 2],
        y: events.positions[row * 2 + 1]
      }));
    // Visits: pair enters and exits per zone group, in time order.
    const open = new Map<number, number>();
    const visits: FollowedVessel['visits'] = [];
    for (const row of best.rows) {
      const group = groups.groupOfZone[events.zones[row]];
      const time = start + events.times[row];
      if (events.types[row] === ENTER) {
        open.set(group, time);
      } else {
        visits.push({group, start: open.get(group) ?? start, end: time});
        open.delete(group);
      }
    }
    for (const [group, time] of open) visits.push({group, start: time, end});
    visits.sort((a, b) => a.start - b.start);
    return {
      track: best.track,
      category:
        VESSEL_CATEGORY_LABELS[VESSEL_CATEGORIES[vessels.category[best.track]]].split(' (')[0],
      crossings,
      visits
    };
  }

  /** Puts the clock shortly before the followed vessel first enters the gate. */
  function startAtFollowedVessel(): void {
    const vessel = followed;
    if (!vessel || !ctx.options.showVessel) return;
    ctx.setOptions({time: Math.max(0, Math.floor((vessel.crossings[0].time - 25 * 60) / 60) * 60)});
  }

  function buildGantt(vessel: FollowedVessel) {
    const rowOrder: number[] = [];
    for (const visit of vessel.visits)
      if (!rowOrder.includes(visit.group)) rowOrder.push(visit.group);
    const from = Math.min(...vessel.visits.map(visit => visit.start));
    const to = Math.max(...vessel.visits.map(visit => visit.end));
    const first = Math.floor(from / 3600) * 3600;
    const last = Math.ceil(to / 3600) * 3600;
    const left = 112;
    const right = 312;
    const rowHeight = 14;
    const top = 6;
    const toX = (time: number) =>
      left + ((time - first) / Math.max(last - first, 1)) * (right - left);
    const height = top + rowOrder.length * rowHeight + 22;
    const escapeMarkup = (text: string) =>
      text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const shorten = (text: string) => (text.length > 19 ? `${text.slice(0, 18)}…` : text);
    const parts: string[] = [];
    rowOrder.forEach((group, row) => {
      const y = top + row * rowHeight;
      const isGate = group === gateGroup;
      parts.push(
        `<text class="${isGate ? 'diagram-accent' : 'diagram-muted'}" x="${left - 6}" y="${y + 10}" text-anchor="end">${escapeMarkup(shorten(groups.names[group]))}</text>`,
        `<line class="diagram-muted" x1="${left}" y1="${y + rowHeight - 1}" x2="${right}" y2="${y + rowHeight - 1}" opacity="0.25"/>`
      );
      for (const visit of vessel.visits) {
        if (visit.group !== group) continue;
        const x = toX(visit.start);
        const width = Math.max(1.6, toX(visit.end) - x);
        parts.push(
          `<rect class="${isGate ? 'diagram-accent' : 'diagram-fill'}" x="${x.toFixed(1)}" y="${y + 2}" width="${width.toFixed(1)}" height="${rowHeight - 5}"/>`
        );
      }
    });
    const axisY = top + rowOrder.length * rowHeight + 4;
    const tickStep = last - first > 8 * 3600 ? 2 : 1;
    for (let time = first; time <= last; time += tickStep * 3600) {
      const x = toX(time);
      parts.push(
        `<line class="diagram-muted" x1="${x.toFixed(1)}" y1="${axisY - 3}" x2="${x.toFixed(1)}" y2="${axisY}"/>`,
        `<text class="diagram-muted" x="${x.toFixed(1)}" y="${axisY + 11}" text-anchor="middle">${formatUtcClock(time)}</text>`
      );
    }
    return {
      kind: 'diagram' as const,
      width: 320,
      height,
      svg: parts.join('\n'),
      description: `Gantt chart of one ${vessel.category.toLowerCase()}: each bar is a visit from the enter event to the exit event of a zone; times are UTC.`
    };
  }

  function updateFollowed(): void {
    const vessel = followed;
    ctx.setReadout('followed', vessel ? `a ${vessel.category.toLowerCase()}` : null);
    ctx.setChart('visitChart', vessel ? buildGantt(vessel) : null);
    if (!ctx.options.showVessel || !vessel) {
      if (vesselAnnotated) {
        markersKey = '';
        ctx.setAnnotations('vessel', null);
        ctx.setAnnotationTime(null);
        lastAnnotationTime = Number.NaN;
        vesselAnnotated = false;
      }
      selectionHighlights();
      return;
    }
    const markers: MapAnnotation[] = vessel.crossings.map((crossing, index) => ({
      kind: 'marker',
      coordinate: vessels.unproject(crossing.x, crossing.y),
      number: index + 1,
      text: `Crossing ${index + 1}: ${crossing.type === ENTER ? 'enters' : 'exits'} the gate zone`,
      timeRange: [crossing.time, SECONDS_PER_DAY + 1],
      id: `vessel-crossing:${index}`
    }));
    const key = JSON.stringify(markers);
    if (key !== markersKey) {
      markersKey = key;
      ctx.setAnnotations('vessel', markers);
    }
    vesselAnnotated = true;
    selectionHighlights();
  }

  /** The followed track and the selected zone, as achromatic highlights. */
  function selectionHighlights(): void {
    const vessel = followed;
    const key = `${vessel && ctx.options.showVessel ? vessel.track : -1}|${selectedGroup}`;
    if (key === highlightKey) return;
    highlightKey = key;
    const list: MapHighlight[] = [];
    if (vessel && ctx.options.showVessel) {
      const coordinates: LngLat[] = [];
      for (
        let vertex = vessels.offsets[vessel.track];
        vertex < vessels.offsets[vessel.track + 1];
        vertex++
      ) {
        coordinates.push([vessels.lngLat[vertex * 2], vessels.lngLat[vertex * 2 + 1]]);
      }
      list.push({kind: 'line', coordinates});
    }
    if (selectedGroup !== NO_GROUP) {
      list.push({kind: 'polygon', rings: geometry.groupRings[selectedGroup]});
    }
    ctx.setHighlight(list.length ? list : null);
  }

  function selectGroup(group: number): void {
    selectedGroup = group;
    selectionHighlights();
  }

  // ---- Compact pulses and stops --------------------------------------------------------------------
  function rebuildPulses(): void {
    const events = eventTable;
    if (!events) return;
    const key = `${events.version}|${ctx.options.zoneKind}`;
    if (key === pulseKey) return;
    pulseKey = key;
    const tracks = new Uint32Array(events.count);
    const times = new Float32Array(events.count);
    const types = new Uint32Array(events.count);
    const positions = new Float32Array(events.count * 2);
    const hourly = new Array<number>(24).fill(0);
    let count = 0;
    for (let row = 0; row < events.count; row++) {
      if (!isShown(groups.groupOfZone[events.zones[row]])) continue;
      tracks[count] = events.tracks[row];
      times[count] = events.times[row];
      types[count] = events.types[row];
      positions[count * 2] = events.positions[row * 2];
      positions[count * 2 + 1] = events.positions[row * 2 + 1];
      const hour = Math.floor((trackStartTimes[events.tracks[row]] + events.times[row]) / 3600);
      if (hour >= 0 && hour < 24) hourly[hour]++;
      count++;
    }
    if (count > 0) {
      pulseTracks.write(tracks.subarray(0, count));
      pulseTimes.write(times.subarray(0, count));
      pulseTypes.write(types.subarray(0, count));
      pulsePositions.write(positions.subarray(0, count * 2));
    }
    pulseDraw.buffer.write(Uint32Array.of(6, count, 0, 0));
    ctx.setTimelineData({domain: [0, SECONDS_PER_DAY], histogram: hourly});
  }

  function rebuildStops(): void {
    const stays = stopped.stays;
    if (!stays?.centroids) return;
    const key = `${stays.version}|${ctx.options.zoneKind}`;
    if (key === stopKey) return;
    stopKey = key;
    const centroids = new Float32Array(stays.count * 2);
    const durations = new Float32Array(stays.count);
    let count = 0;
    for (let row = 0; row < stays.count; row++) {
      const zone = stays.zones[row];
      if (zone >= zoneCount || !isShown(groups.groupOfZone[zone])) continue;
      centroids[count * 2] = stays.centroids[row * 2];
      centroids[count * 2 + 1] = stays.centroids[row * 2 + 1];
      durations[count] = stays.seconds[row];
      count++;
    }
    if (count > 0) {
      shownStopCentroids.write(centroids.subarray(0, count * 2));
      shownStopDurations.write(durations.subarray(0, count));
    }
    stopDraw.buffer.write(Uint32Array.of(6, count, 0, 0));
  }

  // ---- The one place everything is recomputed ------------------------------------------------------
  function refresh(): void {
    if (destroyed) return;
    const options = ctx.options;
    insideGroups = inside.stats ? reduceToGroups(inside.stats, groups) : null;
    stoppedGroups = stopped.stats ? reduceToGroups(stopped.stats, groups) : null;
    insideValues = getGroupValuesFor(insideGroups);
    stoppedValues = getGroupValuesFor(stoppedGroups);
    writeDisplay(inside, insideValues);
    writeDisplay(stopped, stoppedValues);

    const frozen = getFrozenBreaks(options.metric, options.unit);
    const ground = getGround();
    const definition = getDefinition();
    const labels = getZoneValueLabels(options.metric, options.unit, definition);
    if (frozen) {
      const factor = getUnitBreakFactor(options.metric, options.unit);
      table = makeZoneClassTable({
        breaks: frozen.breaks,
        ground,
        metric: options.metric,
        unit: options.unit,
        definition,
        extent: [frozen.extent[0] * factor, frozen.extent[1] * factor],
        method:
          options.metric === 'total' && options.unit === 'present'
            ? `${frozen.method}; vessels present = vessel-hours / 24, so it shares the breaks of the total`
            : frozen.method
      });
      const primary = options.compareVariants ? insideValues : getPrimaryValues();
      const shownValues = primary
        ? Array.from(primary).filter((value, group) => Number.isFinite(value) && isShown(group))
        : [];
      ctx.setLegendData('zoneLegend', {
        table,
        counts: getClassCounts(shownValues, table.breaks),
        title: labels.title,
        basis: labels.basis,
        ground,
        compare: options.compareVariants
      });
    } else {
      table = null;
      ctx.setLegendData('zoneLegend', null);
    }

    const rankValues = options.compareVariants ? insideValues : getPrimaryValues();
    updateRanking(rankValues);
    ctx.setReadout(
      'zonesUsed',
      rankValues
        ? `${Array.from(rankValues).filter(Number.isFinite).length} of ${groups.groupCount} zones`
        : null
    );
    updateDefinitionReadouts();
    updateStayReadouts();
    updateEventReadouts();
    updateNotes(rankValues);
    rebuildPulses();
    rebuildStops();
    updateFollowed();
    ctx.setCost({
      records: vertexCount,
      passes: inside.compiled.stats.nodeOrder.length + stopped.compiled.stats.nodeOrder.length
    });
    ctx.requestLayers();
  }

  // ---- Tooltip -------------------------------------------------------------------------------------
  function describeGroup(group: number): TooltipContent {
    const {metric, unit} = ctx.options;
    const rows: TooltipRow[] = [];
    const area = geometry.groupAreasKm2[group];
    const labels = getZoneValueLabels(metric, unit, getDefinition());
    const swatchOf = (value: number) => getTableColor(value);
    const insideValue = insideValues?.[group] ?? Number.NaN;
    const stoppedValue = stoppedValues?.[group] ?? Number.NaN;
    if (ctx.options.compareVariants) {
      rows.push(
        {
          label: 'Inside',
          value: formatZoneValue(insideValue, metric, unit),
          swatch: swatchOf(insideValue),
          emphasis: true
        },
        {
          label: 'Stopped',
          value: formatZoneValue(stoppedValue, metric, unit),
          swatch: swatchOf(stoppedValue)
        }
      );
    } else {
      const value = getPrimaryValues()?.[group] ?? Number.NaN;
      rows.push({
        label: labels.title,
        value: formatZoneValue(value, metric, unit),
        swatch: swatchOf(value),
        emphasis: true
      });
    }
    if (insideGroups) {
      const hours = insideGroups.sums[group] / 3600;
      const visits = insideGroups.counts[group];
      rows.push(
        {
          label: 'Present on average',
          value: formatQuantity(hours / (SECONDS_PER_DAY / 3600)),
          unit: 'vessels'
        },
        {
          label: 'Visits',
          value: formatCount(visits),
          unit:
            visits > 0
              ? `mean ${formatDwell(insideGroups.sums[group] / visits)}, longest ${formatDwell(insideGroups.maximums[group])}`
              : undefined
        },
        {
          label: 'Area',
          value: formatQuantity(area),
          unit: `km² · ${formatQuantity(hours / Math.max(area, 1e-9))} vessel-hours per km²`
        }
      );
    }
    if (stoppedGroups && stoppedGroups.counts[group] > 0) {
      rows.push({
        label: 'Stopped',
        value: formatQuantity(stoppedGroups.sums[group] / 3600),
        unit: `vessel-hours in ${formatCount(stoppedGroups.counts[group])} stops`
      });
    }
    const pieces = groups.members[group].length;
    return {
      title: groups.names[group],
      subtitle: `${ZONE_KIND_LABELS[groups.kinds[group]]}${pieces > 1 ? ` · ${pieces} pieces added up` : ''}`,
      rows,
      anchor: geometry.groupLabelPoints[group],
      highlight: {kind: 'polygon', rings: geometry.groupRings[group]}
    };
  }

  // ---- Boot --------------------------------------------------------------------------------------
  writeStopParameters();
  inside = buildInside(Number(ctx.options.eventsPerVessel));
  stopped = buildStopped();

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [inside.compiled, stopped.compiled],

    setOption(id, _value, state) {
      switch (id) {
        case 'eventsPerVessel':
          inside = buildInside(Number(state.eventsPerVessel));
          eventTable = null;
          followed = null;
          markChanged();
          refresh();
          break;
        case 'metric':
        case 'unit':
        case 'zoneKind':
        case 'variant':
        case 'compareVariants':
        case 'notes':
          refresh();
          break;
        case 'showVessel':
          startAtFollowedVessel();
          refresh();
          break;
        case 'stopSpeedKnots':
        case 'stopMinutes':
          writeStopParameters();
          stopped.needsEncode = true;
          markChanged();
          refresh();
          break;
        case 'playing':
          if (state.playing && declineFirstPlay) {
            declineFirstPlay = false;
            ctx.setOptions({playing: false});
          }
          break;
        case 'showEvents':
        case 'showStops':
        case 'showTracks':
        case 'fillOpacity':
          ctx.requestLayers();
          break;
        default:
          break;
      }
    },

    onThemeChange() {
      refresh();
    },

    onGroundChange() {
      refresh();
    },

    onLegendFilter(_id, classes) {
      legendHighlight = classes;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      // Paused, the clock sits on the slider, so every story step and deep link is deterministic.
      playhead = clock.advance(frame);
      ctx.setReadout('clock', `${formatUtcClock(playhead)} UTC (${formatEasternClock(playhead)})`);
      eventClock.write(Float32Array.of(playhead, PULSE_SECONDS, 0, 0));
      if (ctx.options.showVessel && followed && Math.abs(playhead - lastAnnotationTime) > 20) {
        lastAnnotationTime = playhead;
        ctx.setAnnotationTime(playhead);
      }
      for (const state of [inside, stopped]) {
        if (!state.needsEncode) continue;
        state.compiled.encode(commandEncoder, {parameters: undefined});
        state.needsEncode = false;
        settleStale = true;
        lastChange = performance.now();
      }
      if (settleStale && performance.now() - lastChange > SETTLE_MILLISECONDS) {
        for (const state of [inside, stopped]) {
          state.tableReader.markStale();
          state.listReader.markStale();
          state.eventReader?.markStale();
        }
        settleStale = false;
      }
      for (const state of [inside, stopped]) {
        state.tableReader.flush(commandEncoder);
        state.listReader.flush(commandEncoder);
        state.eventReader?.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const ground = getGround();
      const layers: Layer[] = [];
      if (table) {
        const classProps = getClassTableLayerProps(table);
        const fill = (state: VariantState, side?: 'a' | 'b') =>
          new SpatialAnalysisPolygonLayer({
            id: `zone-fill-${side ?? state.variant}`,
            coordinateOrigin,
            triangles: fillTriangles,
            features: fillFeatures,
            vertexCount: fillVertexCount,
            values: state.displayValues,
            valueFormat: 'float32',
            colormap: 'uniform',
            noDataColor: [0, 0, 0, 0],
            ...classProps,
            highlightClasses: legendHighlight,
            compareSide: side,
            opacity: options.fillOpacity
          });
        if (options.compareVariants) layers.push(fill(inside, 'a'), fill(stopped, 'b'));
        else layers.push(fill(getPrimary()));
      }
      if (options.showTracks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'zone-tracks',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 0.7,
            color: getTrackInk(ground)
          })
        );
      }
      const styles = getOutlineStyles(ground);
      for (const style of OUTLINE_STYLE_NAMES) {
        const lines = outlineBuffers[style];
        if (lines.count === 0) continue;
        const look = styles[style];
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `zone-outline-${style}`,
            coordinateOrigin,
            segments: lines.buffer,
            instanceCount: lines.count,
            widthPixels: look.widthPixels,
            color: look.color,
            ...(look.casing
              ? {outlineColor: look.casing, outlineWidthPixels: look.casingPixels}
              : {}),
            ...(look.dashArray ? {dashArray: look.dashArray, cap: 'butt' as const} : {})
          })
        );
      }
      if (options.showEvents) {
        const inks = getEventInks(ground);
        layers.push(
          new ZoneEventMarkerLayer({
            id: 'zone-events',
            coordinateOrigin,
            positions: pulsePositions,
            eventTracks: pulseTracks,
            eventTimes: pulseTimes,
            eventTypes: pulseTypes,
            trackStartTimes: trackStartBuffer,
            clock: eventClock.buffer,
            drawCommands: pulseDraw,
            enterColor: inks.enter,
            exitColor: inks.exit,
            sizePixels: 7
          })
        );
      }
      if (options.showStops) {
        layers.push(
          new StopMarkerLayer({
            id: 'zone-stops',
            coordinateOrigin,
            centroids: shownStopCentroids,
            durations: shownStopDurations,
            drawCommands: stopDraw,
            baseRadiusPixels: 3.5,
            radiusPerSqrtSecond: 0.07,
            maximumRadiusPixels: 12,
            classBreaks: STOP_BREAKS_SECONDS,
            classColors: getWaitingClasses(ground, 5),
            ringColor: getStopRing(ground),
            ringWidthPixels: 1.2,
            opacity: 0.95
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = vessels.project(event.coordinate[0], event.coordinate[1]);
      const zone = findZone(zones, x, y);
      return zone < 0 ? null : describeGroup(groups.groupOfZone[zone]);
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const [x, y] = vessels.project(event.coordinate[0], event.coordinate[1]);
      const zone = findZone(zones, x, y);
      const group = zone < 0 ? NO_GROUP : groups.groupOfZone[zone];
      selectGroup(group === selectedGroup ? NO_GROUP : group);
      return group !== NO_GROUP;
    },

    destroy() {
      destroyed = true;
      for (const state of [inside, stopped]) {
        state.tableReader.stop();
        state.listReader.stop();
        state.eventReader?.stop();
      }
      resources.destroy();
    }
  };
}
