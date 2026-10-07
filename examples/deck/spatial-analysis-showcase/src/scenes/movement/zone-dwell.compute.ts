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
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {createPlaybackClock} from '../../engine/playback';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {StopMarkerLayer, ZoneEventMarkerLayer} from './b12-layers';
import {
  formatDuration,
  formatEasternClock,
  formatUtcClock,
  loadVesselTracks,
  METERS_PER_KNOT_SECOND,
  VESSEL_CATEGORIES,
  VESSEL_CATEGORY_LABELS
} from './b12-tracks';
import {
  buildZoneSet,
  findZone,
  rasterizeZones,
  ZONE_KIND_COLORS,
  ZONE_KIND_LABELS,
  ZONE_KINDS,
  type ZoneKind
} from './b12-zones';

/** Option state of the zone dwell scene. */
export type ZoneDwellOptions = {
  variant: 'inside' | 'stopped';
  metric: 'total' | 'mean' | 'longest' | 'visits';
  zoneKind: 'all' | ZoneKind;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  fillOpacity: number;
  showOutlines: boolean;
  showTracks: boolean;
  stopSpeedKnots: number;
  stopMinutes: number;
  eventsPerVessel: '8' | '16' | '32' | '64';
  showEvents: boolean;
  showRestingEvents: boolean;
  playing: boolean;
  time: number;
  playbackSpeed: number;
  loop: boolean;
  pulseMinutes: number;
};

const STOP_CAPACITY = 4096;
const VISIT_CAPACITY = 8192;
const MAX_EVENTS_PER_TRACK = 64;
const EVENT_CANDIDATE_CAPACITY = 1 << 20;
const STOP_CANDIDATE_CAPACITY = 1 << 18;
const RASTER_CELL_METERS = 45;
const SECONDS_PER_DAY = 86400;
const SETTLE_MILLISECONDS = 200;
const NO_ZONE = -1;
const METRIC_CODES = {total: 0, mean: 1, longest: 2, visits: 3} as const;

type ZoneTable = {
  counts: Buffer;
  sums: Buffer;
  means: Buffer;
  maximums: Buffer;
};

type ZoneStats = {
  counts: Uint32Array;
  sums: Float32Array;
  means: Float32Array;
  maximums: Float32Array;
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

type VariantGraph = {
  variant: 'inside' | 'stopped';
  compiled: CompiledGPUCommandGraph<void>;
  display: CompiledGPUCommandGraph<void>;
  displayValues: Buffer;
  table: ZoneTable;
  tableReader: SummaryReader;
  listReader: SummaryReader;
  needsEncode: boolean;
  needsDisplay: boolean;
  eventsPerVessel?: number;
};

/**
 * Zone dwell: how long vessels spend in each harbor zone, two ways. The `inside` variant runs
 * `addFleetDwellZoneEventsRecipe` (`GPUZoneEvents` enter and exit events, then a dense per-zone
 * `GPUGroupStatistics`): any time inside a zone counts. The `stopped` variant runs
 * `addFleetDwellRecipe` (`GPUTrajectoryMetrics` stops, `GPUPointInPolygonJoin` to the zones, the
 * same statistics): only time spent stopped counts. Both write one dense row per zone, which a
 * small kernel turns into the fill value of a rasterized zone map.
 */
export async function createZoneDwell(
  ctx: SceneContext<ZoneDwellOptions>
): Promise<SceneInstance<ZoneDwellOptions>> {
  const vessels = loadVesselTracks(ctx.datasets.get('ais-vessels'));
  const zonesDataset = ctx.datasets.get('ais-zones');
  if (!zonesDataset.geojson) throw new Error('ais-zones has no geometry');
  const zones = buildZoneSet(zonesDataset.geojson, vessels.project);
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = vessels;
  const zoneCount = zones.zoneCount;
  const edgeCount = zones.edgeZones.length;
  const resources = new SpatialAnalysisResources(device, 'zone-dwell');
  const coordinateOrigin: [number, number, number] = [vessels.origin[0], vessels.origin[1], 0];

  // ---- Static inputs --------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', vessels.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', vessels.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', vessels.offsets);
  const segmentsBuffer = resources.createBuffer('segments', vessels.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', vessels.segmentTracks);
  const categoryBuffer = resources.createBuffer('category', vessels.category);
  const trackStartTimes = new Float32Array(trackCount);
  for (let track = 0; track < trackCount; track++) {
    trackStartTimes[track] = vessels.timestamps[vessels.offsets[track]];
  }
  const trackStartBuffer = resources.createBuffer('track-start-times', trackStartTimes);
  const zoneKindsBuffer = resources.createBuffer('zone-kinds', zones.kinds);
  const outlineBuffer = resources.createBuffer('outline', zones.outlineSegments);
  const edgeZonesBuffer = resources.createBuffer('edge-zones', zones.edgeZones);
  let maxZoneEdges = 0;
  const edgesPerZone = new Uint32Array(zoneCount);
  for (const zone of zones.edgeZones) edgesPerZone[zone]++;
  for (const count of edgesPerZone) maxZoneEdges = Math.max(maxZoneEdges, count);
  const selectedOutline = resources.createBuffer('selected-outline', maxZoneEdges * 16);

  // Fill raster: cell -> zone row.
  const pad = RASTER_CELL_METERS * 4;
  const rasterBounds: [number, number, number, number] = [
    zones.bounds[0] - pad,
    zones.bounds[1] - pad,
    zones.bounds[2] + pad,
    zones.bounds[3] + pad
  ];
  const rasterWidth = Math.ceil((rasterBounds[2] - rasterBounds[0]) / RASTER_CELL_METERS);
  const rasterHeight = Math.ceil((rasterBounds[3] - rasterBounds[1]) / RASTER_CELL_METERS);
  const cellRowsBuffer = resources.createBuffer(
    'cell-rows',
    rasterizeZones(zones, rasterBounds, rasterWidth, rasterHeight)
  );

  // ---- Shared draw records and parameters ------------------------------------------------------
  const eventDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'zone-event-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const eventCapacity = vessels.trackCount * MAX_EVENTS_PER_TRACK;
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

  const stopDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'zone-stop-draw',
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
  const displayParameters = resources.createParameterBuffer('display', 'uint32', 4);

  // ---- State -----------------------------------------------------------------------------------
  let destroyed = false;
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'playing', speed: 'playbackSpeed', loop: 'loop'},
    {range: [0, SECONDS_PER_DAY], rate: 1, step: 60}
  );
  let playhead = ctx.options.time;
  let selectedZone = NO_ZONE;
  let settleStale = true;
  let lastChange = performance.now();
  let displayMaximum = 1;
  let stats: ZoneStats | null = null;
  let longestStays: {track: number; zone: number; seconds: number}[] = [];
  let lastFlags: SummaryFlags = {
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
  let active: VariantGraph | null = null;
  const variants = new Map<string, VariantGraph>();

  ctx.setReadout(
    'tracks',
    `${formatCount(trackCount)} tracks / ${formatCount(vessels.vesselCount)} vessels`
  );
  ctx.setReadout('zones', `${zoneCount} zones, ${formatCount(edgeCount)} boundary edges`);

  const metricCode = () => METRIC_CODES[ctx.options.metric];
  const kindCode = () =>
    ctx.options.zoneKind === 'all'
      ? 0xffffffff
      : ZONE_KINDS.indexOf(ctx.options.zoneKind as ZoneKind);
  const getValueScale = () => (ctx.options.metric === 'visits' ? 1 : 1 / 3600);

  function writeDisplayParameters(): void {
    displayParameters.write(Uint32Array.of(metricCode(), kindCode(), 0, 0));
  }

  function writeStopParameters(): void {
    stopParameters.write(
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: ctx.options.stopSpeedKnots * METERS_PER_KNOT_SECOND,
        stopMinimumDuration: ctx.options.stopMinutes * 60
      })
    );
  }

  /** Builds the display kernel graph for a variant's dense zone table. */
  function buildDisplay(
    id: string,
    table: ZoneTable,
    displayValues: Buffer
  ): CompiledGPUCommandGraph<void> {
    const graph = new GPUCommandGraph<void>(device, {id: `zone-display-${id}`});
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    addKernelPass(graph, {
      id: `zone-display-${id}`,
      invocationCount: zoneCount + 1,
      declarations: `const ZONES: u32 = ${zoneCount}u;
fn isNonFinite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u; }`,
      bindings: [
        {
          name: 'counts',
          view: view('counts', table.counts, 'uint32', zoneCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'sums',
          view: view('sums', table.sums, 'float32', zoneCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'means',
          view: view('means', table.means, 'float32', zoneCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'maxima',
          view: view('maxima', table.maximums, 'float32', zoneCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'kinds',
          view: view('kinds', zoneKindsBuffer, 'uint32', zoneCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'parameters',
          view: displayParameters.importToGraph(graph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'display',
          view: view('display', displayValues, 'float32', zoneCount + 1),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var value = nan;
  if (index < ZONES) {
    let metric = u32(bitcast<f32>(parameters[parametersOffset]));
    let kind = parameters[parametersOffset + 1u];
    if (kind == 0xffffffffu || kinds[kindsOffset + index] == kind) {
      if (metric == 0u) { value = sums[sumsOffset + index]; }
      else if (metric == 1u) { value = means[meansOffset + index]; }
      else if (metric == 2u) { value = maxima[maximaOffset + index]; }
      else { value = f32(counts[countsOffset + index]); }
    }
    if (isNonFinite(value) || value <= 0.0) { value = nan; }
  }
  display[displayOffset + index] = value;`
    });
    return resources.track(graph.compile());
  }

  function createTable(id: string): ZoneTable {
    return {
      counts: resources.createBuffer(`${id}-zone-counts`, zoneCount * 4),
      sums: resources.createBuffer(`${id}-zone-sums`, zoneCount * 4),
      means: resources.createBuffer(`${id}-zone-means`, zoneCount * 4),
      maximums: resources.createBuffer(`${id}-zone-maximums`, zoneCount * 4)
    };
  }

  function createTableReader(id: string, table: ZoneTable, variant: VariantGraph['variant']) {
    const bytes = zoneCount * 4;
    return new SummaryReader(
      resources,
      `${id}-table`,
      [
        {buffer: table.counts, size: bytes},
        {buffer: table.sums, size: bytes},
        {buffer: table.means, size: bytes},
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
        if (destroyed || active?.variant !== variant) return;
        const words = new Uint32Array(raw);
        const floats = new Float32Array(raw);
        stats = {
          counts: words.slice(0, zoneCount),
          sums: floats.slice(zoneCount, zoneCount * 2),
          means: floats.slice(zoneCount * 2, zoneCount * 3),
          maximums: floats.slice(zoneCount * 3, zoneCount * 4)
        };
        const flags = zoneCount * 4;
        summarize(
          {
            events: words[flags],
            eventOverflow: words[flags + 1],
            candidates: words[flags + 2],
            candidateOverflow: words[flags + 3],
            trackOverflow: words[flags + 4],
            listOverflow: words[flags + 5],
            stops: words[flags + 6],
            stopOverflow: words[flags + 7],
            joinOverflow: words[flags + 8]
          },
          variant
        );
      }
    );
  }

  function createListReader(id: string, variant: VariantGraph['variant']) {
    if (variant === 'inside') {
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
          if (destroyed || active?.variant !== variant) return;
          const words = new Uint32Array(raw);
          const floats = new Float32Array(raw);
          const count = Math.min(words[0], VISIT_CAPACITY);
          const rows: typeof longestStays = [];
          for (let row = 0; row < count; row++) {
            rows.push({
              track: words[1 + row],
              zone: words[1 + VISIT_CAPACITY + row],
              seconds: floats[1 + VISIT_CAPACITY * 2 + row]
            });
          }
          setLongestStays(rows);
        }
      );
    }
    return new SummaryReader(
      resources,
      `${id}-stops`,
      [
        {buffer: stopCount, size: 4},
        {buffer: stopIds, size: STOP_CAPACITY * 4},
        {buffer: stopZones, size: STOP_CAPACITY * 4},
        {buffer: stopDurations, size: STOP_CAPACITY * 4}
      ],
      raw => {
        if (destroyed || active?.variant !== variant) return;
        const words = new Uint32Array(raw);
        const floats = new Float32Array(raw);
        const count = Math.min(words[0], STOP_CAPACITY);
        const rows: typeof longestStays = [];
        for (let row = 0; row < count; row++) {
          const zone = words[1 + STOP_CAPACITY + row];
          if (zone >= zoneCount) continue;
          rows.push({
            track: words[1 + row],
            zone,
            seconds: floats[1 + STOP_CAPACITY * 2 + row]
          });
        }
        setLongestStays(rows);
      }
    );
  }

  function setLongestStays(rows: typeof longestStays): void {
    longestStays = rows.sort((a, b) => b.seconds - a.seconds).slice(0, 3);
    for (let rank = 0; rank < 3; rank++) {
      const stay = longestStays[rank];
      ctx.setReadout(
        `stay${rank + 1}`,
        stay
          ? `${formatDuration(stay.seconds)}: ${describeVessel(stay.track)} in ${zones.names[stay.zone]}`
          : '-'
      );
    }
  }

  function describeVessel(track: number): string {
    const category = VESSEL_CATEGORIES[vessels.category[track]];
    return `${VESSEL_CATEGORY_LABELS[category].split(' (')[0]} ${vessels.mmsi[track]}`;
  }

  function summarize(flags: SummaryFlags, variant: VariantGraph['variant']): void {
    if (!stats) return;
    lastFlags = flags;
    let visited = 0;
    let totalSeconds = 0;
    let topZone = -1;
    let topValue = 0;
    let maximum = 0;
    const metric = ctx.options.metric;
    const kind = ctx.options.zoneKind;
    for (let zone = 0; zone < zoneCount; zone++) {
      if (stats.counts[zone] > 0) {
        visited++;
        totalSeconds += stats.sums[zone];
      }
      if (kind !== 'all' && ZONE_KINDS[zones.kinds[zone]] !== kind) continue;
      const value = getMetricValue(zone, metric);
      if (Number.isFinite(value) && value > maximum) maximum = value;
      if (Number.isFinite(value) && value > topValue) {
        topValue = value;
        topZone = zone;
      }
    }
    ctx.setReadout('zonesUsed', `${visited} of ${zoneCount}`);
    ctx.setReadout('totalDwell', `${formatCount(totalSeconds / 3600)} vessel-hours`);
    ctx.setReadout(
      'topZone',
      topZone >= 0 ? `${zones.names[topZone]}: ${formatMetric(topValue, metric)}` : 'none'
    );
    if (variant === 'inside') {
      ctx.setReadout(
        'events',
        `${formatCount(flags.events)} kept${flags.eventOverflow ? ' (OVERFLOW)' : ''}`
      );
      ctx.setReadout(
        'candidates',
        `${formatCount(flags.candidates)} of ${formatCount(EVENT_CANDIDATE_CAPACITY)}${flags.candidateOverflow ? ' (OVERFLOW)' : ''}`
      );
      ctx.setReadout(
        'overflow',
        flags.trackOverflow || flags.listOverflow || flags.candidateOverflow
          ? [
              flags.candidateOverflow ? 'candidate scratch' : '',
              flags.trackOverflow
                ? 'a vessel exceeded the per-vessel event cap (statistics are still exact)'
                : '',
              flags.listOverflow ? 'event list' : ''
            ]
              .filter(Boolean)
              .join(', ')
          : 'none'
      );
    } else {
      ctx.setReadout(
        'events',
        `${formatCount(flags.stops)} stops${flags.stopOverflow ? ' (list truncated)' : ''}`
      );
      ctx.setReadout('candidates', 'n/a (stops variant)');
      ctx.setReadout(
        'overflow',
        flags.joinOverflow || flags.stopOverflow ? 'point-in-polygon or stop list' : 'none'
      );
    }
    const scaled = maximum * getValueScale();
    if (maximum > 0 && Math.abs(scaled - displayMaximum) > displayMaximum * 0.03) {
      displayMaximum = scaled;
      ctx.requestLayers();
    } else if (maximum > 0) {
      displayMaximum = scaled;
    }
    ctx.setLegendExtent('zones', [0, displayMaximum]);
    updateCharts(metric, kind);
    describeSelected();
  }

  /** Ranked zones and total dwell by zone kind. */
  function updateCharts(metric: ZoneDwellOptions['metric'], kind: ZoneDwellOptions['zoneKind']) {
    if (!stats) return;
    const scale = getValueScale();
    const ranked: {zone: number; value: number}[] = [];
    const kindTotals = new Float64Array(ZONE_KINDS.length);
    for (let zone = 0; zone < zoneCount; zone++) {
      if (stats.counts[zone] > 0) kindTotals[zones.kinds[zone]] += stats.sums[zone] / 3600;
      if (kind !== 'all' && ZONE_KINDS[zones.kinds[zone]] !== kind) continue;
      const value = getMetricValue(zone, metric);
      if (Number.isFinite(value) && value > 0) ranked.push({zone, value});
    }
    ranked.sort((a, b) => b.value - a.value);
    const top = ranked.slice(0, 8);
    const unit =
      metric === 'visits' ? (ctx.options.variant === 'inside' ? 'visits' : 'stops') : 'hours';
    ctx.setChart(
      'rankChart',
      top.length
        ? {
            kind: 'bars',
            values: top.map(entry => entry.value * scale),
            labels: top.map(entry =>
              zones.names[entry.zone].replace(/ approx\.$/, '').slice(0, 14)
            ),
            highlight: [0],
            height: 150,
            yLabel: unit,
            formatY: value => (value >= 10 ? value.toFixed(0) : value.toFixed(1)),
            description: 'The eight zones with the highest value of the selected statistic.'
          }
        : null
    );
    ctx.setChart('kindChart', {
      kind: 'bars',
      values: kindTotals,
      labels: ZONE_KINDS.map(zoneKind => ZONE_KIND_LABELS[zoneKind].split(' (')[0].slice(0, 11)),
      color: 2,
      height: 120,
      yLabel: 'vessel-hours',
      formatY: value => value.toFixed(0),
      description: 'Total vessel-hours inside each kind of zone, whatever statistic the map shows.'
    });
  }

  function getMetricValue(zone: number, metric: ZoneDwellOptions['metric']): number {
    if (!stats) return NaN;
    if (metric === 'visits') return stats.counts[zone];
    if (metric === 'total') return stats.sums[zone];
    if (metric === 'mean') return stats.means[zone];
    return stats.maximums[zone];
  }

  function formatMetric(value: number, metric: ZoneDwellOptions['metric']): string {
    if (metric === 'visits')
      return `${formatCount(value)} ${ctx.options.variant === 'inside' ? 'visits' : 'stops'}`;
    if (metric === 'total') return `${formatCount(value / 3600)} vessel-hours`;
    return formatDuration(value);
  }

  function describeSelected(): void {
    if (selectedZone === NO_ZONE) {
      ctx.setReadout('selectedZone', 'click a zone');
      return;
    }
    ctx.setReadout('selectedZone', describeZone(selectedZone));
  }

  function describeZone(zone: number): string {
    const kind = ZONE_KINDS[zones.kinds[zone]];
    const header = `${zones.names[zone]} [${kind}]`;
    if (!stats || stats.counts[zone] === 0)
      return `${header}: no ${ctx.options.variant === 'inside' ? 'visits' : 'stops'}`;
    return `${header}: ${formatCount(stats.counts[zone])} ${ctx.options.variant === 'inside' ? 'visits' : 'stops'}, ${formatCount(stats.sums[zone] / 3600)} h total, ${formatDuration(stats.means[zone])} mean, ${formatDuration(stats.maximums[zone])} longest`;
  }

  function writeSelectedOutline(): void {
    const segments = new Float32Array(maxZoneEdges * 4).fill(Number.NaN);
    if (selectedZone !== NO_ZONE) {
      let row = 0;
      for (let edge = 0; edge < edgeCount; edge++) {
        if (zones.edgeZones[edge] !== selectedZone) continue;
        segments.set(zones.outlineSegments.subarray(edge * 4, edge * 4 + 4), row * 4);
        row++;
      }
    }
    selectedOutline.write(segments);
  }

  // ---- Variant graphs ------------------------------------------------------------------------
  function buildInside(eventsPerVessel: number): VariantGraph {
    const table = createTable('inside');
    const displayValues = resources.createBuffer('inside-display', (zoneCount + 1) * 4);
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
      edgeStarts: importGraphBuffer(
        graph,
        'edge-starts',
        resources.createBuffer('edge-starts', zones.edgeStarts),
        'float32x2',
        edgeCount
      ),
      edgeEnds: importGraphBuffer(
        graph,
        'edge-ends',
        resources.createBuffer('edge-ends', zones.edgeEnds),
        'float32x2',
        edgeCount
      ),
      edgeZones: view('edge-zones', edgeZonesBuffer, 'uint32', edgeCount),
      zoneCount,
      candidateCapacity: EVENT_CANDIDATE_CAPACITY,
      maxEventsPerTrack: eventsPerVessel,
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
      visitTable: {
        output: {
          ids: view('visit-tracks', visitTracks, 'uint32', VISIT_CAPACITY),
          count: view('visit-count', visitCount, 'uint32', 1),
          overflow: view('visit-overflow', visitOverflow, 'uint32', 1)
        },
        zones: view('visit-zones', visitZones, 'uint32', VISIT_CAPACITY),
        dwellTimes: view('visit-dwell', visitDwell, 'float32', VISIT_CAPACITY)
      },
      table: {
        counts: view('zone-counts', table.counts, 'uint32', zoneCount),
        sumValues: view('zone-sums', table.sums, 'float32', zoneCount),
        means: view('zone-means', table.means, 'float32', zoneCount),
        maximums: view('zone-maximums', table.maximums, 'float32', zoneCount)
      }
    });
    const compiled = resources.track(graph.compile());
    return {
      variant: 'inside',
      compiled,
      display: buildDisplay('inside', table, displayValues),
      displayValues,
      table,
      tableReader: createTableReader('inside', table, 'inside'),
      listReader: createListReader('inside', 'inside'),
      needsEncode: true,
      needsDisplay: true,
      eventsPerVessel
    };
  }

  function buildStopped(): VariantGraph {
    const table = createTable('stopped');
    const displayValues = resources.createBuffer('stopped-display', (zoneCount + 1) * 4);
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
          resources.createBuffer('polygon-positions', zones.polygonPositions),
          'float32x2',
          zones.polygonPositions.length / 2
        ),
        featureOffsets: importGraphBuffer(
          graph,
          'feature-offsets',
          resources.createBuffer('feature-offsets', zones.featureOffsets),
          'uint32',
          zones.featureOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          graph,
          'polygon-offsets',
          resources.createBuffer('polygon-offsets', zones.polygonOffsets),
          'uint32',
          zones.polygonOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          graph,
          'ring-offsets',
          resources.createBuffer('ring-offsets', zones.ringOffsets),
          'uint32',
          zones.ringOffsets.length
        ),
        candidateCapacity: STOP_CANDIDATE_CAPACITY
      },
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
      stopZones: view('stop-zones', stopZones, 'uint32', STOP_CAPACITY),
      joinOverflow: view('join-overflow', joinOverflow, 'uint32', 1),
      table: {
        counts: view('zone-counts', table.counts, 'uint32', zoneCount),
        sumValues: view('zone-sums', table.sums, 'float32', zoneCount),
        means: view('zone-means', table.means, 'float32', zoneCount),
        maximums: view('zone-maximums', table.maximums, 'float32', zoneCount)
      }
    });
    const compiled = resources.track(graph.compile());
    return {
      variant: 'stopped',
      compiled,
      display: buildDisplay('stopped', table, displayValues),
      displayValues,
      table,
      tableReader: createTableReader('stopped', table, 'stopped'),
      listReader: createListReader('stopped', 'stopped'),
      needsEncode: true,
      needsDisplay: true
    };
  }

  function activate(variant: ZoneDwellOptions['variant']): void {
    const key = variant === 'inside' ? `inside-${ctx.options.eventsPerVessel}` : 'stopped';
    let next = variants.get(key);
    if (!next) {
      next =
        variant === 'inside' ? buildInside(Number(ctx.options.eventsPerVessel)) : buildStopped();
      variants.set(key, next);
    }
    active = next;
    next.needsEncode = true;
    next.needsDisplay = true;
    stats = null;
    longestStays = [];
    markChanged();
  }

  const markChanged = () => {
    lastChange = performance.now();
    settleStale = true;
  };

  writeDisplayParameters();
  writeStopParameters();
  writeSelectedOutline();
  activate(ctx.options.variant);
  describeSelected();

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => (active ? [active.compiled, active.display] : []),

    setOption(id, _value, state) {
      switch (id) {
        case 'variant':
        case 'eventsPerVessel':
          activate(state.variant);
          ctx.requestLayers();
          break;
        case 'metric':
        case 'zoneKind':
          writeDisplayParameters();
          for (const variant of variants.values()) variant.needsDisplay = true;
          if (stats) summarize(lastFlags, state.variant);
          markChanged();
          ctx.requestLayers();
          break;
        case 'stopSpeedKnots':
        case 'stopMinutes':
          writeStopParameters();
          for (const [key, variant] of variants) if (key === 'stopped') variant.needsEncode = true;
          markChanged();
          break;
        case 'time':
        case 'loop':
        case 'playing':
        case 'playbackSpeed':
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
      // Paused, the clock sits on the slider, so every story step and deep link is deterministic.
      playhead = clock.advance(frame);
      ctx.setReadout('clock', `${formatUtcClock(playhead)} UTC (${formatEasternClock(playhead)})`);
      eventClock.write(
        Float32Array.of(playhead, options.pulseMinutes * 60, options.showRestingEvents ? 1 : 0, 0)
      );
      const current = active;
      if (!current) return;
      if (current.needsEncode) {
        current.compiled.encode(commandEncoder, {parameters: undefined});
        const countSource = current.variant === 'inside' ? eventCount : stopCount;
        const draw = current.variant === 'inside' ? eventDraw : stopDraw;
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: countSource,
          destinationBuffer: draw.buffer,
          destinationOffset: 4,
          size: 4
        });
        current.needsEncode = false;
        current.needsDisplay = true;
        settleStale = true;
        lastChange = performance.now();
      }
      if (current.needsDisplay) {
        current.display.encode(commandEncoder, {parameters: undefined});
        current.needsDisplay = false;
      }
      if (settleStale && performance.now() - lastChange > SETTLE_MILLISECONDS) {
        current.tableReader.markStale();
        current.listReader.markStale();
        settleStale = false;
      }
      current.tableReader.flush(commandEncoder);
      current.listReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const current = active;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (!current) return layers;
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: `zone-fill-${current.variant}-${options.metric}`,
          coordinateOrigin,
          gridSize: [rasterWidth, rasterHeight],
          bounds: rasterBounds,
          rowOrigin: 'south',
          values: current.displayValues,
          valueFormat: 'float32',
          valueIndices: cellRowsBuffer,
          valueScale: getValueScale(),
          valueRange: [0, Math.max(displayMaximum, 1e-6)],
          colormap: options.ramp,
          sqrtScale: true,
          noDataColor: [0, 0, 0, 0],
          opacity: options.fillOpacity
        })
      );
      if (options.showTracks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'zone-tracks',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            color: dark ? [190, 200, 220, 30] : [60, 70, 90, 34],
            values: categoryBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer
          })
        );
      }
      if (options.showOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'zone-outlines',
            coordinateOrigin,
            segments: outlineBuffer,
            instanceCount: edgeCount,
            widthPixels: 1.5,
            values: zoneKindsBuffer,
            valueFormat: 'uint32',
            valueIndices: edgeZonesBuffer,
            colormap: 'category',
            palette: ZONE_KIND_COLORS,
            opacity: 0.9
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'zone-selected',
          coordinateOrigin,
          segments: selectedOutline,
          instanceCount: maxZoneEdges,
          widthPixels: 4,
          color: dark ? [255, 255, 255, 255] : [20, 24, 32, 255]
        })
      );
      if (options.showEvents && current.variant === 'inside') {
        layers.push(
          new ZoneEventMarkerLayer({
            id: 'zone-events',
            coordinateOrigin,
            positions: eventPositions,
            eventTracks,
            eventTimes,
            eventTypes,
            trackStartTimes: trackStartBuffer,
            clock: eventClock.buffer,
            drawCommands: eventDraw,
            sizePixels: 5
          })
        );
      }
      if (options.showEvents && current.variant === 'stopped') {
        layers.push(
          new StopMarkerLayer({
            id: 'zone-stops',
            coordinateOrigin,
            centroids: stopCentroids,
            durations: stopDurations,
            drawCommands: stopDraw,
            baseRadiusPixels: 3,
            radiusPerSqrtSecond: 0.12,
            maximumRadiusPixels: 16,
            durationForFullColor: 6 * 3600,
            opacity: 0.9
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = vessels.project(event.coordinate[0], event.coordinate[1]);
      const zone = findZone(zones, x, y);
      return zone < 0 ? null : describeZone(zone);
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const [x, y] = vessels.project(event.coordinate[0], event.coordinate[1]);
      const zone = findZone(zones, x, y);
      selectedZone = zone === selectedZone ? NO_ZONE : zone;
      writeSelectedOutline();
      describeSelected();
      ctx.requestLayers();
      return zone >= 0;
    },

    destroy() {
      destroyed = true;
      for (const variant of variants.values()) {
        variant.tableReader.stop();
        variant.listReader.stop();
      }
      resources.destroy();
    }
  };
}
