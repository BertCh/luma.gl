// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {CHICAGO, nearestPlaceLabel} from '../../cartography/gazetteer';
import {
  getClassIndexOf,
  getClassTableLayerProps,
  makeClassTable
} from '../../cartography/class-table';
import {formatCount, formatOrdinal, formatPercent, formatSigned} from '../../cartography/live-text';
import {getGeometryPolygons} from '../../cartography/picking';
import type {ClassTable, MapAnnotation, MapHighlight} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {SpatialAnalysisFlowLayer} from '../../engine/flow-layer';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {MatrixChartData} from '../chart-types';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {
  CHICAGO_ORIGIN,
  readCommuteSource,
  readTaxiSource,
  type FlowSource
} from './b11-flow-sources';
import {getZoneAt} from './b11-zone-raster';
import {
  CONTEXT_INK,
  FLOW_HALO,
  FLOW_INK,
  FLOW_INK_HOT,
  buildFlowArrows,
  inkFor,
  withInkAlpha,
  type FlowArrows
} from './flows-style';
import {
  INTERIOR_CIRCLE_COUNT,
  MAXIMUM_FLOW_WIDTH,
  MAXIMUM_INTERIOR_RADIUS,
  TOP_FLOW_COUNT,
  formatTaxiDayHour
} from './taxi-flows-constants';
import {
  type Bounds,
  type Lattice,
  type LatticeKind,
  MINIMUM_ZONE_SIZE,
  formatDensity,
  getAreaDisplayName,
  getCenterBounds,
  getFlowScale,
  getLattice,
  getLatticeBounds,
  getLatticeCellAreaKm2,
  getLatticeGridSize,
  getLatticeZone,
  getLatticeZoneCenter,
  getRateDivisor,
  getZoneDensityBreaks,
  type FlowScale
} from './taxi-flows-data';

/** Option state of the taxi-flows scene. */
export type TaxiFlowOptions = {
  source: 'taxi' | 'commute';
  zones: 'native' | 'hexagon' | 'grid';
  zoneSize: number;
  excludeSelf: boolean;
  sumOrder: 'sorted' | 'atomic';
  taxiWeight: 'trips' | 'fare' | 'duration';
  commuteWeight: 'all' | 'low' | 'mid' | 'high';
  /** Playhead of the taxi day: the first hour of the window, counted from 04:00. */
  hourStart: number;
  /** The window `[from, to)` in hours of the taxi day (0 is 04:00, 24 is 04:00 next day). */
  hours: readonly [number, number];
  dayType: 'all' | 'weekday' | 'weekend';
  play: boolean;
  playSpeed: number;
  loop: boolean;
  totals: 'departures' | 'arrivals';
  flowStyle: 'raw' | 'designed';
  arcs: number;
  annotate: 'none' | 'top-flows' | 'interior' | 'top-flow';
  showZones: boolean;
  showOutlines: boolean;
};

/** Flows drawn in the hot ink (the heaviest, on the night ground). */
const HOT_FLOW_COUNT = 3;
/** Curvature of designed flows (fraction of the flow length), as the layer's default. */
const FLOW_CURVATURE = 0.15;
const HEADER_WORDS = 4;
/** Frames a replaced graph's buffers stay alive so in-flight frames never touch destroyed buffers. */
const RETIRE_FRAMES = 4;
const TAXI_PAIR_CAPACITY = 16384;
const COMMUTE_PAIR_CAPACITY = 131072;

/** Per-source buffers, created the first time a source is shown. */
type PreparedSource = {
  source: FlowSource;
  originIds: Buffer;
  destinationIds: Buffer;
  hours: Buffer | null;
  mask: Buffer;
  weights: Buffer;
  zoneIdRaster: Buffer;
  outline: Buffer;
  /** Lattice inputs, built the first time a lattice is requested. */
  originPositions?: Buffer;
  destinationPositions?: Buffer;
  centerBounds: Bounds;
};

type FlowGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  prepared: PreparedSource;
  kind: TaxiFlowOptions['zones'];
  sumOrder: TaxiFlowOptions['sumOrder'];
  excludeSelf: boolean;
  /** Number of zone rows of every zone output (native count or lattice capacity). */
  zoneCount: number;
  capacityGrid: [number, number];
  pairCapacity: number;
  count: Buffer;
  originZones: Buffer;
  destinationZones: Buffer;
  flowWeights: Buffer;
  zoneOutWeights: Buffer;
  zoneInWeights: Buffer;
  zoneOutCounts: Buffer;
  /** The zone backdrop value per zone plus one NaN sentinel row, written from the readback. */
  density: Buffer;
  reader: SummaryReader;
};

/** Everything a summary readback tells the scene, kept for tooltips, annotations and charts. */
type Snapshot = {
  graph: FlowGraph;
  lattice: Lattice | null;
  out: Float32Array;
  incoming: Float32Array;
  flowRows: number;
  topOrigin: Uint32Array;
  topDestination: Uint32Array;
  topWeight: Float32Array;
  designed: FlowArrows;
  total: number;
  interior: number;
  between: number;
  /** Same-zone rows of the list, heaviest first. */
  interiorRows: {zone: number; weight: number}[];
  windowHours: number;
  dayType: TaxiFlowOptions['dayType'];
};

/**
 * Origin-destination flow aggregation. One compiled `GPUFlowAggregation` turns every source row
 * (a taxi record of origin area, destination area, weekday or weekend, hour of the taxi day,
 * trips; or a home tract, work tract and jobs) into a weight-sorted top-512 flow list and
 * per-zone departure and arrival totals. The few kilobytes of that list are read back and drawn
 * as designed arrows (`SpatialAnalysisFlowLayer`); the zone totals become a classed per-km2
 * backdrop. Zones are the dataset's own areas (`ids`), or a hexagon or square lattice whose size
 * is a per-frame parameter. The time window, day mask, weights and lattice size rewrite buffers;
 * only the source, the zone kind, self-flow exclusion and the sum order rebuild.
 */
export async function createTaxiFlows(
  ctx: SceneContext<TaxiFlowOptions>
): Promise<SceneInstance<TaxiFlowOptions>> {
  const {device} = ctx;
  const projection = ctx.datasets.get('chicago-taxi-od').getProjection(CHICAGO_ORIGIN);
  const resources = new SpatialAnalysisResources(device, 'taxi-flows');
  const areas = ctx.datasets.get('chicago-community-areas');
  const sources = new Map<FlowSource['id'], PreparedSource>();

  const radiusBuffer = resources.createParameterBuffer('radius', 'float32', 1);
  const activeGridBuffer = resources.createParameterBuffer('active-grid', 'uint32', 2);
  const boundsBuffer = resources.createParameterBuffer('bounds', 'float32', 4);
  const windowBuffer = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );

  // Flow buffers, written from every readback: the designed arrows (heaviest last through
  // `ids`), the heaviest few for the hot ink, the straight raw lines and the interior circles.
  const k = TOP_FLOW_COUNT;
  const designedFlows = resources.createBuffer('designed-flows', k * 16);
  const designedValues = resources.createBuffer('designed-values', k * 4);
  const designedOrder = resources.createBuffer('designed-order', k * 4);
  const hotOrder = resources.createBuffer('hot-order', HOT_FLOW_COUNT * 4);
  const rawFlows = resources.createBuffer('raw-flows', k * 16);
  const rawValues = resources.createBuffer('raw-values', k * 4);
  const interiorPositions = resources.createBuffer('interior-positions', INTERIOR_CIRCLE_COUNT * 8);
  const interiorSizes = resources.createBuffer('interior-sizes', INTERIOR_CIRCLE_COUNT * 4);
  let designedCount = 0;
  let hotCount = 0;
  let rawCount = 0;
  let interiorCount = 0;

  let graph: FlowGraph | null = null;
  let snapshot: Snapshot | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let statsStale = true;
  let windowLength = Math.max(1, ctx.options.hours[1] - ctx.options.hours[0]);
  let lastWindow: [number, number] = [ctx.options.hours[0], ctx.options.hours[1]];
  const clock = createPlaybackClock(
    ctx,
    {time: 'hourStart', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, Math.max(0, 24 - windowLength)], rate: 1, step: 1, notify: false}
  );
  let hourly: {
    key: string;
    weekday: Float64Array;
    weekend: Float64Array;
  } | null = null;
  let hourlyWindowKey = '';
  let timelineKey = '';
  let zoneTable: ClassTable | null = null;
  let flowScale: FlowScale = {maxFlow: 1, maxInterior: 1};
  let flowScaleKey = '';
  let selectedPair: {origin: number; destination: number} | null = null;
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];
  let comparison: {alt: FlowGraph; reader: SummaryReader; main: FlowGraph} | null = null;
  let comparing = false;
  const breaksByKey = new Map<string, {breaks: number[]; extent: [number, number]}>();

  const areaNames = areas.manifest as unknown as {names?: string[]};

  function prepare(id: FlowSource['id']): PreparedSource {
    const existing = sources.get(id);
    if (existing) return existing;
    const source =
      id === 'taxi'
        ? readTaxiSource(ctx.datasets.get('chicago-taxi-od'), areas, projection)
        : readCommuteSource(
            ctx.datasets.get('chicago-lodes-od'),
            ctx.datasets.get('chicago-tracts'),
            areaNames.names ?? [],
            projection
          );
    const mask = new Uint32Array(source.rowCount).fill(1);
    const defaultWeight = id === 'taxi' ? 'trips' : 'all';
    const name = `${id}`;
    const prepared: PreparedSource = {
      source,
      originIds: resources.createBuffer(`${name}-origin`, source.origin),
      destinationIds: resources.createBuffer(`${name}-destination`, source.destination),
      hours: source.hour ? resources.createBuffer(`${name}-hour`, source.hour) : null,
      mask: resources.createBuffer(`${name}-mask`, mask),
      weights: resources.createBuffer(`${name}-weights`, source.weights[defaultWeight]),
      zoneIdRaster: resources.createBuffer(`${name}-zone-raster`, source.raster.zoneIds),
      outline: resources.createBuffer(`${name}-outline`, source.raster.outline),
      centerBounds: getCenterBounds(source.centers, source.zoneCount)
    };
    sources.set(id, prepared);
    return prepared;
  }

  function getLatticePositions(prepared: PreparedSource): void {
    if (prepared.originPositions) return;
    const {source} = prepared;
    const origins = new Float32Array(source.rowCount * 2);
    const destinations = new Float32Array(source.rowCount * 2);
    for (let row = 0; row < source.rowCount; row++) {
      origins.set(
        source.centers.subarray(source.origin[row] * 2, source.origin[row] * 2 + 2),
        row * 2
      );
      destinations.set(
        source.centers.subarray(source.destination[row] * 2, source.destination[row] * 2 + 2),
        row * 2
      );
    }
    prepared.originPositions = resources.createBuffer(`${source.id}-origin-xy`, origins);
    prepared.destinationPositions = resources.createBuffer(
      `${source.id}-destination-xy`,
      destinations
    );
  }

  function getActiveLattice(current: FlowGraph): Lattice | null {
    if (current.kind === 'native') return null;
    return getLattice(
      current.prepared.centerBounds,
      current.kind as LatticeKind,
      ctx.options.zoneSize
    );
  }

  function getWeightKey(): string {
    return ctx.options.source === 'taxi' ? ctx.options.taxiWeight : ctx.options.commuteWeight;
  }

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 3);
    statsStale = true;
  }

  function writeWeights(): void {
    const prepared = graph?.prepared ?? prepare(ctx.options.source);
    prepared.weights.write(prepared.source.weights[getWeightKey()]);
    refreshFlowScale();
    refreshZoneTable();
    markChanged();
  }

  function writeMask(): void {
    const prepared = graph?.prepared;
    if (!prepared?.source.dayType) return;
    const {dayType} = ctx.options;
    const rows = prepared.source.rowCount;
    const mask = new Uint32Array(rows);
    for (let row = 0; row < rows; row++) {
      const weekend = prepared.source.dayType[row] === 1;
      mask[row] = dayType === 'all' || (dayType === 'weekend') === weekend ? 1 : 0;
    }
    prepared.mask.write(mask);
    markChanged();
  }

  /** The one width scale of the dataset, computed from every row (see `getFlowScale`). */
  function refreshFlowScale(): void {
    const prepared = graph?.prepared ?? prepare(ctx.options.source);
    const key = `${prepared.source.id}:${getWeightKey()}`;
    if (key === flowScaleKey) return;
    flowScaleKey = key;
    flowScale = getFlowScale(prepared.source, getWeightKey(), prepared.centerBounds);
    ctx.setLegendData('flowScale', flowScale);
    ctx.setReadout('scaleMax', formatWeight(flowScale.maxFlow));
  }

  /** The zone-backdrop class table: fixed breaks of the source, colours of the ground. */
  function refreshZoneTable(): void {
    const prepared = graph?.prepared ?? prepare(ctx.options.source);
    const {source} = prepared;
    const ground = ctx.ground();
    const totals = ctx.options.totals;
    const weightKey = getWeightKey();
    const key = `${source.id}:${weightKey}:${source.id === 'commute' ? 'arrivals' : totals}`;
    let entry = breaksByKey.get(key);
    if (!entry) {
      // Breaks are a property of the dataset, not of the window: computed once, then fixed.
      entry = getZoneDensityBreaks(source, weightKey, totals);
      breaksByKey.set(key, entry);
    }
    const units = source.weightUnits[weightKey];
    zoneTable = makeClassTable({
      breaks: entry.breaks,
      // Cool Blues behind the warm flow figure on night; warm YlOrBr jobs as co-subject on paper.
      scheme: source.id === 'commute' ? 'YlOrBr' : 'Blues',
      ground,
      alpha: ground === 'dark' ? 130 : 225,
      unit: source.id === 'commute' ? 'jobs' : units.unit,
      extent: entry.extent,
      method:
        source.id === 'commute'
          ? 'Quantiles of jobs arriving per km², one set of breaks for every earnings weight'
          : 'Quantiles of an average day in 2023, breaks fixed across windows and day types',
      noData: {label: source.id === 'commute' ? 'No jobs arriving' : 'No trips in the window'},
      format: formatDensity
    });
    ctx.setLegendData('zones', {
      table: zoneTable,
      source: source.id,
      ground,
      excludeSelf: ctx.options.excludeSelf
    });
  }

  function getWindow(): [number, number] {
    return lastWindow;
  }

  function getWindowHours(): number {
    const [start, end] = getWindow();
    return graph?.prepared.hours ? end - start : 1;
  }

  /** Share of each day type's weight in every pickup hour of the taxi day, for the current weight. */
  function updateHourlyChart(start: number, end: number): void {
    const source = graph?.prepared.source;
    if (!source?.hour || !source.dayType) {
      if (hourlyWindowKey !== 'none') {
        hourlyWindowKey = 'none';
        ctx.setChart('hourlyChart', null);
      }
      return;
    }
    const key = `${source.id}:${getWeightKey()}`;
    if (!hourly || hourly.key !== key) {
      const weights = source.weights[getWeightKey()];
      const weekday = new Float64Array(24);
      const weekend = new Float64Array(24);
      for (let row = 0; row < source.rowCount; row++) {
        const bucket = Math.min(23, Math.max(0, Math.floor(source.hour[row])));
        (source.dayType[row] === 1 ? weekend : weekday)[bucket] += weights[row];
      }
      hourly = {key, weekday, weekend};
      hourlyWindowKey = '';
    }
    const windowKey = `${key}|${start}|${end}`;
    if (windowKey === hourlyWindowKey) return;
    hourlyWindowKey = windowKey;
    const share = (values: Float64Array) => {
      const total = values.reduce((sum, value) => sum + value, 0) || 1;
      return values.map(value => (100 * value) / total);
    };
    const hours = Array.from({length: 24}, (_, hour) => hour + 0.5);
    const full = end - start >= 24;
    ctx.setChart('hourlyChart', {
      kind: 'line',
      series: [
        {label: 'weekdays', x: hours, y: share(hourly.weekday), color: 0},
        {label: 'weekends', x: hours, y: share(hourly.weekend), color: 3}
      ],
      xDomain: [0, 24],
      bands: full ? [] : [{from: start, to: end, tone: 'signal'}],
      xLabel: 'pickup hour, the taxi day starts at 04:00',
      yLabel: '% of the day',
      height: 130,
      formatX: value => String((Math.round(value) + 4) % 24).padStart(2, '0'),
      formatY: value => `${value.toFixed(0)}%`,
      description:
        'Share of each day type’s taxi trips in every pickup hour, weekdays against weekends, starting at 04:00. The shaded band is the time window.'
    });
    const {dayType} = ctx.options;
    if (timelineKey !== `${key}|${dayType}`) {
      timelineKey = `${key}|${dayType}`;
      const all = new Float64Array(24);
      for (let hour = 0; hour < 24; hour++) {
        all[hour] =
          (dayType === 'weekend' ? 0 : hourly.weekday[hour]) +
          (dayType === 'weekday' ? 0 : hourly.weekend[hour]);
      }
      ctx.setTimelineData({domain: [0, 24], histogram: Array.from(all)});
    }
  }

  /** Builds and compiles one flow graph; the caller owns the returned resources. */
  function buildGraph(
    prepared: PreparedSource,
    kind: TaxiFlowOptions['zones'],
    sumOrder: TaxiFlowOptions['sumOrder'],
    excludeSelf: boolean
  ): FlowGraph {
    const {source} = prepared;
    const id = ++serial;
    const graphResources = new SpatialAnalysisResources(device, `taxi-flows-${id}`);
    let capacityGrid: [number, number] = [source.zoneCount, 1];
    if (kind !== 'native') {
      getLatticePositions(prepared);
      const capacityBounds = getLatticeBounds(
        prepared.centerBounds,
        kind as LatticeKind,
        MINIMUM_ZONE_SIZE
      );
      capacityGrid = getLatticeGridSize(capacityBounds, kind as LatticeKind, MINIMUM_ZONE_SIZE);
    }
    const zoneCount = kind === 'native' ? source.zoneCount : capacityGrid[0] * capacityGrid[1];
    const sentinel = (rows: number) => {
      const values = new Float32Array(rows + 1);
      values[rows] = Number.NaN;
      return values;
    };
    const ids = graphResources.createBuffer('ids', k * 4);
    const count = graphResources.createBuffer('count', 4);
    const requiredCount = graphResources.createBuffer('total-count', 4);
    const overflow = graphResources.createBuffer('overflow', 4);
    const pairOverflow = graphResources.createBuffer('pair-overflow', 4);
    const originZones = graphResources.createBuffer('flow-origin', k * 4);
    const destinationZones = graphResources.createBuffer('flow-destination', k * 4);
    const flowWeights = graphResources.createBuffer('flow-weights', k * 4);
    const zoneOutWeights = graphResources.createBuffer('zone-out-weights', sentinel(zoneCount));
    const zoneInWeights = graphResources.createBuffer('zone-in-weights', sentinel(zoneCount));
    const zoneOutCounts = graphResources.createBuffer('zone-out-counts', zoneCount * 4);
    const density = graphResources.createBuffer('density', sentinel(zoneCount));
    const pairCapacity = source.id === 'taxi' ? TAXI_PAIR_CAPACITY : COMMUTE_PAIR_CAPACITY;

    const commandGraph = new GPUCommandGraph<void>(device, {id: `taxi-flows-${id}`});
    const rows = source.rowCount;
    const zones =
      kind === 'native'
        ? ({kind: 'ids', zoneCount} as const)
        : kind === 'hexagon'
          ? ({
              kind: 'hexagon',
              bounds: boundsBuffer.importToGraph(commandGraph),
              gridSize: capacityGrid,
              activeGridSize: activeGridBuffer.importToGraph(commandGraph),
              radius: radiusBuffer.importToGraph(commandGraph)
            } as const)
          : ({
              kind: 'grid',
              bounds: boundsBuffer.importToGraph(commandGraph),
              gridSize: capacityGrid,
              activeGridSize: activeGridBuffer.importToGraph(commandGraph)
            } as const);
    const positions =
      kind === 'native'
        ? {
            originZoneIds: importGraphBuffer(
              commandGraph,
              'origin-ids',
              prepared.originIds,
              'uint32',
              rows
            ),
            destinationZoneIds: importGraphBuffer(
              commandGraph,
              'destination-ids',
              prepared.destinationIds,
              'uint32',
              rows
            )
          }
        : {
            origins: importGraphBuffer(
              commandGraph,
              'origin-xy',
              prepared.originPositions!,
              'float32x2',
              rows
            ),
            destinations: importGraphBuffer(
              commandGraph,
              'destination-xy',
              prepared.destinationPositions!,
              'float32x2',
              rows
            )
          };
    commandGraph.add(
      new GPUFlowAggregation({
        id: 'flows',
        zones,
        ...positions,
        weights: importGraphBuffer(commandGraph, 'weights', prepared.weights, 'float32', rows),
        mask: importGraphBuffer(commandGraph, 'mask', prepared.mask, 'uint32', rows),
        ...(prepared.hours
          ? {
              timeWindow: {
                timestamps: importGraphBuffer(
                  commandGraph,
                  'hours',
                  prepared.hours,
                  'float32',
                  rows
                ),
                window: windowBuffer.importToGraph(commandGraph)
              }
            }
          : {}),
        excludeSelfFlows: excludeSelf,
        sumOrder,
        pairCapacity,
        // rows * maxProbeCount must fit in a uint32; 128 probes is far above the expected chain length.
        maxProbeCount: 128,
        output: {
          ids: importGraphBuffer(commandGraph, 'ids', ids, 'uint32', k),
          count: importGraphBuffer(commandGraph, 'count', count, 'uint32', 1),
          overflow: importGraphBuffer(commandGraph, 'overflow', overflow, 'uint32', 1),
          requiredCount: importGraphBuffer(commandGraph, 'total-count', requiredCount, 'uint32', 1)
        },
        flowOriginZoneIds: importGraphBuffer(commandGraph, 'flow-origin', originZones, 'uint32', k),
        flowDestinationZoneIds: importGraphBuffer(
          commandGraph,
          'flow-destination',
          destinationZones,
          'uint32',
          k
        ),
        flowWeights: importGraphBuffer(commandGraph, 'flow-weights', flowWeights, 'float32', k),
        pairOverflow: importGraphBuffer(commandGraph, 'pair-overflow', pairOverflow, 'uint32', 1),
        zoneOutWeights: importGraphBuffer(
          commandGraph,
          'zone-out-w',
          zoneOutWeights,
          'float32',
          zoneCount
        ),
        zoneInWeights: importGraphBuffer(
          commandGraph,
          'zone-in-w',
          zoneInWeights,
          'float32',
          zoneCount
        ),
        zoneOutCounts: importGraphBuffer(
          commandGraph,
          'zone-out-c',
          zoneOutCounts,
          'uint32',
          zoneCount
        )
      })
    );
    const compiled = graphResources.track(commandGraph.compile());
    const built: FlowGraph = {
      resources: graphResources,
      compiled,
      prepared,
      kind,
      sumOrder,
      excludeSelf,
      zoneCount,
      capacityGrid,
      pairCapacity,
      count,
      originZones,
      destinationZones,
      flowWeights,
      zoneOutWeights,
      zoneInWeights,
      zoneOutCounts,
      density,
      // Header (count, total, overflow, pair overflow), top-K ids and weights, then zone totals.
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      graphResources,
      `taxi-flows-${id}`,
      [
        {buffer: count, size: 4},
        {buffer: requiredCount, size: 4},
        {buffer: overflow, size: 4},
        {buffer: pairOverflow, size: 4},
        {buffer: originZones, size: k * 4},
        {buffer: destinationZones, size: k * 4},
        {buffer: flowWeights, size: k * 4},
        {buffer: zoneOutWeights, size: zoneCount * 4},
        {buffer: zoneInWeights, size: zoneCount * 4},
        {buffer: zoneOutCounts, size: zoneCount * 4}
      ],
      bytes => {
        if (!destroyed && graph === built) processStatistics(built, bytes);
      }
    );
    return built;
  }

  function rebuild(): void {
    const {source, zones, excludeSelf, sumOrder} = ctx.options;
    const prepared = prepare(source);
    if (graph) retired.push({resources: graph.resources, frames: 0});
    graph = buildGraph(prepared, zones, sumOrder, excludeSelf);
    snapshot = null;
    designedCount = 0;
    hotCount = 0;
    rawCount = 0;
    interiorCount = 0;
    selectedPair = null;
    ctx.setHighlight(null);
    // The new graph reads this source's current weights, mask and lattice parameters.
    prepared.weights.write(prepared.source.weights[getWeightKey()]);
    writeMask();
    writeLattice();
    refreshFlowScale();
    refreshZoneTable();
    ctx.setAnnotations('findings', null);
    ctx.setCost({records: prepared.source.rowCount});
    markChanged();
  }

  // ---------------------------------------------------------------------------------------------
  // Zones: names, centres and lattice furniture
  // ---------------------------------------------------------------------------------------------

  function writeLattice(): void {
    if (!graph) return;
    const lattice = getActiveLattice(graph);
    if (!lattice) {
      ctx.setReadout(
        'zones',
        `${graph.prepared.source.zoneCount} ${graph.prepared.source.id === 'taxi' ? 'community areas' : 'census tracts'}`
      );
      updateLatticeFurniture(null);
      return;
    }
    radiusBuffer.write(Float32Array.of(lattice.size));
    activeGridBuffer.write(Uint32Array.of(lattice.columns, lattice.rows));
    boundsBuffer.write(Float32Array.from(lattice.bounds));
    ctx.setReadout(
      'zones',
      `${lattice.columns} × ${lattice.rows} ${lattice.kind === 'hexagon' ? 'hexagons' : 'cells'}, ${formatCount(lattice.size / 1000)} km ${lattice.kind === 'hexagon' ? 'radius' : 'wide'}`
    );
    updateLatticeFurniture(lattice);
  }

  const furniture: {
    title: {sample: string};
    scaleBar: {units: 'metric'; ticks?: number[]};
  } = {title: {sample: ''}, scaleBar: {units: 'metric'}};

  /** The scale-bar tick and the dashed ring at the Loop, both at the lattice size. */
  function updateLatticeFurniture(lattice: Lattice | null): void {
    furniture.scaleBar = {units: 'metric', ticks: lattice ? [lattice.size] : undefined};
    ctx.setFurniture(furniture);
    const loop = CHICAGO.places.loop;
    ctx.setAnnotations(
      'lattice',
      lattice && loop
        ? [
            {
              kind: 'ring',
              coordinate: loop.lngLat,
              radiusMeters: lattice.size,
              dashed: true,
              text: `${lattice.kind === 'hexagon' ? 'hexagon radius ' : 'cell '}${formatCount(lattice.size / 1000)} km`
            }
          ]
        : null
    );
  }

  function getZoneCenter(current: Snapshot, zone: number): [number, number] | null {
    const {source} = current.graph.prepared;
    if (current.lattice) {
      return zone >= 0 && zone < current.graph.zoneCount
        ? getLatticeZoneCenter(current.lattice, zone)
        : null;
    }
    return zone >= 0 && zone < source.zoneCount
      ? [source.centers[zone * 2], source.centers[zone * 2 + 1]]
      : null;
  }

  function getZoneName(current: Snapshot, zone: number): string {
    const {source} = current.graph.prepared;
    if (!current.lattice) return getAreaDisplayName(source.zoneNames[zone] ?? `Zone ${zone}`);
    const center = getLatticeZoneCenter(current.lattice, zone);
    const [longitude, latitude] = projection.unproject(center[0], center[1]);
    const near = nearestPlaceLabel(CHICAGO, [longitude, latitude], {
      kinds: ['district', 'neighborhood']
    });
    const noun = current.lattice.kind === 'hexagon' ? 'Hexagon' : 'Cell';
    return near ? `${noun} ${near}` : noun;
  }

  function getZoneArea(current: Snapshot, zone: number): number {
    return current.lattice
      ? getLatticeCellAreaKm2(current.lattice)
      : (current.graph.prepared.source.zoneAreaKm2[zone] ?? 1);
  }

  function formatWeight(value: number): string {
    const units = (graph?.prepared ?? prepare(ctx.options.source)).source.weightUnits[
      getWeightKey()
    ];
    return `${units.prefix}${formatCount(value)} ${units.unit}`;
  }

  /** Midpoint of a designed flow (the curve bends left of travel by `FLOW_CURVATURE`). */
  function getFlowMidpoint(arrows: FlowArrows, index: number): [number, number] {
    const [x0, y0, x1, y1] = arrows.flows.subarray(index * 4, index * 4 + 4);
    const dx = x1 - x0;
    const dy = y1 - y0;
    const bend = 0.5 * FLOW_CURVATURE;
    return [(x0 + x1) / 2 - dy * bend, (y0 + y1) / 2 + dx * bend];
  }

  function toLngLat(point: readonly [number, number]): [number, number] {
    return projection.unproject(point[0], point[1]);
  }

  // ---------------------------------------------------------------------------------------------
  // The readback: arrows, circles, backdrop, readouts, charts, notes
  // ---------------------------------------------------------------------------------------------

  function processStatistics(current: FlowGraph, bytes: ArrayBuffer): void {
    const zoneCount = current.zoneCount;
    const words = new Uint32Array(bytes);
    const [flowRows, pairCount, , pairOverflow] = words;
    const topOrigin = words.subarray(HEADER_WORDS, HEADER_WORDS + k);
    const topDestination = words.subarray(HEADER_WORDS + k, HEADER_WORDS + 2 * k);
    const floats = new Float32Array(bytes);
    const topWeight = floats.subarray(HEADER_WORDS + 2 * k, HEADER_WORDS + 3 * k);
    const outStart = HEADER_WORDS + 3 * k;
    const out = floats.slice(outStart, outStart + zoneCount);
    const incoming = floats.slice(outStart + zoneCount, outStart + 2 * zoneCount);
    const rowCounts = words.subarray(outStart + 2 * zoneCount, outStart + 3 * zoneCount);
    const options = ctx.options;
    const {source} = current.prepared;
    const lattice = getActiveLattice(current);
    const listed = Math.min(flowRows, k);

    let total = 0;
    let rows = 0;
    let busiestIn = 0;
    let maximumIn = 0;
    let activeZones = 0;
    for (let zone = 0; zone < zoneCount; zone++) {
      total += out[zone];
      rows += rowCounts[zone];
      if (incoming[zone] > maximumIn) {
        maximumIn = incoming[zone];
        busiestIn = zone;
      }
      if (out[zone] > 0 || incoming[zone] > 0) activeZones++;
    }
    // Same-zone pairs of the ranked list: the list holds almost every one of them, since the
    // diagonal is among the heaviest pairs.
    const interiorRows: {zone: number; weight: number}[] = [];
    let interior = 0;
    for (let row = 0; row < listed; row++) {
      if (topOrigin[row] === topDestination[row]) {
        interior += topWeight[row];
        interiorRows.push({zone: topOrigin[row], weight: topWeight[row]});
      }
    }
    const between = total - interior;

    const next: Snapshot = {
      graph: current,
      lattice,
      out,
      incoming,
      flowRows: listed,
      topOrigin: Uint32Array.from(topOrigin.subarray(0, listed)),
      topDestination: Uint32Array.from(topDestination.subarray(0, listed)),
      topWeight: Float32Array.from(topWeight.subarray(0, listed)),
      designed: undefined as unknown as FlowArrows,
      total,
      interior,
      between,
      interiorRows,
      windowHours: getWindowHours(),
      dayType: options.dayType
    };
    snapshot = next;
    const getCenter = (zone: number) => getZoneCenter(next, zone);
    const arrowInput = {
      originZones: next.topOrigin,
      destinationZones: next.topDestination,
      weights: next.topWeight,
      count: listed,
      getZoneCenter: getCenter,
      skipSelf: true
    };
    next.designed = buildFlowArrows({...arrowInput, limit: options.arcs});
    const raw = buildFlowArrows({...arrowInput, limit: k});

    // Designed arrows: heaviest last, and the heaviest few again in the hot ink.
    designedCount = next.designed.count;
    hotCount = Math.min(HOT_FLOW_COUNT, designedCount);
    if (designedCount > 0) {
      designedFlows.write(next.designed.flows);
      designedValues.write(next.designed.weights);
      designedOrder.write(next.designed.order);
      hotOrder.write(Uint32Array.from(next.designed.order.subarray(designedCount - hotCount)));
    }
    // Raw lines: the same list, straight, in the list's own order.
    rawCount = raw.count;
    if (rawCount > 0) {
      rawFlows.write(raw.flows);
      rawValues.write(raw.weights);
    }

    // Interior circles: the largest same-zone flows, largest drawn first.
    const circles = interiorRows
      .slice()
      .sort((a, b) => b.weight - a.weight)
      .slice(0, INTERIOR_CIRCLE_COUNT);
    const circlePositions = new Float32Array(INTERIOR_CIRCLE_COUNT * 2);
    const circleSizes = new Float32Array(INTERIOR_CIRCLE_COUNT);
    interiorCount = 0;
    for (const {zone, weight} of circles) {
      const center = getCenter(zone);
      if (!center) continue;
      circlePositions.set(center, interiorCount * 2);
      circleSizes[interiorCount] = weight;
      interiorCount++;
    }
    interiorPositions.write(circlePositions);
    interiorSizes.write(circleSizes);

    writeDensity(next);
    ctx.requestLayers();

    // Readouts.
    const shown = next.designed.count;
    let shownWeight = 0;
    for (let flow = 0; flow < shown; flow++) shownWeight += next.designed.weights[flow];
    ctx.setReadout('volume', total > 0 ? formatWeight(total) : 'none in window');
    ctx.setReadout('pairs', pairCount);
    ctx.setReadout('drawn', shown);
    ctx.setReadout('share', between > 0 ? shownWeight / between : null);
    ctx.setReadout('interiorShare', total > 0 && !options.excludeSelf ? interior / total : null);
    const biggestInterior = circles[0];
    ctx.setReadout(
      'interiorTop',
      biggestInterior
        ? `${getZoneName(next, biggestInterior.zone)}: ${formatWeight(biggestInterior.weight)}`
        : null
    );
    ctx.setReadout(
      'truncated',
      pairCount > k ? `yes, ${formatCount(pairCount - k)} pairs not listed` : 'no'
    );
    ctx.setReadout(
      'pairTable',
      `${formatCount(pairCount)} of ${formatCount(current.pairCapacity)} slots`
    );
    ctx.setReadout('pairOverflow', pairOverflow ? 'yes: totals incomplete' : 'no');
    ctx.setReadout('activeZones', `${formatCount(activeZones)} of ${formatCount(zoneCount)}`);
    ctx.setCost({records: rows});
    const topIndex = next.designed.count > 0 ? 0 : -1;
    ctx.setReadout(
      'topFlow',
      topIndex >= 0
        ? `${getZoneName(next, next.designed.originZones[0])} to ${getZoneName(next, next.designed.destinationZones[0])}: ${formatWeight(next.designed.weights[0])}`
        : null
    );
    let topTwentyFive = 0;
    for (let flow = 0; flow < Math.min(25, raw.count); flow++) topTwentyFive += raw.weights[flow];
    ctx.setReadout('top25Share', between > 0 ? topTwentyFive / between : null);
    if (source.id === 'commute') {
      ctx.setReadout(
        'jobsTop',
        maximumIn > 0 ? `${formatCount(maximumIn)} jobs, ${getZoneName(next, busiestIn)}` : null
      );
    }

    updateConcentrationChart(next, raw);
    updateMatrix(next);
    updateNotes(next);
  }

  /** Writes the classed backdrop values: the zone total per km2 per hour of the window. */
  function writeDensity(current: Snapshot): void {
    const {graph: owner} = current;
    const {source} = owner.prepared;
    const values = new Float32Array(owner.zoneCount + 1);
    values[owner.zoneCount] = Number.NaN;
    const arrivals = source.id === 'commute' || ctx.options.totals === 'arrivals';
    const zoneTotals = arrivals ? current.incoming : current.out;
    const divisor = getRateDivisor(source.id, current.dayType, current.windowHours);
    for (let zone = 0; zone < owner.zoneCount; zone++) {
      const weight = zoneTotals[zone];
      values[zone] = weight > 0 ? weight / Math.max(getZoneArea(current, zone), 1e-6) / divisor : 0;
    }
    owner.density.write(values);
  }

  function updateConcentrationChart(current: Snapshot, all: FlowArrows): void {
    const {designed, between} = current;
    // The cumulative curve runs over every between-zone pair of the list, not only the drawn ones.
    if (all.count < 2 || between <= 0) {
      ctx.setChart('concentrationChart', null);
      return;
    }
    const ranks = new Float64Array(all.count);
    const share = new Float64Array(all.count);
    let running = 0;
    for (let flow = 0; flow < all.count; flow++) {
      running += all.weights[flow];
      ranks[flow] = flow + 1;
      share[flow] = (100 * running) / between;
    }
    ctx.setChart('concentrationChart', {
      kind: 'line',
      series: [{label: 'cumulative share', x: ranks, y: share, area: true, color: 2}],
      yDomain: [0, 100],
      markers: [{x: Math.max(1, designed.count), label: 'drawn'}],
      link: {option: 'arcs', label: value => `${value} flows`},
      xLabel: 'flows, largest first',
      yLabel: '% of trips between areas',
      height: 120,
      formatY: value => `${value.toFixed(0)}%`,
      description:
        'Cumulative share of the trips between areas carried by the largest flows. A steep curve means a few pairs dominate; the rule is the number of flows drawn.'
    });
  }

  /** The origin-destination matrix of the ranked list, areas ordered by throughput. */
  function updateMatrix(current: Snapshot): void {
    const {source} = current.graph.prepared;
    if (source.id !== 'taxi' || current.lattice || current.total <= 0) {
      ctx.setChart('odMatrix', null);
      return;
    }
    const count = source.zoneCount;
    const throughput = Array.from(
      {length: count},
      (_, zone) => current.out[zone] + current.incoming[zone]
    );
    const order = Int32Array.from(
      Array.from({length: count}, (_, zone) => zone).sort((a, b) => throughput[b] - throughput[a])
    );
    const position = new Int32Array(count);
    order.forEach((zone, index) => {
      position[zone] = index;
    });
    const values = new Float32Array(count * count).fill(Number.NaN);
    for (let row = 0; row < current.flowRows; row++) {
      values[position[current.topOrigin[row]] * count + position[current.topDestination[row]]] =
        current.topWeight[row];
    }
    const names = Array.from(order, zone => getAreaDisplayName(source.zoneNames[zone]));
    const fractions = [0.01, 0.05, 0.15, 0.4];
    const chart: MatrixChartData = {
      kind: 'matrix',
      values,
      rows: count,
      columns: count,
      rowLabels: names,
      columnLabels: names,
      ramp: 'oranges',
      breaks: fractions.map(fraction => fraction * flowScale.maxFlow),
      diagonal: true,
      highlight: selectedPair
        ? {
            row: position[selectedPair.origin],
            column: position[selectedPair.destination]
          }
        : undefined,
      onCellClick: (row, column) => selectPair(order[row], order[column]),
      formatCell: value => formatCount(value),
      xLabel: 'destination',
      yLabel: 'origin',
      height: 300,
      description:
        'Trips between the 77 community areas of the ranked list, origin down and destination across, areas ordered by throughput. The outlined diagonal is trips that start and end in one area.'
    };
    ctx.setChart('odMatrix', chart);
  }

  let zoneRings: Map<number, MapHighlight> | null = null;

  function getZoneHighlight(zone: number): MapHighlight | null {
    if (!zoneRings) {
      zoneRings = new Map();
      for (const feature of areas.geojson?.features ?? []) {
        const id = Number(feature.properties?.id) - 1;
        const rings = getGeometryPolygons(feature.geometry).map(polygon =>
          polygon[0].map(vertex => [vertex[0], vertex[1]] as [number, number])
        );
        zoneRings.set(id, {kind: 'polygon', rings});
      }
    }
    return zoneRings.get(zone) ?? null;
  }

  function selectPair(origin: number, destination: number): void {
    selectedPair =
      selectedPair?.origin === origin && selectedPair.destination === destination
        ? null
        : {origin, destination};
    const highlights = selectedPair
      ? [getZoneHighlight(origin), getZoneHighlight(destination)].filter(
          (entry): entry is MapHighlight => entry !== null
        )
      : [];
    ctx.setHighlight(highlights.length ? highlights : null);
    if (snapshot) updateMatrix(snapshot);
  }

  /** Finding notes on the map, from the readback: the step picks which through `annotate`. */
  function updateNotes(current: Snapshot): void {
    const {annotate} = ctx.options;
    const notes: MapAnnotation[] = [];
    const flowNote = (index: number): MapAnnotation => ({
      kind: 'note',
      coordinate: toLngLat(getFlowMidpoint(current.designed, index)),
      title: `${getZoneName(current, current.designed.originZones[index])} to ${getZoneName(current, current.designed.destinationZones[index])}`,
      text: formatWeight(current.designed.weights[index]),
      id: `flow-note-${index}`
    });
    if (annotate === 'top-flows') {
      for (let index = 0; index < Math.min(3, current.designed.count); index++) {
        notes.push(flowNote(index));
      }
    } else if (annotate === 'top-flow' && current.designed.count > 0) {
      notes.push(flowNote(0));
    } else if (annotate === 'interior' && current.interiorRows.length) {
      const largest = current.interiorRows.reduce((a, b) => (b.weight > a.weight ? b : a));
      const center = getZoneCenter(current, largest.zone);
      if (center) {
        notes.push({
          kind: 'note',
          coordinate: toLngLat(center),
          title: getZoneName(current, largest.zone),
          text: `${formatWeight(largest.weight)} start and end here`,
          id: 'interior-note'
        });
      }
    }
    ctx.setAnnotations('findings', notes.length ? notes : null);
  }

  // ---------------------------------------------------------------------------------------------
  // Tooltips
  // ---------------------------------------------------------------------------------------------

  function getTooltip(coordinate: readonly [number, number]): TooltipContent | null {
    if (!graph || !snapshot || snapshot.graph !== graph) return null;
    const current = snapshot;
    const {source} = graph.prepared;
    const [x, y] = projection.project(coordinate[0], coordinate[1]);
    // A flow under the pointer: the nearest midpoint of the drawn flows, within 12 pixels.
    if (ctx.options.flowStyle === 'designed') {
      const reach = 12 * ctx.getMetersPerPixel();
      let nearest = -1;
      let nearestDistance = reach;
      for (let index = 0; index < current.designed.count; index++) {
        const [mx, my] = getFlowMidpoint(current.designed, index);
        const distance = Math.hypot(mx - x, my - y);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = index;
        }
      }
      if (nearest >= 0) {
        const weight = current.designed.weights[nearest];
        return {
          title: `${getZoneName(current, current.designed.originZones[nearest])} to ${getZoneName(current, current.designed.destinationZones[nearest])}`,
          subtitle: `Flow, rank ${nearest + 1} of ${current.designed.count} drawn`,
          rows: [
            {
              label: source.id === 'taxi' ? 'Trips' : 'Jobs',
              value: formatCount(weight),
              emphasis: true
            },
            {
              label: 'Share of flow between areas',
              value: formatPercent(current.between > 0 ? weight / current.between : 0, 1)
            }
          ]
        };
      }
    }
    let zone = -1;
    if (current.lattice) zone = getLatticeZone(current.lattice, x, y);
    else zone = getZoneAt(source.raster, x, y);
    if (zone < 0 || zone >= current.out.length) return null;
    const arrivals = source.id === 'commute' || ctx.options.totals === 'arrivals';
    const total = arrivals ? current.incoming[zone] : current.out[zone];
    const divisor = getRateDivisor(source.id, current.dayType, current.windowHours);
    const density = total > 0 ? total / Math.max(getZoneArea(current, zone), 1e-6) / divisor : 0;
    const units = source.weightUnits[getWeightKey()];
    const swatch =
      zoneTable && density > 0 ? zoneTable.colors[getClassIndexOf(zoneTable, density)] : undefined;
    const noun = source.id === 'commute' ? 'Jobs arriving' : arrivals ? 'Arrivals' : 'Departures';
    const rows: TooltipRow[] = [
      {
        label: `${noun} per km²`,
        value: formatDensity(density),
        unit: source.id === 'commute' ? 'jobs' : `${units.unit} per hour`,
        swatch: swatch ? [swatch[0], swatch[1], swatch[2], 255] : undefined,
        emphasis: true
      },
      {label: arrivals ? 'Arrivals' : 'Departures', value: formatCount(total), unit: units.unit},
      ...(source.id === 'taxi'
        ? [
            {
              label: arrivals ? 'Departures' : 'Arrivals',
              value: formatCount(arrivals ? current.out[zone] : current.incoming[zone]),
              unit: units.unit
            },
            {
              label: 'Net (arrivals minus departures)',
              value: formatSigned(current.incoming[zone] - current.out[zone]),
              unit: units.unit
            }
          ]
        : []),
      {
        label: `Share of ${source.id === 'taxi' ? 'all trips' : 'all jobs'} in the window`,
        value: formatPercent(current.total > 0 ? total / current.total : 0, 1)
      }
    ];
    return {
      title: getZoneName(current, zone),
      subtitle: current.lattice
        ? 'Zone of the lattice'
        : source.id === 'taxi'
          ? 'Community area'
          : 'Census tract',
      rows,
      note: `Rank ${formatOrdinal(1 + countGreater(arrivals ? current.incoming : current.out, total))} of ${formatCount(current.out.length)} zones`
    };
  }

  function countGreater(values: Float32Array, value: number): number {
    let greater = 0;
    for (let index = 0; index < values.length; index++) if (values[index] > value) greater++;
    return greater;
  }

  // ---------------------------------------------------------------------------------------------
  // Summation orders
  // ---------------------------------------------------------------------------------------------

  /** Times the current sum order against the other one and compares their zone totals. */
  async function compareSumOrders(): Promise<void> {
    if (!graph || comparison || comparing) return;
    comparing = true;
    const main = graph;
    const otherOrder = main.sumOrder === 'sorted' ? 'atomic' : 'sorted';
    const alt = buildGraph(main.prepared, main.kind, otherOrder, ctx.options.excludeSelf);
    try {
      const first = await measureCompiledGraph(device, main.compiled, {
        parameters: undefined,
        completionBuffer: main.count
      });
      const second = await measureCompiledGraph(device, alt.compiled, {
        parameters: undefined,
        completionBuffer: alt.count
      });
      if (destroyed || graph !== main) {
        alt.resources.destroy();
        return;
      }
      const label = (order: string, value: number) => `${order} ${value.toFixed(2)} ms`;
      ctx.setReadout(
        'sumTiming',
        `${label(main.sumOrder, first.milliseconds)} · ${label(otherOrder, second.milliseconds)}`
      );
      // Both graphs encode in the next frames; the reader then diffs their zone totals once.
      const reader = new SummaryReader(
        alt.resources,
        'taxi-flows-compare',
        [
          {buffer: main.zoneOutWeights, size: main.zoneCount * 4},
          {buffer: alt.zoneOutWeights, size: alt.zoneCount * 4}
        ],
        bytes => {
          const values = new Float32Array(bytes);
          const firstTotals = values.subarray(0, main.zoneCount);
          const secondTotals = values.subarray(main.zoneCount);
          let difference = 0;
          let largest = 0;
          for (let zone = 0; zone < main.zoneCount; zone++) {
            difference = Math.max(difference, Math.abs(firstTotals[zone] - secondTotals[zone]));
            largest = Math.max(largest, firstTotals[zone]);
          }
          ctx.setReadout(
            'sumDifference',
            difference === 0
              ? 'identical (bitwise)'
              : `${difference.toExponential(2)} (largest zone ${formatCount(largest)})`
          );
          retired.push({resources: alt.resources, frames: 0});
          comparison = null;
        }
      );
      comparison = {alt, reader, main};
      markChanged();
    } catch (error) {
      alt.resources.destroy();
      if (!destroyed) ctx.setReadout('sumDifference', `comparison failed: ${String(error)}`);
    } finally {
      comparing = false;
    }
  }

  // The standing sample line of the cartouche, counted from the loaded rows.
  {
    const taxi = prepare('taxi').source;
    let trips = 0;
    for (let row = 0; row < taxi.rowCount; row++) trips += taxi.weights.trips[row];
    furniture.title.sample = `${formatCount(trips)} taxi trips, ${taxi.zoneCount} community areas, 2023`;
  }
  rebuild();

  return {
    getCompiledGraphs: () => (graph ? [graph.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id, value, state) {
      switch (id) {
        case 'source':
        case 'zones':
        case 'excludeSelf':
        case 'sumOrder':
          rebuild();
          ctx.requestLayers();
          break;
        case 'taxiWeight':
        case 'commuteWeight':
          writeWeights();
          break;
        case 'dayType':
          writeMask();
          refreshZoneTable();
          break;
        case 'zoneSize':
          writeLattice();
          markChanged();
          ctx.requestLayers();
          break;
        case 'hours': {
          const [from, to] = state.hours;
          windowLength = Math.max(1, to - from);
          clock.setRange(0, Math.max(0, 24 - windowLength));
          if (state.hourStart !== from) ctx.setOptions({hourStart: from});
          markChanged();
          break;
        }
        case 'hourStart': {
          const [from, to] = state.hours;
          const length = Math.max(1, to - from);
          const start = Math.min(Number(value), Math.max(0, 24 - length));
          if (start !== from) ctx.setOptions({hours: [start, start + length]});
          if (start !== Number(value)) ctx.setOptions({hourStart: start});
          markChanged();
          break;
        }
        case 'play':
          // Leaving play settles the playhead on the window the sweep reached.
          if (!value) clock.seek(lastWindow[0]);
          markChanged();
          break;
        case 'playSpeed':
        case 'loop':
          markChanged();
          break;
        case 'totals':
          refreshZoneTable();
          if (snapshot) writeDensity(snapshot);
          ctx.requestLayers();
          break;
        case 'annotate':
          if (snapshot) updateNotes(snapshot);
          break;
        case 'arcs':
        case 'flowStyle':
          markChanged();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'compareSum') void compareSumOrders();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onGroundChange() {
      refreshZoneTable();
      ctx.requestLayers();
    },

    onCompareChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      return event.coordinate ? getTooltip(event.coordinate) : null;
    },

    encode(commandEncoder, frame) {
      if (!graph) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      const taxi = Boolean(graph.prepared.hours);
      const {play} = ctx.options;
      if (taxi) {
        clock.advance(frame);
        let start: number;
        let end: number;
        if (play) {
          // Playing slides the window one whole hour at a time and writes it back for the bar.
          start = Math.floor(clock.time);
          end = Math.min(24, start + windowLength);
          if (start !== lastWindow[0]) ctx.setOptions({hours: [start, end]});
        } else {
          [start, end] = ctx.options.hours;
        }
        if (start !== lastWindow[0] || end !== lastWindow[1]) {
          lastWindow = [start, end];
          markChanged();
        }
        // The window test is closed at both ends (start <= t <= end) and pickup hours are whole
        // numbers, so [17, 20) is written as [17, 19.5] to keep the 20:00 hour out.
        windowBuffer.write(getGPUTimeWindowParameterValues({start, end: end - 0.5}));
        if (frame.frameIndex % 6 === 0) {
          ctx.setReadout('window', `${formatTaxiDayHour(start)} to ${formatTaxiDayHour(end)}`);
          updateHourlyChart(start, end);
        }
      } else if (frame.frameIndex === 0) {
        ctx.setReadout('window', 'not applicable (no timestamps)');
      }
      if (encodeFrames > 0) {
        graph.compiled.encode(commandEncoder, {parameters: undefined});
        if (comparison) comparison.alt.compiled.encode(commandEncoder, {parameters: undefined});
        encodeFrames--;
      }
      if (comparison && encodeFrames === 0 && !comparison.reader.isPending) {
        comparison.reader.request(commandEncoder);
      }
      // One summary read per change, a few frames after the graph last ran.
      if (statsStale && encodeFrames === 0 && !graph.reader.isPending && !(taxi && play)) {
        statsStale = false;
        graph.reader.request(commandEncoder);
      } else if (taxi && play && frame.frameIndex % 8 === 0 && !graph.reader.isPending) {
        graph.reader.request(commandEncoder);
      } else {
        graph.reader.flush(commandEncoder);
      }
      comparison?.reader.flush(commandEncoder);
    },

    getLayers() {
      if (!graph) return [];
      const options = ctx.options;
      const {prepared} = graph;
      const {source} = prepared;
      const coordinateOrigin: [number, number, number] = [CHICAGO_ORIGIN[0], CHICAGO_ORIGIN[1], 0];
      const ground = ctx.ground();
      const dark = ground === 'dark';
      const lattice = getActiveLattice(graph);
      const compare = ctx.getCompare();
      const layers: Layer[] = [];

      // 1. The quiet backdrop: classed zone totals per km2 (the flows own the warm hue).
      if (options.showZones && zoneTable) {
        const classes = getClassTableLayerProps(zoneTable);
        const common = {
          values: graph.density,
          valueFormat: 'float32' as const,
          colormap: 'greys' as const,
          ...classes,
          discardAtOrBelow: 0
        };
        layers.push(
          new SpatialAnalysisRasterLayer(
            lattice
              ? {
                  id: `flow-zones-${graph.kind}`,
                  coordinateOrigin,
                  gridSize: [lattice.columns, lattice.rows],
                  bounds: lattice.bounds,
                  binning: lattice.kind === 'hexagon' ? 'hexagon' : 'grid',
                  hexagonRadius: lattice.size,
                  ...common
                }
              : {
                  id: `flow-zones-${source.id}`,
                  coordinateOrigin,
                  gridSize: [source.raster.columns, source.raster.rows],
                  bounds: source.raster.bounds,
                  binning: 'grid',
                  valueIndices: prepared.zoneIdRaster,
                  ...common
                }
          )
        );
      }
      // 2. Zone outlines: 0.5 px context ink.
      if (options.showOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `flow-outline-${source.id}`,
            coordinateOrigin,
            segments: prepared.outline,
            instanceCount: source.raster.outline.length / 4,
            widthPixels: 0.5,
            color: inkFor(CONTEXT_INK, ground)
          })
        );
      }
      // 3. Interior circles under the arrows: area by the square root of the trips.
      if (!options.excludeSelf && interiorCount > 0) {
        const ink = inkFor(FLOW_INK, ground);
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'flow-interior',
            coordinateOrigin,
            positions: interiorPositions,
            instanceCount: interiorCount,
            sizeValues: interiorSizes,
            sizeMaximumValue: Math.max(flowScale.maxInterior, 1),
            sizeScale: 'sqrt',
            radiusPixels: MAXIMUM_INTERIOR_RADIUS,
            shape: 'circle',
            color: withInkAlpha(ink, 64),
            outlineColor: ink,
            outlineWidthPixels: 1
          })
        );
      }
      // 4. Flows: raw straight lines, and the designed arrows (heaviest last).
      const ink = inkFor(FLOW_INK, ground);
      const common = {
        coordinateOrigin,
        maxValue: Math.max(flowScale.maxFlow, 1),
        maxWidthPixels: MAXIMUM_FLOW_WIDTH,
        valueFormat: 'float32' as const
      };
      const showRaw = compare ? true : options.flowStyle === 'raw';
      const showDesigned = compare ? true : options.flowStyle === 'designed';
      if (showRaw && rawCount > 0) {
        layers.push(
          new SpatialAnalysisFlowLayer({
            ...common,
            id: 'flow-raw',
            flows: rawFlows,
            values: rawValues,
            instanceCount: rawCount,
            widthFromValues: false,
            widthPixels: 1,
            curvature: 0,
            arrowheads: false,
            color: withInkAlpha(ink, dark ? 64 : 80),
            blending: dark ? 'additive' : 'normal',
            compareSide: compare ? 'b' : undefined
          })
        );
      }
      if (showDesigned && designedCount > 0) {
        const designed = {
          ...common,
          flows: designedFlows,
          values: designedValues,
          ids: designedOrder,
          curvature: FLOW_CURVATURE,
          arrowheads: true,
          endOffsetPixels: 1,
          blending: dark ? ('additive' as const) : ('normal' as const),
          outlineColor: inkFor(FLOW_HALO, ground),
          outlineWidthPixels: dark ? 0 : 0.8,
          compareSide: compare ? ('a' as const) : undefined
        };
        layers.push(
          new SpatialAnalysisFlowLayer({
            ...designed,
            id: 'flow-designed',
            instanceCount: designedCount,
            color: ink
          })
        );
        // The heaviest few in the hot ink, on the night ground only.
        if (dark && hotCount > 0) {
          layers.push(
            new SpatialAnalysisFlowLayer({
              ...designed,
              id: 'flow-hot',
              ids: hotOrder,
              instanceCount: hotCount,
              color: inkFor(FLOW_INK_HOT, ground)
            })
          );
        }
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      comparison?.reader.stop();
      graph?.reader.stop();
      graph?.resources.destroy();
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
    }
  };
}
