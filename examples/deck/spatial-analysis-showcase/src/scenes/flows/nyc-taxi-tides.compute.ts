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
import {getClassIndex} from '../../cartography/breaks';
import {getClassTableLayerProps} from '../../cartography/class-table';
import {type Gazetteer, NYC, nearestPlaceLabel} from '../../cartography/gazetteer';
import {formatCount, formatDistance, formatSigned} from '../../cartography/live-text';
import type {LngLat, MapAnnotation} from '../../cartography/types';
import {SpatialAnalysisFlowLayer} from '../../engine/flow-layer';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {buildFlowArrows, FLOW_HALO, FLOW_INK} from './flows-style';
import {
  buildZoneOutlines,
  getLattice,
  getZoneAt,
  getZoneCenter,
  getZoneCorners,
  type Lattice,
  type ZoneKind
} from './nyc-taxi-tides-lattice';
import {
  BALANCED_TRIPS,
  formatLongWindow,
  formatShare,
  formatShortWindow,
  getHatchColor,
  getMagnitudeTable,
  getNetTable,
  getShareTable,
  getZoneOutlineColor,
  getZoneTooltip,
  MASKED_SHARE
} from './nyc-taxi-tides-style';
import {
  AREA_PRESETS,
  formatTaxiTime,
  loadTaxiTrips,
  NYC_TAXI_HOURS,
  NYC_TAXI_ORIGIN,
  type TaxiTrips
} from './nyc-taxi-data';

/** Option state of the nyc-taxi-tides scene. */
export type NycTaxiTidesOptions = {
  play: boolean;
  time: number;
  playSpeed: number;
  loop: boolean;
  windowHours: number;
  normalise: 'net' | 'share';
  minVolume: number;
  compareMidpoint: boolean;
  zoneSize: number;
  zones: ZoneKind;
  excludeSelf: boolean;
  sumOrder: 'sorted' | 'atomic';
  flowCount: number;
  showRadius: boolean;
  showOutlines: boolean;
};

/**
 * First and last hour the time slider reaches. The archive has no trips that began on 31 December
 * and none that begin after 38.75 h, so the arrivals of the first and last hour are incomplete.
 */
export const TIDES_FIRST_HOUR = 1;
export const TIDES_LAST_HOUR = 37.5;
/** Rows of the top-K flow list (compile-time capacity). */
const TOP_FLOW_COUNT = 256;
/** Smallest lattice size on the slider: it sets the compile-time lattice capacity. */
const MINIMUM_ZONE_SIZE = 400;
/** Words before the zone totals in the readback: flow rows, pairs and two overflow flags. */
const HEADER_WORDS = 4;
/** Distinct origin and destination pairs the hash table keeps. */
const PAIR_CAPACITY = 524288;
const RETIRE_FRAMES = 4;
/** Playback readback interval in frames. */
const PLAYBACK_READ_FRAMES = 6;
/** Fill opacity of the zones: the ground shows through a little. */
const FILL_OPACITY = {light: 0.88, dark: 0.9} as const;
/** Flow widths: 1.2 px for the smallest, 3.5 px for the heaviest. */
const FLOW_WIDTH_PIXELS = [1.2, 3.5] as const;
/** Trips in the time window are counted per hour for the time bar. */
const TIME_BAR_BINS = 36;

/** The gazetteer without the borough poles, so a finding is named by a place, not a borough. */
const NYC_PLACES: Gazetteer = {
  ...NYC,
  places: Object.fromEntries(
    Object.entries(NYC.places).filter(
      ([id]) => !['manhattan', 'brooklyn', 'queens', 'bronx', 'staten-island'].includes(id)
    )
  )
};

type SideGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  count: Buffer;
  requiredCount: Buffer;
  pairOverflow: Buffer;
  originZones: Buffer;
  destinationZones: Buffer;
  flowWeights: Buffer;
  zoneOutWeights: Buffer;
  zoneInWeights: Buffer;
  zoneOutCounts: Buffer;
  zoneInCounts: Buffer;
};

type TideGraphs = {
  resources: SpatialAnalysisResources;
  kind: ZoneKind;
  zoneCount: number;
  departures: SideGraph;
  arrivals: SideGraph;
  /** Net or share per zone (NaN where there are no trips) plus one NaN sentinel row. */
  values: Buffer;
  /** |net| per zone, for the "wrong map" of the compare step. */
  magnitudes: Buffer;
  /** Outline segments of the zones with data, six per zone at most. */
  outlines: Buffer;
  reader: SummaryReader;
};

/** A named box in the planar frame, for the hourly net charts and the airport readouts. */
type TideBox = {
  label: string;
  centerX: number;
  centerY: number;
  halfWidth: number;
  halfHeight: number;
};

/** Statistics of one set of zone totals. */
type TideStats = {
  lattice: Lattice;
  gainZone: number;
  lossZone: number;
  gain: number;
  loss: number;
  citySum: number;
  maxNet: number;
  activeZones: number;
  balancedZones: number;
  hiddenZones: number;
  sharpestZone: number;
  netClassCounts: number[];
  shareClassCounts: number[];
  lgaNet: number;
  jfkNet: number;
};

/**
 * Net taxi arrivals minus departures on a hexagon lattice, by time window. Two `GPUFlowAggregation`
 * graphs read the same 440,000 origin and destination pairs: one gates trips on their pickup time
 * (departures), the other on their dropoff time (arrivals). Both windows are one parameter buffer
 * that the playback clock rewrites every frame; the zone size is a per-frame parameter under a
 * compile-time capacity. The zone totals come back once per update; the CPU assembles the net, the
 * imbalance share and the top-K arrows, and the layers draw from the buffers it writes.
 */
export async function createNycTaxiTides(
  ctx: SceneContext<NycTaxiTidesOptions>
): Promise<SceneInstance<NycTaxiTidesOptions>> {
  const {device} = ctx;
  const trips: TaxiTrips = loadTaxiTrips(ctx.datasets.get('poopdeck-nyc-taxi'));
  const count = trips.count;
  const resources = new SpatialAnalysisResources(device, 'taxi-tides');
  const coordinateOrigin: [number, number, number] = [NYC_TAXI_ORIGIN[0], NYC_TAXI_ORIGIN[1], 0];

  const originsBuffer = resources.createBuffer('origins', trips.pickup);
  const destinationsBuffer = resources.createBuffer('destinations', trips.dropoff);
  const pickupHourBuffer = resources.createBuffer('pickup-hour', trips.pickupHour);
  const dropoffHourBuffer = resources.createBuffer('dropoff-hour', trips.dropoffHour);
  const weightsBuffer = resources.createBuffer('weights', new Float32Array(count).fill(1));
  const radiusBuffer = resources.createParameterBuffer('radius', 'float32', 1);
  const activeGridBuffer = resources.createParameterBuffer('active-grid', 'uint32', 2);
  const boundsBuffer = resources.createParameterBuffer('bounds', 'float32', 4);
  const windowBuffer = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  // The heaviest flows as arrows: endpoints, weights and draw order, rewritten after each readback.
  const flowEndpointsBuffer = resources.createBuffer('flow-endpoints', TOP_FLOW_COUNT * 16);
  const flowWeightsBuffer = resources.createBuffer('flow-values', TOP_FLOW_COUNT * 4);
  const flowOrderBuffer = resources.createBuffer('flow-order', TOP_FLOW_COUNT * 4);

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [TIDES_FIRST_HOUR, TIDES_LAST_HOUR], rate: 1, step: 0.25}
  );

  // ---- Context computed once from the trips on the CPU -----------------------------------------
  const boxes = createBoxes(trips);
  const hourlyNet = getHourlyBoxNet(trips, boxes);
  const medianTripMinutes = getMedianTripMinutes(trips);
  publishTimeBar();

  let graphs: TideGraphs | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let statsStale = true;
  let lastWindowStart = Number.NaN;
  let lastWindowEnd = Number.NaN;
  let windowStart = ctx.options.time;
  let windowEnd = ctx.options.time + ctx.options.windowHours;
  let lastOutCounts = new Uint32Array(0);
  let lastInCounts = new Uint32Array(0);
  let lastFlowOrigins = new Uint32Array(0);
  let lastFlowDestinations = new Uint32Array(0);
  let lastFlowWeights = new Float32Array(0);
  let lastFlowRows = 0;
  let stats: TideStats | null = null;
  let drawnFlows = 0;
  let flowMaximum = 0;
  let outlineCount = 0;
  let legendHighlight: number[] | null = null;
  let lastFurnitureKey = '';
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 3);
    statsStale = true;
  }

  function getCurrentLattice(kind: ZoneKind = graphs?.kind ?? ctx.options.zones): Lattice {
    return getLattice(kind, ctx.options.zoneSize, trips.bounds);
  }

  /**
   * Blanks the zone buffers until the next readback: the totals on hand belong to the previous
   * lattice and would be drawn on the new one.
   */
  function clearZoneValues(): void {
    if (!graphs) return;
    const blank = new Float32Array(graphs.zoneCount + 1).fill(Number.NaN);
    graphs.values.write(blank);
    graphs.magnitudes.write(blank);
    lastOutCounts = new Uint32Array(0);
    lastInCounts = new Uint32Array(0);
    stats = null;
    outlineCount = 0;
    drawnFlows = 0;
    ctx.setAnnotations('tide-notes', null);
    ctx.setAnnotations('hex-radius', null);
  }

  function writeLattice(): void {
    if (!graphs) return;
    const lattice = getCurrentLattice(graphs.kind);
    radiusBuffer.write(Float32Array.of(lattice.size));
    activeGridBuffer.write(Uint32Array.of(lattice.grid[0], lattice.grid[1]));
    boundsBuffer.write(Float32Array.from(lattice.bounds));
    ctx.setReadout(
      'zones',
      `${lattice.grid[0]} × ${lattice.grid[1]} ${graphs.kind === 'hexagon' ? 'hexagons' : 'cells'} (capacity ${formatCount(graphs.zoneCount)})`
    );
    ctx.setReadout('cellSize', formatDistance(lattice.size));
  }

  function buildSide(
    graphResources: SpatialAnalysisResources,
    name: string,
    kind: ZoneKind,
    capacityGrid: [number, number],
    zoneCount: number,
    gateBuffer: Buffer
  ): SideGraph {
    const sentinel = (rows: number) => {
      const values = new Float32Array(rows + 1);
      values[rows] = Number.NaN;
      return values;
    };
    const k = TOP_FLOW_COUNT;
    const ids = graphResources.createBuffer(`${name}-ids`, k * 4);
    const flowCount = graphResources.createBuffer(`${name}-count`, 4);
    const requiredCount = graphResources.createBuffer(`${name}-total-count`, 4);
    const overflow = graphResources.createBuffer(`${name}-overflow`, 4);
    const pairOverflow = graphResources.createBuffer(`${name}-pair-overflow`, 4);
    const originZones = graphResources.createBuffer(`${name}-flow-origin`, k * 4);
    const destinationZones = graphResources.createBuffer(`${name}-flow-destination`, k * 4);
    const flowWeights = graphResources.createBuffer(`${name}-flow-weights`, k * 4);
    const zoneOutWeights = graphResources.createBuffer(`${name}-out-w`, sentinel(zoneCount));
    const zoneInWeights = graphResources.createBuffer(`${name}-in-w`, sentinel(zoneCount));
    const zoneOutCounts = graphResources.createBuffer(`${name}-out-c`, zoneCount * 4);
    const zoneInCounts = graphResources.createBuffer(`${name}-in-c`, zoneCount * 4);

    const commandGraph = new GPUCommandGraph<void>(device, {id: `taxi-tides-${name}-${serial}`});
    const zones =
      kind === 'hexagon'
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
    const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
      id: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(commandGraph, `${name}-${id}`, buffer, format, length);
    commandGraph.add(
      new GPUFlowAggregation({
        id: `flows-${name}`,
        zones,
        origins: view('origins', originsBuffer, 'float32x2', count),
        destinations: view('destinations', destinationsBuffer, 'float32x2', count),
        weights: view('weights', weightsBuffer, 'float32', count),
        timeWindow: {
          timestamps: view('gate', gateBuffer, 'float32', count),
          window: windowBuffer.importToGraph(commandGraph)
        },
        excludeSelfFlows: ctx.options.excludeSelf,
        sumOrder: ctx.options.sumOrder,
        pairCapacity: PAIR_CAPACITY,
        // rows * maxProbeCount must fit in a uint32.
        maxProbeCount: 128,
        output: {
          ids: view('ids', ids, 'uint32', k),
          count: view('count', flowCount, 'uint32', 1),
          overflow: view('overflow', overflow, 'uint32', 1),
          requiredCount: view('total-count', requiredCount, 'uint32', 1)
        },
        flowOriginZoneIds: view('flow-origin', originZones, 'uint32', k),
        flowDestinationZoneIds: view('flow-destination', destinationZones, 'uint32', k),
        flowWeights: view('flow-weights', flowWeights, 'float32', k),
        pairOverflow: view('pair-overflow', pairOverflow, 'uint32', 1),
        zoneOutWeights: view('out-w', zoneOutWeights, 'float32', zoneCount),
        zoneInWeights: view('in-w', zoneInWeights, 'float32', zoneCount),
        zoneOutCounts: view('out-c', zoneOutCounts, 'uint32', zoneCount),
        zoneInCounts: view('in-c', zoneInCounts, 'uint32', zoneCount)
      })
    );
    return {
      compiled: graphResources.track(commandGraph.compile()),
      count: flowCount,
      requiredCount,
      pairOverflow,
      originZones,
      destinationZones,
      flowWeights,
      zoneOutWeights,
      zoneInWeights,
      zoneOutCounts,
      zoneInCounts
    };
  }

  function buildGraphs(): TideGraphs {
    const id = ++serial;
    const kind = ctx.options.zones;
    const graphResources = new SpatialAnalysisResources(device, `taxi-tides-${id}`);
    const capacity = getLattice(kind, MINIMUM_ZONE_SIZE, trips.bounds);
    const zoneCount = capacity.grid[0] * capacity.grid[1];
    const nanRows = () => new Float32Array(zoneCount + 1).fill(Number.NaN);
    const values = graphResources.createBuffer('values', nanRows());
    const magnitudes = graphResources.createBuffer('magnitudes', nanRows());
    const outlines = graphResources.createBuffer('outlines', zoneCount * 6 * 16);
    const departures = buildSide(
      graphResources,
      'dep',
      kind,
      capacity.grid,
      zoneCount,
      pickupHourBuffer
    );
    const arrivals = buildSide(
      graphResources,
      'arr',
      kind,
      capacity.grid,
      zoneCount,
      dropoffHourBuffer
    );
    const built: TideGraphs = {
      resources: graphResources,
      kind,
      zoneCount,
      departures,
      arrivals,
      values,
      magnitudes,
      outlines,
      reader: undefined as unknown as SummaryReader
    };
    const k = TOP_FLOW_COUNT;
    built.reader = new SummaryReader(
      graphResources,
      `taxi-tides-${id}`,
      [
        {buffer: departures.count, size: 4},
        {buffer: departures.requiredCount, size: 4},
        {buffer: departures.pairOverflow, size: 4},
        {buffer: arrivals.pairOverflow, size: 4},
        {buffer: departures.zoneOutCounts, size: zoneCount * 4},
        {buffer: arrivals.zoneInCounts, size: zoneCount * 4},
        {buffer: departures.originZones, size: k * 4},
        {buffer: departures.destinationZones, size: k * 4},
        {buffer: departures.flowWeights, size: k * 4}
      ],
      bytes => {
        if (!destroyed && graphs === built) processReadback(built, bytes);
      }
    );
    return built;
  }

  function rebuild(): void {
    if (graphs) retired.push({resources: graphs.resources, frames: 0});
    graphs = buildGraphs();
    stats = null;
    outlineCount = 0;
    drawnFlows = 0;
    writeLattice();
    markChanged();
  }

  // ---- Tables ------------------------------------------------------------------------------------
  const getTables = () => {
    const ground = ctx.ground();
    return {
      ground,
      net: getNetTable(ground),
      share: getShareTable(ground, ctx.options.minVolume),
      magnitude: getMagnitudeTable(ground)
    };
  };

  /** Legend data is re-published only when it changed (each call re-renders the legends). */
  const publishedLegendData = new Map<string, string>();
  function publishLegend(key: string, value: unknown): void {
    const serialized = JSON.stringify(value);
    if (publishedLegendData.get(key) === serialized) return;
    publishedLegendData.set(key, serialized);
    ctx.setLegendData(key, value);
  }

  function publishLegendData(): void {
    publishLegend('ground', ctx.ground());
    if (stats && !ctx.options.play) {
      publishLegend('counts', {net: stats.netClassCounts, share: stats.shareClassCounts});
    }
    publishLegend('flowMaximum', flowMaximum);
  }

  // ---- Readback -> net, share, outlines, arrows --------------------------------------------------
  function processReadback(current: TideGraphs, bytes: ArrayBuffer): void {
    const zoneCount = current.zoneCount;
    const k = TOP_FLOW_COUNT;
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const [flowRows, pairCount, departurePairOverflow, arrivalPairOverflow] = words;
    let offset = HEADER_WORDS;
    lastOutCounts = words.slice(offset, offset + zoneCount);
    offset += zoneCount;
    lastInCounts = words.slice(offset, offset + zoneCount);
    offset += zoneCount;
    lastFlowOrigins = words.slice(offset, offset + k);
    offset += k;
    lastFlowDestinations = words.slice(offset, offset + k);
    offset += k;
    lastFlowWeights = floats.slice(offset, offset + k);
    lastFlowRows = Math.min(flowRows, k);
    ctx.setReadout('flowRows', flowRows);
    ctx.setReadout('pairs', pairCount);
    ctx.setReadout(
      'pairOverflow',
      departurePairOverflow || arrivalPairOverflow ? 'yes: zone totals incomplete' : 'no'
    );
    applyCounts(current);
    updateFlows();
  }

  /** Assembles net, share and magnitude from the last zone totals and publishes everything. */
  function applyCounts(current: TideGraphs = graphs as TideGraphs): void {
    if (!current || lastOutCounts.length === 0) return;
    const {minVolume, normalise} = ctx.options;
    const zoneCount = current.zoneCount;
    const lattice = getCurrentLattice(current.kind);
    const tables = getTables();
    const values = new Float32Array(zoneCount + 1).fill(Number.NaN);
    const magnitudes = new Float32Array(zoneCount + 1).fill(Number.NaN);
    const netClassCounts = new Array<number>(tables.net.breaks.length + 1).fill(0);
    const shareClassCounts = new Array<number>(tables.share.breaks.length + 1).fill(0);
    const next: TideStats = {
      lattice,
      gainZone: -1,
      lossZone: -1,
      gain: 0,
      loss: 0,
      citySum: 0,
      maxNet: 0,
      activeZones: 0,
      balancedZones: 0,
      hiddenZones: 0,
      sharpestZone: -1,
      netClassCounts,
      shareClassCounts,
      lgaNet: 0,
      jfkNet: 0
    };
    let departureTrips = 0;
    let arrivalTrips = 0;
    let sharpestShare = 0;
    const lga = boxes.laguardia;
    const jfk = boxes.jfk;
    for (let zone = 0; zone < zoneCount; zone++) {
      const departures = lastOutCounts[zone];
      const arrivals = lastInCounts[zone];
      const volume = departures + arrivals;
      departureTrips += departures;
      arrivalTrips += arrivals;
      if (volume === 0) continue;
      const net = arrivals - departures;
      const share = net / volume;
      const masked = volume < minVolume;
      next.activeZones++;
      next.citySum += net;
      next.maxNet = Math.max(next.maxNet, Math.abs(net));
      if (Math.abs(net) <= BALANCED_TRIPS) next.balancedZones++;
      if (masked) next.hiddenZones++;
      if (net > next.gain) {
        next.gain = net;
        next.gainZone = zone;
      }
      if (net < next.loss) {
        next.loss = net;
        next.lossZone = zone;
      }
      if (!masked && Math.abs(share) > sharpestShare) {
        sharpestShare = Math.abs(share);
        next.sharpestZone = zone;
      }
      values[zone] = normalise === 'net' ? net : masked ? MASKED_SHARE : share;
      magnitudes[zone] = Math.abs(net);
      netClassCounts[getClassIndex(net, tables.net.breaks)]++;
      shareClassCounts[masked ? 0 : getClassIndex(share, tables.share.breaks)]++;
      const [x, y] = getZoneCenter(lattice, zone);
      if (isInside(lga, x, y)) next.lgaNet += net;
      if (isInside(jfk, x, y)) next.jfkNet += net;
    }
    stats = next;
    current.values.write(values);
    current.magnitudes.write(magnitudes);
    updateOutlines(current, lattice, values);

    ctx.setReadout('departures', departureTrips);
    ctx.setReadout('arrivals', arrivalTrips);
    ctx.setReadout('activeZones', `${formatCount(next.activeZones)} of ${formatCount(zoneCount)}`);
    ctx.setReadout(
      'gain',
      next.gainZone >= 0 ? describeZone(lattice, next.gainZone, next.gain) : null
    );
    ctx.setReadout(
      'loss',
      next.lossZone >= 0 ? describeZone(lattice, next.lossZone, next.loss) : null
    );
    ctx.setReadout('citySum', `${formatSigned(next.citySum)} trips`);
    ctx.setReadout('maxNet', `${formatCount(next.maxNet)} trips`);
    ctx.setReadout('nonBalanced', next.activeZones - next.balancedZones);
    ctx.setReadout(
      'balanced',
      `${formatCount(next.balancedZones)} of ${formatCount(next.activeZones)}`
    );
    ctx.setReadout('hidden', next.hiddenZones);
    ctx.setReadout(
      'sharpest',
      next.sharpestZone >= 0 ? describeSharpest(lattice, next.sharpestZone) : null
    );
    ctx.setReadout('lgaNet', `${formatSigned(next.lgaNet)} trips`);
    ctx.setReadout('jfkNet', `${formatSigned(next.jfkNet)} trips`);
    publishLegendData();
    updateNotes();
    updateRadiusRing();
    ctx.requestLayers();
  }

  function isInside(box: TideBox, x: number, y: number): boolean {
    return Math.abs(x - box.centerX) < box.halfWidth && Math.abs(y - box.centerY) < box.halfHeight;
  }

  function getZoneLngLat(lattice: Lattice, zone: number): LngLat {
    const [x, y] = getZoneCenter(lattice, zone);
    return trips.unproject(x, y);
  }

  function getZonePlace(lattice: Lattice, zone: number): string | null {
    return nearestPlaceLabel(NYC_PLACES, getZoneLngLat(lattice, zone), {maxDistanceMeters: 9000});
  }

  function describeZone(lattice: Lattice, zone: number, net: number): string {
    const place = getZonePlace(lattice, zone);
    return `${formatSigned(net)} trips${place ? `, ${place}` : ''}`;
  }

  function describeSharpest(lattice: Lattice, zone: number): string {
    const arrivals = lastInCounts[zone];
    const departures = lastOutCounts[zone];
    const volume = arrivals + departures;
    const place = getZonePlace(lattice, zone);
    return `${formatShare((arrivals - departures) / volume)} of ${formatCount(volume)} trips${place ? `, ${place}` : ''}`;
  }

  function updateOutlines(current: TideGraphs, lattice: Lattice, values: Float32Array): void {
    if (!ctx.options.showOutlines) {
      outlineCount = 0;
      return;
    }
    const segments = new Float32Array(current.zoneCount * 6 * 4);
    outlineCount = buildZoneOutlines(lattice, zone => !Number.isNaN(values[zone]), segments);
    current.outlines.write(segments.subarray(0, Math.max(4, outlineCount * 4)));
  }

  /** Top-K arrows from the last readback; the width scale only ever grows (ONE maximum). */
  function updateFlows(): void {
    const {flowCount} = ctx.options;
    if (!graphs || lastFlowRows === 0) {
      drawnFlows = 0;
      return;
    }
    for (let row = 0; row < lastFlowRows; row++) {
      flowMaximum = Math.max(flowMaximum, lastFlowWeights[row]);
    }
    if (flowCount === 0) {
      drawnFlows = 0;
      publishLegendData();
      return;
    }
    const lattice = getCurrentLattice(graphs.kind);
    const arrows = buildFlowArrows({
      originZones: lastFlowOrigins,
      destinationZones: lastFlowDestinations,
      weights: lastFlowWeights,
      count: lastFlowRows,
      limit: flowCount,
      skipSelf: true,
      getZoneCenter: zone => getZoneCenter(lattice, zone)
    });
    if (arrows.count > 0) {
      flowEndpointsBuffer.write(arrows.flows);
      flowWeightsBuffer.write(arrows.weights);
      flowOrderBuffer.write(arrows.order);
    }
    drawnFlows = arrows.count;
    publishLegendData();
    ctx.requestLayers();
  }

  // ---- Notes, ring, furniture -------------------------------------------------------------------
  function updateNotes(): void {
    const options = ctx.options;
    if (!stats || options.play || options.compareMidpoint) {
      ctx.setAnnotations('tide-notes', null);
      return;
    }
    const {lattice} = stats;
    const notes: MapAnnotation[] = [];
    if (options.normalise === 'net') {
      if (stats.gainZone >= 0) {
        notes.push({
          kind: 'note',
          id: 'tide-gain',
          coordinate: getZoneLngLat(lattice, stats.gainZone),
          title: `${formatSigned(stats.gain)} trips`,
          text: getZonePlace(lattice, stats.gainZone) ?? undefined,
          tone: 'accent',
          priority: 6
        });
      }
      if (stats.lossZone >= 0) {
        notes.push({
          kind: 'note',
          id: 'tide-loss',
          coordinate: getZoneLngLat(lattice, stats.lossZone),
          title: `${formatSigned(stats.loss)} trips`,
          text: getZonePlace(lattice, stats.lossZone) ?? undefined,
          tone: 'ink',
          priority: 5
        });
      }
    } else if (stats.sharpestZone >= 0) {
      const arrivals = lastInCounts[stats.sharpestZone];
      const departures = lastOutCounts[stats.sharpestZone];
      notes.push({
        kind: 'note',
        id: 'tide-sharpest',
        coordinate: getZoneLngLat(lattice, stats.sharpestZone),
        title: `${formatShare((arrivals - departures) / (arrivals + departures))} of traffic`,
        text: getZonePlace(lattice, stats.sharpestZone) ?? undefined,
        tone: 'accent',
        priority: 6
      });
    }
    ctx.setAnnotations('tide-notes', notes.length ? notes : null);
  }

  /** A dashed ring of one zone radius around the biggest gain, with the radius on it. */
  function updateRadiusRing(): void {
    if (!ctx.options.showRadius || !stats || stats.gainZone < 0) {
      ctx.setAnnotations('hex-radius', null);
      return;
    }
    const {zoneSize, zones} = ctx.options;
    const radius = zones === 'hexagon' ? zoneSize : zoneSize / 2;
    ctx.setAnnotations('hex-radius', [
      {
        kind: 'ring',
        id: 'hex-radius',
        coordinate: getZoneLngLat(stats.lattice, stats.gainZone),
        radiusMeters: radius,
        text: `radius ${formatDistance(radius)}`,
        dashed: true
      }
    ]);
  }

  function updateFurniture(): void {
    const {normalise, zoneSize, zones, compareMidpoint} = ctx.options;
    const size = formatDistance(zoneSize);
    const unit = zones === 'hexagon' ? 'hexagon' : 'cell';
    const when = formatLongWindow(
      roundToQuarter(windowStart),
      Math.min(roundToQuarter(windowEnd), NYC_TAXI_HOURS)
    );
    const subtitle = compareMidpoint
      ? `Size of the net, and net arrivals, per ${size} ${unit}, ${when}`
      : normalise === 'net'
        ? `Net taxi arrivals per ${size} ${unit}, ${when}`
        : `Imbalance share per ${size} ${unit}, ${when}`;
    const key = `${subtitle}|${zoneSize}`;
    if (key === lastFurnitureKey) return;
    lastFurnitureKey = key;
    ctx.setFurniture({
      title: {
        subtitle,
        sample: `${formatCount(count)} yellow-cab trips, a sample`,
        chips: ['Sample']
      },
      scaleBar: {units: 'metric', ticks: [zoneSize]}
    });
  }

  // ---- Charts -------------------------------------------------------------------------------------
  function publishCharts(): void {
    const tables = getTables();
    const hours = Array.from(hourlyNet.midtown, (_, index) => index + 0.5);
    ctx.setChart('dayChart', {
      kind: 'line',
      series: [
        {label: 'Midtown', x: hours, y: Array.from(hourlyNet.midtown), color: 0},
        {label: 'Upper East Side', x: hours, y: Array.from(hourlyNet.ues), color: 1}
      ],
      xLabel: 'Hours since midnight on Thu 1 Jan',
      yLabel: 'arrivals - departures (trips per hour)',
      height: 130,
      xDomain: [TIDES_FIRST_HOUR, Math.floor(NYC_TAXI_HOURS)],
      formatX: value =>
        `${value < 24 ? 'Thu' : 'Fri'} ${String(Math.floor(value) % 24).padStart(2, '0')}h`,
      guides: [{y: 0}],
      bands: [
        {from: 0, to: 6, label: 'night'},
        {from: 18, to: 30}
      ],
      link: {option: 'time', label: value => formatTaxiTime(value).split(' ').pop() ?? ''},
      description:
        'Net taxi arrivals per hour in a Midtown box and an Upper East Side box, with night shaded and the window start marked.'
    });
    const panelHours = [30, 32, 34, 37];
    const boxList = [boxes.midtown, boxes.ues, boxes.laguardia, boxes.jfk];
    const keys = ['midtown', 'ues', 'laguardia', 'jfk'] as const;
    ctx.setChart('panelChart', {
      kind: 'multiples',
      columns: 2,
      titles: panelHours.map(hour => formatShortWindow(hour, hour + 1)),
      description:
        'Net taxi arrivals in four boxes (Midtown, Upper East Side, LaGuardia, JFK) in four Friday hours: the sign flips through the morning.',
      charts: panelHours.map(hour => {
        const values = keys.map(key => hourlyNet[key][hour]);
        return {
          kind: 'bars' as const,
          values,
          labels: boxList.map(box => box.label),
          colors: values.map(value => tables.net.colors[getClassIndex(value, tables.net.breaks)]),
          height: 96,
          table: false
        };
      })
    });
  }

  function publishTimeBar(): void {
    const histogram = new Array<number>(TIME_BAR_BINS).fill(0);
    const span = TIDES_LAST_HOUR - TIDES_FIRST_HOUR;
    for (let row = 0; row < count; row++) {
      const bin = Math.floor(((trips.pickupHour[row] - TIDES_FIRST_HOUR) / span) * TIME_BAR_BINS);
      if (bin >= 0 && bin < TIME_BAR_BINS) histogram[bin]++;
    }
    ctx.setTimelineData({domain: [TIDES_FIRST_HOUR, TIDES_LAST_HOUR], histogram});
  }

  // ---- Tooltip -----------------------------------------------------------------------------------
  function getTooltip(coordinate: readonly [number, number]) {
    if (!graphs || !stats || lastOutCounts.length === 0) return null;
    const lattice = stats.lattice;
    const [x, y] = trips.project(coordinate[0], coordinate[1]);
    const zone = getZoneAt(lattice, x, y);
    if (zone < 0 || zone >= lastOutCounts.length) return null;
    const arrivals = lastInCounts[zone];
    const departures = lastOutCounts[zone];
    if (arrivals + departures === 0) return null;
    const tables = getTables();
    const content = getZoneTooltip({
      place: getZonePlace(lattice, zone),
      zoneNoun: lattice.kind === 'hexagon' ? 'Hexagon' : 'Cell',
      arrivals,
      departures,
      mode: ctx.options.normalise,
      netTable: tables.net,
      shareTable: tables.share,
      minVolume: ctx.options.minVolume,
      windowLabel: formatLongWindow(
        roundToQuarter(windowStart),
        Math.min(roundToQuarter(windowEnd), NYC_TAXI_HOURS)
      )
    });
    const ring = getZoneCorners(lattice, zone).map(([cornerX, cornerY]) =>
      trips.unproject(cornerX, cornerY)
    );
    return {...content, highlight: {kind: 'polygon' as const, rings: [[...ring, ring[0]]]}};
  }

  ctx.setReadout('rows', count);
  ctx.setReadout('medianTrip', `${medianTripMinutes.toFixed(0)} min`);
  ctx.setCost({records: count, passes: 2});
  rebuild();
  publishLegendData();
  publishCharts();
  updateFurniture();

  return {
    getCompiledGraphs: () =>
      graphs
        ? ([
            graphs.departures.compiled,
            graphs.arrivals.compiled
          ] as CompiledGPUCommandGraph<never>[])
        : [],

    setOption(id) {
      switch (id) {
        case 'zones':
        case 'excludeSelf':
        case 'sumOrder':
          flowMaximum = 0;
          rebuild();
          updateFurniture();
          ctx.requestLayers();
          break;
        case 'zoneSize':
          flowMaximum = 0;
          clearZoneValues();
          writeLattice();
          markChanged();
          updateFurniture();
          ctx.requestLayers();
          break;
        case 'normalise':
        case 'minVolume':
          applyCounts();
          updateFurniture();
          ctx.requestLayers();
          break;
        case 'windowHours':
          lastWindowStart = Number.NaN;
          markChanged();
          break;
        case 'flowCount':
          updateFlows();
          ctx.requestLayers();
          break;
        case 'compareMidpoint':
          updateNotes();
          updateFurniture();
          ctx.requestLayers();
          break;
        case 'showRadius':
          updateRadiusRing();
          break;
        case 'play':
          updateNotes();
          if (!ctx.options.play) markChanged();
          break;
        case 'showOutlines':
          applyCounts();
          break;
        case 'time':
        case 'playSpeed':
        case 'loop':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      publishCharts();
      ctx.requestLayers();
    },

    // The class tables are authored per ground, so a ground flip rebuilds them.
    onGroundChange() {
      publishLegendData();
      publishCharts();
      applyCounts();
      ctx.requestLayers();
    },

    onLegendFilter(_id, classes) {
      legendHighlight = classes === null ? null : [...classes];
      ctx.requestLayers();
    },

    getTooltip(event) {
      return event.coordinate ? getTooltip(event.coordinate) : null;
    },

    encode(commandEncoder, frame) {
      if (!graphs) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      const playhead = clock.advance(frame);
      const start = playhead;
      const end = playhead + ctx.options.windowHours;
      if (start !== lastWindowStart || end !== lastWindowEnd) {
        lastWindowStart = start;
        lastWindowEnd = end;
        windowStart = start;
        windowEnd = end;
        windowBuffer.write(getGPUTimeWindowParameterValues({start, end}));
        markChanged();
        ctx.setReadout('clock', formatTaxiTime(playhead));
        ctx.setReadout(
          'window',
          formatShortWindow(roundToQuarter(start), Math.min(roundToQuarter(end), NYC_TAXI_HOURS))
        );
        updateFurniture();
      }
      if (encodeFrames > 0) {
        graphs.departures.compiled.encode(commandEncoder, {parameters: undefined});
        graphs.arrivals.compiled.encode(commandEncoder, {parameters: undefined});
        encodeFrames--;
      }
      const playing = ctx.options.play;
      if (statsStale && encodeFrames === 0 && !graphs.reader.isPending && !playing) {
        statsStale = false;
        graphs.reader.request(commandEncoder);
      } else if (
        playing &&
        frame.frameIndex % PLAYBACK_READ_FRAMES === 0 &&
        !graphs.reader.isPending
      ) {
        graphs.reader.request(commandEncoder);
      } else {
        graphs.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!graphs) return [];
      const options = ctx.options;
      const tables = getTables();
      const {ground} = tables;
      const lattice = getCurrentLattice(graphs.kind);
      const share = options.normalise === 'share';
      const table = share ? tables.share : tables.net;
      const compare = options.compareMidpoint && !share;
      const zoneProps = {
        coordinateOrigin,
        gridSize: lattice.grid,
        bounds: lattice.bounds,
        binning: graphs.kind,
        hexagonRadius: lattice.size,
        valueFormat: 'float32' as const,
        colormap: 'grayscale' as const,
        noDataColor: [0, 0, 0, 0] as [number, number, number, number],
        opacity: FILL_OPACITY[ground],
        hatchColor: getHatchColor(ground),
        hatchSpacingPixels: 5
      };
      const layers: Layer[] = [];
      if (compare) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...zoneProps,
            id: `taxi-tides-magnitude-${graphs.kind}`,
            values: graphs.magnitudes,
            ...getClassTableLayerProps(tables.magnitude),
            compareSide: 'a'
          })
        );
      }
      layers.push(
        new SpatialAnalysisRasterLayer({
          ...zoneProps,
          id: `taxi-tides-zones-${graphs.kind}`,
          values: graphs.values,
          ...getClassTableLayerProps(table),
          highlightClasses: legendHighlight,
          ...(compare ? {compareSide: 'b' as const} : {})
        })
      );
      if (options.showOutlines && outlineCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-tides-outlines',
            coordinateOrigin,
            segments: graphs.outlines,
            instanceCount: outlineCount,
            widthPixels: 0.5,
            cap: 'butt',
            color: getZoneOutlineColor(ground)
          })
        );
      }
      if (options.flowCount > 0 && drawnFlows > 0) {
        layers.push(
          new SpatialAnalysisFlowLayer({
            id: 'taxi-tides-flows',
            coordinateOrigin,
            flows: flowEndpointsBuffer,
            values: flowWeightsBuffer,
            valueFormat: 'float32',
            ids: flowOrderBuffer,
            instanceCount: drawnFlows,
            maxValue: Math.max(flowMaximum, 1),
            minWidthPixels: FLOW_WIDTH_PIXELS[0],
            maxWidthPixels: FLOW_WIDTH_PIXELS[1],
            curvature: 0.15,
            arrowheads: true,
            color: FLOW_INK[ground],
            outlineColor: FLOW_HALO[ground],
            outlineWidthPixels: 0.8
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      graphs?.reader.stop();
      graphs?.resources.destroy();
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
    }
  };
}

/** Window labels move in quarter hours, the step of the time slider. */
function roundToQuarter(hours: number): number {
  return Math.round(hours * 4) / 4;
}

// ---------------------------------------------------------------------------------------------
// CPU context: named boxes, their hourly net, the median trip
// ---------------------------------------------------------------------------------------------

type BoxId = 'midtown' | 'ues' | 'laguardia' | 'jfk';

/**
 * The four boxes the story talks about. Midtown and the airports come from the dataset's own
 * presets; the Upper East Side box sits just east of Central Park, offset from the park's
 * gazetteer point (the gazetteer has no neighbourhood centroid for it).
 */
function createBoxes(trips: TaxiTrips): Record<BoxId, TideBox> {
  const fromPreset = (label: string, preset: (typeof AREA_PRESETS)[keyof typeof AREA_PRESETS]) => {
    const [centerX, centerY] = trips.project(preset.center[0], preset.center[1]);
    return {
      label,
      centerX,
      centerY,
      halfWidth: preset.halfWidth,
      halfHeight: preset.halfHeight
    };
  };
  const [parkX, parkY] = trips.project(...NYC.places['central-park'].lngLat);
  return {
    midtown: fromPreset('Midtown', AREA_PRESETS.midtown),
    ues: {
      label: 'Upper East Side',
      centerX: parkX + 900,
      centerY: parkY - 950,
      halfWidth: 1000,
      halfHeight: 1800
    },
    laguardia: fromPreset('LaGuardia', AREA_PRESETS.laguardia),
    jfk: fromPreset('JFK', AREA_PRESETS.jfk)
  };
}

/**
 * Hourly arrivals minus departures of each box over the trips that cross its edge. The first and
 * last hours miss trips that began outside the data, so they are NaN and not drawn.
 */
function getHourlyBoxNet(
  trips: TaxiTrips,
  boxes: Record<BoxId, TideBox>
): Record<BoxId, Float64Array> {
  const hours = Math.ceil(NYC_TAXI_HOURS) + 1;
  const result = {} as Record<BoxId, Float64Array>;
  const ids = Object.keys(boxes) as BoxId[];
  for (const id of ids) result[id] = new Float64Array(hours);
  const inside = (box: TideBox, array: Float32Array, row: number) =>
    Math.abs(array[row * 2] - box.centerX) < box.halfWidth &&
    Math.abs(array[row * 2 + 1] - box.centerY) < box.halfHeight;
  for (let row = 0; row < trips.count; row++) {
    for (const id of ids) {
      const box = boxes[id];
      const fromInside = inside(box, trips.pickup, row);
      const toInside = inside(box, trips.dropoff, row);
      if (fromInside === toInside) continue;
      if (fromInside) result[id][Math.min(hours - 1, Math.floor(trips.pickupHour[row]))]--;
      else result[id][Math.min(hours - 1, Math.floor(trips.dropoffHour[row]))]++;
    }
  }
  for (const id of ids) {
    result[id][0] = Number.NaN;
    for (let hour = Math.floor(NYC_TAXI_HOURS); hour < hours; hour++) result[id][hour] = Number.NaN;
  }
  return result;
}

/** Median routed trip time in minutes (durations are 15 s steps, so a count per step is exact). */
function getMedianTripMinutes(trips: TaxiTrips): number {
  const steps = new Uint32Array(256);
  for (let row = 0; row < trips.count; row++) {
    const step = Math.round(((trips.dropoffHour[row] - trips.pickupHour[row]) * 3600) / 15);
    steps[Math.min(255, Math.max(0, step))]++;
  }
  let seen = 0;
  for (let step = 0; step < steps.length; step++) {
    seen += steps[step];
    if (seen >= trips.count / 2) return (step * 15) / 60;
  }
  return 0;
}
