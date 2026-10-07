// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassIndexOf, getClassTableLayerProps} from '../../cartography/class-table';
import {formatCount, formatSigned, liveText} from '../../cartography/live-text';
import {FLOW_HALO, FLOW_INK, buildFlowArrows, formatClockHour} from './flows-style';
import type {LngLat, MapAnnotation} from '../../cartography/types';
import {SpatialAnalysisFlowLayer} from '../../engine/flow-layer';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent} from '../scene';
import {
  buildStationProfiles,
  findStationNearPixel,
  getMonthDayCounts,
  MONTREAL_ORIGIN,
  readBixiFlows,
  shortStationName,
  SLOT_COUNT
} from './bixi-data';
import {
  getNetTable,
  getTidesScale,
  TIDES_DISC_MAX_PIXELS,
  TIDES_DISC_MIN_PIXELS,
  type TidesScale,
  type TidesScaleMode,
  type TidesUnits
} from './bixi-tides-style';

/** Option state of the bixi-tides scene. */
export type BixiTidesOptions = {
  /** Start hour of the one-hour window, local Montreal time. */
  hour: number;
  play: boolean;
  speed: number;
  loop: boolean;
  dayType: 'weekday' | 'weekend';
  /** `rate`: rides per average day-type hour. `total`: the raw August sum. */
  units: TidesUnits;
  /** `fixed`: one scale for every hour. `perHour`: rescale to each hour's maximum (the failing state). */
  scaleMode: TidesScaleMode;
  /** Start hour of the second window drawn by the compare step. */
  compareHour: number;
  flowCount: number;
  /** Width law of the flow arrows: square root of the flow (area-true), or linear. */
  flowWidth: 'sqrt' | 'linear';
  /** Station outlined and charted: none, the biggest morning drain or fill, or the clicked one. */
  selection: 'none' | 'top-drain' | 'top-fill' | 'clicked';
  excludeSelf: boolean;
  sumOrder: 'sorted' | 'atomic';
};

/** Rows of the top-K flow list (compile-time capacity). */
export const TIDES_TOP_FLOW_COUNT = 512;
/** First and last start hour of the playback (the slider range). */
export const TIDES_FIRST_HOUR = 5;
export const TIDES_LAST_HOUR = 23;
/** Longest flow arrow, in CSS pixels (the shortest is {@link FLOW_MIN_PIXELS}). */
export const TIDES_FLOW_MAX_PIXELS = 6;
const FLOW_MIN_PIXELS = 1;
const HEADER_WORDS = 4;
const RETIRE_FRAMES = 4;
const ENCODE_FRAMES = 3;
/** Morning slot used to pick the station the last step preselects. */
const MORNING_HOUR = 8;

type TidesGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  sumOrder: BixiTidesOptions['sumOrder'];
  windowBuffer: GPUParameterBuffer<'float32'>;
  reader: SummaryReader;
};

/** The totals a one-hour window read back from the GPU, for one day type. */
type WindowStats = {
  hour: number;
  dayType: 'weekday' | 'weekend';
  flowRows: number;
  pairCount: number;
  pairOverflow: boolean;
  topOrigin: Uint32Array;
  topDestination: Uint32Array;
  topWeight: Float32Array;
  out: Float32Array;
  incoming: Float32Array;
};

/** One window on screen: its graph, its read-back stats and the buffers its layers draw. */
type TidesView = {
  slot: 'a' | 'b';
  graph: TidesGraph | null;
  stats: WindowStats | null;
  /** Window the graph last encoded (`-1`: the parameter buffer must be rewritten). */
  hour: number;
  requestKey: {hour: number; dayType: 'weekday' | 'weekend'};
  frames: number;
  stale: boolean;
  values: Buffer;
  sizes: Buffer;
  order: Buffer;
  flowEndpoints: Buffer;
  flowWeights: Buffer;
  flowOrder: Buffer;
  drawnFlows: number;
  valueArray: Float32Array;
};

/**
 * Net flow of BIXI stations through the day. A compiled `GPUFlowAggregation` takes the 685,000
 * (origin, destination, weekday or weekend, hour, rides) rows, gates them with a one-hour window,
 * sums the pairs in a GPU hash table and writes a top-512 flow list plus per-station departure and
 * arrival totals. The scene turns the two totals into net bikes per average day-type hour on the
 * CPU (a few KB) and writes one value, one size and one draw-order buffer the discs read.
 *
 * Two copies of the graph (the live window and the compare window) share the row buffers and the
 * day-type mask; only the window, the day type and the display units are buffer writes.
 */
export async function createBixiTides(
  ctx: SceneContext<BixiTidesOptions>
): Promise<SceneInstance<BixiTidesOptions>> {
  const {device} = ctx;
  const flows = readBixiFlows(ctx.datasets.get('bixi-flows'));
  const {zoneCount, stationCount, slices} = flows;
  const rows = slices.rowCount;
  const dayCounts = getMonthDayCounts(flows);
  const resources = new SpatialAnalysisResources(device, 'bixi-tides');
  const coordinateOrigin: [number, number, number] = [MONTREAL_ORIGIN[0], MONTREAL_ORIGIN[1], 0];
  const profiles = buildStationProfiles(flows);

  // ---- CPU context computed once from the same rows the GPU reads ------------------------------
  const daysOf = (dayType: 'weekday' | 'weekend') =>
    dayType === 'weekday' ? dayCounts.weekday : dayCounts.weekend;
  const slotOf = (dayType: 'weekday' | 'weekend', hour: number) =>
    (dayType === 'weekday' ? 0 : 24) + hour;
  const profileNet = (station: number, dayType: 'weekday' | 'weekend', hour: number) => {
    const index = station * SLOT_COUNT + slotOf(dayType, hour);
    return profiles.incoming[index] - profiles.out[index];
  };

  // One maximum for each flow unit, over every hour and day type, so widths compare across views.
  let flowMaximumRate = 1;
  let flowMaximumTotal = 1;
  for (let row = 0; row < rows; row++) {
    const count = slices.count[row];
    flowMaximumTotal = Math.max(flowMaximumTotal, count);
    flowMaximumRate = Math.max(
      flowMaximumRate,
      count / (slices.dayType[row] === 0 ? dayCounts.weekday : dayCounts.weekend)
    );
  }

  // Rides leaving per average hour and the totals of the two day types, from the slice rows.
  const cityWeekday = new Float64Array(24);
  const cityWeekend = new Float64Array(24);
  let weekdayRides = 0;
  let weekendRides = 0;
  for (let row = 0; row < rows; row++) {
    if (slices.origin[row] >= stationCount) continue;
    if (slices.dayType[row] === 0) {
      cityWeekday[slices.hour[row]] += slices.count[row];
      weekdayRides += slices.count[row];
    } else {
      cityWeekend[slices.hour[row]] += slices.count[row];
      weekendRides += slices.count[row];
    }
  }
  const cityWeekdayRate = Array.from(cityWeekday, value => value / dayCounts.weekday);
  const cityWeekendRate = Array.from(cityWeekend, value => value / dayCounts.weekend);

  let topDrainStation = 0;
  let topFillStation = 0;
  for (let station = 1; station < stationCount; station++) {
    if (
      profileNet(station, 'weekday', MORNING_HOUR) <
      profileNet(topDrainStation, 'weekday', MORNING_HOUR)
    ) {
      topDrainStation = station;
    }
    if (
      profileNet(station, 'weekday', MORNING_HOUR) >
      profileNet(topFillStation, 'weekday', MORNING_HOUR)
    ) {
      topFillStation = station;
    }
  }

  // ---- GPU inputs shared by both windows -------------------------------------------------------
  const originIds = resources.createBuffer('origin', slices.origin);
  const destinationIds = resources.createBuffer('destination', slices.destination);
  const hours = resources.createBuffer('hours', slices.hour);
  const weights = resources.createBuffer('weights', slices.count);
  const mask = resources.createBuffer('mask', new Uint32Array(rows).fill(1));
  const centers = resources.createBuffer('centers', flows.centers);

  const makeView = (slot: 'a' | 'b'): TidesView => {
    const nanValues = new Float32Array(zoneCount).fill(Number.NaN);
    return {
      slot,
      graph: null,
      stats: null,
      hour: -1,
      requestKey: {hour: 0, dayType: 'weekday'},
      frames: ENCODE_FRAMES,
      stale: true,
      values: resources.createBuffer(`values-${slot}`, nanValues),
      sizes: resources.createBuffer(`sizes-${slot}`, zoneCount * 4),
      order: resources.createBuffer(`order-${slot}`, stationCount * 4),
      flowEndpoints: resources.createBuffer(`flow-endpoints-${slot}`, TIDES_TOP_FLOW_COUNT * 16),
      flowWeights: resources.createBuffer(`flow-values-${slot}`, TIDES_TOP_FLOW_COUNT * 4),
      flowOrder: resources.createBuffer(`flow-order-${slot}`, TIDES_TOP_FLOW_COUNT * 4),
      drawnFlows: 0,
      valueArray: new Float32Array(zoneCount)
    };
  };
  const viewA = makeView('a');
  const viewB = makeView('b');
  const views = [viewA, viewB] as const;

  let serial = 0;
  let destroyed = false;
  let comparing = false;
  let clickedStation = -1;
  let legendHighlight: number[] | null = null;
  let lastFurnitureKey = '';
  let lastTable = getNetTable(
    getTidesScale({
      units: 'rate',
      scaleMode: 'fixed',
      dayType: 'weekday',
      hourMaximum: 1,
      weekdayCount: dayCounts.weekday
    }),
    ctx.ground()
  );
  let lastScale: TidesScale = getTidesScale({
    units: 'rate',
    scaleMode: 'fixed',
    dayType: 'weekday',
    hourMaximum: 1,
    weekdayCount: dayCounts.weekday
  });
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];
  const clock = createPlaybackClock(
    ctx,
    {time: 'hour', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [TIDES_FIRST_HOUR, TIDES_LAST_HOUR], rate: 1, step: 1, notify: false}
  );

  function getHour(): number {
    const {hour, play} = ctx.options;
    const start = play ? clock.time : hour;
    return Math.min(23, Math.max(0, Math.floor(start + 1e-6)));
  }

  function getCompareHour(): number {
    return Math.min(23, Math.max(0, Math.round(ctx.options.compareHour)));
  }

  function markChanged(): void {
    for (const view of views) {
      view.frames = Math.max(view.frames, ENCODE_FRAMES);
      view.stale = true;
    }
  }

  function writeMask(): void {
    const weekend = ctx.options.dayType === 'weekend';
    const values = new Uint32Array(rows);
    for (let row = 0; row < rows; row++) {
      values[row] = (slices.dayType[row] === 1) === weekend ? 1 : 0;
    }
    mask.write(values);
    markChanged();
  }

  // ---- The graph ---------------------------------------------------------------------------------
  function buildGraph(view: TidesView): TidesGraph {
    const id = ++serial;
    const graphResources = new SpatialAnalysisResources(device, `bixi-tides-${view.slot}-${id}`);
    const k = TIDES_TOP_FLOW_COUNT;
    const ids = graphResources.createBuffer('ids', k * 4);
    const count = graphResources.createBuffer('count', 4);
    const totalCount = graphResources.createBuffer('total-count', 4);
    const overflow = graphResources.createBuffer('overflow', 4);
    const pairOverflow = graphResources.createBuffer('pair-overflow', 4);
    const originZones = graphResources.createBuffer('flow-origin', k * 4);
    const destinationZones = graphResources.createBuffer('flow-destination', k * 4);
    const flowWeights = graphResources.createBuffer('flow-weights', k * 4);
    const zoneOutWeights = graphResources.createBuffer('zone-out-weights', zoneCount * 4);
    const zoneInWeights = graphResources.createBuffer('zone-in-weights', zoneCount * 4);
    const zoneOutCounts = graphResources.createBuffer('zone-out-counts', zoneCount * 4);
    const windowBuffer = graphResources.createParameterBuffer(
      'window',
      'float32',
      GPU_TIME_WINDOW_PARAMETER_LENGTH
    );
    const commandGraph = new GPUCommandGraph<void>(device, {
      id: `bixi-tides-${view.slot}-${id}`
    });
    commandGraph.add(
      new GPUFlowAggregation({
        id: `tides-${view.slot}`,
        zones: {kind: 'ids', zoneCount},
        originZoneIds: importGraphBuffer(commandGraph, 'origin-ids', originIds, 'uint32', rows),
        destinationZoneIds: importGraphBuffer(
          commandGraph,
          'destination-ids',
          destinationIds,
          'uint32',
          rows
        ),
        weights: importGraphBuffer(commandGraph, 'weights', weights, 'float32', rows),
        mask: importGraphBuffer(commandGraph, 'mask', mask, 'uint32', rows),
        timeWindow: {
          timestamps: importGraphBuffer(commandGraph, 'hours', hours, 'float32', rows),
          window: windowBuffer.importToGraph(commandGraph)
        },
        excludeSelfFlows: ctx.options.excludeSelf,
        sumOrder: ctx.options.sumOrder,
        pairCapacity: 131072,
        // rows * maxProbeCount must fit in a uint32; 128 probes is far above the expected chain length.
        maxProbeCount: 128,
        output: {
          ids: importGraphBuffer(commandGraph, 'ids', ids, 'uint32', k),
          count: importGraphBuffer(commandGraph, 'count', count, 'uint32', 1),
          overflow: importGraphBuffer(commandGraph, 'overflow', overflow, 'uint32', 1),
          totalCount: importGraphBuffer(commandGraph, 'total-count', totalCount, 'uint32', 1)
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
    const built: TidesGraph = {
      resources: graphResources,
      compiled,
      sumOrder: ctx.options.sumOrder,
      windowBuffer,
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      graphResources,
      `bixi-tides-${view.slot}-${id}`,
      [
        {buffer: count, size: 4},
        {buffer: totalCount, size: 4},
        {buffer: overflow, size: 4},
        {buffer: pairOverflow, size: 4},
        {buffer: originZones, size: k * 4},
        {buffer: destinationZones, size: k * 4},
        {buffer: flowWeights, size: k * 4},
        {buffer: zoneOutWeights, size: zoneCount * 4},
        {buffer: zoneInWeights, size: zoneCount * 4}
      ],
      bytes => {
        if (!destroyed && view.graph === built) processStatistics(view, bytes);
      }
    );
    return built;
  }

  function rebuild(): void {
    for (const view of views) {
      if (view.graph) retired.push({resources: view.graph.resources, frames: 0});
      view.graph = buildGraph(view);
      view.hour = -1;
    }
    markChanged();
  }

  // ---- Reading a window back ---------------------------------------------------------------------
  function processStatistics(view: TidesView, bytes: ArrayBuffer): void {
    const k = TIDES_TOP_FLOW_COUNT;
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const outStart = HEADER_WORDS + 3 * k;
    view.stats = {
      hour: view.requestKey.hour,
      dayType: view.requestKey.dayType,
      flowRows: words[0],
      pairCount: words[1],
      pairOverflow: words[3] !== 0,
      topOrigin: words.slice(HEADER_WORDS, HEADER_WORDS + k),
      topDestination: words.slice(HEADER_WORDS + k, HEADER_WORDS + 2 * k),
      topWeight: floats.slice(HEADER_WORDS + 2 * k, HEADER_WORDS + 3 * k),
      out: floats.slice(outStart, outStart + zoneCount),
      incoming: floats.slice(outStart + zoneCount, outStart + 2 * zoneCount)
    };
    applyDisplay();
    if (view.slot === 'a') publishWindowReadouts();
  }

  /** Net of one station in the shown unit: a rate per average hour, or the August total. */
  function getShownNet(stats: WindowStats, station: number): number {
    const net = stats.incoming[station] - stats.out[station];
    return ctx.options.units === 'total' ? net : net / daysOf(stats.dayType);
  }

  /** Rewrites every value, size, order and flow buffer from the stats and the display options. */
  function applyDisplay(): void {
    const statsA = viewA.stats;
    if (!statsA) return;
    const options = ctx.options;
    let hourMaximum = 0;
    for (let station = 0; station < stationCount; station++) {
      hourMaximum = Math.max(hourMaximum, Math.abs(getShownNet(statsA, station)));
    }
    lastScale = getTidesScale({
      units: options.units,
      scaleMode: options.scaleMode,
      dayType: statsA.dayType,
      hourMaximum,
      weekdayCount: dayCounts.weekday
    });
    lastTable = getNetTable(lastScale, ctx.ground());
    for (const view of views) {
      if (view.slot === 'b' && !comparing) continue;
      if (view.stats) writeView(view, view.stats);
    }
    publishLegend();
    publishNotes();
    publishCompare();
    ctx.requestLayers();
  }

  function writeView(view: TidesView, stats: WindowStats): void {
    const values = view.valueArray;
    values.fill(Number.NaN);
    const sizes = new Float32Array(zoneCount);
    const order: number[] = [];
    for (let station = 0; station < stationCount; station++) {
      const value = getShownNet(stats, station);
      values[station] = value;
      sizes[station] = Math.abs(value);
      order.push(station);
    }
    // Largest discs first, so the small ones stay visible on top.
    order.sort((a, b) => sizes[b] - sizes[a] || a - b);
    view.values.write(values);
    view.sizes.write(sizes);
    view.order.write(Uint32Array.from(order));
    writeFlows(view, stats);
  }

  function writeFlows(view: TidesView, stats: WindowStats): void {
    const {flowCount, flowWidth, units} = ctx.options;
    view.drawnFlows = 0;
    if (flowCount <= 0 || stats.flowRows === 0) return;
    const days = daysOf(stats.dayType);
    const arrows = buildFlowArrows({
      originZones: stats.topOrigin,
      destinationZones: stats.topDestination,
      weights: stats.topWeight,
      count: stats.flowRows,
      limit: flowCount,
      skipSelf: true,
      getZoneCenter: zone =>
        zone < stationCount ? [flows.centers[zone * 2], flows.centers[zone * 2 + 1]] : null
    });
    if (arrows.count === 0) return;
    const maximum = units === 'total' ? flowMaximumTotal : flowMaximumRate;
    const shown = new Float32Array(arrows.count);
    for (let flow = 0; flow < arrows.count; flow++) {
      const value = units === 'total' ? arrows.weights[flow] : arrows.weights[flow] / days;
      // The layer's width is 0.5 + k sqrt(value / maximum): feeding value^2 / maximum makes it linear.
      shown[flow] = flowWidth === 'linear' ? (value * value) / maximum : value;
    }
    view.flowEndpoints.write(arrows.flows);
    view.flowWeights.write(shown);
    view.flowOrder.write(arrows.order);
    view.drawnFlows = arrows.count;
  }

  // ---- Readouts, legend, notes, charts -----------------------------------------------------------
  function getUnitWord(): string {
    return ctx.options.units === 'total' ? 'rides in August' : 'rides per hour';
  }

  function publishWindowReadouts(): void {
    const stats = viewA.stats;
    const graph = viewA.graph;
    if (!stats) return;
    const days = daysOf(stats.dayType);
    const dayWord = stats.dayType === 'weekday' ? 'weekdays' : 'weekends';
    ctx.setReadout(
      'window',
      `${formatClockHour(stats.hour)} to ${formatClockHour(stats.hour + 1)}, ${dayWord}`
    );
    let departures = 0;
    let drain = 0;
    let fill = 0;
    let drainStation = 0;
    let fillStation = 0;
    let mismatches = 0;
    for (let station = 0; station < stationCount; station++) {
      departures += stats.out[station];
      const net = getShownNet(stats, station);
      if (net < drain) {
        drain = net;
        drainStation = station;
      }
      if (net > fill) {
        fill = net;
        fillStation = station;
      }
      // While playing, a read can lag the window by a frame: the check runs when paused.
      if (!ctx.options.play) {
        const expected = profileNet(station, stats.dayType, stats.hour);
        if (Math.abs(stats.incoming[station] - stats.out[station] - expected) > 0.5) mismatches++;
      }
    }
    const word = getUnitWord();
    ctx.setReadout(
      'windowRides',
      `${formatCount(departures / (ctx.options.units === 'total' ? 1 : days))} ${ctx.options.units === 'total' ? 'rides in August' : 'rides per hour'}`
    );
    ctx.setReadout(
      'topDrain',
      drain < 0
        ? `${shortStationName(flows.names[drainStation], 30)}: ${formatSigned(drain, 1)} ${word}`
        : null
    );
    ctx.setReadout(
      'topFill',
      fill > 0
        ? `${shortStationName(flows.names[fillStation], 30)}: ${formatSigned(fill, 1)} ${word}`
        : null
    );
    if (stats.flowRows > 0) {
      const weight = stats.topWeight[0] / (ctx.options.units === 'total' ? 1 : days);
      ctx.setReadout(
        'busiestFlow',
        `${shortStationName(zoneName(stats.topOrigin[0]), 22)} to ${shortStationName(zoneName(stats.topDestination[0]), 22)}: ${formatCount(weight)} ${word}`
      );
    } else {
      ctx.setReadout('busiestFlow', null);
    }
    ctx.setReadout('rows', rows);
    ctx.setReadout('pairs', stats.pairCount);
    ctx.setReadout('pairOverflow', stats.pairOverflow ? 'yes: totals incomplete' : 'no');
    ctx.setReadout(
      'cpuMatch',
      ctx.options.play
        ? 'checked when paused'
        : mismatches === 0
          ? `yes, all ${formatCount(stationCount)} stations`
          : `no: ${formatCount(mismatches)} stations differ`
    );
    if (graph) {
      ctx.setCost({
        records: rows,
        passes: graph.compiled.stats.nodeOrder.length * (comparing ? 2 : 1)
      });
    }
    publishCharts();
  }

  function zoneName(zone: number): string {
    return zone < stationCount ? flows.names[zone] : 'other stations';
  }

  /** Share of out-of-balance stations whose sign flips between the two windows. */
  function publishCompare(): void {
    const a = viewA.stats;
    const b = viewB.stats;
    if (!comparing || !a || !b) {
      ctx.setReadout('reversal', null);
      return;
    }
    let both = 0;
    let flipped = 0;
    for (let station = 0; station < stationCount; station++) {
      const netA = (a.incoming[station] - a.out[station]) / daysOf(a.dayType);
      const netB = (b.incoming[station] - b.out[station]) / daysOf(b.dayType);
      if (Math.abs(netA) >= 1 && Math.abs(netB) >= 1) {
        both++;
        if (netA * netB < 0) flipped++;
      }
    }
    ctx.setReadout('reversal', both > 0 ? flipped / both : null);
  }

  function publishLegend(): void {
    const stats = viewA.stats;
    const counts = new Array<number>(lastTable.colors.length).fill(0);
    if (stats) {
      for (let station = 0; station < stationCount; station++) {
        counts[getClassIndexOf(lastTable, getShownNet(stats, station))]++;
      }
    }
    ctx.setLegendData('tides', {
      table: lastTable,
      scale: lastScale,
      counts,
      flowMaximum: ctx.options.units === 'total' ? flowMaximumTotal : flowMaximumRate,
      ground: ctx.ground()
    });
  }

  /** Finding notes on the biggest drain and fill, anchored to their stations. */
  function publishNotes(): void {
    const stats = viewA.stats;
    const options = ctx.options;
    if (
      !stats ||
      options.play ||
      comparing ||
      options.units !== 'rate' ||
      options.selection !== 'none'
    ) {
      ctx.setAnnotations('tides-notes', null);
      return;
    }
    let drainStation = -1;
    let fillStation = -1;
    let drain = 0;
    let fill = 0;
    for (let station = 0; station < stationCount; station++) {
      const net = getShownNet(stats, station);
      if (net < drain) {
        drain = net;
        drainStation = station;
      }
      if (net > fill) {
        fill = net;
        fillStation = station;
      }
    }
    const notes: MapAnnotation[] = [];
    const lngLat = (station: number): LngLat => [
      flows.lngLat[station * 2],
      flows.lngLat[station * 2 + 1]
    ];
    if (drainStation >= 0) {
      notes.push({
        kind: 'note',
        id: 'tides-drain',
        coordinate: lngLat(drainStation),
        title: liveText('{net:signed:1} rides per hour', {net: drain}),
        text: `${shortStationName(flows.names[drainStation], 24)} drains`,
        tone: 'ink',
        priority: 6
      });
    }
    if (fillStation >= 0) {
      notes.push({
        kind: 'note',
        id: 'tides-fill',
        coordinate: lngLat(fillStation),
        title: liveText('{net:signed:1} rides per hour', {net: fill}),
        text: `${shortStationName(flows.names[fillStation], 24)} fills`,
        tone: 'accent',
        priority: 6
      });
    }
    ctx.setAnnotations('tides-notes', notes);
  }

  function publishCharts(): void {
    const hourStarts = Array.from({length: 24}, (_, hour) => hour);
    const hour = getHour();
    ctx.setChart('cityChart', {
      kind: 'line',
      series: [
        {label: 'weekdays', x: hourStarts, y: cityWeekdayRate, color: 0},
        {label: 'weekends', x: hourStarts, y: cityWeekendRate, color: 3}
      ],
      xDomain: [0, 23],
      xLabel: 'start hour (Montreal time)',
      yLabel: 'rides leaving per average hour',
      height: 130,
      formatX: value => `${Math.round(value)}`,
      formatY: value => formatCount(value),
      link: {option: 'hour', label: value => formatClockHour(value)},
      description: `Rides leaving all stations in each hour of an average weekday and an average weekend day; the marker is the window start (${formatClockHour(hour)}).`
    });
    publishStationChart();
  }

  function getSelectedStation(
    selection: BixiTidesOptions['selection'] = ctx.options.selection
  ): number {
    switch (selection) {
      case 'top-drain':
        return topDrainStation;
      case 'top-fill':
        return topFillStation;
      case 'clicked':
        return clickedStation;
      default:
        return -1;
    }
  }

  function publishStationChart(selection?: BixiTidesOptions['selection']): void {
    const station = getSelectedStation(selection);
    if (station < 0) {
      ctx.setChart('stationChart', null);
      ctx.setReadout('station', 'click a station');
      return;
    }
    const hourStarts = Array.from({length: 24}, (_, hour) => hour);
    const weekday = hourStarts.map(
      hour => profileNet(station, 'weekday', hour) / dayCounts.weekday
    );
    const weekend = hourStarts.map(
      hour => profileNet(station, 'weekend', hour) / dayCounts.weekend
    );
    ctx.setChart('stationChart', {
      kind: 'line',
      series: [
        {label: 'weekdays', x: hourStarts, y: weekday, color: 0},
        {label: 'weekends', x: hourStarts, y: weekend, color: 3}
      ],
      guides: [{y: 0, label: 'balanced'}],
      xDomain: [0, 23],
      xLabel: 'start hour (Montreal time)',
      yLabel: 'net bikes per average hour',
      height: 130,
      formatX: value => `${Math.round(value)}`,
      formatY: value => formatSigned(value, 1),
      link: {option: 'hour', label: value => formatClockHour(value)},
      description: `Net bikes (arrivals minus departures) at ${flows.names[station]} for each hour of an average weekday and weekend day. Above zero the dock fills, below it drains.`
    });
    let worstHour = 0;
    let bestHour = 0;
    for (let hour = 1; hour < 24; hour++) {
      if (weekday[hour] < weekday[worstHour]) worstHour = hour;
      if (weekday[hour] > weekday[bestHour]) bestHour = hour;
    }
    ctx.setReadout(
      'station',
      `${flows.names[station]}\n${flows.boroughNames[flows.borough[station]]}\nweekdays: most leave at ${formatClockHour(worstHour)} (${formatSigned(weekday[worstHour], 1)} per hour)\nmost arrive at ${formatClockHour(bestHour)} (${formatSigned(weekday[bestHour], 1)} per hour)`
    );
  }

  function applySelection(flyThere: boolean, selection?: BixiTidesOptions['selection']): void {
    const station = getSelectedStation(selection);
    publishStationChart(selection);
    publishNotes();
    if (station < 0) {
      ctx.setHighlight(null);
      return;
    }
    const coordinate: LngLat = [flows.lngLat[station * 2], flows.lngLat[station * 2 + 1]];
    ctx.setHighlight({kind: 'point', coordinate, radiusPixels: 13, pulse: true});
    if (flyThere && !ctx.options.play) {
      ctx.flyTo(
        {longitude: coordinate[0], latitude: coordinate[1], zoom: 13},
        {transitionMs: ctx.reducedMotion() ? 0 : 1400}
      );
    }
  }

  function updateFurniture(): void {
    const {units, dayType} = ctx.options;
    const dayWord = dayType === 'weekday' ? 'weekday' : 'weekend day';
    const subtitle =
      units === 'total'
        ? 'Arrivals minus departures per station, rides in August 2024'
        : `Arrivals minus departures per station, per average ${dayWord} hour, August 2024`;
    if (subtitle === lastFurnitureKey) return;
    lastFurnitureKey = subtitle;
    ctx.setFurniture({
      title: {
        subtitle,
        sample: `${formatCount(weekdayRides + weekendRides)} station-to-station rides, ${formatCount(stationCount)} stations`
      },
      scaleBar: {units: 'metric'}
    });
  }

  function getTooltip(station: number): TooltipContent | null {
    const stats = viewA.stats;
    if (!stats) return null;
    const total = ctx.options.units === 'total';
    const days = daysOf(stats.dayType);
    const divisor = total ? 1 : days;
    const unit = total ? 'rides in August' : 'rides per hour';
    const net = getShownNet(stats, station);
    return {
      title: flows.names[station],
      subtitle: `${flows.boroughNames[flows.borough[station]]}, ${formatClockHour(stats.hour)} to ${formatClockHour(stats.hour + 1)}, ${stats.dayType === 'weekday' ? 'weekdays' : 'weekends'}`,
      rows: [
        {
          label: 'Net (arriving minus leaving)',
          value: formatSigned(net, total ? 0 : 1),
          unit,
          swatch: lastTable.colors[getClassIndexOf(lastTable, net)],
          emphasis: true
        },
        {
          label: 'Leaving',
          value: formatCount(stats.out[station] / divisor),
          unit
        },
        {
          label: 'Arriving',
          value: formatCount(stats.incoming[station] / divisor),
          unit
        }
      ],
      highlight: {
        kind: 'point',
        coordinate: [flows.lngLat[station * 2], flows.lngLat[station * 2 + 1]]
      }
    };
  }

  ctx.setReadout(
    'weekdayTotal',
    `${formatCount(weekdayRides)} rides, ${formatCount(weekdayRides / dayCounts.weekday)} per day`
  );
  ctx.setReadout(
    'weekendTotal',
    `${formatCount(weekendRides)} rides, ${formatCount(weekendRides / dayCounts.weekend)} per day`
  );
  ctx.setReadout(
    'dayCounts',
    `${dayCounts.weekday} weekdays and ${dayCounts.weekend} weekend days`
  );
  rebuild();
  writeMask();
  updateFurniture();
  publishCharts();
  applySelection(false);

  return {
    getCompiledGraphs: () =>
      views.flatMap(view =>
        view.graph ? [view.graph.compiled as CompiledGPUCommandGraph<never>] : []
      ),

    setOption(id) {
      switch (id) {
        case 'excludeSelf':
        case 'sumOrder':
          rebuild();
          ctx.requestLayers();
          break;
        case 'dayType':
          writeMask();
          updateFurniture();
          publishCharts();
          break;
        case 'units':
        case 'scaleMode':
        case 'flowCount':
        case 'flowWidth':
          updateFurniture();
          applyDisplay();
          break;
        case 'compareHour':
          markChanged();
          break;
        case 'selection':
          applySelection(
            ctx.options.selection === 'top-drain' || ctx.options.selection === 'top-fill'
          );
          break;
        case 'hour':
        case 'play':
        case 'speed':
        case 'loop':
          publishNotes();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      applyDisplay();
    },

    // The class tables are authored per ground, so a ground flip rebuilds them.
    onGroundChange() {
      applyDisplay();
    },

    onLegendFilter(_id, classes) {
      legendHighlight = classes === null ? null : [...classes];
      ctx.requestLayers();
    },

    getTooltip(event) {
      const station = findStationNearPixel(
        ctx.getViewport(),
        flows.lngLat,
        stationCount,
        event.pixel
      );
      return station >= 0 ? getTooltip(station) : null;
    },

    onClick(event) {
      const station = findStationNearPixel(
        ctx.getViewport(),
        flows.lngLat,
        stationCount,
        event.pixel,
        14
      );
      if (station < 0) return false;
      const unselect = ctx.options.selection === 'clicked' && clickedStation === station;
      clickedStation = unselect ? -1 : station;
      const selection = unselect ? 'none' : 'clicked';
      ctx.setOptions({selection}, {notify: true});
      // Refresh now: the option write is coalesced per frame and a repeat click may be a no-op.
      applySelection(false, selection);
      return true;
    },

    encode(commandEncoder, frame) {
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      clock.advance(frame);
      const nowComparing = ctx.getCompare() !== null;
      if (nowComparing !== comparing) {
        comparing = nowComparing;
        markChanged();
        applyDisplay();
      }
      for (const view of views) {
        const graph = view.graph;
        if (!graph || (view.slot === 'b' && !comparing)) continue;
        const hour = view.slot === 'a' ? getHour() : getCompareHour();
        if (hour !== view.hour) {
          view.hour = hour;
          // The window is closed at both ends: [hour, hour + 0.5] holds only the hour slot `hour`.
          graph.windowBuffer.write(getGPUTimeWindowParameterValues({start: hour, end: hour + 0.5}));
          view.frames = Math.max(view.frames, ENCODE_FRAMES);
          view.stale = true;
          if (view.slot === 'a') {
            ctx.setReadout(
              'window',
              `${formatClockHour(hour)} to ${formatClockHour(hour + 1)}, ${ctx.options.dayType === 'weekday' ? 'weekdays' : 'weekends'}`
            );
          }
        }
        if (view.frames > 0) {
          graph.compiled.encode(commandEncoder, {parameters: undefined});
          view.frames--;
        }
        if (view.stale && view.frames === 0 && !graph.reader.isPending) {
          view.stale = false;
          view.requestKey = {hour, dayType: ctx.options.dayType};
          graph.reader.request(commandEncoder);
        } else {
          graph.reader.flush(commandEncoder);
        }
      }
    },

    getLayers() {
      const options = ctx.options;
      const ground = ctx.ground();
      const layers: Layer[] = [];
      const haloColor =
        ground === 'dark' ? ([20, 24, 28, 235] as const) : ([255, 255, 255, 235] as const);
      const classProps = getClassTableLayerProps(lastTable);
      const maximum = options.units === 'total' ? flowMaximumTotal : flowMaximumRate;
      for (const view of views) {
        if (!view.stats || (view.slot === 'b' && !comparing)) continue;
        // In a compare step the live window is side `a` and the second window side `b`.
        const side = comparing ? {compareSide: view.slot} : {};
        if (options.flowCount > 0 && view.drawnFlows > 0) {
          layers.push(
            new SpatialAnalysisFlowLayer({
              id: `bixi-tides-flows-${view.slot}`,
              coordinateOrigin,
              flows: view.flowEndpoints,
              values: view.flowWeights,
              valueFormat: 'float32',
              ids: view.flowOrder,
              instanceCount: view.drawnFlows,
              maxValue: maximum,
              minWidthPixels: FLOW_MIN_PIXELS,
              maxWidthPixels: TIDES_FLOW_MAX_PIXELS,
              curvature: 0.15,
              arrowheads: true,
              endOffsetPixels: 4,
              color: FLOW_INK[ground],
              outlineColor: FLOW_HALO[ground],
              outlineWidthPixels: 0.8,
              ...side
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `bixi-tides-stations-${view.slot}`,
            coordinateOrigin,
            positions: centers,
            ids: view.order,
            instanceCount: stationCount,
            values: view.values,
            valueFormat: 'float32',
            sizeValues: view.sizes,
            sizeMaximumValue: lastScale.sizeMaximum,
            sizeScale: 'sqrt',
            radiusPixels: TIDES_DISC_MAX_PIXELS,
            radiusMinPixels: TIDES_DISC_MIN_PIXELS,
            radiusMaxPixels: TIDES_DISC_MAX_PIXELS,
            shape: 'circle',
            outlineColor: haloColor,
            outlineWidthPixels: 1,
            noDataColor: [0, 0, 0, 0],
            highlightClasses: legendHighlight,
            ...classProps,
            ...side
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      ctx.setHighlight(null);
      ctx.setAnnotations('tides-notes', null);
      for (const view of views) {
        view.graph?.reader.stop();
        view.graph?.resources.destroy();
      }
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
    }
  };
}
