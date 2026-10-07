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
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {ARC_SEGMENTS, FlowArcLayer} from './b11-flow-layers';
import {
  buildStationProfiles,
  findStationNearPixel,
  formatCompact,
  formatHourOfDay,
  MONTREAL_ORIGIN,
  readBixiFlows,
  shortStationName,
  SLOT_COUNT
} from './bixi-data';

/** Option state of the bixi-tides scene. */
export type BixiTidesOptions = {
  hour: number;
  play: boolean;
  speed: number;
  loop: boolean;
  hourLength: number;
  dayType: 'weekday' | 'weekend' | 'all';
  totals: 'net' | 'departures' | 'arrivals';
  minBalance: number;
  excludeSelf: boolean;
  sumOrder: 'sorted' | 'atomic';
  arcs: number;
  arcWidth: number;
  arcOpacity: number;
  showStations: boolean;
  stationSize: number;
  ramp: 'magma' | 'viridis' | 'inferno' | 'cividis';
};

/** Rows of the top-K flow list (compile-time capacity). */
export const TIDES_TOP_FLOW_COUNT = 512;
const HEADER_WORDS = 4;
const RETIRE_FRAMES = 4;
/** Weekdays and weekend days of August 2024, to turn totals into rides per day. */
const WEEKDAYS = 22;
const WEEKEND_DAYS = 9;
const ORIGIN_COLOR = [255, 176, 64, 255] as const;
const DESTINATION_COLOR = [64, 224, 255, 255] as const;

type TidesGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  sumOrder: BixiTidesOptions['sumOrder'];
  count: Buffer;
  originZones: Buffer;
  destinationZones: Buffer;
  flowWeights: Buffer;
  zoneOutWeights: Buffer;
  zoneInWeights: Buffer;
  drawCommands: DrawCommandBuffer;
  reader: SummaryReader;
};

/**
 * Net flow of BIXI stations through the day. One compiled `GPUFlowAggregation` takes 685,000
 * (origin, destination, weekday or weekend, hour, rides) rows, gates them with a time window,
 * sums the pairs in a GPU hash table and writes a top-512 flow list plus per-station departure
 * and arrival totals. The scene turns the two totals into a net balance (arrivals minus
 * departures) that colours the stations. Window, day type and the balance threshold are buffer
 * writes; only self-flow exclusion and the sum order rebuild the graph.
 */
export async function createBixiTides(
  ctx: SceneContext<BixiTidesOptions>
): Promise<SceneInstance<BixiTidesOptions>> {
  const {device} = ctx;
  const flows = readBixiFlows(ctx.datasets.get('bixi-flows'));
  const {zoneCount, stationCount, slices} = flows;
  const rows = slices.rowCount;
  const resources = new SpatialAnalysisResources(device, 'bixi-tides');
  const coordinateOrigin: [number, number, number] = [MONTREAL_ORIGIN[0], MONTREAL_ORIGIN[1], 0];
  const profiles = buildStationProfiles(flows);

  const originIds = resources.createBuffer('origin', slices.origin);
  const destinationIds = resources.createBuffer('destination', slices.destination);
  const hours = resources.createBuffer('hours', slices.hour);
  const weights = resources.createBuffer('weights', slices.count);
  const mask = resources.createBuffer('mask', new Uint32Array(rows).fill(1));
  const centers = resources.createBuffer('centers', flows.centers);
  const windowBuffer = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );

  let graph: TidesGraph | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let statsStale = true;
  let selected = -1;
  let colorMaximum = 1;
  let lastOut = new Float32Array(zoneCount);
  let lastIn = new Float32Array(zoneCount);
  let lastWindowKey = '';
  let lastChartKey = '';
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];
  const clock = createPlaybackClock(
    ctx,
    {time: 'hour', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, 23.75], rate: 1, step: 0.25, notify: true}
  );
  // Value shown per zone: net, departures or arrivals; NaN hides a station.
  const totalsBuffer = resources.createBuffer('totals', zoneCount * 4);
  const selectedBuffer = resources.createBuffer('selected', Uint32Array.of(0));

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 3);
    statsStale = true;
  }

  function writeMask(): void {
    const {dayType} = ctx.options;
    const values = new Uint32Array(rows);
    for (let row = 0; row < rows; row++) {
      const weekend = slices.dayType[row] === 1;
      values[row] = dayType === 'all' || (dayType === 'weekend') === weekend ? 1 : 0;
    }
    mask.write(values);
    markChanged();
  }

  function getWindow(): [number, number] {
    const {hour, hourLength, play} = ctx.options;
    const start = play ? clock.time : hour;
    return [start, Math.min(24, start + hourLength)];
  }

  function buildGraph(): TidesGraph {
    const id = ++serial;
    const graphResources = new SpatialAnalysisResources(device, `bixi-tides-${id}`);
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
    const drawCommands = graphResources.track(
      new DrawCommandBuffer(device, {
        id: `bixi-tides-draw-${id}`,
        type: 'draw',
        commands: [{vertexCount: ARC_SEGMENTS * 6, instanceCount: 0}]
      })
    );
    const commandGraph = new GPUCommandGraph<void>(device, {id: `bixi-tides-${id}`});
    commandGraph.add(
      new GPUFlowAggregation({
        id: 'tides',
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
        ),
        drawInstanceCount: commandGraph.importGPUData('arcs', drawCommands.getInstanceCountData(0))
      })
    );
    const compiled = graphResources.track(commandGraph.compile());
    const built: TidesGraph = {
      resources: graphResources,
      compiled,
      sumOrder: ctx.options.sumOrder,
      count,
      originZones,
      destinationZones,
      flowWeights,
      zoneOutWeights,
      zoneInWeights,
      drawCommands,
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      graphResources,
      `bixi-tides-${id}`,
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
        if (!destroyed && graph === built) processStatistics(bytes);
      }
    );
    return built;
  }

  function rebuild(): void {
    if (graph) retired.push({resources: graph.resources, frames: 0});
    graph = buildGraph();
    markChanged();
  }

  function formatRides(value: number): string {
    return `${formatCompact(Math.round(value))} rides`;
  }

  /** Average rides per day in one slot range, for the selected day type. */
  function perDay(weekday: number, weekend: number): number {
    const {dayType} = ctx.options;
    if (dayType === 'weekday') return weekday / WEEKDAYS;
    if (dayType === 'weekend') return weekend / WEEKEND_DAYS;
    return (weekday + weekend) / (WEEKDAYS + WEEKEND_DAYS);
  }

  function updateCityChart(start: number, end: number): void {
    const key = `${start}|${end}`;
    if (key === lastWindowKey) return;
    lastWindowKey = key;
    if (!cityProfile) {
      const weekday = new Float64Array(24);
      const weekend = new Float64Array(24);
      for (let row = 0; row < rows; row++) {
        // Rows are slices of departures; sentinel rows (to the sentinel zone) are departures too.
        if (slices.origin[row] >= stationCount) continue;
        (slices.dayType[row] === 1 ? weekend : weekday)[slices.hour[row]] += slices.count[row];
      }
      cityProfile = {
        weekday: weekday.map(value => value / WEEKDAYS),
        weekend: weekend.map(value => value / WEEKEND_DAYS)
      };
    }
    const x = Array.from({length: 24}, (_, hour) => hour + 0.5);
    const full = end - start >= 24;
    ctx.setChart('cityChart', {
      kind: 'line',
      series: [
        {label: 'weekdays', x, y: cityProfile.weekday, color: 0},
        {label: 'weekends', x, y: cityProfile.weekend, color: 3}
      ],
      xDomain: [0, 24],
      markers: full
        ? []
        : [
            {x: start, label: formatHourOfDay(start)},
            {x: end, label: formatHourOfDay(end)}
          ],
      xLabel: 'start hour (local)',
      yLabel: 'rides per hour',
      height: 130,
      formatX: value => `${Math.round(value)}`,
      formatY: value => formatCompact(value),
      description:
        'Average rides starting in each hour of the day, weekdays against weekends, over all of Montreal. Rules mark the time window.'
    });
  }
  let cityProfile: {weekday: Float64Array; weekend: Float64Array} | null = null;

  function updateStationChart(start: number): void {
    if (selected < 0) {
      if (lastChartKey !== 'none') {
        lastChartKey = 'none';
        ctx.setChart('stationChart', null);
        ctx.setReadout('station', 'click a station');
      }
      return;
    }
    const key = `${selected}|${ctx.options.dayType}|${Math.floor(start)}`;
    if (key === lastChartKey) return;
    lastChartKey = key;
    const net = new Float64Array(24);
    let dailyNet = 0;
    let worstHour = 0;
    let bestHour = 0;
    for (let hour = 0; hour < 24; hour++) {
      const weekdaySlot = selected * SLOT_COUNT + hour;
      const weekendSlot = selected * SLOT_COUNT + 24 + hour;
      net[hour] = perDay(
        profiles.incoming[weekdaySlot] - profiles.out[weekdaySlot],
        profiles.incoming[weekendSlot] - profiles.out[weekendSlot]
      );
      dailyNet += net[hour];
      if (net[hour] < net[worstHour]) worstHour = hour;
      if (net[hour] > net[bestHour]) bestHour = hour;
    }
    ctx.setChart('stationChart', {
      kind: 'sparkline',
      values: net,
      highlight: Math.min(23, Math.floor(start)),
      height: 48,
      description: `Net balance of ${flows.names[selected]} for each hour of the day: rides arriving minus rides leaving. The dot is the start of the window.`
    });
    ctx.setReadout(
      'station',
      `${flows.names[selected]}\nmost bikes leave at ${formatHourOfDay(worstHour)} (${net[worstHour].toFixed(1)} per day)\nmost arrive at ${formatHourOfDay(bestHour)} (+${net[bestHour].toFixed(1)} per day)\nover the day: ${dailyNet >= 0 ? '+' : ''}${dailyNet.toFixed(1)} rides`
    );
  }

  function processStatistics(bytes: ArrayBuffer): void {
    const k = TIDES_TOP_FLOW_COUNT;
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const [flowRows, pairCount, , pairOverflow] = words;
    const topOrigin = words.subarray(HEADER_WORDS, HEADER_WORDS + k);
    const topDestination = words.subarray(HEADER_WORDS + k, HEADER_WORDS + 2 * k);
    const topWeight = floats.subarray(HEADER_WORDS + 2 * k, HEADER_WORDS + 3 * k);
    const outStart = HEADER_WORDS + 3 * k;
    const out = floats.slice(outStart, outStart + zoneCount);
    const incoming = floats.slice(outStart + zoneCount, outStart + 2 * zoneCount);
    lastOut = out;
    lastIn = incoming;
    const {totals, minBalance} = ctx.options;
    const shown = new Float32Array(zoneCount);
    shown[stationCount] = Number.NaN;
    let total = 0;
    let maximum = 0;
    let busiest = 0;
    const drains: [number, number][] = [];
    for (let station = 0; station < stationCount; station++) {
      total += out[station];
      const net = incoming[station] - out[station];
      drains.push([net, station]);
      const value =
        totals === 'net' ? net : totals === 'arrivals' ? incoming[station] : out[station];
      const hidden = totals === 'net' && Math.abs(net) < minBalance;
      shown[station] = hidden ? Number.NaN : value;
      if (Math.abs(value) > maximum && !hidden) maximum = Math.abs(value);
      if (out[station] > out[busiest]) busiest = station;
    }
    totalsBuffer.write(shown);
    const nextMaximum = Math.max(maximum, 1);
    if (Math.abs(nextMaximum - colorMaximum) > 0.005 * colorMaximum) {
      colorMaximum = nextMaximum;
      ctx.requestLayers();
    }
    ctx.setLegendExtent(
      'stations',
      totals === 'net' ? [-colorMaximum, colorMaximum] : [0, colorMaximum]
    );

    drains.sort((x, y) => x[0] - y[0]);
    const top = Math.min(5, stationCount);
    const draining = drains.slice(0, top);
    const filling = drains.slice(-top);
    ctx.setChart('extremesChart', {
      kind: 'bars',
      values: [...draining.map(entry => entry[0]), ...filling.map(entry => entry[0])],
      labels: [...draining, ...filling].map(entry => shortStationName(flows.names[entry[1]], 12)),
      highlight: Array.from({length: top}, (_, index) => top + index),
      height: 140,
      yLabel: 'arrivals - departures',
      formatY: formatCompact,
      description:
        'The five stations that lose the most bikes in the window (left, muted) and the five that gain the most (right, accent), in rides over the whole month.'
    });
    ctx.setReadout(
      'topDrain',
      draining[0] ? `${flows.names[draining[0][1]]}: ${formatRides(draining[0][0])}` : null
    );
    ctx.setReadout(
      'topFill',
      filling.length
        ? `${flows.names[filling[filling.length - 1][1]]}: +${formatRides(filling[filling.length - 1][0])}`
        : null
    );
    const {dayType} = ctx.options;
    const days =
      dayType === 'weekday'
        ? WEEKDAYS
        : dayType === 'weekend'
          ? WEEKEND_DAYS
          : WEEKDAYS + WEEKEND_DAYS;
    ctx.setReadout(
      'volume',
      total > 0
        ? `${formatCompact(total / days)} rides per day (${formatCompact(total)} in August)`
        : 'none in window'
    );
    ctx.setReadout('records', pairCount);
    ctx.setReadout(
      'busiest',
      total > 0 ? `${flows.names[busiest]} (${formatRides(out[busiest])})` : null
    );
    ctx.setReadout('pairOverflow', pairOverflow ? 'yes: totals incomplete' : 'no');
    const flowsShown = Math.min(flowRows, ctx.options.arcs);
    let shownWeight = 0;
    for (let flow = 0; flow < flowsShown; flow++) shownWeight += topWeight[flow];
    ctx.setReadout('share', total > 0 ? shownWeight / total : null);
    for (let rank = 0; rank < 3; rank++) {
      ctx.setReadout(
        `flow${rank + 1}`,
        rank < flowRows
          ? `${shortStationName(zoneName(topOrigin[rank]), 22)} to ${shortStationName(zoneName(topDestination[rank]), 22)}: ${formatRides(topWeight[rank])}`
          : null
      );
    }
    updateStationChart(getWindow()[0]);
  }

  function zoneName(zone: number): string {
    return zone < stationCount ? flows.names[zone] : 'other stations';
  }

  function getTooltip(station: number): string {
    const out = lastOut[station] ?? 0;
    const incoming = lastIn[station] ?? 0;
    const net = incoming - out;
    return `${flows.names[station]}\n${flows.boroughNames[flows.borough[station]]}\nleaving ${formatRides(out)}, arriving ${formatRides(incoming)}\nnet ${net >= 0 ? '+' : ''}${formatRides(net)}`;
  }

  rebuild();
  writeMask();

  return {
    getCompiledGraphs: () => (graph ? [graph.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id) {
      switch (id) {
        case 'excludeSelf':
        case 'sumOrder':
          rebuild();
          ctx.requestLayers();
          break;
        case 'dayType':
          writeMask();
          lastChartKey = '';
          break;
        case 'hour':
        case 'hourLength':
        case 'totals':
        case 'minBalance':
        case 'arcs':
          markChanged();
          break;
        case 'play':
        case 'speed':
        case 'loop':
          markChanged();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
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
      selected = station === selected ? -1 : station;
      if (selected >= 0) selectedBuffer.write(Uint32Array.of(selected));
      lastChartKey = '';
      updateStationChart(getWindow()[0]);
      return station >= 0;
    },

    encode(commandEncoder, frame) {
      if (!graph) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      const {play} = ctx.options;
      clock.advance(frame);
      if (play) markChanged();
      const [start, end] = getWindow();
      windowBuffer.write(getGPUTimeWindowParameterValues({start, end}));
      if (frame.frameIndex % 6 === 0) {
        ctx.setReadout(
          'window',
          `${formatHourOfDay(start)} to ${end >= 24 ? '24:00' : formatHourOfDay(end)}`
        );
        updateCityChart(start, end);
        updateStationChart(start);
      }
      if (encodeFrames > 0) {
        graph.compiled.encode(commandEncoder, {parameters: undefined});
        encodeFrames--;
      }
      if (statsStale && encodeFrames === 0 && !graph.reader.isPending && !play) {
        statsStale = false;
        graph.reader.request(commandEncoder);
      } else if (play && frame.frameIndex % 8 === 0 && !graph.reader.isPending) {
        graph.reader.request(commandEncoder);
      } else {
        graph.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!graph) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const net = options.totals === 'net';
      const layers: Layer[] = [
        new FlowArcLayer({
          id: 'bixi-tides-arcs',
          coordinateOrigin,
          flowOriginZoneIds: graph.originZones,
          flowDestinationZoneIds: graph.destinationZones,
          flowWeights: graph.flowWeights,
          flowCount: graph.count,
          drawCommands: graph.drawCommands,
          zoneKind: 'ids',
          zoneCenters: centers,
          limit: options.arcs,
          widthMinPixels: 1,
          widthMaxPixels: options.arcWidth,
          opacity: options.arcOpacity,
          originColor: [...ORIGIN_COLOR],
          destinationColor: [...DESTINATION_COLOR]
        })
      ];
      if (options.showStations) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'bixi-tides-stations',
            coordinateOrigin,
            positions: centers,
            instanceCount: stationCount,
            values: totalsBuffer,
            valueFormat: 'float32',
            colormap: net ? 'diverging' : options.ramp,
            valueRange: net ? [-colorMaximum, colorMaximum] : [0, colorMaximum],
            sqrtScale: !net,
            noDataColor: [0, 0, 0, 0],
            radiusPixels: options.stationSize
          })
        );
      }
      if (selected >= 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'bixi-tides-selected',
            coordinateOrigin,
            positions: centers,
            ids: selectedBuffer,
            instanceCount: 1,
            radiusPixels: options.stationSize + 6,
            color: dark ? [255, 255, 255, 255] : [20, 24, 40, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'bixi-tides-selected-fill',
            coordinateOrigin,
            positions: centers,
            ids: selectedBuffer,
            instanceCount: 1,
            values: totalsBuffer,
            valueFormat: 'float32',
            colormap: net ? 'diverging' : options.ramp,
            valueRange: net ? [-colorMaximum, colorMaximum] : [0, colorMaximum],
            sqrtScale: !net,
            noDataColor: [128, 128, 128, 255],
            radiusPixels: options.stationSize + 3
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      graph?.reader.stop();
      graph?.resources.destroy();
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
    }
  };
}
