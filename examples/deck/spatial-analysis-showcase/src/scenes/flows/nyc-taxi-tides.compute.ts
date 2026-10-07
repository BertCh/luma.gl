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
  getGPUPointDensityHexagonCell,
  getGPUPointDensityHexagonCenter,
  getGPUPointDensityHexagonGridSize
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {ARC_SEGMENTS, FlowArcLayer} from './b11-flow-layers';
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
  zones: 'hexagon' | 'grid';
  zoneSize: number;
  excludeSelf: boolean;
  sumOrder: 'sorted' | 'atomic';
  weight: 'trips' | 'fare' | 'passengers';
  totals: 'net' | 'arrivals' | 'departures';
  scaleMode: 'fixed' | 'auto';
  colorRange: number;
  ramp: 'magma' | 'viridis' | 'inferno' | 'cividis';
  zoneOpacity: number;
  arcs: number;
  arcWidth: number;
  arcOpacity: number;
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
const HEADER_WORDS = 4;
/** Distinct origin and destination pairs the hash table keeps. */
const PAIR_CAPACITY = 524288;
const RETIRE_FRAMES = 4;
const ORIGIN_COLOR = [255, 176, 64, 255] as const;
const DESTINATION_COLOR = [64, 224, 255, 255] as const;

type Bounds = [number, number, number, number];

/** One aggregation graph: departures gate on the pickup time, arrivals on the dropoff time. */
type SideGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  count: Buffer;
  totalCount: Buffer;
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
  kind: NycTaxiTidesOptions['zones'];
  zoneCount: number;
  departures: SideGraph;
  arrivals: SideGraph;
  /** Arrivals minus departures per zone plus one NaN sentinel row, written from the readback. */
  net: Buffer;
  drawCommands: DrawCommandBuffer;
  reader: SummaryReader;
};

/**
 * Net arrivals minus departures on a hexagon lattice, by time window. Two `GPUFlowAggregation`
 * graphs read the same 440,000 origin and destination pairs: one gates trips on their pickup time
 * (departures), the other on their dropoff time (arrivals). Both windows are one parameter buffer
 * that the playback clock rewrites every frame; the lattice size is a per-frame parameter under a
 * compile-time capacity. The zone totals come back once per update, the net per zone is written to
 * a buffer that colors the lattice, and the top flows are drawn as arcs straight from GPU storage.
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

  const meanFare = trips.fare.reduce((sum, value) => sum + value, 0) / count;
  const meanPassengers = trips.passengers.reduce((sum, value) => sum + value, 0) / count;
  const MEAN_WEIGHT = {trips: 1, fare: meanFare, passengers: meanPassengers} as const;
  const WEIGHT_UNITS = {trips: 'trips', fare: 'USD of fares', passengers: 'passengers'} as const;

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [TIDES_FIRST_HOUR, TIDES_LAST_HOUR], rate: 1, step: 0.25}
  );

  let graphs: TideGraphs | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let statsStale = true;
  let colorMaximum = 1;
  let lastWindowStart = Number.NaN;
  let lastWindowEnd = Number.NaN;
  let lastChartStep = -1;
  let lastDepartures = new Float32Array(0);
  let lastArrivals = new Float32Array(0);
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];

  // CPU context chart: hourly net trips of the Midtown box, independent of the lattice and weight.
  const midtown = (() => {
    const preset = AREA_PRESETS.midtown;
    const [centerX, centerY] = trips.project(preset.center[0], preset.center[1]);
    const inside = (row: number, array: Float32Array) =>
      Math.abs(array[row * 2] - centerX) < preset.halfWidth &&
      Math.abs(array[row * 2 + 1] - centerY) < preset.halfHeight;
    const hours = Math.ceil(NYC_TAXI_HOURS) + 1;
    const net = new Float64Array(hours);
    for (let row = 0; row < count; row++) {
      const fromInside = inside(row, trips.pickup);
      const toInside = inside(row, trips.dropoff);
      if (fromInside === toInside) continue;
      if (fromInside) net[Math.min(hours - 1, Math.floor(trips.pickupHour[row]))]--;
      else net[Math.min(hours - 1, Math.floor(trips.dropoffHour[row]))]++;
    }
    // The first and last hours miss trips that began outside the data, so they are not drawn.
    net[0] = Number.NaN;
    for (let hour = Math.floor(NYC_TAXI_HOURS); hour < hours; hour++) net[hour] = Number.NaN;
    return net;
  })();

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 3);
    statsStale = true;
  }

  function getLatticeBounds(kind: NycTaxiTidesOptions['zones'], size: number): Bounds {
    const [minX, minY, maxX, maxY] = trips.bounds;
    const pad = kind === 'hexagon' ? 1.2 * size : 100;
    return [minX - pad, minY - pad, maxX + pad, maxY + pad];
  }

  function getGridSize(
    bounds: Bounds,
    kind: NycTaxiTidesOptions['zones'],
    size: number
  ): [number, number] {
    return kind === 'hexagon'
      ? getGPUPointDensityHexagonGridSize(bounds, size)
      : [Math.ceil((bounds[2] - bounds[0]) / size), Math.ceil((bounds[3] - bounds[1]) / size)];
  }

  function getActiveLattice(kind: NycTaxiTidesOptions['zones'], size: number) {
    const bounds = getLatticeBounds(kind, size);
    return {bounds, grid: getGridSize(bounds, kind, size)};
  }

  function writeLattice(): void {
    if (!graphs) return;
    const {zoneSize} = ctx.options;
    const {bounds, grid} = getActiveLattice(graphs.kind, zoneSize);
    radiusBuffer.write(Float32Array.of(zoneSize));
    activeGridBuffer.write(Uint32Array.of(grid[0], grid[1]));
    boundsBuffer.write(Float32Array.from(bounds));
    ctx.setReadout(
      'zones',
      `${grid[0]} × ${grid[1]} ${graphs.kind === 'hexagon' ? 'hexagons' : 'cells'} (capacity ${formatCount(graphs.zoneCount)})`
    );
  }

  function writeWeights(): void {
    const weights =
      ctx.options.weight === 'fare'
        ? trips.fare
        : ctx.options.weight === 'passengers'
          ? Float32Array.from(trips.passengers)
          : new Float32Array(count).fill(1);
    weightsBuffer.write(weights);
    markChanged();
  }

  function buildSide(
    graphResources: SpatialAnalysisResources,
    name: string,
    kind: NycTaxiTidesOptions['zones'],
    capacityGrid: [number, number],
    zoneCount: number,
    gateBuffer: Buffer,
    drawCommands: DrawCommandBuffer | null
  ): SideGraph {
    const sentinel = (rows: number) => {
      const values = new Float32Array(rows + 1);
      values[rows] = Number.NaN;
      return values;
    };
    const k = TOP_FLOW_COUNT;
    const ids = graphResources.createBuffer(`${name}-ids`, k * 4);
    const flowCount = graphResources.createBuffer(`${name}-count`, 4);
    const totalCount = graphResources.createBuffer(`${name}-total-count`, 4);
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
          totalCount: view('total-count', totalCount, 'uint32', 1)
        },
        flowOriginZoneIds: view('flow-origin', originZones, 'uint32', k),
        flowDestinationZoneIds: view('flow-destination', destinationZones, 'uint32', k),
        flowWeights: view('flow-weights', flowWeights, 'float32', k),
        pairOverflow: view('pair-overflow', pairOverflow, 'uint32', 1),
        zoneOutWeights: view('out-w', zoneOutWeights, 'float32', zoneCount),
        zoneInWeights: view('in-w', zoneInWeights, 'float32', zoneCount),
        zoneOutCounts: view('out-c', zoneOutCounts, 'uint32', zoneCount),
        zoneInCounts: view('in-c', zoneInCounts, 'uint32', zoneCount),
        ...(drawCommands
          ? {
              drawInstanceCount: commandGraph.importGPUData(
                `${name}-arcs`,
                drawCommands.getInstanceCountData(0)
              )
            }
          : {})
      })
    );
    return {
      compiled: graphResources.track(commandGraph.compile()),
      count: flowCount,
      totalCount,
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
    const capacityBounds = getLatticeBounds(kind, MINIMUM_ZONE_SIZE);
    const capacityGrid = getGridSize(capacityBounds, kind, MINIMUM_ZONE_SIZE);
    const zoneCount = capacityGrid[0] * capacityGrid[1];
    const net = graphResources.createBuffer('net', zoneCount * 4 + 4);
    const drawCommands = graphResources.track(
      new DrawCommandBuffer(device, {
        id: `taxi-tides-draw-${id}`,
        type: 'draw',
        commands: [{vertexCount: ARC_SEGMENTS * 6, instanceCount: 0}]
      })
    );
    const departures = buildSide(
      graphResources,
      'dep',
      kind,
      capacityGrid,
      zoneCount,
      pickupHourBuffer,
      drawCommands
    );
    const arrivals = buildSide(
      graphResources,
      'arr',
      kind,
      capacityGrid,
      zoneCount,
      dropoffHourBuffer,
      null
    );
    const built: TideGraphs = {
      resources: graphResources,
      kind,
      zoneCount,
      departures,
      arrivals,
      net,
      drawCommands,
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      graphResources,
      `taxi-tides-${id}`,
      [
        {buffer: departures.count, size: 4},
        {buffer: departures.totalCount, size: 4},
        {buffer: departures.pairOverflow, size: 4},
        {buffer: arrivals.pairOverflow, size: 4},
        {buffer: departures.zoneOutWeights, size: zoneCount * 4},
        {buffer: arrivals.zoneInWeights, size: zoneCount * 4},
        {buffer: departures.zoneOutCounts, size: zoneCount * 4},
        {buffer: arrivals.zoneInCounts, size: zoneCount * 4}
      ],
      bytes => {
        if (!destroyed && graphs === built) processStatistics(built, bytes);
      }
    );
    return built;
  }

  function rebuild(): void {
    if (graphs) retired.push({resources: graphs.resources, frames: 0});
    graphs = buildGraphs();
    writeLattice();
    colorMaximum = 1;
    markChanged();
  }

  function getZoneLabel(zone: number): string {
    if (!graphs) return `Zone ${zone}`;
    const {bounds, grid} = getActiveLattice(graphs.kind, ctx.options.zoneSize);
    const column = zone % grid[0];
    const row = Math.floor(zone / grid[0]);
    let x: number;
    let y: number;
    if (graphs.kind === 'hexagon') {
      [x, y] = getGPUPointDensityHexagonCenter(
        column,
        row,
        bounds[0],
        bounds[1],
        ctx.options.zoneSize
      );
    } else {
      const size = ctx.options.zoneSize;
      x = bounds[0] + (column + 0.5) * size;
      y = bounds[1] + (row + 0.5) * size;
    }
    const [longitude, latitude] = trips.unproject(x, y);
    return `${latitude.toFixed(3)}°N ${Math.abs(longitude).toFixed(3)}°W`;
  }

  function formatWeight(value: number): string {
    const unit = WEIGHT_UNITS[ctx.options.weight];
    return `${value < 0 ? '-' : ''}${formatCount(Math.abs(value))} ${unit}`;
  }

  function getColorMaximum(measured: number): number {
    return ctx.options.scaleMode === 'fixed'
      ? Math.max(1, ctx.options.colorRange * MEAN_WEIGHT[ctx.options.weight])
      : Math.max(1, measured);
  }

  function processStatistics(current: TideGraphs, bytes: ArrayBuffer): void {
    const zoneCount = current.zoneCount;
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const [flowRows, pairCount, departurePairOverflow, arrivalPairOverflow] = words;
    const outStart = HEADER_WORDS;
    const out = floats.slice(outStart, outStart + zoneCount);
    const incoming = floats.slice(outStart + zoneCount, outStart + 2 * zoneCount);
    const outCounts = words.subarray(outStart + 2 * zoneCount, outStart + 3 * zoneCount);
    const inCounts = words.subarray(outStart + 3 * zoneCount, outStart + 4 * zoneCount);
    lastDepartures = out;
    lastArrivals = incoming;
    const net = new Float32Array(zoneCount + 1);
    net[zoneCount] = Number.NaN;
    let maximumNet = 0;
    let maximumOut = 0;
    let maximumIn = 0;
    let gainZone = -1;
    let lossZone = -1;
    let gain = 0;
    let loss = 0;
    let departureTrips = 0;
    let arrivalTrips = 0;
    let activeZones = 0;
    for (let zone = 0; zone < zoneCount; zone++) {
      net[zone] = incoming[zone] - out[zone];
      maximumNet = Math.max(maximumNet, Math.abs(net[zone]));
      maximumOut = Math.max(maximumOut, out[zone]);
      maximumIn = Math.max(maximumIn, incoming[zone]);
      if (net[zone] > gain) {
        gain = net[zone];
        gainZone = zone;
      }
      if (net[zone] < loss) {
        loss = net[zone];
        lossZone = zone;
      }
      departureTrips += outCounts[zone];
      arrivalTrips += inCounts[zone];
      if (outCounts[zone] > 0 || inCounts[zone] > 0) activeZones++;
    }
    current.net.write(net);
    const {totals} = ctx.options;
    const measured = totals === 'net' ? maximumNet : totals === 'arrivals' ? maximumIn : maximumOut;
    const nextMaximum = getColorMaximum(measured);
    if (Math.abs(nextMaximum - colorMaximum) > 0.005 * Math.max(colorMaximum, 1)) {
      colorMaximum = nextMaximum;
      ctx.requestLayers();
      ctx.setLegendExtent(
        'zones',
        totals === 'net' ? [-colorMaximum, colorMaximum] : [0, colorMaximum]
      );
    }
    ctx.setReadout('departures', departureTrips);
    ctx.setReadout('arrivals', arrivalTrips);
    ctx.setReadout(
      'gain',
      gainZone >= 0 ? `${formatWeight(gain)} near ${getZoneLabel(gainZone)}` : null
    );
    ctx.setReadout(
      'loss',
      lossZone >= 0 ? `${formatWeight(loss)} near ${getZoneLabel(lossZone)}` : null
    );
    ctx.setReadout('flowRows', flowRows);
    ctx.setReadout('pairs', pairCount);
    ctx.setReadout(
      'pairOverflow',
      departurePairOverflow || arrivalPairOverflow ? 'yes: zone totals incomplete' : 'no'
    );
    ctx.setReadout('activeZones', `${formatCount(activeZones)} of ${formatCount(zoneCount)}`);
  }

  function updateChart(playhead: number): void {
    const step = Math.floor(playhead * 4);
    if (step === lastChartStep) return;
    lastChartStep = step;
    const hours = Array.from(midtown, (_, index) => index + 0.5);
    ctx.setChart('midtownChart', {
      kind: 'line',
      series: [{label: 'net trips per hour', x: hours, y: Array.from(midtown), area: true}],
      xLabel: 'Hours since midnight on Thu 1 Jan',
      yLabel: 'arrivals - departures',
      height: 120,
      formatX: value =>
        `${value < 24 ? 'Thu' : 'Fri'} ${String(Math.floor(value) % 24).padStart(2, '0')}h`,
      guides: [{y: 0}],
      markers: [{x: playhead, label: formatTaxiTime(playhead).split(' ').pop()}],
      description: 'Net taxi arrivals in the Midtown box by hour, with the playhead marked.'
    });
  }

  function getTooltip(coordinate: readonly [number, number]): string | null {
    if (!graphs || lastDepartures.length === 0) return null;
    const [x, y] = trips.project(coordinate[0], coordinate[1]);
    const {zoneSize} = ctx.options;
    const {bounds, grid} = getActiveLattice(graphs.kind, zoneSize);
    let column: number;
    let row: number;
    if (graphs.kind === 'hexagon') {
      [column, row] = getGPUPointDensityHexagonCell(x, y, bounds[0], bounds[1], zoneSize);
    } else {
      column = Math.floor(((x - bounds[0]) / (bounds[2] - bounds[0])) * grid[0]);
      row = Math.floor(((y - bounds[1]) / (bounds[3] - bounds[1])) * grid[1]);
    }
    if (column < 0 || row < 0 || column >= grid[0] || row >= grid[1]) return null;
    const zone = row * grid[0] + column;
    if (zone >= lastDepartures.length) return null;
    const net = lastArrivals[zone] - lastDepartures[zone];
    return `${getZoneLabel(zone)}\nDepartures: ${formatWeight(lastDepartures[zone])}\nArrivals: ${formatWeight(lastArrivals[zone])}\nNet: ${net >= 0 ? '+' : ''}${formatWeight(net)}`;
  }

  ctx.setReadout('rows', count);
  rebuild();
  writeWeights();
  updateChart(ctx.options.time);

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
          rebuild();
          ctx.requestLayers();
          break;
        case 'weight':
          writeWeights();
          colorMaximum = 1;
          ctx.requestLayers();
          break;
        case 'zoneSize':
          writeLattice();
          markChanged();
          ctx.requestLayers();
          break;
        case 'windowHours':
        case 'scaleMode':
        case 'colorRange':
          colorMaximum = 1;
          lastWindowStart = Number.NaN;
          markChanged();
          break;
        case 'totals':
          colorMaximum = 1;
          markChanged();
          ctx.requestLayers();
          break;
        case 'time':
        case 'play':
        case 'playSpeed':
        case 'loop':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
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
        windowBuffer.write(getGPUTimeWindowParameterValues({start, end}));
        markChanged();
        ctx.setReadout('clock', formatTaxiTime(playhead));
        ctx.setReadout(
          'window',
          `${formatTaxiTime(start).split(' ').pop()} to ${formatTaxiTime(Math.min(end, NYC_TAXI_HOURS)).split(' ').pop()}`
        );
        updateChart(playhead);
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
      } else if (playing && frame.frameIndex % 6 === 0 && !graphs.reader.isPending) {
        graphs.reader.request(commandEncoder);
      } else {
        graphs.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!graphs) return [];
      const options = ctx.options;
      const lattice = getActiveLattice(graphs.kind, options.zoneSize);
      const isNet = options.totals === 'net';
      const values = isNet
        ? graphs.net
        : options.totals === 'arrivals'
          ? graphs.arrivals.zoneInWeights
          : graphs.departures.zoneOutWeights;
      const layers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          id: `taxi-tides-zones-${graphs.kind}`,
          coordinateOrigin,
          gridSize: lattice.grid,
          bounds: lattice.bounds,
          binning: graphs.kind === 'hexagon' ? 'hexagon' : 'grid',
          hexagonRadius: options.zoneSize,
          values,
          valueFormat: 'float32',
          colormap: isNet ? 'diverging' : options.ramp,
          valueRange: isNet ? [-colorMaximum, colorMaximum] : [0, colorMaximum],
          sqrtScale: !isNet,
          ...(isNet ? {} : {discardAtOrBelow: 0}),
          color: [255, 255, 255, 255],
          opacity: options.zoneOpacity
        })
      ];
      if (options.arcs > 0) {
        layers.push(
          new FlowArcLayer({
            id: `taxi-tides-arcs-${graphs.kind}`,
            coordinateOrigin,
            flowOriginZoneIds: graphs.departures.originZones,
            flowDestinationZoneIds: graphs.departures.destinationZones,
            flowWeights: graphs.departures.flowWeights,
            flowCount: graphs.departures.count,
            drawCommands: graphs.drawCommands,
            zoneKind: graphs.kind,
            gridSize: lattice.grid,
            bounds: lattice.bounds,
            hexagonRadius: options.zoneSize,
            limit: options.arcs,
            widthMinPixels: 1,
            widthMaxPixels: options.arcWidth,
            opacity: options.arcOpacity,
            originColor: [...ORIGIN_COLOR],
            destinationColor: [...DESTINATION_COLOR]
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
