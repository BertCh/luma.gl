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
import {getGPUPointDensityHexagonCell} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  CHICAGO_ORIGIN,
  readCommuteSource,
  readTaxiSource,
  type FlowSource
} from './b11-flow-sources';
import {ARC_SEGMENTS, FlowArcLayer} from './b11-flow-layers';
import {getZoneAt} from './b11-zone-raster';

/** Option state of the taxi-flows scene. */
export type TaxiFlowOptions = {
  source: 'taxi' | 'commute';
  zones: 'native' | 'hexagon' | 'grid';
  zoneSize: number;
  excludeSelf: boolean;
  sumOrder: 'sorted' | 'atomic';
  taxiWeight: 'trips' | 'fare' | 'duration';
  commuteWeight: 'all' | 'low' | 'mid' | 'high';
  hourStart: number;
  hourLength: number;
  dayType: 'all' | 'weekday' | 'weekend';
  play: boolean;
  playSpeed: number;
  loop: boolean;
  totals: 'departures' | 'arrivals' | 'net';
  arcs: number;
  arcWidth: number;
  arcOpacity: number;
  showZones: boolean;
  showOutlines: boolean;
  ramp: 'magma' | 'viridis' | 'inferno' | 'cividis' | 'grayscale';
  zoneOpacity: number;
};

/** Rows of the top-K flow list (compile-time capacity). */
export const TOP_FLOW_COUNT = 512;
const SQRT3 = Math.sqrt(3);
/** Smallest lattice size on the slider: it sets the compile-time lattice capacity. */
const MINIMUM_ZONE_SIZE = 400;
const HEADER_WORDS = 4;
/** Frames a replaced graph's buffers stay alive so in-flight frames never touch destroyed buffers. */
const RETIRE_FRAMES = 4;
const ORIGIN_COLOR = [255, 176, 64, 255] as const;
const DESTINATION_COLOR = [64, 224, 255, 255] as const;

type Bounds = [number, number, number, number];

/** Per-source buffers, created the first time a source is shown. */
type PreparedSource = {
  source: FlowSource;
  originIds: Buffer;
  destinationIds: Buffer;
  hours: Buffer | null;
  mask: Buffer;
  weights: Buffer;
  centers: Buffer;
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
  /** Number of zone rows of every zone output (native count or lattice capacity). */
  zoneCount: number;
  capacityGrid: [number, number];
  count: Buffer;
  originZones: Buffer;
  destinationZones: Buffer;
  flowWeights: Buffer;
  zoneOutWeights: Buffer;
  zoneInWeights: Buffer;
  zoneOutCounts: Buffer;
  /** Arrivals minus departures per zone plus one NaN sentinel row, written from the readback. */
  net: Buffer;
  drawCommands: DrawCommandBuffer;
  reader: SummaryReader;
};

/**
 * Origin-destination flow aggregation. One compiled `GPUFlowAggregation` turns every source row
 * (a taxi record of origin area, destination area, weekday or weekend, hour of day, trips; or a
 * home tract, work tract and jobs) into a weight-sorted top-512 flow list and per-zone departure
 * and arrival totals. Zones are the dataset's own areas (`ids`), or a hexagon or square lattice
 * whose size is a per-frame parameter. The time window, weekday mask, weights and lattice size
 * rewrite buffers; only the source, the zone kind, self-flow exclusion and the sum order rebuild.
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

  let graph: FlowGraph | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let statsStale = true;
  const clock = createPlaybackClock(
    ctx,
    {time: 'hourStart', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, 23.75], rate: 1, step: 0.25, notify: true}
  );
  let hourlyShares: {weekday: Float64Array; weekend: Float64Array} | null = null;
  let hourlyKey = '';
  let hourlyWindowKey = '';
  /** Largest zone value of the current statistics, used for the legend and the layer color range. */
  let colorMaximum = 1;
  let lastOut = new Float32Array(0);
  let lastIn = new Float32Array(0);
  let lastNames: (zone: number) => string = zone => `Zone ${zone}`;
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];
  let comparison: {alt: FlowGraph; reader: SummaryReader; main: FlowGraph} | null = null;
  let comparing = false;

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
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let zone = 0; zone < source.zoneCount; zone++) {
      minX = Math.min(minX, source.centers[zone * 2]);
      maxX = Math.max(maxX, source.centers[zone * 2]);
      minY = Math.min(minY, source.centers[zone * 2 + 1]);
      maxY = Math.max(maxY, source.centers[zone * 2 + 1]);
    }
    const name = `${id}`;
    const prepared: PreparedSource = {
      source,
      originIds: resources.createBuffer(`${name}-origin`, source.origin),
      destinationIds: resources.createBuffer(`${name}-destination`, source.destination),
      hours: source.hour ? resources.createBuffer(`${name}-hour`, source.hour) : null,
      mask: resources.createBuffer(`${name}-mask`, mask),
      weights: resources.createBuffer(`${name}-weights`, source.weights[defaultWeight]),
      centers: resources.createBuffer(`${name}-centers`, source.centers),
      zoneIdRaster: resources.createBuffer(`${name}-zone-raster`, source.raster.zoneIds),
      outline: resources.createBuffer(`${name}-outline`, source.raster.outline),
      centerBounds: [minX, minY, maxX, maxY]
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

  /** Lattice bounds for a zone size: the zone centers padded so every center is inside a cell. */
  function getLatticeBounds(
    prepared: PreparedSource,
    kind: TaxiFlowOptions['zones'],
    size: number
  ): Bounds {
    const [minX, minY, maxX, maxY] = prepared.centerBounds;
    const pad = kind === 'hexagon' ? 1.2 * size : 100;
    return [minX - pad, minY - pad, maxX + pad, maxY + pad];
  }

  function getGridSize(
    bounds: Bounds,
    kind: TaxiFlowOptions['zones'],
    size: number
  ): [number, number] {
    const width = bounds[2] - bounds[0];
    const height = bounds[3] - bounds[1];
    return kind === 'hexagon'
      ? [Math.ceil(width / (SQRT3 * size)) + 1, Math.ceil(height / (1.5 * size)) + 1]
      : [Math.ceil(width / size), Math.ceil(height / size)];
  }

  /** Lattice geometry for the current zone size. */
  function getActiveLattice(
    prepared: PreparedSource,
    kind: TaxiFlowOptions['zones'],
    size: number
  ): {bounds: Bounds; grid: [number, number]} {
    const bounds = getLatticeBounds(prepared, kind, size);
    return {bounds, grid: getGridSize(bounds, kind, size)};
  }

  function writeLattice(): void {
    if (!graph) return;
    if (graph.kind === 'native') {
      ctx.setReadout('zones', `${graph.prepared.source.zoneCount} areas (native)`);
      return;
    }
    const {zoneSize} = ctx.options;
    const {bounds, grid} = getActiveLattice(graph.prepared, graph.kind, zoneSize);
    radiusBuffer.write(Float32Array.of(zoneSize));
    activeGridBuffer.write(Uint32Array.of(grid[0], grid[1]));
    boundsBuffer.write(Float32Array.from(bounds));
    ctx.setReadout(
      'zones',
      `${grid[0]} × ${grid[1]} ${graph.kind === 'hexagon' ? 'hexagons' : 'cells'} (capacity ${formatCount(graph.zoneCount)})`
    );
  }

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 3);
    statsStale = true;
  }

  function getWeightKey(): string {
    return ctx.options.source === 'taxi' ? ctx.options.taxiWeight : ctx.options.commuteWeight;
  }

  function writeWeights(): void {
    const prepared = graph?.prepared ?? prepare(ctx.options.source);
    prepared.weights.write(prepared.source.weights[getWeightKey()]);
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

  function getWindow(): [number, number] {
    const {hourStart, hourLength, play} = ctx.options;
    // The clock holds the exact playhead; its slider write-back is rounded to a quarter hour.
    const start = play ? clock.time : hourStart;
    return [start, Math.min(24, start + hourLength)];
  }

  /** Share of each day type's weight in every pickup hour, for the current weight. */
  function updateHourlyChart(start: number, end: number): void {
    const prepared = graph?.prepared;
    const source = prepared?.source;
    if (!source?.hour || !source.dayType) {
      if (hourlyKey !== 'none') {
        hourlyKey = 'none';
        hourlyWindowKey = '';
        ctx.setChart('hourlyChart', null);
      }
      return;
    }
    const key = `${source.id}:${getWeightKey()}`;
    if (key !== hourlyKey) {
      hourlyKey = key;
      const weights = source.weights[getWeightKey()];
      const weekday = new Float64Array(24);
      const weekend = new Float64Array(24);
      for (let row = 0; row < source.rowCount; row++) {
        const bucket = Math.min(23, Math.max(0, Math.floor(source.hour[row])));
        (source.dayType[row] === 1 ? weekend : weekday)[bucket] += weights[row];
      }
      const normalize = (values: Float64Array) => {
        const total = values.reduce((sum, value) => sum + value, 0) || 1;
        return values.map(value => (100 * value) / total);
      };
      hourlyShares = {weekday: normalize(weekday), weekend: normalize(weekend)};
    }
    if (!hourlyShares) return;
    const windowKey = `${key}|${start}|${end}`;
    if (windowKey === hourlyWindowKey) return;
    hourlyWindowKey = windowKey;
    const hours = Array.from({length: 24}, (_, hour) => hour + 0.5);
    const full = end - start >= 24;
    ctx.setChart('hourlyChart', {
      kind: 'line',
      series: [
        {label: 'weekdays', x: hours, y: hourlyShares.weekday, color: 0},
        {label: 'weekends', x: hours, y: hourlyShares.weekend, color: 3}
      ],
      xDomain: [0, 24],
      markers: full
        ? []
        : [
            {x: start, label: formatHour(start)},
            {x: end, label: formatHour(end)}
          ],
      xLabel: 'pickup hour',
      yLabel: '% of the day',
      height: 130,
      formatX: value => `${Math.round(value)}`,
      formatY: value => `${value.toFixed(0)}%`,
      description:
        'Share of the day total in each pickup hour, weekdays against weekends, for the weight on the map. Rules mark the time window.'
    });
  }

  function formatHour(hour: number): string {
    const whole = Math.floor(hour);
    const minutes = Math.round((hour - whole) * 60);
    return `${String(whole).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
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
      const capacityBounds = getLatticeBounds(prepared, kind, MINIMUM_ZONE_SIZE);
      capacityGrid = getGridSize(capacityBounds, kind, MINIMUM_ZONE_SIZE);
    }
    const zoneCount = kind === 'native' ? source.zoneCount : capacityGrid[0] * capacityGrid[1];
    const sentinel = (rows: number) => {
      const values = new Float32Array(rows + 1);
      values[rows] = Number.NaN;
      return values;
    };
    const k = TOP_FLOW_COUNT;
    const ids = graphResources.createBuffer('ids', k * 4);
    const count = graphResources.createBuffer('count', 4);
    const totalCount = graphResources.createBuffer('total-count', 4);
    const overflow = graphResources.createBuffer('overflow', 4);
    const pairOverflow = graphResources.createBuffer('pair-overflow', 4);
    const originZones = graphResources.createBuffer('flow-origin', k * 4);
    const destinationZones = graphResources.createBuffer('flow-destination', k * 4);
    const flowWeights = graphResources.createBuffer('flow-weights', k * 4);
    const zoneOutWeights = graphResources.createBuffer('zone-out-weights', sentinel(zoneCount));
    const zoneInWeights = graphResources.createBuffer('zone-in-weights', sentinel(zoneCount));
    const zoneOutCounts = graphResources.createBuffer('zone-out-counts', zoneCount * 4);
    const net = graphResources.createBuffer('net', sentinel(zoneCount));
    const drawCommands = graphResources.track(
      new DrawCommandBuffer(device, {
        id: `taxi-flows-draw-${id}`,
        type: 'draw',
        commands: [{vertexCount: ARC_SEGMENTS * 6, instanceCount: 0}]
      })
    );

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
        pairCapacity: source.id === 'taxi' ? 16384 : 131072,
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
    const built: FlowGraph = {
      resources: graphResources,
      compiled,
      prepared,
      kind,
      sumOrder,
      zoneCount,
      capacityGrid,
      count,
      originZones,
      destinationZones,
      flowWeights,
      zoneOutWeights,
      zoneInWeights,
      zoneOutCounts,
      net,
      drawCommands,
      // Header (count, total, overflow, pair overflow), top-K ids and weights, then zone totals.
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      graphResources,
      `taxi-flows-${id}`,
      [
        {buffer: count, size: 4},
        {buffer: totalCount, size: 4},
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
    // The new graph reads this source's current weights, mask and lattice parameters.
    prepared.weights.write(prepared.source.weights[getWeightKey()]);
    writeMask();
    writeLattice();
    colorMaximum = 1;
    markChanged();
  }

  function getZoneName(zone: number): string {
    if (!graph) return `Zone ${zone}`;
    if (graph.kind === 'native') return graph.prepared.source.zoneNames[zone] ?? `Zone ${zone}`;
    const columns = getActiveLattice(graph.prepared, graph.kind, ctx.options.zoneSize).grid[0];
    return `${graph.kind === 'hexagon' ? 'Hexagon' : 'Cell'} ${zone % columns}, ${Math.floor(zone / columns)}`;
  }

  function compactNumber(value: number): string {
    return value >= 1e6
      ? `${(value / 1e6).toFixed(1)}M`
      : value >= 1e3
        ? `${(value / 1e3).toFixed(value >= 1e4 ? 0 : 1)}k`
        : value.toFixed(0);
  }

  function formatWeight(value: number): string {
    const units = graph!.prepared.source.weightUnits[getWeightKey()];
    return `${units.prefix}${formatCount(value)} ${units.unit}`;
  }

  function processStatistics(current: FlowGraph, bytes: ArrayBuffer): void {
    const k = TOP_FLOW_COUNT;
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
    lastOut = out;
    lastIn = incoming;
    let total = 0;
    let rows = 0;
    let maximumOut = 0;
    let maximumIn = 0;
    let busiestOut = 0;
    let busiestIn = 0;
    let activeZones = 0;
    const net = new Float32Array(zoneCount + 1);
    net[zoneCount] = Number.NaN;
    let maximumNet = 0;
    for (let zone = 0; zone < zoneCount; zone++) {
      total += out[zone];
      rows += rowCounts[zone];
      if (out[zone] > maximumOut) {
        maximumOut = out[zone];
        busiestOut = zone;
      }
      if (incoming[zone] > maximumIn) {
        maximumIn = incoming[zone];
        busiestIn = zone;
      }
      if (out[zone] > 0 || incoming[zone] > 0) activeZones++;
      net[zone] = incoming[zone] - out[zone];
      maximumNet = Math.max(maximumNet, Math.abs(net[zone]));
    }
    current.net.write(net);
    const {totals, arcs} = ctx.options;
    const nextMaximum =
      totals === 'net' ? maximumNet : totals === 'arrivals' ? maximumIn : maximumOut;
    if (Math.abs(nextMaximum - colorMaximum) > 0.005 * Math.max(colorMaximum, 1)) {
      colorMaximum = Math.max(nextMaximum, 1);
      ctx.requestLayers();
    }
    ctx.setLegendExtent(
      'zones',
      totals === 'net' ? [-colorMaximum, colorMaximum] : [0, colorMaximum]
    );
    const shown = Math.min(flowRows, arcs);
    let shownWeight = 0;
    for (let flow = 0; flow < shown; flow++) shownWeight += topWeight[flow];
    ctx.setReadout('volume', total > 0 ? formatWeight(total) : 'none in window');
    ctx.setReadout('records', rows);
    ctx.setReadout('pairs', pairCount);
    ctx.setReadout('drawn', shown);
    ctx.setReadout('share', total > 0 ? shownWeight / total : null);
    ctx.setReadout(
      'truncated',
      pairCount > k ? `yes (${formatCount(pairCount - k)} pairs not listed)` : 'no'
    );
    ctx.setReadout('pairOverflow', pairOverflow ? 'yes: totals incomplete' : 'no');
    ctx.setReadout('activeZones', `${formatCount(activeZones)} of ${formatCount(zoneCount)}`);
    ctx.setReadout(
      'busiestOrigin',
      total > 0 ? `${getZoneName(busiestOut)} (${formatWeight(maximumOut)})` : null
    );
    ctx.setReadout(
      'busiestDestination',
      total > 0 ? `${getZoneName(busiestIn)} (${formatWeight(maximumIn)})` : null
    );
    for (let rank = 0; rank < 3; rank++) {
      ctx.setReadout(
        `flow${rank + 1}`,
        rank < flowRows
          ? `${getZoneName(topOrigin[rank])} → ${getZoneName(topDestination[rank])}: ${formatWeight(topWeight[rank])}`
          : null
      );
    }
    // Top flows: the heaviest few as bars, and how fast the cumulative share rises with rank.
    const topCount = Math.min(flowRows, 10);
    ctx.setChart(
      'topFlowsChart',
      topCount
        ? {
            kind: 'bars',
            values: Array.from(topWeight.subarray(0, topCount)),
            labels: Array.from({length: topCount}, (_, rank) => `${rank + 1}`),
            highlight: [0],
            height: 110,
            yLabel: ctx.options.source === 'taxi' ? 'weight' : 'jobs',
            xLabel: 'rank of the flow',
            formatY: compactNumber,
            description: 'Weight of the ten largest flows, from the ranked list.'
          }
        : null
    );
    const listed = Math.min(flowRows, k);
    if (listed > 1 && total > 0) {
      const ranks = new Float64Array(listed);
      const share = new Float64Array(listed);
      let running = 0;
      for (let flow = 0; flow < listed; flow++) {
        running += topWeight[flow];
        ranks[flow] = flow + 1;
        share[flow] = (100 * running) / total;
      }
      ctx.setChart('concentrationChart', {
        kind: 'line',
        series: [{label: 'cumulative share', x: ranks, y: share, area: true, color: 2}],
        yDomain: [0, 100],
        markers: [{x: Math.max(1, shown), label: 'drawn'}],
        xLabel: 'flows, largest first',
        yLabel: '% of all flow',
        height: 120,
        formatY: value => `${value.toFixed(0)}%`,
        description:
          'Cumulative share of the total weight carried by the largest flows. A steep curve means a few pairs dominate; the rule is the number of arcs drawn.'
      });
    } else {
      ctx.setChart('concentrationChart', null);
    }
    lastNames = getZoneName;
  }

  function getTooltip(coordinate: readonly [number, number]): string | null {
    if (!graph || lastOut.length === 0) return null;
    const [x, y] = projection.project(coordinate[0], coordinate[1]);
    let zone = -1;
    if (graph.kind === 'native') {
      zone = getZoneAt(graph.prepared.source.raster, x, y);
    } else {
      const {zoneSize} = ctx.options;
      const {bounds, grid} = getActiveLattice(graph.prepared, graph.kind, zoneSize);
      let column: number;
      let row: number;
      if (graph.kind === 'hexagon') {
        [column, row] = getGPUPointDensityHexagonCell(x, y, bounds[0], bounds[1], zoneSize);
      } else {
        column = Math.floor(((x - bounds[0]) / (bounds[2] - bounds[0])) * grid[0]);
        row = Math.floor(((y - bounds[1]) / (bounds[3] - bounds[1])) * grid[1]);
      }
      if (column >= 0 && row >= 0 && column < grid[0] && row < grid[1])
        zone = row * grid[0] + column;
    }
    if (zone < 0 || zone >= lastOut.length) return null;
    return `${lastNames(zone)}\nDepartures: ${formatWeight(lastOut[zone])}\nArrivals: ${formatWeight(lastIn[zone])}`;
  }

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

  rebuild();

  return {
    getCompiledGraphs: () => (graph ? [graph.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id, _value, state) {
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
          break;
        case 'zoneSize':
          writeLattice();
          markChanged();
          ctx.requestLayers();
          break;
        case 'hourStart':
        case 'hourLength':
          markChanged();
          break;
        case 'play':
        case 'playSpeed':
        case 'loop':
          markChanged();
          break;
        case 'totals':
          colorMaximum = 1;
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
      const {play} = ctx.options;
      clock.advance(frame);
      if (play) markChanged();
      const [start, end] = getWindow();
      if (graph.prepared.hours) {
        windowBuffer.write(getGPUTimeWindowParameterValues({start, end}));
        if (frame.frameIndex % 6 === 0) {
          ctx.setReadout('window', `${formatHour(start)} to ${formatHour(end)}`);
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
      if (statsStale && encodeFrames === 0 && !graph.reader.isPending && !play) {
        statsStale = false;
        graph.reader.request(commandEncoder);
      } else if (play && frame.frameIndex % 8 === 0 && !graph.reader.isPending) {
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
      const dark = ctx.theme() === 'dark';
      const lattice =
        graph.kind === 'native' ? null : getActiveLattice(prepared, graph.kind, options.zoneSize);
      const isNet = options.totals === 'net';
      const values = isNet
        ? graph.net
        : options.totals === 'arrivals'
          ? graph.zoneInWeights
          : graph.zoneOutWeights;
      const common = {
        values,
        valueFormat: 'float32' as const,
        colormap: isNet ? ('diverging' as const) : options.ramp,
        valueRange: (isNet ? [-colorMaximum, colorMaximum] : [0, colorMaximum]) as [number, number],
        sqrtScale: !isNet,
        ...(isNet ? {} : {discardAtOrBelow: 0}),
        color: [255, 255, 255, 255] as [number, number, number, number],
        opacity: options.zoneOpacity
      };
      const layers: Layer[] = [];
      if (options.showZones) {
        if (!lattice) {
          layers.push(
            new SpatialAnalysisRasterLayer({
              id: `flow-zones-${source.id}`,
              coordinateOrigin,
              gridSize: [source.raster.columns, source.raster.rows],
              bounds: source.raster.bounds,
              binning: 'grid',
              valueIndices: prepared.zoneIdRaster,
              ...common
            })
          );
        } else {
          layers.push(
            new SpatialAnalysisRasterLayer({
              id: `flow-zones-${graph.kind}`,
              coordinateOrigin,
              gridSize: lattice.grid,
              bounds: lattice.bounds,
              binning: graph.kind === 'hexagon' ? 'hexagon' : 'grid',
              hexagonRadius: options.zoneSize,
              ...common
            })
          );
        }
      }
      if (options.showOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `flow-outline-${source.id}`,
            coordinateOrigin,
            segments: prepared.outline,
            instanceCount: source.raster.outline.length / 4,
            widthPixels: 1,
            color: dark ? [225, 230, 245, 70] : [40, 50, 80, 70]
          })
        );
      }
      layers.push(
        new FlowArcLayer({
          id: `flow-arcs-${graph.kind}-${source.id}`,
          coordinateOrigin,
          flowOriginZoneIds: graph.originZones,
          flowDestinationZoneIds: graph.destinationZones,
          flowWeights: graph.flowWeights,
          flowCount: graph.count,
          drawCommands: graph.drawCommands,
          zoneKind: graph.kind === 'native' ? 'ids' : graph.kind,
          zoneCenters: prepared.centers,
          gridSize: lattice?.grid,
          bounds: lattice?.bounds,
          hexagonRadius: options.zoneSize,
          limit: options.arcs,
          widthMinPixels: 1,
          widthMaxPixels: options.arcWidth,
          opacity: options.arcOpacity,
          originColor: [...ORIGIN_COLOR],
          destinationColor: [...DESTINATION_COLOR]
        })
      );
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
