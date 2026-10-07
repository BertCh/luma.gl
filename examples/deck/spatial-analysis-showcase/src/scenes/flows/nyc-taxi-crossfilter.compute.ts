// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCrossfilter} from '@luma.gl/experimental/gpu-crossfilter';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getClassTableLayerProps, getClassIndexOf} from '../../cartography/class-table';
import {NYC, nearestPlaceLabel} from '../../cartography/gazetteer';
import {formatCount, formatPercent, liveText} from '../../cartography/live-text';
import {createNearestIndex, type NearestIndex} from '../../cartography/picking';
import type {LngLat, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {getViewportMetricBounds, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, ScenePointerEvent, TooltipRow} from '../scene';
import {DROPOFF_INK, PICKUP_INK} from './flows-style';
import {
  buildTaxiClassTables,
  GHOST_ALPHA,
  GHOST_INK,
  getAreaBoundsMeters,
  getHourColor,
  getPlaceCenter,
  LINK_INK,
  PASSENGER_COLORS,
  type TaxiAreaId,
  type TaxiClassTables
} from './nyc-taxi-crossfilter-style';
import {
  DISTANCE_DOMAIN,
  FARE_DOMAIN,
  formatTaxiTime,
  getTaxiHourOfDay,
  getTaxiTripSegments,
  HOUR_DOMAIN,
  loadTaxiTrips,
  NYC_TAXI_ORIGIN,
  PASSENGER_DOMAIN,
  type TaxiTrips
} from './nyc-taxi-data';

/** Option state of the nyc-taxi-crossfilter scene. */
export type NycTaxiCrossfilterOptions = {
  hours: readonly [number, number];
  distance: readonly [number, number];
  fare: readonly [number, number];
  passengers: readonly [number, number];
  area: 'none' | 'midtown' | 'downtown' | 'jfk' | 'laguardia' | 'drawn';
  brushMap: boolean;
  day: 'all' | 'jan1' | 'jan2';
  selfExclude: boolean;
  show: 'pickups' | 'dropoffs' | 'both';
  colorBy: 'none' | 'fare' | 'distance' | 'hour' | 'passengers';
  fareClasses: 'quantile' | 'equal' | 'swipe';
  blending: 'normal' | 'additive';
  showLinks: boolean;
  showFiltered: boolean;
  pointScale: number;
};

const HOUR_BINS = 39;
const DISTANCE_BINS = 30;
/** Half-dollar bins: the modal fare is exact, and four of them make one two-dollar chart bar. */
const FARE_BINS = 120;
const FARE_CHART_BINS = 30;
/** Dense group keys 0 to 6 (0 never occurs). */
const PASSENGER_GROUPS = 7;
/** Frames a replaced graph stays alive so in-flight frames never touch destroyed buffers. */
const RETIRE_FRAMES = 4;
/** Value that stands for "no bound" on a brush that sits at the end of its slider. */
const UNBOUNDED = 1e9;
/** Trip links are drawn only for selections this small (about a tenth of the trips). */
const LINK_LIMIT = 60000;
/** Milliseconds the camera must rest before the in-view share is counted. */
const SETTLE_MILLISECONDS = 400;
/** Hover reach in CSS pixels. */
const HOVER_PIXELS = 12;
/** The selected dots: radius by zoom (design sheet: 1.0 / 1.5 / 2.5 px). */
const SELECTED_RADIUS_STOPS: readonly (readonly [number, number])[] = [
  [10.5, 1],
  [11, 1.5],
  [12.5, 1.5],
  [13, 2.5]
];
const GHOST_RADIUS_STOPS: readonly (readonly [number, number])[] = [
  [10.5, 0.8],
  [13, 1.2]
];

type Bounds = [number, number, number, number];
type Brush = [number, number] | null;

type FilterGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  filter: GPUCrossfilter<void>;
  selfExclude: boolean;
  reader: SummaryReader;
  resources: SpatialAnalysisResources;
};

/** Returns an inclusive or half-open brush, or `null` when the handles cover the whole slider. */
function toBrush(range: readonly [number, number], domain: readonly [number, number]): Brush {
  const [low, high] = range;
  if (low <= domain[0] && high >= domain[1]) return null;
  return [low <= domain[0] ? -UNBOUNDED : low, high >= domain[1] ? UNBOUNDED : high];
}

const formatMoney = (value: number) => `$${value.toFixed(2)}`;

/**
 * Crossfilter over 440,000 taxi trips. One `GPUCrossfilter` controller holds a map rectangle and
 * four scalar brushes; linked histograms, group statistics, a count and a compact list of visible
 * trip ids are all compute passes over the same buffers. A brush only rewrites a five-word
 * selection state; the graph is encoded again and a few hundred numbers come back for the charts.
 * The map draws the visible ids straight from GPU storage with an indirect count, so a trip that
 * a brush removed is neither drawn nor counted, with no CPU round trip. The class breaks of the
 * fare, distance and party-size colours are computed once at load and never move with a brush.
 */
export async function createNycTaxiCrossfilter(
  ctx: SceneContext<NycTaxiCrossfilterOptions>
): Promise<SceneInstance<NycTaxiCrossfilterOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get('poopdeck-nyc-taxi');
  const trips: TaxiTrips = loadTaxiTrips(dataset);
  const projection = dataset.getProjection(NYC_TAXI_ORIGIN);
  const count = trips.count;
  const resources = new SpatialAnalysisResources(device, 'taxi-crossfilter');
  const coordinateOrigin: [number, number, number] = [NYC_TAXI_ORIGIN[0], NYC_TAXI_ORIGIN[1], 0];
  const tables: TaxiClassTables = buildTaxiClassTables(trips);
  ctx.setLegendData('tables', tables);

  // ---- Totals of every trip, for the "share of the money" and "versus all trips" readouts -------
  let fareTotalAll = 0;
  let distanceTotalAll = 0;
  for (let row = 0; row < count; row++) {
    fareTotalAll += trips.fare[row];
    distanceTotalAll += trips.distance[row];
  }
  const perMileAll = distanceTotalAll > 0 ? fareTotalAll / distanceTotalAll : 0;
  const equalMajorShare = Math.max(...tables.fareEqualCounts) / count;
  const quantileMajorShare = Math.max(...tables.fareQuantileCounts) / count;

  // ---- Source columns, uploaded once ------------------------------------------------------------
  let gpuBytes = 0;
  const makeBuffer = (name: string, data: Float32Array | Uint32Array | number): Buffer => {
    const buffer = resources.createBuffer(name, data);
    gpuBytes += buffer.byteLength;
    return buffer;
  };
  const pickupX = new Float32Array(count);
  const pickupY = new Float32Array(count);
  for (let row = 0; row < count; row++) {
    pickupX[row] = trips.pickup[row * 2];
    pickupY[row] = trips.pickup[row * 2 + 1];
  }
  const pickupXBuffer = makeBuffer('pickup-x', pickupX);
  const pickupYBuffer = makeBuffer('pickup-y', pickupY);
  const pickupPositions = makeBuffer('pickup-xy', trips.pickup);
  const dropoffPositions = makeBuffer('dropoff-xy', trips.dropoff);
  const hourBuffer = makeBuffer('hour', trips.pickupHour);
  const hourOfDayBuffer = makeBuffer('hour-of-day', getTaxiHourOfDay(trips.pickupHour));
  const distanceBuffer = makeBuffer('distance', trips.distance);
  const fareBuffer = makeBuffer('fare', trips.fare);
  const passengerBuffer = makeBuffer('passengers', Float32Array.from(trips.passengers));
  const passengerKeys = makeBuffer('passenger-keys', trips.passengers);
  const liveMaskBuffer = makeBuffer('live-mask', new Uint32Array(count).fill(1));
  const visibleIds = makeBuffer('visible-ids', count * 4);
  const tripSegments = makeBuffer('trip-links', getTaxiTripSegments(trips));
  const filteredDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'taxi-crossfilter-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  // ---- Linked-view outputs (same buffers for every graph rebuild) -------------------------------
  const hourBins = makeBuffer('hour-bins', HOUR_BINS * 4);
  const distanceBins = makeBuffer('distance-bins', DISTANCE_BINS * 4);
  const fareBins = makeBuffer('fare-bins', FARE_BINS * 4);
  const passengerCounts = makeBuffer('passenger-counts', PASSENGER_GROUPS * 4);
  const selectedPassengers = makeBuffer('selected-passengers', PASSENGER_GROUPS * 4);
  const fareSums = makeBuffer('fare-sums', PASSENGER_GROUPS * 4);
  const distanceSums = makeBuffer('distance-sums', PASSENGER_GROUPS * 4);
  const selectedCount = makeBuffer('selected-count', 4);
  const summarySources = [
    {buffer: hourBins, size: HOUR_BINS * 4},
    {buffer: distanceBins, size: DISTANCE_BINS * 4},
    {buffer: fareBins, size: FARE_BINS * 4},
    {buffer: passengerCounts, size: PASSENGER_GROUPS * 4},
    {buffer: selectedPassengers, size: PASSENGER_GROUPS * 4},
    {buffer: fareSums, size: PASSENGER_GROUPS * 4},
    {buffer: distanceSums, size: PASSENGER_GROUPS * 4},
    {buffer: selectedCount, size: 4}
  ];
  const readbackBytes = summarySources.reduce((total, source) => total + source.size, 0);

  let destroyed = false;
  let graph: FilterGraph | null = null;
  let serial = 0;
  let encodeFrames = 3;
  let summaryStale = true;
  let mapBrush: Bounds | null = null;
  let dragStart: [number, number] | null = null;
  /** Rows alive after the day switch (the live mask). */
  let liveCount = count;
  let linksOn = false;
  const retired: {graph: FilterGraph; frames: number}[] = [];

  // CPU mirror of the brushes, used only for the hover tooltip (the GPU decides everything else).
  const cpuBrush: {
    hours: Brush;
    distance: Brush;
    fare: Brush;
    passengers: Brush;
    area: Bounds | null;
  } = {hours: null, distance: null, fare: null, passengers: null, area: null};

  // ---- Furniture and views -----------------------------------------------------------------------
  const runtimeFurniture: {
    title: {sample: string};
    scaleBar: {units: 'metric'; ticks?: number[]};
  } = {
    title: {sample: `${formatCount(count)} trips, a sample of the TLC records`},
    scaleBar: {units: 'metric'}
  };
  ctx.setFurniture(runtimeFurniture);
  ctx.setAnnotationHalo('heavy');
  let lastTickMeters: number | null | undefined;

  let viewBounds: Bounds | null = null;
  let viewChangedAt = performance.now();
  let viewStale = true;

  function buildGraph(selfExclude: boolean): FilterGraph {
    const id = ++serial;
    const graphResources = new SpatialAnalysisResources(device, `taxi-crossfilter-${id}`);
    const commandGraph = new GPUCommandGraph<void>(device, {id: `taxi-crossfilter-${id}`});
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length = count
    ) => importGraphBuffer(commandGraph, name, buffer, format, length);
    const x = view('pickup-x', pickupXBuffer, 'float32');
    const y = view('pickup-y', pickupYBuffer, 'float32');
    const hour = view('hour', hourBuffer, 'float32');
    const distance = view('distance', distanceBuffer, 'float32');
    const fare = view('fare', fareBuffer, 'float32');
    const passengers = view('passengers', passengerBuffer, 'float32');
    const keys = view('passenger-keys', passengerKeys, 'uint32');
    const own = {includeOwnSelection: !selfExclude};
    const filter = new GPUCrossfilter<void>(commandGraph, {
      id: 'taxi',
      dimensions: [
        {id: 'map', kind: 'bounds', x, y, rejectNonFinite: true},
        {id: 'hour', kind: 'range', input: hour, exclusiveMaximum: true},
        {id: 'distance', kind: 'range', input: distance, exclusiveMaximum: true},
        {id: 'fare', kind: 'range', input: fare, exclusiveMaximum: true},
        {id: 'passengers', kind: 'range', input: passengers}
      ],
      // Rows of the day that is switched off are dead: they leave every view below.
      liveMask: view('live-mask', liveMaskBuffer, 'uint32'),
      views: [
        {
          id: 'hour-bins',
          kind: 'histogram',
          dimension: 'hour',
          input: hour,
          domain: HOUR_DOMAIN,
          output: view('hour-bins', hourBins, 'uint32', HOUR_BINS),
          ...own
        },
        {
          id: 'distance-bins',
          kind: 'histogram',
          dimension: 'distance',
          input: distance,
          domain: DISTANCE_DOMAIN,
          output: view('distance-bins', distanceBins, 'uint32', DISTANCE_BINS),
          ...own
        },
        {
          id: 'fare-bins',
          kind: 'histogram',
          dimension: 'fare',
          input: fare,
          domain: FARE_DOMAIN,
          output: view('fare-bins', fareBins, 'uint32', FARE_BINS),
          ...own
        },
        {
          id: 'passenger-counts',
          kind: 'group',
          dimension: 'passengers',
          keys,
          output: view('passenger-counts', passengerCounts, 'uint32', PASSENGER_GROUPS),
          ...own
        },
        {
          id: 'selected-passengers',
          kind: 'group',
          keys,
          output: view('selected-passengers', selectedPassengers, 'uint32', PASSENGER_GROUPS)
        },
        {
          id: 'fare-sums',
          kind: 'group',
          keys,
          operation: 'sum',
          values: fare,
          output: view('fare-sums', fareSums, 'float32', PASSENGER_GROUPS)
        },
        {
          id: 'distance-sums',
          kind: 'group',
          keys,
          operation: 'sum',
          values: distance,
          output: view('distance-sums', distanceSums, 'float32', PASSENGER_GROUPS)
        },
        {
          id: 'selected-count',
          kind: 'count',
          output: view('selected-count', selectedCount, 'uint32', 1)
        },
        {
          id: 'visible-trips',
          kind: 'visibility',
          output: view('visible-ids', visibleIds, 'uint32'),
          count: commandGraph.importGPUData(
            'visible-draw-count',
            filteredDraw.getInstanceCountData(0)
          )
        }
      ]
    });
    filter.addToGraph(commandGraph);
    // Destroyed in reverse order: the compiled graph goes before the controller's selection buffers.
    graphResources.track({destroy: () => filter.destroy()});
    const compiled = graphResources.track(commandGraph.compile());
    const built: FilterGraph = {
      compiled,
      filter,
      selfExclude,
      resources: graphResources,
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      graphResources,
      `taxi-crossfilter-${id}`,
      summarySources,
      bytes => {
        if (!destroyed && graph === built) processSummary(bytes);
      }
    );
    return built;
  }

  /** The map rectangle of the current `area` option, in meters. */
  function getAreaBounds(): Bounds | null {
    const {area} = ctx.options;
    if (area === 'none') return null;
    if (area === 'drawn') return mapBrush;
    return getAreaBoundsMeters(trips, area as TaxiAreaId);
  }

  /** True when any brush or the day switch removes trips. */
  function isFiltering(): boolean {
    const options = ctx.options;
    return (
      Boolean(toBrush(options.hours, HOUR_DOMAIN)) ||
      Boolean(toBrush(options.distance, DISTANCE_DOMAIN)) ||
      Boolean(toBrush(options.fare, FARE_DOMAIN)) ||
      Boolean(toBrush(options.passengers, PASSENGER_DOMAIN)) ||
      Boolean(getAreaBounds()) ||
      options.day !== 'all'
    );
  }

  /** The brush rectangle as a dashed frame with its size, and a scale-bar tick at its half-width. */
  function updateBrushFurniture(area: Bounds | null): void {
    if (!area) {
      ctx.setAnnotations('brush', null);
    } else {
      const [west, south] = trips.unproject(area[0], area[1]);
      const [east, north] = trips.unproject(area[2], area[3]);
      const widthKm = (area[2] - area[0]) / 1000;
      const heightKm = (area[3] - area[1]) / 1000;
      ctx.setAnnotations('brush', [
        {
          kind: 'frame',
          id: 'brush-frame',
          bounds: [west, south, east, north],
          text: `${widthKm.toFixed(1)} × ${heightKm.toFixed(1)} km`
        }
      ]);
    }
    const tick = area ? Math.round((area[2] - area[0]) / 2 / 100) * 100 : null;
    if (tick !== lastTickMeters) {
      lastTickMeters = tick;
      runtimeFurniture.scaleBar = {units: 'metric', ticks: tick ? [tick] : undefined};
      ctx.setFurniture(runtimeFurniture);
    }
  }

  /** Writes every brush into the controller's five-word selection states. */
  function applyBrushes(): void {
    if (!graph) return;
    const {filter} = graph;
    const options = ctx.options;
    const set = (dimension: string, brush: Brush) =>
      brush ? filter.setRange(dimension, brush) : filter.clear(dimension);
    cpuBrush.hours = toBrush(options.hours, HOUR_DOMAIN);
    cpuBrush.distance = toBrush(options.distance, DISTANCE_DOMAIN);
    cpuBrush.fare = toBrush(options.fare, FARE_DOMAIN);
    cpuBrush.passengers = toBrush(options.passengers, PASSENGER_DOMAIN);
    set('hour', cpuBrush.hours);
    set('distance', cpuBrush.distance);
    set('fare', cpuBrush.fare);
    set('passengers', cpuBrush.passengers);
    const area = getAreaBounds();
    cpuBrush.area = area;
    if (area) filter.setBounds('map', area);
    else filter.clear('map');
    updateBrushFurniture(area);
    markChanged();
  }

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 2);
    summaryStale = true;
  }

  function writeLiveMask(): void {
    const {day} = ctx.options;
    const mask = new Uint32Array(count);
    let alive = 0;
    for (let row = 0; row < count; row++) {
      const first = trips.pickupHour[row] < 24;
      mask[row] = day === 'all' || (day === 'jan1') === first ? 1 : 0;
      alive += mask[row];
    }
    liveCount = alive;
    liveMaskBuffer.write(mask);
    markChanged();
  }

  function rebuild(): void {
    if (graph) retired.push({graph, frames: 0});
    graph = buildGraph(ctx.options.selfExclude);
    ctx.setCost({records: count, passes: graph.compiled.stats.nodeOrder.length});
    applyBrushes();
  }

  const hourLabel = (hours: number) =>
    `${hours < 24 ? 'Thu' : 'Fri'} ${String(Math.floor(hours) % 24).padStart(2, '0')}h`;

  /** Colour of one hour bar of the hour chart: the cyclic ramp of the map. */
  const hourBarColors = Array.from({length: HOUR_BINS}, (_, bin) => getHourColor(bin + 0.5));

  /** Where the nearest selected trip sits under the cursor (tooltip only; the GPU decides the rest). */
  function passesBrushes(row: number): boolean {
    const {day} = ctx.options;
    const hour = trips.pickupHour[row];
    if (day !== 'all' && (day === 'jan1') !== hour < 24) return false;
    const inBrush = (brush: Brush, value: number) =>
      !brush || (value >= brush[0] && value < brush[1]);
    if (!inBrush(cpuBrush.hours, hour)) return false;
    if (!inBrush(cpuBrush.distance, trips.distance[row])) return false;
    if (!inBrush(cpuBrush.fare, trips.fare[row])) return false;
    const passengers = trips.passengers[row];
    if (cpuBrush.passengers) {
      if (passengers < ctx.options.passengers[0] || passengers > ctx.options.passengers[1]) {
        return false;
      }
    }
    const area = cpuBrush.area;
    if (area) {
      const x = trips.pickup[row * 2];
      const y = trips.pickup[row * 2 + 1];
      if (x < area[0] || x > area[2] || y < area[1] || y > area[3]) return false;
    }
    return true;
  }

  const nearestIndexes: {pickup?: NearestIndex; dropoff?: NearestIndex} = {};
  function getNearestIndex(end: 'pickup' | 'dropoff'): NearestIndex {
    let index = nearestIndexes[end];
    if (!index) {
      const meters = end === 'pickup' ? trips.pickup : trips.dropoff;
      const lngLat = new Float64Array(count * 2);
      for (let row = 0; row < count; row++) {
        const [longitude, latitude] = trips.unproject(meters[row * 2], meters[row * 2 + 1]);
        lngLat[row * 2] = longitude;
        lngLat[row * 2 + 1] = latitude;
      }
      index = createNearestIndex(lngLat);
      nearestIndexes[end] = index;
    }
    return index;
  }

  /** The selected trip closest to a pointer position, or -1. */
  function findTrip(coordinate: LngLat): {row: number; end: 'pickup' | 'dropoff'} | null {
    const {show} = ctx.options;
    const ends: ('pickup' | 'dropoff')[] =
      show === 'pickups' ? ['pickup'] : show === 'dropoffs' ? ['dropoff'] : ['pickup', 'dropoff'];
    const reach = ctx.getMetersPerPixel() * HOVER_PIXELS;
    const [pointerX, pointerY] = trips.project(coordinate[0], coordinate[1]);
    let best: {row: number; end: 'pickup' | 'dropoff'} | null = null;
    let bestDistance = Infinity;
    for (const end of ends) {
      const candidates = getNearestIndex(end).within(coordinate, reach);
      const meters = end === 'pickup' ? trips.pickup : trips.dropoff;
      for (let item = 0; item < Math.min(candidates.length, 300); item++) {
        const row = candidates[item];
        if (!passesBrushes(row)) continue;
        const distance = Math.hypot(meters[row * 2] - pointerX, meters[row * 2 + 1] - pointerY);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = {row, end};
        }
        break;
      }
    }
    return best;
  }

  /** The class colour of a trip's mapped value, for the tooltip swatch. */
  function getSwatch(row: number): readonly [number, number, number, number] | undefined {
    const {colorBy, fareClasses} = ctx.options;
    const pick = (table: TaxiClassTables['fareQuantile'], value: number) =>
      table.colors[getClassIndexOf(table, value)] as [number, number, number, number];
    switch (colorBy) {
      case 'fare':
        return pick(
          fareClasses === 'equal' ? tables.fareEqual : tables.fareQuantile,
          trips.fare[row]
        );
      case 'distance':
        return pick(tables.distance, trips.distance[row]);
      case 'passengers':
        return PASSENGER_COLORS[trips.passengers[row] - 1] as [number, number, number, number];
      case 'hour':
        return getHourColor(trips.pickupHour[row] % 24);
      default:
        return undefined;
    }
  }

  function highlightBins(
    range: readonly [number, number],
    domain: readonly [number, number],
    bins: number
  ): number[] {
    if (!toBrush(range, domain)) return [];
    const width = (domain[1] - domain[0]) / bins;
    const indexes: number[] = [];
    for (let bin = 0; bin < bins; bin++) {
      const start = domain[0] + bin * width;
      if (start + width > range[0] && start < range[1]) indexes.push(bin);
    }
    return indexes;
  }

  function processSummary(bytes: ArrayBuffer): void {
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    let offset = 0;
    const take = (length: number) => {
      const start = offset;
      offset += length;
      return start;
    };
    const hourStart = take(HOUR_BINS);
    const distanceStart = take(DISTANCE_BINS);
    const fareStart = take(FARE_BINS);
    const paxStart = take(PASSENGER_GROUPS);
    const selectedStart = take(PASSENGER_GROUPS);
    const fareSumStart = take(PASSENGER_GROUPS);
    const distanceSumStart = take(PASSENGER_GROUPS);
    const countStart = take(1);
    const options = ctx.options;
    const selected = words[countStart];
    const shouldLink = selected < LINK_LIMIT;
    if (shouldLink !== linksOn) {
      linksOn = shouldLink;
      ctx.requestLayers();
    }

    const sum = (start: number, length: number) => {
      let total = 0;
      for (let index = start; index < start + length; index++) total += words[index];
      return total;
    };

    // Hour chart: bars take the cyclic ramp when the map is coloured by hour, the brush is a band.
    const hourBrushed = Boolean(toBrush(options.hours, HOUR_DOMAIN));
    ctx.setChart('hourChart', {
      kind: 'histogram',
      values: Array.from(words.subarray(hourStart, hourStart + HOUR_BINS)),
      xDomain: HOUR_DOMAIN,
      height: 110,
      xLabel: 'Pickup time (hour of Thu 1 Jan to Fri 2 Jan)',
      yLabel: 'trips',
      formatX: hourLabel,
      ...(options.colorBy === 'hour'
        ? {colors: hourBarColors}
        : {highlight: highlightBins(options.hours, HOUR_DOMAIN, HOUR_BINS)}),
      bands: hourBrushed
        ? [
            {
              from: Math.max(options.hours[0], HOUR_DOMAIN[0]),
              to: options.hours[1],
              tone: 'signal' as const
            }
          ]
        : [],
      description: 'Trips per pickup hour; the brushed hours are shaded.'
    });
    const distanceClassed = options.colorBy === 'distance';
    ctx.setChart('distanceChart', {
      kind: 'histogram',
      values: Array.from(words.subarray(distanceStart, distanceStart + DISTANCE_BINS)),
      xDomain: DISTANCE_DOMAIN,
      height: 110,
      xLabel: 'Trip distance (miles, longer trips off scale)',
      yLabel: 'trips',
      ...(distanceClassed
        ? {
            breaks: tables.distance.breaks.filter(value => value < DISTANCE_DOMAIN[1]),
            classColors: tables.distance.colors
          }
        : {highlight: highlightBins(options.distance, DISTANCE_DOMAIN, DISTANCE_BINS)}),
      bands: toBrush(options.distance, DISTANCE_DOMAIN)
        ? [
            {
              from: Math.max(options.distance[0], DISTANCE_DOMAIN[0]),
              to: options.distance[1],
              tone: 'signal' as const
            }
          ]
        : []
    });
    // Fare: 120 half-dollar bins from the GPU, summed four at a time into 30 two-dollar bars, with
    // the class edges of both tables so the failing and the good classification show together.
    const fareHalfDollars = Array.from(words.subarray(fareStart, fareStart + FARE_BINS));
    const fareBars = Array.from({length: FARE_CHART_BINS}, (_, bar) => {
      const group = FARE_BINS / FARE_CHART_BINS;
      let total = 0;
      for (let item = 0; item < group; item++) total += fareHalfDollars[bar * group + item];
      return total;
    });
    const equalEdges = tables.fareEqual.breaks.filter(value => value < FARE_DOMAIN[1]);
    ctx.setChart('fareChart', {
      kind: 'histogram',
      values: fareBars,
      xDomain: FARE_DOMAIN,
      height: 120,
      xLabel: 'Metered fare (USD, no tips, higher fares off scale)',
      yLabel: 'trips',
      formatX: value => `$${value.toFixed(0)}`,
      breaks: tables.fareQuantile.breaks.filter(value => value < FARE_DOMAIN[1]),
      classColors: tables.fareQuantile.colors,
      markers: equalEdges.map((value, index) => ({
        x: value,
        label: index === 0 ? 'equal-interval edges' : undefined
      })),
      description:
        'Trips by fare. Bars are coloured by the quantile classes; the vertical rules are the edges of six equal-width classes, which sit far to the right of nearly every trip.'
    });
    ctx.setChart('passengerChart', {
      kind: 'bars',
      values: Array.from(words.subarray(paxStart + 1, paxStart + PASSENGER_GROUPS)),
      labels: ['1', '2', '3', '4', '5', '6'],
      height: 100,
      xLabel: 'Passengers',
      yLabel: 'trips',
      ...(options.colorBy === 'passengers'
        ? {colors: PASSENGER_COLORS}
        : {
            highlight: toBrush(options.passengers, PASSENGER_DOMAIN)
              ? Array.from({length: 6}, (_, index) => index).filter(
                  index => index + 1 >= options.passengers[0] && index + 1 <= options.passengers[1]
                )
              : []
          })
    });

    // The selectivity funnel: each bar is the trips that pass every brush except one (the self-
    // excluding chart's own count), so the distance between a bar and the last is what that brush
    // alone removes.
    const exceptHours = sum(hourStart, HOUR_BINS);
    const exceptDistance = sum(distanceStart, DISTANCE_BINS);
    const exceptFare = sum(fareStart, FARE_BINS);
    const exceptParty = sum(paxStart + 1, PASSENGER_GROUPS - 1);
    ctx.setChart('funnelChart', {
      kind: 'bars',
      horizontal: true,
      values: [liveCount, exceptHours, exceptDistance, exceptFare, exceptParty, selected],
      labels: options.selfExclude
        ? [
            'All live trips',
            'All but hours',
            'All but distance',
            'All but fare',
            'All but party',
            'Every brush'
          ]
        : [
            'All live trips',
            'Hour chart',
            'Distance chart',
            'Fare chart',
            'Party chart',
            'Every brush'
          ],
      highlight: [5],
      xLabel: 'Trips passing',
      height: 150,
      description: options.selfExclude
        ? 'Trips that pass every brush except the one named: each chart is allowed to see all the others.'
        : 'With self-exclusion off every chart counts only what its own brush keeps, so the bars collapse onto the selection.'
    });

    let fareTotal = 0;
    let distanceTotal = 0;
    let passengerTotal = 0;
    for (let group = 1; group < PASSENGER_GROUPS; group++) {
      fareTotal += floats[fareSumStart + group];
      distanceTotal += floats[distanceSumStart + group];
      passengerTotal += words[selectedStart + group] * group;
    }
    const perMile = distanceTotal > 0 ? fareTotal / distanceTotal : 0;
    let modalFare = 0;
    let modalCount = 0;
    for (let bin = 0; bin < FARE_BINS; bin++) {
      if (fareHalfDollars[bin] > modalCount) {
        modalCount = fareHalfDollars[bin];
        modalFare = (bin * (FARE_DOMAIN[1] - FARE_DOMAIN[0])) / FARE_BINS;
      }
    }
    const fareBinned = exceptFare;
    ctx.setReadout('selected', selected);
    ctx.setReadout('share', selected / count);
    ctx.setReadout('fareShare', selected > 0 ? fareTotal / fareTotalAll : null);
    ctx.setReadout('fareMean', selected > 0 ? formatMoney(fareTotal / selected) : null);
    ctx.setReadout(
      'distanceMean',
      selected > 0 ? `${(distanceTotal / selected).toFixed(2)} mi` : null
    );
    ctx.setReadout(
      'perMileDelta',
      distanceTotal > 0 ? `${formatMoney(perMile)} vs ${formatMoney(perMileAll)} per mile` : null
    );
    ctx.setReadout('passengersMean', selected > 0 ? (passengerTotal / selected).toFixed(2) : null);
    ctx.setReadout(
      'modalFare',
      modalCount > 0
        ? `${formatMoney(modalFare)} (${formatPercent(modalCount / Math.max(1, fareBinned), 0)} of these trips)`
        : null
    );
    ctx.setReadout(
      'window',
      toBrush(options.hours, HOUR_DOMAIN)
        ? `${formatTaxiTime(options.hours[0])} to ${formatTaxiTime(options.hours[1])}`
        : 'all 38 hours'
    );

    // Finding notes read from the numbers above: the long trips at JFK, the box in the area brush.
    const notes: MapAnnotation[] = [];
    const longTrips = options.distance[0] >= 5 && options.distance[1] >= DISTANCE_DOMAIN[1];
    if (longTrips && selected > 0) {
      notes.push({
        kind: 'note',
        id: 'long-trips-note',
        coordinate: getPlaceCenter('jfk'),
        title: liveText('{share:percent} of trips, {money:percent} of fares', {
          share: selected / count,
          money: fareTotal / fareTotalAll
        }),
        text: 'long trips, drawn with their links'
      });
    }
    const area = cpuBrush.area;
    if (area && selected > 0) {
      const [centerLongitude, centerLatitude] = trips.unproject(
        (area[0] + area[2]) / 2,
        (area[1] + area[3]) / 2
      );
      notes.push({
        kind: 'note',
        id: 'area-note',
        coordinate: [centerLongitude, centerLatitude],
        title: liveText('{n:integer} pickups inside', {n: selected}),
        text: `mean fare ${formatMoney(fareTotal / selected)}, ${
          nearestPlaceLabel(NYC, [centerLongitude, centerLatitude], {maxDistanceMeters: 6000}) ??
          'New York'
        }`
      });
    }
    ctx.setAnnotations('findings', notes.length ? notes : null);
  }

  function toBoundsPoint(event: ScenePointerEvent): [number, number] | null {
    if (!event.coordinate) return null;
    return trips.project(event.coordinate[0], event.coordinate[1]);
  }

  /** Share of trips whose pickup is in the visible map, counted once the camera settles. */
  function countInView(bounds: Bounds): void {
    let inside = 0;
    for (let row = 0; row < count; row++) {
      const x = trips.pickup[row * 2];
      const y = trips.pickup[row * 2 + 1];
      if (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3]) inside++;
    }
    ctx.setReadout('inView', inside / count);
  }

  ctx.setReadout('rows', count);
  ctx.setReadout('views', '5 dimensions, 9 views');
  ctx.setReadout('readbackBytes', readbackBytes);
  ctx.setReadout('gpuBytes', gpuBytes);
  ctx.setReadout('equalMajor', equalMajorShare);
  ctx.setReadout('quantileMajor', quantileMajorShare);
  writeLiveMask();
  rebuild();

  return {
    getCompiledGraphs: () => (graph ? [graph.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id) {
      switch (id) {
        case 'selfExclude':
          rebuild();
          break;
        case 'day':
          writeLiveMask();
          break;
        case 'hours':
        case 'distance':
        case 'fare':
        case 'passengers':
        case 'area':
          applyBrushes();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'clearAll') {
        mapBrush = null;
        ctx.setOptions(
          {
            hours: [HOUR_DOMAIN[0], HOUR_DOMAIN[1]],
            distance: [DISTANCE_DOMAIN[0], DISTANCE_DOMAIN[1]],
            fare: [FARE_DOMAIN[0], FARE_DOMAIN[1]],
            passengers: [PASSENGER_DOMAIN[0], PASSENGER_DOMAIN[1]],
            area: 'none',
            day: 'all'
          },
          {notify: true}
        );
        // Notified writes call `setOption`, which rewrites the brushes, the live mask and the layers.
        graph?.filter.clearAll();
        applyBrushes();
        ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    // The story sits on the night ground in both themes, so a ground change only repaints.
    onGroundChange() {
      ctx.requestLayers();
    },

    onDragStart(event) {
      if (!ctx.options.brushMap && !event.shiftKey) return false;
      const point = toBoundsPoint(event);
      if (!point) return false;
      dragStart = point;
      mapBrush = [point[0], point[1], point[0], point[1]];
      ctx.setOptions({area: 'drawn'}, {notify: true});
      applyBrushes();
      return true;
    },
    onDrag(event) {
      const point = toBoundsPoint(event);
      if (!point || !dragStart) return;
      mapBrush = [
        Math.min(dragStart[0], point[0]),
        Math.min(dragStart[1], point[1]),
        Math.max(dragStart[0], point[0]),
        Math.max(dragStart[1], point[1])
      ];
      applyBrushes();
    },
    onDragEnd() {
      dragStart = null;
      markChanged();
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate || dragStart) return null;
      const found = findTrip(event.coordinate);
      if (!found) return null;
      const {row, end} = found;
      const fare = trips.fare[row];
      const distance = trips.distance[row];
      const {colorBy} = ctx.options;
      const swatch = getSwatch(row);
      const rows: TooltipRow[] = [];
      if (swatch && colorBy !== 'none') {
        const label =
          colorBy === 'fare'
            ? {label: 'Fare', value: formatMoney(fare)}
            : colorBy === 'distance'
              ? {label: 'Distance', value: distance.toFixed(1), unit: 'mi'}
              : colorBy === 'passengers'
                ? {label: 'Passengers', value: trips.passengers[row]}
                : {label: 'Hour of day', value: formatTaxiTime(trips.pickupHour[row]).slice(-5)};
        rows.push({...label, swatch, emphasis: true});
      }
      if (colorBy !== 'fare') rows.push({label: 'Fare', value: formatMoney(fare), unit: 'no tip'});
      if (colorBy !== 'distance') {
        rows.push({label: 'Distance', value: distance.toFixed(1), unit: 'mi'});
      }
      if (colorBy !== 'passengers') {
        rows.push({label: 'Passengers', value: trips.passengers[row]});
      }
      if (distance > 0) {
        rows.push({label: 'Fare per mile', value: formatMoney(fare / distance)});
      }
      rows.push({
        label: 'Drop-off',
        value: formatTaxiTime(trips.dropoffHour[row]).slice(-5),
        unit: 'routed time'
      });
      const meters = end === 'pickup' ? trips.pickup : trips.dropoff;
      const lngLat = trips.unproject(meters[row * 2], meters[row * 2 + 1]) as LngLat;
      return {
        title: `${end === 'pickup' ? 'Pickup' : 'Drop-off'} ${formatTaxiTime(
          end === 'pickup' ? trips.pickupHour[row] : trips.dropoffHour[row]
        )}`,
        subtitle: nearestPlaceLabel(NYC, lngLat, {maxDistanceMeters: 6000}) ?? undefined,
        rows,
        highlight: {kind: 'point', coordinate: lngLat, radiusPixels: 6}
      };
    },

    encode(commandEncoder, frame) {
      if (!graph) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].graph.resources.destroy();
          retired.splice(index, 1);
        }
      }
      const bounds = getViewportMetricBounds(frame.viewport, projection);
      if (!viewBounds || bounds.some((value, index) => value !== viewBounds![index])) {
        viewChangedAt = performance.now();
        viewStale = true;
      }
      viewBounds = bounds;
      if (viewStale && performance.now() - viewChangedAt > SETTLE_MILLISECONDS) {
        viewStale = false;
        countInView(bounds);
      }
      if (encodeFrames > 0) {
        graph.compiled.encode(commandEncoder, {parameters: undefined});
        encodeFrames--;
        graph.reader.flush(commandEncoder);
        return;
      }
      if (summaryStale && !graph.reader.isPending) {
        summaryStale = false;
        graph.reader.request(commandEncoder);
      } else {
        graph.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const layers: Layer[] = [];
      const showPickups = options.show !== 'dropoffs';
      const showDropoffs = options.show !== 'pickups';
      const both = options.show === 'both';
      const additive = options.blending === 'additive';
      const scaled = (stops: readonly (readonly [number, number])[]) =>
        stops.map(([zoom, radius]) => [zoom, radius * options.pointScale] as [number, number]);

      // Context: the rows a brush removed, a faint ghost that never takes a ramp colour.
      if (options.showFiltered && isFiltering()) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'taxi-all-points',
            coordinateOrigin,
            positions: options.show === 'dropoffs' ? dropoffPositions : pickupPositions,
            instanceCount: count,
            radiusPixels: scaled(GHOST_RADIUS_STOPS),
            blending: 'additive',
            color: [GHOST_INK[0], GHOST_INK[1], GHOST_INK[2], GHOST_ALPHA]
          })
        );
      }
      // Supporting input: straight trip links for small selections, ends drawn over them.
      if (both && options.showLinks && linksOn) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-trip-links',
            coordinateOrigin,
            segments: tripSegments,
            ids: visibleIds,
            drawCommands: filteredDraw,
            widthPixels: 0.6,
            blending: 'additive',
            color: [LINK_INK[0], LINK_INK[1], LINK_INK[2], LINK_INK[3] ?? 26]
          })
        );
      }

      const common = {
        coordinateOrigin,
        ids: visibleIds,
        drawCommands: filteredDraw,
        radiusPixels: scaled(SELECTED_RADIUS_STOPS),
        blending: options.blending,
        opacity: additive ? 0.4 : 0.6
      };
      const colorProps = (side?: 'equal' | 'quantile') => {
        switch (options.colorBy) {
          case 'fare': {
            const table =
              side === 'equal' || (!side && options.fareClasses === 'equal')
                ? tables.fareEqual
                : tables.fareQuantile;
            return {
              values: fareBuffer,
              valueFormat: 'float32' as const,
              colormap: 'uniform' as const,
              ...getClassTableLayerProps(table)
            };
          }
          case 'distance':
            return {
              values: distanceBuffer,
              valueFormat: 'float32' as const,
              colormap: 'uniform' as const,
              ...getClassTableLayerProps(tables.distance)
            };
          case 'passengers':
            return {
              values: passengerBuffer,
              valueFormat: 'float32' as const,
              colormap: 'uniform' as const,
              ...getClassTableLayerProps(tables.passengers)
            };
          case 'hour':
            return {
              values: hourOfDayBuffer,
              valueFormat: 'float32' as const,
              colormap: 'romao' as const,
              valueRange: [0, 24] as const
            };
          default:
            return {color: PICKUP_INK.dark};
        }
      };
      if (showDropoffs) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'taxi-visible-dropoffs',
            ...common,
            positions: dropoffPositions,
            ...(both ? {color: DROPOFF_INK.dark} : colorProps())
          })
        );
      }
      if (showPickups) {
        const swipe = !both && options.colorBy === 'fare' && options.fareClasses === 'swipe';
        if (swipe) {
          for (const side of ['equal', 'quantile'] as const) {
            layers.push(
              new SpatialAnalysisPointLayer({
                id: `taxi-visible-pickups-${side}`,
                ...common,
                positions: pickupPositions,
                ...colorProps(side),
                compareSide: side === 'equal' ? 'a' : 'b'
              })
            );
          }
        } else {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'taxi-visible-pickups',
              ...common,
              positions: pickupPositions,
              ...(both ? {color: PICKUP_INK.dark} : colorProps())
            })
          );
        }
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      graph?.reader.stop();
      graph?.resources.destroy();
      for (const entry of retired) entry.graph.resources.destroy();
      resources.destroy();
    }
  };
}
