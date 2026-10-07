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
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';
import {
  DISTANCE_DOMAIN,
  FARE_DOMAIN,
  formatTaxiTime,
  HOUR_DOMAIN,
  loadTaxiTrips,
  AREA_PRESETS,
  NYC_TAXI_ORIGIN,
  PASSENGER_DOMAIN,
  TAXI_COLOR_RANGES,
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
  colorBy: 'fare' | 'distance' | 'time' | 'passengers';
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  pointSize: number;
  opacity: number;
  showFiltered: boolean;
};

const HOUR_BINS = 39;
const DISTANCE_BINS = 30;
const FARE_BINS = 30;
/** Dense group keys 0 to 6 (0 never occurs). */
const PASSENGER_GROUPS = 7;
/** Frames a replaced graph stays alive so in-flight frames never touch destroyed buffers. */
const RETIRE_FRAMES = 4;
/** Value that stands for "no bound" on a brush that sits at the end of its slider. */
const UNBOUNDED = 1e9;

type Bounds = [number, number, number, number];

type FilterGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  filter: GPUCrossfilter<void>;
  selfExclude: boolean;
  reader: SummaryReader;
  resources: SpatialAnalysisResources;
};

/** Returns an inclusive or half-open brush, or `null` when the handles cover the whole slider. */
function toBrush(
  range: readonly [number, number],
  domain: readonly [number, number]
): [number, number] | null {
  const [low, high] = range;
  if (low <= domain[0] && high >= domain[1]) return null;
  return [low <= domain[0] ? -UNBOUNDED : low, high >= domain[1] ? UNBOUNDED : high];
}

/**
 * Million-row-class crossfilter over 440,000 taxi trips. One `GPUCrossfilter` controller holds a
 * map rectangle and four scalar brushes; linked histograms, group statistics, a count and a
 * compact list of visible trip ids are all compute passes over the same buffers. A brush only
 * rewrites a five-word selection state; the graph is encoded again and a few hundred numbers come
 * back for the charts. The map draws the visible ids straight from GPU storage with an indirect count.
 */
export async function createNycTaxiCrossfilter(
  ctx: SceneContext<NycTaxiCrossfilterOptions>
): Promise<SceneInstance<NycTaxiCrossfilterOptions>> {
  const {device} = ctx;
  const trips: TaxiTrips = loadTaxiTrips(ctx.datasets.get('poopdeck-nyc-taxi'));
  const count = trips.count;
  const resources = new SpatialAnalysisResources(device, 'taxi-crossfilter');
  const coordinateOrigin: [number, number, number] = [NYC_TAXI_ORIGIN[0], NYC_TAXI_ORIGIN[1], 0];

  // ---- Source columns, uploaded once ------------------------------------------------------------
  const pickupX = new Float32Array(count);
  const pickupY = new Float32Array(count);
  for (let row = 0; row < count; row++) {
    pickupX[row] = trips.pickup[row * 2];
    pickupY[row] = trips.pickup[row * 2 + 1];
  }
  const pickupXBuffer = resources.createBuffer('pickup-x', pickupX);
  const pickupYBuffer = resources.createBuffer('pickup-y', pickupY);
  const pickupPositions = resources.createBuffer('pickup-xy', trips.pickup);
  const dropoffPositions = resources.createBuffer('dropoff-xy', trips.dropoff);
  const hourBuffer = resources.createBuffer('hour', trips.pickupHour);
  const distanceBuffer = resources.createBuffer('distance', trips.distance);
  const fareBuffer = resources.createBuffer('fare', trips.fare);
  const passengerValues = Float32Array.from(trips.passengers);
  const passengerBuffer = resources.createBuffer('passengers', passengerValues);
  const passengerKeys = resources.createBuffer('passenger-keys', trips.passengers);
  const liveMaskBuffer = resources.createBuffer('live-mask', new Uint32Array(count).fill(1));
  const visibleIds = resources.createBuffer('visible-ids', count * 4);
  const filteredDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'taxi-crossfilter-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const brushOutline = resources.createBuffer('brush-outline', 16 * 4);

  // ---- Linked-view outputs (same buffers for every graph rebuild) -------------------------------
  const hourBins = resources.createBuffer('hour-bins', HOUR_BINS * 4);
  const distanceBins = resources.createBuffer('distance-bins', DISTANCE_BINS * 4);
  const fareBins = resources.createBuffer('fare-bins', FARE_BINS * 4);
  const passengerCounts = resources.createBuffer('passenger-counts', PASSENGER_GROUPS * 4);
  const selectedPassengers = resources.createBuffer('selected-passengers', PASSENGER_GROUPS * 4);
  const fareSums = resources.createBuffer('fare-sums', PASSENGER_GROUPS * 4);
  const distanceSums = resources.createBuffer('distance-sums', PASSENGER_GROUPS * 4);
  const selectedCount = resources.createBuffer('selected-count', 4);

  let destroyed = false;
  let graph: FilterGraph | null = null;
  let serial = 0;
  let encodeFrames = 3;
  let summaryStale = true;
  let mapBrush: Bounds | null = null;
  let dragStart: [number, number] | null = null;
  const retired: {graph: FilterGraph; frames: number}[] = [];

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
      [
        {buffer: hourBins, size: HOUR_BINS * 4},
        {buffer: distanceBins, size: DISTANCE_BINS * 4},
        {buffer: fareBins, size: FARE_BINS * 4},
        {buffer: passengerCounts, size: PASSENGER_GROUPS * 4},
        {buffer: selectedPassengers, size: PASSENGER_GROUPS * 4},
        {buffer: fareSums, size: PASSENGER_GROUPS * 4},
        {buffer: distanceSums, size: PASSENGER_GROUPS * 4},
        {buffer: selectedCount, size: 4}
      ],
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
    const preset = AREA_PRESETS[area];
    const [centerX, centerY] = trips.project(preset.center[0], preset.center[1]);
    return [
      centerX - preset.halfWidth,
      centerY - preset.halfHeight,
      centerX + preset.halfWidth,
      centerY + preset.halfHeight
    ];
  }

  /** Writes every brush into the controller's five-word selection states. */
  function applyBrushes(): void {
    if (!graph) return;
    const {filter} = graph;
    const options = ctx.options;
    const set = (dimension: string, brush: [number, number] | null) =>
      brush ? filter.setRange(dimension, brush) : filter.clear(dimension);
    set('hour', toBrush(options.hours, HOUR_DOMAIN));
    set('distance', toBrush(options.distance, DISTANCE_DOMAIN));
    set('fare', toBrush(options.fare, FARE_DOMAIN));
    set('passengers', toBrush(options.passengers, PASSENGER_DOMAIN));
    const area = getAreaBounds();
    if (area) filter.setBounds('map', area);
    else filter.clear('map');
    const outline = new Float32Array(16);
    if (area) {
      const [minX, minY, maxX, maxY] = area;
      outline.set([
        minX,
        minY,
        maxX,
        minY,
        maxX,
        minY,
        maxX,
        maxY,
        maxX,
        maxY,
        minX,
        maxY,
        minX,
        maxY,
        minX,
        minY
      ]);
    }
    brushOutline.write(outline);
    markChanged();
  }

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 2);
    summaryStale = true;
  }

  function writeLiveMask(): void {
    const {day} = ctx.options;
    const mask = new Uint32Array(count);
    for (let row = 0; row < count; row++) {
      const first = trips.pickupHour[row] < 24;
      mask[row] = day === 'all' || (day === 'jan1') === first ? 1 : 0;
    }
    liveMaskBuffer.write(mask);
    markChanged();
  }

  function rebuild(): void {
    if (graph) retired.push({graph, frames: 0});
    graph = buildGraph(ctx.options.selfExclude);
    applyBrushes();
  }

  const hourLabel = (hours: number) =>
    `${hours < 24 ? 'Thu' : 'Fri'} ${String(Math.floor(hours) % 24).padStart(2, '0')}h`;

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

    ctx.setChart('hourChart', {
      kind: 'histogram',
      values: Array.from(words.subarray(hourStart, hourStart + HOUR_BINS)),
      xDomain: HOUR_DOMAIN,
      height: 110,
      xLabel: 'Pickup time (hour of Thu 1 Jan to Fri 2 Jan)',
      yLabel: 'trips',
      formatX: hourLabel,
      highlight: highlightBins(options.hours, HOUR_DOMAIN, HOUR_BINS),
      description: 'Trips per pickup hour, with the brushed hours highlighted.'
    });
    ctx.setChart('distanceChart', {
      kind: 'histogram',
      values: Array.from(words.subarray(distanceStart, distanceStart + DISTANCE_BINS)),
      xDomain: DISTANCE_DOMAIN,
      height: 110,
      xLabel: 'Trip distance (miles, longer trips off scale)',
      yLabel: 'trips',
      highlight: highlightBins(options.distance, DISTANCE_DOMAIN, DISTANCE_BINS)
    });
    ctx.setChart('fareChart', {
      kind: 'histogram',
      values: Array.from(words.subarray(fareStart, fareStart + FARE_BINS)),
      xDomain: FARE_DOMAIN,
      height: 110,
      xLabel: 'Metered fare (USD, no tips, higher fares off scale)',
      yLabel: 'trips',
      formatX: value => `$${value.toFixed(0)}`,
      highlight: highlightBins(options.fare, FARE_DOMAIN, FARE_BINS)
    });
    const passengerLabels = ['1', '2', '3', '4', '5', '6'];
    const passengerHighlight: number[] = [];
    for (let group = 1; group < PASSENGER_GROUPS; group++) {
      if (group >= options.passengers[0] && group <= options.passengers[1]) {
        passengerHighlight.push(group - 1);
      }
    }
    ctx.setChart('passengerChart', {
      kind: 'bars',
      values: Array.from(words.subarray(paxStart + 1, paxStart + PASSENGER_GROUPS)),
      labels: passengerLabels,
      height: 100,
      xLabel: 'Passengers',
      yLabel: 'trips',
      highlight: toBrush(options.passengers, PASSENGER_DOMAIN) ? passengerHighlight : []
    });

    let fareTotal = 0;
    let distanceTotal = 0;
    let passengerTotal = 0;
    const meanFareByParty: number[] = [];
    for (let group = 1; group < PASSENGER_GROUPS; group++) {
      const groupCount = words[selectedStart + group];
      fareTotal += floats[fareSumStart + group];
      distanceTotal += floats[distanceSumStart + group];
      passengerTotal += groupCount * group;
      meanFareByParty.push(groupCount > 0 ? floats[fareSumStart + group] / groupCount : 0);
    }
    ctx.setChart('partyFareChart', {
      kind: 'bars',
      values: meanFareByParty,
      labels: passengerLabels,
      height: 100,
      xLabel: 'Passengers',
      yLabel: 'mean fare (USD)',
      color: 2
    });
    ctx.setReadout('selected', selected);
    ctx.setReadout('share', selected / count);
    ctx.setReadout('fareTotal', selected > 0 ? `$${formatCount(fareTotal)}` : null);
    ctx.setReadout('fareMean', selected > 0 ? `$${(fareTotal / selected).toFixed(2)}` : null);
    ctx.setReadout(
      'distanceMean',
      selected > 0 ? `${(distanceTotal / selected).toFixed(2)} mi` : null
    );
    ctx.setReadout(
      'perMile',
      distanceTotal > 0 ? `$${(fareTotal / distanceTotal).toFixed(2)} per mile` : null
    );
    ctx.setReadout('passengersMean', selected > 0 ? (passengerTotal / selected).toFixed(2) : null);
    ctx.setReadout(
      'window',
      toBrush(options.hours, HOUR_DOMAIN)
        ? `${formatTaxiTime(options.hours[0])} to ${formatTaxiTime(options.hours[1])}`
        : 'all 38 hours'
    );
  }

  function toBoundsPoint(event: ScenePointerEvent): [number, number] | null {
    if (!event.coordinate) return null;
    return trips.project(event.coordinate[0], event.coordinate[1]);
  }

  function updateDraw(): void {
    markChanged();
    ctx.requestLayers();
  }

  ctx.setReadout('rows', count);
  ctx.setReadout('dimensions', '5 (map rectangle plus 4 ranges)');
  ctx.setReadout('views', '9 (3 histograms, 4 groups, count, visible ids)');
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
      updateDraw();
    },

    getTooltip() {
      return null;
    },

    encode(commandEncoder, _frame) {
      if (!graph) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].graph.resources.destroy();
          retired.splice(index, 1);
        }
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
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const showPickups = options.show !== 'dropoffs';
      const showDropoffs = options.show !== 'pickups';
      if (options.showFiltered) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'taxi-all-points',
            coordinateOrigin,
            positions: options.show === 'dropoffs' ? dropoffPositions : pickupPositions,
            instanceCount: count,
            radiusPixels: 0.8,
            color: dark ? [150, 160, 185, 22] : [70, 80, 110, 28]
          })
        );
      }
      const valuesByColor = {
        fare: fareBuffer,
        distance: distanceBuffer,
        time: hourBuffer,
        passengers: passengerBuffer
      };
      const colored = options.show !== 'both';
      const common = {
        coordinateOrigin,
        ids: visibleIds,
        drawCommands: filteredDraw,
        radiusPixels: options.pointSize,
        opacity: options.opacity
      };
      if (showPickups) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'taxi-visible-pickups',
            ...common,
            positions: pickupPositions,
            ...(colored
              ? {
                  values: valuesByColor[options.colorBy],
                  valueFormat: 'float32' as const,
                  colormap: options.ramp,
                  valueRange: TAXI_COLOR_RANGES[options.colorBy] as unknown as [number, number]
                }
              : {color: [255, 170, 60, 255] as [number, number, number, number]})
          })
        );
      }
      if (showDropoffs) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'taxi-visible-dropoffs',
            ...common,
            positions: dropoffPositions,
            ...(colored
              ? {
                  values: valuesByColor[options.colorBy],
                  valueFormat: 'float32' as const,
                  colormap: options.ramp,
                  valueRange: TAXI_COLOR_RANGES[options.colorBy] as unknown as [number, number]
                }
              : {color: [70, 215, 255, 255] as [number, number, number, number]})
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'taxi-map-brush',
          coordinateOrigin,
          segments: brushOutline,
          instanceCount: getAreaBounds() ? 4 : 0,
          widthPixels: 2,
          color: dark ? [255, 255, 255, 230] : [20, 30, 60, 230]
        })
      );
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
