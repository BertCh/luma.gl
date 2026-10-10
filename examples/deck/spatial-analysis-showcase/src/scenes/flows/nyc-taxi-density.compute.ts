// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  createGPUPointDensityGaussianKernel,
  GPUPointDensity
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassIndexOf, getClassTableLayerProps} from '../../cartography/class-table';
import {nearestPlaceLabel, NYC} from '../../cartography/gazetteer';
import {formatCount, formatDistance, liveText} from '../../cartography/live-text';
import type {ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {sampleRamp} from '../../engine/ramps';
import {getViewportMetricBounds, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipRow} from '../scene';
import {
  buildDensityReference,
  DENSITY_RAMP,
  DENSITY_RAMP_RANGE,
  DROPOFF_DENSITY_RAMP,
  type DensityReference,
  formatDensity,
  formatWindowTime,
  makeMeanFareTable,
  makeQuantileDensityTable,
  WINDOW_HOURS
} from './nyc-taxi-density-style';
import {
  countBins,
  loadTaxiTrips,
  NYC_TAXI_HOURS,
  NYC_TAXI_ORIGIN,
  type TaxiTrips
} from './nyc-taxi-data';

/** Option state of the nyc-taxi-density scene. */
export type NycTaxiDensityOptions = {
  /** `rides`: trips per km2; `fare`: the mean fare of the rides that start in a cell. */
  measure: 'rides' | 'fare';
  show: 'pickups' | 'dropoffs';
  /** A swipe of two versions of the same field: two stretches, or pickups against drop-offs. */
  swipe: 'off' | 'stretch' | 'ends';
  scale: 'linear' | 'sqrt' | 'quantile';
  binning: 'grid' | 'hexagon';
  resolution: 'coarse' | 'medium' | 'fine';
  smoothing: 'off' | 'gaussian';
  sigma: number;
  hours: readonly [number, number];
  minCount: number;
  showGrid: boolean;
  opacity: number;
};

type GraphKind = 'rides' | 'mean' | 'tally';
type Side = 'pickups' | 'dropoffs';
type GraphSpec = {kind: GraphKind; side: Side};

const GRID_SIZES = {coarse: [70, 44], medium: [110, 70], fine: [170, 106]} as const;
const HEXAGON_SIZES = {coarse: [34, 26], medium: [52, 38], fine: [80, 58]} as const;
const SQRT3 = Math.sqrt(3);
/** Compile-time kernel; per-frame weights select the actual sigma. */
const KERNEL_RADIUS = 8;
const KERNEL_WIDTH = KERNEL_RADIUS * 2 + 1;
const SETTLE_MILLISECONDS = 500;
const HOURS = 39;
/** Smoothed cells below this many rides are not drawn (as in the nature story). */
const SMOOTHED_FLOOR = 0.3;
/** Bars of the legend strip and of the cell chart. */
const LEGEND_BINS = 16;
const CHART_BINS = 24;
/** Local maxima labelled on the map, and their separation in cells. */
const PEAK_COUNT = 3;
const PEAK_SEPARATION_CELLS = 9;
/** Cells smaller than this on screen are not outlined. */
const MINIMUM_OUTLINE_PIXELS = 10;
/** Cells at most this far (in cells) from a place are searched for its mean fare. */
const PLACE_SEARCH_CELLS = 4;
const GRID_LINE_CAPACITY = (GRID_SIZES.fine[0] + 1 + GRID_SIZES.fine[1] + 1) * 16;

type DensityGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  gridSize: readonly [number, number];
  values: Buffer;
  reader: SummaryReader;
};

/** One read-back field: the cell values, the raw ride counts and what the camera looked at. */
type FieldSnapshot = {
  values: Float32Array;
  counts: Uint32Array;
  extentMaximum: number;
  gridSize: readonly [number, number];
  bounds: Float32Array;
  binning: 'grid' | 'hexagon';
  /** Ground area of one cell, km2. */
  cellAreaKm2: number;
};

const formatMoney = (value: number) => `$${value.toFixed(2)}`;

/**
 * Pickup and drop-off density of 440,000 taxi trips. The pickups and drop-offs are one 880,000-point
 * buffer with two masks (which end, which hours); the grid follows the camera, the kernel is a
 * weight buffer, and the lattice, the resolution, the side and the statistic select between lazily
 * compiled graphs. Every colour scale is written in trips per km2, so a cell that gets smaller as
 * you zoom does not change the colours; the scales are read from the data once at load.
 */
export async function createNycTaxiDensity(
  ctx: SceneContext<NycTaxiDensityOptions>
): Promise<SceneInstance<NycTaxiDensityOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get('poopdeck-nyc-taxi');
  const trips: TaxiTrips = loadTaxiTrips(dataset);
  const count = trips.count;
  const pointCount = count * 2;
  const projection = dataset.getProjection(NYC_TAXI_ORIGIN);
  const resources = new SpatialAnalysisResources(device, 'taxi-density');
  const coordinateOrigin: [number, number, number] = [NYC_TAXI_ORIGIN[0], NYC_TAXI_ORIGIN[1], 0];
  const reference: DensityReference = buildDensityReference(trips);
  ctx.setLegendData('clip', reference.fullClip);

  const positions = new Float32Array(pointCount * 2);
  positions.set(trips.pickup, 0);
  positions.set(trips.dropoff, count * 2);
  const positionsBuffer = resources.createBuffer('positions', positions);
  const pickupMask = resources.createBuffer('mask-pickups', new Uint32Array(pointCount).fill(1));
  const dropoffMask = resources.createBuffer('mask-dropoffs', new Uint32Array(pointCount).fill(1));
  const fareWeights = new Float32Array(pointCount);
  fareWeights.set(trips.fare, 0);
  fareWeights.set(trips.fare, count);
  const weightsBuffer = resources.createBuffer('fare-weights', fareWeights);
  const bounds = resources.createParameterBuffer('bounds', 'float32', 4);
  const hexagonRadius = resources.createParameterBuffer(
    'hexagon-radius',
    'float32',
    1,
    Float32Array.of(300)
  );
  const kernel = resources.createParameterBuffer('kernel', 'float32', KERNEL_WIDTH * KERNEL_WIDTH);
  const gridLines = resources.createBuffer('grid-lines', GRID_LINE_CAPACITY);

  let destroyed = false;
  /** True when analysis inputs changed and the selected density graphs need another encoding. */
  let densityDirty = true;
  /** Grid geometry is CPU-derived and can change without invalidating the density field. */
  let gridDirty = true;
  let settleStale = true;
  let lastChangeTime = performance.now();
  let lastBounds: Float32Array | null = null;
  /** Bounds and cell area of the frame whose field the readers copied. */
  let readBounds: Float32Array | null = null;
  let readAreaKm2 = 1;
  /** Cell of the current frame (follows the camera). */
  let cellAreaKm2 = 1;
  let cellWidthMeters = 1;
  let cellHeightMeters = 1;
  let lastLayerArea = 0;
  let gridLineCount = 0;
  let includedPickups = count;
  let includedDropoffs = count;
  let hoverIndex = -1;
  const graphs = new Map<string, DensityGraph>();
  let primary: DensityGraph | null = null;
  let secondary: DensityGraph | null = null;
  const snapshots: {primary: FieldSnapshot | null; secondary: FieldSnapshot | null} = {
    primary: null,
    secondary: null
  };
  let fareTable: ClassTable = makeMeanFareTable(
    reference.meanFareBreaks,
    ctx.ground(),
    ctx.options.minCount
  );
  ctx.setLegendData('fareTable', fareTable);

  // CPU context chart: trips picked up and dropped off per hour, independent of every option.
  const pickupsPerHour = countBins(trips.pickupHour, 0, HOURS, HOURS);
  const dropoffsPerHour = countBins(trips.dropoffHour, 0, HOURS, HOURS);

  const markChanged = () => {
    densityDirty = true;
    settleStale = true;
    lastChangeTime = performance.now();
  };

  // ---- Which graphs the current options need ------------------------------------------------------
  function getSpecs(options: NycTaxiDensityOptions): {
    primary: GraphSpec;
    secondary: GraphSpec | null;
  } {
    if (options.measure === 'fare') {
      return {
        primary: {kind: 'mean', side: 'pickups'},
        secondary: {kind: 'tally', side: 'pickups'}
      };
    }
    if (options.swipe === 'ends') {
      return {
        primary: {kind: 'rides', side: 'pickups'},
        secondary: {kind: 'rides', side: 'dropoffs'}
      };
    }
    return {primary: {kind: 'rides', side: options.show}, secondary: null};
  }

  const getKey = (options: NycTaxiDensityOptions, spec: GraphSpec) =>
    `${options.binning}:${options.resolution}:${spec.side}:${spec.kind}`;

  function buildGraph(options: NycTaxiDensityOptions, spec: GraphSpec): DensityGraph {
    const id = getKey(options, spec).replace(/:/g, '-');
    const gridSize =
      options.binning === 'grid'
        ? GRID_SIZES[options.resolution]
        : HEXAGON_SIZES[options.resolution];
    const cellCount = gridSize[0] * gridSize[1];
    const values = resources.createBuffer(`${id}-values`, cellCount * 4);
    const counts = resources.createBuffer(`${id}-counts`, cellCount * 4);
    const extent = resources.createBuffer(`${id}-extent`, 8);
    const graph = new GPUCommandGraph<void>(device, {id: `taxi-density-${id}`});
    const weighted = spec.kind === 'mean';
    // Only the ride field is blurred: a mean of blurred means, or a blurred tally, is not a rate.
    const smoothed = spec.kind === 'rides' && options.binning === 'grid';
    graph.add(
      new GPUPointDensity({
        id: 'density',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', pointCount),
        mask: importGraphBuffer(
          graph,
          'mask',
          spec.side === 'pickups' ? pickupMask : dropoffMask,
          'uint32',
          pointCount
        ),
        ...(weighted
          ? {weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', pointCount)}
          : {}),
        bounds: bounds.importToGraph(graph),
        gridSize,
        binning: options.binning,
        ...(options.binning === 'hexagon'
          ? {hexagonRadius: hexagonRadius.importToGraph(graph)}
          : smoothed
            ? {
                smoothing: {
                  kernel: kernel.importToGraph(graph),
                  kernelWidth: KERNEL_WIDTH,
                  kernelHeight: KERNEL_WIDTH,
                  strategy: 'direct'
                }
              }
            : {}),
        statistic: weighted ? 'mean' : 'count',
        output: {
          values: importGraphBuffer(graph, 'values', values, 'float32', cellCount),
          counts: importGraphBuffer(graph, 'counts', counts, 'uint32', cellCount),
          extent: importGraphBuffer(graph, 'extent', extent, 'float32', 2)
        }
      })
    );
    const built: DensityGraph = {
      compiled: resources.track(graph.compile()),
      gridSize,
      values,
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      resources,
      `${id}-field`,
      [
        {buffer: values, size: cellCount * 4},
        {buffer: counts, size: cellCount * 4},
        {buffer: extent, size: 8}
      ],
      bytes => {
        if (destroyed || !readBounds) return;
        const snapshot = parseField(bytes, gridSize, options.binning);
        if (built === primary) snapshots.primary = snapshot;
        else if (built === secondary) snapshots.secondary = snapshot;
        else return;
        processFields();
      }
    );
    return built;
  }

  function parseField(
    bytes: ArrayBuffer,
    gridSize: readonly [number, number],
    binning: 'grid' | 'hexagon'
  ): FieldSnapshot {
    const cells = gridSize[0] * gridSize[1];
    return {
      values: new Float32Array(bytes.slice(0, cells * 4)),
      counts: new Uint32Array(bytes.slice(cells * 4, cells * 8)),
      extentMaximum: new Float32Array(bytes, cells * 8, 2)[1],
      gridSize,
      bounds: readBounds ?? new Float32Array(4),
      binning,
      cellAreaKm2: readAreaKm2
    };
  }

  function selectGraphs(): void {
    const options = ctx.options;
    const specs = getSpecs(options);
    const fetch = (spec: GraphSpec) => {
      const key = getKey(options, spec);
      let graph = graphs.get(key);
      if (!graph) {
        graph = buildGraph(options, spec);
        graphs.set(key, graph);
      }
      return graph;
    };
    primary = fetch(specs.primary);
    secondary = specs.secondary ? fetch(specs.secondary) : null;
    snapshots.primary = null;
    snapshots.secondary = null;
    ctx.setAnnotations('peaks', null);
    const [columns, rows] = primary.gridSize;
    ctx.setReadout(
      'grid',
      `${columns} × ${rows} ${options.binning === 'hexagon' ? 'hexagons' : 'cells'}`
    );
    gridDirty = true;
    publishCost();
    markChanged();
  }

  /** True when the single field on the map is the drop-offs. */
  const showsDropoffs = () =>
    ctx.options.measure === 'rides' &&
    ctx.options.show === 'dropoffs' &&
    ctx.options.swipe !== 'ends';

  function publishCost(): void {
    if (!primary) return;
    ctx.setCost({
      records: showsDropoffs() ? includedDropoffs : includedPickups,
      passes:
        primary.compiled.stats.nodeOrder.length + (secondary?.compiled.stats.nodeOrder.length ?? 0)
    });
  }

  // ---- Masks, kernel, weights ---------------------------------------------------------------------
  function isInHours(hour: number): boolean {
    const {hours} = ctx.options;
    return hour >= hours[0] && hour < hours[1];
  }

  function writeMasks(): void {
    const {hours} = ctx.options;
    const pickups = new Uint32Array(pointCount);
    const dropoffs = new Uint32Array(pointCount);
    let pickupsIncluded = 0;
    let dropoffsIncluded = 0;
    for (let row = 0; row < count; row++) {
      if (isInHours(trips.pickupHour[row])) {
        pickups[row] = 1;
        pickupsIncluded++;
      }
      if (isInHours(trips.dropoffHour[row])) {
        dropoffs[count + row] = 1;
        dropoffsIncluded++;
      }
    }
    pickupMask.write(pickups);
    dropoffMask.write(dropoffs);
    includedPickups = pickupsIncluded;
    includedDropoffs = dropoffsIncluded;
    ctx.setReadout('kept', showsDropoffs() ? dropoffsIncluded : pickupsIncluded);
    ctx.setReadout(
      'window',
      hours[0] <= 0 && hours[1] >= NYC_TAXI_HOURS
        ? 'all 38 hours'
        : `${formatWindowTime(hours[0])} to ${formatWindowTime(hours[1])}`
    );
    updatePulse();
    publishCost();
    markChanged();
  }

  function writeKernel(): void {
    const {smoothing, sigma} = ctx.options;
    const active = smoothing === 'off' ? 0 : sigma;
    const radius = active > 0 ? Math.min(KERNEL_RADIUS, Math.ceil(3 * active)) : 0;
    const small = createGPUPointDensityGaussianKernel(radius, active > 0 ? active : undefined);
    const size = radius * 2 + 1;
    const offset = KERNEL_RADIUS - radius;
    const weights = new Float32Array(KERNEL_WIDTH * KERNEL_WIDTH);
    for (let row = 0; row < size; row++) {
      for (let column = 0; column < size; column++) {
        weights[(row + offset) * KERNEL_WIDTH + column + offset] = small[row * size + column];
      }
    }
    kernel.write(weights);
    markChanged();
  }

  // ---- Scales (all in trips per km2, fixed at load) -----------------------------------------------
  const getWindowHours = () => {
    const {hours} = ctx.options;
    return Math.min(hours[1], NYC_TAXI_HOURS) - hours[0];
  };
  const isWholePeriod = () => {
    const {hours} = ctx.options;
    return hours[0] <= 0 && hours[1] >= NYC_TAXI_HOURS;
  };
  /** The brightest colour: the whole-period 98th percentile, or a window of one locked rate. */
  const getClip = () =>
    isWholePeriod() ? reference.fullClip : reference.windowClipPerHour * getWindowHours();
  const getQuantileTable = (ramp: typeof DENSITY_RAMP | typeof DROPOFF_DENSITY_RAMP) => {
    const factor = getClip() / reference.fullClip;
    return makeQuantileDensityTable(
      reference.quantileBreaks.map(value => value * factor),
      ramp,
      getClip()
    );
  };
  const isSmoothed = () => ctx.options.binning === 'grid' && ctx.options.smoothing !== 'off';
  const publishScale = () => {
    ctx.setLegendData('clip', getClip());
    ctx.setLegendData('quantileTable', getQuantileTable(DENSITY_RAMP));
    ctx.setLegendData('dropoffQuantileTable', getQuantileTable(DROPOFF_DENSITY_RAMP));
  };

  function publishFareTable(): void {
    fareTable = makeMeanFareTable(reference.meanFareBreaks, ctx.ground(), ctx.options.minCount);
    ctx.setLegendData('fareTable', fareTable);
  }

  // ---- Reading the fields -------------------------------------------------------------------------
  const unprojectCell = (snapshot: FieldSnapshot, index: number): LngLat => {
    const [columns, rows] = snapshot.gridSize;
    const cellWidth = (snapshot.bounds[2] - snapshot.bounds[0]) / columns;
    const cellHeight = (snapshot.bounds[3] - snapshot.bounds[1]) / rows;
    return projection.unproject(
      snapshot.bounds[0] + ((index % columns) + 0.5) * cellWidth,
      snapshot.bounds[1] + (Math.floor(index / columns) + 0.5) * cellHeight
    ) as LngLat;
  };

  /** Local maxima of the field above `floor`, strongest first, at least a few cells apart. */
  function findPeaks(snapshot: FieldSnapshot, floor: number): number[] {
    const [columns, rows] = snapshot.gridSize;
    const {values} = snapshot;
    const maxima: number[] = [];
    for (let index = 0; index < values.length; index++) {
      const value = values[index];
      if (!(value > floor)) continue;
      const column = index % columns;
      const row = Math.floor(index / columns);
      let isMaximum = true;
      for (let dy = -1; dy <= 1 && isMaximum; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const x = column + dx;
          const y = row + dy;
          if (x < 0 || y < 0 || x >= columns || y >= rows) continue;
          const other = values[y * columns + x];
          // Ties go to the cell with the lower index, so a plateau counts once.
          if (other > value || (other === value && y * columns + x < index)) {
            isMaximum = false;
            break;
          }
        }
      }
      if (isMaximum) maxima.push(index);
    }
    const chosen: number[] = [];
    for (const index of maxima.sort((a, b) => values[b] - values[a])) {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const separated = chosen.every(other => {
        const dx = (other % columns) - column;
        const dy = Math.floor(other / columns) - row;
        return dx * dx + dy * dy >= PEAK_SEPARATION_CELLS ** 2;
      });
      if (separated) chosen.push(index);
      if (chosen.length === PEAK_COUNT) break;
    }
    return chosen;
  }

  const getCellIndexAt = (snapshot: FieldSnapshot, lngLat: LngLat): number => {
    const [columns, rows] = snapshot.gridSize;
    const [x, y] = projection.project(lngLat[0], lngLat[1]);
    const column = Math.floor(
      ((x - snapshot.bounds[0]) / (snapshot.bounds[2] - snapshot.bounds[0])) * columns
    );
    const row = Math.floor(
      ((y - snapshot.bounds[1]) / (snapshot.bounds[3] - snapshot.bounds[1])) * rows
    );
    if (column < 0 || row < 0 || column >= columns || row >= rows) return -1;
    return row * columns + column;
  };

  /** Re-derives everything the settled fields teach: readouts, legend strip, notes, charts. */
  function processFields(): void {
    const a = snapshots.primary;
    if (!a || !readBounds) return;
    const needsSecond = secondary !== null;
    if (needsSecond && !snapshots.secondary) return;
    const options = ctx.options;
    if (options.measure === 'rides') processRides(a, snapshots.secondary);
    else if (snapshots.secondary) processFare(a, snapshots.secondary);
    updateScaleTick(a);
    updateHalo();
  }

  function processRides(snapshot: FieldSnapshot, other: FieldSnapshot | null): void {
    const options = ctx.options;
    const clip = getClip();
    const area = snapshot.cellAreaKm2;
    const floor = isSmoothed() ? SMOOTHED_FLOOR : 0;
    publishScale();
    const peakValue = snapshot.extentMaximum;
    ctx.setReadout('peakCount', peakValue > 0 ? `${formatCount(peakValue)} rides` : null);
    ctx.setReadout(
      'peakDensity',
      peakValue > 0 ? `${formatDensity(peakValue / area)} trips per km²` : null
    );

    // Legend strip: the visible cells binned over the fixed colour range (bars are square roots).
    const bins = new Array<number>(LEGEND_BINS).fill(0);
    const densities: number[] = [];
    for (const value of snapshot.values) {
      if (!(value > floor) || !Number.isFinite(value)) continue;
      const density = value / area;
      densities.push(density);
      bins[Math.min(LEGEND_BINS - 1, Math.floor((density / clip) * LEGEND_BINS))]++;
    }
    ctx.setLegendData('histogram', bins);

    // The cell chart: on a linear axis the whole city is one bar, which is why the stretch exists.
    const sqrtAxis = options.scale !== 'linear';
    const axisValue = (density: number) =>
      Math.min(1, sqrtAxis ? Math.sqrt(density / clip) : density / clip);
    const chartBins = new Array<number>(CHART_BINS).fill(0);
    for (const density of densities) {
      chartBins[Math.min(CHART_BINS - 1, Math.floor(axisValue(density) * CHART_BINS))]++;
    }
    const table = options.scale === 'quantile' ? getQuantileTable(DENSITY_RAMP) : null;
    ctx.setChart('valueChart', {
      kind: 'histogram',
      values: chartBins,
      xDomain: [0, 1],
      height: 110,
      title: sqrtAxis ? 'Cells by density (square-root axis)' : 'Cells by density (linear axis)',
      xLabel: 'Trips per km²',
      yLabel: 'Cells',
      formatX: value => formatDensity((sqrtAxis ? value * value : value) * clip),
      ...(table
        ? {
            breaks: table.breaks.map(value => Math.min(1, Math.sqrt(value / clip))),
            classColors: table.colors
          }
        : {}),
      table: false,
      description:
        'Visible cells by trips per km². Most cells are quiet and a few are very busy; the quantile classes are marked where they cut the cells.'
    });

    // Place of the busiest cell(s) and the finding notes (local maxima), only on a plain map.
    const place = (target: FieldSnapshot) => {
      let best = -1;
      let bestValue = 0;
      for (let index = 0; index < target.values.length; index++) {
        if (target.values[index] > bestValue) {
          bestValue = target.values[index];
          best = index;
        }
      }
      return best < 0
        ? null
        : nearestPlaceLabel(NYC, unprojectCell(target, best), {maxDistanceMeters: 8000});
    };
    ctx.setReadout('peakPlace', place(snapshot));
    ctx.setReadout('peakPlaceB', other ? place(other) : null);
    if (snapshot.binning === 'grid' && options.swipe === 'off') {
      const notes: MapAnnotation[] = findPeaks(snapshot, peakValue * 0.02).map((index, rank) => {
        const coordinate = unprojectCell(snapshot, index);
        return {
          kind: 'note',
          id: `peak-${rank}`,
          coordinate,
          title: liveText('{density:integer} trips per km²', {
            density: snapshot.values[index] / area
          }),
          text: nearestPlaceLabel(NYC, coordinate, {maxDistanceMeters: 8000}) ?? undefined,
          tone: rank === 0 ? 'accent' : 'ink',
          priority: 5 - rank
        } satisfies MapAnnotation;
      });
      ctx.setAnnotations('peaks', notes.length ? notes : null);
    } else {
      ctx.setAnnotations('peaks', null);
    }
    ctx.setReadout('hiddenCells', null);
    ctx.setReadout('jfkFare', null);
    ctx.setReadout('lgaFare', null);
  }

  /** Mean fare of the cell nearest a place that holds at least `minimum` rides, or null. */
  function findPlaceCell(
    means: FieldSnapshot,
    tally: FieldSnapshot,
    lngLat: LngLat,
    minimum: number
  ): number {
    if (means.binning !== 'grid') return -1;
    const index = getCellIndexAt(means, lngLat);
    if (index < 0) return -1;
    const [columns, rows] = means.gridSize;
    const column = index % columns;
    const row = Math.floor(index / columns);
    let best = -1;
    let bestDistance = Infinity;
    for (let dy = -PLACE_SEARCH_CELLS; dy <= PLACE_SEARCH_CELLS; dy++) {
      for (let dx = -PLACE_SEARCH_CELLS; dx <= PLACE_SEARCH_CELLS; dx++) {
        const x = column + dx;
        const y = row + dy;
        if (x < 0 || y < 0 || x >= columns || y >= rows) continue;
        const other = y * columns + x;
        if (tally.values[other] < minimum) continue;
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = other;
        }
      }
    }
    return best;
  }

  function processFare(means: FieldSnapshot, tally: FieldSnapshot): void {
    const minimum = ctx.options.minCount;
    let nonEmpty = 0;
    let hidden = 0;
    for (const value of tally.values) {
      if (value > 0) {
        nonEmpty++;
        if (value < minimum) hidden++;
      }
    }
    ctx.setReadout(
      'hiddenCells',
      nonEmpty > 0 ? `${formatCount(hidden)} of ${formatCount(nonEmpty)} cells` : null
    );
    const notes: MapAnnotation[] = [];
    for (const [readout, placeId] of [
      ['jfkFare', 'jfk'],
      ['lgaFare', 'lga']
    ] as const) {
      const lngLat = NYC.places[placeId].lngLat as LngLat;
      const cell = findPlaceCell(means, tally, lngLat, minimum);
      if (cell < 0) {
        ctx.setReadout(readout, null);
        continue;
      }
      const fare = means.values[cell];
      const rides = tally.values[cell];
      ctx.setReadout(readout, `${formatMoney(fare)} (${formatCount(rides)} rides)`);
      notes.push({
        kind: 'note',
        id: `fare-${placeId}`,
        coordinate: unprojectCell(means, cell),
        title: liveText('{fare} a ride', {fare: formatMoney(fare)}),
        text: `${NYC.places[placeId].name}, ${formatCount(rides)} rides`,
        tone: 'accent'
      });
    }
    ctx.setAnnotations('peaks', notes.length ? notes : null);
    ctx.setReadout('peakCount', null);
    ctx.setReadout('peakDensity', null);
    ctx.setReadout('peakPlace', null);
    ctx.setReadout('peakPlaceB', null);
    ctx.setLegendData('histogram', null);
    ctx.setChart('valueChart', null);
  }

  // ---- Charts, furniture ---------------------------------------------------------------------------
  function updatePulse(): void {
    const {hours, show} = ctx.options;
    const whole = hours[0] <= 0 && hours[1] >= NYC_TAXI_HOURS;
    ctx.setChart('pulseChart', {
      kind: 'timeline',
      x: Array.from({length: HOURS}, (_, index) => index),
      y: Array.from(show === 'dropoffs' ? dropoffsPerHour : pickupsPerHour),
      mode: 'bars',
      window: whole ? undefined : [hours[0], hours[1]],
      height: 110,
      xLabel: 'Hours since midnight on Thu 1 Jan',
      yLabel: show === 'dropoffs' ? 'drop-offs per hour' : 'pickups per hour',
      formatX: value =>
        `${value < 24 ? 'Thu' : 'Fri'} ${String(Math.floor(value) % 24).padStart(2, '0')}h`,
      onScrub: time => {
        const length = whole ? WINDOW_HOURS : getWindowHours();
        const start = Math.max(
          0,
          Math.min(Math.round((time - length / 2) * 2) / 2, NYC_TAXI_HOURS - length)
        );
        ctx.setOptions({hours: [start, start + length]}, {notify: true});
      },
      description:
        'Taxi pickups (or drop-offs) per hour. The bracketed window is the one the map shows; click the chart to move it.'
    });
  }

  const runtimeFurniture: {
    title: {sample: string};
    scaleBar: {units: 'metric'; ticks?: number[]};
  } = {
    title: {
      sample: `${formatCount(count)} trips as ${formatCount(pointCount)} pickup and drop-off points`
    },
    scaleBar: {units: 'metric'}
  };
  ctx.setFurniture(runtimeFurniture);
  let lastCellText = '';
  /** Publishes the cell readouts only when their text changes (the cell follows the camera). */
  function publishCell(size: string, area: string): void {
    if (size === lastCellText) return;
    lastCellText = size;
    ctx.setReadout('cellSize', size);
    ctx.setReadout('cellArea', area || null);
  }

  let lastTickMeters: number | null | undefined;
  /** Scale-bar tick at the cell size, only while the grid is drawn (the step that teaches it). */
  function updateScaleTick(snapshot: FieldSnapshot): void {
    const [columns] = snapshot.gridSize;
    const cellWidth = (snapshot.bounds[2] - snapshot.bounds[0]) / columns;
    const teaching = ctx.options.showGrid && snapshot.binning === 'grid';
    const meters = teaching ? Math.round(cellWidth / 10) * 10 : null;
    if (meters === lastTickMeters) return;
    lastTickMeters = meters;
    runtimeFurniture.scaleBar = {units: 'metric', ticks: meters ? [meters] : undefined};
    ctx.setFurniture(runtimeFurniture);
  }

  let lastHalo: 'normal' | 'heavy' | null = null;
  /** Heavy label halos where names sit over the glow; normal ones on paper. */
  function updateHalo(): void {
    const weight = ctx.ground() === 'dark' ? 'heavy' : 'normal';
    if (weight === lastHalo) return;
    lastHalo = weight;
    ctx.setAnnotationHalo(weight);
  }

  /** Colour a density has on the map, for tooltip swatches. */
  function getDensitySwatch(
    density: number,
    ramp: typeof DENSITY_RAMP | typeof DROPOFF_DENSITY_RAMP
  ): readonly [number, number, number, number] {
    const {scale} = ctx.options;
    if (scale === 'quantile') {
      const table = getQuantileTable(ramp);
      return table.colors[getClassIndexOf(table, density)] as [number, number, number, number];
    }
    const t = Math.min(1, Math.max(0, density / getClip()));
    const [red, green, blue] = sampleRamp(
      ramp,
      scale === 'sqrt' ? Math.sqrt(t) : t,
      false,
      DENSITY_RAMP_RANGE
    );
    return [red, green, blue, 255];
  }

  /** Writes the cell edges of the current frame; returns the segment count (0 when cells are tiny). */
  function writeGridLines(frameBounds: Float32Array, columns: number, rows: number): number {
    if (cellWidthMeters / ctx.getMetersPerPixel() < MINIMUM_OUTLINE_PIXELS) return 0;
    const segments = new Float32Array((columns + 1 + rows + 1) * 4);
    let offset = 0;
    for (let column = 0; column <= columns; column++) {
      const x = frameBounds[0] + column * cellWidthMeters;
      segments.set([x, frameBounds[1], x, frameBounds[3]], offset);
      offset += 4;
    }
    for (let row = 0; row <= rows; row++) {
      const y = frameBounds[1] + row * cellHeightMeters;
      segments.set([frameBounds[0], y, frameBounds[2], y], offset);
      offset += 4;
    }
    gridLines.write(segments);
    return columns + 1 + rows + 1;
  }

  ctx.setReadout('points', pointCount);
  ctx.setReadout('corePickups', reference.coreShare.pickups);
  ctx.setReadout('coreDropoffs', reference.coreShare.dropoffs);
  writeKernel();
  writeMasks();
  publishScale();
  selectGraphs();

  return {
    getCompiledGraphs: () =>
      [primary?.compiled, secondary?.compiled].filter(Boolean) as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      switch (id) {
        case 'binning':
        case 'resolution':
        case 'measure':
        case 'show':
        case 'swipe':
          if (id === 'show') writeMasks();
          selectGraphs();
          publishScale();
          processFields();
          ctx.requestLayers();
          break;
        case 'smoothing':
        case 'sigma':
          writeKernel();
          ctx.requestLayers();
          break;
        case 'hours':
          writeMasks();
          publishScale();
          ctx.requestLayers();
          break;
        case 'minCount':
          publishFareTable();
          processFields();
          ctx.requestLayers();
          break;
        case 'showGrid':
          gridDirty = true;
          ctx.requestLayers();
          break;
        case 'scale':
          publishScale();
          processFields();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      publishFareTable();
      ctx.requestLayers();
    },

    // The fare step sits on paper (a light or dark paper by theme); the count steps on night.
    onGroundChange() {
      publishFareTable();
      updateHalo();
      ctx.requestLayers();
    },

    getTooltip(event) {
      const field = snapshots.primary;
      const options = ctx.options;
      const clear = () => {
        if (hoverIndex !== -1) {
          hoverIndex = -1;
          ctx.setLegendData('marker', null);
        }
        return null;
      };
      if (!field || !event.coordinate || field.binning !== 'grid') return clear();
      if (snapshots.secondary === null && secondary) return clear();
      // The field is only valid for the frame it was read at.
      if (!lastBounds || field.bounds.some((value, index) => value !== lastBounds![index])) {
        return clear();
      }
      const index = getCellIndexAt(field, event.coordinate as LngLat);
      if (index < 0) return clear();
      const [columns, rows] = field.gridSize;
      const cellWidth = (field.bounds[2] - field.bounds[0]) / columns;
      const cellHeight = (field.bounds[3] - field.bounds[1]) / rows;
      const west = field.bounds[0] + (index % columns) * cellWidth;
      const south = field.bounds[1] + Math.floor(index / columns) * cellHeight;
      const [west0, south0] = projection.unproject(west, south);
      const [east1, north1] = projection.unproject(west + cellWidth, south + cellHeight);
      const center = projection.unproject(west + cellWidth / 2, south + cellHeight / 2) as LngLat;
      const subtitle = nearestPlaceLabel(NYC, center, {maxDistanceMeters: 8000}) ?? undefined;
      const highlight = {kind: 'box', bounds: [west0, south0, east1, north1]} as const;
      const size = `${formatDistance(cellWidth)} cell`;
      const tooltipRows: TooltipRow[] = [];
      if (options.measure === 'fare') {
        const tally = snapshots.secondary;
        const rides = tally?.values[index] ?? 0;
        if (!(rides > 0)) return clear();
        const mean = field.values[index];
        hoverIndex = index;
        const hiddenCell = rides < options.minCount;
        tooltipRows.push(
          {
            label: 'Mean fare',
            value: hiddenCell ? 'not drawn' : formatMoney(mean),
            unit: hiddenCell ? undefined : 'per ride',
            swatch: hiddenCell
              ? undefined
              : (fareTable.colors[getClassIndexOf(fareTable, mean)] as [
                  number,
                  number,
                  number,
                  number
                ]),
            emphasis: true
          },
          {label: 'Rides behind it', value: formatCount(rides)}
        );
        return {
          title: size,
          subtitle,
          rows: tooltipRows,
          note: hiddenCell ? `Hidden: fewer than ${options.minCount} rides` : undefined,
          highlight
        };
      }
      const value = field.values[index];
      const floor = isSmoothed() ? SMOOTHED_FLOOR : 0;
      if (!(value > floor)) return clear();
      const area = field.cellAreaKm2;
      if (hoverIndex !== index) {
        hoverIndex = index;
        ctx.setLegendData('marker', value / area);
      }
      const endsMode = options.swipe === 'ends' && snapshots.secondary;
      const sideName = endsMode || options.show === 'pickups' ? 'Pickups' : 'Drop-offs';
      tooltipRows.push(
        {
          label: sideName,
          value: formatDensity(value / area),
          unit: 'trips per km²',
          swatch: getDensitySwatch(
            value / area,
            endsMode || options.show === 'pickups' ? DENSITY_RAMP : DROPOFF_DENSITY_RAMP
          ),
          emphasis: true
        },
        {
          label: 'Rides in the cell',
          value: formatCount(field.counts[index]),
          unit: isSmoothed() ? `(${formatCount(value)} smoothed)` : undefined
        }
      );
      if (endsMode && snapshots.secondary) {
        const otherValue = snapshots.secondary.values[index];
        tooltipRows.push({
          label: 'Drop-offs',
          value: formatDensity(otherValue / area),
          unit: 'trips per km²',
          swatch: getDensitySwatch(otherValue / area, DROPOFF_DENSITY_RAMP)
        });
      }
      return {title: size, subtitle, rows: tooltipRows, highlight};
    },

    encode(commandEncoder, frame) {
      if (!primary) return;
      const options = ctx.options;
      const viewBounds = getViewportMetricBounds(frame.viewport, projection);
      const [columns, rows] = primary.gridSize;
      const boundsData = Float32Array.from(viewBounds);
      const boundsChanged =
        !lastBounds || boundsData.some((value, index) => value !== lastBounds![index]);
      if (boundsChanged) markChanged();
      if (options.binning === 'hexagon') {
        const radius = Math.max(
          (viewBounds[2] - viewBounds[0]) / (SQRT3 * (columns - 1)),
          (viewBounds[3] - viewBounds[1]) / (1.5 * (rows - 1))
        );
        if (densityDirty) hexagonRadius.write(Float32Array.of(radius));
        cellWidthMeters = radius * SQRT3;
        cellHeightMeters = radius * 1.5;
        cellAreaKm2 = (1.5 * SQRT3 * radius * radius) / 1e6;
        publishCell(`${formatDistance(radius)} hexagon radius`, '');
      } else {
        cellWidthMeters = (viewBounds[2] - viewBounds[0]) / columns;
        cellHeightMeters = (viewBounds[3] - viewBounds[1]) / rows;
        cellAreaKm2 = (cellWidthMeters * cellHeightMeters) / 1e6;
        publishCell(
          `${formatDistance(cellWidthMeters)} across`,
          `${(cellAreaKm2 * 100).toFixed(cellAreaKm2 < 0.01 ? 2 : 1)} ha`
        );
      }
      // The colours are per km2, so the layers need the new cell area whenever the zoom changes it.
      if (Math.abs(cellAreaKm2 - lastLayerArea) > 0.002 * lastLayerArea) {
        lastLayerArea = cellAreaKm2;
        ctx.requestLayers();
      }
      if ((boundsChanged || gridDirty) && options.showGrid && options.binning === 'grid') {
        gridLineCount = writeGridLines(boundsData, columns, rows);
      }
      gridDirty = false;
      lastBounds = boundsData;
      if (densityDirty) {
        bounds.write(boundsData);
        primary.compiled.encode(commandEncoder, {parameters: undefined});
        secondary?.compiled.encode(commandEncoder, {parameters: undefined});
        densityDirty = false;
      }
      const settled = performance.now() - lastChangeTime > SETTLE_MILLISECONDS;
      const busy = primary.reader.isPending || Boolean(secondary?.reader.isPending);
      if (settleStale && settled && !busy) {
        readBounds = boundsData;
        readAreaKm2 = cellAreaKm2;
        primary.reader.request(commandEncoder);
        secondary?.reader.request(commandEncoder);
        settleStale = false;
      } else {
        primary.reader.flush(commandEncoder);
        secondary?.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!primary) return [];
      const options = ctx.options;
      const dark = ctx.ground() === 'dark';
      const layers: Layer[] = [];
      const area = Math.max(cellAreaKm2, 1e-9);
      const common = (graph: DensityGraph) => ({
        coordinateOrigin,
        gridSize: graph.gridSize,
        bounds: bounds.buffer,
        binning: options.binning,
        hexagonRadius: hexagonRadius.buffer,
        values: graph.values,
        valueFormat: 'float32' as const
      });

      if (options.measure === 'rides') {
        const clip = getClip();
        const floor = isSmoothed() ? SMOOTHED_FLOOR : 0;
        const rideLayer = (
          graph: DensityGraph,
          side: Side,
          scale: NycTaxiDensityOptions['scale'],
          compareSide?: 'a' | 'b'
        ) => {
          const ramp = side === 'pickups' ? DENSITY_RAMP : DROPOFF_DENSITY_RAMP;
          return new SpatialAnalysisRasterLayer({
            id: `density-${side}-${compareSide ?? 'only'}-${options.binning}`,
            ...common(graph),
            ...(scale === 'quantile'
              ? {colormap: 'uniform' as const, ...getClassTableLayerProps(getQuantileTable(ramp))}
              : {
                  colormap: ramp,
                  valueRange: [0, clip] as const,
                  sqrtScale: scale === 'sqrt',
                  rampRange: DENSITY_RAMP_RANGE
                }),
            // Counts become trips per km2: the scale below is in those units at every zoom.
            valueScale: 1 / area,
            discardAtOrBelow: floor / area,
            opacity: options.opacity,
            compareSide
          });
        };
        if (options.swipe === 'ends' && secondary) {
          layers.push(
            rideLayer(primary, 'pickups', options.scale, 'a'),
            rideLayer(secondary, 'dropoffs', options.scale, 'b')
          );
        } else if (options.swipe === 'stretch') {
          layers.push(
            rideLayer(primary, options.show, 'linear', 'a'),
            rideLayer(primary, options.show, options.scale, 'b')
          );
        } else {
          layers.push(rideLayer(primary, options.show, options.scale));
        }
      } else if (secondary) {
        // Paper ground: the mean fare of each cell, classed; cells with too few rides are masked
        // by the tally channel and drawn as a hatch.
        const minimum = options.minCount;
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `density-mean-fare-${options.binning}`,
            ...common(primary),
            colormap: 'uniform',
            ...getClassTableLayerProps(fareTable),
            discardAtOrBelow: 0,
            noDataColor: [0, 0, 0, 0],
            instanceChannels: secondary.values,
            channelStride: 1,
            channels: {alpha: 0},
            alphaDomain: [minimum - 0.5, minimum - 0.4],
            alphaOutput: [0, 1],
            opacity: 0.88
          }),
          new SpatialAnalysisRasterLayer({
            id: `density-low-n-${options.binning}`,
            ...common(secondary),
            colormap: 'uniform',
            classBreaks: minimum > 1 ? [1, minimum] : [1],
            classColors:
              minimum > 1
                ? [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                  ]
                : [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                  ],
            hatchClasses: minimum > 1 ? [1] : [],
            hatchColor: dark ? [214, 221, 234, 150] : [74, 86, 99, 175],
            hatchSpacingPixels: 5,
            hatchWidthPixels: 1,
            opacity: 1
          })
        );
      }
      if (options.showGrid && options.binning === 'grid' && gridLineCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'density-cell-outlines',
            coordinateOrigin,
            segments: gridLines,
            instanceCount: gridLineCount,
            widthPixels: 0.5,
            color: [255, 255, 255, 31]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const entry of graphs.values()) entry.reader.stop();
      resources.destroy();
    }
  };
}
