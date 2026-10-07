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
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  countBins,
  formatTaxiTime,
  loadTaxiTrips,
  NYC_TAXI_ORIGIN,
  type TaxiTrips
} from './nyc-taxi-data';

/** Option state of the nyc-taxi-density scene. */
export type NycTaxiDensityOptions = {
  show: 'pickups' | 'dropoffs' | 'balance';
  measure: 'trips' | 'fare' | 'passengers' | 'distance';
  statistic: 'total' | 'average';
  binning: 'grid' | 'hexagon';
  resolution: 'coarse' | 'medium' | 'fine';
  smoothing: 'off' | 'gaussian';
  sigma: number;
  hours: readonly [number, number];
  invertHours: boolean;
  ramp: 'inferno' | 'magma' | 'viridis' | 'cividis';
  lossRamp: 'cividis' | 'viridis';
  opacity: number;
};

type GpuStatistic = 'count' | 'sum' | 'mean';

const GRID_SIZES = {coarse: [70, 44], medium: [110, 70], fine: [170, 106]} as const;
const HEXAGON_SIZES = {coarse: [34, 26], medium: [52, 38], fine: [80, 58]} as const;
const SQRT3 = Math.sqrt(3);
/** Compile-time kernel; per-frame weights select the actual sigma. */
const KERNEL_RADIUS = 8;
const KERNEL_WIDTH = KERNEL_RADIUS * 2 + 1;
const HISTOGRAM_BINS = 16;
const SETTLE_MILLISECONDS = 500;
const HOURS = 39;

type DensityGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  gridSize: readonly [number, number];
  values: Buffer;
  extent: Buffer;
  reader: SummaryReader;
};

/** Rows 0 to count-1 are pickups, rows count to 2*count-1 the dropoffs of the same trips. */
function getGpuStatistic(options: NycTaxiDensityOptions): GpuStatistic {
  if (options.show === 'balance') return 'sum';
  if (options.measure === 'trips') return 'count';
  return options.statistic === 'average' ? 'mean' : 'sum';
}

/**
 * Pickup and dropoff density of 440,000 taxi trips. The pickups and dropoffs are one 880,000-point
 * buffer: a mask selects which half and which hours count, and signed weights turn the same graph
 * into a balance map (dropoffs minus pickups). The grid follows the camera, the kernel is a weight
 * buffer, and the statistic, lattice and resolution select between lazily compiled graphs.
 */
export async function createNycTaxiDensity(
  ctx: SceneContext<NycTaxiDensityOptions>
): Promise<SceneInstance<NycTaxiDensityOptions>> {
  const {device} = ctx;
  const trips: TaxiTrips = loadTaxiTrips(ctx.datasets.get('poopdeck-nyc-taxi'));
  const count = trips.count;
  const pointCount = count * 2;
  const projection = ctx.datasets.get('poopdeck-nyc-taxi').getProjection(NYC_TAXI_ORIGIN);
  const resources = new SpatialAnalysisResources(device, 'taxi-density');
  const coordinateOrigin: [number, number, number] = [NYC_TAXI_ORIGIN[0], NYC_TAXI_ORIGIN[1], 0];

  const positions = new Float32Array(pointCount * 2);
  positions.set(trips.pickup, 0);
  positions.set(trips.dropoff, count * 2);
  const positionsBuffer = resources.createBuffer('positions', positions);
  const maskBuffer = resources.createBuffer('mask', new Uint32Array(pointCount).fill(1));
  const weightsBuffer = resources.createBuffer('weights', new Float32Array(pointCount).fill(1));
  const bounds = resources.createParameterBuffer('bounds', 'float32', 4);
  const hexagonRadius = resources.createParameterBuffer(
    'hexagon-radius',
    'float32',
    1,
    Float32Array.of(300)
  );
  const kernel = resources.createParameterBuffer('kernel', 'float32', KERNEL_WIDTH * KERNEL_WIDTH);

  const measures = {
    trips: new Float32Array(count).fill(1),
    fare: trips.fare,
    passengers: Float32Array.from(trips.passengers),
    distance: trips.distance
  };

  let destroyed = false;
  let settleStale = true;
  let lastChangeTime = performance.now();
  let lastBounds: Float32Array | null = null;
  /** Largest absolute value of the balance field, from the last extent readback. */
  let balanceMaximum = 1;
  const graphs = new Map<string, DensityGraph>();
  let current: DensityGraph | null = null;
  let currentKey = '';

  const markChanged = () => {
    settleStale = true;
    lastChangeTime = performance.now();
  };

  const getKey = (options: NycTaxiDensityOptions) =>
    `${options.binning}:${options.resolution}:${getGpuStatistic(options)}`;

  // CPU context chart: trips picked up and dropped off per hour, independent of every option.
  const pickupsPerHour = countBins(trips.pickupHour, 0, HOURS, HOURS);
  const dropoffsPerHour = countBins(trips.dropoffHour, 0, HOURS, HOURS);

  function buildGraph(options: NycTaxiDensityOptions): DensityGraph {
    const key = getKey(options);
    const id = key.replace(/:/g, '-');
    const statistic = getGpuStatistic(options);
    const gridSize =
      options.binning === 'grid'
        ? GRID_SIZES[options.resolution]
        : HEXAGON_SIZES[options.resolution];
    const cellCount = gridSize[0] * gridSize[1];
    const values = resources.createBuffer(`${id}-values`, cellCount * 4);
    const extent = resources.createBuffer(`${id}-extent`, 8);
    const histogram = resources.createBuffer(`${id}-histogram`, HISTOGRAM_BINS * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `taxi-density-${id}`});
    const weighted = statistic !== 'count';
    graph.add(
      new GPUPointDensity({
        id: 'density',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', pointCount),
        mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', pointCount),
        ...(weighted
          ? {weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', pointCount)}
          : {}),
        bounds: bounds.importToGraph(graph),
        gridSize,
        binning: options.binning,
        ...(options.binning === 'hexagon'
          ? {hexagonRadius: hexagonRadius.importToGraph(graph)}
          : {
              smoothing: {
                kernel: kernel.importToGraph(graph),
                kernelWidth: KERNEL_WIDTH,
                kernelHeight: KERNEL_WIDTH,
                strategy: 'direct'
              }
            }),
        statistic,
        output: {
          values: importGraphBuffer(graph, 'values', values, 'float32', cellCount),
          extent: importGraphBuffer(graph, 'extent', extent, 'float32', 2),
          histogram: importGraphBuffer(graph, 'histogram', histogram, 'uint32', HISTOGRAM_BINS)
        }
      })
    );
    const built: DensityGraph = {
      compiled: resources.track(graph.compile()),
      gridSize,
      values,
      extent,
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      resources,
      `${id}-summary`,
      [
        {buffer: extent, size: 8},
        {buffer: histogram, size: HISTOGRAM_BINS * 4}
      ],
      bytes => {
        if (!destroyed && current === built) processSummary(bytes);
      }
    );
    return built;
  }

  function selectGraph(): void {
    const key = getKey(ctx.options);
    if (key === currentKey && current) return;
    let next = graphs.get(key);
    if (!next) {
      next = buildGraph(ctx.options);
      graphs.set(key, next);
    }
    current = next;
    currentKey = key;
    ctx.setReadout(
      'grid',
      `${next.gridSize[0]} × ${next.gridSize[1]} ${ctx.options.binning === 'hexagon' ? 'hexagons' : 'cells'}`
    );
    ctx.setReadout('statisticUsed', getGpuStatistic(ctx.options));
    markChanged();
  }

  function isInHours(hour: number): boolean {
    const {hours, invertHours} = ctx.options;
    return (hour >= hours[0] && hour < hours[1]) !== invertHours;
  }

  function writeMask(): void {
    const {show, hours, invertHours} = ctx.options;
    const mask = new Uint32Array(pointCount);
    let included = 0;
    for (let row = 0; row < count; row++) {
      if (show !== 'dropoffs' && isInHours(trips.pickupHour[row])) {
        mask[row] = 1;
        included++;
      }
      if (show !== 'pickups' && isInHours(trips.dropoffHour[row])) {
        mask[count + row] = 1;
        included++;
      }
    }
    maskBuffer.write(mask);
    ctx.setReadout('included', `${formatCount(included)} points of ${formatCount(pointCount)}`);
    ctx.setReadout(
      'window',
      hours[0] <= 0 && hours[1] >= HOURS && !invertHours
        ? 'all 38 hours'
        : `${invertHours ? 'outside ' : ''}${formatTaxiTime(hours[0])} to ${formatTaxiTime(hours[1])}`
    );
    updateChart();
    markChanged();
  }

  function writeWeights(): void {
    const {show, measure} = ctx.options;
    const source = measures[measure];
    const weights = new Float32Array(pointCount);
    const pickupSign = show === 'balance' ? -1 : 1;
    for (let row = 0; row < count; row++) {
      weights[row] = pickupSign * source[row];
      weights[count + row] = source[row];
    }
    weightsBuffer.write(weights);
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
    ctx.setReadout('kernelRadius', radius === 0 ? 'none' : `${radius} cells`);
    markChanged();
  }

  function updateChart(): void {
    const {hours, invertHours} = ctx.options;
    const x = Array.from({length: HOURS}, (_, index) => index + 0.5);
    const brushed = hours[0] > 0 || hours[1] < HOURS;
    ctx.setChart('pulseChart', {
      kind: 'line',
      series: [
        {label: 'pickups', x, y: Array.from(pickupsPerHour), color: 0},
        {label: 'dropoffs', x, y: Array.from(dropoffsPerHour), color: 2, dashed: true}
      ],
      xLabel: 'Hours since midnight on Thu 1 Jan',
      yLabel: 'trips per hour',
      height: 120,
      formatX: value =>
        `${value < 24 ? 'Thu' : 'Fri'} ${String(Math.floor(value) % 24).padStart(2, '0')}h`,
      markers: brushed && !invertHours ? [{x: hours[0]}, {x: hours[1]}] : [],
      description: 'Taxi pickups and dropoffs per hour, with the selected window marked.'
    });
  }

  function processSummary(bytes: ArrayBuffer): void {
    const floats = new Float32Array(bytes, 0, 2);
    const histogram = new Uint32Array(bytes, 8, HISTOGRAM_BINS);
    const [low, high] = floats;
    const {show, measure} = ctx.options;
    if (show === 'balance') {
      const next = Math.max(1, Math.abs(low), Math.abs(high));
      if (Math.abs(next - balanceMaximum) > 0.005 * balanceMaximum) {
        balanceMaximum = next;
        ctx.requestLayers();
      }
      ctx.setLegendExtent('density', [0, balanceMaximum]);
      ctx.setLegendExtent('loss', [0, balanceMaximum]);
      ctx.setReadout('peak', `+${formatCount(high)} / ${formatCount(low)}`);
    } else {
      ctx.setLegendExtent('density', [low, high]);
      ctx.setReadout(
        'peak',
        measure === 'trips' && getGpuStatistic(ctx.options) === 'count'
          ? formatCount(high)
          : high.toFixed(high < 10 ? 2 : 0)
      );
    }
    ctx.setChart('valueChart', {
      kind: 'histogram',
      values: Array.from(histogram),
      xDomain: [low, high > low ? high : low + 1],
      height: 100,
      xLabel: 'Cell value (most cells are empty)',
      yLabel: 'cells',
      formatX: value =>
        Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(1)}k` : value.toFixed(0)
    });
  }

  ctx.setReadout('points', pointCount);
  writeKernel();
  writeWeights();
  writeMask();
  selectGraph();

  return {
    getCompiledGraphs: () =>
      current ? ([current.compiled] as CompiledGPUCommandGraph<never>[]) : [],

    setOption(id) {
      switch (id) {
        case 'binning':
        case 'resolution':
        case 'statistic':
          selectGraph();
          ctx.requestLayers();
          break;
        case 'show':
          writeMask();
          writeWeights();
          selectGraph();
          balanceMaximum = 1;
          ctx.requestLayers();
          break;
        case 'measure':
          writeWeights();
          selectGraph();
          ctx.requestLayers();
          break;
        case 'smoothing':
        case 'sigma':
          writeKernel();
          ctx.requestLayers();
          break;
        case 'hours':
        case 'invertHours':
          writeMask();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (!current) return;
      const {binning} = ctx.options;
      const viewBounds = getViewportMetricBounds(frame.viewport, projection);
      if (binning === 'hexagon') {
        const [columnsCount, rowsCount] = current.gridSize;
        const radius = Math.max(
          (viewBounds[2] - viewBounds[0]) / (SQRT3 * (columnsCount - 1)),
          (viewBounds[3] - viewBounds[1]) / (1.5 * (rowsCount - 1))
        );
        hexagonRadius.write(Float32Array.of(radius));
        ctx.setReadout('cellSize', `${radius.toFixed(0)} m hexagon radius`);
      } else {
        ctx.setReadout(
          'cellSize',
          `${((viewBounds[2] - viewBounds[0]) / current.gridSize[0]).toFixed(0)} m`
        );
      }
      const boundsData = Float32Array.from(viewBounds);
      if (!lastBounds || boundsData.some((value, index) => value !== lastBounds![index])) {
        markChanged();
      }
      lastBounds = boundsData;
      bounds.write(boundsData);
      current.compiled.encode(commandEncoder, {parameters: undefined});
      const settled = performance.now() - lastChangeTime > SETTLE_MILLISECONDS;
      if (settleStale && settled && !current.reader.isPending) {
        current.reader.request(commandEncoder);
        settleStale = false;
      } else {
        current.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!current) return [];
      const {show, ramp, lossRamp, opacity, binning, smoothing} = ctx.options;
      const statistic = getGpuStatistic(ctx.options);
      const smoothed = binning === 'grid' && smoothing !== 'off';
      const common = {
        coordinateOrigin,
        gridSize: current.gridSize,
        bounds: bounds.buffer,
        binning,
        hexagonRadius: hexagonRadius.buffer,
        values: current.values,
        valueFormat: 'float32' as const,
        color: [255, 255, 255, Math.round(opacity * 255)] as [number, number, number, number]
      };
      const empty = smoothed ? (statistic === 'count' ? 0.3 : 0.002) : 0;
      if (show === 'balance') {
        // One graph holds the signed field; two layers show each sign with its own ramp.
        const epsilon = Math.max(0.002 * balanceMaximum, smoothed ? 0.05 : 0);
        return [
          new SpatialAnalysisRasterLayer({
            id: `taxi-density-gain-${binning}`,
            ...common,
            colormap: ramp,
            valueRange: [0, balanceMaximum],
            sqrtScale: true,
            discardAtOrBelow: epsilon
          }),
          new SpatialAnalysisRasterLayer({
            id: `taxi-density-loss-${binning}`,
            ...common,
            colormap: lossRamp,
            valueScale: -1,
            valueRange: [0, balanceMaximum],
            sqrtScale: true,
            discardAtOrBelow: epsilon
          })
        ] as Layer[];
      }
      return [
        new SpatialAnalysisRasterLayer({
          id: `taxi-density-${show}-${binning}`,
          ...common,
          colormap: ramp,
          extent: current.extent,
          sqrtScale: statistic !== 'mean',
          discardAtOrBelow: empty
        })
      ] as Layer[];
    },

    destroy() {
      destroyed = true;
      for (const entry of graphs.values()) entry.reader.stop();
      resources.destroy();
    }
  };
}
