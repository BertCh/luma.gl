// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  createGPUPointDensityGaussianKernel,
  createGPUPointDensityGaussianKernel1D,
  GPUPointDensity,
  type GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../../engine/layers';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  fillNatureMask,
  formatHourWindow,
  readNatureColumns,
  type NatureColumns
} from './b1-nature-data';

/** Option state of the nature-density scene. */
export type NatureDensityOptions = {
  binning: 'grid' | 'hexagon';
  resolution: 'coarse' | 'medium' | 'fine';
  statistic: 'count' | 'sum' | 'mean';
  weight: 'researchGrade' | 'introduced' | 'animal';
  sumAccumulation: 'workgroup' | 'atomic';
  smoothing: 'off' | 'gaussian-2d' | 'gaussian-separable';
  sigma: number;
  hours: readonly [number, number];
  invertHours: boolean;
  dayType: 'all' | 'weekdays' | 'weekends';
  category: string;
  ramp: 'inferno' | 'magma' | 'viridis' | 'cividis';
  opacity: number;
  showPoints: boolean;
};

const GRID_SIZES = {coarse: [70, 44], medium: [110, 70], fine: [170, 106]} as const;
const HEXAGON_SIZES = {coarse: [34, 26], medium: [52, 38], fine: [80, 58]} as const;
const SQRT3 = Math.sqrt(3);
/** Compile-time kernel; per-frame weights select the actual sigma. */
const KERNEL_RADIUS = 8;
const KERNEL_WIDTH = KERNEL_RADIUS * 2 + 1;
const HISTOGRAM_BINS = 16;
const SETTLE_MILLISECONDS = 500;

type DensityGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  values: Buffer;
  extent: Buffer;
  extentReader: SummaryReader;
};

type DensitySet = {
  gridSize: readonly [number, number];
  dense?: DensityGraph;
  separable?: DensityGraph;
  hexagon?: DensityGraph;
  compareReader: SummaryReader | null;
};

/**
 * Viewport-following density of Chicago nature observations. Everything the analyst steers (bounds, hexagon
 * radius, Gaussian sigma, the hour/day/category mask and the weight attribute) is a buffer write.
 * Compile-time choices (lattice, resolution, statistic, accumulation) select between lazily compiled
 * graph sets that are cached, so switching back never recompiles.
 */
export async function createNatureDensity(
  ctx: SceneContext<NatureDensityOptions>
): Promise<SceneInstance<NatureDensityOptions>> {
  const observations = ctx.datasets.get('chicago-nature');
  const {device} = ctx;
  const columns: NatureColumns = readNatureColumns(observations);
  const origin = columns.origin;
  const projection = observations.getProjection(origin);
  const pointCount = columns.count;
  const categoryIndex = (name: string) =>
    name === 'all' ? -1 : columns.categoryNames.indexOf(name);

  const resources = new SpatialAnalysisResources(device, 'nature-density');
  const positionsBuffer = resources.createBuffer('positions', columns.positions);
  const maskBuffer = resources.createBuffer('mask', new Uint32Array(pointCount).fill(1));
  const weightsBuffer = resources.createBuffer('weights', columns[ctx.options.weight]);
  const bounds = resources.createParameterBuffer('bounds', 'float32', 4);
  const hexagonRadius = resources.createParameterBuffer(
    'hexagon-radius',
    'float32',
    1,
    Float32Array.of(50)
  );
  const kernel = resources.createParameterBuffer('kernel', 'float32', KERNEL_WIDTH * KERNEL_WIDTH);
  const lineKernel = resources.createParameterBuffer('line-kernel', 'float32', KERNEL_WIDTH);

  let destroyed = false;
  let measuring = false;
  let settleStale = true;
  let lastChangeTime = performance.now();
  let lastBounds: Float32Array | null = null;
  const sets = new Map<string, DensitySet>();
  let current: DensitySet | null = null;
  let currentKey = '';

  ctx.setReadout('points', pointCount);
  writeKernels(kernel, lineKernel, ctx.options.smoothing === 'off' ? 0 : ctx.options.sigma);
  ctx.setReadout('kernelRadius', `${getKernelRadius(ctx.options.sigma)} cells`);

  const markChanged = () => {
    settleStale = true;
    lastChangeTime = performance.now();
  };

  const getKey = (options: NatureDensityOptions) =>
    `${options.binning}:${options.resolution}:${options.statistic}:${options.sumAccumulation}`;

  const getDisplayed = (): DensityGraph | undefined => {
    if (!current) return undefined;
    if (ctx.options.binning === 'hexagon') return current.hexagon;
    return ctx.options.smoothing === 'gaussian-separable' ? current.separable : current.dense;
  };

  function buildDensityGraph(
    id: string,
    options: NatureDensityOptions,
    gridSize: readonly [number, number],
    binning: 'grid' | 'hexagon',
    useSeparable: boolean
  ): DensityGraph {
    const cellCount = gridSize[0] * gridSize[1];
    const values = resources.createBuffer(`${id}-values`, cellCount * 4);
    const extent = resources.createBuffer(`${id}-extent`, 8);
    const histogram = resources.createBuffer(`${id}-histogram`, HISTOGRAM_BINS * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `nature-density-${id}`});
    const lineView = useSeparable ? lineKernel.importToGraph(graph) : undefined;
    const weighted = options.statistic !== 'count';
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
        binning,
        ...(binning === 'hexagon'
          ? {hexagonRadius: hexagonRadius.importToGraph(graph)}
          : {
              smoothing: lineView
                ? {
                    separableKernel: {horizontal: lineView, vertical: lineView},
                    kernelWidth: KERNEL_WIDTH,
                    kernelHeight: KERNEL_WIDTH
                  }
                : {
                    kernel: kernel.importToGraph(graph),
                    kernelWidth: KERNEL_WIDTH,
                    kernelHeight: KERNEL_WIDTH,
                    strategy: 'direct'
                  }
            }),
        statistic: options.statistic,
        ...(weighted ? {sumAccumulation: options.sumAccumulation} : {}),
        output: {
          values: importGraphBuffer(graph, 'values', values, 'float32', cellCount),
          extent: importGraphBuffer(graph, 'extent', extent, 'float32', 2),
          histogram: importGraphBuffer(graph, 'histogram', histogram, 'uint32', HISTOGRAM_BINS)
        }
      })
    );
    const extentReader = new SummaryReader(
      resources,
      `${id}-extent`,
      [{buffer: extent, size: 8}],
      bytes => {
        if (destroyed || getDisplayed()?.extent !== extent) return;
        const [low, high] = new Float32Array(bytes);
        ctx.setLegendExtent('density', [low, high]);
        ctx.setReadout('peak', high);
      }
    );
    return {compiled: resources.track(graph.compile()), values, extent, extentReader};
  }

  function buildSet(options: NatureDensityOptions): DensitySet {
    const key = getKey(options);
    const id = key.replace(/:/g, '-');
    if (options.binning === 'grid') {
      const gridSize = GRID_SIZES[options.resolution];
      const dense = buildDensityGraph(`${id}-dense`, options, gridSize, 'grid', false);
      const separable = buildDensityGraph(`${id}-separable`, options, gridSize, 'grid', true);
      const cellCount = gridSize[0] * gridSize[1];
      const compareReader = new SummaryReader(
        resources,
        `${id}-compare`,
        [
          {buffer: dense.values, size: cellCount * 4},
          {buffer: separable.values, size: cellCount * 4}
        ],
        bytes => {
          if (destroyed) return;
          const denseValues = new Float32Array(bytes, 0, cellCount);
          const separableValues = new Float32Array(bytes, cellCount * 4, cellCount);
          let maximumDifference = 0;
          let maximumValue = 0;
          for (let index = 0; index < cellCount; index++) {
            maximumDifference = Math.max(
              maximumDifference,
              Math.abs(denseValues[index] - separableValues[index])
            );
            maximumValue = Math.max(maximumValue, denseValues[index]);
          }
          ctx.setReadout(
            'difference',
            `${maximumDifference.toExponential(2)} (field max ${maximumValue.toPrecision(3)})`
          );
        }
      );
      return {gridSize, dense, separable, compareReader};
    }
    const gridSize = HEXAGON_SIZES[options.resolution];
    return {
      gridSize,
      hexagon: buildDensityGraph(`${id}-hexagon`, options, gridSize, 'hexagon', false),
      compareReader: null
    };
  }

  function selectSet(): void {
    const key = getKey(ctx.options);
    if (key === currentKey && current) return;
    let next = sets.get(key);
    if (!next) {
      next = buildSet(ctx.options);
      sets.set(key, next);
    }
    current = next;
    currentKey = key;
    const [columnsCount, rowsCount] = next.gridSize;
    ctx.setReadout(
      'grid',
      `${columnsCount} × ${rowsCount} ${ctx.options.binning === 'hexagon' ? 'hexagons' : 'cells'}`
    );
    if (ctx.options.binning === 'hexagon') {
      for (const id of ['difference', 'denseTime', 'separableTime', 'speedup'])
        ctx.setReadout(id, 'n/a (square grid only)');
    } else {
      for (const id of ['difference', 'denseTime', 'separableTime', 'speedup'])
        ctx.setReadout(id, null);
    }
    markChanged();
  }

  function writeMask(): void {
    const {hours, invertHours, dayType, category} = ctx.options;
    const mask = new Uint32Array(pointCount);
    const included = fillNatureMask(
      columns,
      {hours, invertHours, dayType, category: categoryIndex(category)},
      mask
    );
    maskBuffer.write(mask);
    ctx.setReadout('included', `${formatCount(included)} of ${formatCount(pointCount)}`);
    ctx.setReadout('window', formatHourWindow(hours, invertHours));
    markChanged();
  }

  function writeWeights(): void {
    weightsBuffer.write(columns[ctx.options.weight]);
    markChanged();
  }

  async function measureSmoothing(): Promise<void> {
    const set = current;
    if (measuring || destroyed || !set?.dense || !set.separable) return;
    measuring = true;
    try {
      const options = {parameters: undefined, completionBuffer: set.dense.extent};
      const denseTiming = await measureCompiledGraph(device, set.dense.compiled, options);
      const separableTiming = await measureCompiledGraph(device, set.separable.compiled, options);
      if (destroyed || current !== set) return;
      ctx.setReadout('denseTime', `${denseTiming.milliseconds.toFixed(2)} ms`);
      ctx.setReadout('separableTime', `${separableTiming.milliseconds.toFixed(2)} ms`);
      const ratio = denseTiming.milliseconds / separableTiming.milliseconds;
      ctx.setReadout('speedup', Number.isFinite(ratio) ? `${ratio.toFixed(2)}x` : 'n/a');
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  writeMask();
  writeWeights();
  selectSet();

  return {
    getCompiledGraphs: () => {
      if (!current) return [];
      const graphs = [current.dense, current.separable, current.hexagon]
        .filter((graph): graph is DensityGraph => Boolean(graph))
        .map(graph => graph.compiled);
      return graphs as unknown as CompiledGPUCommandGraph<never>[];
    },

    setOption(id, _value, state) {
      if (['binning', 'resolution', 'statistic', 'sumAccumulation'].includes(id)) {
        selectSet();
        ctx.requestLayers();
      } else if (id === 'smoothing' || id === 'sigma') {
        writeKernels(kernel, lineKernel, state.smoothing === 'off' ? 0 : state.sigma);
        markChanged();
        if (id === 'sigma') ctx.setReadout('kernelRadius', `${getKernelRadius(state.sigma)} cells`);
        ctx.requestLayers();
      } else if (['hours', 'invertHours', 'dayType', 'category'].includes(id)) {
        writeMask();
      } else if (id === 'weight') {
        writeWeights();
        ctx.requestLayers();
      } else {
        ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'measure') void measureSmoothing();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const displayed = getDisplayed();
      if (!displayed || !current) return;
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
      if (!lastBounds || boundsData.some((value, index) => value !== lastBounds![index]))
        markChanged();
      lastBounds = boundsData;
      bounds.write(boundsData);
      displayed.compiled.encode(commandEncoder, {parameters: undefined});
      const settled = performance.now() - lastChangeTime > SETTLE_MILLISECONDS;
      if (settleStale && settled && !displayed.extentReader.isPending) {
        displayed.extentReader.request(commandEncoder);
        const {dense, separable, compareReader} = current;
        if (dense && separable && compareReader && !compareReader.isPending) {
          const other = displayed === dense ? separable : dense;
          other.compiled.encode(commandEncoder, {parameters: undefined});
          compareReader.request(commandEncoder);
        }
        settleStale = false;
      } else {
        displayed.extentReader.flush(commandEncoder);
        current.compareReader?.flush(commandEncoder);
      }
    },

    getLayers() {
      const displayed = getDisplayed();
      if (!displayed || !current) return [];
      const {binning, ramp, opacity, showPoints, statistic} = ctx.options;
      const layers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          id: `nature-density-${binning}`,
          coordinateOrigin: [origin[0], origin[1], 0],
          gridSize: current.gridSize,
          bounds: bounds.buffer,
          binning,
          hexagonRadius: hexagonRadius.buffer,
          values: displayed.values,
          valueFormat: 'float32',
          ...(statistic === 'mean' ? {valueRange: [0, 1] as const} : {extent: displayed.extent}),
          colormap: ramp,
          sqrtScale: statistic !== 'mean',
          discardAtOrBelow:
            binning === 'grid' && ctx.options.smoothing !== 'off'
              ? statistic === 'mean'
                ? 0.002
                : 0.3
              : 0,
          color: [255, 255, 255, Math.round(opacity * 255)]
        })
      ];
      if (showPoints) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'nature-density-points',
            coordinateOrigin: [origin[0], origin[1], 0],
            positions: positionsBuffer,
            instanceCount: pointCount,
            radiusPixels: 1,
            color: ctx.theme() === 'dark' ? [120, 220, 255, 90] : [20, 90, 160, 80]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const set of sets.values()) {
        set.compareReader?.stop();
        for (const graph of [set.dense, set.separable, set.hexagon]) graph?.extentReader.stop();
      }
      resources.destroy();
    }
  };
}

function getKernelRadius(sigma: number): number {
  return Math.min(KERNEL_RADIUS, Math.ceil(3 * sigma));
}

/** Writes the 2D and 1D Gaussians for `sigma` (0 writes a unit impulse, smoothing off). */
function writeKernels(
  kernel: GPUParameterBuffer<'float32'>,
  lineKernel: GPUParameterBuffer<'float32'>,
  sigma: number
): void {
  const radius = sigma > 0 ? getKernelRadius(sigma) : 0;
  const small = createGPUPointDensityGaussianKernel(radius, sigma > 0 ? sigma : undefined);
  const smallLine = createGPUPointDensityGaussianKernel1D(radius, sigma > 0 ? sigma : undefined);
  const size = radius * 2 + 1;
  const offset = KERNEL_RADIUS - radius;
  const weights = new Float32Array(KERNEL_WIDTH * KERNEL_WIDTH);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      weights[(row + offset) * KERNEL_WIDTH + column + offset] = small[row * size + column];
    }
  }
  const lineWeights = new Float32Array(KERNEL_WIDTH);
  lineWeights.set(smallLine, offset);
  kernel.write(weights);
  lineKernel.write(lineWeights);
}
