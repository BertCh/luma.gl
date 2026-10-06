// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  createGPUPointDensityGaussianKernel,
  createGPUPointDensityGaussianKernel1D,
  GPUPointDensity
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {type GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../spatial-analysis-resources';
import {SummaryReader} from './summary-reader';
import {formatCompiledGraphTiming, formatSpeedup, measureCompiledGraph} from './vector-timing';

const GRID_SIZE: readonly [number, number] = [160, 100];
const HEXAGON_GRID_SIZE: readonly [number, number] = [72, 52];
const SQRT3 = Math.sqrt(3);
/** Compile-time kernel size; per-frame weights select the actual smoothing sigma. */
const KERNEL_RADIUS = 8;
const KERNEL_WIDTH = KERNEL_RADIUS * 2 + 1;
const HISTOGRAM_BINS = 16;
/** Milliseconds the camera or controls stay still before the comparison is refreshed. */
const COMPARE_INTERVAL_MS = 500;

type Binning = 'grid' | 'hexagon';
type Smoothing = 'off' | 'gaussian-2d' | 'gaussian-separable';
type TimeWindow = 'all' | 'first' | 'middle' | 'last';

type DensityOutputs = {
  values: Buffer;
  extent: Buffer;
};

type DensityGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  outputs: DensityOutputs;
};

/**
 * Viewport-following density of every New York trip vertex. The bounds (and hexagon radius) are
 * rewritten from the camera each frame, the Gaussian smoothing sigma from a slider and the
 * trip-time mask from a select, all without recompiling. The grid compiles two smoothing graphs,
 * a dense 2D Gaussian and two separable 1D passes, so the panel can time them against each other
 * and read back the largest difference. Switching grid and hexagon is a compile-time change and
 * rebuilds the graphs.
 */
export const densityMode: SpatialAnalysisModeDefinition = {
  id: 'density',
  title: 'Density',
  contributors: ['GPUPointDensity'],
  description:
    'Heatmap of New York trip vertices binned on the GPU over the visible map. Pan and zoom: ' +
    'the grid follows the camera through a parameter buffer, never a recompile. Pick 2D or ' +
    'separable Gaussian smoothing and compare their GPU cost and difference.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(trips.origin);
    const resources = new SpatialAnalysisResources(device, 'density');
    const pointCount = trips.vertexTimestamps.length;
    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const maskBuffer = resources.createBuffer('mask', new Uint32Array(pointCount).fill(1));
    const bounds = resources.createParameterBuffer('bounds', 'float32', 4);
    const hexagonRadius = resources.createParameterBuffer(
      'hexagon-radius',
      'float32',
      1,
      Float32Array.of(50)
    );
    const kernel = resources.createParameterBuffer(
      'kernel',
      'float32',
      KERNEL_WIDTH * KERNEL_WIDTH
    );
    const lineKernel = resources.createParameterBuffer('line-kernel', 'float32', KERNEL_WIDTH);

    let binning: Binning = 'grid';
    let smoothing = 'gaussian-separable' as Smoothing;
    let sigma = 1.5;
    let timeWindow: TimeWindow = 'all';
    let showPoints = false;
    let currentHexagonRadius = 50;
    let destroyed = false;
    let measuring = false;
    let compareStale = true;
    let lastChangeTime = performance.now();
    let graphs: {dense?: DensityGraph; separable?: DensityGraph; hexagon?: DensityGraph} = {};

    writeKernels(kernel, lineKernel, smoothing === 'off' ? 0 : sigma);

    let compareReader: SummaryReader | null = null;
    let lastBounds: Float32Array | null = null;

    function getDisplayed(): DensityGraph | undefined {
      if (binning === 'hexagon') return graphs.hexagon;
      return smoothing === 'gaussian-separable' ? graphs.separable : graphs.dense;
    }

    function markChanged(): void {
      compareStale = true;
      lastChangeTime = performance.now();
    }

    function buildDensityGraph(
      id: string,
      gridSize: readonly [number, number],
      useSeparable: boolean
    ): DensityGraph {
      const cellCount = gridSize[0] * gridSize[1];
      const values = resources.createBuffer(`${id}-values`, cellCount * 4);
      const extent = resources.createBuffer(`${id}-extent`, 8);
      const histogram = resources.createBuffer(`${id}-histogram`, HISTOGRAM_BINS * 4);
      const graph = new GPUCommandGraph<void>(device, {id: `density-${id}`});
      const lineView = useSeparable ? lineKernel.importToGraph(graph) : undefined;
      graph.add(
        new GPUPointDensity({
          id: 'density',
          positions: importGraphBuffer(
            graph,
            'positions',
            positionsBuffer,
            'float32x2',
            pointCount
          ),
          mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', pointCount),
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
          statistic: 'count',
          output: {
            values: importGraphBuffer(graph, 'values', values, 'float32', cellCount),
            extent: importGraphBuffer(graph, 'extent', extent, 'float32', 2),
            histogram: importGraphBuffer(graph, 'histogram', histogram, 'uint32', HISTOGRAM_BINS)
          }
        })
      );
      return {compiled: resources.track(graph.compile()), outputs: {values, extent}};
    }

    function buildGraphs(): void {
      for (const graph of Object.values(graphs)) resources.release(graph.compiled);
      compareReader?.stop();
      const gridSize = binning === 'grid' ? GRID_SIZE : HEXAGON_GRID_SIZE;
      if (binning === 'grid') {
        graphs = {
          dense: buildDensityGraph('dense', gridSize, false),
          separable: buildDensityGraph('separable', gridSize, true)
        };
        const cellBytes = gridSize[0] * gridSize[1] * 4;
        compareReader = new SummaryReader(
          resources,
          'density-compare',
          [
            {buffer: graphs.dense!.outputs.values, size: cellBytes},
            {buffer: graphs.separable!.outputs.values, size: cellBytes}
          ],
          bytes => {
            if (destroyed) return;
            const cellCount = gridSize[0] * gridSize[1];
            const dense = new Float32Array(bytes, 0, cellCount);
            const separable = new Float32Array(bytes, cellCount * 4, cellCount);
            let maximumDifference = 0;
            let maximumValue = 0;
            for (let index = 0; index < cellCount; index++) {
              maximumDifference = Math.max(
                maximumDifference,
                Math.abs(dense[index] - separable[index])
              );
              maximumValue = Math.max(maximumValue, dense[index]);
            }
            differenceReadout.setValue(
              `${maximumDifference.toExponential(2)} (field max ${maximumValue.toFixed(2)})`
            );
          }
        );
      } else {
        graphs = {hexagon: buildDensityGraph('hexagon', gridSize, false)};
        compareReader = null;
        differenceReadout.setValue('n/a (hexagon has no smoothing)');
        denseTimeReadout.setValue('n/a');
        separableTimeReadout.setValue('n/a');
        speedupReadout.setValue('n/a');
      }
      gridReadout.setValue(
        `${gridSize[0]} × ${gridSize[1]} ${binning === 'grid' ? 'cells' : 'hexagons'}`
      );
      markChanged();
    }

    function writeMask(): void {
      const [start, end] = trips.timeRange;
      const third = (end - start) / 3;
      const windowStart =
        start + third * (timeWindow === 'middle' ? 1 : timeWindow === 'last' ? 2 : 0);
      const windowEnd = timeWindow === 'all' ? Infinity : windowStart + third;
      const mask = new Uint32Array(pointCount);
      let included = 0;
      for (let index = 0; index < pointCount; index++) {
        const time = trips.vertexTimestamps[index];
        const inside = timeWindow === 'all' || (time >= windowStart && time < windowEnd) ? 1 : 0;
        mask[index] = inside;
        included += inside;
      }
      maskBuffer.write(mask);
      includedReadout.setValue(`${formatCount(included)} of ${formatCount(pointCount)}`);
      markChanged();
    }

    /** Times both smoothing graphs outside the frame for the current sigma. */
    async function measureSmoothing(): Promise<void> {
      const {dense, separable} = graphs;
      if (measuring || destroyed || !dense || !separable) return;
      measuring = true;
      measureButton.setDisabled(true);
      try {
        const options = {parameters: undefined, completionBuffer: dense.outputs.extent};
        const denseTiming = await measureCompiledGraph(device, dense.compiled, options);
        const separableTiming = await measureCompiledGraph(device, separable.compiled, options);
        if (destroyed || graphs.dense !== dense) return;
        denseTimeReadout.setValue(formatCompiledGraphTiming(denseTiming));
        separableTimeReadout.setValue(formatCompiledGraphTiming(separableTiming));
        speedupReadout.setValue(
          formatSpeedup(denseTiming.milliseconds, separableTiming.milliseconds)
        );
      } catch {
        // Device destroyed or measurement aborted.
      } finally {
        measuring = false;
        if (!destroyed) measureButton.setDisabled(false);
      }
    }

    context.controls.addSelect<Binning>({
      label: 'Binning (compile-time)',
      options: [
        {value: 'grid', label: 'Square grid'},
        {value: 'hexagon', label: 'Hexagons'}
      ],
      value: binning,
      onChange: value => {
        binning = value;
        smoothingSelect.setDisabled(binning === 'hexagon');
        sigmaControl.setDisabled(binning === 'hexagon');
        buildGraphs();
        context.updateLayers();
      }
    });
    const smoothingSelect = context.controls.addSelect<Smoothing>({
      label: 'Smoothing (per-frame kernel buffers)',
      options: [
        {value: 'off', label: 'Off'},
        {value: 'gaussian-2d', label: '2D Gaussian (dense kernel)'},
        {value: 'gaussian-separable', label: 'Separable Gaussian (two 1D passes)'}
      ],
      value: smoothing,
      onChange: value => {
        smoothing = value;
        writeKernels(kernel, lineKernel, smoothing === 'off' ? 0 : sigma);
        markChanged();
        context.updateLayers();
      }
    });
    const sigmaControl = context.controls.addSlider({
      label: 'Gaussian sigma (cells)',
      min: 0.5,
      max: 2.5,
      step: 0.25,
      value: sigma,
      format: value => `${value.toFixed(2)} cells (radius ${getKernelRadius(value)})`,
      onChange: value => {
        sigma = value;
        writeKernels(kernel, lineKernel, smoothing === 'off' ? 0 : sigma);
        markChanged();
        scheduleMeasurement();
      }
    });
    context.controls.addSelect<TimeWindow>({
      label: 'Trip time window (mask buffer)',
      options: [
        {value: 'all', label: 'Whole period'},
        {value: 'first', label: 'First third'},
        {value: 'middle', label: 'Middle third'},
        {value: 'last', label: 'Last third'}
      ],
      value: timeWindow,
      onChange: value => {
        timeWindow = value;
        writeMask();
      }
    });
    context.controls.addToggle({
      label: 'Show source points',
      value: showPoints,
      onChange: value => {
        showPoints = value;
        context.updateLayers();
      }
    });
    const measureButton = context.controls.addButton({
      label: 'Time 2D vs separable smoothing',
      onClick: () => void measureSmoothing()
    });
    context.controls.addLegend({
      title: 'Points per cell (GPU extent, sqrt scale)',
      gradient: {
        colors: [
          [0, 0, 4],
          [120, 28, 109],
          [237, 105, 37],
          [252, 255, 164]
        ],
        minimumLabel: 'min',
        maximumLabel: 'max'
      }
    });
    context.controls.addReadout('Points', formatCount(pointCount));
    const includedReadout = context.controls.addReadout('Points in time window');
    const gridReadout = context.controls.addReadout('Grid');
    const cellReadout = context.controls.addReadout('Cell size');
    const denseTimeReadout = context.controls.addReadout('2D Gaussian graph', '...');
    const separableTimeReadout = context.controls.addReadout('Separable graph', '...');
    const speedupReadout = context.controls.addReadout('Separable vs 2D', '...');
    const differenceReadout = context.controls.addReadout('Max |2D - separable|', '...');
    context.controls.addNote(
      'Both graphs bin, smooth and reduce the same points for the same sigma; timed outside the frame (GPU timestamps when available, else wall clock). The difference is read back from both fields a moment after the camera or controls settle.'
    );
    context.controls.addReadout('Data', trips.attribution);

    let measureTimer: ReturnType<typeof setTimeout> | undefined;
    function scheduleMeasurement(): void {
      clearTimeout(measureTimer);
      measureTimer = setTimeout(() => void measureSmoothing(), 600);
    }

    writeMask();
    buildGraphs();
    scheduleMeasurement();

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () =>
        Object.values(graphs).map(graph => graph.compiled) as CompiledGPUCommandGraph<never>[],
      encode(commandEncoder, frame) {
        const displayed = getDisplayed();
        if (!displayed) return;
        const viewBounds = getViewportMetricBounds(frame.viewport, projection);
        if (binning === 'hexagon') {
          // Choose the radius so the compile-time lattice covers the viewport.
          const [columns, rows] = HEXAGON_GRID_SIZE;
          currentHexagonRadius = Math.max(
            (viewBounds[2] - viewBounds[0]) / (SQRT3 * (columns - 1)),
            (viewBounds[3] - viewBounds[1]) / (1.5 * (rows - 1))
          );
          hexagonRadius.write(Float32Array.of(currentHexagonRadius));
          cellReadout.setValue(`${currentHexagonRadius.toFixed(0)} m radius`);
        } else {
          cellReadout.setValue(`${((viewBounds[2] - viewBounds[0]) / GRID_SIZE[0]).toFixed(0)} m`);
        }
        const previousBounds = lastBounds;
        const boundsData = Float32Array.from(viewBounds);
        if (!previousBounds || boundsData.some((value, index) => value !== previousBounds[index])) {
          markChanged();
        }
        lastBounds = boundsData;
        bounds.write(boundsData);
        displayed.compiled.encode(commandEncoder, {parameters: undefined});
        if (graphs.dense && graphs.separable && compareReader) {
          const settled = performance.now() - lastChangeTime > COMPARE_INTERVAL_MS;
          if (compareStale && settled && !compareReader.isPending) {
            // Encode the other smoothing graph too so both fields share these bounds.
            const other = displayed === graphs.dense ? graphs.separable : graphs.dense;
            other.compiled.encode(commandEncoder, {parameters: undefined});
            compareStale = false;
            compareReader.request(commandEncoder);
          } else {
            compareReader.flush(commandEncoder);
          }
        }
      },
      getLayers() {
        const outputs = getDisplayed()?.outputs;
        if (!outputs) return [];
        const layers: Layer[] = [
          new SpatialAnalysisRasterLayer({
            id: `density-${binning}`,
            coordinateOrigin: [trips.origin[0], trips.origin[1], 0],
            gridSize: binning === 'grid' ? GRID_SIZE : HEXAGON_GRID_SIZE,
            bounds: bounds.buffer,
            binning,
            hexagonRadius: hexagonRadius.buffer,
            values: outputs.values,
            valueFormat: 'float32',
            extent: outputs.extent,
            colormap: 'inferno',
            sqrtScale: true,
            discardAtOrBelow: 0,
            color: [255, 255, 255, 215]
          })
        ];
        if (showPoints) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'density-points',
              coordinateOrigin: [trips.origin[0], trips.origin[1], 0],
              positions: positionsBuffer,
              instanceCount: pointCount,
              radiusPixels: 1.2,
              color: [120, 220, 255, 140]
            })
          );
        }
        return layers;
      },
      destroy: () => {
        destroyed = true;
        clearTimeout(measureTimer);
        compareReader?.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};

/** Radius in cells that holds three sigma, clamped to the compile-time kernel. */
function getKernelRadius(sigma: number): number {
  return Math.min(KERNEL_RADIUS, Math.ceil(3 * sigma));
}

/** Writes the 2D and 1D Gaussians for `sigma` (0 writes a unit impulse, smoothing off). */
function writeKernels(
  kernel: GPUParameterBuffer<'float32'>,
  lineKernel: GPUParameterBuffer<'float32'>,
  sigma: number
): void {
  // Embed a small Gaussian in the compile-time kernel; unused taps are zero.
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
