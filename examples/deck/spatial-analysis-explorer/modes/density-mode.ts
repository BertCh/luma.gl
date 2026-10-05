// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  createGPUPointDensityGaussianKernel,
  GPUPointDensity
} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer} from '../graph-buffers';
import {type GPUParameterBuffer} from '@luma.gl/experimental/geospatial';
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

const GRID_SIZE: readonly [number, number] = [160, 100];
const HEXAGON_GRID_SIZE: readonly [number, number] = [72, 52];
const SQRT3 = Math.sqrt(3);
/** Compile-time kernel size; per-frame weights select the actual smoothing radius. */
const KERNEL_RADIUS = 4;
const KERNEL_WIDTH = KERNEL_RADIUS * 2 + 1;
const HISTOGRAM_BINS = 16;

type Binning = 'grid' | 'hexagon';

/**
 * Viewport-following density of every New York trip vertex. The bounds (and hexagon radius) are
 * rewritten from the camera each frame and the Gaussian smoothing kernel from a slider, all
 * without recompiling. Switching grid and hexagon is a compile-time change and rebuilds the graph.
 */
export const densityMode: SpatialAnalysisModeDefinition = {
  id: 'density',
  title: 'Density',
  contributors: ['GPUPointDensity'],
  description:
    'Heatmap of New York trip vertices binned on the GPU over the visible map. Pan and zoom: ' +
    'the grid follows the camera through a parameter buffer, never a recompile.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(trips.origin);
    const resources = new SpatialAnalysisResources(device, 'density');
    const pointCount = trips.vertexTimestamps.length;
    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
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

    let binning: Binning = 'grid';
    let smoothingRadius = 1;
    let showPoints = false;
    let currentHexagonRadius = 50;
    let compiled: CompiledGPUCommandGraph<void> | null = null;
    let outputs: {
      values: ReturnType<SpatialAnalysisResources['createBuffer']>;
      extent: ReturnType<SpatialAnalysisResources['createBuffer']>;
    } | null = null;

    writeKernel(kernel, smoothingRadius);

    function buildGraph(): void {
      if (compiled) resources.release(compiled);
      const gridSize = binning === 'grid' ? GRID_SIZE : HEXAGON_GRID_SIZE;
      const cellCount = gridSize[0] * gridSize[1];
      const values = resources.createBuffer(`${binning}-values`, cellCount * 4);
      const extent = resources.createBuffer(`${binning}-extent`, 8);
      const histogram = resources.createBuffer(`${binning}-histogram`, HISTOGRAM_BINS * 4);
      const graph = new GPUCommandGraph<void>(device, {id: `density-${binning}`});
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
          bounds: bounds.importToGraph(graph),
          gridSize,
          binning,
          ...(binning === 'hexagon'
            ? {hexagonRadius: hexagonRadius.importToGraph(graph)}
            : {
                smoothing: {
                  kernel: kernel.importToGraph(graph),
                  kernelWidth: KERNEL_WIDTH,
                  kernelHeight: KERNEL_WIDTH
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
      compiled = resources.track(graph.compile());
      outputs = {values, extent};
      gridReadout.setValue(
        `${gridSize[0]} × ${gridSize[1]} ${binning === 'grid' ? 'cells' : 'hexagons'}`
      );
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
        smoothingControl.setDisabled(binning === 'hexagon');
        buildGraph();
        context.updateLayers();
      }
    });
    const smoothingControl = context.controls.addSlider({
      label: 'Gaussian smoothing radius (per-frame kernel)',
      min: 0,
      max: KERNEL_RADIUS,
      step: 1,
      value: smoothingRadius,
      format: value => (value === 0 ? 'off' : `${value} cells`),
      onChange: value => {
        smoothingRadius = value;
        writeKernel(kernel, smoothingRadius);
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
    const gridReadout = context.controls.addReadout('Grid');
    const cellReadout = context.controls.addReadout('Cell size');
    context.controls.addReadout('Data', trips.attribution);

    buildGraph();

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => (compiled ? [compiled] : []),
      encode(commandEncoder, frame) {
        if (!compiled) return;
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
        bounds.write(Float32Array.from(viewBounds));
        compiled.encode(commandEncoder, {parameters: undefined});
      },
      getLayers() {
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
      destroy: () => resources.destroy()
    };
    return instance;
  }
};

function writeKernel(kernel: GPUParameterBuffer<'float32'>, radius: number): void {
  // Embed a radius-r Gaussian inside the compile-time kernel; unused taps are zero.
  const small = createGPUPointDensityGaussianKernel(radius);
  const size = radius * 2 + 1;
  const weights = new Float32Array(KERNEL_WIDTH * KERNEL_WIDTH);
  const offset = KERNEL_RADIUS - radius;
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      weights[(row + offset) * KERNEL_WIDTH + column + offset] = small[row * size + column];
    }
  }
  kernel.write(weights);
}
