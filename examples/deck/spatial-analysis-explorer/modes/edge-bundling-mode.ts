// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Kernel-density edge bundling of New York taxi origin-destination pairs. Trip endpoints (and each
 * trip's midpoint) are snapped to a 300 m lattice, which gives shared nodes and a deduplicated
 * edge list. One compiled `GPUEdgeBundling` graph turns the straight edges into bundled polylines
 * on the GPU; iterations, kernel radius, smoothing (stiffness) and step scale are parameter-buffer
 * values, so moving a slider re-encodes the same graph. The GPU gates every iteration beyond the
 * active count, so the cost follows the slider rather than the compiled maximum.
 */

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUEdgeBundling,
  createGPUEdgeBundlingParameterValues,
  GPU_EDGE_BUNDLING_DEFAULTS,
  type GPUEdgeBundlingParameterValues
} from '@luma.gl/experimental/gpu-network';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {BundledPathLayer} from './edge-bundling-layers';
import {SummaryReader} from './summary-reader';

/** Compile-time iteration capacity; the slider picks how many run. */
const MAXIMUM_ITERATIONS = 32;
const POINTS_PER_EDGE = 16;
const DENSITY_RESOLUTION = 256;
/** Snap lattice cell size in meters. */
const CELL_SIZE_METERS = 300;
const MAXIMUM_EDGES = 6000;
const START_COLOR = [255, 150, 60, 255] as const;
const END_COLOR = [60, 220, 255, 255] as const;

export const edgeBundlingMode: SpatialAnalysisModeDefinition = {
  id: 'edge-bundling',
  title: 'Edge bundling',
  contributors: ['GPUEdgeBundling'],
  description:
    'Taxi origin-destination pairs bundled by kernel-density edge bundling on the GPU. Slide the ' +
    'iterations from 0 (straight lines) upward and watch parallel flows merge into corridors; ' +
    'stiffness and kernel radius reshape the bundles. Every slider writes a parameter buffer and ' +
    're-encodes the same compiled graph.',
  initialViewState: {longitude: -73.97, latitude: 40.74, zoom: 11.7},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(trips.origin);
    const resources = new SpatialAnalysisResources(device, 'edge-bundling');

    // Snap origin, midpoint and destination of every trip to lattice cells; connect the three
    // pairs, drop self-pairs and duplicates, and keep the most frequent pairs.
    const cellIds = new Map<string, number>();
    const cellCenters: number[] = [];
    const getCell = (x: number, y: number): number => {
      const column = Math.round(x / CELL_SIZE_METERS);
      const row = Math.round(y / CELL_SIZE_METERS);
      const key = `${column},${row}`;
      let id = cellIds.get(key);
      if (id === undefined) {
        id = cellCenters.length / 2;
        cellIds.set(key, id);
        // Lon/lat degrees: the contributor scales longitude by cos(latitude) itself.
        cellCenters.push(
          ...projection.unproject(column * CELL_SIZE_METERS, row * CELL_SIZE_METERS)
        );
      }
      return id;
    };
    const pairCounts = new Map<number, number>();
    const tripCount = trips.vendors.length;
    // The dataset has far-away outlier endpoints (other cities) that would stretch the work box,
    // so flows with an endpoint outside the 1st to 99th percentile per axis are skipped.
    const bounds = getRobustBounds(trips, tripCount);
    const isInside = (vertex: number) => {
      const x = trips.vertexPositions[vertex * 2];
      const y = trips.vertexPositions[vertex * 2 + 1];
      return x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3];
    };
    for (let trip = 0; trip < tripCount; trip++) {
      const first = trips.tripOffsets[trip];
      const last = trips.tripOffsets[trip + 1] - 1;
      const middle = (first + last) >> 1;
      if (!isInside(first) || !isInside(middle) || !isInside(last)) continue;
      const cells = [first, middle, last].map(vertex =>
        getCell(trips.vertexPositions[vertex * 2], trips.vertexPositions[vertex * 2 + 1])
      );
      for (const [a, b] of [
        [0, 2],
        [0, 1],
        [1, 2]
      ]) {
        if (cells[a] === cells[b]) continue;
        const key = cells[a] * 0x100000 + cells[b];
        pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
      }
    }
    const ranked = [...pairCounts.entries()]
      .sort((left, right) => right[1] - left[1] || left[0] - right[0])
      .slice(0, MAXIMUM_EDGES);
    const edgeCount = Math.max(ranked.length, 1);
    const sources = new Uint32Array(edgeCount);
    const targets = new Uint32Array(edgeCount);
    ranked.forEach(([key], edge) => {
      sources[edge] = Math.floor(key / 0x100000);
      targets[edge] = key % 0x100000;
    });
    if (cellCenters.length === 0) cellCenters.push(trips.origin[0], trips.origin[1]);
    const vertexCount = cellCenters.length / 2;

    const positions = resources.createBuffer('positions', Float32Array.from(cellCenters));
    const sourceBuffer = resources.createBuffer('sources', sources);
    const targetBuffer = resources.createBuffer('targets', targets);
    const paths = resources.createBuffer('paths', edgeCount * POINTS_PER_EDGE * 8);
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'edge-bundling-draw',
        type: 'draw',
        commands: [{vertexCount: POINTS_PER_EDGE, instanceCount: 0}]
      })
    );

    let parameterValues: GPUEdgeBundlingParameterValues = {
      activeIterations: 15,
      kernelRadius: GPU_EDGE_BUNDLING_DEFAULTS.kernelRadius,
      lambda: GPU_EDGE_BUNDLING_DEFAULTS.lambda,
      smoothing: GPU_EDGE_BUNDLING_DEFAULTS.smoothing,
      stepScale: GPU_EDGE_BUNDLING_DEFAULTS.stepScale
    };
    const parameters = resources.createParameterBuffer(
      'parameters',
      'uint32',
      5,
      createGPUEdgeBundlingParameterValues(parameterValues, 'uint32')
    );

    const graph = new GPUCommandGraph<void>(device, {id: 'edge-bundling'});
    graph.add(
      new GPUEdgeBundling({
        id: 'edge-bundling',
        positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', vertexCount),
        sourceVertices: importGraphBuffer(graph, 'sources', sourceBuffer, 'uint32', edgeCount),
        targetVertices: importGraphBuffer(graph, 'targets', targetBuffer, 'uint32', edgeCount),
        geographic: true,
        pointsPerEdge: POINTS_PER_EDGE,
        iterations: MAXIMUM_ITERATIONS,
        densityResolution: DENSITY_RESOLUTION,
        parameters: parameters.importToGraph(graph),
        paths: importGraphBuffer(graph, 'paths', paths, 'float32x2', edgeCount * POINTS_PER_EDGE),
        drawRecord: drawCommands.importToGraph(graph).words
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    let dirty = true;
    const writeParameters = (changes: GPUEdgeBundlingParameterValues) => {
      parameterValues = {...parameterValues, ...changes};
      parameters.write(createGPUEdgeBundlingParameterValues(parameterValues, 'uint32'));
      dirty = true;
    };

    const record = context.controls.addReadout('Draw record (GPU)');
    const reader = new SummaryReader(
      resources,
      'edge-bundling',
      [{buffer: drawCommands.buffer, size: 16}],
      bytes => {
        const words = new Uint32Array(bytes);
        record.setValue(`${words[0]} points x ${formatCount(words[1])} edges`);
      }
    );

    context.controls.addSlider({
      label: 'Iterations (GPU-gated)',
      min: 0,
      max: MAXIMUM_ITERATIONS,
      step: 1,
      value: parameterValues.activeIterations ?? 0,
      format: value => (value === 0 ? '0 (straight)' : String(value)),
      onChange: value => writeParameters({activeIterations: value})
    });
    context.controls.addSlider({
      label: 'Stiffness (Laplacian smoothing)',
      min: 0,
      max: 1,
      step: 0.05,
      value: parameterValues.smoothing ?? 0.5,
      format: value => value.toFixed(2),
      onChange: value => writeParameters({smoothing: value})
    });
    context.controls.addSlider({
      label: 'Kernel radius (box fraction)',
      min: 0.005,
      max: 0.08,
      step: 0.005,
      value: parameterValues.kernelRadius ?? 0.03,
      format: value => value.toFixed(3),
      onChange: value => writeParameters({kernelRadius: value})
    });
    context.controls.addSlider({
      label: 'Step scale',
      min: 0.25,
      max: 2,
      step: 0.05,
      value: parameterValues.stepScale ?? 1,
      format: value => value.toFixed(2),
      onChange: value => writeParameters({stepScale: value})
    });
    context.controls.addLegend({
      title: 'Flow direction',
      gradient: {
        colors: [START_COLOR, END_COLOR],
        minimumLabel: 'Origin',
        maximumLabel: 'Destination'
      }
    });
    context.controls.addReadout('Places (lattice nodes)', formatCount(vertexCount));
    context.controls.addReadout('Edges (OD pairs)', formatCount(ranked.length));
    context.controls.addReadout('Control points', formatCount(ranked.length * POINTS_PER_EDGE));
    context.controls.addReadout('Data', trips.attribution);

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        // Paths persist in their buffer, so the graph only re-encodes when a parameter changed.
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
        }
        if (frame.frameIndex === 1) reader.request(commandEncoder);
        reader.flush(commandEncoder);
      },
      getLayers() {
        const layers: Layer[] = [
          new BundledPathLayer({
            id: 'edge-bundling-paths',
            paths,
            drawCommands,
            pointsPerPath: POINTS_PER_EDGE,
            startColor: START_COLOR,
            endColor: END_COLOR,
            opacity: 0.3
          })
        ];
        return layers;
      },
      destroy() {
        resources.destroy();
      }
    };
    return instance;
  }
};

/** Per-axis 1st and 99th percentile of every trip's first and last vertex, in planar meters. */
function getRobustBounds(
  trips: {vertexPositions: Float32Array; tripOffsets: Uint32Array},
  tripCount: number
): [number, number, number, number] {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let trip = 0; trip < tripCount; trip++) {
    for (const vertex of [trips.tripOffsets[trip], trips.tripOffsets[trip + 1] - 1]) {
      xs.push(trips.vertexPositions[vertex * 2]);
      ys.push(trips.vertexPositions[vertex * 2 + 1]);
    }
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  const quantile = (values: number[], fraction: number) =>
    values[Math.round(fraction * (values.length - 1))] ?? 0;
  return [quantile(xs, 0.01), quantile(ys, 0.01), quantile(xs, 0.99), quantile(ys, 0.99)];
}
