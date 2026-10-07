// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH,
  GPU_NETWORK_REACHABILITY_NONE,
  getGPUNetworkIsochroneParameterValues,
  GPUNetworkCostMatrix,
  GPUNetworkIsochrones,
  GPUNetworkReachability
} from '@luma.gl/experimental/gpu-network';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {IsobandTriangleLayer} from './b9-network-layers';
import {createPackedPalette} from './b9-shared';
import {findNearestStation, findStation, loadRailGraph, TRANSIT_HUB_NAMES} from './transit-data';

/** Option state of the transit reachability scene. */
export type TransitReachabilityOptions = {
  origin: string;
  trainTypes: 'all' | 'fast' | 'stopping';
  dwellSeconds: number;
  costLimitMinutes: number;
  bandMinutes: '15' | '20' | '30';
  bandCount: number;
  lastMile: 'none' | 'walk' | 'bike';
  lastMileMinutes: number;
  localIterations: '16' | '32' | '64';
  showBands: boolean;
  showTree: boolean;
  showEdges: boolean;
  showStations: boolean;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
};

/** Speeds of the last mile in meters per second. */
export const LAST_MILE_SPEEDS = {walk: 1.34, bike: 4.2} as const;

const MAXIMUM_BREAKS = 4;
const MAXIMUM_ROUNDS = 16;
const MATRIX_ROUNDS = 16;
const RASTER_WIDTH = 800;
const MAXIMUM_BUFFER_PIXELS = 16;
const TRIANGLE_CAPACITY = 1_000_000;
const PALETTE_SIZE = 64;
const BAND_ALPHA = 170;
const EXTENT_PADDING = 12_000;
/** Value of the `origin` option while the origin is a clicked station. */
const CUSTOM_ORIGIN = 'clicked';
const DISABLED_EDGE_ALPHA = 0.08;

type ReachGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  localIterations: number;
};

/** Imports every buffer once per graph (a graph rejects two imports of one buffer). */
function createImporter(graph: GPUCommandGraph<void>) {
  const cache = new Map<Buffer, GraphDataView>();
  return <Format extends 'uint32' | 'float32' | 'float32x2'>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    let view = cache.get(buffer);
    if (!view) {
      view = importGraphBuffer(graph, buffer.id, buffer, format, length) as GraphDataView;
      cache.set(buffer, view);
    }
    return view as unknown as GraphDataView<Format>;
  };
}

/**
 * Transit reachability. `GPUNetworkReachability` runs a single-source shortest-path search over
 * the scheduled rail graph (costs in seconds), `GPUNetworkIsochrones` splats every station's cost
 * to a raster with a walking or cycling buffer and contours it into bands, and
 * `GPUNetworkCostMatrix` computes the travel time between fourteen hub stations and every other
 * station. Train types and dwell are edge-weight writes; nothing recompiles.
 */
export async function createTransitReachability(
  ctx: SceneContext<TransitReachabilityOptions>
): Promise<SceneInstance<TransitReachabilityOptions>> {
  const rail = loadRailGraph(ctx.datasets.get('gtfs-nl-rail-graph'));
  const {device} = ctx;
  const {nodeCount, edgeCount} = rail;
  const resources = new SpatialAnalysisResources(device, 'reach');
  const coordinateOrigin: [number, number, number] = [rail.origin[0], rail.origin[1], 0];

  const hubNodes = TRANSIT_HUB_NAMES.map(name => findStation(rail, name)).filter(node => node >= 0);
  const hubLabels = hubNodes.map(node => rail.names[node].replace(/ Centraal$/, ''));
  const hubCount = hubNodes.length;

  // Raster extent: the stations plus room for a last-mile buffer.
  const extent: [number, number, number, number] = [
    rail.bounds[0] - EXTENT_PADDING,
    rail.bounds[1] - EXTENT_PADDING,
    rail.bounds[2] + EXTENT_PADDING,
    rail.bounds[3] + EXTENT_PADDING
  ];
  const rasterHeight = Math.round(
    (RASTER_WIDTH * (extent[3] - extent[1])) / (extent[2] - extent[0])
  );
  const pixelSize = (extent[2] - extent[0]) / RASTER_WIDTH;

  // ---- Static buffers ---------------------------------------------------------------------------
  const offsetsBuffer = resources.createBuffer('offsets', rail.offsets);
  const neighborsBuffer = resources.createBuffer('neighbors', rail.targets);
  const weightsBuffer = resources.createBuffer('weights', edgeCount * 4);
  const nodePositionsBuffer = resources.createBuffer('node-positions', rail.nodePositions);
  const segmentsBuffer = resources.createBuffer('segments', rail.segments);
  const edgeAlphaBuffer = resources.createBuffer('edge-alpha', edgeCount * 4);
  // Station-only CSR for the isochrone splat: one zero-length self edge per station, so every
  // station is sampled once at its own cost and the line between stations is not "reachable".
  const selfOffsets = Uint32Array.from({length: nodeCount + 1}, (_, index) => index);
  const selfNeighbors = Uint32Array.from({length: nodeCount}, (_, index) => index);
  const selfOffsetsBuffer = resources.createBuffer('self-offsets', selfOffsets);
  const selfNeighborsBuffer = resources.createBuffer('self-neighbors', selfNeighbors);
  const selfWeightsBuffer = resources.createBuffer('self-weights', new Float32Array(nodeCount));
  const hubNodesBuffer = resources.createBuffer('hub-nodes', Uint32Array.from(hubNodes));
  const identityBuffer = resources.createBuffer(
    'identity',
    Uint32Array.from({length: nodeCount}, (_, index) => index)
  );
  const treeSegmentsBuffer = resources.createBuffer('tree-segments', nodeCount * 16);
  const originPositionBuffer = resources.createBuffer('origin-position', 8);

  // ---- Parameter buffers ------------------------------------------------------------------------
  const originParameter = resources.createParameterBuffer('origin', 'uint32', 1);
  const costLimitParameter = resources.createParameterBuffer('cost-limit', 'float32', 1);
  const breaksParameter = resources.createParameterBuffer('breaks', 'float32', MAXIMUM_BREAKS);
  const isochroneParameters = resources.createParameterBuffer(
    'isochrone-parameters',
    'float32',
    GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH
  );
  const paletteBuffer = resources.createBuffer(
    'palette',
    createPackedPalette(ctx.options.ramp, PALETTE_SIZE, BAND_ALPHA)
  );

  // ---- Outputs ----------------------------------------------------------------------------------
  const costsBuffer = resources.createBuffer('costs', nodeCount * 4);
  const predecessorsBuffer = resources.createBuffer('predecessors', nodeCount * 4);
  const convergedBuffer = resources.createBuffer('converged', 4);
  const iterationsBuffer = resources.createBuffer('iterations', 4);
  const triangles = resources.createBuffer('triangles', TRIANGLE_CAPACITY * 3 * 8);
  const triangleBands = resources.createBuffer('triangle-bands', TRIANGLE_CAPACITY * 4);
  const triangleCount = resources.createBuffer('triangle-count', 4);
  const triangleOverflow = resources.createBuffer('triangle-overflow', 4);
  const bandVertexCount = resources.createBuffer('band-vertex-count', 4);
  const matrixBuffer = resources.createBuffer('matrix', Math.max(hubCount, 1) * nodeCount * 4);
  const matrixConverged = resources.createBuffer('matrix-converged', 4);
  const drawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'reach-bands-draw',
      type: 'draw',
      commands: [{vertexCount: 0, instanceCount: 1}]
    })
  );

  // ---- Graphs -----------------------------------------------------------------------------------
  let reach: ReachGraph | null = null;

  function buildReachGraph(localIterations: number): ReachGraph {
    if (reach) resources.release(reach.compiled);
    const graph = new GPUCommandGraph<void>(device, {id: `reach-${localIterations}`});
    const view = createImporter(graph);
    const costs = view(costsBuffer, 'float32', nodeCount);
    graph.add(
      new GPUNetworkReachability({
        id: 'search',
        offsets: view(offsetsBuffer, 'uint32', nodeCount + 1),
        neighbors: view(neighborsBuffer, 'uint32', edgeCount),
        weights: view(weightsBuffer, 'float32', edgeCount),
        sources: originParameter.importToGraph(graph),
        costLimit: costLimitParameter.importToGraph(graph),
        maxIterations: MAXIMUM_ROUNDS,
        localIterations,
        costs,
        predecessors: view(predecessorsBuffer, 'uint32', nodeCount),
        converged: view(convergedBuffer, 'uint32', 1),
        iterationCount: view(iterationsBuffer, 'uint32', 1)
      })
    );
    graph.add(
      new GPUNetworkIsochrones({
        id: 'bands',
        offsets: view(selfOffsetsBuffer, 'uint32', nodeCount + 1),
        neighbors: view(selfNeighborsBuffer, 'uint32', nodeCount),
        weights: view(selfWeightsBuffer, 'float32', nodeCount),
        nodePositions: view(nodePositionsBuffer, 'float32x2', nodeCount),
        costs,
        breaks: breaksParameter.importToGraph(graph),
        parameters: isochroneParameters.importToGraph(graph),
        raster: {
          width: RASTER_WIDTH,
          height: rasterHeight,
          mode: 'min',
          maximumBufferPixels: MAXIMUM_BUFFER_PIXELS,
          maximumSamplesPerEdge: 2,
          output: {
            triangles: view(triangles, 'float32x2', TRIANGLE_CAPACITY * 3),
            triangleBands: view(triangleBands, 'uint32', TRIANGLE_CAPACITY),
            count: view(triangleCount, 'uint32', 1),
            overflow: view(triangleOverflow, 'uint32', 1),
            vertexCount: view(bandVertexCount, 'uint32', 1)
          }
        }
      })
    );
    reach = {compiled: resources.track(graph.compile()), localIterations};
    return reach;
  }

  const matrixGraph = new GPUCommandGraph<void>(device, {id: 'reach-matrix'});
  {
    const view = createImporter(matrixGraph);
    matrixGraph.add(
      new GPUNetworkCostMatrix({
        id: 'hub-matrix',
        offsets: view(offsetsBuffer, 'uint32', nodeCount + 1),
        neighbors: view(neighborsBuffer, 'uint32', edgeCount),
        weights: view(weightsBuffer, 'float32', edgeCount),
        seedNodes: view(hubNodesBuffer, 'uint32', Math.max(hubCount, 1)),
        laneCount: Math.max(hubCount, 1),
        maxIterations: MATRIX_ROUNDS,
        localIterations: 16,
        costs: view(matrixBuffer, 'float32', Math.max(hubCount, 1) * nodeCount),
        converged: view(matrixConverged, 'uint32', 1)
      })
    );
  }
  const matrixCompiled = resources.track(matrixGraph.compile());

  // ---- State ------------------------------------------------------------------------------------
  let destroyed = false;
  let reachDirty = 2;
  let matrixDirty = 2;
  let originNode = Math.max(0, hubNodes[hubLabels.indexOf('Utrecht')] ?? hubNodes[0] ?? 0);
  let customNode = originNode;
  let costs = new Float32Array(nodeCount).fill(Number.POSITIVE_INFINITY);
  let matrix = new Float32Array(Math.max(hubCount, 1) * nodeCount);
  let paletteRamp = ctx.options.ramp;
  let builtOptions = {local: ctx.options.localIterations};

  const hubValue = (label: string) => `hub:${label}`;
  const originFromOptions = (value: string): number => {
    if (value === CUSTOM_ORIGIN) return customNode;
    const index = hubLabels.findIndex(label => hubValue(label) === value);
    return index >= 0 ? hubNodes[index] : originNode;
  };

  function writeWeights(): void {
    const options = ctx.options;
    const weights = new Float32Array(edgeCount);
    const alpha = new Float32Array(edgeCount);
    for (let edge = 0; edge < edgeCount; edge++) {
      const cls = rail.edgeClass[edge];
      const allowed =
        options.trainTypes === 'all' ||
        (options.trainTypes === 'fast' && cls <= 1) ||
        (options.trainTypes === 'stopping' && cls === 2);
      // Negative weights are impassable. Dwell is added at every station passed.
      weights[edge] = allowed ? rail.travelTime[edge] + options.dwellSeconds : -1;
      alpha[edge] = allowed ? 0.9 : DISABLED_EDGE_ALPHA;
    }
    weightsBuffer.write(weights);
    edgeAlphaBuffer.write(alpha);
    reachDirty = Math.max(reachDirty, 1);
    matrixDirty = Math.max(matrixDirty, 1);
  }

  function writeOrigin(): void {
    originParameter.write(Uint32Array.of(originNode));
    originPositionBuffer.write(rail.nodePositions.slice(originNode * 2, originNode * 2 + 2));
    ctx.setReadout('origin', rail.names[originNode]);
    reachDirty = Math.max(reachDirty, 1);
    updateHubChart();
  }

  function getBreaks(): number[] {
    const options = ctx.options;
    const minutes = Number(options.bandMinutes);
    return Array.from(
      {length: MAXIMUM_BREAKS},
      (_, band) => Math.min(band + 1, options.bandCount) * minutes * 60
    );
  }

  function writeBudget(): void {
    const options = ctx.options;
    const breaks = getBreaks();
    breaksParameter.write(Float32Array.from(breaks));
    costLimitParameter.write(Float32Array.of(options.costLimitMinutes * 60));
    const speed = options.lastMile === 'none' ? 0 : LAST_MILE_SPEEDS[options.lastMile];
    const wanted = speed * options.lastMileMinutes * 60;
    const effective = Math.min(wanted, MAXIMUM_BUFFER_PIXELS * pixelSize);
    isochroneParameters.write(
      getGPUNetworkIsochroneParameterValues({
        breakCount: options.bandCount,
        extent,
        bufferRadius: effective,
        walkCostPerUnit: speed > 0 ? 1 / speed : 0
      })
    );
    ctx.setReadout(
      'lastMile',
      speed === 0
        ? 'stations only'
        : `${(wanted / 1000).toFixed(1)} km${effective < wanted ? ` (capped at ${(effective / 1000).toFixed(1)} km by the buffer limit)` : ''}`
    );
    reachDirty = Math.max(reachDirty, 1);
  }

  function writePalette(): void {
    if (paletteRamp === ctx.options.ramp) return;
    paletteRamp = ctx.options.ramp;
    paletteBuffer.write(createPackedPalette(paletteRamp, PALETTE_SIZE, BAND_ALPHA));
  }

  const formatMinutes = (seconds: number) =>
    Number.isFinite(seconds) ? `${Math.round(seconds / 60)} min` : 'not reached';

  function updateHubChart(): void {
    if (!hubCount) {
      ctx.setChart('hubTimes', null);
      return;
    }
    const values = hubLabels.map((_, hub) => {
      const seconds = matrix[hub * nodeCount + originNode];
      return Number.isFinite(seconds) ? seconds / 60 : 0;
    });
    const hubOfOrigin = hubNodes.indexOf(originNode);
    ctx.setChart('hubTimes', {
      kind: 'bars',
      height: 150,
      values,
      labels: hubLabels,
      highlight: hubOfOrigin >= 0 ? [hubOfOrigin] : undefined,
      yLabel: 'minutes',
      formatY: value => value.toFixed(0),
      description: `Scheduled rail travel time between ${rail.names[originNode]} and each hub station, from the cost matrix. A zero bar is the station itself or no connection.`
    });
  }

  function updateReachChart(): void {
    const options = ctx.options;
    const horizon = Math.max(options.costLimitMinutes, 30);
    const step = 5;
    const x: number[] = [];
    const y: number[] = [];
    for (let minutes = 0; minutes <= horizon; minutes += step) {
      let count = 0;
      for (let node = 0; node < nodeCount; node++) if (costs[node] <= minutes * 60) count++;
      x.push(minutes);
      y.push(count);
    }
    const minutesPerBand = Number(options.bandMinutes);
    ctx.setChart('reachCurve', {
      kind: 'line',
      height: 130,
      xLabel: 'rail travel time (min)',
      yLabel: 'stations reached',
      xDomain: [0, horizon],
      formatX: value => value.toFixed(0),
      formatY: value => formatCount(value),
      markers: Array.from({length: options.bandCount}, (_, band) => ({
        x: (band + 1) * minutesPerBand
      })).filter(marker => marker.x <= horizon),
      series: [{label: 'stations', x, y, area: true}],
      description: `Stations reachable within each travel time from ${rail.names[originNode]}; vertical lines are the isochrone breaks.`
    });
  }

  // ---- Readbacks --------------------------------------------------------------------------------
  const reachReader = new SummaryReader(
    resources,
    'reach-costs',
    [
      {buffer: costsBuffer, size: nodeCount * 4},
      {buffer: predecessorsBuffer, size: nodeCount * 4},
      {buffer: convergedBuffer, size: 4},
      {buffer: iterationsBuffer, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const floats = new Float32Array(bytes);
      const words = new Uint32Array(bytes);
      costs = floats.slice(0, nodeCount);
      const predecessors = words.slice(nodeCount, nodeCount * 2);
      const converged = words[nodeCount * 2];
      const rounds = words[nodeCount * 2 + 1];
      // Fastest-route tree: one segment from each station's predecessor to the station.
      const tree = new Float32Array(nodeCount * 4).fill(Number.NaN);
      for (let node = 0; node < nodeCount; node++) {
        const predecessor = predecessors[node];
        if (predecessor === GPU_NETWORK_REACHABILITY_NONE || predecessor >= nodeCount) continue;
        tree.set(rail.nodePositions.subarray(predecessor * 2, predecessor * 2 + 2), node * 4);
        tree.set(rail.nodePositions.subarray(node * 2, node * 2 + 2), node * 4 + 2);
      }
      treeSegmentsBuffer.write(tree);
      const breaks = getBreaks();
      const withinBreak = new Array<number>(ctx.options.bandCount).fill(0);
      let farthest = 0;
      let farthestNode = originNode;
      let reached = 0;
      for (let node = 0; node < nodeCount; node++) {
        const cost = costs[node];
        if (!Number.isFinite(cost)) continue;
        reached++;
        if (cost > farthest) {
          farthest = cost;
          farthestNode = node;
        }
        for (let band = 0; band < withinBreak.length; band++) {
          if (cost <= breaks[band]) withinBreak[band]++;
        }
      }
      ctx.setReadout('reached', `${formatCount(reached)} of ${formatCount(nodeCount)} stations`);
      ctx.setReadout(
        'bandTable',
        withinBreak
          .map(
            (count, band) =>
              `within ${String(Math.round(breaks[band] / 60)).padStart(3)} min  ${formatCount(count).padStart(4)} stations`
          )
          .join('\n')
      );
      ctx.setReadout('farthest', `${rail.names[farthestNode]}, ${formatMinutes(farthest)}`);
      ctx.setReadout(
        'solver',
        `${rounds} rounds${converged ? ', converged' : ', NOT converged (raise local iterations)'}`
      );
      updateReachChart();
      ctx.requestLayers();
    }
  );

  const matrixReader = new SummaryReader(
    resources,
    'reach-matrix',
    [
      {buffer: matrixBuffer, size: Math.max(hubCount, 1) * nodeCount * 4},
      {buffer: matrixConverged, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      matrix = new Float32Array(bytes.slice(0, Math.max(hubCount, 1) * nodeCount * 4));
      ctx.setReadout(
        'matrix',
        `${hubCount} hubs x ${nodeCount} stations${words[Math.max(hubCount, 1) * nodeCount] ? '' : ', NOT converged'}`
      );
      // The mean hub-to-hub time summarises the matrix.
      let sum = 0;
      let count = 0;
      for (let a = 0; a < hubCount; a++) {
        for (let b = 0; b < hubCount; b++) {
          if (a === b) continue;
          const seconds = matrix[a * nodeCount + hubNodes[b]];
          if (Number.isFinite(seconds)) {
            sum += seconds;
            count++;
          }
        }
      }
      ctx.setReadout('hubMean', count ? `${formatMinutes(sum / count)} on average` : null);
      updateHubChart();
    }
  );

  // ---- Initial state ----------------------------------------------------------------------------
  originNode = originFromOptions(ctx.options.origin);
  buildReachGraph(Number(ctx.options.localIterations));
  writeWeights();
  writeBudget();
  writePalette();
  writeOrigin();
  ctx.setReadout('stations', `${formatCount(nodeCount)} stations, ${formatCount(edgeCount)} edges`);
  ctx.setReadout(
    'raster',
    `${RASTER_WIDTH} x ${rasterHeight}, ${pixelSize.toFixed(0)} m per pixel`
  );

  // ---- Interaction ------------------------------------------------------------------------------
  function pickStation(pixel: readonly [number, number], radius = 16): number {
    const viewport = ctx.getViewport();
    if (!viewport) return -1;
    let best = -1;
    let bestDistance = radius * radius;
    for (let node = 0; node < nodeCount; node++) {
      const [x, y] = viewport.project([rail.nodeLngLat[node * 2], rail.nodeLngLat[node * 2 + 1]]);
      const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
      if (squared < bestDistance) {
        bestDistance = squared;
        best = node;
      }
    }
    return best;
  }

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [...(reach ? [reach.compiled] : []), matrixCompiled],

    setOption(id, _value, state) {
      switch (id) {
        case 'origin':
          originNode = originFromOptions(state.origin);
          writeOrigin();
          ctx.requestLayers();
          break;
        case 'trainTypes':
        case 'dwellSeconds':
          writeWeights();
          ctx.requestLayers();
          break;
        case 'costLimitMinutes':
        case 'bandMinutes':
        case 'bandCount':
        case 'lastMile':
        case 'lastMileMinutes':
          writeBudget();
          ctx.requestLayers();
          break;
        case 'ramp':
          writePalette();
          ctx.requestLayers();
          break;
        case 'localIterations':
          if (builtOptions.local !== state.localIterations) {
            builtOptions = {local: state.localIterations};
            buildReachGraph(Number(state.localIterations));
            reachDirty = Math.max(reachDirty, 1);
          }
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (matrixDirty > 0) {
        matrixCompiled.encode(commandEncoder, {parameters: undefined});
        matrixDirty--;
        matrixReader.markStale();
      }
      matrixReader.flush(commandEncoder);
      if (reachDirty > 0 && reach) {
        reach.compiled.encode(commandEncoder, {parameters: undefined});
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: bandVertexCount,
          destinationBuffer: drawCommands.buffer,
          destinationOffset: 0,
          size: 4
        });
        reachDirty--;
        reachReader.markStale();
      }
      reachReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const horizon = Number(options.bandMinutes) * options.bandCount * 60;
      if (options.showBands) {
        layers.push(
          new IsobandTriangleLayer({
            id: 'reach-bands',
            coordinateOrigin,
            gridSize: [1, 1],
            bounds: [0, 0, 1, 1],
            triangles,
            triangleBands,
            values: paletteBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            extent: isochroneParameters.buffer,
            drawCommands,
            drawCommandIndex: 0
          })
        );
      }
      if (options.showEdges) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'reach-edges',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: edgeCount,
            weights: edgeAlphaBuffer,
            widthPixels: 1,
            color: dark ? [215, 222, 240, 170] : [40, 48, 70, 170]
          })
        );
      }
      if (options.showTree) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'reach-tree',
            coordinateOrigin,
            segments: treeSegmentsBuffer,
            instanceCount: nodeCount,
            values: costsBuffer,
            valueFormat: 'float32',
            valueIndices: identityBuffer,
            colormap: 'inferno',
            valueRange: [0, horizon],
            widthPixels: 2.6
          })
        );
      }
      if (options.showStations) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'reach-stations',
            coordinateOrigin,
            positions: nodePositionsBuffer,
            instanceCount: nodeCount,
            radiusPixels: 2.6,
            color: dark ? [245, 247, 252, 235] : [25, 30, 45, 235]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'reach-origin',
          coordinateOrigin,
          positions: originPositionBuffer,
          instanceCount: 1,
          radiusPixels: 8,
          color: [255, 90, 40, 255]
        })
      );
      return layers;
    },

    getTooltip(event) {
      const node = pickStation(event.pixel);
      if (node < 0) return null;
      const cost = costs[node];
      return `${rail.names[node]}: ${formatMinutes(cost)} from ${rail.names[originNode]} (${rail.departuresPerHour[node].toFixed(0)} trains per hour leave it at the morning peak)`;
    },

    onClick(event) {
      let node = pickStation(event.pixel, 40);
      if (node < 0 && event.coordinate) {
        const [x, y] = rail.project(event.coordinate[0], event.coordinate[1]);
        const nearest = findNearestStation(rail, x, y);
        if (nearest.distance < 15_000) node = nearest.node;
      }
      if (node < 0) return false;
      customNode = node;
      originNode = node;
      ctx.setOptions({origin: CUSTOM_ORIGIN});
      writeOrigin();
      ctx.requestLayers();
      return true;
    },

    destroy() {
      destroyed = true;
      reachReader.stop();
      matrixReader.stop();
      resources.destroy();
    }
  };
}
