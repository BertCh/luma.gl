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
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {geodesicCircle} from '../../cartography/reference-geometry';
import {fetchGeoJson, type GeoJsonCollection} from '../../data/loaders';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {IsobandTriangleLayer} from './b9-network-layers';
import {findNearestStation, findStation, loadRailGraph, TRANSIT_HUB_NAMES} from './transit-data';

/** Option state of the transit reachability scene. */
export type TransitReachabilityOptions = {
  origin: string;
  trainTypes: 'all' | 'fast' | 'stopping';
  dwellSeconds: number;
  costLimitMinutes: number;
  lastMile: 'none' | 'walk' | 'bike';
  lastMileMinutes: number;
  serviceHour: string;
  localIterations: '16' | '32' | '64';
  showBands: boolean;
  showTree: boolean;
  showEdges: boolean;
  showStations: boolean;
  showDistanceReference: boolean;
};

/** Speeds of the last mile in meters per second. */
export const LAST_MILE_SPEEDS = {walk: 1.34, bike: 4.2} as const;

const MAXIMUM_BREAKS = 4;
const MAXIMUM_ROUNDS = 16;
const MATRIX_ROUNDS = 16;
const RASTER_WIDTH = 800;
const MAXIMUM_BUFFER_PIXELS = 16;
const TRIANGLE_CAPACITY = 1_000_000;
const BAND_ALPHA = 170;
const EXTENT_PADDING = 12_000;
/** Value of the `origin` option while the origin is a clicked station. */
const CUSTOM_ORIGIN = 'clicked';
const DISABLED_EDGE_ALPHA = 0.08;

const REACH_BREAK_SECONDS = [30 * 60, 60 * 60, 90 * 60] as const;
const REACH_BAND_COLORS = [
  [8, 48, 107, BAND_ALPHA],
  [43, 140, 190, BAND_ALPHA],
  [173, 216, 191, BAND_ALPHA],
  [0, 0, 0, 0]
] as const;
const REFERENCE_RING_VERTEX_COUNT = 97;

function packColors(colors: readonly (readonly number[])[]): Uint32Array {
  return Uint32Array.from(
    colors.map(color => (color[0] | (color[1] << 8) | (color[2] << 16) | (color[3] << 24)) >>> 0)
  );
}

/** Builds a regional water cover with Natural Earth land rings punched out, plus detailed lakes. */
function makeWaterMask(land: GeoJsonCollection, lakes: GeoJsonCollection): GeoJsonCollection {
  const holes: number[][][] = [];
  for (const feature of land.features) {
    const geometry = feature.geometry;
    if (!geometry) continue;
    const polygons =
      geometry.type === 'Polygon'
        ? [geometry.coordinates as number[][][]]
        : geometry.type === 'MultiPolygon'
          ? (geometry.coordinates as number[][][][])
          : [];
    for (const polygon of polygons) {
      if (polygon[0]?.length >= 3) holes.push(polygon[0]);
    }
  }
  return {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        properties: {name: 'Natural Earth regional water mask'},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [2.2, 50.4],
              [7.6, 50.4],
              [7.6, 54.5],
              [2.2, 54.5],
              [2.2, 50.4]
            ],
            ...holes
          ]
        }
      },
      ...lakes.features
    ]
  };
}

/** Turns Natural Earth line features into scene-local planar segments for national context. */
function getBoundarySegments(
  boundaries: GeoJsonCollection,
  project: (longitude: number, latitude: number) => readonly [number, number]
): Float32Array {
  const values: number[] = [];
  const appendLine = (line: unknown) => {
    if (!Array.isArray(line)) return;
    for (let index = 1; index < line.length; index++) {
      const previous = line[index - 1] as number[];
      const next = line[index] as number[];
      if (!Array.isArray(previous) || !Array.isArray(next)) continue;
      const [x0, y0] = project(previous[0], previous[1]);
      const [x1, y1] = project(next[0], next[1]);
      values.push(x0, y0, x1, y1);
    }
  };
  for (const feature of boundaries.features) {
    const geometry = feature.geometry;
    if (!geometry) continue;
    if (geometry.type === 'LineString') appendLine(geometry.coordinates);
    if (geometry.type === 'MultiLineString') {
      for (const line of geometry.coordinates as unknown[]) appendLine(line);
    }
  }
  return Float32Array.from(values);
}

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
  const naturalEarth = ctx.datasets.get('natural-earth');
  const {device} = ctx;
  const {nodeCount, edgeCount} = rail;
  const resources = new SpatialAnalysisResources(device, 'reach');
  const coordinateOrigin: [number, number, number] = [rail.origin[0], rail.origin[1], 0];

  // Draw this cover after the GPU isobands: planar last-mile buffers are invalid over water.
  // Detailed Natural Earth land holes retain the real coast; lakes include IJsselmeer.
  const land = await fetchGeoJson(naturalEarth.fileUrl('ne_10m_land_nl.geojson'), ctx.signal);
  const lakes = await fetchGeoJson(naturalEarth.fileUrl('ne_10m_lakes_nl.geojson'), ctx.signal);
  const boundaries = await fetchGeoJson(
    naturalEarth.fileUrl('ne_10m_boundary_lines_nl.geojson'),
    ctx.signal
  );
  const waterMask = makeWaterMask(land, lakes);
  const waterMesh = buildPolygonMesh(waterMask, (longitude, latitude) =>
    rail.project(longitude, latitude)
  );
  const waterPolygons = createPolygonMeshBuffers(resources, waterMesh, 'reach-water-mask');
  const boundarySegments = getBoundarySegments(boundaries, (longitude, latitude) =>
    rail.project(longitude, latitude)
  );

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
  const hubPositionsBuffer = resources.createBuffer(
    'hub-positions',
    Float32Array.from(
      hubNodes.flatMap(node => [...rail.nodePositions.subarray(node * 2, node * 2 + 2)])
    )
  );
  const boundarySegmentsBuffer = resources.createBuffer('boundary-segments', boundarySegments);
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
  const treeSegmentsBuffer = resources.createBuffer('tree-segments', nodeCount * 16);
  const originPositionBuffer = resources.createBuffer('origin-position', 8);
  const referenceRingBuffer = resources.createBuffer(
    'reference-ring',
    (REFERENCE_RING_VERTEX_COUNT - 1) * 4 * 4
  );

  // ---- Parameter buffers ------------------------------------------------------------------------
  const originParameter = resources.createParameterBuffer('origin', 'uint32', 1);
  const costLimitParameter = resources.createParameterBuffer('cost-limit', 'float32', 1);
  const breaksParameter = resources.createParameterBuffer('breaks', 'float32', MAXIMUM_BREAKS);
  const isochroneParameters = resources.createParameterBuffer(
    'isochrone-parameters',
    'float32',
    GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH
  );
  const paletteBuffer = resources.createBuffer('palette', packColors(REACH_BAND_COLORS));

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
  let builtOptions = {local: ctx.options.localIterations};

  function publishFurniture(): void {
    ctx.setFurniture({
      title: {
        subtitle: `Rail reach from ${rail.names[originNode]} · ${ctx.options.costLimitMinutes} min horizon · scheduled median`,
        chips: ['Best-case connections']
      },
      scaleBar: {units: 'metric', latitude: 52}
    });
  }

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
    const ring = geodesicCircle(
      [rail.nodeLngLat[originNode * 2], rail.nodeLngLat[originNode * 2 + 1]],
      60_000,
      REFERENCE_RING_VERTEX_COUNT - 1
    )[0];
    const segments = new Float32Array((REFERENCE_RING_VERTEX_COUNT - 1) * 4);
    for (let point = 0; point < REFERENCE_RING_VERTEX_COUNT - 1; point++) {
      const [x0, y0] = rail.project(ring[point][0], ring[point][1]);
      const [x1, y1] = rail.project(ring[point + 1][0], ring[point + 1][1]);
      segments.set([x0, y0, x1, y1], point * 4);
    }
    referenceRingBuffer.write(segments);
    ctx.setReadout('origin', rail.names[originNode]);
    publishFurniture();
    reachDirty = Math.max(reachDirty, 1);
    updateHubChart();
  }

  function getBreaks(): number[] {
    return [...REACH_BREAK_SECONDS, REACH_BREAK_SECONDS[REACH_BREAK_SECONDS.length - 1]];
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
        breakCount: REACH_BREAK_SECONDS.length,
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

  const formatMinutes = (seconds: number) =>
    Number.isFinite(seconds) ? `${Math.round(seconds / 60)} min` : 'not reached';

  function updateHubChart(): void {
    if (!hubCount) {
      ctx.setChart('hubTimes', null);
      return;
    }
    const values = hubLabels.map((_, hub) => {
      const seconds = matrix[hub * nodeCount + originNode];
      return Number.isFinite(seconds) ? seconds / 60 : Number.NaN;
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
      description: `Scheduled rail travel time between ${rail.names[originNode]} and each hub station, from the cost matrix. The origin is zero; an em dash is unreachable.`
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
    ctx.setChart('reachCurve', {
      kind: 'line',
      height: 130,
      xLabel: 'rail travel time (min)',
      yLabel: 'stations reached',
      xDomain: [0, horizon],
      formatX: value => value.toFixed(0),
      formatY: value => formatCount(value),
      markers: REACH_BREAK_SECONDS.map(seconds => ({x: seconds / 60})).filter(
        marker => marker.x <= horizon
      ),
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
      const withinBreak = new Array<number>(REACH_BREAK_SECONDS.length).fill(0);
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
      const values = new Float64Array(hubCount * hubCount);
      for (let row = 0; row < hubCount; row++) {
        for (let column = 0; column < hubCount; column++) {
          const seconds = matrix[row * nodeCount + hubNodes[column]];
          values[row * hubCount + column] = Number.isFinite(seconds) ? seconds / 60 : Number.NaN;
        }
      }
      ctx.setChart('hubMatrix', {
        kind: 'matrix',
        height: 260,
        values,
        rows: hubCount,
        columns: hubCount,
        rowLabels: hubLabels,
        columnLabels: hubLabels,
        ramp: 'YlGnBu',
        reverse: true,
        highlight: {column: hubNodes.indexOf(originNode)},
        formatCell: value => (Number.isFinite(value) ? `${Math.round(value)}` : '—'),
        description:
          'Hub-by-hub scheduled cost matrix in minutes. Blank cells are unreachable and are never encoded as the origin’s zero.'
      });
    }
  );

  // ---- Initial state ----------------------------------------------------------------------------
  originNode = originFromOptions(ctx.options.origin);
  buildReachGraph(Number(ctx.options.localIterations));
  writeWeights();
  writeBudget();
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
        case 'lastMile':
        case 'lastMileMinutes':
          writeBudget();
          publishFurniture();
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
      const dark = ctx.ground() === 'dark';
      const layers: Layer[] = [];
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
      if (options.showBands && waterPolygons.vertexCount > 0) {
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: 'reach-water-mask',
            coordinateOrigin,
            triangles: waterPolygons.triangles,
            features: waterPolygons.triangleFeatures,
            vertexCount: waterPolygons.vertexCount,
            colormap: 'uniform',
            color: dark ? [10, 14, 19, 245] : [214, 224, 230, 248]
          })
        );
      }
      if (boundarySegments.length > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'reach-natural-earth-boundaries',
            coordinateOrigin,
            segments: boundarySegmentsBuffer,
            instanceCount: boundarySegments.length / 4,
            widthPixels: 0.65,
            dashArray: [3, 3],
            color: dark ? [220, 228, 242, 105] : [55, 67, 88, 100]
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
            widthPixels: 0.75,
            color: dark ? [215, 222, 240, 92] : [40, 48, 70, 76]
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
            widthPixels: 1.5,
            color: dark ? [234, 239, 246, 190] : [30, 37, 52, 185]
          })
        );
      }
      if (options.showDistanceReference) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'reach-geodesic-60km',
            coordinateOrigin,
            segments: referenceRingBuffer,
            instanceCount: REFERENCE_RING_VERTEX_COUNT - 1,
            widthPixels: 1.4,
            dashArray: [6, 4],
            color: dark ? [250, 244, 220, 225] : [36, 45, 62, 220]
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
            radiusPixels: 2,
            color: dark ? [245, 247, 252, 235] : [25, 30, 45, 235]
          })
        );
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'reach-hubs',
            coordinateOrigin,
            positions: hubPositionsBuffer,
            instanceCount: hubCount,
            radiusPixels: 4.5,
            shape: 'ring',
            outlineWidthPixels: 1.25,
            color: dark ? [250, 245, 224, 245] : [32, 40, 56, 240]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'reach-origin-ring',
          coordinateOrigin,
          positions: originPositionBuffer,
          instanceCount: 1,
          radiusPixels: 8,
          shape: 'ring',
          outlineWidthPixels: 1.75,
          fillOpacity: 0,
          color: [255, 90, 40, 255]
        }),
        new SpatialAnalysisPointLayer({
          id: 'reach-origin',
          coordinateOrigin,
          positions: originPositionBuffer,
          instanceCount: 1,
          radiusPixels: 3.25,
          color: [255, 90, 40, 255],
          outlineColor: dark ? [10, 14, 19, 255] : [255, 255, 255, 255],
          outlineWidthPixels: 1
        })
      );
      return layers;
    },

    getTooltip(event) {
      const node = pickStation(event.pixel);
      if (node < 0) return null;
      const cost = costs[node];
      const hour = Number(ctx.options.serviceHour);
      let departures = 0;
      for (let edge = rail.offsets[node]; edge < rail.offsets[node + 1]; edge++) {
        departures += rail.tripsPerHour[edge * 24 + hour];
      }
      return `${rail.names[node]}: ${formatMinutes(cost)} from ${rail.names[originNode]} · ${departures} outgoing trains/h at ${String(hour).padStart(2, '0')}:00 · modelled cost: scheduled in-vehicle time + ${ctx.options.dwellSeconds} s dwell per edge`;
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
