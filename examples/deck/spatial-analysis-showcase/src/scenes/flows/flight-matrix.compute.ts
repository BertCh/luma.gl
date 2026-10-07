// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  computeAdjacencyMatrixOrder,
  decodeGPUNetworkCoarseningSummary,
  encodeGPUAdjacencyMatrixWindow,
  GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE,
  GPU_NETWORK_COARSENING_SUMMARY_LENGTH,
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder,
  GPUNetworkCoarsening
} from '@luma.gl/experimental/gpu-network';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {LocalMetricProjection} from '../../engine/projection';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  expandAntimeridianEdges,
  readWorldNetwork,
  type ExpandedEdges,
  type FlightNetwork
} from './b11-flight-data';
import {binLog2} from '../movement/f-chart-helpers';
import {FlowArcLayer, SizedDiscLayer} from './b11-flow-layers';
import {CONTINENT_COLORS, CONTINENT_NAMES, loadAirportTable} from './b11-geography';

/** Option state of the flight-matrix scene. */
export type FlightMatrixOptions = {
  view: 'matrix' | 'map' | 'both';
  order: 'input' | 'degree' | 'continent' | 'continent-degree' | 'country';
  statistic: 'count' | 'weight';
  weightBy: 'routes' | 'airlines' | 'distance';
  resolution: '256' | '512' | '1024' | '2048';
  focus: 'all' | 'Europe' | 'Asia' | 'North America' | 'South America' | 'Africa' | 'Oceania';
  zoom: number;
  panX: number;
  panY: number;
  coarsenBy: 'continent' | 'country';
  showGroups: boolean;
  showEdges: boolean;
  showAirports: boolean;
  showBlocks: boolean;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  arcWidth: number;
};

/** Side of the matrix card in meters, drawn around longitude 0, latitude 0. */
export const MATRIX_SIDE = 10_000_000;
const GROUP_CAPACITY = 256;
const SUPEREDGE_CAPACITY = 4096;
const SETTLE_MILLISECONDS = 300;
const RETIRE_FRAMES = 4;
const MAXIMUM_COUNT_READBACK_RESOLUTION = 1024;
const MATRIX_ORIGIN: [number, number, number] = [0, 0, 0];

type Groups = {
  continent: Uint32Array;
  country: Uint32Array;
  countryNames: string[];
  countryContinent: Uint32Array;
  degreeKey: Uint32Array;
};

type MatrixGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  resolution: number;
  reader: SummaryReader;
  layout: Record<string, {offset: number; length: number}>;
  counts: Buffer;
  weightSums: Buffer;
  edgeIds: Buffer;
  edgeTargets: Buffer;
  edgeCounts: Buffer;
  edgeCount: Buffer;
  groupCentroid: Buffer;
  groupVertexCount: Buffer;
};

/**
 * Adjacency matrix and coarsening of the world airline network. One compiled graph holds three
 * contributors: `GPUAdjacencyMatrixOrder` sorts airports by a group label (and a tie key such as
 * degree) into a position permutation; `GPUAdjacencyMatrix` bins the airport pairs into a
 * resolution x resolution image through that permutation and a zoom window; and
 * `GPUNetworkCoarsening` collapses the network by continent or country into supernodes and
 * superedges. Ordering, window, weights and labels are buffer writes; only the matrix resolution
 * rebuilds.
 */
export async function createFlightMatrix(
  ctx: SceneContext<FlightMatrixOptions>
): Promise<SceneInstance<FlightMatrixOptions>> {
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'flight-matrix');
  const airports = await loadAirportTable('openflights', ctx.signal);
  const network = readWorldNetwork(ctx.datasets.get('openflights'), airports);
  const expanded: ExpandedEdges = expandAntimeridianEdges(network);
  const nodeCount = network.nodeCount;
  const groups = buildGroups(network);
  const matrixProjection = new LocalMetricProjection([0, 0]);

  // Undirected CSR: every edge appears in both directions.
  const slotCount = network.edgeCount * 2;
  const offsets = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < network.edgeCount; edge++) {
    offsets[network.source[edge] + 1]++;
    offsets[network.target[edge] + 1]++;
  }
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const neighbors = new Uint32Array(slotCount);
  const slotEdge = new Uint32Array(slotCount);
  const cursor = offsets.slice(0, nodeCount);
  for (let edge = 0; edge < network.edgeCount; edge++) {
    for (const [from, to] of [
      [network.source[edge], network.target[edge]],
      [network.target[edge], network.source[edge]]
    ]) {
      neighbors[cursor[from]] = to;
      slotEdge[cursor[from]] = edge;
      cursor[from]++;
    }
  }

  const offsetsBuffer = resources.createBuffer('offsets', offsets);
  const neighborsBuffer = resources.createBuffer('neighbors', neighbors);
  const weightsBuffer = resources.createBuffer('weights', slotCount * 4);
  const groupsBuffer = resources.createBuffer('order-groups', nodeCount * 4);
  const tieBuffer = resources.createBuffer('order-ties', nodeCount * 4);
  const orderBuffer = resources.createBuffer('order', nodeCount * 4);
  const labelsBuffer = resources.createBuffer('labels', nodeCount * 4);
  const positionsBuffer = resources.createBuffer('positions', network.lonLat);
  const degreeValues = resources.createBuffer('degree-values', Float32Array.from(network.degree));
  const groupColorBuffer = resources.createBuffer('group-colors', GROUP_CAPACITY * 4);
  const windowBuffer = resources.createParameterBuffer('window', 'uint32', 4);
  const edgeSegments = resources.createBuffer(
    'edge-segments',
    (() => {
      const segments = new Float32Array(expanded.source.length * 4);
      for (let edge = 0; edge < expanded.source.length; edge++) {
        segments.set(
          expanded.positions.subarray(expanded.source[edge] * 2, expanded.source[edge] * 2 + 2),
          edge * 4
        );
        segments.set(
          expanded.positions.subarray(expanded.target[edge] * 2, expanded.target[edge] * 2 + 2),
          edge * 4 + 2
        );
      }
      return segments;
    })()
  );
  const airportColors = resources.createBuffer('airport-colors', groups.continent);

  // Dynamic overlay segments of the matrix card (block boundaries and continent strips).
  const lineCapacity = (GROUP_CAPACITY + 8) * 2;
  const countryLines = resources.createBuffer('country-lines', lineCapacity * 16);
  const continentLines = resources.createBuffer('continent-lines', 64 * 16);
  const stripSegments = resources.createBuffer('strip-segments', 16 * 16);
  const stripColors = resources.createBuffer('strip-colors', 16 * 4);
  const cardBackdrop = resources.createBuffer('card-backdrop', 4);
  const counts = {countryLines: 0, continentLines: 0, strips: 0};

  let graph: MatrixGraph | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let statsStale = true;
  let lastChange = performance.now();
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];
  /** Position of every airport after the CPU twin of the ordering. */
  let cpuOrder: Uint32Array = new Uint32Array(nodeCount);
  let inverseOrder = new Uint32Array(nodeCount);
  let continentStarts: number[] = [];
  let countryStarts: number[] = [];
  let window = {rowStart: 0, rowEnd: nodeCount, colStart: 0, colEnd: nodeCount};
  // Latest readback.
  let lastMaxCount = 1;
  let lastMaxWeight = 1;
  let lastCounts: Uint32Array | null = null;
  let lastGroupVertexCount: Uint32Array = new Uint32Array(GROUP_CAPACITY);
  let lastGroupIntra: Uint32Array = new Uint32Array(GROUP_CAPACITY);
  let lastCentroid: Float32Array = new Float32Array(GROUP_CAPACITY * 2);
  let lastSuperTargets: Uint32Array = new Uint32Array(0);
  let lastSuperSources: Uint32Array = new Uint32Array(0);
  let lastSuperCounts: Uint32Array = new Uint32Array(0);
  let maximumGroupVertices = 1;
  let maximumSuperedge = 1;
  let coarsenGroupNames: string[] = [];

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 2);
    statsStale = true;
    lastChange = performance.now();
  }

  function writeOrderInputs(): void {
    const {order} = ctx.options;
    const zeros = new Uint32Array(nodeCount);
    const groupKeys =
      order === 'continent' || order === 'continent-degree'
        ? groups.continent
        : order === 'country'
          ? groups.country
          : zeros;
    const tieKeys =
      order === 'degree' || order === 'continent-degree' || order === 'country'
        ? groups.degreeKey
        : zeros;
    groupsBuffer.write(groupKeys);
    tieBuffer.write(tieKeys);
    // The CPU twin gives the same permutation: used for block boundaries, hover and verification.
    cpuOrder = computeAdjacencyMatrixOrder(groupKeys, order === 'input' ? undefined : tieKeys);
    inverseOrder = new Uint32Array(nodeCount);
    for (let vertex = 0; vertex < nodeCount; vertex++) inverseOrder[cpuOrder[vertex]] = vertex;
    // Continent-major orderings keep continents contiguous; boundaries are cumulative sizes.
    const hasBlocks = order === 'continent' || order === 'continent-degree' || order === 'country';
    continentStarts = [];
    countryStarts = [];
    if (hasBlocks) {
      const perContinent = new Array(CONTINENT_NAMES.length).fill(0);
      for (const continent of groups.continent) perContinent[continent]++;
      let running = 0;
      for (const size of perContinent) {
        continentStarts.push(running);
        running += size;
      }
      continentStarts.push(running);
      if (order === 'country') {
        const perCountry = new Array(groups.countryNames.length).fill(0);
        for (const country of groups.country) perCountry[country]++;
        running = 0;
        for (const size of perCountry) {
          countryStarts.push(running);
          running += size;
        }
        countryStarts.push(running);
      }
    }
    writeWindow();
    markChanged();
  }

  function writeWeights(): void {
    const {weightBy} = ctx.options;
    const weights = new Float32Array(slotCount);
    for (let slot = 0; slot < slotCount; slot++) {
      const edge = slotEdge[slot];
      weights[slot] =
        weightBy === 'routes'
          ? network.traffic[edge]
          : weightBy === 'airlines'
            ? network.airlines![edge]
            : network.distanceKm[edge] / 100;
    }
    weightsBuffer.write(weights);
    markChanged();
  }

  function writeLabels(): void {
    const {coarsenBy} = ctx.options;
    labelsBuffer.write(coarsenBy === 'continent' ? groups.continent : groups.country);
    const colors = new Uint32Array(GROUP_CAPACITY);
    if (coarsenBy === 'continent') {
      for (let group = 0; group < CONTINENT_NAMES.length; group++) colors[group] = group;
    } else {
      groups.countryContinent.forEach((continent, group) => {
        colors[group] = continent;
      });
    }
    groupColorBuffer.write(colors);
    coarsenGroupNames = coarsenBy === 'continent' ? [...CONTINENT_NAMES] : groups.countryNames;
    markChanged();
  }

  function writeWindow(): void {
    const {focus, zoom, panX, panY} = ctx.options;
    const focusIndex = CONTINENT_NAMES.indexOf(focus as (typeof CONTINENT_NAMES)[number]);
    if (focus !== 'all' && focusIndex >= 0 && continentStarts.length > 0) {
      window = {
        rowStart: continentStarts[focusIndex],
        rowEnd: continentStarts[focusIndex + 1],
        colStart: continentStarts[focusIndex],
        colEnd: continentStarts[focusIndex + 1]
      };
    } else {
      const size = Math.max(8, Math.round(nodeCount / zoom));
      const clampStart = (center: number) =>
        Math.min(nodeCount - size, Math.max(0, Math.round(center * nodeCount - size / 2)));
      window = {
        rowStart: clampStart(panY),
        rowEnd: clampStart(panY) + size,
        colStart: clampStart(panX),
        colEnd: clampStart(panX) + size
      };
    }
    const resolution = graph?.resolution ?? Number(ctx.options.resolution);
    windowBuffer.write(encodeGPUAdjacencyMatrixWindow(window, resolution));
    ctx.setReadout(
      'window',
      `rows ${formatCount(window.rowStart)} to ${formatCount(window.rowEnd)}, columns ${formatCount(window.colStart)} to ${formatCount(window.colEnd)}`
    );
    writeOverlays();
    markChanged();
  }

  /** Writes block boundaries and continent strips in card coordinates. */
  function writeOverlays(): void {
    const half = MATRIX_SIDE / 2;
    const columnExtent = Math.max(1, window.colEnd - window.colStart);
    const rowExtent = Math.max(1, window.rowEnd - window.rowStart);
    const toX = (position: number) =>
      -half + ((position - window.colStart) / columnExtent) * MATRIX_SIDE;
    const toY = (position: number) =>
      half - ((position - window.rowStart) / rowExtent) * MATRIX_SIDE;
    const frame = [
      [-half, -half, half, -half],
      [half, -half, half, half],
      [half, half, -half, half],
      [-half, half, -half, -half]
    ];
    const boundary = (starts: number[], list: number[][]) => {
      for (const position of starts.slice(1, -1)) {
        if (position > window.colStart && position < window.colEnd) {
          list.push([toX(position), -half, toX(position), half]);
        }
        if (position > window.rowStart && position < window.rowEnd) {
          list.push([-half, toY(position), half, toY(position)]);
        }
      }
    };
    const continentList: number[][] = [...frame];
    boundary(continentStarts, continentList);
    const countryList: number[][] = [];
    boundary(countryStarts, countryList);
    const flat = (list: number[][]) => Float32Array.from(list.flat());
    continentLines.write(flat(continentList.slice(0, 64)));
    counts.continentLines = Math.min(64, continentList.length);
    countryLines.write(flat(countryList.slice(0, lineCapacity)));
    counts.countryLines = Math.min(lineCapacity, countryList.length);
    // Continent strips above and left of the card, clipped to the window.
    const strips: number[][] = [];
    const colors: number[] = [];
    const gap = MATRIX_SIDE * 0.018;
    for (let index = 0; index + 1 < continentStarts.length; index++) {
      const from = continentStarts[index];
      const to = continentStarts[index + 1];
      if (to <= from) continue;
      const colFrom = Math.max(from, window.colStart);
      const colTo = Math.min(to, window.colEnd);
      if (colTo > colFrom) {
        strips.push([toX(colFrom), half + gap, toX(colTo), half + gap]);
        colors.push(index);
      }
      const rowFrom = Math.max(from, window.rowStart);
      const rowTo = Math.min(to, window.rowEnd);
      if (rowTo > rowFrom) {
        strips.push([-half - gap, toY(rowFrom), -half - gap, toY(rowTo)]);
        colors.push(index);
      }
    }
    stripSegments.write(flat(strips.slice(0, 16)));
    stripColors.write(Uint32Array.from(colors.slice(0, 16)));
    counts.strips = Math.min(16, strips.length);
  }

  function buildGraph(): MatrixGraph {
    const resolution = Number(ctx.options.resolution);
    const id = ++serial;
    const graphResources = new SpatialAnalysisResources(device, `flight-matrix-${id}`);
    const cellCount = resolution * resolution;
    const counts = graphResources.createBuffer('counts', cellCount * 4);
    const weightSums = graphResources.createBuffer('weight-sums', cellCount * 4);
    const maxCount = graphResources.createBuffer('max-count', 4);
    const maxWeightSum = graphResources.createBuffer('max-weight-sum', 4);
    const groupVertexCount = graphResources.createBuffer('group-vertex-count', GROUP_CAPACITY * 4);
    const groupIntra = graphResources.createBuffer('group-intra', GROUP_CAPACITY * 4);
    const groupIntraWeight = graphResources.createBuffer('group-intra-weight', GROUP_CAPACITY * 4);
    const groupCentroid = graphResources.createBuffer('group-centroid', GROUP_CAPACITY * 8);
    const groupValueSum = graphResources.createBuffer('group-value-sum', GROUP_CAPACITY * 4);
    const edgeIds = graphResources.createBuffer('super-ids', SUPEREDGE_CAPACITY * 4);
    const edgeCount = graphResources.createBuffer('super-count', 4);
    const edgeOverflow = graphResources.createBuffer('super-overflow', 4);
    const edgeTargets = graphResources.createBuffer('super-targets', SUPEREDGE_CAPACITY * 4);
    const edgeCounts = graphResources.createBuffer('super-counts', SUPEREDGE_CAPACITY * 4);
    const edgeWeights = graphResources.createBuffer('super-weights', SUPEREDGE_CAPACITY * 4);
    const summary = graphResources.createBuffer(
      'summary',
      GPU_NETWORK_COARSENING_SUMMARY_LENGTH * 4
    );

    const commandGraph = new GPUCommandGraph<void>(device, {id: `flight-matrix-${id}`});
    const u32 = (name: string, buffer: Buffer, length?: number) =>
      importGraphBuffer(commandGraph, name, buffer, 'uint32', length);
    const f32 = (name: string, buffer: Buffer, length?: number) =>
      importGraphBuffer(commandGraph, name, buffer, 'float32', length);
    const orderView = u32('order', orderBuffer, nodeCount);
    const offsetView = u32('offsets', offsetsBuffer, nodeCount + 1);
    const neighborView = u32('neighbors', neighborsBuffer, slotCount);
    const weightView = f32('weights', weightsBuffer, slotCount);
    commandGraph.add(
      new GPUAdjacencyMatrixOrder({
        id: 'order',
        groups: u32('groups', groupsBuffer, nodeCount),
        tieKeys: u32('ties', tieBuffer, nodeCount),
        order: orderView
      })
    );
    commandGraph.add(
      new GPUAdjacencyMatrix({
        id: 'matrix',
        offsets: offsetView,
        neighbors: neighborView,
        weights: weightView,
        order: orderView,
        window: windowBuffer.importToGraph(commandGraph),
        resolution,
        weightScale: GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE,
        output: {
          counts: u32('counts', counts, cellCount),
          weightSums: u32('weight-sums', weightSums, cellCount),
          maxCount: u32('max-count', maxCount, 1),
          maxWeightSum: u32('max-weight-sum', maxWeightSum, 1)
        }
      })
    );
    commandGraph.add(
      new GPUNetworkCoarsening({
        id: 'coarsen',
        offsets: offsetView,
        neighbors: neighborView,
        weights: weightView,
        labels: u32('labels', labelsBuffer, nodeCount),
        positions: importGraphBuffer(
          commandGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          nodeCount
        ),
        vertexValues: f32('degree', degreeValues, nodeCount),
        groupCapacity: GROUP_CAPACITY,
        groupVertexCount: u32('group-vertex-count', groupVertexCount, GROUP_CAPACITY),
        groupIntraEdgeCount: u32('group-intra', groupIntra, GROUP_CAPACITY),
        groupIntraWeight: f32('group-intra-weight', groupIntraWeight, GROUP_CAPACITY),
        groupCentroid: importGraphBuffer(
          commandGraph,
          'group-centroid',
          groupCentroid,
          'float32x2',
          GROUP_CAPACITY
        ),
        groupValueSum: f32('group-value-sum', groupValueSum, GROUP_CAPACITY),
        edges: {
          ids: u32('super-ids', edgeIds, SUPEREDGE_CAPACITY),
          count: u32('super-count', edgeCount, 1),
          overflow: u32('super-overflow', edgeOverflow, 1)
        },
        edgeTargets: u32('super-targets', edgeTargets, SUPEREDGE_CAPACITY),
        edgeCounts: u32('super-counts', edgeCounts, SUPEREDGE_CAPACITY),
        edgeWeights: f32('super-weights', edgeWeights, SUPEREDGE_CAPACITY),
        summary: u32('summary', summary, GPU_NETWORK_COARSENING_SUMMARY_LENGTH)
      })
    );
    const compiled = graphResources.track(commandGraph.compile());

    const readCounts = resolution <= MAXIMUM_COUNT_READBACK_RESOLUTION;
    const sourceList: {name: string; buffer: Buffer; size: number}[] = [
      {name: 'maxCount', buffer: maxCount, size: 4},
      {name: 'maxWeight', buffer: maxWeightSum, size: 4},
      {name: 'order', buffer: orderBuffer, size: nodeCount * 4},
      {name: 'summary', buffer: summary, size: GPU_NETWORK_COARSENING_SUMMARY_LENGTH * 4},
      {name: 'groupVertexCount', buffer: groupVertexCount, size: GROUP_CAPACITY * 4},
      {name: 'groupIntra', buffer: groupIntra, size: GROUP_CAPACITY * 4},
      {name: 'groupCentroid', buffer: groupCentroid, size: GROUP_CAPACITY * 8},
      {name: 'superIds', buffer: edgeIds, size: SUPEREDGE_CAPACITY * 4},
      {name: 'superTargets', buffer: edgeTargets, size: SUPEREDGE_CAPACITY * 4},
      {name: 'superCounts', buffer: edgeCounts, size: SUPEREDGE_CAPACITY * 4},
      {name: 'superCount', buffer: edgeCount, size: 4},
      {name: 'superOverflow', buffer: edgeOverflow, size: 4}
    ];
    if (readCounts) sourceList.push({name: 'counts', buffer: counts, size: cellCount * 4});
    const layout: Record<string, {offset: number; length: number}> = {};
    let offset = 0;
    for (const source of sourceList) {
      layout[source.name] = {offset, length: source.size / 4};
      offset += source.size;
    }
    const built: MatrixGraph = {
      resources: graphResources,
      compiled,
      resolution,
      reader: undefined as unknown as SummaryReader,
      layout,
      counts,
      weightSums,
      edgeIds,
      edgeTargets,
      edgeCounts,
      edgeCount,
      groupCentroid,
      groupVertexCount
    };
    built.reader = new SummaryReader(
      graphResources,
      `flight-matrix-${id}`,
      sourceList.map(({buffer, size}) => ({buffer, size})),
      bytes => {
        if (!destroyed && graph === built) processReadback(built, bytes);
      }
    );
    return built;
  }

  function rebuild(): void {
    if (graph) retired.push({resources: graph.resources, frames: 0});
    graph = buildGraph();
    ctx.setReadout('resolution', `${graph.resolution} × ${graph.resolution} cells`);
    writeWindow();
    markChanged();
  }

  // Airport degree distribution (static), in power-of-two bins: a heavy-tailed hub structure.
  {
    const {counts, labels} = binLog2(network.degree, 10);
    ctx.setChart('degreeChart', {
      kind: 'bars',
      values: counts,
      labels,
      color: 2,
      height: 110,
      xLabel: 'routes at an airport',
      yLabel: 'airports',
      formatY: value => (value >= 1000 ? `${Math.round(value / 1000)}k` : `${value}`),
      description:
        'Airports by number of routes, in power-of-two bins. Most airports have a few routes; a handful of hubs have hundreds.'
    });
  }

  function processReadback(current: MatrixGraph, bytes: ArrayBuffer): void {
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const get = (name: string): Uint32Array => {
      const {offset, length} = current.layout[name];
      return words.slice(offset / 4, offset / 4 + length);
    };
    const getFloat = (name: string): Float32Array => {
      const {offset, length} = current.layout[name];
      return floats.slice(offset / 4, offset / 4 + length);
    };
    lastMaxCount = Math.max(1, get('maxCount')[0]);
    lastMaxWeight = Math.max(1, get('maxWeight')[0] / GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE);
    const gpuOrder = get('order');
    let matching = 0;
    for (let vertex = 0; vertex < nodeCount; vertex++) {
      if (gpuOrder[vertex] === cpuOrder[vertex]) matching++;
    }
    ctx.setReadout(
      'orderCheck',
      matching === nodeCount
        ? `identical to computeAdjacencyMatrixOrder (${formatCount(nodeCount)} airports)`
        : `MISMATCH: ${formatCount(nodeCount - matching)} airports differ`
    );
    lastCounts = current.layout.counts ? get('counts') : null;
    if (lastCounts) {
      let occupied = 0;
      for (const count of lastCounts) if (count > 0) occupied++;
      ctx.setReadout('filled', occupied / lastCounts.length);
      const {counts: cellBins, labels: cellLabels} = binLog2(lastCounts, 10);
      ctx.setChart('cellChart', {
        kind: 'bars',
        values: cellBins,
        labels: cellLabels,
        height: 110,
        xLabel: 'pairs in a cell',
        yLabel: 'cells',
        formatY: value => (value >= 1000 ? `${Math.round(value / 1000)}k` : `${value}`),
        description:
          'Number of occupied matrix cells by how many airport pairs they hold, in power-of-two bins. Most cells hold one or two routes: the matrix is sparse.'
      });
      ctx.setReadout('occupied', occupied);
    } else {
      ctx.setReadout('filled', null);
      ctx.setReadout('occupied', null);
      ctx.setChart('cellChart', null);
    }
    ctx.setReadout('maxCell', ctx.options.statistic === 'count' ? lastMaxCount : lastMaxWeight);
    ctx.setLegendExtent(
      'matrix',
      ctx.options.statistic === 'count' ? [0, lastMaxCount] : [0, lastMaxWeight]
    );

    const summary = decodeGPUNetworkCoarseningSummary(get('summary'));
    lastGroupVertexCount = get('groupVertexCount');
    lastGroupIntra = get('groupIntra');
    lastCentroid = getFloat('groupCentroid');
    const superCount = Math.min(get('superCount')[0], SUPEREDGE_CAPACITY);
    lastSuperSources = get('superIds').slice(0, superCount);
    lastSuperTargets = get('superTargets').slice(0, superCount);
    lastSuperCounts = get('superCounts').slice(0, superCount);
    maximumGroupVertices = Math.max(1, ...lastGroupVertexCount);
    maximumSuperedge = Math.max(1, ...lastSuperCounts);
    const ranked = Array.from({length: GROUP_CAPACITY}, (_, group) => group)
      .filter(group => lastGroupVertexCount[group] > 0)
      .sort((a, b) => lastGroupVertexCount[b] - lastGroupVertexCount[a])
      .slice(0, 8);
    ctx.setChart('groupChart', {
      kind: 'bars',
      values: ranked.map(group => lastGroupVertexCount[group]),
      labels: ranked.map(group => (coarsenGroupNames[group] ?? `G${group}`).slice(0, 10)),
      highlight: [0],
      height: 120,
      yLabel: 'airports',
      formatY: value => value.toFixed(0),
      description: 'Airports in each of the eight largest groups of the coarsened network.'
    });
    ctx.setReadout('groups', summary.groupCount);
    ctx.setReadout(
      'superedges',
      `${formatCount(summary.superedgeCount)}${get('superOverflow')[0] ? ' (list truncated)' : ''}`
    );
    ctx.setReadout(
      'intraShare',
      summary.countedEdgeCount > 0 ? summary.intraEdgeCount / summary.countedEdgeCount : null
    );
    ctx.setReadout('interEdges', summary.interEdgeCount);
    ctx.setReadout('dropped', summary.droppedEdgeCount + summary.overflowedVertexCount);
    let largest = 0;
    for (let group = 0; group < GROUP_CAPACITY; group++) {
      if (lastGroupVertexCount[group] > lastGroupVertexCount[largest]) largest = group;
    }
    ctx.setReadout(
      'largestGroup',
      `${coarsenGroupNames[largest] ?? `Group ${largest}`} (${formatCount(lastGroupVertexCount[largest])} airports)`
    );
    ctx.requestLayers();
  }

  function getGroupTooltip(pixel: readonly [number, number]): string | null {
    const viewport = ctx.getViewport();
    if (!viewport) return null;
    let best = -1;
    let bestDistance = 26;
    for (let group = 0; group < GROUP_CAPACITY; group++) {
      if (lastGroupVertexCount[group] === 0) continue;
      const [x, y] = viewport.project([lastCentroid[group * 2], lastCentroid[group * 2 + 1]]);
      const distance = Math.hypot(x - pixel[0], y - pixel[1]);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = group;
      }
    }
    if (best < 0) return null;
    let external = 0;
    for (let edge = 0; edge < lastSuperCounts.length; edge++) {
      if (lastSuperSources[edge] === best || lastSuperTargets[edge] === best)
        external += lastSuperCounts[edge];
    }
    const intra = lastGroupIntra[best];
    const share = intra + external > 0 ? (intra / (intra + external)) * 100 : 0;
    return `${coarsenGroupNames[best] ?? `Group ${best}`}\n${formatCount(lastGroupVertexCount[best])} airports\n${formatCount(intra)} routes stay inside (${share.toFixed(0)}%)\n${formatCount(external)} routes leave`;
  }

  function getMatrixTooltip(coordinate: readonly [number, number]): string | null {
    const [x, y] = matrixProjection.project(coordinate[0], coordinate[1]);
    const u = (x + MATRIX_SIDE / 2) / MATRIX_SIDE;
    const v = (MATRIX_SIDE / 2 - y) / MATRIX_SIDE;
    if (u < 0 || u >= 1 || v < 0 || v >= 1 || !graph) return null;
    const column = window.colStart + Math.floor(u * (window.colEnd - window.colStart));
    const row = window.rowStart + Math.floor(v * (window.rowEnd - window.rowStart));
    const rowAirport = network.airports[inverseOrder[Math.min(row, nodeCount - 1)]];
    const columnAirport = network.airports[inverseOrder[Math.min(column, nodeCount - 1)]];
    const resolution = graph.resolution;
    const cell = lastCounts
      ? lastCounts[Math.floor(v * resolution) * resolution + Math.floor(u * resolution)]
      : null;
    const lines = [
      `Row: ${rowAirport.iata} ${rowAirport.city}, ${rowAirport.country}`,
      `Column: ${columnAirport.iata} ${columnAirport.city}, ${columnAirport.country}`
    ];
    if (cell !== null) lines.push(`This cell holds ${formatCount(cell)} airport pairs`);
    return lines.join('\n');
  }

  writeLabels();
  writeWeights();
  writeOrderInputs();
  rebuild();

  return {
    getCompiledGraphs: () => (graph ? [graph.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id) {
      switch (id) {
        case 'resolution':
          rebuild();
          ctx.requestLayers();
          break;
        case 'order':
          writeOrderInputs();
          ctx.requestLayers();
          break;
        case 'weightBy':
          writeWeights();
          break;
        case 'coarsenBy':
          writeLabels();
          ctx.requestLayers();
          break;
        case 'focus':
        case 'zoom':
        case 'panX':
        case 'panY':
          writeWindow();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      const {view} = ctx.options;
      if (view !== 'map' && event.coordinate) {
        const text = getMatrixTooltip(event.coordinate);
        if (text) return text;
      }
      return view !== 'matrix' && ctx.options.showGroups ? getGroupTooltip(event.pixel) : null;
    },

    encode(commandEncoder) {
      if (!graph) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      if (encodeFrames > 0) {
        graph.compiled.encode(commandEncoder, {parameters: undefined});
        encodeFrames--;
      }
      if (
        statsStale &&
        encodeFrames === 0 &&
        performance.now() - lastChange > SETTLE_MILLISECONDS
      ) {
        if (!graph.reader.isPending) {
          statsStale = false;
          graph.reader.request(commandEncoder);
        }
      } else {
        graph.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!graph) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const lngLat = COORDINATE_SYSTEM.LNGLAT;
      const layers: Layer[] = [];
      const showMap = options.view !== 'matrix';
      const showMatrix = options.view !== 'map';
      if (showMap) {
        if (options.showEdges) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'matrix-edges',
              coordinateSystem: lngLat,
              segments: edgeSegments,
              instanceCount: expanded.source.length,
              widthPixels: 0.7,
              color: dark ? [170, 185, 225, 34] : [60, 70, 100, 38]
            })
          );
        }
        if (options.showGroups) {
          layers.push(
            new FlowArcLayer({
              id: `matrix-superedges-${options.coarsenBy}`,
              coordinateSystem: lngLat,
              flowOriginZoneIds: graph.edgeIds,
              flowDestinationZoneIds: graph.edgeTargets,
              flowWeights: graph.edgeCounts,
              weightFormat: 'uint32',
              flowCount: graph.edgeCount,
              instanceCount: SUPEREDGE_CAPACITY,
              zoneKind: 'ids',
              zoneCenters: graph.groupCentroid,
              maximumWeight: maximumSuperedge,
              heaviestLast: false,
              widthMinPixels: 1,
              widthMaxPixels: options.arcWidth,
              bulge: 0.1,
              opacity: 0.8,
              originColor: dark ? [255, 205, 120, 255] : [210, 110, 20, 255],
              destinationColor: dark ? [255, 205, 120, 255] : [210, 110, 20, 255]
            }),
            new SizedDiscLayer({
              id: `matrix-supernodes-${options.coarsenBy}`,
              positions: graph.groupCentroid,
              rowCount: GROUP_CAPACITY,
              values: graph.groupVertexCount,
              valueFormat: 'uint32',
              maximumValue: maximumGroupVertices,
              colorIndices: groupColorBuffer,
              palette: CONTINENT_COLORS,
              minRadiusPixels: 5,
              maxRadiusPixels: options.coarsenBy === 'continent' ? 40 : 30,
              opacity: 0.92
            })
          );
        }
        if (options.showAirports) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'matrix-airports',
              coordinateSystem: lngLat,
              positions: positionsBuffer,
              instanceCount: nodeCount,
              radiusPixels: 1.6,
              values: airportColors,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: CONTINENT_COLORS,
              opacity: options.showGroups ? 0.55 : 0.9
            })
          );
        }
      }
      if (showMatrix) {
        const half = MATRIX_SIDE / 2;
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'matrix-backdrop',
            coordinateOrigin: MATRIX_ORIGIN,
            gridSize: [1, 1],
            bounds: [-half * 1.06, -half * 1.06, half * 1.06, half * 1.06],
            binning: 'grid',
            colormap: 'uniform',
            color: dark ? [12, 16, 26, 238] : [252, 252, 254, 244]
          }),
          new SpatialAnalysisRasterLayer({
            id: 'matrix-cells',
            coordinateOrigin: MATRIX_ORIGIN,
            gridSize: [graph.resolution, graph.resolution],
            bounds: [-half, -half, half, half],
            binning: 'grid',
            rowOrigin: 'north',
            values: options.statistic === 'count' ? graph.counts : graph.weightSums,
            valueFormat: 'uint32',
            valueScale:
              options.statistic === 'count' ? 1 : 1 / GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE,
            valueRange: [0, options.statistic === 'count' ? lastMaxCount : lastMaxWeight],
            colormap: options.ramp,
            sqrtScale: true,
            discardAtOrBelow: 0,
            color: [255, 255, 255, 255]
          })
        );
        if (options.showBlocks) {
          if (counts.countryLines > 0) {
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: 'matrix-country-lines',
                coordinateOrigin: MATRIX_ORIGIN,
                segments: countryLines,
                instanceCount: counts.countryLines,
                widthPixels: 0.6,
                color: dark ? [255, 255, 255, 38] : [20, 30, 60, 40]
              })
            );
          }
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'matrix-continent-lines',
              coordinateOrigin: MATRIX_ORIGIN,
              segments: continentLines,
              instanceCount: counts.continentLines,
              widthPixels: 1.4,
              color: dark ? [255, 255, 255, 110] : [20, 30, 60, 120]
            })
          );
          if (counts.strips > 0) {
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: 'matrix-strips',
                coordinateOrigin: MATRIX_ORIGIN,
                segments: stripSegments,
                instanceCount: counts.strips,
                widthPixels: 7,
                values: stripColors,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: CONTINENT_COLORS
              })
            );
          }
        }
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      graph?.reader.stop();
      graph?.resources.destroy();
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
      void cardBackdrop;
    }
  };
}

/** Group labels of every airport: continent, country (continent-major, big countries first), degree key. */
function buildGroups(network: FlightNetwork): Groups {
  const countryCounts = new Map<string, number>();
  for (const airport of network.airports) {
    countryCounts.set(airport.country, (countryCounts.get(airport.country) ?? 0) + 1);
  }
  const continentOfCountry = new Map<string, number>();
  network.airports.forEach((airport, index) => {
    continentOfCountry.set(airport.country, network.continent[index]);
  });
  const countryList = [...countryCounts.keys()].sort(
    (a, b) =>
      continentOfCountry.get(a)! - continentOfCountry.get(b)! ||
      countryCounts.get(b)! - countryCounts.get(a)! ||
      a.localeCompare(b)
  );
  const countryIds = new Map(countryList.map((country, index) => [country, index]));
  const country = new Uint32Array(network.nodeCount);
  const continent = new Uint32Array(network.nodeCount);
  const degreeKey = new Uint32Array(network.nodeCount);
  const maximumDegree = Math.max(...network.degree);
  network.airports.forEach((airport, index) => {
    country[index] = countryIds.get(airport.country)!;
    continent[index] = network.continent[index];
    degreeKey[index] = maximumDegree - network.degree[index];
  });
  return {
    continent,
    country,
    countryNames: countryList,
    countryContinent: Uint32Array.from(countryList.map(name => continentOfCountry.get(name)!)),
    degreeKey
  };
}
