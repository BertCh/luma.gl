// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  computeAdjacencyMatrixOrder,
  decodeGPUNetworkCoarseningSummary,
  encodeGPUAdjacencyMatrixWindow,
  GPU_NETWORK_COARSENING_SUMMARY_LENGTH,
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder,
  GPUNetworkCoarsening
} from '@luma.gl/experimental/gpu-network';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getClassIndexOf,
  getClassLabel,
  getClassTableLayerProps
} from '../../cartography/class-table';
import {formatCount, formatPercent, liveText} from '../../cartography/live-text';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisFlowLayer} from '../../engine/flow-layer';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {LocalMetricProjection} from '../../engine/projection';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {MapAnnotation, SceneContext, SceneInstance, TooltipContent} from '../scene';
import {expandAntimeridianEdges, readWorldNetwork, type ExpandedEdges} from './b11-flight-data';
import {CONTINENT_NAMES, loadAirportTable} from './b11-geography';
import {
  binLog2,
  buildGroups,
  computeChance,
  getBlockStarts,
  getIntraShare,
  getLargestSuperedge,
  getMatrixWindow,
  getMembersByDegree,
  getOrderBlocks,
  getOrderKeys,
  pickHubs,
  type ChanceModel,
  type MatrixOrder,
  type MatrixWindow
} from './flight-matrix-stats';
import {
  COUNT_LABELS,
  getChanceTable,
  getContinentPalette,
  getCountTable,
  getSheetColors,
  LABEL_GAP_SHARE,
  MATRIX_HALF,
  MATRIX_SIDE,
  SHEET_HALF,
  STRIP_GAP_SHARE
} from './flight-matrix-style';
import {BETWEEN_GROUPS_INK, FLOW_HALO, FLOW_INK, inkFor, withInkAlpha} from './flows-style';

/** Option state of the flight-matrix scene. */
export type FlightMatrixOptions = {
  view: 'map' | 'matrix';
  order: MatrixOrder;
  /** Matrix resolution as a power of two: 8 is 256 bins per side. */
  resolutionPower: number;
  focus: 'all' | 'Europe' | 'Asia' | 'North America' | 'South America' | 'Africa' | 'Oceania';
  zoom: number;
  panX: number;
  panY: number;
  coarsenBy: 'continent' | 'country';
  showRoutes: boolean;
  showAirports: boolean;
  showGroups: boolean;
  showBlocks: boolean;
  showChance: boolean;
  arcWidth: number;
};

const GROUP_CAPACITY = 256;
const SUPEREDGE_CAPACITY = 16_384;
/** At most this many superedges are drawn (the heaviest); the continent grouping has fewer. */
const SUPEREDGE_DRAW_LIMIT = 320;
const SETTLE_MILLISECONDS = 300;
const MAXIMUM_COUNT_READBACK_RESOLUTION = 1024;
const MATRIX_ORIGIN: [number, number, number] = [0, 0, 0];
const ORDER_TRANSITION_MILLISECONDS = 900;
const HUB_LABEL_COUNT = 8;
const NOTE_BLOCK_COUNT = 3;
const AIRPORT_MAX_RADIUS_PIXELS = 6.5;
const ROUTE_WIDTH_PIXELS = 0.7;
const INTRA_ROUTE_ALPHA = 40;
const INTER_ROUTE_ALPHA = 56;

/** One drawn supernode (a group of airports at its busiest airport). */
type SuperNode = {
  group: number;
  lonLat: readonly [number, number];
  airports: number;
  inside: number;
  leaving: number;
  name: string;
  color: number;
};

/** A compiled matrix graph of one resolution with its own count buffer and reader. */
type MatrixGraph = {
  resolution: number;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  windowBuffer: GPUParameterBuffer<'uint32'>;
  counts: Buffer;
  reader: SummaryReader;
  readsCounts: boolean;
};

/** Cubic ease in and out. */
function easeInOut(progress: number): number {
  return progress < 0.5 ? 4 * progress ** 3 : 1 - (-2 * progress + 2) ** 3 / 2;
}

/**
 * Adjacency matrix and coarsening of the world airline network. Three small graphs share the
 * buffers: `GPUAdjacencyMatrixOrder` sorts the airports by a group label (and a tie key such as
 * degree) into a position permutation; `GPUAdjacencyMatrix` bins the airport pairs through that
 * permutation and a zoom window into an R x R image (one compiled graph per resolution, compiled
 * when first asked for); `GPUNetworkCoarsening` collapses the network by continent or country into
 * supernodes and superedges. Ordering, window and labels are buffer writes; only the matrix
 * resolution compiles. A change of order is drawn as an animated reorder: every airport slides
 * from its old position to its new one while the GPU re-bins the matrix each frame.
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
  const continentCount = CONTINENT_NAMES.length;
  const groups = buildGroups(network);
  const countryCount = groups.countryNames.length;
  const projection = new LocalMetricProjection([0, 0]);

  // Undirected CSR: every route pair appears in both directions.
  const slotCount = network.edgeCount * 2;
  const offsets = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < network.edgeCount; edge++) {
    offsets[network.source[edge] + 1]++;
    offsets[network.target[edge] + 1]++;
  }
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const neighbors = new Uint32Array(slotCount);
  const cursor = offsets.slice(0, nodeCount);
  for (let edge = 0; edge < network.edgeCount; edge++) {
    neighbors[cursor[network.source[edge]]++] = network.target[edge];
    neighbors[cursor[network.target[edge]]++] = network.source[edge];
  }

  // CPU twins and standing statistics (the readouts of the story are computed from the data).
  const members = {
    continent: getMembersByDegree(groups.continent, continentCount, network.degree),
    shuffle: getMembersByDegree(groups.shuffled, continentCount, network.degree),
    country: getMembersByDegree(groups.country, countryCount, network.degree)
  };
  const chances: Record<'continent' | 'shuffle' | 'country', ChanceModel> = {
    continent: computeChance(groups.continent, continentCount, network.source, network.target),
    shuffle: computeChance(groups.shuffled, continentCount, network.source, network.target),
    country: computeChance(groups.country, countryCount, network.source, network.target)
  };
  const countryIntraShare = getIntraShare(groups.country, network.source, network.target);
  const superedgeMaximum = Math.max(
    1,
    getLargestSuperedge(groups.continent, network.source, network.target),
    getLargestSuperedge(groups.country, network.source, network.target)
  );
  const hubVertices = pickHubs(network, members.continent, HUB_LABEL_COUNT);
  const maximumDegree = Math.max(...network.degree);

  // GPU buffers shared by the graphs.
  const offsetsBuffer = resources.createBuffer('offsets', offsets);
  const neighborsBuffer = resources.createBuffer('neighbors', neighbors);
  const groupsBuffer = resources.createBuffer('order-groups', nodeCount * 4);
  const tieBuffer = resources.createBuffer('order-ties', nodeCount * 4);
  /** What `GPUAdjacencyMatrixOrder` computed. */
  const orderBuffer = resources.createBuffer('order', nodeCount * 4);
  /** What the matrix reads: the GPU order at rest, the sliding positions during a reorder. */
  const displayOrderBuffer = resources.createBuffer('display-order', nodeCount * 4);
  const labelsBuffer = resources.createBuffer('labels', nodeCount * 4);
  const positionsBuffer = resources.createBuffer('positions', network.lonLat);
  const degreeBuffer = resources.createBuffer('degree', Float32Array.from(network.degree));
  const airportColors = resources.createBuffer('airport-colors', groups.continent);
  const airportOrder = resources.createBuffer(
    'airport-order',
    Uint32Array.from(
      Array.from({length: nodeCount}, (_, vertex) => vertex).sort(
        (a, b) => network.degree[b] - network.degree[a] || a - b
      )
    )
  );

  // Route segments of the map: inside a continent (coloured by it) and between continents.
  const intraList: number[] = [];
  const intraContinents: number[] = [];
  const interList: number[] = [];
  for (let edge = 0; edge < expanded.source.length; edge++) {
    const original = expanded.original[edge];
    const a = network.source[original];
    const b = network.target[original];
    const sourcePosition = expanded.source[edge] * 2;
    const targetPosition = expanded.target[edge] * 2;
    const row = [
      expanded.positions[sourcePosition],
      expanded.positions[sourcePosition + 1],
      expanded.positions[targetPosition],
      expanded.positions[targetPosition + 1]
    ];
    if (groups.continent[a] === groups.continent[b]) {
      intraList.push(...row);
      intraContinents.push(groups.continent[a]);
    } else {
      interList.push(...row);
    }
  }
  const intraRouteCount = intraContinents.length;
  const interRouteCount = interList.length / 4;
  const intraSegments = resources.createBuffer('intra-segments', Float32Array.from(intraList));
  const intraColors = resources.createBuffer('intra-colors', Uint32Array.from(intraContinents));
  const interSegments = resources.createBuffer('inter-segments', Float32Array.from(interList));

  // Overlays of the matrix card (card coordinates, rewritten when the window or order changes).
  const lineCapacity = (GROUP_CAPACITY + 8) * 2;
  const countryLines = resources.createBuffer('country-lines', lineCapacity * 16);
  const continentLines = resources.createBuffer('continent-lines', 64 * 16);
  const frameLines = resources.createBuffer(
    'frame-lines',
    Float32Array.from([
      ...[-MATRIX_HALF, -MATRIX_HALF, MATRIX_HALF, -MATRIX_HALF],
      ...[MATRIX_HALF, -MATRIX_HALF, MATRIX_HALF, MATRIX_HALF],
      ...[MATRIX_HALF, MATRIX_HALF, -MATRIX_HALF, MATRIX_HALF],
      ...[-MATRIX_HALF, MATRIX_HALF, -MATRIX_HALF, -MATRIX_HALF]
    ])
  );
  const stripSegments = resources.createBuffer('strip-segments', 16 * 16);
  const stripColors = resources.createBuffer('strip-colors', 16 * 4);
  const overlayCounts = {countryLines: 0, continentLines: 0, strips: 0};

  // Supernodes and superedges of the map (rewritten after every coarsening readback).
  const nodePositions = resources.createBuffer('node-positions', GROUP_CAPACITY * 8);
  const nodeSizes = resources.createBuffer('node-sizes', GROUP_CAPACITY * 4);
  const nodeColors = resources.createBuffer('node-colors', GROUP_CAPACITY * 4);
  const flowEndpoints = resources.createBuffer('flow-endpoints', SUPEREDGE_DRAW_LIMIT * 2 * 16);
  const flowWeights = resources.createBuffer('flow-weights', SUPEREDGE_DRAW_LIMIT * 2 * 4);
  const flowOrder = resources.createBuffer('flow-order', SUPEREDGE_DRAW_LIMIT * 2 * 4);
  let superNodes: SuperNode[] = [];
  let superFlowCount = 0;
  let superNodeMaximum = 1;
  let superMode: 'continent' | 'country' = ctx.options.coarsenBy;

  // State of the matrix.
  const matrixGraphs = new Map<number, MatrixGraph>();
  let activeGraph: MatrixGraph | null = null;
  let orderGraphHandle: {
    resources: SpatialAnalysisResources;
    compiled: CompiledGPUCommandGraph<void>;
  };
  let coarsenGraphHandle: {
    resources: SpatialAnalysisResources;
    compiled: CompiledGPUCommandGraph<void>;
  };
  let metaReader: SummaryReader;
  let metaLayout: Record<string, {offset: number; length: number}> = {};
  let destroyed = false;
  let orderFrames = 3;
  let matrixFrames = 3;
  let coarsenFrames = 3;
  let statsStale = true;
  let lastChange = performance.now();
  let requestedMode: 'continent' | 'country' = ctx.options.coarsenBy;

  let currentOrder: MatrixOrder = ctx.options.order;
  /** Position of every airport after the CPU twin of the ordering. */
  let cpuOrder: Uint32Array = new Uint32Array(nodeCount);
  let inverseOrder = new Uint32Array(nodeCount);
  /** The positions the matrix is binned with right now (the CPU copy of `displayOrderBuffer`). */
  let displayedOrder = new Uint32Array(nodeCount);
  let blockLabels: Uint32Array | null = null;
  let blockMembers: number[][] = [];
  let continentStarts: number[] = [];
  let countryStarts: number[] = [];
  let matrixWindow: MatrixWindow = getMatrixWindow(
    {start: 0, extent: nodeCount},
    {start: 0, extent: nodeCount},
    2 ** ctx.options.resolutionPower
  );
  let lastCounts: Uint32Array | null = null;
  let lastCountsWindow: MatrixWindow | null = null;
  let lastCountsResolution = 0;
  let lastSuperSources: Uint32Array = new Uint32Array(0);
  let lastSuperTargets: Uint32Array = new Uint32Array(0);
  let lastSuperCounts: Uint32Array = new Uint32Array(0);
  let lastGroupVertexCount: Uint32Array = new Uint32Array(GROUP_CAPACITY);
  let lastGroupIntra: Uint32Array = new Uint32Array(GROUP_CAPACITY);
  let animation: {
    from: Uint32Array;
    to: Uint32Array;
    start: number;
    frame: number | null;
  } | null = null;
  const scratchOrder = new Uint32Array(nodeCount);

  // ---------------------------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------------------------

  function markChanged(): void {
    matrixFrames = Math.max(matrixFrames, 2);
    statsStale = true;
    lastChange = performance.now();
  }

  const getGroundTables = () => {
    const ground = ctx.ground();
    return {ground, count: getCountTable(ground), chance: getChanceTable(ground)};
  };

  const ORDER_WORDS: Record<MatrixOrder, string> = {
    input: 'IATA code, A to Z',
    shuffle: 'shuffled continent labels (null)',
    degree: 'routes, busiest first',
    continent: 'continent, busiest first',
    country: 'country, busiest first'
  };

  /** The standing sample line, and on the matrix steps the order in the cartouche subtitle. */
  function publishFurniture(): void {
    const sample = `${formatCount(nodeCount)} airports, ${formatCount(network.edgeCount)} route pairs, 2014`;
    ctx.setFurniture({
      title:
        ctx.options.view === 'matrix'
          ? {sample, subtitle: `Airport × airport, ordered by ${ORDER_WORDS[ctx.options.order]}`}
          : {sample}
    });
  }

  function publishLegendData(): void {
    const {ground, count, chance} = getGroundTables();
    ctx.setLegendData('ground', ground);
    ctx.setLegendData('countTable', count);
    ctx.setLegendData('chanceTable', chance);
    ctx.setLegendData('maximumDegree', maximumDegree);
    ctx.setLegendData(
      'continentSizes',
      getBlockStarts(groups.continent, continentCount)
        .slice(1)
        .map((end, index, all) => end - (all[index - 1] ?? 0))
    );
  }

  const getGroupName = (mode: 'continent' | 'country', group: number): string =>
    mode === 'continent'
      ? (CONTINENT_NAMES[group] ?? `Group ${group}`)
      : (groups.countryNames[group] ?? `Group ${group}`);

  /** Card metres to the `[longitude, latitude]` annotations use. */
  const cardToLngLat = (x: number, y: number): [number, number] => projection.unproject(x, y);

  const positionToX = (position: number): number =>
    -MATRIX_HALF + ((position - matrixWindow.colStart) / matrixWindow.span) * MATRIX_SIDE;
  const positionToY = (position: number): number =>
    MATRIX_HALF - ((position - matrixWindow.rowStart) / matrixWindow.span) * MATRIX_SIDE;

  function getOverlayOpacity(): number {
    if (!animation) return 1;
    const progress = Math.min(
      1,
      (performance.now() - animation.start) / ORDER_TRANSITION_MILLISECONDS
    );
    return Math.max(0, (progress - 0.7) / 0.3);
  }

  // ---------------------------------------------------------------------------------------------
  // Order, window, labels
  // ---------------------------------------------------------------------------------------------

  function writeOrderInputs(animate: boolean): void {
    const order = ctx.options.order;
    const {groupKeys, tieKeys} = getOrderKeys(groups, order);
    groupsBuffer.write(groupKeys);
    tieBuffer.write(tieKeys);
    // The CPU twin gives the same permutation: block boundaries, hover and verification.
    const previousDisplay = Uint32Array.from(displayedOrder);
    cpuOrder = computeAdjacencyMatrixOrder(groupKeys, tieKeys);
    inverseOrder = new Uint32Array(nodeCount);
    for (let vertex = 0; vertex < nodeCount; vertex++) inverseOrder[cpuOrder[vertex]] = vertex;
    currentOrder = order;
    blockLabels = getOrderBlocks(groups, order);
    continentStarts =
      order === 'continent' || order === 'shuffle' || order === 'country'
        ? getBlockStarts(groups.continent, continentCount)
        : [];
    countryStarts = order === 'country' ? getBlockStarts(groups.country, countryCount) : [];
    blockMembers =
      order === 'continent'
        ? members.continent
        : order === 'shuffle'
          ? members.shuffle
          : order === 'country'
            ? members.country
            : [];
    orderFrames = 3;
    publishOrderStatistics();
    writeWindow();
    markChanged();
    if (animate && !ctx.reducedMotion() && activeGraph) {
      startAnimation(previousDisplay, cpuOrder);
    } else {
      stopAnimation();
      displayedOrder = Uint32Array.from(cpuOrder);
      displayOrderBuffer.write(displayedOrder);
    }
    updateAnnotations();
  }

  /** Readouts and chart that depend on the grouping of the current order. */
  function publishOrderStatistics(): void {
    const key =
      currentOrder === 'shuffle' ? 'shuffle' : currentOrder === 'country' ? 'country' : 'continent';
    const model = chances[key];
    const hasBlocks = blockLabels !== null;
    ctx.setReadout('blockShare', hasBlocks ? model.observedShare : null);
    ctx.setReadout('expectedShare', hasBlocks ? model.expectedShare : null);
    ctx.setReadout('modularity', hasBlocks ? model.modularity : null);
    if (key === 'country') {
      ctx.setChart('chanceMatrix', null);
      return;
    }
    const table = getChanceTable(ctx.ground());
    // A pair of groups with no route at all is an empty cell, not the lowest class.
    const values = Array.from(model.logRatio, value =>
      Number.isFinite(value) ? value : Number.NaN
    );
    ctx.setChart('chanceMatrix', {
      kind: 'matrix',
      values,
      rows: continentCount,
      columns: continentCount,
      rowLabels: [...CONTINENT_NAMES],
      columnLabels: CONTINENT_NAMES.map(name =>
        name.replace('North America', 'N. America').replace('South America', 'S. America')
      ),
      cellColors: values.map(value =>
        Number.isNaN(value)
          ? (table.noData?.color ?? [200, 200, 200, 255])
          : table.colors[Math.max(0, getClassIndexOf(table, value))]
      ),
      midpoint: 0,
      diagonal: true,
      marginals: 'none',
      formatCell: value =>
        Number.isNaN(value)
          ? 'none'
          : `${2 ** value >= 10 ? Math.round(2 ** value) : (2 ** value).toFixed(1)}x`,
      title:
        key === 'shuffle'
          ? 'Routes against chance, shuffled labels'
          : 'Routes against chance, by continent',
      description:
        'Routes between continents divided by the number expected if every airport kept its number of routes but the route ends were paired at random. Orange is more than chance, purple fewer, grey as expected. The diagonal is routes that stay inside a continent.',
      table: false
    });
  }

  function getAxisWindows(): {
    rows: {start: number; extent: number};
    columns: {start: number; extent: number};
  } {
    const {focus, zoom, panX, panY} = ctx.options;
    const focusIndex = CONTINENT_NAMES.indexOf(focus as (typeof CONTINENT_NAMES)[number]);
    if (focus !== 'all' && focusIndex >= 0 && continentStarts.length > 0) {
      const start = continentStarts[focusIndex];
      const extent = Math.max(1, continentStarts[focusIndex + 1] - start);
      return {rows: {start, extent}, columns: {start, extent}};
    }
    const extent = Math.max(8, Math.round(nodeCount / zoom));
    const clampStart = (center: number) =>
      Math.min(nodeCount - extent, Math.max(0, Math.round(center * nodeCount - extent / 2)));
    return {
      rows: {start: clampStart(panY), extent},
      columns: {start: clampStart(panX), extent}
    };
  }

  function writeWindow(): void {
    const resolution = activeGraph?.resolution ?? 2 ** ctx.options.resolutionPower;
    const {rows, columns} = getAxisWindows();
    matrixWindow = getMatrixWindow(rows, columns, resolution);
    if (activeGraph) {
      activeGraph.windowBuffer.write(
        encodeGPUAdjacencyMatrixWindow(
          {
            rowStart: matrixWindow.rowStart,
            rowEnd: matrixWindow.rowEnd,
            colStart: matrixWindow.colStart,
            colEnd: matrixWindow.colEnd
          },
          resolution
        )
      );
    }
    const {positionsPerCell, cells, extent} = matrixWindow;
    ctx.setReadout(
      'cellScale',
      positionsPerCell === 1
        ? `1 airport per cell, ${formatCount(cells)} × ${formatCount(cells)} cells${resolution > extent ? ' (clamped)' : ''}`
        : `${positionsPerCell} × ${positionsPerCell} airports per cell, ${formatCount(cells)} × ${formatCount(cells)} cells`
    );
    ctx.setReadout('blockAirports', extent);
    ctx.setReadout('fillBound', Math.min(1, slotCount / (cells * cells)));
    ctx.setReadout(
      'window',
      `rows ${formatCount(matrixWindow.rowStart)} to ${formatCount(matrixWindow.rowStart + extent)}, columns ${formatCount(matrixWindow.colStart)} to ${formatCount(matrixWindow.colStart + extent)}`
    );
    writeOverlays();
    markChanged();
    updateAnnotations();
  }

  function writeLabels(): void {
    const {coarsenBy} = ctx.options;
    labelsBuffer.write(coarsenBy === 'continent' ? groups.continent : groups.country);
    coarsenFrames = 3;
    markChanged();
  }

  /** Block boundaries, frame and continent strips in card coordinates, clipped to the window. */
  function writeOverlays(): void {
    const w = matrixWindow;
    const columnEnd = w.colStart + w.extent;
    const rowEnd = w.rowStart + w.extent;
    const boundary = (starts: number[], list: number[][]) => {
      for (const position of starts.slice(1, -1)) {
        if (position > w.colStart && position < columnEnd) {
          list.push([positionToX(position), -MATRIX_HALF, positionToX(position), MATRIX_HALF]);
        }
        if (position > w.rowStart && position < rowEnd) {
          list.push([-MATRIX_HALF, positionToY(position), MATRIX_HALF, positionToY(position)]);
        }
      }
    };
    const continentList: number[][] = [];
    boundary(continentStarts, continentList);
    const countryList: number[][] = [];
    boundary(countryStarts, countryList);
    const flat = (list: number[][]) => Float32Array.from(list.flat());
    continentLines.write(flat(continentList.slice(0, 64)));
    overlayCounts.continentLines = Math.min(64, continentList.length);
    countryLines.write(flat(countryList.slice(0, lineCapacity)));
    overlayCounts.countryLines = Math.min(lineCapacity, countryList.length);
    const strips: number[][] = [];
    const colors: number[] = [];
    const gap = MATRIX_SIDE * STRIP_GAP_SHARE;
    for (let index = 0; index + 1 < continentStarts.length; index++) {
      const from = continentStarts[index];
      const to = continentStarts[index + 1];
      if (to <= from) continue;
      const columnFrom = Math.max(from, w.colStart);
      const columnTo = Math.min(to, columnEnd);
      if (columnTo > columnFrom) {
        strips.push([
          positionToX(columnFrom),
          MATRIX_HALF + gap,
          positionToX(columnTo),
          MATRIX_HALF + gap
        ]);
        colors.push(index);
      }
      const rowFrom = Math.max(from, w.rowStart);
      const rowTo = Math.min(to, rowEnd);
      if (rowTo > rowFrom) {
        strips.push([
          -MATRIX_HALF - gap,
          positionToY(rowFrom),
          -MATRIX_HALF - gap,
          positionToY(rowTo)
        ]);
        colors.push(index);
      }
    }
    stripSegments.write(flat(strips.slice(0, 16)));
    stripColors.write(Uint32Array.from(colors.slice(0, 16)));
    overlayCounts.strips = Math.min(16, strips.length);
  }

  // ---------------------------------------------------------------------------------------------
  // Animated reorder
  // ---------------------------------------------------------------------------------------------

  function stopAnimation(): void {
    if (animation?.frame != null) cancelAnimationFrame(animation.frame);
    animation = null;
  }

  function startAnimation(from: Uint32Array, to: Uint32Array): void {
    stopAnimation();
    animation = {from, to, start: performance.now(), frame: null};
    ctx.setAnnotations('matrix-axes', null);
    const tick = () => {
      if (!animation || destroyed) return;
      const progress = Math.min(
        1,
        (performance.now() - animation.start) / ORDER_TRANSITION_MILLISECONDS
      );
      const eased = easeInOut(progress);
      for (let vertex = 0; vertex < nodeCount; vertex++) {
        const start = animation.from[vertex];
        scratchOrder[vertex] = Math.round(start + (animation.to[vertex] - start) * eased);
      }
      displayedOrder = Uint32Array.from(scratchOrder);
      displayOrderBuffer.write(scratchOrder);
      matrixFrames = Math.max(matrixFrames, 2);
      lastChange = performance.now();
      statsStale = true;
      ctx.requestLayers();
      if (progress >= 1) {
        animation = null;
        displayedOrder = Uint32Array.from(cpuOrder);
        // Hand the matrix back to the GPU permutation: copy it over the sliding positions.
        orderFrames = Math.max(orderFrames, 2);
        markChanged();
        updateAnnotations();
        ctx.requestLayers();
        return;
      }
      animation.frame = requestAnimationFrame(tick);
    };
    animation.frame = requestAnimationFrame(tick);
  }

  // ---------------------------------------------------------------------------------------------
  // Graphs
  // ---------------------------------------------------------------------------------------------

  function buildOrderGraph() {
    const graphResources = new SpatialAnalysisResources(device, 'flight-matrix-order');
    const commandGraph = new GPUCommandGraph<void>(device, {id: 'flight-matrix-order'});
    commandGraph.add(
      new GPUAdjacencyMatrixOrder({
        id: 'order',
        groups: importGraphBuffer(commandGraph, 'groups', groupsBuffer, 'uint32', nodeCount),
        tieKeys: importGraphBuffer(commandGraph, 'ties', tieBuffer, 'uint32', nodeCount),
        order: importGraphBuffer(commandGraph, 'order', orderBuffer, 'uint32', nodeCount)
      })
    );
    return {resources: graphResources, compiled: graphResources.track(commandGraph.compile())};
  }

  function buildMatrixGraph(resolution: number): MatrixGraph {
    const graphResources = new SpatialAnalysisResources(device, `flight-matrix-${resolution}`);
    const cellCount = resolution * resolution;
    const counts = graphResources.createBuffer('counts', cellCount * 4);
    const maxCount = graphResources.createBuffer('max-count', 4);
    const windowBuffer = graphResources.createParameterBuffer('window', 'uint32', 4);
    const commandGraph = new GPUCommandGraph<void>(device, {id: `flight-matrix-${resolution}`});
    commandGraph.add(
      new GPUAdjacencyMatrix({
        id: 'matrix',
        offsets: importGraphBuffer(commandGraph, 'offsets', offsetsBuffer, 'uint32', nodeCount + 1),
        neighbors: importGraphBuffer(
          commandGraph,
          'neighbors',
          neighborsBuffer,
          'uint32',
          slotCount
        ),
        order: importGraphBuffer(commandGraph, 'order', displayOrderBuffer, 'uint32', nodeCount),
        window: windowBuffer.importToGraph(commandGraph),
        resolution,
        output: {
          counts: importGraphBuffer(commandGraph, 'counts', counts, 'uint32', cellCount),
          maxCount: importGraphBuffer(commandGraph, 'max-count', maxCount, 'uint32', 1)
        }
      })
    );
    const compiled = graphResources.track(commandGraph.compile());
    const readsCounts = resolution <= MAXIMUM_COUNT_READBACK_RESOLUTION;
    const sources = [{buffer: maxCount, size: 4}];
    if (readsCounts) sources.push({buffer: counts, size: cellCount * 4});
    const built: MatrixGraph = {
      resolution,
      resources: graphResources,
      compiled,
      windowBuffer,
      counts,
      reader: undefined as unknown as SummaryReader,
      readsCounts
    };
    built.reader = new SummaryReader(
      graphResources,
      `flight-matrix-${resolution}`,
      sources,
      bytes => {
        if (!destroyed && activeGraph === built) processMatrix(built, bytes);
      }
    );
    return built;
  }

  /** The slots the matrix pass visits and the compute passes of the three graphs. */
  function publishCost(): void {
    ctx.setCost({
      records: slotCount,
      passes:
        orderGraphHandle.compiled.stats.nodeOrder.length +
        (activeGraph?.compiled.stats.nodeOrder.length ?? 0) +
        coarsenGraphHandle.compiled.stats.nodeOrder.length
    });
  }

  function activateResolution(): void {
    const resolution = 2 ** ctx.options.resolutionPower;
    let next = matrixGraphs.get(resolution);
    if (!next) {
      next = buildMatrixGraph(resolution);
      matrixGraphs.set(resolution, next);
    }
    activeGraph = next;
    lastCounts = null;
    ctx.setReadout('resolution', `${resolution} × ${resolution} bins`);
    writeWindow();
    if (orderGraphHandle && coarsenGraphHandle) publishCost();
  }

  function buildCoarsenGraph() {
    const graphResources = new SpatialAnalysisResources(device, 'flight-matrix-coarsen');
    const groupVertexCount = graphResources.createBuffer('group-vertex-count', GROUP_CAPACITY * 4);
    const groupIntra = graphResources.createBuffer('group-intra', GROUP_CAPACITY * 4);
    const edgeIds = graphResources.createBuffer('super-ids', SUPEREDGE_CAPACITY * 4);
    const edgeCount = graphResources.createBuffer('super-count', 4);
    const edgeOverflow = graphResources.createBuffer('super-overflow', 4);
    const edgeTargets = graphResources.createBuffer('super-targets', SUPEREDGE_CAPACITY * 4);
    const edgeCounts = graphResources.createBuffer('super-counts', SUPEREDGE_CAPACITY * 4);
    const summary = graphResources.createBuffer(
      'summary',
      GPU_NETWORK_COARSENING_SUMMARY_LENGTH * 4
    );
    const commandGraph = new GPUCommandGraph<void>(device, {id: 'flight-matrix-coarsen'});
    const u32 = (name: string, buffer: Buffer, length: number) =>
      importGraphBuffer(commandGraph, name, buffer, 'uint32', length);
    commandGraph.add(
      new GPUNetworkCoarsening({
        id: 'coarsen',
        offsets: u32('offsets', offsetsBuffer, nodeCount + 1),
        neighbors: u32('neighbors', neighborsBuffer, slotCount),
        labels: u32('labels', labelsBuffer, nodeCount),
        groupCapacity: GROUP_CAPACITY,
        groupVertexCount: u32('group-vertex-count', groupVertexCount, GROUP_CAPACITY),
        groupIntraEdgeCount: u32('group-intra', groupIntra, GROUP_CAPACITY),
        edges: {
          ids: u32('super-ids', edgeIds, SUPEREDGE_CAPACITY),
          count: u32('super-count', edgeCount, 1),
          overflow: u32('super-overflow', edgeOverflow, 1)
        },
        edgeTargets: u32('super-targets', edgeTargets, SUPEREDGE_CAPACITY),
        edgeCounts: u32('super-counts', edgeCounts, SUPEREDGE_CAPACITY),
        summary: u32('summary', summary, GPU_NETWORK_COARSENING_SUMMARY_LENGTH)
      })
    );
    const compiled = graphResources.track(commandGraph.compile());
    // One readback of everything small: the order (to check it), the summary and the group tables.
    const sourceList = [
      {name: 'order', buffer: orderBuffer, size: nodeCount * 4},
      {name: 'summary', buffer: summary, size: GPU_NETWORK_COARSENING_SUMMARY_LENGTH * 4},
      {name: 'groupVertexCount', buffer: groupVertexCount, size: GROUP_CAPACITY * 4},
      {name: 'groupIntra', buffer: groupIntra, size: GROUP_CAPACITY * 4},
      {name: 'superIds', buffer: edgeIds, size: SUPEREDGE_CAPACITY * 4},
      {name: 'superTargets', buffer: edgeTargets, size: SUPEREDGE_CAPACITY * 4},
      {name: 'superCounts', buffer: edgeCounts, size: SUPEREDGE_CAPACITY * 4},
      {name: 'superCount', buffer: edgeCount, size: 4},
      {name: 'superOverflow', buffer: edgeOverflow, size: 4}
    ];
    let offset = 0;
    metaLayout = {};
    for (const source of sourceList) {
      metaLayout[source.name] = {offset, length: source.size / 4};
      offset += source.size;
    }
    metaReader = new SummaryReader(
      graphResources,
      'flight-matrix-meta',
      sourceList.map(({buffer, size}) => ({buffer, size})),
      bytes => {
        if (!destroyed) processMeta(bytes);
      }
    );
    return {resources: graphResources, compiled};
  }

  // ---------------------------------------------------------------------------------------------
  // Readbacks
  // ---------------------------------------------------------------------------------------------

  function processMatrix(current: MatrixGraph, bytes: ArrayBuffer): void {
    if (animation) {
      statsStale = true;
      return;
    }
    const words = new Uint32Array(bytes);
    lastCounts = current.readsCounts ? words.slice(1, 1 + current.resolution ** 2) : null;
    lastCountsWindow = matrixWindow;
    lastCountsResolution = current.resolution;
    summariseCounts();
    ctx.requestLayers();
  }

  /** Cells per class, occupied cells and the cell chart of the visible `cells x cells` part. */
  function summariseCounts(): void {
    const table = getCountTable(ctx.ground());
    if (!lastCounts || !lastCountsWindow) {
      ctx.setReadout('occupied', null);
      ctx.setChart('cellChart', null);
      ctx.setLegendData('cellCounts', undefined);
      ctx.setLegendData('emptyCells', undefined);
      return;
    }
    const {cells} = lastCountsWindow;
    const perClass = new Array<number>(table.colors.length).fill(0);
    let occupied = 0;
    for (let row = 0; row < cells; row++) {
      for (let column = 0; column < cells; column++) {
        const count = lastCounts[row * lastCountsResolution + column];
        if (count > 0) {
          occupied++;
          perClass[Math.max(0, getClassIndexOf(table, count))]++;
        }
      }
    }
    ctx.setReadout('occupied', occupied);
    ctx.setLegendData('cellCounts', perClass);
    ctx.setLegendData('emptyCells', cells * cells - occupied);
    ctx.setChart('cellChart', {
      kind: 'bars',
      values: perClass,
      labels: [...COUNT_LABELS],
      colors: table.colors,
      height: 110,
      xLabel: 'airport pairs in a cell',
      yLabel: 'cells',
      formatY: value => (value >= 1000 ? `${Math.round(value / 1000)}k` : `${value}`),
      table: false,
      description:
        'Occupied matrix cells by how many airport pairs they hold. Most cells hold one or two pairs: the matrix is sparse, and its colours are small integers.'
    });
  }

  function processMeta(bytes: ArrayBuffer): void {
    const words = new Uint32Array(bytes);
    const get = (name: string): Uint32Array => {
      const {offset, length} = metaLayout[name];
      return words.slice(offset / 4, offset / 4 + length);
    };
    const gpuOrder = get('order');
    let matching = 0;
    for (let vertex = 0; vertex < nodeCount; vertex++) {
      if (gpuOrder[vertex] === cpuOrder[vertex]) matching++;
    }
    ctx.setReadout(
      'orderCheck',
      matching === nodeCount
        ? `identical to the CPU twin (${formatCount(nodeCount)} airports)`
        : `MISMATCH: ${formatCount(nodeCount - matching)} airports differ`
    );
    const summary = decodeGPUNetworkCoarseningSummary(get('summary'));
    lastGroupVertexCount = get('groupVertexCount');
    lastGroupIntra = get('groupIntra');
    const superCount = Math.min(get('superCount')[0], SUPEREDGE_CAPACITY);
    lastSuperSources = get('superIds').slice(0, superCount);
    lastSuperTargets = get('superTargets').slice(0, superCount);
    lastSuperCounts = get('superCounts').slice(0, superCount);
    superMode = requestedMode;
    ctx.setReadout('groups', summary.groupCount);
    ctx.setReadout(
      'superedges',
      `${formatCount(summary.superedgeCount)}${get('superOverflow')[0] ? ' (list truncated)' : ''}`
    );
    ctx.setReadout(
      'intraShare',
      summary.countedEdgeCount > 0 ? summary.intraEdgeCount / summary.countedEdgeCount : null
    );
    ctx.setReadout('dropped', summary.droppedEdgeCount + summary.overflowedVertexCount);
    let largest = -1;
    for (let edge = 0; edge < lastSuperCounts.length; edge++) {
      if (largest < 0 || lastSuperCounts[edge] > lastSuperCounts[largest]) largest = edge;
    }
    ctx.setReadout(
      'largestPair',
      largest >= 0
        ? `${getGroupName(superMode, lastSuperSources[largest])} and ${getGroupName(superMode, lastSuperTargets[largest])}: ${formatCount(lastSuperCounts[largest])} routes`
        : null
    );
    rebuildSuperDrawing();
    updateAnnotations();
    ctx.requestLayers();
  }

  /** Supernodes at the busiest airport of each group and the heaviest superedges as flows. */
  function rebuildSuperDrawing(): void {
    const mode = superMode;
    const groupMembers = mode === 'continent' ? members.continent : members.country;
    const external = new Float64Array(GROUP_CAPACITY);
    for (let edge = 0; edge < lastSuperCounts.length; edge++) {
      external[lastSuperSources[edge]] += lastSuperCounts[edge];
      external[lastSuperTargets[edge]] += lastSuperCounts[edge];
    }
    const nodes: SuperNode[] = [];
    for (let group = 0; group < GROUP_CAPACITY; group++) {
      const count = lastGroupVertexCount[group];
      if (count === 0 || !groupMembers[group]?.length) continue;
      const anchor = groupMembers[group][0];
      nodes.push({
        group,
        lonLat: [network.lonLat[anchor * 2], network.lonLat[anchor * 2 + 1]],
        airports: count,
        inside: lastGroupIntra[group],
        leaving: external[group],
        name: getGroupName(mode, group),
        color: mode === 'continent' ? group : groups.countryContinent[group]
      });
    }
    // Largest first: the small discs sit on top of the big ones.
    nodes.sort((a, b) => b.airports - a.airports || a.group - b.group);
    superNodes = nodes;
    superNodeMaximum = Math.max(1, ...nodes.map(node => node.airports));
    if (nodes.length > 0) {
      nodePositions.write(
        Float32Array.from(nodes.flatMap(node => [node.lonLat[0], node.lonLat[1]]))
      );
      nodeSizes.write(Float32Array.from(nodes.map(node => node.airports)));
      nodeColors.write(Uint32Array.from(nodes.map(node => node.color)));
    }

    const anchorOf = new Map(nodes.map(node => [node.group, node.lonLat] as const));
    const ranked = Array.from({length: lastSuperCounts.length}, (_, edge) => edge)
      .sort((a, b) => lastSuperCounts[b] - lastSuperCounts[a] || a - b)
      .slice(0, SUPEREDGE_DRAW_LIMIT);
    const endpoints: number[] = [];
    const weights: number[] = [];
    for (const edge of ranked) {
      const from = anchorOf.get(lastSuperSources[edge]);
      const to = anchorOf.get(lastSuperTargets[edge]);
      if (!from || !to) continue;
      const count = lastSuperCounts[edge];
      const difference = to[0] - from[0];
      const pairs: [readonly [number, number], readonly [number, number]][] = [];
      if (Math.abs(difference) <= 180) {
        pairs.push([from, to]);
      } else {
        // A route over the antimeridian is two flows, each leaving the visible world on its own side.
        const shift = difference > 0 ? -360 : 360;
        pairs.push([from, [to[0] + shift, to[1]]], [[from[0] - shift, from[1]], to]);
      }
      for (const [start, end] of pairs) {
        if (weights.length >= SUPEREDGE_DRAW_LIMIT * 2) break;
        endpoints.push(
          ...projection.project(start[0], start[1]),
          ...projection.project(end[0], end[1])
        );
        weights.push(count);
      }
    }
    superFlowCount = weights.length;
    if (weights.length > 0) {
      flowEndpoints.write(Float32Array.from(endpoints));
      flowWeights.write(Float32Array.from(weights));
      // Heaviest drawn last.
      flowOrder.write(
        Uint32Array.from(
          Array.from({length: weights.length}, (_, index) => index).sort(
            (a, b) => weights[a] - weights[b] || a - b
          )
        )
      );
    }
    const radius = mode === 'continent' ? 38 : 26;
    ctx.setLegendData('superNodes', {
      mode,
      maximum: superNodeMaximum,
      maximumRadius: radius,
      flowMaximum: superedgeMaximum
    });
  }

  // ---------------------------------------------------------------------------------------------
  // Annotations
  // ---------------------------------------------------------------------------------------------

  function updateAnnotations(): void {
    const {view, showGroups, showBlocks} = ctx.options;
    if (view === 'map') {
      ctx.setAnnotations('matrix-axes', null);
      if (showGroups) {
        ctx.setAnnotations('hubs', null);
        ctx.setAnnotations(
          'supernodes',
          superNodes.slice(0, 6).map(
            (node, index): MapAnnotation => ({
              kind: 'point',
              coordinate: node.lonLat,
              text: node.name,
              detail: `${formatPercent(node.inside / Math.max(1, node.inside + node.leaving))} of routes stay inside`,
              marker: 'none',
              rank: 'subject',
              priority: 10 - index,
              id: `super:${superMode}:${node.group}`
            })
          )
        );
      } else {
        ctx.setAnnotations('supernodes', null);
        ctx.setAnnotations(
          'hubs',
          hubVertices.map(
            (vertex, index): MapAnnotation => ({
              kind: 'point',
              coordinate: [network.lonLat[vertex * 2], network.lonLat[vertex * 2 + 1]],
              text: airports[vertex].iata,
              detail: airports[vertex].city,
              rank: index < continentCount ? 'subject' : 'context',
              priority: 10 - index,
              id: `hub:${airports[vertex].iata}`
            })
          )
        );
      }
      return;
    }
    ctx.setAnnotations('hubs', null);
    ctx.setAnnotations('supernodes', null);
    if (animation || !showBlocks) {
      ctx.setAnnotations('matrix-axes', null);
      return;
    }
    const list: MapAnnotation[] = [];
    const w = matrixWindow;
    const labelX = -MATRIX_HALF - MATRIX_SIDE * LABEL_GAP_SHARE;
    const captions: Record<MatrixOrder, string> = {
      input: 'A to Z by IATA code',
      shuffle: 'Shuffled labels, same sizes',
      degree: 'Busiest first',
      continent: 'By continent, busiest first',
      country: 'By country, busiest first'
    };
    list.push({
      kind: 'area',
      coordinate: cardToLngLat(0, MATRIX_HALF + MATRIX_SIDE * 0.06),
      text: captions[currentOrder],
      size: 'small',
      priority: 1,
      id: 'matrix-caption'
    });
    const rowEnd = w.rowStart + w.extent;
    const focusIndex = CONTINENT_NAMES.indexOf(
      ctx.options.focus as (typeof CONTINENT_NAMES)[number]
    );
    const visibleBlocks = (starts: number[]) => {
      const result: {group: number; from: number; to: number}[] = [];
      for (let group = 0; group + 1 < starts.length; group++) {
        const from = Math.max(starts[group], w.rowStart);
        const to = Math.min(starts[group + 1], rowEnd);
        if (to > from) result.push({group, from, to});
      }
      return result;
    };
    if (continentStarts.length > 0) {
      const blockPlaces: {group: number; from: number; to: number; text: string; detail: string}[] =
        [];
      if (currentOrder === 'country' && focusIndex >= 0) {
        // One continent in the window: name its largest countries from the data.
        const countries = visibleBlocks(countryStarts)
          .filter(block => groups.countryContinent[block.group] === focusIndex)
          .sort((a, b) => b.to - b.from - (a.to - a.from))
          .slice(0, 6);
        for (const block of countries) {
          blockPlaces.push({
            ...block,
            text: groups.countryNames[block.group],
            detail: airports[members.country[block.group][0]].iata
          });
        }
      } else {
        for (const block of visibleBlocks(continentStarts)) {
          if (block.to - block.from < w.extent * 0.03) continue;
          const hubs = (blockMembers[block.group] ?? [])
            .slice(0, 3)
            .map(vertex => airports[vertex].iata)
            .join(' ');
          blockPlaces.push({
            ...block,
            text:
              currentOrder === 'shuffle'
                ? `Shuffled ${CONTINENT_NAMES[block.group]}`
                : CONTINENT_NAMES[block.group],
            detail: hubs
          });
        }
      }
      blockPlaces.forEach((block, index) => {
        list.push({
          kind: 'point',
          coordinate: cardToLngLat(labelX, positionToY((block.from + block.to) / 2)),
          text: block.text,
          detail: block.detail,
          anchor: 'w',
          marker: 'none',
          rank: 'subject',
          priority: 8 - index,
          id: `axis:${currentOrder}:${block.group}`
        });
      });
      // Notes: what share of each big block's route ends stays inside it (from the chance model).
      if (focusIndex < 0 && currentOrder !== 'country') {
        const model = currentOrder === 'shuffle' ? chances.shuffle : chances.continent;
        if (currentOrder === 'shuffle') {
          list.push({
            kind: 'note',
            coordinate: cardToLngLat(
              positionToX((continentStarts[0] + continentStarts[1]) / 2),
              positionToY((continentStarts[0] + continentStarts[1]) / 2)
            ),
            title: liveText('{share:percent} stay inside their group', {
              share: model.observedShare
            }),
            text: 'With labels shuffled, no better than chance',
            anchor: 'ne',
            distance: 46,
            priority: 6,
            id: 'matrix-note:shuffle'
          });
        } else {
          const biggest = Array.from({length: continentCount}, (_, group) => group)
            .sort(
              (a, b) =>
                model.observed[b * continentCount + b] - model.observed[a * continentCount + a]
            )
            .slice(0, NOTE_BLOCK_COUNT);
          for (const group of biggest) {
            const centre = (continentStarts[group] + continentStarts[group + 1]) / 2;
            list.push({
              kind: 'note',
              coordinate: cardToLngLat(positionToX(centre), positionToY(centre)),
              title: liveText('{share:percent} of routes stay home', {
                share: model.observed[group * continentCount + group] / model.groupDegree[group]
              }),
              text: CONTINENT_NAMES[group],
              anchor: 'ne',
              distance: 40,
              priority: 5,
              id: `matrix-note:${group}`
            });
          }
        }
      }
    }
    ctx.setAnnotations('matrix-axes', list);
  }

  // ---------------------------------------------------------------------------------------------
  // Tooltips
  // ---------------------------------------------------------------------------------------------

  function getMatrixTooltip(coordinate: readonly [number, number]): TooltipContent | null {
    if (animation || !activeGraph) return null;
    const [x, y] = projection.project(coordinate[0], coordinate[1]);
    const u = (x + MATRIX_HALF) / MATRIX_SIDE;
    const v = (MATRIX_HALF - y) / MATRIX_SIDE;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return null;
    const w = matrixWindow;
    const column = Math.floor(u * w.cells);
    const row = Math.floor(v * w.cells);
    const rowFrom = w.rowStart + row * w.positionsPerCell;
    const columnFrom = w.colStart + column * w.positionsPerCell;
    const describe = (from: number) => {
      const list: string[] = [];
      let hub = -1;
      for (
        let position = from;
        position < from + w.positionsPerCell && position < nodeCount;
        position++
      ) {
        const vertex = inverseOrder[position];
        if (hub < 0 || network.degree[vertex] > network.degree[hub]) hub = vertex;
        if (list.length < 4) list.push(airports[vertex].iata);
      }
      const last = Math.min(nodeCount, from + w.positionsPerCell);
      return {
        hub,
        range:
          last - from > 1
            ? `positions ${formatCount(from)} to ${formatCount(last - 1)}`
            : `position ${formatCount(from)}`,
        names: list.join(' ') + (last - from > list.length ? ' ...' : '')
      };
    };
    if (rowFrom >= nodeCount || columnFrom >= nodeCount) return null;
    const rows = describe(rowFrom);
    const columns = describe(columnFrom);
    const table = getCountTable(ctx.ground());
    const count =
      lastCounts && lastCountsWindow === w && lastCountsResolution === activeGraph.resolution
        ? lastCounts[row * activeGraph.resolution + column]
        : null;
    const tooltipRows: TooltipContent['rows'] = [
      count === null
        ? {label: 'Airport pairs', value: 'not read back at this resolution', emphasis: true}
        : {
            label: 'Airport pairs',
            value: count === 0 ? 'none' : formatCount(count),
            unit: count === 0 ? undefined : 'in this cell',
            swatch:
              count === 0
                ? getSheetColors(ctx.ground()).plot
                : table.colors[getClassIndexOf(table, count)],
            emphasis: true
          },
      ...(count !== null && count > 0
        ? [{label: 'Class', value: getClassLabel(table, getClassIndexOf(table, count))}]
        : []),
      {label: 'Rows', value: `${rows.range}: ${rows.names}`},
      {label: 'Columns', value: `${columns.range}: ${columns.names}`}
    ];
    return {
      title: `${airports[rows.hub].iata} and ${airports[columns.hub].iata}`,
      subtitle: 'Busiest airport of the row and column cells',
      rows: tooltipRows
    };
  }

  function getAirportTooltip(pixel: readonly [number, number]): TooltipContent | null {
    const viewport = ctx.getViewport();
    if (!viewport) return null;
    let best = -1;
    let bestDistance = 12;
    for (let vertex = 0; vertex < nodeCount; vertex++) {
      const [px, py] = viewport.project([
        network.lonLat[vertex * 2],
        network.lonLat[vertex * 2 + 1]
      ]);
      const distance = Math.hypot(px - pixel[0], py - pixel[1]);
      if (
        distance < bestDistance ||
        (distance === bestDistance && best >= 0 && network.degree[vertex] > network.degree[best])
      ) {
        bestDistance = distance;
        best = vertex;
      }
    }
    if (best < 0) return null;
    const airport = airports[best];
    const palette = getContinentPalette(ctx.ground());
    return {
      title: `${airport.iata} ${airport.name}`,
      subtitle: `${airport.city}, ${airport.country}`,
      rows: [
        {label: 'Airports reached', value: formatCount(network.degree[best]), emphasis: true},
        {
          label: 'Continent',
          value: CONTINENT_NAMES[groups.continent[best]],
          swatch: palette[groups.continent[best]]
        }
      ]
    };
  }

  function getSuperNodeTooltip(pixel: readonly [number, number]): TooltipContent | null {
    const viewport = ctx.getViewport();
    if (!viewport) return null;
    const radius = superMode === 'continent' ? 38 : 26;
    let best: SuperNode | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const node of superNodes) {
      const [px, py] = viewport.project([node.lonLat[0], node.lonLat[1]]);
      const distance = Math.hypot(px - pixel[0], py - pixel[1]);
      const reach = Math.max(8, radius * Math.sqrt(node.airports / superNodeMaximum));
      if (distance <= reach && distance < bestDistance) {
        bestDistance = distance;
        best = node;
      }
    }
    if (!best) return null;
    const palette = getContinentPalette(ctx.ground());
    return {
      title: best.name,
      subtitle: `Anchored at ${airports[(superMode === 'continent' ? members.continent : members.country)[best.group][0]].iata}, its busiest airport`,
      rows: [
        {
          label: 'Airports',
          value: formatCount(best.airports),
          swatch: palette[best.color],
          emphasis: true
        },
        {label: 'Routes inside', value: formatCount(best.inside)},
        {label: 'Routes leaving', value: formatCount(best.leaving)},
        {
          label: 'Share staying inside',
          value: formatPercent(best.inside / Math.max(1, best.inside + best.leaving))
        }
      ]
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Start up
  // ---------------------------------------------------------------------------------------------

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
      table: false,
      description:
        'Airports by number of routes, in power-of-two bins. Most airports have a few routes; a handful of hubs have hundreds.'
    });
    ctx.setReadout('airports', nodeCount);
    ctx.setReadout('routes', network.edgeCount);
    ctx.setReadout('shareContinent', chances.continent.observedShare);
    ctx.setReadout('shareCountry', countryIntraShare);
    ctx.setStatus('Compiling the order, matrix and coarsening graphs');
  }
  orderGraphHandle = buildOrderGraph();
  coarsenGraphHandle = buildCoarsenGraph();
  writeLabels();
  writeOrderInputs(false);
  activateResolution();
  publishLegendData();
  publishFurniture();
  updateAnnotations();
  publishCost();

  // ---------------------------------------------------------------------------------------------
  // The instance
  // ---------------------------------------------------------------------------------------------

  return {
    getCompiledGraphs: () =>
      [
        orderGraphHandle.compiled,
        ...(activeGraph ? [activeGraph.compiled] : []),
        coarsenGraphHandle.compiled
      ] as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      switch (id) {
        case 'resolutionPower':
          activateResolution();
          ctx.requestLayers();
          break;
        case 'order':
          writeOrderInputs(true);
          publishFurniture();
          ctx.requestLayers();
          break;
        case 'coarsenBy':
          writeLabels();
          requestedMode = ctx.options.coarsenBy;
          ctx.requestLayers();
          break;
        case 'focus':
        case 'zoom':
        case 'panX':
        case 'panY':
          writeWindow();
          ctx.requestLayers();
          break;
        case 'view':
          publishFurniture();
          updateAnnotations();
          ctx.requestLayers();
          break;
        case 'showBlocks':
          updateAnnotations();
          ctx.requestLayers();
          break;
        case 'showGroups':
          updateAnnotations();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      publishLegendData();
      summariseCounts();
      publishOrderStatistics();
      ctx.requestLayers();
    },

    onGroundChange() {
      publishLegendData();
      summariseCounts();
      publishOrderStatistics();
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (ctx.options.view === 'matrix') {
        return event.coordinate ? getMatrixTooltip(event.coordinate) : null;
      }
      if (ctx.options.showGroups) {
        const node = getSuperNodeTooltip(event.pixel);
        if (node) return node;
      }
      return ctx.options.showAirports ? getAirportTooltip(event.pixel) : null;
    },

    encode(commandEncoder) {
      if (!activeGraph) return;
      if (orderFrames > 0) {
        orderGraphHandle.compiled.encode(commandEncoder, {parameters: undefined});
        if (!animation) {
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: orderBuffer,
            destinationBuffer: displayOrderBuffer,
            size: nodeCount * 4
          });
        }
        orderFrames--;
      }
      if (matrixFrames > 0) {
        activeGraph.compiled.encode(commandEncoder, {parameters: undefined});
        matrixFrames--;
      }
      if (coarsenFrames > 0) {
        coarsenGraphHandle.compiled.encode(commandEncoder, {parameters: undefined});
        coarsenFrames--;
      }
      const settled =
        orderFrames === 0 &&
        matrixFrames === 0 &&
        coarsenFrames === 0 &&
        !animation &&
        performance.now() - lastChange > SETTLE_MILLISECONDS;
      if (statsStale && settled) {
        if (!metaReader.isPending && !activeGraph.reader.isPending) {
          statsStale = false;
          requestedMode = ctx.options.coarsenBy;
          metaReader.request(commandEncoder);
          activeGraph.reader.request(commandEncoder);
        }
      } else {
        metaReader.flush(commandEncoder);
        activeGraph.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!activeGraph) return [];
      const options = ctx.options;
      const ground = ctx.ground();
      const lngLat = COORDINATE_SYSTEM.LNGLAT;
      const layers: Layer[] = [];
      const dark = ground === 'dark';
      const sheet = getSheetColors(ground);
      const continentPalette = getContinentPalette(ground);
      const halo = inkFor(FLOW_HALO, ground);
      if (options.view === 'map') {
        const blending = dark ? ('additive' as const) : ('normal' as const);
        if (options.showRoutes && !options.showGroups) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'matrix-routes-inside',
              coordinateSystem: lngLat,
              segments: intraSegments,
              instanceCount: intraRouteCount,
              widthPixels: ROUTE_WIDTH_PIXELS,
              values: intraColors,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: getContinentPalette(ground, INTRA_ROUTE_ALPHA),
              blending
            }),
            new SpatialAnalysisSegmentLayer({
              id: 'matrix-routes-between',
              coordinateSystem: lngLat,
              segments: interSegments,
              instanceCount: interRouteCount,
              widthPixels: ROUTE_WIDTH_PIXELS,
              color: withInkAlpha(inkFor(BETWEEN_GROUPS_INK, ground), INTER_ROUTE_ALPHA),
              blending
            })
          );
        }
        if (options.showGroups && superFlowCount > 0) {
          layers.push(
            new SpatialAnalysisFlowLayer({
              id: `matrix-superedges-${superMode}`,
              coordinateOrigin: MATRIX_ORIGIN,
              flows: flowEndpoints,
              values: flowWeights,
              valueFormat: 'float32',
              ids: flowOrder,
              instanceCount: superFlowCount,
              maxValue: superedgeMaximum,
              maxWidthPixels: options.arcWidth,
              minWidthPixels: 0.8,
              curvature: 0.12,
              arrowheads: false,
              color: inkFor(FLOW_INK, ground),
              outlineColor: halo,
              outlineWidthPixels: dark ? 0 : 0.8,
              blending
            })
          );
        }
        if (options.showGroups && superNodes.length > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `matrix-supernodes-${superMode}`,
              coordinateSystem: lngLat,
              positions: nodePositions,
              instanceCount: superNodes.length,
              sizeValues: nodeSizes,
              sizeMaximumValue: superNodeMaximum,
              radiusPixels: superMode === 'continent' ? 38 : 26,
              radiusMinPixels: 2.5,
              values: nodeColors,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: continentPalette,
              outlineColor: halo,
              outlineWidthPixels: 1.5,
              opacity: 0.94
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
              ids: airportOrder,
              sizeValues: degreeBuffer,
              sizeMaximumValue: maximumDegree,
              radiusPixels: AIRPORT_MAX_RADIUS_PIXELS,
              radiusMinPixels: 1.1,
              values: airportColors,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: continentPalette,
              outlineColor: halo,
              outlineWidthPixels: 0.6,
              opacity: options.showGroups ? 0.5 : 0.95
            })
          );
        }
        return layers;
      }

      // The matrix card on its own sheet: a matrix is not geography.
      const countTable = getCountTable(ground);
      const cell = MATRIX_SIDE / matrixWindow.cells;
      const resolution = activeGraph.resolution;
      const overlayOpacity = getOverlayOpacity();
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: 'matrix-sheet',
          coordinateOrigin: MATRIX_ORIGIN,
          gridSize: [1, 1],
          bounds: [-SHEET_HALF, -SHEET_HALF, SHEET_HALF, SHEET_HALF],
          binning: 'grid',
          colormap: 'uniform',
          color: sheet.sheet
        }),
        new SpatialAnalysisRasterLayer({
          id: 'matrix-plot',
          coordinateOrigin: MATRIX_ORIGIN,
          gridSize: [1, 1],
          bounds: [-MATRIX_HALF, -MATRIX_HALF, MATRIX_HALF, MATRIX_HALF],
          binning: 'grid',
          colormap: 'uniform',
          color: sheet.plot
        })
      );
      if (
        options.showChance &&
        continentStarts.length > 0 &&
        blockLabels !== null &&
        currentOrder !== 'country'
      ) {
        // Each continent-by-continent block tinted by observed over expected, under the cells.
        const model = currentOrder === 'shuffle' ? chances.shuffle : chances.continent;
        const chanceTable = getChanceTable(ground);
        const w = matrixWindow;
        const clip = (from: number, to: number, start: number) => {
          const low = Math.max(from, start);
          const high = Math.min(to, start + w.extent);
          return high > low ? [low, high] : null;
        };
        for (let rowGroup = 0; rowGroup < continentCount; rowGroup++) {
          const rows = clip(continentStarts[rowGroup], continentStarts[rowGroup + 1], w.rowStart);
          if (!rows) continue;
          for (let columnGroup = 0; columnGroup < continentCount; columnGroup++) {
            const columns = clip(
              continentStarts[columnGroup],
              continentStarts[columnGroup + 1],
              w.colStart
            );
            if (!columns) continue;
            const ratio = model.logRatio[rowGroup * continentCount + columnGroup];
            // Two groups with no route between them have nothing to tint.
            if (!Number.isFinite(ratio)) continue;
            const color = chanceTable.colors[Math.max(0, getClassIndexOf(chanceTable, ratio))];
            layers.push(
              new SpatialAnalysisRasterLayer({
                id: `matrix-chance-${rowGroup}-${columnGroup}`,
                coordinateOrigin: MATRIX_ORIGIN,
                gridSize: [1, 1],
                bounds: [
                  positionToX(columns[0]),
                  positionToY(rows[1]),
                  positionToX(columns[1]),
                  positionToY(rows[0])
                ],
                binning: 'grid',
                colormap: 'uniform',
                color: [color[0], color[1], color[2], dark ? 120 : 150],
                opacity: overlayOpacity
              })
            );
          }
        }
      }
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: `matrix-cells-${resolution}`,
          coordinateOrigin: MATRIX_ORIGIN,
          gridSize: [resolution, resolution],
          bounds: [
            -MATRIX_HALF,
            MATRIX_HALF - resolution * cell,
            -MATRIX_HALF + resolution * cell,
            MATRIX_HALF
          ],
          binning: 'grid',
          rowOrigin: 'north',
          values: activeGraph.counts,
          valueFormat: 'uint32',
          colormap: 'greys',
          discardAtOrBelow: 0,
          ...getClassTableLayerProps(countTable)
        }),
        // Cells that belong to other blocks (outside the shown window) are covered with the sheet.
        new SpatialAnalysisRasterLayer({
          id: 'matrix-mask-right',
          coordinateOrigin: MATRIX_ORIGIN,
          gridSize: [1, 1],
          bounds: [MATRIX_HALF, -SHEET_HALF, SHEET_HALF, SHEET_HALF],
          binning: 'grid',
          colormap: 'uniform',
          color: sheet.sheet
        }),
        new SpatialAnalysisRasterLayer({
          id: 'matrix-mask-bottom',
          coordinateOrigin: MATRIX_ORIGIN,
          gridSize: [1, 1],
          bounds: [-SHEET_HALF, -SHEET_HALF, SHEET_HALF, -MATRIX_HALF],
          binning: 'grid',
          colormap: 'uniform',
          color: sheet.sheet
        })
      );
      if (options.showBlocks) {
        if (overlayCounts.countryLines > 0) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'matrix-country-lines',
              coordinateOrigin: MATRIX_ORIGIN,
              segments: countryLines,
              instanceCount: overlayCounts.countryLines,
              widthPixels: 0.5,
              color: withInkAlpha(sheet.rule, 46),
              opacity: overlayOpacity
            })
          );
        }
        if (overlayCounts.continentLines > 0) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'matrix-continent-lines',
              coordinateOrigin: MATRIX_ORIGIN,
              segments: continentLines,
              instanceCount: overlayCounts.continentLines,
              widthPixels: 1.25,
              color: withInkAlpha(sheet.rule, 140),
              opacity: overlayOpacity
            })
          );
        }
        if (overlayCounts.strips > 0) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'matrix-strips',
              coordinateOrigin: MATRIX_ORIGIN,
              segments: stripSegments,
              instanceCount: overlayCounts.strips,
              widthPixels: 8,
              cap: 'butt',
              values: stripColors,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: continentPalette,
              opacity: overlayOpacity
            })
          );
        }
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'matrix-frame',
          coordinateOrigin: MATRIX_ORIGIN,
          segments: frameLines,
          instanceCount: 4,
          widthPixels: 1,
          cap: 'square',
          color: withInkAlpha(sheet.rule, 170)
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      stopAnimation();
      metaReader.stop();
      for (const graph of matrixGraphs.values()) {
        graph.reader.stop();
        graph.resources.destroy();
      }
      coarsenGraphHandle.resources.destroy();
      orderGraphHandle.resources.destroy();
      resources.destroy();
    }
  };
}
