// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {
  createGPUEdgeBundlingParameterValues,
  GPUEdgeBundling
} from '@luma.gl/experimental/gpu-network';
import {
  getGPUGreatCircleArcsParameterValues,
  GPUGreatCircleArcs,
  GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUGraphForceLayout, GPUGraphSpatialForceLayout} from '@luma.gl/gpgpu/gpu-graph';
import {haversineMeters} from '../../cartography/anchors';
import {formatCount, formatOrdinal, liveText} from '../../cartography/live-text';
import {createNearestIndex} from '../../cartography/picking';
import {getProportionalRadius} from '../../cartography/proportional';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {
  ChartData,
  LngLat,
  MapAnnotation,
  MapHighlight,
  SceneContext,
  SceneInstance,
  ScenePointerEvent,
  TooltipContent,
  TooltipRow
} from '../scene';
import {
  copyCountToDrawRecord,
  createGraphImporter,
  createPathOutputBuffers,
  importPathOutput
} from '../geometry/b3-common';
import {
  expandAntimeridianEdges,
  readWorldNetwork,
  type ExpandedEdges,
  type FlightNetwork
} from './b11-flight-data';
import {CONTINENT_NAMES, loadAirportTable} from './b11-geography';
import {
  buildAnalysisGraph,
  buildOptimizationRun,
  createNetworkStack,
  decodeAnalysis,
  getAnalysisSources,
  GraphVectorFactory,
  type AnalysisSummary,
  type OptimizationResult,
  type OptimizationRun
} from './airline-network-graph';
import {
  ADDITIVE_ROUTE_PARAMETERS,
  CategoryArcLayer,
  CategoryPathLayer,
  NORMAL_ROUTE_PARAMETERS,
  type RouteDrawPass
} from './airline-network-layers';
import {
  BETWEEN_GROUPS_INDEX,
  DISC_OUTLINE_COLOR,
  DISC_OUTLINE_PIXELS,
  EGO_DIM_ALPHA,
  getNodePalette,
  getRoutePalette,
  NEUTRAL_NODE_INK,
  NEUTRAL_ROUTE_INK,
  OTHER_GROUP_INDEX
} from './airline-network-palette';
import {
  colorPartition,
  getBridgeCounts,
  getLogHistogram,
  getPearsonCorrelation,
  getSpearmanCorrelation,
  getTouchShare,
  sortRowsDescending,
  type Coloring
} from './airline-network-stats';

/** Option state of the airline-network scene. */
export type AirlineNetworkOptions = {
  view: 'arcs' | 'bundles' | 'morph';
  colorBy: 'none' | 'continent' | 'community';
  sizeBy: 'pagerank' | 'degree' | 'core' | 'bridges' | 'uniform';
  labels: 'hubs' | 'groups' | 'disagreement' | 'bridges' | 'none';
  ego: boolean;
  maxRadius: number;
  edgeOpacity: number;
  edgeFilter: 'all' | 'within' | 'between';
  minRecords: number;
  arcSegmentKm: number;
  pageRankDamping: number;
  pageRankIterations: number;
  propagationRounds: number;
  communityMethod: 'propagation' | 'modularity';
  resolution: number;
  rounds: '64' | '128' | '256' | '512';
  minimumGain: number;
  morph: number;
  animateMorph: boolean;
  morphSeconds: number;
  layoutRunning: boolean;
  layoutMode: 'exact' | 'spatial';
  repulsion: number;
  attraction: number;
  gravity: number;
  layoutDamping: number;
  maxVelocity: number;
  iterationsPerFrame: '1' | '2' | '4' | '8';
  theta: number;
  autoFit: boolean;
  layoutScale: number;
  bundleIterations: number;
  kernelRadius: number;
  decay: number;
  stiffness: number;
  stepScale: number;
  pointsPerEdge: '8' | '16' | '24';
  densityResolution: '128' | '256' | '512';
};

/** One entry of a group legend, as the scene file draws it. */
export type NetworkLegendEntry = {
  color: readonly [number, number, number, number];
  label: string;
  count: number;
};

/** Data the compute module hands to the scene's `legends` (through `ctx.setLegendData`). */
export type NetworkLegendData = {
  continent: NetworkLegendEntry[];
  community: NetworkLegendEntry[];
  /** The between-groups ink, for the "route between two groups" entry. */
  between: readonly [number, number, number, number];
  /** Routes between groups in the partition currently shown on the map. */
  betweenCount: number;
  /** True while the swipe compare shows continents beside communities. */
  comparing: boolean;
  /** Largest value of every size metric (the full-radius value of the size legend). */
  sizeMaxima: Record<'pagerank' | 'degree' | 'core' | 'bridges', number>;
};

/** Resolutions of the modularity sweep chart. */
export const SWEEP_RESOLUTIONS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3] as const;

const MAXIMUM_BUNDLE_ITERATIONS = 32;
const ARC_MAXIMUM_SEGMENTS = 64;
const WORLD_COPIES = [-360, 0, 360] as const;
const SPATIAL_GRID: readonly [number, number] = [14, 14];
const SPATIAL_BOUNDS: readonly [number, number, number, number] = [-6, -6, 6, 6];
const RETIRE_FRAMES = 4;
const LAYOUT_READ_FRAMES = 20;
const MAP_CENTER: readonly [number, number] = [10, 15];
const MAP_HALF_SIZE: readonly [number, number] = [165, 58];
const LNGLAT = COORDINATE_SYSTEM.LNGLAT;
const PAGE_RANK_BARS = 20;
/** Radius of an airport disc when every airport has the same size. */
const UNIFORM_RADIUS_PIXELS = 3;
/** Smallest disc radius when discs are sized by a metric. */
const MINIMUM_RADIUS_PIXELS = 1.5;
/** Hub labels per step, so the labels never outnumber the six places a frame can hold. */
const HUB_LABEL_COUNT = 4;
const BRIDGE_LABEL_COUNT = 5;
const GROUP_LABEL_COUNT = 5;
/** Neighbours ringed around a selected airport. */
const EGO_RING_LIMIT = 80;
/** Debounce of the compile-time sliders (damping, resolution): the graph builds once the reader pauses. */
const REBUILD_DEBOUNCE_MS = 260;
/** Largest category float of a selected route is `EGO_FLAG + slot`. */
const EGO_FLAG = 16;
/** Pointer distance, in CSS pixels, that still counts as pointing at an airport. */
const PICK_RADIUS_PIXELS = 24;

type Destroyable = {destroy: () => void};

/** Layout state read back every few frames: positions, auto-fit and spatial index status. */
type LayoutSnapshot = {
  positions: Float32Array;
  spatialCount: number;
  spatialOverflow: boolean;
};

/**
 * The OpenFlights airline network as a graph, using `@luma.gl/gpgpu/gpu-graph` directly:
 * degree, PageRank, core numbers, label propagation, modularity and its optimization, and a
 * progressive force layout whose positions buffer is morphed against the geographic positions.
 * Routes are drawn as `GPUGreatCircleArcs` or `GPUEdgeBundling` paths colored by community.
 *
 * Compile-time: damping, iteration and round counts, the modularity resolution, the force-layout
 * constants and the bundling grid (each is a shader constant, so changing one builds a new
 * graph). Per frame: the morph, the filters, the bundling kernel and the arc resolution.
 */
export async function createAirlineNetwork(
  ctx: SceneContext<AirlineNetworkOptions>
): Promise<SceneInstance<AirlineNetworkOptions>> {
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'airline-network');
  const airports = await loadAirportTable('openflights', ctx.signal);
  const network: FlightNetwork = readWorldNetwork(ctx.datasets.get('openflights'), airports);
  const vertexCount = network.nodeCount;
  const edgeCount = network.edgeCount;
  const continentLabels = Uint32Array.from(network.continent);

  // Persistent GPU state -------------------------------------------------------------------
  const factory = resources.track(new GraphVectorFactory(device, 'airline-network'));
  const stack = createNetworkStack(
    factory,
    vertexCount,
    network.source,
    network.target,
    network.continent
  );
  const geoPositions = resources.createBuffer('geo-positions', network.lonLat);
  const layoutPositions = factory.coordinates(
    'layout-positions',
    new Float32Array(vertexCount * 2),
    Buffer.VERTEX
  );
  const layoutVelocities = factory.coordinates(
    'layout-velocities',
    new Float32Array(vertexCount * 2)
  );
  const pinned = factory.scalar('layout-pinned', 'uint32', vertexCount);
  const resetRequest = factory.scalar('layout-reset', 'uint32', 1, Uint32Array.of(1));
  const spatialCells = SPATIAL_GRID[0] * SPATIAL_GRID[1];
  const spatialCellOffsets = factory.scalar('spatial-offsets', 'uint32', spatialCells + 1);
  const spatialVertexIds = factory.scalar('spatial-ids', 'uint32', vertexCount);
  const spatialCenters = factory.coordinates('spatial-centers', new Float32Array(spatialCells * 2));
  const spatialCount = factory.scalar('spatial-count', 'uint32', 1);
  const spatialOverflow = factory.scalar('spatial-overflow', 'uint32', 1);

  const morphPositions = resources.createBuffer('morph-positions', vertexCount * 8);
  const segments = resources.createBuffer('morph-segments', edgeCount * 16);
  const segmentWeights = resources.createBuffer('morph-weights', edgeCount * 4);
  const edgeMask = resources.createBuffer('edge-mask', new Uint32Array(edgeCount).fill(1));
  // Draw order of the morph segments: within-group routes first, between-group routes last.
  const edgeOrder = resources.createBuffer(
    'edge-order',
    Uint32Array.from({length: edgeCount}, (_, edge) => edge)
  );
  // Palette slot of every route (morph view) and of every airport, by partition.
  const edgeIndex = resources.createBuffer('edge-index', new Uint32Array(edgeCount));
  const edgeCategoryContinent = resources.createBuffer(
    'edge-category-continent',
    new Float32Array(edgeCount)
  );
  const edgeCategoryCommunity = resources.createBuffer(
    'edge-category-community',
    new Float32Array(edgeCount)
  );
  const nodeSlotsContinent = resources.createBuffer(
    'node-slots-continent',
    new Uint32Array(vertexCount)
  );
  const nodeSlotsCommunity = resources.createBuffer(
    'node-slots-community',
    new Uint32Array(vertexCount)
  );
  // Disc size metric, its draw order (largest first) and the selection highlight channel.
  const sizeValues = resources.createBuffer('size-values', new Float32Array(vertexCount));
  const drawOrder = resources.createBuffer(
    'draw-order',
    Uint32Array.from({length: vertexCount}, (_, airport) => airport)
  );
  const highlightChannel = resources.createBuffer(
    'highlight-channel',
    new Float32Array(vertexCount)
  );
  const morphParameters = resources.createParameterBuffer('morph-parameters', 'float32', 8);

  // Great-circle arcs: one compiled graph, rerun when the resolution changes ------------------
  const arcSources = new Float32Array(edgeCount * 2);
  const arcTargets = new Float32Array(edgeCount * 2);
  for (let edge = 0; edge < edgeCount; edge++) {
    const a = network.source[edge];
    const b = network.target[edge];
    arcSources.set(network.lonLat.subarray(a * 2, a * 2 + 2), edge * 2);
    arcTargets.set(network.lonLat.subarray(b * 2, b * 2 + 2), edge * 2);
  }
  const arcs = createPathOutputBuffers(
    resources,
    'arcs',
    edgeCount * (ARC_MAXIMUM_SEGMENTS + 1),
    edgeCount
  );
  const arcParameters = resources.createParameterBuffer(
    'arc-parameters',
    'float32',
    GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH
  );
  const arcGraph = new GPUCommandGraph<void>(device, {id: 'airline-arcs'});
  {
    const imp = createGraphImporter(arcGraph);
    arcGraph.add(
      new GPUGreatCircleArcs({
        id: 'airline-arcs',
        sources: imp(
          'sources',
          resources.createBuffer('arc-sources', arcSources),
          'float32x2',
          edgeCount
        ),
        targets: imp(
          'targets',
          resources.createBuffer('arc-targets', arcTargets),
          'float32x2',
          edgeCount
        ),
        maximumSegments: ARC_MAXIMUM_SEGMENTS,
        parameters: arcParameters.importToGraph(arcGraph),
        output: importPathOutput(arcGraph, arcs)
      })
    );
  }
  const compiledArcs = arcGraph.compile();
  const arcReader = new SummaryReader(
    resources,
    'arcs',
    [
      {buffer: arcs.count, size: 4},
      {buffer: arcs.overflow, size: 4}
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      ctx.setReadout(
        'arcVertices',
        `${formatCount(words[0])} vertices, ${(words[0] / edgeCount).toFixed(1)} per arc${words[1] ? ' (overflow)' : ''}`
      );
    }
  );

  // Morph kernels: mix geography with the layout positions, then build the segment rows -------
  const morphGraph = new GPUCommandGraph<void>(device, {id: 'airline-morph'});
  {
    const imp = createGraphImporter(morphGraph);
    const parameters = morphParameters.importToGraph(morphGraph);
    const geo = imp('geo', geoPositions, 'float32x2', vertexCount);
    const morphed = imp('morphed', morphPositions, 'float32x2', vertexCount);
    addKernelPass(morphGraph, {
      id: 'airline-morph-positions',
      invocationCount: vertexCount,
      bindings: [
        {name: 'geo', view: geo, type: 'f32', access: 'read'},
        {
          name: 'layoutPositions',
          view: imp('layout', factory.getBuffer(layoutPositions), 'float32x2', vertexCount),
          type: 'f32',
          access: 'read'
        },
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'morphed', view: morphed, type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let blend = parameters[parametersOffset];
  let eased = blend * blend * (3.0 - 2.0 * blend);
  let scale = parameters[parametersOffset + 1u];
  let layoutCenter = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let mapCenter = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
  let geoPoint = vec2<f32>(geo[geoOffset + 2u * index], geo[geoOffset + 2u * index + 1u]);
  let layoutPoint = vec2<f32>(layoutPositions[layoutPositionsOffset + 2u * index], layoutPositions[layoutPositionsOffset + 2u * index + 1u]);
  let mapped = mapCenter + (layoutPoint - layoutCenter) * scale;
  let result = mix(geoPoint, mapped, eased);
  morphed[morphedOffset + 2u * index] = result.x;
  morphed[morphedOffset + 2u * index + 1u] = result.y;`
    });
    addKernelPass(morphGraph, {
      id: 'airline-morph-segments',
      invocationCount: edgeCount,
      bindings: [
        {name: 'morphed', view: morphed, type: 'f32', access: 'read'},
        {name: 'geo', view: geo, type: 'f32', access: 'read'},
        {
          name: 'sources',
          view: imp(
            'edge-sources',
            factory.getBuffer(stack.graph.sourceVertices),
            'uint32',
            edgeCount
          ),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'targets',
          view: imp(
            'edge-targets',
            factory.getBuffer(stack.graph.targetVertices),
            'uint32',
            edgeCount
          ),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'mask',
          view: imp('edge-mask', edgeMask, 'uint32', edgeCount),
          type: 'u32',
          access: 'read'
        },
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {
          name: 'segments',
          view: imp('segments', segments, 'float32', edgeCount * 4),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'weights',
          view: imp('weights', segmentWeights, 'float32', edgeCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let a = sources[sourcesOffset + index];
  let b = targets[targetsOffset + index];
  segments[segmentsOffset + 4u * index] = morphed[morphedOffset + 2u * a];
  segments[segmentsOffset + 4u * index + 1u] = morphed[morphedOffset + 2u * a + 1u];
  segments[segmentsOffset + 4u * index + 2u] = morphed[morphedOffset + 2u * b];
  segments[segmentsOffset + 4u * index + 3u] = morphed[morphedOffset + 2u * b + 1u];
  // A pair that spans the antimeridian is a straight line across the whole map in geography.
  // Hide it at the start of the morph and fade it in once the endpoints have moved.
  let blend = parameters[parametersOffset];
  var weight = f32(mask[maskOffset + index]);
  if (abs(geo[geoOffset + 2u * a] - geo[geoOffset + 2u * b]) > 180.0) {
    weight = weight * smoothstep(0.35, 0.8, blend);
  }
  weights[weightsOffset + index] = weight;`
    });
  }
  const compiledMorph = morphGraph.compile();

  // Edge bundling (built the first time the bundles view is needed) ---------------------------
  const expanded: ExpandedEdges = expandAntimeridianEdges(network);
  const bundleEdgeCount = expanded.source.length;
  const bundlePositions = resources.createBuffer('bundle-positions', expanded.positions);
  const bundleSources = resources.createBuffer('bundle-sources', expanded.source);
  const bundleTargets = resources.createBuffer('bundle-targets', expanded.target);
  const bundleMask = resources.createBuffer(
    'bundle-mask',
    new Uint32Array(bundleEdgeCount).fill(1)
  );
  const bundleCategory = resources.createBuffer(
    'bundle-category',
    new Float32Array(bundleEdgeCount)
  );
  const bundleParameters = resources.createParameterBuffer('bundle-parameters', 'uint32', 5);
  type BundlingGraph = {
    resources: SpatialAnalysisResources;
    compiled: CompiledGPUCommandGraph<void>;
    paths: Buffer;
    pointsPerEdge: number;
  };
  let bundling: BundlingGraph | null = null;
  let bundlingFrames = 0;
  let bundlingSerial = 0;

  // Mutable scene state -----------------------------------------------------------------------
  const retired: {item: Destroyable; frames: number}[] = [];
  const retire = (item: Destroyable) => retired.push({item, frames: 0});
  const retireOptimizationRun = (run: OptimizationRun) => {
    run.reader.stop();
    retire(run.resources);
  };
  let destroyed = false;
  let analysis: AnalysisSummary | null = null;
  let analysisGraph: CompiledGPUCommandGraph<void> | null = null;
  let analysisSerial = 0;
  let pageRankOrder: Uint32Array = new Uint32Array(0);
  let degreeOrder: Uint32Array = new Uint32Array(0);
  let rankOfAirport: Uint32Array = new Uint32Array(0);
  let activeRun: OptimizationRun | null = null;
  let activeResult: OptimizationResult | null = null;
  let activeSettingsKey = '';
  let sweepRun: OptimizationRun | null = null;
  let sweepSerial = 0;
  const sweepResults: {
    resolution: number;
    optimized: number;
    propagation: number;
    continents: number;
  }[] = [];
  let layoutGraph: CompiledGPUCommandGraph<void> | null = null;
  let layoutSteps = 0;
  let layoutSnapshot: LayoutSnapshot | null = null;
  let layoutFit = {scale: 40, centerX: 0, centerY: 0};
  let fitInitialized = false;
  let layoutFrames = 0;
  let morphLastWritten = ctx.options.morph;
  let morphDirection = 1;
  let morphLastWriteSeconds = -Infinity;
  let morphValue = ctx.options.morph;
  // Colourings of the two partitions (continents, communities) once the analysis has arrived.
  let continentColoring: Coloring | null = null;
  let communityColoring: Coloring | null = null;
  // Disc size metric: the full-radius value, and whether the size buffer holds it.
  let sizeMaximum = 1;
  let sizeReady = false;
  let sizeColumn: Float32Array | null = null;
  const sizeMaxima: NetworkLegendData['sizeMaxima'] = {pagerank: 1, degree: 1, core: 1, bridges: 1};
  // Selected airport (click-to-ego) and the routes that touch it.
  let egoAirport = -1;
  let egoEdgeFlags: Uint8Array | null = null;
  let legendComparing = false;
  let analysisTimer: ReturnType<typeof setTimeout> | undefined;
  let optimizationTimer: ReturnType<typeof setTimeout> | undefined;
  const nearestIndex = createNearestIndex(network.lonLat);
  const layoutReader = new SummaryReader(
    resources,
    'layout-positions',
    [
      {buffer: factory.getBuffer(layoutPositions), size: vertexCount * 8},
      {buffer: factory.getBuffer(spatialCount), size: 4},
      {buffer: factory.getBuffer(spatialOverflow), size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes, vertexCount * 8);
      layoutSnapshot = {
        positions: new Float32Array(bytes.slice(0, vertexCount * 8)),
        spatialCount: words[0],
        spatialOverflow: words[1] !== 0
      };
      processLayout(layoutSnapshot);
    }
  );
  const analysisReader = new SummaryReader(
    resources,
    'analysis',
    getAnalysisSources(stack),
    bytes => {
      if (destroyed) return;
      analysis = decodeAnalysis(bytes.slice(0), vertexCount);
      processAnalysis(analysis);
    }
  );

  // ------------------------------------------------------------------------------------------
  // Compile-time rebuilds
  // ------------------------------------------------------------------------------------------

  function buildAnalysis(): void {
    const {pageRankDamping, pageRankIterations, propagationRounds} = ctx.options;
    if (analysisGraph) retire(analysisGraph);
    analysisGraph = buildAnalysisGraph(
      device,
      stack,
      {pageRankDamping, pageRankIterations, propagationRounds},
      `airline-analysis-${++analysisSerial}`
    );
    analysisReader.markStale();
    pendingAnalysisEncode = true;
  }
  let pendingAnalysisEncode = false;

  function getOptimizationKey(): string {
    const {resolution, rounds, minimumGain} = ctx.options;
    return `${resolution}|${rounds}|${minimumGain}`;
  }

  /** Builds the optimization run for the current resolution and publishes its labels. */
  function buildActiveOptimization(): void {
    if (!analysis) return;
    // The previous result stays on the map until the new graph reports, so a slider drag does not
    // flash the label-propagation colours.
    if (activeRun) retireOptimizationRun(activeRun);
    activeSettingsKey = getOptimizationKey();
    const settings = {
      resolution: ctx.options.resolution,
      rounds: Number(ctx.options.rounds),
      minimumGain: ctx.options.minimumGain
    };
    const key = activeSettingsKey;
    const run = buildOptimizationRun(device, stack, settings, `airline-opt-${key}`, result => {
      if (destroyed || activeRun !== run) return;
      activeResult = result;
      processOptimization(result);
    });
    activeRun = run;
    ctx.setReadout('optimizerStatus', 'running 0%');
  }

  /** Drops the sweep in progress and its chart. */
  function cancelSweep(): void {
    sweepSerial++;
    sweepResults.length = 0;
    if (sweepRun) retireOptimizationRun(sweepRun);
    sweepRun = null;
  }

  /** Restarts the modularity sweep, one compiled graph per resolution, built outside `encode`. */
  function startSweep(): void {
    cancelSweep();
    renderModularityChart();
    scheduleSweepStep(sweepSerial);
  }

  function scheduleSweepStep(serial: number): void {
    setTimeout(() => {
      if (destroyed || serial !== sweepSerial || !analysis) return;
      const index = sweepResults.length;
      if (index >= SWEEP_RESOLUTIONS.length) return;
      const resolution = SWEEP_RESOLUTIONS[index];
      const run: OptimizationRun = buildOptimizationRun(
        device,
        stack,
        {
          resolution,
          rounds: Number(ctx.options.rounds),
          minimumGain: ctx.options.minimumGain
        },
        `airline-sweep-${serial}-${index}`,
        result => {
          if (destroyed || serial !== sweepSerial) return;
          sweepResults.push({
            resolution,
            optimized: result.modularity,
            propagation: result.propagationModularity,
            continents: result.continentModularity
          });
          retireOptimizationRun(run);
          if (sweepRun === run) sweepRun = null;
          renderModularityChart();
          scheduleSweepStep(serial);
        }
      );
      sweepRun = run;
    }, 60);
  }

  function buildLayout(): void {
    const options = ctx.options;
    if (layoutGraph) retire(layoutGraph);
    const graph = new GPUCommandGraph<void>(device, {id: `airline-layout-${++layoutFrames}`});
    const layout = new GPUGraphForceLayout({
      id: 'airline-layout',
      topology: stack.topology,
      positions: layoutPositions,
      velocities: layoutVelocities,
      pinned,
      reset: resetRequest,
      seed: 20140601,
      iterationsPerFrame: Number(options.iterationsPerFrame),
      repulsion: options.repulsion,
      attraction: options.attraction,
      gravity: options.gravity,
      damping: options.layoutDamping,
      maxVelocity: options.maxVelocity
    });
    if (options.layoutMode === 'spatial') {
      new GPUGraphSpatialForceLayout({
        id: 'airline-layout-spatial',
        layout,
        gridSize: SPATIAL_GRID,
        bounds: SPATIAL_BOUNDS,
        theta: options.theta,
        nearCellRadius: 1,
        cellOffsets: spatialCellOffsets,
        vertexIds: spatialVertexIds,
        cellCenters: spatialCenters,
        count: spatialCount,
        overflow: spatialOverflow
      }).addToGraph(graph);
    } else {
      layout.addToGraph(graph);
    }
    layoutGraph = graph.compile();
  }

  function buildBundling(): void {
    const pointsPerEdge = Number(ctx.options.pointsPerEdge);
    const densityResolution = Number(ctx.options.densityResolution);
    if (bundling) retire(bundling.resources);
    const graphResources = new SpatialAnalysisResources(
      device,
      `airline-bundling-${++bundlingSerial}`
    );
    const paths = graphResources.createBuffer('paths', bundleEdgeCount * pointsPerEdge * 8);
    const graph = new GPUCommandGraph<void>(device, {id: `airline-bundling-${bundlingSerial}`});
    graph.add(
      new GPUEdgeBundling({
        id: 'bundling',
        positions: importGraphBuffer(
          graph,
          'positions',
          bundlePositions,
          'float32x2',
          expanded.positions.length / 2
        ),
        sourceVertices: importGraphBuffer(
          graph,
          'sources',
          bundleSources,
          'uint32',
          bundleEdgeCount
        ),
        targetVertices: importGraphBuffer(
          graph,
          'targets',
          bundleTargets,
          'uint32',
          bundleEdgeCount
        ),
        edgeMask: importGraphBuffer(graph, 'mask', bundleMask, 'uint32', bundleEdgeCount),
        geographic: true,
        pointsPerEdge,
        iterations: MAXIMUM_BUNDLE_ITERATIONS,
        densityResolution,
        parameters: bundleParameters.importToGraph(graph),
        paths: importGraphBuffer(
          graph,
          'paths',
          paths,
          'float32x2',
          bundleEdgeCount * pointsPerEdge
        )
      })
    );
    const compiled = graphResources.track(graph.compile());
    bundling = {resources: graphResources, compiled, paths, pointsPerEdge};
    writeBundleParameters();
  }

  function ensureBundling(): void {
    if (!bundling) buildBundling();
  }

  // ------------------------------------------------------------------------------------------
  // Parameter writes
  // ------------------------------------------------------------------------------------------

  function writeBundleParameters(): void {
    const {bundleIterations, kernelRadius, decay, stiffness, stepScale} = ctx.options;
    bundleParameters.write(
      createGPUEdgeBundlingParameterValues(
        {
          activeIterations: bundleIterations,
          kernelRadius,
          lambda: decay,
          smoothing: stiffness,
          stepScale
        },
        'uint32'
      )
    );
    bundlingFrames = Math.max(bundlingFrames, 2);
  }

  function writeArcParameters(): void {
    arcParameters.write(
      getGPUGreatCircleArcsParameterValues({
        maximumSegmentLength: ctx.options.arcSegmentKm * 1000,
        minimumSegments: 2
      })
    );
    pendingArcs = true;
  }
  let pendingArcs = true;

  function writeMorphParameters(): void {
    const options = ctx.options;
    const fit = options.autoFit ? layoutFit : {scale: options.layoutScale, centerX: 0, centerY: 0};
    morphParameters.write(
      Float32Array.of(
        morphValue,
        fit.scale,
        fit.centerX,
        fit.centerY,
        options.autoFit ? MAP_CENTER[0] : 0,
        options.autoFit ? MAP_CENTER[1] : 15,
        0,
        0
      )
    );
  }

  /** Advances the geography-to-layout morph when the animation is on, or follows the slider. */
  function advanceMorph(deltaSeconds: number, timeSeconds: number): void {
    const options = ctx.options;
    if (options.morph !== morphLastWritten) {
      morphValue = options.morph;
      morphLastWritten = options.morph;
    }
    if (options.animateMorph) {
      morphValue += (morphDirection * deltaSeconds) / Math.max(options.morphSeconds, 0.5);
      if (morphValue >= 1) {
        morphValue = 1;
        morphDirection = -1;
      } else if (morphValue <= 0) {
        morphValue = 0;
        morphDirection = 1;
      }
      if (timeSeconds - morphLastWriteSeconds > 1 / 20) {
        morphLastWriteSeconds = timeSeconds;
        morphLastWritten = Math.round(morphValue * 100) / 100;
        ctx.setOptions({morph: morphLastWritten});
      }
    }
  }

  // ------------------------------------------------------------------------------------------
  // Readback processing (CPU bookkeeping on small summaries)
  // ------------------------------------------------------------------------------------------

  function describeAirport(airport: number): string {
    const record = network.airports[airport];
    return `${record.iata} (${record.city})`;
  }

  function processAnalysis(result: AnalysisSummary): void {
    pageRankOrder = sortRowsDescending(result.pageRank);
    degreeOrder = sortRowsDescending(result.degree);
    rankOfAirport = new Uint32Array(vertexCount);
    pageRankOrder.forEach((airport, rank) => {
      rankOfAirport[airport] = rank;
    });
    ctx.setReadout('airports', vertexCount);
    ctx.setReadout('routes', edgeCount);
    ctx.setReadout(
      'engineStatus',
      result.adjacencyOverflow
        ? 'adjacency overflow: results are zeroed'
        : result.valid
          ? 'CSR complete, partitions valid'
          : 'a partition failed validation'
    );
    ctx.setReadout('residual', result.residual.toExponential(1));
    ctx.setReadout('degeneracy', result.degeneracy);
    ctx.setReadout(
      'coreStatus',
      result.coreConverged ? 'core numbers converged' : 'core numbers are upper bounds'
    );
    ctx.setReadout(
      'propagationStatus',
      result.propagationConverged ? 'converged' : 'stopped at its round budget'
    );
    ctx.setReadout('topHub', describeAirport(pageRankOrder[0]));
    ctx.setReadout(
      'topHubs',
      Array.from(pageRankOrder.subarray(0, 8), airport => network.airports[airport].iata).join(', ')
    );
    ctx.setReadout(
      'topByDegree',
      Array.from(degreeOrder.subarray(0, 8), airport => network.airports[airport].iata).join(', ')
    );
    ctx.setReadout('rankAgreement', getSpearmanCorrelation(result.pageRank, result.degree));
    ctx.setReadout(
      'hubShare',
      getTouchShare(new Set(degreeOrder.subarray(0, 10)), network.source, network.target)
    );
    const sortedDegrees = Array.from(result.degree).sort((a, b) => a - b);
    ctx.setReadout('medianDegree', sortedDegrees[Math.floor(vertexCount / 2)]);
    ctx.setReadout('maxDegree', sortedDegrees[vertexCount - 1]);
    ctx.setReadout(
      'leafShare',
      result.degree.reduce((count, degree) => count + (degree === 1 ? 1 : 0), 0) / vertexCount
    );
    ctx.setCost({
      records: edgeCount,
      note: `${ctx.options.pageRankIterations} PageRank rounds, ${ctx.options.propagationRounds} propagation rounds`
    });
    renderPageRankChart(result);
    renderDegreeChart(result);
    recomputeColorings();
    if (!activeRun || activeSettingsKey !== getOptimizationKey()) buildActiveOptimization();
  }

  function processOptimization(result: OptimizationResult): void {
    ctx.setReadout(
      'optimizerStatus',
      !result.valid
        ? 'failed validation'
        : result.converged
          ? 'local optimum reached'
          : 'stopped at its round budget'
    );
    recomputeColorings();
    renderModularityChart();
  }

  function renderPageRankChart(result: AnalysisSummary): void {
    const rows = Array.from(pageRankOrder.subarray(0, PAGE_RANK_BARS));
    const degreeTop = new Set(degreeOrder.subarray(0, PAGE_RANK_BARS));
    ctx.setChart('pageRankChart', {
      kind: 'bars',
      values: rows.map(airport => result.pageRank[airport] * 1000),
      labels: rows.map(airport => network.airports[airport].iata),
      highlight: rows.flatMap((airport, index) => (degreeTop.has(airport) ? [] : [index])),
      yLabel: 'PageRank x 1000',
      height: 150,
      formatY: value => value.toFixed(1),
      onBarClick: index => selectAirport(rows[index], true),
      description:
        'The 20 airports with the highest PageRank. Highlighted bars are not among the 20 airports with the most connections. Click a bar to mark the airport on the map.'
    });
  }

  function renderDegreeChart(result: AnalysisSummary): void {
    const bins = 24;
    const {counts, maximumLog} = getLogHistogram(result.degree, bins);
    ctx.setChart('degreeChart', {
      kind: 'histogram',
      values: counts.map(count => Math.log10(1 + count)),
      xDomain: [0, maximumLog],
      xLabel: 'connections per airport (log scale)',
      yLabel: 'airports (log scale)',
      height: 130,
      formatX: value => String(Math.round(10 ** value)),
      formatY: value => formatCount(10 ** value - 1),
      description:
        'Histogram of the number of distinct airports each airport connects to, with logarithmic bins on both axes.'
    });
  }

  function renderModularityChart(): void {
    const x = sweepResults.map(entry => entry.resolution);
    const chart: ChartData | null =
      sweepResults.length === 0
        ? null
        : {
            kind: 'line',
            xLabel: 'resolution gamma',
            yLabel: 'modularity Q',
            height: 160,
            xDomain: [SWEEP_RESOLUTIONS[0], SWEEP_RESOLUTIONS[SWEEP_RESOLUTIONS.length - 1]],
            series: [
              {
                label: 'optimized',
                x,
                y: sweepResults.map(entry => entry.optimized),
                color: 0,
                points: true
              },
              {label: 'propagation', x, y: sweepResults.map(entry => entry.propagation), color: 1},
              {
                label: 'continents',
                x,
                y: sweepResults.map(entry => entry.continents),
                color: 2,
                dashed: true
              }
            ],
            markers: [{x: ctx.options.resolution, label: 'now'}],
            link: {option: 'resolution', label: value => `gamma = ${value.toFixed(1)}`},
            formatY: value => value.toFixed(2),
            description:
              'Modularity of three partitions scored at each resolution: the optimized communities, label propagation, and the six continents. Click or drag to set the resolution.'
          };
    ctx.setChart('modularityChart', chart);
    ctx.setReadout(
      'sweepStatus',
      sweepResults.length >= SWEEP_RESOLUTIONS.length
        ? `${SWEEP_RESOLUTIONS.length} resolutions done`
        : `${sweepResults.length} of ${SWEEP_RESOLUTIONS.length} resolutions done`
    );
  }

  /** Community labels chosen by the method option, or label propagation until the optimizer reports. */
  function getCommunityLabels(): Uint32Array | null {
    if (!analysis) return null;
    if (ctx.options.communityMethod === 'modularity' && activeResult?.valid) {
      return activeResult.labels;
    }
    return analysis.propagation;
  }

  /**
   * The partition the map and the between-group figures are about. With no colour (the unanalysed
   * first view) the figures still need a partition, so they use the communities.
   */
  function getDisplayedColoring(): Coloring | null {
    return ctx.options.colorBy === 'continent' ? continentColoring : communityColoring;
  }

  function updateCommunityReadouts(): void {
    const labels = getCommunityLabels();
    const coloring = communityColoring;
    if (!labels || !analysis || !coloring) return;
    const useOptimized = labels !== analysis.propagation;
    ctx.setReadout('communities', coloring.groups.length);
    ctx.setReadout('purity', coloring.continentPurity);
    ctx.setReadout(
      'qCommunities',
      useOptimized && activeResult ? activeResult.modularity : analysis.propagationModularity
    );
    ctx.setReadout(
      'qContinents',
      useOptimized && activeResult ? activeResult.continentModularity : analysis.continentModularity
    );
    ctx.setReadout('withinShare', 1 - coloring.betweenCount / edgeCount);
    ctx.setReadout(
      'communityList',
      coloring.groups
        .slice(0, 6)
        .map((group, rank) => {
          const hub = network.airports[group.topAirport].iata;
          const continent = CONTINENT_NAMES[group.dominantContinent];
          return `${rank + 1}. ${formatCount(group.size)} airports, hub ${hub}, ${Math.round(group.dominantShare * 100)}% ${continent}`;
        })
        .join('\n')
    );
  }

  function processLayout(snapshot: LayoutSnapshot): void {
    const positions = snapshot.positions;
    // Robust extent: the 2nd to 98th percentile, so a few far-flung leaves do not shrink the map.
    const xs = new Float32Array(vertexCount);
    const ys = new Float32Array(vertexCount);
    for (let airport = 0; airport < vertexCount; airport++) {
      xs[airport] = positions[airport * 2];
      ys[airport] = positions[airport * 2 + 1];
    }
    xs.sort();
    ys.sort();
    const low = Math.floor(vertexCount * 0.02);
    const high = Math.min(vertexCount - 1, Math.ceil(vertexCount * 0.98));
    const halfX = Math.max((xs[high] - xs[low]) / 2, 1e-4) * 1.05;
    const halfY = Math.max((ys[high] - ys[low]) / 2, 1e-4) * 1.05;
    const target = {
      scale: Math.min(MAP_HALF_SIZE[0] / halfX, MAP_HALF_SIZE[1] / halfY),
      centerX: (xs[high] + xs[low]) / 2,
      centerY: (ys[high] + ys[low]) / 2
    };
    const mix = fitInitialized ? 0.3 : 1;
    fitInitialized = true;
    layoutFit = {
      scale: layoutFit.scale + (target.scale - layoutFit.scale) * mix,
      centerX: layoutFit.centerX + (target.centerX - layoutFit.centerX) * mix,
      centerY: layoutFit.centerY + (target.centerY - layoutFit.centerY) * mix
    };
    const geographic = new Float32Array(edgeCount);
    const layoutLength = new Float32Array(edgeCount);
    for (let edge = 0; edge < edgeCount; edge++) {
      const a = network.source[edge];
      const b = network.target[edge];
      geographic[edge] = network.distanceKm[edge];
      layoutLength[edge] = Math.hypot(
        positions[a * 2] - positions[b * 2],
        positions[a * 2 + 1] - positions[b * 2 + 1]
      );
    }
    ctx.setReadout('layoutCorrelation', getPearsonCorrelation(geographic, layoutLength));
    ctx.setReadout('layoutSteps', layoutSteps);
    ctx.setReadout(
      'layoutStatus',
      ctx.options.layoutMode === 'spatial'
        ? snapshot.spatialOverflow || snapshot.spatialCount < vertexCount
          ? `spatial index: ${formatCount(vertexCount - snapshot.spatialCount)} airports outside the grid, layout frozen`
          : 'spatial index holds every airport'
        : 'exact all-pairs repulsion'
    );
  }

  // ------------------------------------------------------------------------------------------
  // Colouring, sizes, filters and the selection
  // ------------------------------------------------------------------------------------------

  /** Whether a route passes the display filters (the graph algorithms always see every route). */
  function keepsRoute(edge: number, slot: number): boolean {
    const {edgeFilter, minRecords} = ctx.options;
    if (network.traffic[edge] < minRecords) return false;
    if (edgeFilter === 'all') return true;
    return edgeFilter === 'within' ? slot !== BETWEEN_GROUPS_INDEX : slot === BETWEEN_GROUPS_INDEX;
  }

  /** The per-route category floats of a partition: slot, `EGO_FLAG + slot` when selected, NaN when filtered out. */
  function getEdgeCategories(coloring: Coloring): Float32Array {
    const categories = new Float32Array(edgeCount);
    for (let edge = 0; edge < edgeCount; edge++) {
      const slot = coloring.edgeSlots[edge];
      categories[edge] = keepsRoute(edge, slot)
        ? egoEdgeFlags?.[edge]
          ? EGO_FLAG + slot
          : slot
        : Number.NaN;
    }
    return categories;
  }

  /** Writes node slots, route categories, the morph mask and order, and the bundle categories. */
  function writeColorBuffers(): void {
    if (!continentColoring || !communityColoring) return;
    nodeSlotsContinent.write(continentColoring.nodeSlots);
    nodeSlotsCommunity.write(communityColoring.nodeSlots);
    const continentCategories = getEdgeCategories(continentColoring);
    const communityCategories = getEdgeCategories(communityColoring);
    edgeCategoryContinent.write(continentCategories);
    edgeCategoryCommunity.write(communityCategories);

    const displayed = getDisplayedColoring();
    const categories = displayed === continentColoring ? continentCategories : communityCategories;
    const slots = displayed ?? communityColoring;
    const mask = new Uint32Array(edgeCount);
    const expandedCategories = new Float32Array(bundleEdgeCount);
    const expandedMask = new Uint32Array(bundleEdgeCount);
    let live = 0;
    for (let edge = 0; edge < edgeCount; edge++) {
      mask[edge] = Number.isNaN(categories[edge]) ? 0 : 1;
      live += mask[edge];
    }
    for (let edge = 0; edge < bundleEdgeCount; edge++) {
      const original = expanded.original[edge];
      expandedCategories[edge] = categories[original];
      expandedMask[edge] = mask[original];
    }
    // Morph draw order: within-group routes first, between-group routes over them.
    const order = new Uint32Array(edgeCount);
    let within = 0;
    let between = edgeCount;
    for (let edge = 0; edge < edgeCount; edge++) {
      if (slots.edgeSlots[edge] === BETWEEN_GROUPS_INDEX) order[--between] = edge;
      else order[within++] = edge;
    }
    edgeIndex.write(slots.edgeSlots);
    edgeMask.write(mask);
    edgeOrder.write(order);
    bundleCategory.write(expandedCategories);
    bundleMask.write(expandedMask);
    if (bundling) bundlingFrames = Math.max(bundlingFrames, 2);
    ctx.setReadout('edges', `${formatCount(live)} of ${formatCount(edgeCount)}`);
  }

  /** Recomputes both colourings from the current analysis and partition, then everything that reads them. */
  function recomputeColorings(): void {
    const labels = getCommunityLabels();
    if (!analysis || !labels) return;
    continentColoring = colorPartition(
      'continent',
      continentLabels,
      analysis.pageRank,
      network.continent,
      network.source,
      network.target
    );
    communityColoring = colorPartition(
      'community',
      labels,
      analysis.pageRank,
      network.continent,
      network.source,
      network.target
    );
    writeColorBuffers();
    updateCommunityReadouts();
    updateBridgeReadouts();
    refreshSizes();
    refreshLegend();
    refreshAnnotations();
    ctx.requestLayers();
  }

  /** Between-group share and the bridge airports of the partition the map shows. */
  function updateBridgeReadouts(): void {
    const coloring = getDisplayedColoring() ?? communityColoring;
    if (!coloring) return;
    ctx.setReadout('betweenShare', coloring.betweenCount / edgeCount);
    const bridges = getBridgeCounts(coloring.labels, network.source, network.target);
    const top = sortRowsDescending(bridges).subarray(0, BRIDGE_LABEL_COUNT);
    ctx.setReadout(
      'bridgeAirports',
      Array.from(top, airport => network.airports[airport].iata).join(', ')
    );
  }

  /** The size column of a metric, one float per airport. */
  function getSizeColumn(metric: AirlineNetworkOptions['sizeBy']): Float32Array | null {
    if (!analysis) return null;
    switch (metric) {
      case 'pagerank':
        return analysis.pageRank;
      case 'degree':
        return Float32Array.from(analysis.degree);
      case 'core':
        return Float32Array.from(analysis.coreNumber);
      case 'bridges': {
        const coloring = getDisplayedColoring() ?? communityColoring;
        return coloring ? getBridgeCounts(coloring.labels, network.source, network.target) : null;
      }
      default:
        return null;
    }
  }

  /** Writes the size metric and the largest-first draw order, and remembers every metric's maximum. */
  function refreshSizes(): void {
    if (!analysis) return;
    for (const metric of ['pagerank', 'degree', 'core', 'bridges'] as const) {
      const column = getSizeColumn(metric);
      if (!column) continue;
      let maximum = 0;
      for (let airport = 0; airport < vertexCount; airport++) {
        maximum = Math.max(maximum, column[airport]);
      }
      sizeMaxima[metric] = Math.max(maximum, 1e-9);
    }
    const metric = ctx.options.sizeBy;
    const column = getSizeColumn(metric);
    if (!column || metric === 'uniform') {
      sizeReady = false;
      sizeColumn = null;
      return;
    }
    sizeColumn = column;
    sizeValues.write(column);
    drawOrder.write(sortRowsDescending(column));
    sizeMaximum = sizeMaxima[metric];
    sizeReady = true;
  }

  /** Pixel radius of an airport's disc under the current size metric. */
  function getDiscRadius(airport: number): number {
    if (ctx.options.sizeBy === 'uniform' || !sizeReady || !sizeColumn) return UNIFORM_RADIUS_PIXELS;
    return getProportionalRadius(sizeColumn[airport], sizeMaximum, ctx.options.maxRadius, {
      minRadiusPixels: MINIMUM_RADIUS_PIXELS
    });
  }

  /** Hands the group legends and the size maxima to the scene's `legends`. */
  function refreshLegend(): void {
    if (!continentColoring || !communityColoring) return;
    const ground = ctx.ground();
    const nodePalette = getNodePalette(ground);
    const entries = (coloring: Coloring, byContinent: boolean): NetworkLegendEntry[] => {
      const list: NetworkLegendEntry[] = [];
      let other = 0;
      let otherGroups = 0;
      for (const group of coloring.groups) {
        if (group.slot === OTHER_GROUP_INDEX && !byContinent) {
          other += group.size;
          otherGroups++;
          continue;
        }
        list.push({
          color: nodePalette[group.slot],
          label: byContinent
            ? CONTINENT_NAMES[group.label]
            : `${network.airports[group.topAirport].iata} community`,
          count: group.size
        });
      }
      if (otherGroups > 0) {
        list.push({
          color: nodePalette[OTHER_GROUP_INDEX],
          label: `${formatCount(otherGroups)} smaller communities`,
          count: other
        });
      }
      return list;
    };
    const data: NetworkLegendData = {
      continent: entries(continentColoring, true),
      community: entries(communityColoring, false),
      between: nodePalette[BETWEEN_GROUPS_INDEX],
      betweenCount: (getDisplayedColoring() ?? communityColoring).betweenCount,
      comparing: legendComparing,
      sizeMaxima: {...sizeMaxima}
    };
    ctx.setLegendData('network', data);
  }

  /** Marks the selected airport and its routes (an achromatic highlight), or clears the selection. */
  function setEgo(airport: number): void {
    egoAirport = airport;
    const channel = new Float32Array(vertexCount);
    const highlights: MapHighlight[] = [];
    if (airport >= 0) {
      egoEdgeFlags = new Uint8Array(edgeCount);
      const neighbours: number[] = [];
      for (let edge = 0; edge < edgeCount; edge++) {
        const a = network.source[edge];
        const b = network.target[edge];
        if (a === airport || b === airport) {
          egoEdgeFlags[edge] = 1;
          neighbours.push(a === airport ? b : a);
        }
      }
      channel[airport] = 1;
      for (const neighbour of neighbours) channel[neighbour] = 1;
      neighbours.sort(
        (a, b) => (analysis?.pageRank[b] ?? 0) - (analysis?.pageRank[a] ?? 0) || a - b
      );
      for (const neighbour of neighbours.slice(0, EGO_RING_LIMIT)) {
        highlights.push({
          kind: 'point',
          coordinate: getAirportLngLat(neighbour),
          radiusPixels: getDiscRadius(neighbour) + 2.5
        });
      }
      highlights.push({
        kind: 'point',
        coordinate: getAirportLngLat(airport),
        radiusPixels: getDiscRadius(airport) + 4,
        pulse: true
      });
    } else {
      egoEdgeFlags = null;
    }
    highlightChannel.write(channel);
    ctx.setHighlight(highlights.length ? highlights : null);
    writeColorBuffers();
    refreshAnnotations();
    ctx.requestLayers();
  }

  /** Selects an airport (chart bar or map click); a bar click also works with the selection off. */
  function selectAirport(airport: number, fromChart = false): void {
    if (ctx.options.ego && ctx.options.view === 'arcs') {
      setEgo(airport === egoAirport && !fromChart ? -1 : airport);
      return;
    }
    ctx.setHighlight({
      kind: 'point',
      coordinate: getAirportLngLat(airport),
      radiusPixels: getDiscRadius(airport) + 4,
      pulse: true
    });
  }

  function getAirportLngLat(airport: number): LngLat {
    return [network.lonLat[airport * 2], network.lonLat[airport * 2 + 1]];
  }

  // ------------------------------------------------------------------------------------------
  // Labels and notes (placed from dataset rows, never from typed coordinates)
  // ------------------------------------------------------------------------------------------

  function getAirportLabel(
    airport: number,
    priority: number
  ): Extract<MapAnnotation, {kind: 'point'}> {
    const record = network.airports[airport];
    return {
      kind: 'point',
      id: `airport:${record.iata}`,
      coordinate: getAirportLngLat(airport),
      text: record.iata,
      detail: record.city,
      marker: 'dot',
      rank: 'subject',
      priority
    };
  }

  /** The rows with the largest value of the metric the discs show (connections for equal discs). */
  function getTopRows(count: number): number[] {
    if (!analysis) return [];
    const metric = ctx.options.sizeBy;
    const column =
      metric === 'uniform' || metric === 'bridges' ? analysis.degree : getSizeColumn(metric);
    return column ? Array.from(sortRowsDescending(column).subarray(0, count)) : [];
  }

  function refreshAnnotations(): void {
    const options = ctx.options;
    const list: MapAnnotation[] = [];
    const coloring = getDisplayedColoring() ?? communityColoring;
    if (analysis && options.view !== 'morph' && options.labels !== 'none') {
      if (options.labels === 'hubs') {
        for (const airport of getTopRows(HUB_LABEL_COUNT)) list.push(getAirportLabel(airport, 6));
      } else if (options.labels === 'bridges' && coloring) {
        const bridges = getBridgeCounts(coloring.labels, network.source, network.target);
        for (const airport of sortRowsDescending(bridges).subarray(0, BRIDGE_LABEL_COUNT)) {
          list.push(getAirportLabel(airport, 6));
        }
      } else if (coloring) {
        // The largest groups, named by their top airport.
        for (const group of coloring.groups.slice(0, GROUP_LABEL_COUNT)) {
          list.push(getAirportLabel(group.topAirport, 6));
        }
        if (options.labels === 'disagreement') {
          // The large group whose dominant continent holds the smallest share: where the
          // communities cut across the continents.
          const largest = coloring.groups.slice(0, 7);
          const widest = largest.reduce((best, group) =>
            group.dominantShare < best.dominantShare ? group : best
          );
          list.push({
            kind: 'note',
            id: 'note:disagreement',
            coordinate: getAirportLngLat(widest.topAirport),
            title: liveText('{share:percent} {continent}', {
              share: widest.dominantShare,
              continent: CONTINENT_NAMES[widest.dominantContinent]
            }),
            text: `of the ${network.airports[widest.topAirport].iata} community; the rest sit on other continents`,
            priority: 8
          });
        }
      }
    }
    if (egoAirport >= 0 && options.view === 'arcs') {
      list.push({...getAirportLabel(egoAirport, 9), marker: 'ring', tone: 'signal'});
    }
    ctx.setAnnotations('airline-network', list.length ? list : null);
  }

  // ------------------------------------------------------------------------------------------
  // Picking and tooltip
  // ------------------------------------------------------------------------------------------

  /** CPU position of an airport on screen: geography, or the morph between it and the layout. */
  function getAirportPosition(airport: number): [number, number] {
    const geoX = network.lonLat[airport * 2];
    const geoY = network.lonLat[airport * 2 + 1];
    if (ctx.options.view !== 'morph' || !layoutSnapshot) return [geoX, geoY];
    const options = ctx.options;
    const fit = options.autoFit ? layoutFit : {scale: options.layoutScale, centerX: 0, centerY: 0};
    const centerLon = options.autoFit ? MAP_CENTER[0] : 0;
    const centerLat = options.autoFit ? MAP_CENTER[1] : 15;
    const eased = morphValue * morphValue * (3 - 2 * morphValue);
    const x = centerLon + (layoutSnapshot.positions[airport * 2] - fit.centerX) * fit.scale;
    const y = centerLat + (layoutSnapshot.positions[airport * 2 + 1] - fit.centerY) * fit.scale;
    return [geoX + (x - geoX) * eased, geoY + (y - geoY) * eased];
  }

  /**
   * The airport under the pointer, or -1. On the map the CPU nearest index finds the candidates
   * within a pointer's reach; among them the disc whose edge is closest to the pointer wins. In
   * the morph view the airports are not at their dataset positions, so every airport is tried.
   */
  function pickAirport(event: ScenePointerEvent): number {
    const viewport = ctx.getViewport();
    if (!viewport || !analysis) return -1;
    const pixel = event.pixel;
    let candidates: number[];
    if (ctx.options.view === 'morph') {
      candidates = Array.from({length: vertexCount}, (_, airport) => airport);
    } else {
      if (!event.coordinate) return -1;
      const longitude = ((((event.coordinate[0] + 180) % 360) + 360) % 360) - 180;
      const reach = viewport.unproject([pixel[0] + PICK_RADIUS_PIXELS, pixel[1]]);
      const radius = haversineMeters(
        [event.coordinate[0], event.coordinate[1]],
        [reach[0], reach[1]]
      );
      candidates = nearestIndex.within([longitude, event.coordinate[1]], radius);
    }
    let best = -1;
    let bestScore = Number.POSITIVE_INFINITY;
    for (const airport of candidates) {
      const [longitude, latitude] = getAirportPosition(airport);
      // The pointer may be over another copy of the world: project the nearest copy.
      const copy = event.coordinate ? Math.round((event.coordinate[0] - longitude) / 360) * 360 : 0;
      const [x, y] = viewport.project([longitude + copy, latitude]);
      const distance = Math.hypot(x - pixel[0], y - pixel[1]);
      const radius = getDiscRadius(airport);
      if (distance > Math.max(radius + 3, 7)) continue;
      const score = distance - radius;
      if (score < bestScore) {
        bestScore = score;
        best = airport;
      }
    }
    return best;
  }

  function getTooltip(event: ScenePointerEvent): TooltipContent | null {
    const airport = pickAirport(event);
    if (airport < 0 || !analysis) return null;
    const record = network.airports[airport];
    const labels = getCommunityLabels();
    const community = communityColoring?.groups.find(group => group.label === labels?.[airport]);
    const palette = getNodePalette(ctx.ground());
    const slot = (ctx.options.colorBy === 'continent' ? continentColoring : communityColoring)
      ?.nodeSlots[airport];
    const metric = ctx.options.sizeBy;
    const swatch = slot === undefined || ctx.options.colorBy === 'none' ? undefined : palette[slot];
    const connections: TooltipRow = {
      label: 'Connections',
      value: formatCount(analysis.degree[airport]),
      unit: 'airports'
    };
    const pageRank: TooltipRow = {
      label: 'PageRank rank',
      value: formatOrdinal(rankOfAirport[airport] + 1),
      unit: `of ${formatCount(vertexCount)}`
    };
    const core: TooltipRow = {
      label: 'Core number',
      value: formatCount(analysis.coreNumber[airport])
    };
    // The mapped value comes first, carrying the swatch of the group colour.
    const mapped = metric === 'pagerank' ? pageRank : metric === 'core' ? core : connections;
    const rows: TooltipRow[] = [
      {...mapped, swatch, emphasis: true},
      ...[connections, pageRank, core].filter(row => row !== mapped),
      {label: 'Continent', value: CONTINENT_NAMES[network.continent[airport]]}
    ];
    if (community) {
      rows.push({
        label: 'Community',
        value: `${network.airports[community.topAirport].iata} community`,
        unit: `${formatCount(community.size)} airports`,
        swatch: palette[community.slot]
      });
    }
    return {
      title: record.iata,
      subtitle: `${record.name}, ${record.city}, ${record.country}`,
      rows,
      anchor: getAirportLngLat(airport),
      highlight: {
        kind: 'point',
        coordinate: getAirportLngLat(airport),
        radiusPixels: getDiscRadius(airport) + 3
      }
    };
  }

  // ------------------------------------------------------------------------------------------
  // Start up
  // ------------------------------------------------------------------------------------------

  writeArcParameters();
  writeBundleParameters();
  buildAnalysis();
  buildLayout();
  if (ctx.options.view === 'bundles') ensureBundling();
  ctx.setReadout('airports', vertexCount);
  ctx.setReadout('routes', edgeCount);
  ctx.setReadout('resolutionNow', ctx.options.resolution.toFixed(1));
  ctx.setReadout('edges', `${formatCount(edgeCount)} of ${formatCount(edgeCount)}`);
  ctx.setFurniture({
    title: {
      sample: `${formatCount(vertexCount)} airports, ${formatCount(edgeCount)} route pairs`
    }
  });

  /** Debounced rebuild of the analysis graph: a slider drag compiles once, when the reader pauses. */
  function scheduleAnalysisRebuild(resetOptimization: boolean): void {
    clearTimeout(analysisTimer);
    analysisTimer = setTimeout(() => {
      if (destroyed) return;
      buildAnalysis();
      if (resetOptimization) {
        if (activeRun) retireOptimizationRun(activeRun);
        activeRun = null;
        activeResult = null;
        cancelSweep();
      }
      ctx.requestLayers();
    }, REBUILD_DEBOUNCE_MS);
  }

  /** Debounced rebuild of the optimization run (resolution, rounds, minimum gain). */
  function scheduleOptimizationRebuild(clearSweep: boolean): void {
    clearTimeout(optimizationTimer);
    optimizationTimer = setTimeout(() => {
      if (destroyed) return;
      buildActiveOptimization();
      if (clearSweep) {
        cancelSweep();
        renderModularityChart();
      }
      ctx.requestLayers();
    }, REBUILD_DEBOUNCE_MS);
  }

  return {
    getCompiledGraphs: () =>
      [
        analysisGraph,
        activeRun?.compiled,
        layoutGraph,
        compiledMorph,
        compiledArcs,
        bundling?.compiled
      ].filter(Boolean) as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      switch (id) {
        case 'pageRankDamping':
        case 'pageRankIterations':
          scheduleAnalysisRebuild(false);
          break;
        case 'propagationRounds':
          scheduleAnalysisRebuild(true);
          break;
        case 'resolution':
          // The chart marker follows the slider at once; the graph builds when the drag settles.
          ctx.setReadout('resolutionNow', ctx.options.resolution.toFixed(1));
          renderModularityChart();
          scheduleOptimizationRebuild(false);
          break;
        case 'rounds':
        case 'minimumGain':
          scheduleOptimizationRebuild(true);
          break;
        case 'communityMethod':
          recomputeColorings();
          break;
        case 'colorBy':
        case 'edgeFilter':
        case 'minRecords':
          writeColorBuffers();
          updateBridgeReadouts();
          refreshSizes();
          refreshAnnotations();
          break;
        case 'sizeBy':
          refreshSizes();
          refreshAnnotations();
          break;
        case 'labels':
          refreshAnnotations();
          break;
        case 'ego':
          if (!ctx.options.ego && egoAirport >= 0) setEgo(-1);
          break;
        case 'arcSegmentKm':
          writeArcParameters();
          break;
        case 'view':
          if (ctx.options.view === 'bundles') ensureBundling();
          if (ctx.options.view === 'arcs') pendingArcs = true;
          if (ctx.options.view !== 'arcs' && egoAirport >= 0) setEgo(-1);
          refreshAnnotations();
          break;
        case 'repulsion':
        case 'attraction':
        case 'gravity':
        case 'layoutDamping':
        case 'maxVelocity':
        case 'iterationsPerFrame':
        case 'layoutMode':
        case 'theta':
          buildLayout();
          break;
        case 'bundleIterations':
        case 'kernelRadius':
        case 'decay':
        case 'stiffness':
        case 'stepScale':
          writeBundleParameters();
          break;
        case 'pointsPerEdge':
        case 'densityResolution':
          buildBundling();
          bundlingFrames = 3;
          break;
        default:
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'resetLayout') {
        factory.getBuffer(resetRequest).write(Uint32Array.of(1));
        layoutSteps = 0;
        layoutFit = {scale: 40, centerX: 0, centerY: 0};
        fitInitialized = false;
      } else if (id === 'rerunSweep') {
        startSweep();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    // The ground colours the palettes: a ground flip rebuilds the legends and the layers.
    onGroundChange() {
      refreshLegend();
      ctx.requestLayers();
    },

    getTooltip,

    onClick(event) {
      if (!ctx.options.ego || ctx.options.view !== 'arcs' || !analysis) return false;
      const airport = pickAirport(event);
      if (airport < 0) {
        if (egoAirport < 0) return false;
        setEgo(-1);
        return true;
      }
      selectAirport(airport);
      return true;
    },

    encode(commandEncoder, frame) {
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].item.destroy();
          retired.splice(index, 1);
        }
      }
      const options = ctx.options;
      advanceMorph(frame.deltaSeconds, frame.timeSeconds);

      if (pendingAnalysisEncode && analysisGraph) {
        analysisGraph.encode(commandEncoder, {parameters: undefined});
        analysisReader.request(commandEncoder);
        pendingAnalysisEncode = false;
      } else {
        analysisReader.flush(commandEncoder);
      }

      const activeExecutionRunning = Boolean(
        activeRun && analysis && !activeRun.execution.completed
      );
      if (activeRun && activeExecutionRunning) {
        const step = activeRun.execution.encodeNext(commandEncoder, {parameters: undefined});
        ctx.setReadout('optimizerStatus', `running ${Math.round(step.progress * 100)}%`);
        if (step.completed) activeRun.reader.request(commandEncoder);
      } else {
        activeRun?.reader.flush(commandEncoder);
      }
      // Never place two expensive candidate rounds in the same browser submission. A manual
      // resolution sweep yields to the interactive active-resolution result.
      if (sweepRun && !activeExecutionRunning && !sweepRun.execution.completed) {
        const step = sweepRun.execution.encodeNext(commandEncoder, {parameters: undefined});
        ctx.setReadout(
          'sweepStatus',
          `${sweepResults.length} of ${SWEEP_RESOLUTIONS.length} resolutions done · ${Math.round(step.progress * 100)}% current`
        );
        if (step.completed) sweepRun.reader.request(commandEncoder);
      } else {
        sweepRun?.reader.flush(commandEncoder);
      }

      if (pendingArcs) {
        compiledArcs.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, arcs.count, arcs.drawCommands);
        arcReader.request(commandEncoder);
        pendingArcs = false;
      } else {
        arcReader.flush(commandEncoder);
      }

      if (bundling && bundlingFrames > 0 && options.view === 'bundles') {
        bundling.compiled.encode(commandEncoder, {parameters: undefined});
        bundlingFrames--;
      }

      if (layoutGraph && options.layoutRunning && options.view === 'morph') {
        layoutGraph.encode(commandEncoder, {parameters: undefined});
        layoutSteps += Number(options.iterationsPerFrame);
        if (frame.frameIndex % LAYOUT_READ_FRAMES === 0 || layoutSteps <= 8) {
          layoutReader.request(commandEncoder);
        } else {
          layoutReader.flush(commandEncoder);
        }
      } else {
        layoutReader.flush(commandEncoder);
      }

      if (options.view === 'morph') {
        writeMorphParameters();
        compiledMorph.encode(commandEncoder, {parameters: undefined});
        if (!options.layoutRunning && !layoutSnapshot) layoutReader.request(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const layers: Layer[] = [];
      const ground = ctx.ground();
      const parameters = ground === 'dark' ? ADDITIVE_ROUTE_PARAMETERS : NORMAL_ROUTE_PARAMETERS;
      const nodePalette = getNodePalette(ground);
      const routePalette = getRoutePalette(ground);
      const neutralNode = NEUTRAL_NODE_INK[ground];
      const neutralRoute = NEUTRAL_ROUTE_INK[ground];
      const coloringReady = Boolean(continentColoring && communityColoring);
      // Swipe compare: continents on side a, communities on side b, whatever "colour by" says.
      const comparing = options.view === 'arcs' && ctx.getCompare() !== null && coloringReady;
      if (comparing !== legendComparing) {
        legendComparing = comparing;
        queueMicrotask(() => {
          if (!destroyed) refreshLegend();
        });
      }
      const neutral = !comparing && (options.colorBy === 'none' || !coloringReady);
      const sides: {side: 'a' | 'b' | undefined; partition: 'continent' | 'community'}[] = comparing
        ? [
            {side: 'a', partition: 'continent'},
            {side: 'b', partition: 'community'}
          ]
        : [
            {
              side: undefined,
              partition: options.colorBy === 'continent' ? 'continent' : 'community'
            }
          ];
      const egoActive = egoAirport >= 0 && options.view === 'arcs';
      // Two passes are only useful when both coloured classes are visible: the second pass puts
      // between-group routes over the within-group routes. Neutral ink and single-class filters
      // would otherwise submit every arc or bundle twice for exactly the same image.
      const passes: RouteDrawPass[] =
        neutral || options.edgeFilter !== 'all' ? ['all'] : ['within', 'between'];

      for (const {side, partition} of sides) {
        const key = `${side ?? 'x'}-${partition}`;
        const categories =
          partition === 'continent' ? edgeCategoryContinent : edgeCategoryCommunity;
        const routeStyle = {
          palette: routePalette,
          neutral: neutralRoute,
          colorMode: neutral ? ('neutral' as const) : ('category' as const),
          opacity: options.edgeOpacity,
          compareSide: side,
          egoActive,
          dimAlpha: EGO_DIM_ALPHA,
          parameters
        };
        if (options.view === 'arcs') {
          for (const drawPass of passes) {
            for (const offset of WORLD_COPIES) {
              layers.push(
                new CategoryArcLayer({
                  id: `network-arcs-${key}-${drawPass}-${offset}`,
                  ...routeStyle,
                  drawPass,
                  positionOffset: [offset, 0],
                  positions: arcs.positions,
                  pathOffsets: arcs.offsets,
                  pathOffsetCount: arcs.offsetCount,
                  vertexCount: arcs.count,
                  drawCommands: arcs.drawCommands,
                  categories,
                  widthPixels: 0.75
                })
              );
            }
          }
        } else if (options.view === 'bundles' && bundling) {
          for (const drawPass of passes) {
            layers.push(
              new CategoryPathLayer({
                id: `network-bundles-${bundling.pointsPerEdge}-${bundlingSerial}-${drawPass}`,
                ...routeStyle,
                drawPass,
                paths: bundling.paths,
                pointsPerPath: bundling.pointsPerEdge,
                pathCount: bundleEdgeCount,
                categories: bundleCategory,
                opacity: Math.min(1, options.edgeOpacity * 1.4)
              })
            );
          }
        } else if (options.view === 'morph') {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'network-morph-edges',
              coordinateSystem: LNGLAT,
              segments,
              weights: segmentWeights,
              ids: edgeOrder,
              instanceCount: edgeCount,
              widthPixels: 0.8,
              values: edgeIndex,
              valueFormat: 'uint32',
              colormap: neutral ? 'uniform' : 'category',
              color: neutralRoute,
              palette: routePalette,
              blending: ground === 'dark' ? 'additive' : 'normal',
              opacity: options.edgeOpacity
            })
          );
        }
      }

      // Airports last: outlined discs, largest first so small airports stay visible.
      const sizeActive = options.sizeBy !== 'uniform' && sizeReady;
      for (const {side, partition} of sides) {
        const slots = partition === 'continent' ? nodeSlotsContinent : nodeSlotsCommunity;
        const copies = options.view === 'morph' ? [0] : WORLD_COPIES;
        for (const offset of copies) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `network-airports-${side ?? 'x'}-${offset}`,
              coordinateSystem: LNGLAT,
              positions: options.view === 'morph' ? morphPositions : geoPositions,
              instanceCount: vertexCount,
              ids: sizeActive ? drawOrder : null,
              sizeValues: sizeActive ? sizeValues : null,
              sizeMaximumValue: sizeMaximum,
              radiusPixels: sizeActive ? options.maxRadius : UNIFORM_RADIUS_PIXELS,
              radiusMinPixels: sizeActive ? MINIMUM_RADIUS_PIXELS : undefined,
              shape: 'circle',
              colormap: neutral ? 'uniform' : 'category',
              color: neutralNode,
              ...(neutral ? {} : {values: slots, valueFormat: 'uint32' as const}),
              palette: nodePalette,
              outlineColor: DISC_OUTLINE_COLOR,
              outlineWidthPixels: DISC_OUTLINE_PIXELS,
              ...(egoActive
                ? {
                    instanceChannels: highlightChannel,
                    channelStride: 1,
                    channels: {highlight: 0},
                    dimOpacity: 0.2
                  }
                : {}),
              positionOffset: [offset, 0],
              compareSide: side,
              opacity: 0.95
            })
          );
        }
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      clearTimeout(analysisTimer);
      clearTimeout(optimizationTimer);
      analysisReader.stop();
      layoutReader.stop();
      arcReader.stop();
      activeRun?.reader.stop();
      sweepRun?.reader.stop();
      for (const entry of retired) entry.item.destroy();
      retired.length = 0;
      sweepRun?.resources.destroy();
      activeRun?.resources.destroy();
      bundling?.resources.destroy();
      analysisGraph?.destroy();
      layoutGraph?.destroy();
      compiledMorph.destroy();
      compiledArcs.destroy();
      resources.destroy();
    }
  };
}
