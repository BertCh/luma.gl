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
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, ChartData} from '../scene';
import {
  copyCountToDrawRecord,
  createGraphImporter,
  createPathOutputBuffers,
  importPathOutput
} from '../geometry/b3-common';
import {PathOutputLayer} from '../geometry/b3-layers';
import {
  expandAntimeridianEdges,
  readWorldNetwork,
  type ExpandedEdges,
  type FlightNetwork
} from './b11-flight-data';
import {CONTINENT_NAMES, loadAirportTable} from './b11-geography';
import {SizedDiscLayer} from './b11-flow-layers';
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
import {CategoryPathLayer} from './airline-network-layers';
import {BETWEEN_GROUPS_INDEX, NETWORK_PALETTE, OTHER_GROUP_INDEX} from './airline-network-palette';
import {
  analyzeGroups,
  getEdgeIndex,
  getLogHistogram,
  getPearsonCorrelation,
  getSpearmanCorrelation,
  sortRowsDescending,
  type GroupAnalysis
} from './airline-network-stats';

/** Option state of the airline-network scene. */
export type AirlineNetworkOptions = {
  view: 'arcs' | 'bundles' | 'morph';
  colorBy: 'community' | 'continent';
  sizeBy: 'pagerank' | 'degree' | 'core' | 'uniform';
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
  const edgeIndex = resources.createBuffer('edge-index', new Uint32Array(edgeCount));
  const edgeCategory = resources.createBuffer('edge-category', new Float32Array(edgeCount));
  const nodeColor = resources.createBuffer('node-color', new Uint32Array(vertexCount));
  const ones = resources.createBuffer('ones', new Float32Array(vertexCount).fill(1));
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
          name: 'layout',
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
  let layoutPoint = vec2<f32>(layout[layoutOffset + 2u * index], layout[layoutOffset + 2u * index + 1u]);
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
    if (activeRun) retire(activeRun.resources);
    activeResult = null;
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
      if (sweepResults.length === 0 && !sweepRun) startSweep();
    });
    activeRun = run;
  }

  /** Drops the sweep in progress and its chart. */
  function cancelSweep(): void {
    sweepSerial++;
    sweepResults.length = 0;
    if (sweepRun) retire(sweepRun.resources);
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
          retire(run.resources);
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

  function processAnalysis(result: AnalysisSummary): void {
    pageRankOrder = sortRowsDescending(result.pageRank);
    degreeOrder = sortRowsDescending(result.degree);
    rankOfAirport = new Uint32Array(vertexCount);
    pageRankOrder.forEach((airport, rank) => {
      rankOfAirport[airport] = rank;
    });
    ctx.setReadout('airports', vertexCount);
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
    const top = Array.from(pageRankOrder.subarray(0, 8), airport => network.airports[airport].iata);
    ctx.setReadout('topHubs', top.join(', '));
    ctx.setReadout(
      'topByDegree',
      Array.from(degreeOrder.subarray(0, 8), airport => network.airports[airport].iata).join(', ')
    );
    ctx.setReadout('rankAgreement', getSpearmanCorrelation(result.pageRank, result.degree));
    const sortedDegrees = Array.from(result.degree).sort((a, b) => a - b);
    ctx.setReadout('medianDegree', sortedDegrees[Math.floor(vertexCount / 2)]);
    ctx.setReadout('maxDegree', sortedDegrees[vertexCount - 1]);
    ctx.setReadout(
      'leafShare',
      result.degree.reduce((count, degree) => count + (degree === 1 ? 1 : 0), 0) / vertexCount
    );
    renderPageRankChart(result);
    renderDegreeChart(result);
    applyColoring();
    updateCommunityReadouts();
    ctx.requestLayers();
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
    applyColoring();
    updateCommunityReadouts();
    renderModularityChart();
    ctx.requestLayers();
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
      description:
        'The 20 airports with the highest PageRank. Highlighted bars are not among the 20 airports with the most connections.'
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
            height: 150,
            series: [
              {label: 'optimized', x, y: sweepResults.map(entry => entry.optimized), color: 0},
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
            formatY: value => value.toFixed(2),
            description:
              'Modularity of three partitions scored at each resolution: the optimized communities, label propagation, and the six continents.'
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

  function getDisplayLabels(): Uint32Array | null {
    return ctx.options.colorBy === 'continent' ? continentLabels : getCommunityLabels();
  }

  function updateCommunityReadouts(): void {
    const labels = getCommunityLabels();
    if (!labels || !analysis) return;
    const groups = analyzeGroups(labels, analysis.pageRank, network.continent);
    const useOptimized = labels !== analysis.propagation;
    ctx.setReadout('communities', groups.groups.length);
    ctx.setReadout('purity', groups.continentPurity);
    ctx.setReadout(
      'qCommunities',
      useOptimized && activeResult ? activeResult.modularity : analysis.propagationModularity
    );
    ctx.setReadout(
      'qContinents',
      useOptimized && activeResult ? activeResult.continentModularity : analysis.continentModularity
    );
    let within = 0;
    for (let edge = 0; edge < edgeCount; edge++) {
      if (labels[network.source[edge]] === labels[network.target[edge]]) within++;
    }
    ctx.setReadout('withinShare', within / edgeCount);
    ctx.setReadout(
      'communityList',
      groups.groups
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
  // Coloring and filters
  // ------------------------------------------------------------------------------------------

  /** Writes node colors, per-route palette indexes and the filter masks for the current options. */
  function applyColoring(): void {
    const labels = getDisplayLabels();
    if (!labels || !analysis) return;
    const groups = analyzeGroups(labels, analysis.pageRank, network.continent);
    nodeColor.write(groups.colorIndex);
    const {edgeFilter, minRecords} = ctx.options;
    const indexes = new Uint32Array(edgeCount);
    const categories = new Float32Array(edgeCount);
    const mask = new Uint32Array(edgeCount);
    const expandedCategories = new Float32Array(bundleEdgeCount);
    const expandedMask = new Uint32Array(bundleEdgeCount);
    let live = 0;
    for (let edge = 0; edge < edgeCount; edge++) {
      const index = getEdgeIndex(
        labels,
        groups.colorIndex,
        network.source[edge],
        network.target[edge]
      );
      indexes[edge] = index;
      const keep =
        network.traffic[edge] >= minRecords &&
        (edgeFilter === 'all' ||
          (edgeFilter === 'within'
            ? index !== BETWEEN_GROUPS_INDEX
            : index === BETWEEN_GROUPS_INDEX));
      categories[edge] = keep ? index : Number.NaN;
      mask[edge] = keep ? 1 : 0;
      live += mask[edge];
    }
    for (let edge = 0; edge < bundleEdgeCount; edge++) {
      const original = expanded.original[edge];
      expandedCategories[edge] = categories[original];
      expandedMask[edge] = mask[original];
    }
    edgeIndex.write(indexes);
    edgeCategory.write(categories);
    edgeMask.write(mask);
    bundleCategory.write(expandedCategories);
    bundleMask.write(expandedMask);
    if (bundling) bundlingFrames = Math.max(bundlingFrames, 2);
    ctx.setReadout('edges', `${formatCount(live)} of ${formatCount(edgeCount)}`);
    ctx.setLegendData('groups', {
      entries: describeGroups(groups, labels === continentLabels),
      showBetween: true
    });
  }

  function describeGroups(
    groups: GroupAnalysis,
    byContinent: boolean
  ): {color: readonly [number, number, number, number]; label: string}[] {
    const entries: {color: readonly [number, number, number, number]; label: string}[] = [];
    groups.groups.slice(0, OTHER_GROUP_INDEX).forEach((group, rank) => {
      const label = byContinent
        ? `${CONTINENT_NAMES[group.label]} (${formatCount(group.size)})`
        : `${network.airports[group.topAirport].iata} community (${formatCount(group.size)})`;
      entries.push({color: NETWORK_PALETTE[rank], label});
    });
    const rest = groups.groups.length - OTHER_GROUP_INDEX;
    if (rest > 0) {
      entries.push({
        color: NETWORK_PALETTE[OTHER_GROUP_INDEX],
        label: `${formatCount(rest)} smaller ${byContinent ? 'groups' : 'communities'}`
      });
    }
    return entries;
  }

  // ------------------------------------------------------------------------------------------
  // Tooltip
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

  function findAirportNear(pixel: readonly [number, number]): string | null {
    const viewport = ctx.getViewport();
    if (!viewport || !analysis) return null;
    let best = -1;
    let bestScore = Number.POSITIVE_INFINITY;
    for (let airport = 0; airport < vertexCount; airport++) {
      const [x, y] = viewport.project(getAirportPosition(airport));
      const distance = Math.hypot(x - pixel[0], y - pixel[1]);
      // Prefer the more important airport when several discs overlap the pointer.
      const score = distance - Math.sqrt(analysis.pageRank[airport] * 4000);
      if (distance < 14 && score < bestScore) {
        bestScore = score;
        best = airport;
      }
    }
    if (best < 0) return null;
    const record = network.airports[best];
    const labels = getCommunityLabels();
    const community = labels
      ? `community of ${labels.reduce((count, label) => count + (label === labels[best] ? 1 : 0), 0)} airports`
      : '';
    return [
      `${record.iata}: ${record.name}`,
      `${record.city}, ${record.country}`,
      `${analysis.degree[best]} connections, PageRank rank ${rankOfAirport[best] + 1}`,
      `core number ${analysis.coreNumber[best]}${community ? `, ${community}` : ''}`
    ].join('\n');
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
  ctx.setReadout('edges', `${formatCount(edgeCount)} of ${formatCount(edgeCount)}`);

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
        case 'propagationRounds':
          buildAnalysis();
          if (activeRun) retire(activeRun.resources);
          activeRun = null;
          activeResult = null;
          cancelSweep();
          break;
        case 'resolution':
        case 'rounds':
        case 'minimumGain':
          buildActiveOptimization();
          if (id !== 'resolution') startSweep();
          else renderModularityChart();
          break;
        case 'communityMethod':
        case 'colorBy':
        case 'edgeFilter':
        case 'minRecords':
          applyColoring();
          updateCommunityReadouts();
          break;
        case 'arcSegmentKm':
          writeArcParameters();
          break;
        case 'view':
          if (ctx.options.view === 'bundles') ensureBundling();
          if (ctx.options.view === 'arcs') pendingArcs = true;
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

    getTooltip(event) {
      return findAirportNear(event.pixel);
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

      if (activeRun?.pending && analysis) {
        activeRun.compiled.encode(commandEncoder, {parameters: undefined});
        activeRun.reader.request(commandEncoder);
        activeRun.pending = false;
      } else {
        activeRun?.reader.flush(commandEncoder);
      }
      if (sweepRun?.pending) {
        sweepRun.compiled.encode(commandEncoder, {parameters: undefined});
        sweepRun.reader.request(commandEncoder);
        sweepRun.pending = false;
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

      if (layoutGraph && options.layoutRunning) {
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
      const dark = ctx.theme() === 'dark';
      const palette = NETWORK_PALETTE;
      if (options.view === 'arcs') {
        for (const offset of WORLD_COPIES) {
          layers.push(
            new PathOutputLayer({
              id: `network-arcs-${offset}`,
              coordinateSystem: LNGLAT,
              positionOffset: [offset, 0],
              positions: arcs.positions,
              pathOffsets: arcs.offsets,
              pathOffsetCount: arcs.offsetCount,
              vertexCount: arcs.count,
              drawCommands: arcs.drawCommands,
              values: edgeCategory,
              colorSource: 'path-value',
              valueMapping: 'category',
              color: [255, 255, 255, 255],
              widthPixels: 0.9,
              opacity: options.edgeOpacity
            })
          );
        }
      } else if (options.view === 'bundles' && bundling) {
        layers.push(
          new CategoryPathLayer({
            id: `network-bundles-${bundling.pointsPerEdge}-${bundlingSerial}`,
            paths: bundling.paths,
            pointsPerPath: bundling.pointsPerEdge,
            pathCount: bundleEdgeCount,
            categories: bundleCategory,
            opacity: Math.min(1, options.edgeOpacity * 1.4)
          })
        );
      } else if (options.view === 'morph') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'network-morph-edges',
            coordinateSystem: LNGLAT,
            segments,
            weights: segmentWeights,
            instanceCount: edgeCount,
            widthPixels: 0.8,
            values: edgeIndex,
            valueFormat: 'uint32',
            colormap: 'category',
            palette,
            opacity: options.edgeOpacity * 0.8
          })
        );
      }
      const sizeBuffer =
        options.sizeBy === 'pagerank'
          ? factory.getBuffer(stack.pageRank)
          : options.sizeBy === 'degree'
            ? factory.getBuffer(stack.degree)
            : options.sizeBy === 'core'
              ? factory.getBuffer(stack.coreNumber)
              : ones;
      const maximum = !analysis
        ? 1
        : options.sizeBy === 'pagerank'
          ? analysis.pageRank[pageRankOrder[0]]
          : options.sizeBy === 'degree'
            ? analysis.degree[degreeOrder[0]]
            : options.sizeBy === 'core'
              ? Math.max(1, analysis.degeneracy)
              : 1;
      layers.push(
        new SizedDiscLayer({
          id: `network-airports-${options.sizeBy}`,
          positions: options.view === 'morph' ? morphPositions : geoPositions,
          rowCount: vertexCount,
          values: sizeBuffer,
          valueFormat:
            options.sizeBy === 'pagerank' || options.sizeBy === 'uniform' ? 'float32' : 'uint32',
          maximumValue: maximum,
          colorIndices: nodeColor,
          palette,
          minRadiusPixels: options.sizeBy === 'uniform' ? 3 : 1.6,
          maxRadiusPixels: options.sizeBy === 'uniform' ? 3 : options.maxRadius,
          opacity: dark ? 0.95 : 0.9
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
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
