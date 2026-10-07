// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  decodeGPUNetworkStatistics,
  encodeGPUNetworkStatisticsParameters,
  getGPUNetworkStatisticsLength,
  getGPUNetworkSubgraphFilterParameterLength,
  getGPUNetworkSubgraphFilterParameterValues,
  decodeGPUNetworkSubgraphFilterCounts,
  GPU_NETWORK_STATISTICS_PARAMETER_LENGTH,
  GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH,
  GPUNetworkAnalyticsColumns,
  GPUNetworkStatistics,
  GPUNetworkSubgraphFilter
} from '@luma.gl/experimental/gpu-network';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  COMMUNITY_COLORS,
  COMPONENT_COLORS,
  LENGTH_MAXIMUM,
  SPEED_MAXIMUM
} from './b10-scene-constants';
import {
  buildRoadGraph,
  formatLength,
  ROAD_CLASS_NAMES,
  SegmentIndex,
  sparkBar,
  type RoadGraph
} from './b10-road-graph';

/** Node metric drawn on the streets. */
export type CentralityMetric =
  | 'pageRank'
  | 'degree'
  | 'inDegree'
  | 'core'
  | 'community'
  | 'component';

/** Option state of the street-centrality scene. */
export type CentralityOptions = {
  metric: CentralityMetric;
  oneWay: boolean;
  damping: '0.5' | '0.7' | '0.85' | '0.95' | '0.99';
  pageRankIterations: '5' | '10' | '20' | '40' | '100';
  communityIterations: '2' | '5' | '10' | '32' | '100';
  componentIterations: '4' | '16' | '32' | '128';
  resolution: number;
  degreeBinning: 'linear' | 'log2';
  degreeBinWidth: number;
  speedRange: readonly [number, number];
  lengthRange: readonly [number, number];
  classRange: readonly [number, number];
  topPercent: number;
  dropIsolated: boolean;
  pairSlots: boolean;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  showRemoved: boolean;
};

const BIN_COUNT = 12;

type Topology = {
  directed: boolean;
  slotCount: number;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  reverseOffsets: Uint32Array | null;
  reverseNeighbors: Uint32Array | null;
  slotSource: Uint32Array;
  slotLength: Float32Array;
  slotClass: Float32Array;
  slotSpeed: Float32Array;
  /** Slot that colors each drawn segment. */
  segmentSlots: Uint32Array;
};

/** Undirected (symmetric CSR) or directed (one-way streets respected) view of the street graph. */
function buildTopology(graph: RoadGraph, directed: boolean): Topology {
  if (!directed) {
    return {
      directed,
      slotCount: graph.slotCount,
      offsets: graph.offsets,
      neighbors: graph.neighbors,
      reverseOffsets: null,
      reverseNeighbors: null,
      slotSource: graph.slotSource,
      slotLength: graph.weights,
      slotClass: graph.slotClass,
      slotSpeed: graph.slotSpeed,
      segmentSlots: graph.segmentSlots
    };
  }
  // Dataset edges are sorted by (source, target), so the edge index is the CSR slot.
  const {nodeCount, edgeCount, edgeSource, edgeTarget} = graph;
  const offsets = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < edgeCount; edge++) offsets[edgeSource[edge] + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const reverseOffsets = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < edgeCount; edge++) reverseOffsets[edgeTarget[edge] + 1]++;
  for (let node = 0; node < nodeCount; node++) reverseOffsets[node + 1] += reverseOffsets[node];
  const fill = reverseOffsets.slice(0, nodeCount);
  const reverseNeighbors = new Uint32Array(edgeCount);
  for (let edge = 0; edge < edgeCount; edge++)
    reverseNeighbors[fill[edgeTarget[edge]]++] = edgeSource[edge];
  return {
    directed,
    slotCount: edgeCount,
    offsets,
    neighbors: edgeTarget,
    reverseOffsets,
    reverseNeighbors,
    slotSource: edgeSource,
    slotLength: Float32Array.from(graph.edgeLength),
    slotClass: Float32Array.from(graph.edgeClass),
    slotSpeed: Float32Array.from(graph.edgeSpeed),
    segmentSlots: graph.segmentEdge
  };
}

/** Everything that depends on the topology; destroyed and rebuilt when the one-way option flips. */
type Pipeline = {
  topology: Topology;
  resources: SpatialAnalysisResources;
  analytics: CompiledGPUCommandGraph<void> | null;
  filterStatistics: CompiledGPUCommandGraph<void> | null;
  style: CompiledGPUCommandGraph<void>;
  buffers: {
    degree: Buffer;
    degreeNormalized: Buffer;
    inDegree: Buffer;
    inDegreeNormalized: Buffer;
    pageRank: Buffer;
    pageRankNormalized: Buffer;
    core: Buffer;
    coreNormalized: Buffer;
    components: Buffer;
    communities: Buffer;
    edgeMask: Buffer;
    slotScalar: Buffer;
    slotLabel: Buffer;
    slotMask: Buffer;
    nodeScalar: Buffer;
    nodeLabel: Buffer;
    statistics: Buffer;
    counts: Buffer;
    segmentSlots: Buffer;
  };
  analyticsReader: SummaryReader;
  statisticsReader: SummaryReader;
  /** Node columns read back after the analytics ran (for tooltips, quantiles and counts). */
  nodeData: {
    degree: Uint32Array;
    pageRank: Float32Array;
    core: Uint32Array;
    community: Uint32Array;
    component: Uint32Array;
    inDegree: Uint32Array;
  } | null;
};

/**
 * Network analytics of the Chicago street graph: node columns (`GPUNetworkAnalyticsColumns`) style
 * the streets, a predicate filter (`GPUNetworkSubgraphFilter`) keeps the roads that pass
 * speed, length, class and importance ranges, and `GPUNetworkStatistics` summarizes whatever is
 * left. The analytics run once per compile-time setting; every filter range is a parameter write
 * that re-runs only the filter, the statistics and a styling kernel.
 */
export async function createStreetCentrality(
  ctx: SceneContext<CentralityOptions>
): Promise<SceneInstance<CentralityOptions>> {
  const roads = ctx.datasets.get('chicago-roads');
  const {device} = ctx;
  const origin = roads.defaultOrigin;
  const graph = buildRoadGraph(roads, origin);
  const {nodeCount} = graph;
  const resources = new SpatialAnalysisResources(device, 'centrality');
  const segmentIndex = new SegmentIndex(graph.segments, graph.bounds);
  const projection = roads.getProjection(origin);
  const segmentsBuffer = resources.createBuffer('segments', graph.segments);
  const filterParameters = resources.createParameterBuffer(
    'filter-parameters',
    'float32',
    getGPUNetworkSubgraphFilterParameterLength({
      vertexColumnCount: 1,
      edgeColumnCount: 3,
      hasEdgeTimes: false
    })
  );
  const statisticsParameters = resources.createParameterBuffer(
    'statistics-parameters',
    'uint32',
    GPU_NETWORK_STATISTICS_PARAMETER_LENGTH
  );
  const styleParameters = resources.createParameterBuffer('style-parameters', 'uint32', 2);
  const undirectedTopology = buildTopology(graph, false);
  const directedTopology = buildTopology(graph, true);

  let destroyed = false;
  let pipeline: Pipeline | null = null;
  let pipelineGeneration = 0;
  let analyticsDirty = true;
  let filterDirty = true;
  let styleDirty = true;
  let importanceThreshold: number | null = null;
  let largestComponentLabel = 0;
  let measuring = false;

  ctx.setReadout(
    'network',
    `${formatCount(nodeCount)} intersections, ${formatCount(graph.edgeCount)} directed edges`
  );
  ctx.setReadout('length', formatLength(graph.networkLength));

  function releasePipeline(): void {
    if (!pipeline) return;
    pipeline.analyticsReader.stop();
    pipeline.statisticsReader.stop();
    pipeline.resources.destroy();
    pipeline = null;
  }

  function buildPipeline(): void {
    releasePipeline();
    const generation = ++pipelineGeneration;
    const topology = ctx.options.oneWay ? directedTopology : undirectedTopology;
    const pipe = new SpatialAnalysisResources(device, `centrality-${generation}`);
    const slots = topology.slotCount;
    const nodeBuffers = (name: string) => pipe.createBuffer(name, nodeCount * 4);
    const buffers: Pipeline['buffers'] = {
      degree: nodeBuffers('degree'),
      degreeNormalized: nodeBuffers('degree-normalized'),
      inDegree: nodeBuffers('in-degree'),
      inDegreeNormalized: nodeBuffers('in-degree-normalized'),
      pageRank: nodeBuffers('page-rank'),
      pageRankNormalized: nodeBuffers('page-rank-normalized'),
      core: nodeBuffers('core'),
      coreNormalized: nodeBuffers('core-normalized'),
      components: nodeBuffers('components'),
      communities: nodeBuffers('communities'),
      edgeMask: pipe.createBuffer('edge-mask', slots * 4),
      slotScalar: pipe.createBuffer('slot-scalar', slots * 4),
      slotLabel: pipe.createBuffer('slot-label', slots * 4),
      slotMask: pipe.createBuffer('slot-mask', slots * 4),
      nodeScalar: nodeBuffers('node-scalar'),
      nodeLabel: nodeBuffers('node-label'),
      statistics: pipe.createBuffer('statistics', getGPUNetworkStatisticsLength(BIN_COUNT) * 4),
      counts: pipe.createBuffer('counts', GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH * 4),
      segmentSlots: pipe.createBuffer('segment-slots', topology.segmentSlots)
    };
    const csr = {
      offsets: pipe.createBuffer('offsets', topology.offsets),
      neighbors: pipe.createBuffer('neighbors', topology.neighbors),
      reverseOffsets: topology.reverseOffsets
        ? pipe.createBuffer('reverse-offsets', topology.reverseOffsets)
        : null,
      reverseNeighbors: topology.reverseNeighbors
        ? pipe.createBuffer('reverse-neighbors', topology.reverseNeighbors)
        : null,
      slotSource: pipe.createBuffer('slot-source', topology.slotSource),
      slotLength: pipe.createBuffer('slot-length', topology.slotLength),
      slotClass: pipe.createBuffer('slot-class', topology.slotClass),
      slotSpeed: pipe.createBuffer('slot-speed', topology.slotSpeed),
      vertexMask: pipe.createBuffer('vertex-mask', nodeCount * 4)
    };
    const scalars = {
      residual: pipe.createBuffer('residual', 4),
      degeneracy: pipe.createBuffer('degeneracy', 4),
      coreConverged: pipe.createBuffer('core-converged', 4),
      componentConverged: pipe.createBuffer('component-converged', 4),
      communityConverged: pipe.createBuffer('community-converged', 4)
    };

    // ---- Styling graph: node metric -> per-slot values, edge mask -> per-slot weights ----
    const styleGraph = new GPUCommandGraph<void>(device, {id: 'centrality-style'});
    const sView = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(styleGraph, name, buffer, format, length);
    const slotSourceView = sView('slot-source', csr.slotSource, 'uint32', slots);
    addKernelPass(styleGraph, {
      id: 'slot-scalar',
      bindings: [
        {
          name: 'nodeScalar',
          view: sView('node-scalar', buffers.nodeScalar, 'float32', nodeCount),
          type: 'f32',
          access: 'read'
        },
        {name: 'slotSource', view: slotSourceView, type: 'u32', access: 'read'},
        {
          name: 'neighbors',
          view: sView('neighbors', csr.neighbors, 'uint32', slots),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'slotScalar',
          view: sView('slot-scalar', buffers.slotScalar, 'float32', slots),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: slots,
      body: `let a = nodeScalar[nodeScalarOffset + slotSource[slotSourceOffset + index]];
  let b = nodeScalar[nodeScalarOffset + neighbors[neighborsOffset + index]];
  slotScalar[slotScalarOffset + index] = 0.5 * (a + b);`
    });
    addKernelPass(styleGraph, {
      id: 'slot-label',
      bindings: [
        {
          name: 'nodeLabel',
          view: sView('node-label', buffers.nodeLabel, 'uint32', nodeCount),
          type: 'u32',
          access: 'read'
        },
        {name: 'slotSource', view: slotSourceView, type: 'u32', access: 'read'},
        {
          name: 'style',
          view: styleParameters.importToGraph(styleGraph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'slotLabel',
          view: sView('slot-label', buffers.slotLabel, 'uint32', slots),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: slots,
      body: `let label = nodeLabel[nodeLabelOffset + slotSource[slotSourceOffset + index]];
  // style[0] = largest component label, style[1] = 1 to highlight islands instead of cycling labels.
  if (style[styleOffset + 1u] == 1u) {
    slotLabel[slotLabelOffset + index] = select(1u, 0u, label == style[styleOffset]);
  } else {
    slotLabel[slotLabelOffset + index] = label;
  }`
    });
    addKernelPass(styleGraph, {
      id: 'slot-mask',
      bindings: [
        {
          name: 'edgeMask',
          view: sView('edge-mask', buffers.edgeMask, 'uint32', slots),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'slotMask',
          view: sView('slot-mask', buffers.slotMask, 'float32', slots),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: slots,
      body: `slotMask[slotMaskOffset + index] = select(0.0, 1.0, edgeMask[edgeMaskOffset + index] != 0u);`
    });
    const style: CompiledGPUCommandGraph<void> = pipe.track(styleGraph.compile());

    const analyticsReader = new SummaryReader(
      pipe,
      'analytics',
      [
        {buffer: buffers.degree, size: nodeCount * 4},
        {buffer: buffers.pageRank, size: nodeCount * 4},
        {buffer: buffers.core, size: nodeCount * 4},
        {buffer: buffers.communities, size: nodeCount * 4},
        {buffer: buffers.components, size: nodeCount * 4},
        {buffer: buffers.inDegree, size: nodeCount * 4},
        {buffer: scalars.residual, size: 4},
        {buffer: scalars.degeneracy, size: 4},
        {buffer: scalars.coreConverged, size: 4},
        {buffer: scalars.componentConverged, size: 4},
        {buffer: scalars.communityConverged, size: 4}
      ],
      bytes => {
        if (destroyed || pipeline?.analyticsReader !== analyticsReader) return;
        handleAnalytics(bytes);
      }
    );
    const statisticsReader = new SummaryReader(
      pipe,
      'statistics',
      [
        {buffer: buffers.statistics, size: getGPUNetworkStatisticsLength(BIN_COUNT) * 4},
        {buffer: buffers.counts, size: GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH * 4}
      ],
      bytes => {
        if (destroyed || pipeline?.statisticsReader !== statisticsReader) return;
        handleStatistics(bytes);
      }
    );
    pipeline = {
      topology,
      resources: pipe,
      analytics: null,
      filterStatistics: null,
      style,
      buffers,
      analyticsReader,
      statisticsReader,
      nodeData: null
    };
    // Keep what the compile helpers need alongside the pipeline.
    pipelineInputs = {csr, scalars};
    buildAnalytics();
    buildFilterStatistics();
    analyticsDirty = true;
    filterDirty = true;
    styleDirty = true;
    importanceThreshold = null;
    ctx.setReadout(
      'slots',
      `${formatCount(slots)} ${topology.directed ? 'directed edges' : 'street slots (both directions)'}`
    );
  }
  let pipelineInputs!: {
    csr: {
      offsets: Buffer;
      neighbors: Buffer;
      reverseOffsets: Buffer | null;
      reverseNeighbors: Buffer | null;
      slotSource: Buffer;
      slotLength: Buffer;
      slotClass: Buffer;
      slotSpeed: Buffer;
      vertexMask: Buffer;
    };
    scalars: Record<
      'residual' | 'degeneracy' | 'coreConverged' | 'componentConverged' | 'communityConverged',
      Buffer
    >;
  };

  /** Compile-time analytics settings: damping, iteration counts, directed or not. */
  function buildAnalytics(): void {
    const current = pipeline;
    if (!current) return;
    if (current.analytics) current.resources.release(current.analytics);
    const {csr, scalars} = pipelineInputs;
    const {buffers, topology} = current;
    const slots = topology.slotCount;
    const analyticsGraph = new GPUCommandGraph<void>(device, {id: 'centrality-analytics'});
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(analyticsGraph, name, buffer, format, length);
    analyticsGraph.add(
      new GPUNetworkAnalyticsColumns({
        id: 'centrality',
        offsets: view('offsets', csr.offsets, 'uint32', nodeCount + 1),
        neighbors: view('neighbors', csr.neighbors, 'uint32', slots),
        ...(csr.reverseOffsets && csr.reverseNeighbors
          ? {
              reverseOffsets: view('reverse-offsets', csr.reverseOffsets, 'uint32', nodeCount + 1),
              reverseNeighbors: view('reverse-neighbors', csr.reverseNeighbors, 'uint32', slots),
              inDegree: {
                output: view('in-degree', buffers.inDegree, 'uint32', nodeCount),
                normalized: view(
                  'in-degree-normalized',
                  buffers.inDegreeNormalized,
                  'float32',
                  nodeCount
                )
              }
            }
          : {}),
        degree: {
          output: view('degree', buffers.degree, 'uint32', nodeCount),
          normalized: view('degree-normalized', buffers.degreeNormalized, 'float32', nodeCount)
        },
        pageRank: {
          output: view('page-rank', buffers.pageRank, 'float32', nodeCount),
          normalized: view(
            'page-rank-normalized',
            buffers.pageRankNormalized,
            'float32',
            nodeCount
          ),
          damping: Number(ctx.options.damping),
          iterations: Number(ctx.options.pageRankIterations),
          residual: view('residual', scalars.residual, 'float32', 1)
        },
        coreNumber: {
          output: view('core', buffers.core, 'uint32', nodeCount),
          normalized: view('core-normalized', buffers.coreNormalized, 'float32', nodeCount),
          iterations: 64,
          degeneracy: view('degeneracy', scalars.degeneracy, 'uint32', 1),
          converged: view('core-converged', scalars.coreConverged, 'uint32', 1)
        },
        components: {
          output: view('components', buffers.components, 'uint32', nodeCount),
          iterations: Number(ctx.options.componentIterations),
          converged: view('component-converged', scalars.componentConverged, 'uint32', 1)
        },
        communities: {
          output: view('communities', buffers.communities, 'uint32', nodeCount),
          iterations: Number(ctx.options.communityIterations),
          converged: view('community-converged', scalars.communityConverged, 'uint32', 1)
        }
      })
    );
    current.analytics = current.resources.track(analyticsGraph.compile());
    analyticsDirty = true;
  }

  /** Compile-time filter and statistics settings: isolated vertices, slot pairing, binning. */
  function buildFilterStatistics(): void {
    const current = pipeline;
    if (!current) return;
    if (current.filterStatistics) {
      current.resources.release(current.filterStatistics);
      current.filterStatistics = null;
    }
    const {csr} = pipelineInputs;
    const {buffers, topology} = current;
    const slots = topology.slotCount;
    const graphForStatistics = new GPUCommandGraph<void>(device, {
      id: 'centrality-filter-statistics'
    });
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graphForStatistics, name, buffer, format, length);
    const offsets = view('offsets', csr.offsets, 'uint32', nodeCount + 1);
    const neighbors = view('neighbors', csr.neighbors, 'uint32', slots);
    const vertexMask = view('vertex-mask', csr.vertexMask, 'uint32', nodeCount);
    const edgeMask = view('edge-mask', buffers.edgeMask, 'uint32', slots);
    const filter = new GPUNetworkSubgraphFilter({
      id: 'filter',
      offsets,
      neighbors,
      directed: topology.directed,
      pairUndirectedSlots: ctx.options.pairSlots,
      dropIsolated: ctx.options.dropIsolated,
      vertexColumns: [
        view('page-rank-normalized', buffers.pageRankNormalized, 'float32', nodeCount)
      ],
      edgeColumns: [
        view('slot-speed', csr.slotSpeed, 'float32', slots),
        view('slot-length', csr.slotLength, 'float32', slots),
        view('slot-class', csr.slotClass, 'float32', slots)
      ],
      parameters: filterParameters.importToGraph(graphForStatistics),
      output: {
        vertexMask,
        edgeMask,
        counts: view('counts', buffers.counts, 'uint32', GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH)
      }
    });
    graphForStatistics.add(filter);
    const statistics = new GPUNetworkStatistics({
      id: 'statistics',
      offsets,
      neighbors,
      directed: topology.directed,
      vertexMask,
      edgeMask,
      communities: view('communities', buffers.communities, 'uint32', nodeCount),
      parameters: statisticsParameters.importToGraph(graphForStatistics),
      degreeBinCount: BIN_COUNT,
      degreeBinning: ctx.options.degreeBinning,
      componentIterations: Number(ctx.options.componentIterations),
      output: view(
        'statistics',
        buffers.statistics,
        'uint32',
        getGPUNetworkStatisticsLength(BIN_COUNT)
      )
    });
    // The statistics contributor owns scratch buffers: track it before the compiled graph.
    current.resources.track(statistics);
    graphForStatistics.add(statistics);
    current.filterStatistics = current.resources.track(graphForStatistics.compile());
    filterDirty = true;
  }

  function writeFilterParameters(): void {
    const {speedRange, lengthRange, classRange} = ctx.options;
    filterParameters.write(
      getGPUNetworkSubgraphFilterParameterValues(
        {vertexColumnCount: 1, edgeColumnCount: 3, hasEdgeTimes: false},
        {
          vertexRanges: [importanceThreshold === null ? null : [importanceThreshold, Infinity]],
          edgeRanges: [
            speedRange[0] <= 0 && speedRange[1] >= SPEED_MAXIMUM
              ? null
              : [speedRange[0], speedRange[1] >= SPEED_MAXIMUM ? Infinity : speedRange[1]],
            lengthRange[0] <= 0 && lengthRange[1] >= LENGTH_MAXIMUM
              ? null
              : [lengthRange[0], lengthRange[1] >= LENGTH_MAXIMUM ? Infinity : lengthRange[1]],
            classRange[0] <= 0 && classRange[1] >= ROAD_CLASS_NAMES.length - 1
              ? null
              : [classRange[0], classRange[1] + 1]
          ]
        }
      )
    );
    statisticsParameters.write(
      encodeGPUNetworkStatisticsParameters({
        resolution: ctx.options.resolution,
        degreeBinWidth: ctx.options.degreeBinWidth
      })
    );
    filterDirty = true;
  }

  function writeStyleParameters(): void {
    styleParameters.write(
      Uint32Array.of(largestComponentLabel, ctx.options.metric === 'component' ? 1 : 0)
    );
    styleDirty = true;
  }

  function handleAnalytics(bytes: ArrayBuffer): void {
    const current = pipeline;
    if (!current) return;
    let offset = 0;
    const next = <T extends Uint32Array | Float32Array>(
      Constructor: {new (buffer: ArrayBuffer, byteOffset: number, length: number): T},
      length: number
    ): T => {
      const result = new Constructor(bytes.slice(offset, offset + length * 4), 0, length);
      offset += length * 4;
      return result;
    };
    const degree = next(Uint32Array, nodeCount);
    const pageRank = next(Float32Array, nodeCount);
    const core = next(Uint32Array, nodeCount);
    const community = next(Uint32Array, nodeCount);
    const component = next(Uint32Array, nodeCount);
    const inDegree = next(Uint32Array, nodeCount);
    const residual = next(Float32Array, 1)[0];
    const degeneracy = next(Uint32Array, 1)[0];
    const coreConverged = next(Uint32Array, 1)[0];
    const componentConverged = next(Uint32Array, 1)[0];
    const communityConverged = next(Uint32Array, 1)[0];
    current.nodeData = {degree, pageRank, core, community, component, inDegree};

    const countLabels = (labels: Uint32Array) => {
      const counts = new Map<number, number>();
      for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
      return counts;
    };
    const communityCounts = countLabels(community);
    const componentCounts = countLabels(component);
    let largest = 0;
    let largestSize = 0;
    for (const [label, size] of componentCounts) {
      if (size > largestSize) {
        largest = label;
        largestSize = size;
      }
    }
    largestComponentLabel = largest;
    writeStyleParameters();
    let maximumPageRank = 0;
    let minimumPageRank = Infinity;
    let sumPageRank = 0;
    for (const value of pageRank) {
      maximumPageRank = Math.max(maximumPageRank, value);
      minimumPageRank = Math.min(minimumPageRank, value);
      sumPageRank += value;
    }
    // Importance quantile: normalized PageRank of the intersection at the chosen percentile.
    const sorted = Float32Array.from(pageRank).sort();
    const range = maximumPageRank - minimumPageRank;
    quantile = (fraction: number) =>
      range > 0
        ? (sorted[Math.min(nodeCount - 1, Math.floor(fraction * nodeCount))] - minimumPageRank) /
          range
        : 0;
    updateImportanceThreshold();
    ctx.setReadout(
      'communities',
      `${formatCount(communityCounts.size)} found, largest ${formatCount(Math.max(...communityCounts.values()))} intersections${communityConverged ? '' : ' (not converged)'}`
    );
    ctx.setReadout(
      'components',
      `${formatCount(componentCounts.size)}, largest ${((100 * largestSize) / nodeCount).toFixed(1)}% of intersections${componentConverged ? '' : ' (not converged)'}`
    );
    ctx.setReadout('pageRankResidual', residual);
    ctx.setReadout(
      'pageRankPeak',
      `${(maximumPageRank / (sumPageRank / nodeCount)).toFixed(1)}x the average`
    );
    ctx.setReadout('degeneracy', `${degeneracy}${coreConverged ? '' : ' (not converged)'}`);
    ctx.setReadout('analyticsTime', null);
  }
  let quantile: (fraction: number) => number = () => 0;

  function updateImportanceThreshold(): void {
    const {topPercent} = ctx.options;
    importanceThreshold = topPercent >= 100 ? null : quantile(1 - topPercent / 100);
    writeFilterParameters();
  }

  function handleStatistics(bytes: ArrayBuffer): void {
    const statisticsWords = getGPUNetworkStatisticsLength(BIN_COUNT);
    const words = new Uint32Array(
      bytes,
      0,
      statisticsWords + GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH
    );
    const result = decodeGPUNetworkStatistics(words.subarray(0, statisticsWords), {
      degreeBinCount: BIN_COUNT
    });
    const counts = decodeGPUNetworkSubgraphFilterCounts(words.subarray(statisticsWords));
    ctx.setReadout(
      'liveVertices',
      `${formatCount(result.liveVertexCount)} of ${formatCount(nodeCount)}`
    );
    ctx.setReadout('liveEdges', formatCount(result.liveEdgeCount));
    ctx.setReadout(
      'filterCounts',
      `${formatCount(counts.liveVertexCount)} vertices, ${formatCount(counts.liveSlotCount)} slots`
    );
    ctx.setReadout(
      'statComponents',
      `${formatCount(result.componentCount)}${result.componentsConverged ? '' : ' (not converged)'}`
    );
    ctx.setReadout(
      'largestComponent',
      result.liveVertexCount > 0
        ? `${formatCount(result.largestComponentSize)} (${((100 * result.largestComponentSize) / result.liveVertexCount).toFixed(1)}%)`
        : null
    );
    ctx.setReadout('isolated', result.isolatedVertexCount);
    ctx.setReadout('maxDegree', result.maxTotalDegree);
    ctx.setReadout('modularity', result.modularityValid ? result.modularity : 'n/a');
    const histogram = result.totalDegreeHistogram;
    const peak = Math.max(...histogram, 1);
    ctx.setReadout('histogram', histogram.map(count => sparkBar(Math.sqrt(count / peak))).join(''));
    const mode =
      ctx.options.degreeBinning === 'log2'
        ? 'bins: 0, 1, 2-3, 4-7, ...'
        : `bins of ${ctx.options.degreeBinWidth}: 0, ${ctx.options.degreeBinWidth}, ...`;
    ctx.setReadout('histogramBins', mode);
  }

  async function measure(): Promise<void> {
    const current = pipeline;
    if (measuring || !current?.filterStatistics || !current.analytics) return;
    measuring = true;
    try {
      const options = {
        parameters: undefined,
        completionBuffer: current.buffers.counts,
        runs: 5,
        repetitions: 2
      };
      const analytics = await measureCompiledGraph(device, current.analytics, options);
      const filtering = await measureCompiledGraph(device, current.filterStatistics, options);
      if (!destroyed) {
        ctx.setReadout('analyticsTime', `${analytics.milliseconds.toFixed(1)} ms`);
        ctx.setReadout('filterTime', `${filtering.milliseconds.toFixed(1)} ms`);
      }
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  buildPipeline();
  writeFilterParameters();
  writeStyleParameters();

  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];

  const encodeSelectedMetric = (
    commandEncoder: Parameters<SceneInstance<CentralityOptions>['encode']>[0]
  ) => {
    const current = pipeline!;
    const {metric} = ctx.options;
    const scalarSource =
      metric === 'degree'
        ? current.buffers.degreeNormalized
        : metric === 'inDegree'
          ? current.buffers.inDegreeNormalized
          : metric === 'core'
            ? current.buffers.coreNormalized
            : current.buffers.pageRankNormalized;
    const labelSource =
      metric === 'component' ? current.buffers.components : current.buffers.communities;
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: scalarSource,
      destinationBuffer: current.buffers.nodeScalar,
      size: nodeCount * 4
    });
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: labelSource,
      destinationBuffer: current.buffers.nodeLabel,
      size: nodeCount * 4
    });
  };

  return {
    getCompiledGraphs: () => {
      const current = pipeline;
      return (
        current?.analytics && current.filterStatistics
          ? [current.analytics, current.filterStatistics, current.style]
          : []
      ) as CompiledGPUCommandGraph<never>[];
    },

    setOption(id) {
      switch (id) {
        case 'oneWay':
          buildPipeline();
          writeFilterParameters();
          writeStyleParameters();
          ctx.requestLayers();
          break;
        case 'damping':
        case 'pageRankIterations':
        case 'communityIterations':
          buildAnalytics();
          break;
        case 'componentIterations':
          buildAnalytics();
          buildFilterStatistics();
          break;
        case 'degreeBinning':
        case 'dropIsolated':
        case 'pairSlots':
          buildFilterStatistics();
          break;
        case 'speedRange':
        case 'lengthRange':
        case 'classRange':
        case 'resolution':
        case 'degreeBinWidth':
          writeFilterParameters();
          break;
        case 'topPercent':
          updateImportanceThreshold();
          break;
        case 'metric':
          writeStyleParameters();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'measure') void measure();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      const current = pipeline;
      if (!event.coordinate || !current) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const segment = segmentIndex.nearest(x, y, 35);
      if (segment < 0) return null;
      const slot = current.topology.segmentSlots[segment];
      const edge = graph.segmentEdge[segment];
      const source = current.topology.slotSource[slot];
      const data = current.nodeData;
      const lines = [
        `${ROAD_CLASS_NAMES[graph.edgeClass[edge]]}, ${Math.round(graph.edgeLength[edge])} m, ${graph.edgeSpeed[edge]} km/h`
      ];
      if (data) {
        const average = 1 / nodeCount;
        lines.push(
          `start intersection: degree ${data.degree[source]}, core ${data.core[source]}, PageRank ${(data.pageRank[source] / average).toFixed(2)}x average`
        );
      }
      return lines.join('\n');
    },

    encode(commandEncoder) {
      const current = pipeline;
      if (!current?.analytics || !current.filterStatistics) return;
      if (analyticsDirty) {
        current.analytics.encode(commandEncoder, {parameters: undefined});
        current.analyticsReader.request(commandEncoder);
        analyticsDirty = false;
        filterDirty = true;
      }
      if (filterDirty) {
        current.filterStatistics.encode(commandEncoder, {parameters: undefined});
        current.statisticsReader.request(commandEncoder);
        filterDirty = false;
        styleDirty = true;
      }
      if (styleDirty) {
        encodeSelectedMetric(commandEncoder);
        current.style.encode(commandEncoder, {parameters: undefined});
        styleDirty = false;
      }
      current.analyticsReader.flush(commandEncoder);
      current.statisticsReader.flush(commandEncoder);
    },

    getLayers() {
      const current = pipeline;
      if (!current) return [];
      const {metric, ramp, showRemoved} = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (showRemoved) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'centrality-removed',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: graph.segmentCount,
            color: dark ? [150, 158, 185, 55] : [90, 100, 125, 60],
            widthPixels: 0.9
          })
        );
      }
      const isLabel = metric === 'community' || metric === 'component';
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'centrality-streets',
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: graph.segmentCount,
          weights: current.buffers.slotMask,
          values: isLabel ? current.buffers.slotLabel : current.buffers.slotScalar,
          valueIndices: current.buffers.segmentSlots,
          ...(isLabel
            ? {
                valueFormat: 'uint32' as const,
                colormap: 'category' as const,
                palette: metric === 'component' ? COMPONENT_COLORS : COMMUNITY_COLORS
              }
            : {
                valueFormat: 'float32' as const,
                colormap: ramp,
                valueRange: [0, 1] as const,
                sqrtScale: true
              }),
          widthPixels: metric === 'component' ? 1.6 : 1.5
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      releasePipeline();
      resources.destroy();
    }
  };
}
