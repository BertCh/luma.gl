// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  createGPUComputeCommandNode,
  createTransientView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraph, GPUGraphTopology, type GPUGraphAdjacency} from '@luma.gl/gpgpu/gpu-graph';
import {
  GPUNetworkAnalyticsColumns,
  GPUNetworkNeighborhood,
  GPUNetworkPathExtraction,
  GPUNetworkReachability,
  type GPUNetworkAnalyticsColumnsProps
} from '@luma.gl/experimental/gpu-network';
import type {GPUGraphNodeColumn} from './gpu-graph-columns';

/**
 * Imports one caller-owned buffer into a command graph and returns a packed view over it, using
 * the public `GPUCommandGraph.importBuffer()` and `createDataView()` methods. The graph never
 * destroys the buffer.
 */
function importGraphBuffer<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph<void>,
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return graph.createDataView(handle, {format, length});
}

const SCALAR_BYTE_LENGTH = 4;
const WORKGROUP_SIZE = 256;
/** Graphs at or above this vertex count run each analytics metric in its own encoded stage. */
const MASSIVE_VERTEX_COUNT = 16_384;
const MAXIMUM_NEIGHBORHOOD_HOPS = 8;
const DEFAULT_PATH_MAXIMUM_ITERATIONS = 64;
/**
 * Hop thresholds of {@link GPUGraphAnalysisColumns.reachabilityBands}: band 0 is the source itself,
 * band 1 is one hop away, and so on. Nodes beyond the last threshold are `NONE`.
 */
const DEFAULT_BAND_HOP_THRESHOLDS: readonly number[] = [0, 1, 2, 3, 4, 6, 8, 12];

/** Names of the analytics columns published by {@link GPUGraphAnalysisColumns}. */
export type GPUGraphAnalysisColumnName =
  | 'degree'
  | 'pageRank'
  | 'coreNumber'
  | 'component'
  | 'community';

/** Optional tuning of {@link GPUGraphAnalysisColumns}. All values are compile-time. */
export type GPUGraphAnalysisColumnsOptions = {
  /** Prefix for graph, node and buffer IDs. Defaults to `'gpu-graph-analysis'`. */
  id?: string;
  /** Relaxation iterations (maximum hop distance) of the A to B path search. Default 64. */
  pathMaximumIterations?: number;
  /** Ascending hop thresholds for `reachabilityBands`. Default `[0, 1, 2, 3, 4, 6, 8, 12]`. */
  bandHopThresholds?: readonly number[];
};

/** Estimated resident and adapter-limit footprint of {@link GPUGraphAnalysisColumns}. */
export type GPUGraphAnalysisColumnsFootprint = {
  /** Bytes of the largest single buffer (the symmetrized neighbor list). */
  largestBufferBytes: number;
  /** Sum of persistent buffers: topology, columns, masks, parameters. Excludes graph transients. */
  residentBytes: number;
};

/**
 * Estimates the footprint of {@link GPUGraphAnalysisColumns} for a graph without allocating.
 *
 * The undirected topology stores `2 * edgeCount` neighbors and edge IDs, which dominates.
 */
export function estimateGPUGraphAnalysisColumnsFootprint(
  vertexCount: number,
  edgeCount: number,
  bandCount: number = DEFAULT_BAND_HOP_THRESHOLDS.length
): GPUGraphAnalysisColumnsFootprint {
  const symmetricEdgeBytes = 2 * edgeCount * SCALAR_BYTE_LENGTH;
  // offsets + 3 raw columns + 3 normalized + component + community + hop distances + mask +
  // path ranks + bands, all one u32 per node.
  const nodeColumnBytes = vertexCount * SCALAR_BYTE_LENGTH * 13 + SCALAR_BYTE_LENGTH;
  return {
    largestBufferBytes: Math.max(symmetricEdgeBytes, nodeColumnBytes / 13),
    residentBytes: 2 * symmetricEdgeBytes + nodeColumnBytes + (16 + bandCount) * SCALAR_BYTE_LENGTH
  };
}

/**
 * Returns why the analysis columns cannot be built for this graph on `device`, or `null` when they
 * can. Checks the adapter's storage-binding and buffer-size limits against the symmetrized CSR.
 */
export function getGPUGraphAnalysisColumnsSkipReason(
  device: Device,
  vertexCount: number,
  edgeCount: number
): string | null {
  if (device.type !== 'webgpu') return 'analysis columns require WebGPU compute';
  if (vertexCount < 1) return 'graph has no vertices';
  if (edgeCount < 1) return 'graph has no edges';
  const {largestBufferBytes} = estimateGPUGraphAnalysisColumnsFootprint(vertexCount, edgeCount);
  const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  if (largestBufferBytes > limit) {
    return `symmetrized adjacency needs ${largestBufferBytes} bytes, adapter limit is ${limit}`;
  }
  return null;
}

/** Fixed layout of {@link GPUGraphAnalysisColumns} measurements, available without readback. */
export type GPUGraphAnalysisColumnsStats = {
  /** Persistent buffer bytes owned by the analysis columns. */
  residentBufferBytes: number;
  /** Physical transient bytes of every compiled graph. */
  transientBufferBytes: number;
  /** Number of analytics stages already encoded, including topology. */
  completedStages: number;
  /** Total number of stages: topology plus analytics. */
  totalStages: number;
  /** Cumulative CPU milliseconds spent encoding topology and analytics stages. */
  analyticsEncodeMilliseconds: number;
  /** CPU milliseconds of the most recent interaction encoding. */
  interactionEncodeMilliseconds: number;
  /** Number of interaction encodings so far. */
  interactionEncodeCount: number;
};

/**
 * GPU-resident, row-aligned analysis contributor outputs for the deck.gl graph layers.
 *
 * Every output is a caller-visible {@link Buffer} with one packed 4-byte row per graph vertex and
 * `STORAGE | COPY_SRC | COPY_DST` usage, so layers bind them directly as storage buffers and
 * nothing is read back. Construction compiles every graph exactly once and never submits or
 * encodes; per-interaction inputs (hover, hops, path endpoints) live in tiny buffers written with
 * `Buffer.write`, so interactions never recompile anything.
 *
 * Topology: the contributors need a symmetric CSR. This class builds ONE undirected `GPUGraphTopology`
 * over an undirected view of the source graph's own source and target vectors (cost: `2E`
 * neighbors and `2E` edge IDs, `16 E` bytes, no copy of the edge list). The directed forward CSR
 * was rejected: it would make hover neighborhoods and click-to-click paths one-way and would
 * leave analytics treating the graph as directed. Hover and path semantics are therefore
 * direction-agnostic, matching the effect's existing `'both'` breadth-first search.
 *
 * Path weights: positions move every frame, so the A to B path uses unit (hop) weights generated
 * on the GPU each interaction encoding. Euclidean weights would invalidate the path continuously.
 *
 * Encoding: {@link encodeAnalytics} encodes topology then every analytics metric once (one metric
 * per call for massive graphs). {@link encodeInteraction} encodes only when an input changed.
 * Both append to the caller's command encoder and never submit.
 */
export class GPUGraphAnalysisColumns {
  /** Normalized `degree`, `pageRank`, `coreNumber` (float32 in [0, 1]) and uint32 labels. */
  readonly columns: Readonly<Record<GPUGraphAnalysisColumnName, GPUGraphNodeColumn>>;
  /** Un-normalized degree (uint32), PageRank (float32, sums to 1) and core number (uint32). */
  readonly rawColumns: Readonly<Record<'degree' | 'pageRank' | 'coreNumber', GPUGraphNodeColumn>>;
  /**
   * Nonzero for every vertex within `hops` of the hovered seed (either direction), zero for all
   * vertices when no vertex is hovered. Suitable as `highlightMask`.
   */
  readonly neighborhoodMask: Buffer;
  /** Hop distance from the hovered seed, or `GPU_NETWORK_REACHABILITY_NONE` outside `hops`. */
  readonly hopDistances: Buffer;
  /**
   * 0 for vertices off the shortest path, otherwise the 1-based position along the path from the
   * source endpoint A (rank 1) to the target endpoint B. All zero when either endpoint is null or
   * B is unreachable from A. Suitable as `pathRanks`.
   */
  readonly pathRanks: Buffer;
  /**
   * Hop band of each vertex relative to path source A: index into the band thresholds, or
   * `GPU_NETWORK_REACHABILITY_NONE` when unreached or beyond the last threshold.
   */
  readonly reachabilityBands: Buffer;
  /** Ascending hop thresholds that define `reachabilityBands`. */
  readonly bandHopThresholds: readonly number[];
  /** Vertex count of every row-aligned output. */
  readonly vertexCount: number;
  /** Compiled one-time topology graph (symmetrized CSR). */
  readonly topologyGraph: CompiledGPUCommandGraph<void>;
  /** Compiled one-time analytics stage graphs, in encode order. */
  readonly analyticsGraphs: readonly CompiledGPUCommandGraph<void>[];
  /** Compiled interaction graph: neighborhood, reachability, bands, path extraction, ranks. */
  readonly interactionGraph: CompiledGPUCommandGraph<void>;
  /** Undirected topology the contributors run on. */
  readonly topology: GPUGraphTopology;

  private readonly device: Device;
  private readonly buffers: Buffer[] = [];
  private readonly vectors: GPUVector[] = [];
  private readonly contributors: {destroy?(): void}[] = [];
  private readonly seeds: Buffer;
  private readonly seedCount: Buffer;
  private readonly hops: Buffer;
  private readonly pathSources: Buffer;
  private readonly pathSourceCount: Buffer;
  private readonly pathTargets: Buffer;
  private readonly pathTargetCount: Buffer;
  private readonly stagesPerCall: number;
  private completedStages = 0;
  private interactionDirty = true;
  private interactionEncodeCount = 0;
  private analyticsEncodeMilliseconds = 0;
  private interactionEncodeMilliseconds = 0;
  private hoveredVertex: number | null = null;
  private neighborhoodHops = 2;
  private pathSource: number | null = null;
  private pathTarget: number | null = null;
  private destroyed = false;

  /**
   * Builds and compiles every graph. Never submits and never reads a buffer.
   *
   * @param device WebGPU device.
   * @param graph Source graph; its source and target vectors are reused, never copied.
   * @param options Compile-time tuning.
   */
  constructor(device: Device, graph: GPUGraph, options: GPUGraphAnalysisColumnsOptions = {}) {
    const skipReason = getGPUGraphAnalysisColumnsSkipReason(
      device,
      graph.vertexCount,
      graph.edgeCount
    );
    if (skipReason) throw new Error(`GPUGraphAnalysisColumns unavailable: ${skipReason}`);
    this.device = device;
    const id = options.id ?? 'gpu-graph-analysis';
    const nodeCount = graph.vertexCount;
    const edgeCount = graph.edgeCount;
    const symmetricCount = edgeCount * 2;
    const massive = nodeCount >= MASSIVE_VERTEX_COUNT;
    this.vertexCount = nodeCount;
    this.stagesPerCall = massive ? 1 : Number.POSITIVE_INFINITY;
    this.bandHopThresholds = options.bandHopThresholds ?? DEFAULT_BAND_HOP_THRESHOLDS;
    const pathIterations = options.pathMaximumIterations ?? DEFAULT_PATH_MAXIMUM_ITERATIONS;

    try {
      // Symmetrized topology over the same source and target vectors.
      const undirected = new GPUGraph({
        vertexCount: nodeCount,
        directed: false,
        sourceVertices: graph.sourceVertices,
        targetVertices: graph.targetVertices
      });
      const offsets = this.createBuffer(`${id}-offsets`, nodeCount + 1);
      const neighbors = this.createBuffer(`${id}-neighbors`, symmetricCount);
      this.topology = new GPUGraphTopology({
        id: `${id}-topology`,
        graph: undirected,
        forward: this.createAdjacency(id, offsets, neighbors, nodeCount, symmetricCount),
        invalidEdgeCount: this.createVector(`${id}-invalid-edges`, 1)
      });
      const topology = new GPUCommandGraph<void>(device, {
        id: `${id}-topology`
      });
      this.topology.addToGraph(topology);
      this.topologyGraph = topology.compile();

      // Analytics columns. Raw outputs are required by the contributor; normalized ones are published.
      const raw = {
        degree: this.createBuffer(`${id}-degree-raw`, nodeCount),
        pageRank: this.createBuffer(`${id}-page-rank-raw`, nodeCount),
        coreNumber: this.createBuffer(`${id}-core-number-raw`, nodeCount)
      };
      const normalized = {
        degree: this.createBuffer(`${id}-degree`, nodeCount),
        pageRank: this.createBuffer(`${id}-page-rank`, nodeCount),
        coreNumber: this.createBuffer(`${id}-core-number`, nodeCount)
      };
      const component = this.createBuffer(`${id}-component`, nodeCount);
      const community = this.createBuffer(`${id}-community`, nodeCount);
      this.columns = {
        degree: {buffer: normalized.degree, format: 'float32'},
        pageRank: {buffer: normalized.pageRank, format: 'float32'},
        coreNumber: {buffer: normalized.coreNumber, format: 'float32'},
        component: {buffer: component, format: 'uint32'},
        community: {buffer: community, format: 'uint32'}
      };
      this.rawColumns = {
        degree: {buffer: raw.degree, format: 'uint32'},
        pageRank: {buffer: raw.pageRank, format: 'float32'},
        coreNumber: {buffer: raw.coreNumber, format: 'uint32'}
      };

      const iterations = massive ? 2 : undefined;
      const metricGroups: readonly (readonly GPUGraphAnalysisColumnName[])[] = massive
        ? [['degree'], ['component'], ['community'], ['pageRank'], ['coreNumber']]
        : [['degree', 'pageRank', 'coreNumber', 'component', 'community']];
      const analyticsGraphs: CompiledGPUCommandGraph<void>[] = [];
      for (const [groupIndex, group] of metricGroups.entries()) {
        const stage = new GPUCommandGraph<void>(device, {
          id: `${id}-analytics-${groupIndex}`
        });
        const importColumn = <Format extends 'uint32' | 'float32'>(
          name: string,
          buffer: Buffer,
          format: Format,
          length: number = nodeCount
        ) => importGraphBuffer(stage, `${id}-${groupIndex}-${name}`, buffer, format, length);
        const props: GPUNetworkAnalyticsColumnsProps = {
          id: `${id}-analytics-${groupIndex}`,
          offsets: importColumn('csr-offsets', offsets, 'uint32', nodeCount + 1),
          neighbors: importColumn('csr-neighbors', neighbors, 'uint32', symmetricCount)
        };
        if (group.includes('degree')) {
          props.degree = {
            output: importColumn('degree-raw', raw.degree, 'uint32'),
            normalized: importColumn('degree-normalized', normalized.degree, 'float32')
          };
        }
        if (group.includes('pageRank')) {
          props.pageRank = {
            output: importColumn('page-rank-raw', raw.pageRank, 'float32'),
            normalized: importColumn('page-rank-normalized', normalized.pageRank, 'float32'),
            iterations: iterations ?? 20
          };
        }
        if (group.includes('coreNumber')) {
          props.coreNumber = {
            output: importColumn('core-number-raw', raw.coreNumber, 'uint32'),
            normalized: importColumn('core-number-normalized', normalized.coreNumber, 'float32'),
            iterations: iterations ?? 32
          };
        }
        if (group.includes('component')) {
          props.components = {
            output: importColumn('component', component, 'uint32'),
            iterations: iterations ?? 32
          };
        }
        if (group.includes('community')) {
          props.communities = {
            output: importColumn('community', community, 'uint32'),
            iterations: iterations ?? 16
          };
        }
        const contributor = new GPUNetworkAnalyticsColumns(props);
        this.contributors.push(contributor);
        stage.add(contributor);
        analyticsGraphs.push(stage.compile());
      }
      this.analyticsGraphs = analyticsGraphs;

      // Interaction inputs and outputs.
      this.seeds = this.createBuffer(`${id}-hover-seed`, 1);
      this.seedCount = this.createBuffer(`${id}-hover-seed-count`, 1);
      this.hops = this.createBuffer(`${id}-hops`, 1);
      this.hops.write(Uint32Array.of(this.neighborhoodHops));
      this.pathSources = this.createBuffer(`${id}-path-source`, 1);
      this.pathSourceCount = this.createBuffer(`${id}-path-source-count`, 1);
      this.pathTargets = this.createBuffer(`${id}-path-target`, 1);
      this.pathTargetCount = this.createBuffer(`${id}-path-target-count`, 1);
      const bandThresholds = this.createBuffer(
        `${id}-band-thresholds`,
        this.bandHopThresholds.length
      );
      bandThresholds.write(Float32Array.from(this.bandHopThresholds));
      this.neighborhoodMask = this.createBuffer(`${id}-neighborhood-mask`, nodeCount);
      this.hopDistances = this.createBuffer(`${id}-hop-distances`, nodeCount);
      this.pathRanks = this.createBuffer(`${id}-path-ranks`, nodeCount);
      this.reachabilityBands = this.createBuffer(`${id}-reachability-bands`, nodeCount);

      const interaction = new GPUCommandGraph<void>(device, {
        id: `${id}-interaction`
      });
      const importInput = <Format extends 'uint32' | 'float32'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ) => importGraphBuffer(interaction, `${id}-i-${name}`, buffer, format, length);
      const csrOffsets = importInput('csr-offsets', offsets, 'uint32', nodeCount + 1);
      const csrNeighbors = importInput('csr-neighbors', neighbors, 'uint32', symmetricCount);
      const hoodMask = importInput('neighborhood-mask', this.neighborhoodMask, 'uint32', nodeCount);
      const neighborhood = new GPUNetworkNeighborhood({
        id: `${id}-neighborhood`,
        offsets: csrOffsets,
        neighbors: csrNeighbors,
        seeds: importInput('hover-seed', this.seeds, 'uint32', 1),
        seedCount: importInput('hover-seed-count', this.seedCount, 'uint32', 1),
        hops: importInput('hops', this.hops, 'uint32', 1),
        maxHops: MAXIMUM_NEIGHBORHOOD_HOPS,
        hopDistances: importInput('hop-distances', this.hopDistances, 'uint32', nodeCount),
        nodeMask: hoodMask
      });
      interaction.add(neighborhood);

      const unitWeights = createTransientView(
        interaction,
        `${id}-unit-weights`,
        'float32',
        symmetricCount
      );
      interaction.add(
        createAnalysisKernelNode(interaction, {
          id: `${id}-unit-weights`,
          bindings: [
            {
              name: 'weights',
              view: unitWeights,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: symmetricCount,
          body: 'weights[weightsOffset + index] = 1.0;'
        })
      );
      const costs = createTransientView(interaction, `${id}-path-costs`, 'float32', nodeCount);
      const predecessors = createTransientView(
        interaction,
        `${id}-path-predecessors`,
        'uint32',
        nodeCount
      );
      const reachability = new GPUNetworkReachability({
        id: `${id}-reachability`,
        offsets: csrOffsets,
        neighbors: csrNeighbors,
        weights: unitWeights,
        sources: importInput('path-source', this.pathSources, 'uint32', 1),
        sourceCount: importInput('path-source-count', this.pathSourceCount, 'uint32', 1),
        maxIterations: pathIterations,
        costs,
        predecessors,
        bandThresholds: importInput(
          'band-thresholds',
          bandThresholds,
          'float32',
          this.bandHopThresholds.length
        ),
        bands: importInput('reachability-bands', this.reachabilityBands, 'uint32', nodeCount)
      });
      interaction.add(reachability);

      // A path has at most `pathIterations + 1` nodes because unit-weight relaxation converges in
      // as many iterations as the hop distance.
      const pathCapacity = pathIterations + 1;
      const pathIds = createTransientView(interaction, `${id}-path-ids`, 'uint32', pathCapacity);
      const pathCount = createTransientView(interaction, `${id}-path-count`, 'uint32', 1);
      const pathOverflow = createTransientView(interaction, `${id}-path-overflow`, 'uint32', 1);
      const extraction = new GPUNetworkPathExtraction({
        id: `${id}-path-extraction`,
        predecessors,
        costs,
        targets: importInput('path-target', this.pathTargets, 'uint32', 1),
        targetCount: importInput('path-target-count', this.pathTargetCount, 'uint32', 1),
        maxPathLength: pathCapacity,
        output: {ids: pathIds, count: pathCount, overflow: pathOverflow}
      });
      interaction.add(extraction);

      // Extraction orders ids source to target, so rank = position + 1 puts A at rank 1.
      const ranks = importInput('path-ranks', this.pathRanks, 'uint32', nodeCount);
      interaction.add(
        createAnalysisKernelNode(interaction, {
          id: `${id}-path-ranks-clear`,
          bindings: [{name: 'ranks', view: ranks, type: 'u32', access: 'read_write'}],
          invocationCount: nodeCount,
          body: 'ranks[ranksOffset + index] = 0u;'
        })
      );
      interaction.add(
        createAnalysisKernelNode(interaction, {
          id: `${id}-path-ranks-scatter`,
          bindings: [
            {name: 'ids', view: pathIds, type: 'u32', access: 'read'},
            {name: 'pathCount', view: pathCount, type: 'u32', access: 'read'},
            {name: 'ranks', view: ranks, type: 'u32', access: 'read_write'}
          ],
          invocationCount: pathCapacity,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;`,
          body: `if (index < pathCount[pathCountOffset]) {
    let node = ids[idsOffset + index];
    if (node < NODE_COUNT) {
      ranks[ranksOffset + node] = index + 1u;
    }
  }`
        })
      );
      this.interactionGraph = interaction.compile();
    } catch (error) {
      this.destroyPartial();
      throw error;
    }
    this.writeHover();
    this.writePath();
  }

  /** True once topology and every analytics stage have been encoded. */
  get isAnalyticsComplete(): boolean {
    return this.completedStages >= this.analyticsGraphs.length + 1;
  }

  /** The hovered vertex last passed to {@link setHoverVertex}. */
  get currentHoverVertex(): number | null {
    return this.hoveredVertex;
  }

  /** Hop radius of {@link neighborhoodMask}. */
  get currentNeighborhoodHops(): number {
    return this.neighborhoodHops;
  }

  /** Path source A and target B last passed to {@link setPathEndpoints}. */
  get currentPathEndpoints(): readonly [number | null, number | null] {
    return [this.pathSource, this.pathTarget];
  }

  /** Immediately available measurements; reads nothing from the GPU. */
  get stats(): GPUGraphAnalysisColumnsStats {
    return {
      residentBufferBytes: this.buffers.reduce((total, buffer) => total + buffer.byteLength, 0),
      transientBufferBytes: [
        this.topologyGraph,
        ...this.analyticsGraphs,
        this.interactionGraph
      ].reduce((total, graph) => total + graph.stats.physicalTransientResourceBytes, 0),
      completedStages: this.completedStages,
      totalStages: this.analyticsGraphs.length + 1,
      analyticsEncodeMilliseconds: this.analyticsEncodeMilliseconds,
      interactionEncodeMilliseconds: this.interactionEncodeMilliseconds,
      interactionEncodeCount: this.interactionEncodeCount
    };
  }

  /**
   * Sets the hovered vertex whose `hops`-neighborhood fills {@link neighborhoodMask}; `null` clears
   * it. Writes one tiny buffer and marks the interaction graph dirty; invalid vertices are ignored.
   */
  setHoverVertex(vertex: number | null): void {
    if (vertex !== null && !this.isValidVertex(vertex)) return;
    this.hoveredVertex = vertex;
    this.writeHover();
    this.interactionDirty = true;
  }

  /** Sets the neighborhood radius, rounded and clamped to `[0, 8]`. */
  setNeighborhoodHops(hops: number): void {
    this.neighborhoodHops = Math.max(0, Math.min(MAXIMUM_NEIGHBORHOOD_HOPS, Math.round(hops)));
    this.hops.write(Uint32Array.of(this.neighborhoodHops));
    this.interactionDirty = true;
  }

  /**
   * Sets the shortest-path endpoints. Either being `null` (or invalid) yields all-zero
   * {@link pathRanks} and an all-`NONE` {@link reachabilityBands}.
   */
  setPathEndpoints(source: number | null, target: number | null): void {
    this.pathSource = source !== null && this.isValidVertex(source) ? source : null;
    this.pathTarget = target !== null && this.isValidVertex(target) ? target : null;
    this.writePath();
    this.interactionDirty = true;
  }

  /**
   * Appends the next analytics stage(s) to `commandEncoder`: topology first, then every metric.
   * Non-massive graphs encode everything in the first call; massive graphs encode one stage per
   * call. Returns whether anything was encoded.
   */
  encodeAnalytics(commandEncoder: CommandEncoder): boolean {
    if (this.destroyed) return false;
    const stages = [this.topologyGraph, ...this.analyticsGraphs];
    let encoded = 0;
    while (this.completedStages < stages.length && encoded < this.stagesPerCall) {
      const encoding = stages[this.completedStages].encode(commandEncoder, {
        parameters: undefined
      });
      this.analyticsEncodeMilliseconds += encoding.stats.cpuEncodeTimeMilliseconds;
      this.completedStages++;
      encoded++;
    }
    return encoded > 0;
  }

  /**
   * Appends neighborhood, reachability, bands, path extraction and rank scatter to
   * `commandEncoder` only when hover, hops or path endpoints changed and topology is built.
   * Returns whether anything was encoded.
   */
  encodeInteraction(commandEncoder: CommandEncoder): boolean {
    if (this.destroyed || !this.interactionDirty || this.completedStages < 1) return false;
    const encoding = this.interactionGraph.encode(commandEncoder, {
      parameters: undefined
    });
    this.interactionEncodeMilliseconds = encoding.stats.cpuEncodeTimeMilliseconds;
    this.interactionEncodeCount++;
    this.interactionDirty = false;
    return true;
  }

  /** Destroys compiled graphs, contributors and every buffer, including the published outputs. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.destroyPartial();
  }

  private destroyPartial(): void {
    this.topologyGraph?.destroy();
    for (const graph of this.analyticsGraphs ?? []) graph.destroy();
    this.interactionGraph?.destroy();
    for (const contributor of this.contributors) contributor.destroy?.();
    for (const vector of this.vectors.reverse()) vector.destroy();
    for (const buffer of this.buffers.reverse()) buffer.destroy();
  }

  private isValidVertex(vertex: number): boolean {
    return Number.isSafeInteger(vertex) && vertex >= 0 && vertex < this.vertexCount;
  }

  private writeHover(): void {
    this.seeds.write(Uint32Array.of(this.hoveredVertex ?? 0));
    this.seedCount.write(Uint32Array.of(this.hoveredVertex === null ? 0 : 1));
  }

  private writePath(): void {
    this.pathSources.write(Uint32Array.of(this.pathSource ?? 0));
    this.pathSourceCount.write(Uint32Array.of(this.pathSource === null ? 0 : 1));
    this.pathTargets.write(Uint32Array.of(this.pathTarget ?? 0));
    this.pathTargetCount.write(
      Uint32Array.of(this.pathSource === null || this.pathTarget === null ? 0 : 1)
    );
  }

  private createBuffer(id: string, length: number): Buffer {
    const buffer = this.device.createBuffer({
      id,
      byteLength: Math.max(length, 1) * SCALAR_BYTE_LENGTH,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    return buffer;
  }

  private createVector(id: string, length: number, buffer?: Buffer): GPUVector<'uint32'> {
    const vector = new GPUVector<'uint32'>({
      type: 'buffer',
      name: id,
      format: 'uint32',
      buffer: buffer ?? this.createBuffer(id, length),
      length,
      ownsBuffer: false
    });
    this.vectors.push(vector);
    return vector;
  }

  private createAdjacency(
    id: string,
    offsets: Buffer,
    neighbors: Buffer,
    nodeCount: number,
    capacity: number
  ): GPUGraphAdjacency {
    return {
      offsets: this.createVector(`${id}-offsets`, nodeCount + 1, offsets),
      neighbors: this.createVector(`${id}-neighbors`, capacity, neighbors),
      edgeIds: this.createVector(`${id}-edge-ids`, capacity),
      count: this.createVector(`${id}-count`, 1),
      overflow: this.createVector(`${id}-overflow`, 1)
    };
  }
}

type AnalysisKernelProps = {
  id: string;
  bindings: readonly {
    name: string;
    view: GraphDataView;
    type: 'u32' | 'f32';
    access: 'read' | 'read_write';
  }[];
  invocationCount: number;
  declarations?: string;
  body: string;
};

/**
 * One linear compute pass over packed storage views, following the linear WGSL kernel pattern.
 * Bindings expose `${name}Offset` element offsets; `index` is the guarded invocation index.
 */
function createAnalysisKernelNode(
  graph: GPUCommandGraph<void>,
  props: AnalysisKernelProps
): GPUCommandNode<void> {
  const layout = getBoundedDispatchLayout(
    props.id,
    Math.max(props.invocationCount, 1),
    WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const declarations = props.bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const bindingReferences = props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ');
  const source = /* wgsl */ `
const INVOCATION_COUNT: u32 = ${props.invocationCount}u;
${declarations}
${props.declarations ?? ''}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${bindingReferences}
  ${getBoundedInvocationIndexSource(layout, WORKGROUP_SIZE)}
  if (index >= INVOCATION_COUNT) {
    return;
  }
  ${props.body}
}`;
  let readByteLength = 0;
  let writeByteLength = 0;
  for (const binding of props.bindings) {
    const byteLength = binding.view.length * binding.view.rowByteLength;
    if (binding.access === 'read') readByteLength += byteLength;
    else writeByteLength += byteLength;
  }
  return createGPUComputeCommandNode<void>({
    id: props.id,
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
    })),
    workload: {
      operation: 'GPUGraphAnalysisColumns',
      commandCount: 1,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      maximumInvocationCount: props.invocationCount,
      readByteLength,
      writeByteLength
    },
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: props.bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          computation.setBindings(
            Object.fromEntries(
              props.bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
            )
          );
          computation.dispatch(computePass, layout.x, layout.y, layout.z);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}
