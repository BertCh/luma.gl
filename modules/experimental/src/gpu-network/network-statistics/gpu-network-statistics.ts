// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGraphConnectedComponents, GPUGraphTopologyView} from '@luma.gl/gpgpu/gpu-graph';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {createLoadBalancedSegmentSumNodes} from '../../utils/load-balanced-for-each';
import type {WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkAdjacency
} from '../../analysis-core/network-topology';
import {
  getWorkgroupCounterKernelSource,
  WORKGROUP_COUNTER_WORKGROUP_SIZE
} from '../network-workgroup-counters';

/** Number of header words that precede the three degree histograms in the summary. */
export const GPU_NETWORK_STATISTICS_HEADER_LENGTH = 16;

/** Word indices of the summary header. */
export const GPU_NETWORK_STATISTICS_WORD = {
  /** Nodes that are live (`nodeMask` nonzero, or every node without a mask). */
  liveNodeCount: 0,
  /** Directed: `liveSlotCount`. Undirected: live non-self-loop slots / 2 plus live self-loop slots. */
  liveEdgeCount: 1,
  /** CSR slots that are live. */
  liveSlotCount: 2,
  /** Weak components among live nodes. */
  componentCount: 3,
  /** Node count of the largest live weak component. */
  largestComponentSize: 4,
  /** 1 when the component labelling reached a fixed point within `maximumComponentIterations`. */
  componentsConverged: 5,
  /** Live nodes with zero live degree. */
  isolatedNodeCount: 6,
  /** Largest live out-degree. */
  maximumOutDegree: 7,
  /** Largest live in-degree. */
  maximumInDegree: 8,
  /** Largest live total degree. */
  maximumTotalDegree: 9,
  /** Modularity as `float32` bits; 0 when invalid. */
  modularity: 10,
  /** 1 when `modularity` is meaningful. */
  modularityValid: 11,
  /** Live slots whose endpoints share a community label. */
  intraCommunitySlotCount: 12,
  /** Live slots u -> u. */
  selfLoopSlotCount: 13
} as const;

/** Number of words of the optional parameter view. */
export const GPU_NETWORK_STATISTICS_PARAMETER_LENGTH = 2;

/** Returns the exact row count of the `output` view for a histogram resolution. */
export function getGPUNetworkStatisticsLength(degreeBinCount: number = 32): number {
  return GPU_NETWORK_STATISTICS_HEADER_LENGTH + 3 * degreeBinCount;
}

/** Per-frame settings of {@link GPUNetworkStatistics}. */
export type GPUNetworkStatisticsSettings = {
  /** Modularity resolution `gamma`. Defaults to 1. */
  resolution?: number;
  /** Linear degree bin width, floored and clamped to at least 1. Defaults to 1. */
  degreeBinWidth?: number;
};

/**
 * Packs the per-frame parameter words: word 0 is the modularity resolution as `float32` bits,
 * word 1 is the linear degree bin width (clamped to at least 1 on the GPU).
 *
 * Write the result into a `'uint32'` view of two words, for example with
 * `GPUParameterBuffer.write`.
 */
export function getGPUNetworkStatisticsParameterValues(
  settings: GPUNetworkStatisticsSettings = {}
): Uint32Array {
  const words = new Uint32Array(GPU_NETWORK_STATISTICS_PARAMETER_LENGTH);
  new Float32Array(words.buffer, 0, 1)[0] = settings.resolution ?? 1;
  words[1] = Math.max(1, Math.floor(settings.degreeBinWidth ?? 1));
  return words;
}

/** Histogram layout needed to decode a summary. */
export type GPUNetworkStatisticsLayout = {
  /** Bins per histogram. Defaults to 32. */
  degreeBinCount?: number;
};

/** Decoded summary. */
export type GPUNetworkStatisticsResult = {
  liveNodeCount: number;
  liveEdgeCount: number;
  liveSlotCount: number;
  componentCount: number;
  largestComponentSize: number;
  componentsConverged: boolean;
  isolatedNodeCount: number;
  maximumOutDegree: number;
  maximumInDegree: number;
  maximumTotalDegree: number;
  /** 0 when `modularityValid` is false. */
  modularity: number;
  modularityValid: boolean;
  intraCommunitySlotCount: number;
  selfLoopSlotCount: number;
  /** Live-node counts per out-degree bin. */
  outDegreeHistogram: number[];
  /** Live-node counts per in-degree bin. */
  inDegreeHistogram: number[];
  /** Live-node counts per total-degree bin. */
  totalDegreeHistogram: number[];
};

/** Decodes a summary read back from the `output` view. */
export function decodeGPUNetworkStatistics(
  words: Uint32Array | ArrayLike<number>,
  layout: GPUNetworkStatisticsLayout = {}
): GPUNetworkStatisticsResult {
  const binCount = layout.degreeBinCount ?? 32;
  if (words.length < getGPUNetworkStatisticsLength(binCount)) {
    throw new Error('decodeGPUNetworkStatistics needs the full summary');
  }
  const word = GPU_NETWORK_STATISTICS_WORD;
  const histogram = (index: number) =>
    Array.from(
      {length: binCount},
      (_, bin) => words[GPU_NETWORK_STATISTICS_HEADER_LENGTH + index * binCount + bin]
    );
  const bits = new Uint32Array([words[word.modularity]]);
  return {
    liveNodeCount: words[word.liveNodeCount],
    liveEdgeCount: words[word.liveEdgeCount],
    liveSlotCount: words[word.liveSlotCount],
    componentCount: words[word.componentCount],
    largestComponentSize: words[word.largestComponentSize],
    componentsConverged: words[word.componentsConverged] !== 0,
    isolatedNodeCount: words[word.isolatedNodeCount],
    maximumOutDegree: words[word.maximumOutDegree],
    maximumInDegree: words[word.maximumInDegree],
    maximumTotalDegree: words[word.maximumTotalDegree],
    modularity: new Float32Array(bits.buffer)[0],
    modularityValid: words[word.modularityValid] !== 0,
    intraCommunitySlotCount: words[word.intraCommunitySlotCount],
    selfLoopSlotCount: words[word.selfLoopSlotCount],
    outDegreeHistogram: histogram(0),
    inDegreeHistogram: histogram(1),
    totalDegreeHistogram: histogram(2)
  };
}

/**
 * Properties for {@link GPUNetworkStatistics}.
 *
 * Compile-time: node count, `directed`, which optional views exist, `degreeBinCount`,
 * `degreeBinning` and `maximumComponentIterations`. Per-frame: the contents of the CSR, both masks, the
 * community labels and the parameter view, so filtering a live network changes the statistics on
 * the next encoding without recompiling.
 */
export type GPUNetworkStatisticsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-statistics'`. */
  id?: string;
  /**
   * Forward CSR network; `weights` are not read. Neighbor indices `>= nodeCount` are dead slots.
   * Undirected graphs list each edge in both directions (a self-loop once), like
   * `GPUNetworkAnalyticsColumns`.
   */
  topology: GPUNetworkAdjacency;
  /**
   * Treat slots as directed edges. Default false. Undirected, a self-loop stored once in the CSR
   * adds 2 to its node's degree (in the degree histograms, maxima and isolated count) and to the
   * modularity volumes, as in networkx and `GPUGraphModularity`; `liveSlotCount`, `liveEdgeCount`
   * and `selfLoopSlotCount` count the stored slot once.
   */
  directed?: boolean;
  /** Optional per-node mask, nonzero = live. */
  nodeMask?: GraphDataView<'uint32'>;
  /**
   * Optional per-slot mask aligned with `topology.neighbors`, nonzero = live. A slot u -> v is live iff
   * this mask (if any) is nonzero, both u and v are live and `v < nodeCount`.
   */
  edgeMask?: GraphDataView<'uint32'>;
  /**
   * Optional community label per node (labels must be `< nodeCount`; a live node with a larger
   * label makes modularity invalid). Enables modularity.
   */
  communities?: GraphDataView<'uint32'>;
  /**
   * Optional per-frame `'uint32'` view of at least two words, see
   * {@link getGPUNetworkStatisticsParameterValues}: word 0 modularity resolution (`float32` bits,
   * default 1), word 1 linear degree bin width (default 1).
   */
  parameters?: GraphDataView<'uint32'>;
  /** Bins per histogram. Compile-time, default 32. */
  degreeBinCount?: number;
  /**
   * Compile-time bin rule, default `'linear'`. Linear: `min(floor(d / binWidth), binCount - 1)`.
   * Log2: bin 0 for d = 0, else `min(1 + floor(log2(d)), binCount - 1)`.
   */
  degreeBinning?: 'linear' | 'log2';
  /**
   * Cap on the component relaxation iterations, compile-time, default 32. The labelling may stop
   * earlier; `componentsConverged` reports whether it reached a fixed point.
   */
  maximumComponentIterations?: number;
  /** Packed `uint32` summary with exactly {@link getGPUNetworkStatisticsLength} rows. */
  output: GraphDataView<'uint32'>;
};

const WORKGROUP_SIZE = 256;

/** Largest `3 * degreeBinCount` kept in workgroup memory by the degree pass (4 KiB). */
const MAXIMUM_PRIVATE_HISTOGRAM_WORDS = 1024;

/**
 * Reduces a CSR network to a compact statistics summary for a graph stats panel: live
 * node/edge/slot counts, weak component count and largest size, isolated nodes, maximum
 * degrees, three degree histograms and optional modularity, all in one caller-owned `uint32` view.
 * Every word is rewritten on every encoding; the CPU decodes it with
 * {@link decodeGPUNetworkStatistics}.
 *
 * Summary layout: words `0..15` are the header ({@link GPU_NETWORK_STATISTICS_WORD}, words 14 and
 * 15 reserved zeros), then out-degree, in-degree and total-degree histograms of `degreeBinCount`
 * bins each. Undirected: in = out = total = live row length, with a self-loop adding 2 (networkx).
 * Directed: total = in + out. Degree statistics cover live nodes only.
 *
 * Degrees are not computed with `GPUGraphDegree`: that algorithm subtracts CSR offsets, which
 * counts every stored slot, while statistics need the live degree after per-node and per-slot
 * masks (a masked CSR would need new offsets from a prefix sum), and the directed in-degree is a
 * scatter over the live slots rather than a reverse CSR. The live row length, in-degree and slot
 * counts therefore come from the same pass that builds the masked neighbors.
 *
 * Components run `GPUGraphConnectedComponents` over a `GPUGraphTopologyView` of a graph-owned masked
 * copy of the neighbors (dead slots become `0xffffffff`), so masked nodes are isolated and excluded
 * from counts.
 *
 * Modularity uses exact `u32` atomic sums (intra-community slots and per-label degree sums) and a
 * two-stage f32 finish (per-workgroup partials, then one workgroup over the partials) in a fixed
 * order, so it is deterministic. `GPUGraphModularity` accumulates floats with atomics and is
 * therefore order dependent; it is deliberately not used. Undirected:
 * `Q = intra/2m - gamma * sum_c (d_c/2m)^2` with `2m = liveSlotCount + selfLoopSlotCount` and
 * `intra` counting each self-loop twice (networkx `modularity`); directed:
 * `Q = intra/m - gamma * sum_c out_c in_c / m^2` with `m = liveSlotCount`. Counts are `u32`, so a
 * graph with 2^32 or more live slots overflows. Invalid (no live slots, no communities or an
 * out-of-range label) gives modularity 0 and `modularityValid` 0.
 *
 * The slot loops (masking, live and self-loop counts, in-degree, intra-community count) are
 * load-balanced CSR loops, so a hub row with 1e5 slots spreads over many workgroups instead of
 * serializing one thread. They require non-decreasing `topology.offsets`: for such offsets the
 * result equals the per-row loop, and decreasing offsets give unspecified counts (no
 * out-of-bounds access).
 *
 * The contributor owns no GPU resources; the masked neighbors, labels and status words are graph
 * transients.
 */
export class GPUNetworkStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkStatisticsProps;
  /** Number of nodes, `topology.offsets.length - 1`. */
  readonly nodeCount: number;
  /** Bins per histogram. */
  readonly degreeBinCount: number;

  constructor(props: GPUNetworkStatisticsProps) {
    this.id = props.id ?? 'network-statistics';
    this.props = props;
    const {id} = this;
    this.nodeCount = validateGPUNetworkTopology(id, props.topology).nodeCount;
    const {nodeCount} = this;
    this.degreeBinCount = props.degreeBinCount ?? 32;
    validateCapacity(id, 'degreeBinCount', this.degreeBinCount, 1, 4096);
    if (props.degreeBinning && props.degreeBinning !== 'linear' && props.degreeBinning !== 'log2') {
      throw new Error(`${id} degreeBinning must be 'linear' or 'log2'`);
    }
    validateCapacity(
      id,
      'maximumComponentIterations',
      props.maximumComponentIterations ?? 32,
      1,
      1024
    );
    for (const [name, view, length] of [
      ['nodeMask', props.nodeMask, nodeCount],
      ['communities', props.communities, nodeCount],
      ['edgeMask', props.edgeMask, props.topology.neighbors.length]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['uint32'], `${id} ${name}`);
      if (view.length !== length) {
        throw new Error(
          `${id} ${name} must contain one row per ${name === 'edgeMask' ? 'neighbor slot' : 'node'}`
        );
      }
    }
    if (props.parameters) {
      validatePackedView(props.parameters, ['uint32'], `${id} parameters`);
      if (props.parameters.length < GPU_NETWORK_STATISTICS_PARAMETER_LENGTH) {
        throw new Error(`${id} parameters must contain at least two uint32 rows`);
      }
    }
    validatePackedView(props.output, ['uint32'], `${id} output`);
    if (props.output.length !== getGPUNetworkStatisticsLength(this.degreeBinCount)) {
      throw new Error(
        `${id} output must contain exactly ${getGPUNetworkStatisticsLength(this.degreeBinCount)} rows`
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output],
      [
        ...getGPUNetworkTopologyViews(props.topology),
        props.nodeMask,
        props.edgeMask,
        props.communities,
        props.parameters
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, degreeBinCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUNetworkTopologyViews(props.topology),
      props.nodeMask,
      props.edgeMask,
      props.communities,
      props.parameters,
      props.output
    ]);
    const directed = Boolean(props.directed);
    // Undirected, a self-loop adds 2 to a degree (networkx; review B-15).
    const countTwice = !directed;
    const hasCommunities = Boolean(props.communities);
    const word = GPU_NETWORK_STATISTICS_WORD;
    const operation = 'GPUNetworkStatistics';
    const {offsets, neighbors} = props.topology;
    const masked = createTransientView(graph, `${id}-masked-neighbors`, 'uint32', neighbors.length);
    const labels = createTransientView(graph, `${id}-component-labels`, 'uint32', nodeCount);
    const converged = createTransientView(graph, `${id}-component-converged`, 'uint32', 1);
    const transient = (name: string) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', nodeCount);
    const outDegree = transient('out-degree');
    const inDegree = directed ? transient('in-degree') : undefined;
    const componentSize = transient('component-size');
    const outSum = hasCommunities ? transient('community-out-sum') : undefined;
    const inSum = hasCommunities && directed ? transient('community-in-sum') : undefined;
    const {output} = props;

    const maskBindings: WGSLKernelBinding[] = [];
    if (props.nodeMask) {
      maskBindings.push({
        name: 'vertexMask',
        view: props.nodeMask,
        type: 'u32',
        access: 'read'
      });
    }
    const isLive = (vertex: string) =>
      props.nodeMask ? `vertexMask[vertexMaskOffset + ${vertex}] != 0u` : 'true';
    const nodes: GPUCommandNode<Parameters>[] = [];
    const fill = (step: string, view: GraphDataView<'uint32'> | undefined) => {
      if (view) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${step}`,
            operation,
            view,
            type: 'u32',
            value: '0u'
          })
        );
      }
    };
    fill('zero-output', output);
    fill('zero-in-degree', inDegree);
    fill('zero-component-size', componentSize);
    fill('zero-out-sum', outSum);
    fill('zero-in-sum', inSum);

    // Pass 1: mask slots, live and self-loop counts per row, in-degree. A load-balanced segment
    // sum (algorithms roadmap decision 3) spreads a hub row over many workgroups instead of one
    // thread. The contribution of a slot also writes the masked neighbor and counts the in-degree:
    // the walk calls it exactly once per slot. Counts are integers, so the result does not depend
    // on the association. Offsets must be non-decreasing; the loop visits the slots
    // `[offsets[0], offsets[nodeCount])` clamped to the neighbor capacity, which is what the old
    // per-row `min(offsets, capacity)` clamp visited for a valid CSR. Slots outside every row are
    // not written, and decreasing offsets give unspecified (but memory safe) counts.
    // Two u32 segment sums, so each output binding holds one word per node (a vec2 would need
    // 8 bytes per node and overflow one binding at the node limit): the live row length (a
    // self-loop counts twice undirected) straight into `outDegree`, then the live self-loops per
    // row from the masked copy.
    const selfLoopCounts = createTransientView(
      graph,
      `${id}-self-loop-counts`,
      'uint32',
      nodeCount
    );
    const passOneBindings: WGSLKernelBinding[] = [
      {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
      ...maskBindings,
      {name: 'maskedOut', view: masked, type: 'u32', access: 'read_write'}
    ];
    if (props.edgeMask) {
      passOneBindings.push({
        name: 'edgeMask',
        view: props.edgeMask,
        type: 'u32',
        access: 'read'
      });
    }
    if (inDegree) {
      passOneBindings.push({
        name: 'inDegree',
        view: inDegree,
        type: 'atomic<u32>',
        access: 'read_write'
      });
    }
    nodes.push(
      ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-slots`,
        operation,
        variant: 'slots',
        offsets,
        itemCapacity: neighbors.length,
        accumulatorType: 'u32',
        output: outDegree,
        bindings: passOneBindings,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  let neighbor = neighbors[neighborsOffset + item];
  var slotLive = ${isLive('row')} && neighbor < NODE_COUNT;
  ${props.nodeMask ? 'if (slotLive) { slotLive = vertexMask[vertexMaskOffset + neighbor] != 0u; }' : ''}
  ${props.edgeMask ? 'if (slotLive) { slotLive = edgeMask[edgeMaskOffset + item] != 0u; }' : ''}
  maskedOut[maskedOutOffset + item] = select(0xffffffffu, neighbor, slotLive);
  if (!slotLive) {
    return 0u;
  }
  ${inDegree ? 'atomicAdd(&inDegree[inDegreeOffset + neighbor], 1u);' : ''}
  return ${countTwice ? 'select(1u, 2u, neighbor == row)' : '1u'};
}`
      }),
      ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-self-loops`,
        operation,
        variant: 'self-loops',
        offsets,
        itemCapacity: neighbors.length,
        accumulatorType: 'u32',
        output: selfLoopCounts,
        bindings: [{name: 'maskedIn', view: masked, type: 'u32', access: 'read'}],
        declarations: `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  return select(0u, 1u, maskedIn[maskedInOffset + item] == row);
}`
      })
    );
    // The two summary words. Slot counts fold in workgroup memory so each word takes one global
    // atomic per workgroup (rule 14).
    const slotCounters = getWorkgroupCounterKernelSource({
      counters: [
        {target: 'summary', word: word.liveSlotCount},
        {target: 'summary', word: word.selfLoopSlotCount}
      ],
      rowCount: 'NODE_COUNT',
      rowSource: `let selfCount = selfLoopCounts[selfLoopCountsOffset + index];
  addCounter(0u, outDegree[outDegreeOffset + index]${countTwice ? ' - selfCount' : ''});
  addCounter(1u, selfCount);`
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slots-finalize`,
        operation,
        variant: 'slots-finalize',
        bindings: [
          {name: 'selfLoopCounts', view: selfLoopCounts, type: 'u32', access: 'read'},
          {name: 'outDegree', view: outDegree, type: 'u32', access: 'read'},
          {name: 'summary', view: output, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: nodeCount,
        workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
${slotCounters.declarations}`,
        body: slotCounters.body
      })
    );

    // Weak components over the masked adjacency.
    nodes.push(
      ...captureGraphCommandNodes(graph, () => {
        const topology = new GPUGraphTopologyView({
          id: `${id}-topology`,
          vertexCount: nodeCount,
          forward: {offsets, neighbors: masked}
        });
        new GPUGraphConnectedComponents({
          id: `${id}-components`,
          topology,
          output: labels,
          iterations: props.maximumComponentIterations,
          converged
        }).addToGraph(graph);
      })
    );

    // Pass 2: live vertices and component sizes.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-component-sizes`,
        operation,
        variant: 'component-sizes',
        bindings: [
          ...maskBindings,
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {
            name: 'componentSize',
            view: componentSize,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: nodeCount,
        body: `if (!(${isLive('index')})) {
    return;
  }
  atomicAdd(&componentSize[componentSizeOffset + labels[labelsOffset + index]], 1u);`
      })
    );

    // Pass 3: degree maxima, histograms, isolated vertices and per-community degree sums.
    const binWidth = props.parameters ? 'max(parameters[parametersOffset + 1u], 1u)' : '1u';
    const binSource =
      props.degreeBinning === 'log2'
        ? `fn getBin(degree: u32) -> u32 {
  if (degree == 0u) {
    return 0u;
  }
  return min(1u + firstLeadingBit(degree), BIN_COUNT - 1u);
}`
        : `fn getBin(degree: u32) -> u32 {
  return min(degree / ${binWidth}, BIN_COUNT - 1u);
}`;
    const degreeBindings: WGSLKernelBinding[] = [
      ...maskBindings,
      {name: 'outDegree', view: outDegree, type: 'u32', access: 'read'}
    ];
    if (inDegree) {
      degreeBindings.push({
        name: 'inDegree',
        view: inDegree,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.communities) {
      degreeBindings.push({
        name: 'communities',
        view: props.communities,
        type: 'u32',
        access: 'read'
      });
    }
    if (outSum) {
      degreeBindings.push({
        name: 'outSum',
        view: outSum,
        type: 'atomic<u32>',
        access: 'read_write'
      });
    }
    if (inSum) {
      degreeBindings.push({
        name: 'inSum',
        view: inSum,
        type: 'atomic<u32>',
        access: 'read_write'
      });
    }
    if (props.parameters && props.degreeBinning !== 'log2') {
      degreeBindings.push({
        name: 'parameters',
        view: props.parameters,
        type: 'u32',
        access: 'read'
      });
    }
    degreeBindings.push({
      name: 'summary',
      view: output,
      type: 'atomic<u32>',
      access: 'read_write'
    });
    const privatizeDegrees = 3 * degreeBinCount <= MAXIMUM_PRIVATE_HISTOGRAM_WORDS;
    const degreeDeclarations = `const NODE_COUNT: u32 = ${nodeCount}u;
const BIN_COUNT: u32 = ${degreeBinCount}u;
const HISTOGRAM_BASE: u32 = ${GPU_NETWORK_STATISTICS_HEADER_LENGTH}u;
${binSource}`;
    // Per-community degree sums stay global: they spread over one address per label.
    const communityStatements = hasCommunities
      ? `let label = communities[communitiesOffset + index];
  if (label >= NODE_COUNT) {
    atomicAdd(&summary[summaryOffset + ${word.modularityValid}u], 1u);
  } else {
    atomicAdd(&outSum[outSumOffset + label], outValue);
    ${inSum ? 'atomicAdd(&inSum[inSumOffset + label], inValue);' : ''}
  }`
      : '';
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-degrees`,
        operation,
        variant: 'degrees',
        record: {datum: {keySpace: 'node'}, formula: 'gpu-network.statistics.degree-histogram'},
        bindings: degreeBindings,
        invocationCount: nodeCount,
        // Privatized: each workgroup folds its vertices into workgroup-memory histograms, maxima
        // and counters, then flushes once, so the ~10 summary words take one global atomic per
        // bin per workgroup instead of up to ten per vertex on the same addresses. It needs
        // 3 * binCount + 5 words of workgroup memory, so wide histograms keep the global path.
        ...(privatizeDegrees
          ? {
              workgroupSize: WORKGROUP_SIZE,
              guardIndex: false,
              declarations: `${degreeDeclarations}
var<workgroup> localHistogram: array<atomic<u32>, ${3 * degreeBinCount}>;
var<workgroup> localMaxima: array<atomic<u32>, 3>;
var<workgroup> localCounts: array<atomic<u32>, 2>;`,
              body: `for (var bin = localInvocationIndex; bin < ${3 * degreeBinCount}u; bin += ${WORKGROUP_SIZE}u) {
    atomicStore(&localHistogram[bin], 0u);
  }
  if (localInvocationIndex < 3u) {
    atomicStore(&localMaxima[localInvocationIndex], 0u);
  }
  if (localInvocationIndex < 2u) {
    atomicStore(&localCounts[localInvocationIndex], 0u);
  }
  workgroupBarrier();
  if (index < NODE_COUNT && ${isLive('index')}) {
    let outValue = outDegree[outDegreeOffset + index];
    let inValue = ${inDegree ? 'inDegree[inDegreeOffset + index]' : 'outValue'};
    let totalValue = ${directed ? 'outValue + inValue' : 'outValue'};
    atomicAdd(&localCounts[1], 1u);
    if (totalValue == 0u) {
      atomicAdd(&localCounts[0], 1u);
    }
    atomicMax(&localMaxima[0], outValue);
    atomicMax(&localMaxima[1], inValue);
    atomicMax(&localMaxima[2], totalValue);
    atomicAdd(&localHistogram[getBin(outValue)], 1u);
    atomicAdd(&localHistogram[BIN_COUNT + getBin(inValue)], 1u);
    atomicAdd(&localHistogram[2u * BIN_COUNT + getBin(totalValue)], 1u);
    ${communityStatements}
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < ${3 * degreeBinCount}u; bin += ${WORKGROUP_SIZE}u) {
    let binCount = atomicLoad(&localHistogram[bin]);
    if (binCount > 0u) {
      atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + bin], binCount);
    }
  }
  if (localInvocationIndex < 3u) {
    let workgroupMaximum = atomicLoad(&localMaxima[localInvocationIndex]);
    if (workgroupMaximum > 0u) {
      atomicMax(&summary[summaryOffset + ${word.maximumOutDegree}u + localInvocationIndex], workgroupMaximum);
    }
  }
  if (localInvocationIndex == 3u) {
    let workgroupIsolated = atomicLoad(&localCounts[0]);
    if (workgroupIsolated > 0u) {
      atomicAdd(&summary[summaryOffset + ${word.isolatedNodeCount}u], workgroupIsolated);
    }
    let workgroupLive = atomicLoad(&localCounts[1]);
    if (workgroupLive > 0u) {
      atomicAdd(&summary[summaryOffset + ${word.liveNodeCount}u], workgroupLive);
    }
  }`
            }
          : {
              declarations: degreeDeclarations,
              body: `if (!(${isLive('index')})) {
    return;
  }
  let outValue = outDegree[outDegreeOffset + index];
  let inValue = ${inDegree ? 'inDegree[inDegreeOffset + index]' : 'outValue'};
  let totalValue = ${directed ? 'outValue + inValue' : 'outValue'};
  atomicAdd(&summary[summaryOffset + ${word.liveNodeCount}u], 1u);
  if (totalValue == 0u) {
    atomicAdd(&summary[summaryOffset + ${word.isolatedNodeCount}u], 1u);
  }
  atomicMax(&summary[summaryOffset + ${word.maximumOutDegree}u], outValue);
  atomicMax(&summary[summaryOffset + ${word.maximumInDegree}u], inValue);
  atomicMax(&summary[summaryOffset + ${word.maximumTotalDegree}u], totalValue);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + getBin(outValue)], 1u);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + BIN_COUNT + getBin(inValue)], 1u);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + 2u * BIN_COUNT + getBin(totalValue)], 1u);
  ${communityStatements}`
            })
      })
    );

    // Pass 4: intra-community live slots. A load-balanced u32 segment sum counts them per row
    // (hub rows spread over many workgroups), then the row counts fold per workgroup into one
    // atomic (rule 14). Same offsets requirement as pass 1.
    if (props.communities) {
      const intraSums = createTransientView(graph, `${id}-intra-sums`, 'uint32', nodeCount);
      nodes.push(
        ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
          id: `${id}-intra-community`,
          operation,
          variant: 'intra-community',
          offsets,
          itemCapacity: neighbors.length,
          accumulatorType: 'u32',
          output: intraSums,
          bindings: [
            {name: 'maskedIn', view: masked, type: 'u32', access: 'read'},
            {name: 'communities', view: props.communities, type: 'u32', access: 'read'}
          ],
          declarations: `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  let neighbor = maskedIn[maskedInOffset + item];
  return select(
    0u,
    1u,
    neighbor != 0xffffffffu &&
      communities[communitiesOffset + neighbor] == communities[communitiesOffset + row]
  );
}`
        })
      );
      const intraCounters = getWorkgroupCounterKernelSource({
        counters: [{target: 'summary', word: word.intraCommunitySlotCount}],
        rowCount: 'NODE_COUNT',
        rowSource: 'addCounter(0u, intraSums[intraSumsOffset + index]);'
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-intra-community-total`,
          operation,
          variant: 'intra-community-total',
          bindings: [
            {name: 'intraSums', view: intraSums, type: 'u32', access: 'read'},
            {name: 'summary', view: output, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: nodeCount,
          workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
${intraCounters.declarations}`,
          body: intraCounters.body
        })
      );
    }

    // Pass 5: two-stage reduction of component sizes and community terms (rule 14). Stage 1 folds
    // 256 labels per workgroup into a partial (count, largest, term sum as f32 bits); stage 2 is
    // one workgroup over the partials that also finishes the header.
    const partialCount = Math.ceil(nodeCount / WORKGROUP_SIZE);
    const partials = createTransientView(
      graph,
      `${id}-finish-partials`,
      'uint32',
      3 * partialCount
    );
    const partialBindings: WGSLKernelBinding[] = [
      {
        name: 'componentSize',
        view: componentSize,
        type: 'u32',
        access: 'read'
      }
    ];
    if (outSum) {
      partialBindings.push({name: 'outSum', view: outSum, type: 'u32', access: 'read'});
    }
    if (inSum) {
      partialBindings.push({name: 'inSum', view: inSum, type: 'u32', access: 'read'});
    }
    partialBindings.push(
      {name: 'summaryIn', view: output, type: 'u32', access: 'read'},
      {name: 'partials', view: partials, type: 'u32', access: 'read_write'}
    );
    const communityTerm = inSum
      ? `(f32(outSum[outSumOffset + index]) / totalSlots) * (f32(inSum[inSumOffset + index]) / totalSlots)`
      : `(f32(outSum[outSumOffset + index]) / totalSlots) * (f32(outSum[outSumOffset + index]) / totalSlots)`;
    const treeReduce = `workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedCount[localInvocationIndex] += sharedCount[localInvocationIndex + stride];
      sharedLargest[localInvocationIndex] =
        max(sharedLargest[localInvocationIndex], sharedLargest[localInvocationIndex + stride]);
      sharedSum[localInvocationIndex] += sharedSum[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`;
    const sharedDeclarations = `var<workgroup> sharedCount: array<u32, ${WORKGROUP_SIZE}>;
var<workgroup> sharedLargest: array<u32, ${WORKGROUP_SIZE}>;
var<workgroup> sharedSum: array<f32, ${WORKGROUP_SIZE}>;`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-partials`,
        operation,
        variant: 'finish-partials',
        bindings: partialBindings,
        invocationCount: partialCount * WORKGROUP_SIZE,
        workgroupSize: WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
${sharedDeclarations}`,
        body: `let liveSlots = summaryIn[summaryInOffset + ${word.liveSlotCount}u];
  let selfLoops = summaryIn[summaryInOffset + ${word.selfLoopSlotCount}u];
  let totalSlots = f32(liveSlots${countTwice ? ' + selfLoops' : ''});
  var componentCount = 0u;
  var largest = 0u;
  var termSum = 0.0;
  if (index < NODE_COUNT) {
    let size = componentSize[componentSizeOffset + index];
    if (size > 0u) {
      componentCount = 1u;
      largest = size;
    }
    ${hasCommunities ? `if (liveSlots > 0u) { termSum = ${communityTerm}; }` : ''}
  }
  sharedCount[localInvocationIndex] = componentCount;
  sharedLargest[localInvocationIndex] = largest;
  sharedSum[localInvocationIndex] = termSum;
  ${treeReduce}
  if (localInvocationIndex == 0u) {
    let partial = 3u * (index / ${WORKGROUP_SIZE}u);
    partials[partialsOffset + partial] = sharedCount[0];
    partials[partialsOffset + partial + 1u] = sharedLargest[0];
    partials[partialsOffset + partial + 2u] = bitcast<u32>(sharedSum[0]);
  }`
      })
    );
    const finishBindings: WGSLKernelBinding[] = [
      {name: 'partials', view: partials, type: 'u32', access: 'read'},
      {name: 'converged', view: converged, type: 'u32', access: 'read'}
    ];
    if (props.parameters) {
      finishBindings.push({
        name: 'parameters',
        view: props.parameters,
        type: 'u32',
        access: 'read'
      });
    }
    finishBindings.push({
      name: 'summary',
      view: output,
      type: 'u32',
      access: 'read_write'
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation,
        variant: 'finish',
        bindings: finishBindings,
        invocationCount: WORKGROUP_SIZE,
        workgroupSize: WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const PARTIAL_COUNT: u32 = ${partialCount}u;
${sharedDeclarations}`,
        body: `let liveSlots = summary[summaryOffset + ${word.liveSlotCount}u];
  let selfLoops = summary[summaryOffset + ${word.selfLoopSlotCount}u];
  let totalSlots = f32(liveSlots${countTwice ? ' + selfLoops' : ''});
  var componentCount = 0u;
  var largest = 0u;
  var termSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${WORKGROUP_SIZE}u) {
    componentCount += partials[partialsOffset + 3u * partial];
    largest = max(largest, partials[partialsOffset + 3u * partial + 1u]);
    termSum += bitcast<f32>(partials[partialsOffset + 3u * partial + 2u]);
  }
  sharedCount[localInvocationIndex] = componentCount;
  sharedLargest[localInvocationIndex] = largest;
  sharedSum[localInvocationIndex] = termSum;
  ${treeReduce}
  if (localInvocationIndex == 0u) {
    let invalidLabels = summary[summaryOffset + ${word.modularityValid}u];
    let intra = summary[summaryOffset + ${word.intraCommunitySlotCount}u]${countTwice ? ' + selfLoops' : ''};
    summary[summaryOffset + ${word.componentCount}u] = sharedCount[0];
    summary[summaryOffset + ${word.largestComponentSize}u] = sharedLargest[0];
    summary[summaryOffset + ${word.componentsConverged}u] = select(0u, 1u, converged[convergedOffset] != 0u);
    summary[summaryOffset + ${word.liveEdgeCount}u] = ${
      directed ? 'liveSlots' : '(liveSlots - selfLoops) / 2u + selfLoops'
    };
    ${modularityFinish(directed, hasCommunities, Boolean(props.parameters))}
    summary[summaryOffset + 14u] = 0u;
    summary[summaryOffset + 15u] = 0u;
  }`
      })
    );
    return nodes;
  }
}

function modularityFinish(
  directed: boolean,
  hasCommunities: boolean,
  hasParameters: boolean
): string {
  const word = GPU_NETWORK_STATISTICS_WORD;
  if (!hasCommunities) {
    return `summary[summaryOffset + ${word.modularity}u] = 0u;
    summary[summaryOffset + ${word.modularityValid}u] = 0u;`;
  }
  const resolution = hasParameters ? 'bitcast<f32>(parameters[parametersOffset])' : '1.0';
  return `let modularityValid = invalidLabels == 0u && liveSlots > 0u;
    var modularity = 0.0;
    if (modularityValid) {
      modularity = f32(intra) / totalSlots - ${resolution} * sharedSum[0];
    }
    summary[summaryOffset + ${word.modularity}u] = bitcast<u32>(modularity);
    summary[summaryOffset + ${word.modularityValid}u] = select(0u, 1u, modularityValid);`;
}
