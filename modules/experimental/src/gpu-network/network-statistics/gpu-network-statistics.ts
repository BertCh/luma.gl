// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGraphConnectedComponents} from '@luma.gl/gpgpu/gpu-graph';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  importGraphBuffer,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createNetworkAnalyticsTopology,
  getCoreDefaultBuffer,
  getGraphViewDefaultBuffer,
  getGraphViewGPUVector
} from '../network-analysis/network-analytics-topology';

/** Number of header words that precede the three degree histograms in the summary. */
export const GPU_NETWORK_STATISTICS_HEADER_LENGTH = 16;

/** Word indices of the summary header. */
export const GPU_NETWORK_STATISTICS_WORD = {
  /** Vertices that are live (`vertexMask` nonzero, or every vertex without a mask). */
  liveVertexCount: 0,
  /** Directed: `liveSlotCount`. Undirected: live non-self-loop slots / 2 plus live self-loop slots. */
  liveEdgeCount: 1,
  /** CSR slots that are live. */
  liveSlotCount: 2,
  /** Weak components among live vertices. */
  componentCount: 3,
  /** Vertex count of the largest live weak component. */
  largestComponentSize: 4,
  /** 1 when the component labelling reached a fixed point within `componentIterations`. */
  componentsConverged: 5,
  /** Live vertices with zero live degree. */
  isolatedVertexCount: 6,
  /** Largest live out-degree. */
  maxOutDegree: 7,
  /** Largest live in-degree. */
  maxInDegree: 8,
  /** Largest live total degree. */
  maxTotalDegree: 9,
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

/**
 * Encodes the per-frame parameter words: word 0 is the modularity resolution as `float32` bits,
 * word 1 is the linear degree bin width (clamped to at least 1 on the GPU).
 *
 * Write the result into a `'uint32'` view of two words, for example with
 * `GPUParameterBuffer.write`.
 */
export function encodeGPUNetworkStatisticsParameters(
  parameters: {resolution?: number; degreeBinWidth?: number} = {}
): Uint32Array {
  const words = new Uint32Array(GPU_NETWORK_STATISTICS_PARAMETER_LENGTH);
  new Float32Array(words.buffer, 0, 1)[0] = parameters.resolution ?? 1;
  words[1] = Math.max(1, Math.floor(parameters.degreeBinWidth ?? 1));
  return words;
}

/** Histogram layout needed to decode a summary. */
export type GPUNetworkStatisticsLayout = {
  /** Bins per histogram. Defaults to 32. */
  degreeBinCount?: number;
};

/** Decoded summary. */
export type GPUNetworkStatisticsResult = {
  liveVertexCount: number;
  liveEdgeCount: number;
  liveSlotCount: number;
  componentCount: number;
  largestComponentSize: number;
  componentsConverged: boolean;
  isolatedVertexCount: number;
  maxOutDegree: number;
  maxInDegree: number;
  maxTotalDegree: number;
  /** 0 when `modularityValid` is false. */
  modularity: number;
  modularityValid: boolean;
  intraCommunitySlotCount: number;
  selfLoopSlotCount: number;
  /** Live-vertex counts per out-degree bin. */
  outDegreeHistogram: number[];
  /** Live-vertex counts per in-degree bin. */
  inDegreeHistogram: number[];
  /** Live-vertex counts per total-degree bin. */
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
    liveVertexCount: words[word.liveVertexCount],
    liveEdgeCount: words[word.liveEdgeCount],
    liveSlotCount: words[word.liveSlotCount],
    componentCount: words[word.componentCount],
    largestComponentSize: words[word.largestComponentSize],
    componentsConverged: words[word.componentsConverged] !== 0,
    isolatedVertexCount: words[word.isolatedVertexCount],
    maxOutDegree: words[word.maxOutDegree],
    maxInDegree: words[word.maxInDegree],
    maxTotalDegree: words[word.maxTotalDegree],
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
 * `degreeBinning` and `componentIterations`. Per-frame: the contents of the CSR, both masks, the
 * community labels and the parameter view, so filtering a live network changes the statistics on
 * the next encoding without recompiling.
 */
export type GPUNetworkStatisticsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-statistics'`. */
  id?: string;
  /** Forward CSR row offsets with `nodeCount + 1` rows, imported with a default buffer. */
  offsets: GraphDataView<'uint32'>;
  /**
   * Forward CSR destination per slot. Indices `>= nodeCount` are dead slots. Undirected graphs
   * list each edge in both directions (a self-loop once), like `GPUNetworkAnalyticsColumns`.
   */
  neighbors: GraphDataView<'uint32'>;
  /** Treat slots as directed edges. Default false. */
  directed?: boolean;
  /** Optional per-vertex mask, nonzero = live. */
  vertexMask?: GraphDataView<'uint32'>;
  /**
   * Optional per-slot mask aligned with `neighbors`, nonzero = live. A slot u -> v is live iff
   * this mask (if any) is nonzero, both u and v are live and `v < nodeCount`.
   */
  edgeMask?: GraphDataView<'uint32'>;
  /**
   * Optional community label per vertex (labels must be `< nodeCount`; a live vertex with a larger
   * label makes modularity invalid). Enables modularity.
   */
  communities?: GraphDataView<'uint32'>;
  /**
   * Optional per-frame `'uint32'` view of at least two words, see
   * {@link encodeGPUNetworkStatisticsParameters}: word 0 modularity resolution (`float32` bits,
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
  /** Component relaxation iterations, compile-time, default 32. */
  componentIterations?: number;
  /** Packed `uint32` summary with exactly {@link getGPUNetworkStatisticsLength} rows. */
  output: GraphDataView<'uint32'>;
};

const WORKGROUP_SIZE = 256;

/**
 * Reduces a CSR network to a compact statistics summary for a graph stats panel: live
 * vertex/edge/slot counts, weak component count and largest size, isolated vertices, maximum
 * degrees, three degree histograms and optional modularity, all in one caller-owned `uint32` view.
 * Every word is rewritten on every encoding; the CPU decodes it with
 * {@link decodeGPUNetworkStatistics}.
 *
 * Summary layout: words `0..15` are the header ({@link GPU_NETWORK_STATISTICS_WORD}, words 14 and
 * 15 reserved zeros), then out-degree, in-degree and total-degree histograms of `degreeBinCount`
 * bins each. Undirected: in = out = total = live row length (not doubled). Directed: total = in +
 * out. Degree statistics cover live vertices only.
 *
 * Components run `GPUGraphConnectedComponents` on a contributor-owned masked copy of the neighbors
 * (dead slots become `0xffffffff`), so masked vertices are isolated and excluded from counts.
 *
 * Modularity uses exact `u32` atomic sums (intra-community slots and per-label degree sums) and a
 * single-workgroup f32 finish, so it is deterministic. `GPUGraphModularity` accumulates floats
 * with atomics and is therefore order dependent; it is deliberately not used. Undirected:
 * `Q = intra/2m - gamma * sum_c (d_c/2m)^2` with `2m = liveSlotCount`; directed:
 * `Q = intra/m - gamma * sum_c out_c in_c / m^2` with `m = liveSlotCount`. Counts are `u32`, so a
 * graph with 2^32 or more live slots overflows. Invalid (no live slots, no communities or an
 * out-of-range label) gives modularity 0 and `modularityValid` 0.
 *
 * Owns a masked-neighbors buffer, a label buffer and small scalars, released by {@link destroy}.
 * Add the contributor to one graph at a time; destroy compiled graphs first.
 */
export class GPUNetworkStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkStatisticsProps;
  /** Number of nodes, `offsets.length - 1`. */
  readonly nodeCount: number;
  /** Bins per histogram. */
  readonly degreeBinCount: number;
  private readonly maskedNeighbors: Buffer;
  private readonly componentLabels: Buffer;
  private readonly componentConverged: Buffer;
  private readonly status: Buffer;

  constructor(props: GPUNetworkStatisticsProps) {
    this.id = props.id ?? 'network-statistics';
    this.props = props;
    const {id} = this;
    validatePackedView(props.offsets, ['uint32'], `${id} offsets`);
    validatePackedView(props.neighbors, ['uint32'], `${id} neighbors`);
    this.nodeCount = props.offsets.length - 1;
    const {nodeCount} = this;
    if (nodeCount < 1) {
      throw new Error(`${id} offsets must contain at least two rows`);
    }
    this.degreeBinCount = props.degreeBinCount ?? 32;
    if (
      !Number.isSafeInteger(this.degreeBinCount) ||
      this.degreeBinCount < 1 ||
      this.degreeBinCount > 4096
    ) {
      throw new Error(`${id} degreeBinCount must be an integer between 1 and 4096`);
    }
    if (props.degreeBinning && props.degreeBinning !== 'linear' && props.degreeBinning !== 'log2') {
      throw new Error(`${id} degreeBinning must be 'linear' or 'log2'`);
    }
    const iterations = props.componentIterations ?? 32;
    if (!Number.isSafeInteger(iterations) || iterations < 1 || iterations > 1024) {
      throw new Error(`${id} componentIterations must be an integer between 1 and 1024`);
    }
    for (const [name, view, length] of [
      ['vertexMask', props.vertexMask, nodeCount],
      ['communities', props.communities, nodeCount],
      ['edgeMask', props.edgeMask, props.neighbors.length]
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
        props.offsets,
        props.neighbors,
        props.vertexMask,
        props.edgeMask,
        props.communities,
        props.parameters
      ]
    );
    // The gpu-graph algorithm binds the offsets through a GPUVector over the same buffer.
    const device = getCoreDefaultBuffer(
      getGraphViewDefaultBuffer(id, 'offsets', props.offsets)
    ).device;
    const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
    const create = (name: string, words: number) =>
      device.createBuffer({
        id: `${id}-${name}`,
        byteLength: Math.max(words, 1) * 4,
        usage
      });
    this.maskedNeighbors = create('masked-neighbors', props.neighbors.length);
    this.componentLabels = create('component-labels', nodeCount);
    this.componentConverged = create('component-converged', 1);
    this.status = create('status', 4);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, degreeBinCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.offsets,
      props.neighbors,
      props.vertexMask,
      props.edgeMask,
      props.communities,
      props.parameters,
      props.output
    ]);
    const directed = Boolean(props.directed);
    const hasCommunities = Boolean(props.communities);
    const word = GPU_NETWORK_STATISTICS_WORD;
    const operation = 'GPUNetworkStatistics';
    const masked = importGraphBuffer(
      graph,
      `${id}-masked-neighbors`,
      this.maskedNeighbors,
      'uint32',
      props.neighbors.length
    );
    const labels = importGraphBuffer(
      graph,
      `${id}-component-labels`,
      this.componentLabels,
      'uint32',
      nodeCount
    );
    const converged = importGraphBuffer(
      graph,
      `${id}-component-converged`,
      this.componentConverged,
      'uint32',
      1
    );
    const transient = (name: string) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', nodeCount);
    const outDegree = transient('out-degree');
    const inDegree = directed ? transient('in-degree') : undefined;
    const componentSize = transient('component-size');
    const outSum = hasCommunities ? transient('community-out-sum') : undefined;
    const inSum = hasCommunities && directed ? transient('community-in-sum') : undefined;
    const {output} = props;

    const maskBindings: WGSLKernelBinding[] = [];
    if (props.vertexMask) {
      maskBindings.push({
        name: 'vertexMask',
        view: props.vertexMask,
        type: 'u32',
        access: 'read'
      });
    }
    const isLive = (vertex: string) =>
      props.vertexMask ? `vertexMask[vertexMaskOffset + ${vertex}] != 0u` : 'true';
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

    // Pass 1: mask slots, live row length, in-degree, live slot and self-loop counts.
    const passOneBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      ...maskBindings,
      {name: 'maskedOut', view: masked, type: 'u32', access: 'read_write'},
      {name: 'outDegree', view: outDegree, type: 'u32', access: 'read_write'}
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
    passOneBindings.push({
      name: 'summary',
      view: output,
      type: 'atomic<u32>',
      access: 'read_write'
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slots`,
        operation,
        variant: 'slots',
        bindings: passOneBindings,
        invocationCount: nodeCount,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;`,
        body: `let vertexLive = ${isLive('index')};
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd = offsets[offsetsOffset + index + 1u];
  var liveCount = 0u;
  var selfCount = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    var slotLive = vertexLive && neighbor < NODE_COUNT;
    ${props.vertexMask ? 'if (slotLive) { slotLive = vertexMask[vertexMaskOffset + neighbor] != 0u; }' : ''}
    ${props.edgeMask ? 'if (slotLive) { slotLive = edgeMask[edgeMaskOffset + slot] != 0u; }' : ''}
    maskedOut[maskedOutOffset + slot] = select(0xffffffffu, neighbor, slotLive);
    if (slotLive) {
      liveCount++;
      if (neighbor == index) {
        selfCount++;
      }
      ${inDegree ? 'atomicAdd(&inDegree[inDegreeOffset + neighbor], 1u);' : ''}
    }
  }
  outDegree[outDegreeOffset + index] = liveCount;
  if (liveCount > 0u) {
    atomicAdd(&summary[summaryOffset + ${word.liveSlotCount}u], liveCount);
  }
  if (selfCount > 0u) {
    atomicAdd(&summary[summaryOffset + ${word.selfLoopSlotCount}u], selfCount);
  }`
      })
    );

    // Weak components over the masked adjacency.
    nodes.push(
      ...captureGraphCommandNodes(graph, () => {
        const topology = createNetworkAnalyticsTopology({
          id,
          nodeCount,
          offsets: props.offsets,
          neighbors: masked,
          status: this.status
        });
        new GPUGraphConnectedComponents({
          id: `${id}-components`,
          topology,
          output: getGraphViewGPUVector(id, 'component labels', labels),
          iterations: props.componentIterations,
          converged: getGraphViewGPUVector(id, 'component converged', converged)
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
          },
          {
            name: 'summary',
            view: output,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: nodeCount,
        body: `if (!(${isLive('index')})) {
    return;
  }
  atomicAdd(&componentSize[componentSizeOffset + labels[labelsOffset + index]], 1u);
  atomicAdd(&summary[summaryOffset + ${word.liveVertexCount}u], 1u);`
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
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-degrees`,
        operation,
        variant: 'degrees',
        bindings: degreeBindings,
        invocationCount: nodeCount,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const BIN_COUNT: u32 = ${degreeBinCount}u;
const HISTOGRAM_BASE: u32 = ${GPU_NETWORK_STATISTICS_HEADER_LENGTH}u;
${binSource}`,
        body: `if (!(${isLive('index')})) {
    return;
  }
  let outValue = outDegree[outDegreeOffset + index];
  let inValue = ${inDegree ? 'inDegree[inDegreeOffset + index]' : 'outValue'};
  let totalValue = ${directed ? 'outValue + inValue' : 'outValue'};
  if (totalValue == 0u) {
    atomicAdd(&summary[summaryOffset + ${word.isolatedVertexCount}u], 1u);
  }
  atomicMax(&summary[summaryOffset + ${word.maxOutDegree}u], outValue);
  atomicMax(&summary[summaryOffset + ${word.maxInDegree}u], inValue);
  atomicMax(&summary[summaryOffset + ${word.maxTotalDegree}u], totalValue);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + getBin(outValue)], 1u);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + BIN_COUNT + getBin(inValue)], 1u);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + 2u * BIN_COUNT + getBin(totalValue)], 1u);
  ${
    hasCommunities
      ? `let label = communities[communitiesOffset + index];
  if (label >= NODE_COUNT) {
    atomicAdd(&summary[summaryOffset + ${word.modularityValid}u], 1u);
  } else {
    atomicAdd(&outSum[outSumOffset + label], outValue);
    ${inSum ? 'atomicAdd(&inSum[inSumOffset + label], inValue);' : ''}
  }`
      : ''
  }`
      })
    );

    // Pass 4: intra-community live slots.
    if (props.communities) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-intra-community`,
          operation,
          variant: 'intra-community',
          bindings: [
            {
              name: 'offsets',
              view: props.offsets,
              type: 'u32',
              access: 'read'
            },
            {name: 'maskedIn', view: masked, type: 'u32', access: 'read'},
            {
              name: 'communities',
              view: props.communities,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'summary',
              view: output,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: nodeCount,
          body: `let label = communities[communitiesOffset + index];
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd = offsets[offsetsOffset + index + 1u];
  var intraCount = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    let neighbor = maskedIn[maskedInOffset + slot];
    if (neighbor != 0xffffffffu && communities[communitiesOffset + neighbor] == label) {
      intraCount++;
    }
  }
  if (intraCount > 0u) {
    atomicAdd(&summary[summaryOffset + ${word.intraCommunitySlotCount}u], intraCount);
  }`
        })
      );
    }

    // Pass 5: one workgroup reduces component sizes and community sums, then finishes the header.
    const finishBindings: WGSLKernelBinding[] = [
      {
        name: 'componentSize',
        view: componentSize,
        type: 'u32',
        access: 'read'
      },
      {name: 'converged', view: converged, type: 'u32', access: 'read'}
    ];
    if (outSum) {
      finishBindings.push({
        name: 'outSum',
        view: outSum,
        type: 'u32',
        access: 'read'
      });
    }
    if (inSum) {
      finishBindings.push({
        name: 'inSum',
        view: inSum,
        type: 'u32',
        access: 'read'
      });
    }
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
    const communityTerm = inSum
      ? `(f32(outSum[outSumOffset + label]) / totalSlots) * (f32(inSum[inSumOffset + label]) / totalSlots)`
      : `(f32(outSum[outSumOffset + label]) / totalSlots) * (f32(outSum[outSumOffset + label]) / totalSlots)`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation,
        variant: 'finish',
        bindings: finishBindings,
        invocationCount: WORKGROUP_SIZE,
        workgroupSize: WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
var<workgroup> sharedCount: array<u32, ${WORKGROUP_SIZE}>;
var<workgroup> sharedLargest: array<u32, ${WORKGROUP_SIZE}>;
var<workgroup> sharedSum: array<f32, ${WORKGROUP_SIZE}>;`,
        body: `let liveSlots = summary[summaryOffset + ${word.liveSlotCount}u];
  let totalSlots = f32(liveSlots);
  var componentCount = 0u;
  var largest = 0u;
  var termSum = 0.0;
  for (var label = localInvocationIndex; label < NODE_COUNT; label += ${WORKGROUP_SIZE}u) {
    let size = componentSize[componentSizeOffset + label];
    if (size > 0u) {
      componentCount++;
      largest = max(largest, size);
    }
    ${hasCommunities ? `if (liveSlots > 0u) { termSum += ${communityTerm}; }` : ''}
  }
  sharedCount[localInvocationIndex] = componentCount;
  sharedLargest[localInvocationIndex] = largest;
  sharedSum[localInvocationIndex] = termSum;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedCount[localInvocationIndex] += sharedCount[localInvocationIndex + stride];
      sharedLargest[localInvocationIndex] =
        max(sharedLargest[localInvocationIndex], sharedLargest[localInvocationIndex + stride]);
      sharedSum[localInvocationIndex] += sharedSum[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let selfLoops = summary[summaryOffset + ${word.selfLoopSlotCount}u];
    let invalidLabels = summary[summaryOffset + ${word.modularityValid}u];
    let intra = summary[summaryOffset + ${word.intraCommunitySlotCount}u];
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

  /** Releases the contributor-owned buffers. Destroy compiled graphs that use them first. */
  destroy(): void {
    this.maskedNeighbors.destroy();
    this.componentLabels.destroy();
    this.componentConverged.destroy();
    this.status.destroy();
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
