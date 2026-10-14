// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import type {GPUNetworkAdjacency} from '../../analysis-core/network-topology';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology
} from '../../analysis-core/network-topology';
import {createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {DECODE_ORDERED_KEY_WGSL, GET_ORDERED_KEY_WGSL} from '../../utils/wgsl-common';
import {
  getWorkgroupCounterKernelSource,
  WORKGROUP_COUNTER_WORKGROUP_SIZE
} from '../network-workgroup-counters';

/**
 * Largest supported `groupCapacity`. A superedge key `source * groupCapacity + target` must stay
 * below `2^32`, and `65535^2 + 1` does.
 */
export const GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY = 65535;

/**
 * Largest superedge capacity (`output.edges.ids` rows) of {@link GPUNetworkCoarsening} with
 * `output.edgeWeights` on `device`: the 64-bit fixed-point weight accumulators hold 8 bytes per
 * superedge slot in one storage binding, so 16777216 on a default device (128 MiB bindings).
 */
export function getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/** Number of words of the optional `summary` view. */
export const GPU_NETWORK_COARSENING_SUMMARY_LENGTH = 8;

/** Word indices of the `summary` view. */
export const GPU_NETWORK_COARSENING_SUMMARY_WORD = {
  /** Live nodes, including nodes whose label is `>= groupCapacity`. */
  liveNodeCount: 0,
  /** Live nodes whose label is `>= groupCapacity`; they are excluded from every group. */
  overflowedNodeCount: 1,
  /** Counted live edges (directed: live slots; undirected: live slots `u -> v` with `u <= v`). */
  countedEdgeCount: 2,
  /** Counted edges dropped because an endpoint label is `>= groupCapacity`. */
  droppedEdgeCount: 3,
  /** Counted edges whose endpoints share a valid group (self-loops included). */
  intraEdgeCount: 4,
  /** Counted edges between two distinct valid groups; the sum of every superedge count. */
  interEdgeCount: 5,
  /** Groups with at least one live node. */
  groupCount: 6,
  /** Distinct superedges, before clamping to the superedge capacity. */
  superedgeCount: 7
} as const;

/** Decoded `summary` words. */
export type GPUNetworkCoarseningSummary = {
  liveNodeCount: number;
  overflowedNodeCount: number;
  countedEdgeCount: number;
  droppedEdgeCount: number;
  intraEdgeCount: number;
  interEdgeCount: number;
  groupCount: number;
  superedgeCount: number;
};

/** Decodes the `summary` words read back from the view. */
export function decodeGPUNetworkCoarseningSummary(
  words: Uint32Array | ArrayLike<number>
): GPUNetworkCoarseningSummary {
  if (words.length < GPU_NETWORK_COARSENING_SUMMARY_LENGTH) {
    throw new Error('decodeGPUNetworkCoarseningSummary needs eight words');
  }
  const word = GPU_NETWORK_COARSENING_SUMMARY_WORD;
  return {
    liveNodeCount: words[word.liveNodeCount],
    overflowedNodeCount: words[word.overflowedNodeCount],
    countedEdgeCount: words[word.countedEdgeCount],
    droppedEdgeCount: words[word.droppedEdgeCount],
    intraEdgeCount: words[word.intraEdgeCount],
    interEdgeCount: words[word.interEdgeCount],
    groupCount: words[word.groupCount],
    superedgeCount: words[word.superedgeCount]
  };
}

/**
 * Properties for {@link GPUNetworkCoarsening}.
 *
 * Compile-time: node and slot counts, `directed`, `groupCapacity`, `fixedPointScale`, which
 * optional views exist and every output capacity. Per-frame: the contents of every input view, so
 * re-labelling or filtering a live network changes the summary on the next encoding.
 */
export type GPUNetworkCoarseningProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-coarsening'`. */
  id?: string;
  /**
   * Forward CSR network with at least one slot. Neighbor indices `>= nodeCount` are dead slots.
   * Optional `weights` are summed per superedge (missing weights are 1); optional `nodePositions`
   * (finite values) feed centroids and bounds.
   */
  topology: GPUNetworkAdjacency;
  /**
   * Treat slots as directed edges. Default false: the CSR lists each undirected edge in both
   * directions (a self-loop once), like `GPUNetworkStatistics`, and only the slot `u -> v` with
   * `u <= v` is counted, so each edge counts once. An undirected `edgeMask` should be symmetric.
   */
  directed?: boolean;
  /** Optional per-node mask, nonzero = live. */
  nodeMask?: GraphDataView<'uint32'>;
  /**
   * Optional per-slot mask aligned with `topology.neighbors`, nonzero = live. A slot `u -> v` is
   * live iff this mask (if any) is nonzero, both endpoints are live and `v < nodeCount`.
   */
  edgeMask?: GraphDataView<'uint32'>;
  /**
   * Group label per node (one `uint32` row per node), for example a community from label
   * propagation, a component or a spatial cell. Dense labels `< groupCapacity` are the group IDs.
   */
  labels: GraphDataView<'uint32'>;
  /** Optional per-node finite `float32` value column summed per group. */
  nodeValues?: GraphDataView<'float32'>;
  /**
   * Compile-time number of dense groups, `1..65535`. Every `groupXxx` output has exactly this many
   * rows. A live node with a label at or above it is excluded and raises `output.edges.overflow`.
   */
  groupCapacity: number;
  /**
   * Power-of-two scale of the 64-bit fixed-point accumulators used for weight, position and value
   * sums. Default 65536. Each addend is rounded to a multiple of `1 / fixedPointScale`.
   */
  fixedPointScale?: number;
  /** Caller-owned outputs. */
  output: GPUNetworkCoarseningOutput;
};

/** GPU-written outputs of {@link GPUNetworkCoarsening}. */
export type GPUNetworkCoarseningOutput = {
  /** Exact-size (`groupCapacity` rows) per-group live node count. */
  groupNodeCount: GraphDataView<'uint32'>;
  /** Optional per-group count of intra-group counted edges (self-loops included). */
  groupIntraEdgeCount?: GraphDataView<'uint32'>;
  /** Optional per-group sum of intra-group edge weights. */
  groupIntraWeight?: GraphDataView<'float32'>;
  /** Optional per-group centroid `sum / count`; requires `topology.nodePositions`. Empty groups are 0. */
  groupCentroid?: GraphDataView<'float32x2'>;
  /** Optional per-group `[minX, minY, maxX, maxY]`; requires `topology.nodePositions`. Empty groups are 0. */
  groupBounds?: GraphDataView<'float32x4'>;
  /** Optional per-group sum of `nodeValues`; requires `nodeValues`. */
  groupValueSum?: GraphDataView<'float32'>;
  /**
   * Bounded superedge list: `ids` is the source group. The capacity is `ids.length`. Rows are
   * sorted ascending by `(source, target)`; undirected superedges have `source < target`. When the
   * list overflows the first rows in that order are kept.
   * `overflow` is also set by label overflow.
   */
  edges: GPUCompactOutput;
  /** Target group per superedge row, `edges.ids.length` rows. */
  edgeTargetGroups: GraphDataView<'uint32'>;
  /** Edge count per superedge row, `edges.ids.length` rows. */
  edgeCounts: GraphDataView<'uint32'>;
  /** Optional summed weight per superedge row, `edges.ids.length` rows. */
  edgeWeights?: GraphDataView<'float32'>;
  /** Optional {@link GPU_NETWORK_COARSENING_SUMMARY_LENGTH}-word diagnostic view. */
  summary?: GraphDataView<'uint32'>;
};

const OPERATION = 'GPUNetworkCoarsening';
const GROUP_ACC_STRIDE = 10;
/** Sorted slots folded per invocation when accumulating superedge counts and weights. */
const SEGMENT_RUN_LENGTH = 16;
const WORD = GPU_NETWORK_COARSENING_SUMMARY_WORD;

/** Shared WGSL for the 64-bit signed fixed-point accumulators. */
function getFixedPointSource(scale: number): string {
  return `const SCALE: f32 = ${scale}.0;
const TWO_32: f32 = 4294967296.0;
const FIXED_LIMIT: f32 = 4.0e18;
fn quantize(value: f32) -> vec2<u32> {
  let negative = value < 0.0;
  let x = min(round(abs(value) * SCALE), FIXED_LIMIT);
  let highFloat = floor(x / TWO_32);
  var low = u32(x - highFloat * TWO_32);
  var high = u32(highFloat);
  if (negative) {
    low = ~low + 1u;
    high = ~high + select(0u, 1u, low == 0u);
  }
  return vec2<u32>(low, high);
}
fn dequantize(low: u32, high: u32) -> f32 {
  let negative = (high & 0x80000000u) != 0u;
  var magnitudeLow = low;
  var magnitudeHigh = high;
  if (negative) {
    magnitudeLow = ~low + 1u;
    magnitudeHigh = ~high + select(0u, 1u, magnitudeLow == 0u);
  }
  let magnitude = f32(magnitudeHigh) * TWO_32 + f32(magnitudeLow);
  return select(magnitude, -magnitude, negative) / SCALE;
}
${GET_ORDERED_KEY_WGSL}${DECODE_ORDERED_KEY_WGSL}`;
}

/** WGSL statements that add `vec2<u32>` `quantized` into the atomic array words `index`, `index + 1`. */
function getFixedPointAddSource(array: string, index: string, quantized: string): string {
  return `{
    let addendOld = atomicAdd(&${array}[${index}], ${quantized}.x);
    let addendCarry = select(0u, 1u, addendOld + ${quantized}.x < addendOld);
    atomicAdd(&${array}[${index} + 1u], ${quantized}.y + addendCarry);
  }`;
}

/**
 * Coarsens a CSR network by a node group label into a summary graph for zoomed-out compound
 * nodes: per-group node counts, centroids, bounds, value sums and intra-group edge statistics,
 * plus a bounded list of superedges (source group, target group, edge count, summed weight).
 *
 * Labels. Labels are arbitrary `uint32`, but the group ID is the label itself and must be below
 * the compile-time `groupCapacity` (at most 65535). A live node with a larger label is excluded
 * from every group and superedge, edges touching it are counted in `droppedEdgeCount`, and
 * `edges.overflow` is set. A GPU-side dense relabel was rejected: first-come hash slot assignment
 * is nondeterministic across runs, and a deterministic rank needs a sort of the vertices plus a
 * hash probe per edge endpoint, a large cost for the labels that matter (communities and
 * components below `nodeCount`, spatial cells indexed by the caller). Remap sparse labels first,
 * for example with a caller-owned lookup table.
 *
 * Exactness. Node, intra and edge counts are `u32` atomics, so they are integer exact. Every sum
 * (weights, positions, values) is accumulated as a two-word signed 64-bit fixed-point integer
 * (`round(value * fixedPointScale)`, carry propagated between two `atomicAdd`s). Integer addition
 * is order independent, so sums are bitwise reproducible; they differ from an exact real sum only
 * by the per-addend rounding to `1 / fixedPointScale` and clamp at about `4e18 / fixedPointScale`.
 * The single final conversion to `float32` (`sum / scale`, then `/ count` for centroids) is
 * identical on every run. Bounds use ordered-integer `atomicMin`/`atomicMax`, so they are exact.
 * Inputs must be finite. Sums wrap at `2^63 / fixedPointScale` in total.
 *
 * Superedges use sort-and-segment: each counted inter-group slot gets the key `a * groupCapacity +
 * b` (`a < b` for undirected edges), keys sort with the stable radix `GPUSort`, a scan numbers the
 * segments, and counts and fixed-point weight sums accumulate per segment. The output order, the
 * kept prefix on overflow, the counts and the weights are all deterministic. A bounded hash table
 * was rejected because a full table drops keys depending on atomic race order.
 *
 * The contributor owns no GPU resources; every transient belongs to the graph. Every output word is
 * rewritten on every encoding.
 */
export class GPUNetworkCoarsening implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkCoarseningProps;
  /** Number of nodes, `topology.offsets.length - 1`. */
  readonly nodeCount: number;
  /** Number of dense groups. */
  readonly groupCapacity: number;
  /** Maximum number of superedge rows, `edges.ids.length`. */
  readonly edgeCapacity: number;

  constructor(props: GPUNetworkCoarseningProps) {
    this.id = props.id ?? 'network-coarsening';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const topologyShape = validateGPUNetworkTopology(id, props.topology);
    validatePackedView(props.labels, ['uint32'], `${id} labels`);
    this.nodeCount = topologyShape.nodeCount;
    const {nodeCount} = this;
    if (topologyShape.edgeCount < 1) {
      throw new Error(`${id} topology.neighbors must contain at least one slot`);
    }
    this.groupCapacity = props.groupCapacity;
    const groupCapacity = this.groupCapacity;
    validateCapacity(
      id,
      'groupCapacity',
      groupCapacity,
      1,
      GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY
    );
    const scale = props.fixedPointScale ?? 65536;
    validatePowerOfTwo(id, 'fixedPointScale', scale, 1, 2 ** 24);
    const rowChecks: [string, GraphDataView | undefined, number, readonly string[]][] = [
      ['nodeMask', props.nodeMask, nodeCount, ['uint32']],
      ['labels', props.labels, nodeCount, ['uint32']],
      ['nodeValues', props.nodeValues, nodeCount, ['float32']],
      ['edgeMask', props.edgeMask, props.topology.neighbors.length, ['uint32']],
      ['output.groupNodeCount', output.groupNodeCount, groupCapacity, ['uint32']],
      ['output.groupIntraEdgeCount', output.groupIntraEdgeCount, groupCapacity, ['uint32']],
      ['output.groupIntraWeight', output.groupIntraWeight, groupCapacity, ['float32']],
      ['output.groupCentroid', output.groupCentroid, groupCapacity, ['float32x2']],
      ['output.groupBounds', output.groupBounds, groupCapacity, ['float32x4']],
      ['output.groupValueSum', output.groupValueSum, groupCapacity, ['float32']]
    ];
    for (const [name, view, length, formats] of rowChecks) {
      if (!view) {
        continue;
      }
      validatePackedView(view, formats as any, `${id} ${name}`);
      if (view.length !== length) {
        throw new Error(`${id} ${name} must contain exactly ${length} rows`);
      }
    }
    if ((output.groupCentroid || output.groupBounds) && !props.topology.nodePositions) {
      throw new Error(
        `${id} output.groupCentroid and output.groupBounds require topology.nodePositions`
      );
    }
    if (output.groupValueSum && !props.nodeValues) {
      throw new Error(`${id} output.groupValueSum requires nodeValues`);
    }
    validateCompactOutput(id, output.edges);
    this.edgeCapacity = output.edges.ids.length;
    if (this.edgeCapacity < 1) {
      throw new Error(`${id} output.edges.ids must contain at least one row`);
    }
    for (const [name, view, formats] of [
      ['output.edgeTargetGroups', output.edgeTargetGroups, ['uint32']],
      ['output.edgeCounts', output.edgeCounts, ['uint32']],
      ['output.edgeWeights', output.edgeWeights, ['float32']]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, formats as any, `${id} ${name}`);
      if (view.length !== this.edgeCapacity) {
        throw new Error(`${id} ${name} must match output.edges.ids length`);
      }
    }
    if (props.output.summary) {
      validatePackedView(props.output.summary, ['uint32'], `${id} output.summary`);
      if (props.output.summary.length !== GPU_NETWORK_COARSENING_SUMMARY_LENGTH) {
        throw new Error(
          `${id} output.summary must contain exactly ${GPU_NETWORK_COARSENING_SUMMARY_LENGTH} rows`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputs(), this.getInputs());
  }

  private getInputs(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      ...getGPUNetworkTopologyViews(props.topology),
      props.nodeMask,
      props.edgeMask,
      props.labels,
      props.nodeValues
    ];
  }

  private getOutputs(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.groupNodeCount,
      output.groupIntraEdgeCount,
      output.groupIntraWeight,
      output.groupCentroid,
      output.groupBounds,
      output.groupValueSum,
      output.edges.ids,
      output.edges.status.count,
      output.edges.status.overflow,
      output.edges.status.requiredCount,
      output.edgeTargetGroups,
      output.edgeCounts,
      output.edgeWeights,
      output.summary
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, groupCapacity, edgeCapacity} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputs(), ...this.getOutputs()]);
    if (props.output.edgeWeights) {
      validateMaximumCount(
        id,
        'output.edges.ids rows with output.edgeWeights',
        edgeCapacity,
        getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(graph.device)
      );
    }
    const slotCount = props.topology.neighbors.length;
    const directed = Boolean(props.directed);
    const scale = props.fixedPointScale ?? 65536;
    const sentinelKey = groupCapacity * groupCapacity;
    const hasPositions = Boolean(props.topology.nodePositions);
    const hasValues = Boolean(props.nodeValues);
    const hasIntra = Boolean(props.output.groupIntraEdgeCount || props.output.groupIntraWeight);
    const hasIntraWeight = Boolean(props.output.groupIntraWeight);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    const effectiveLabels = u32('effective-labels', nodeCount);
    const groupAcc = u32('group-acc', groupCapacity * GROUP_ACC_STRIDE);
    const intraAcc = hasIntra ? u32('intra-acc', groupCapacity * 3) : undefined;
    const stats = props.output.summary ?? u32('stats', GPU_NETWORK_COARSENING_SUMMARY_LENGTH);
    const sortKeys = u32('sort-keys', slotCount);
    const sortIndices = u32('sort-indices', slotCount);
    const sortedKeys = u32('sorted-keys', slotCount);
    const sortedIndices = u32('sorted-indices', slotCount);
    const flags = u32('segment-flags', slotCount);
    const segmentScan = u32('segment-scan', slotCount);
    const segmentTotal = u32('segment-total', 1);
    const overflowFlag = u32('overflow-flag', 1);
    const segmentWeightAcc = props.output.edgeWeights
      ? u32('segment-weight-acc', edgeCapacity * 2)
      : undefined;

    const fill = (
      step: string,
      view: GraphDataView | undefined,
      type: 'u32' | 'f32' = 'u32',
      value = '0u'
    ) => {
      if (view) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${step}`,
            operation: OPERATION,
            view,
            type,
            value
          })
        );
      }
    };
    fill('zero-stats', stats);
    fill('zero-group-counts', props.output.groupNodeCount);
    fill('zero-intra', intraAcc);
    fill('zero-edge-ids', props.output.edges.ids);
    fill('zero-edge-targets', props.output.edgeTargetGroups);
    fill('zero-edge-counts', props.output.edgeCounts);
    fill('zero-edge-weight-acc', segmentWeightAcc);
    // Slots past offsets[NODE_COUNT] belong to no row, so the per-row slots pass never writes them.
    // They take the sentinel key and sort last; otherwise stale words (transients alias) would
    // enter the sort as superedges (review B-09, library rule 11).
    fill('clear-sort-keys', sortKeys, 'u32', `${sentinelKey}u`);

    // Initialise the per-group accumulator: sums are 0, bound minima are the largest ordered key.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-init-groups`,
        operation: OPERATION,
        variant: 'init-groups',
        bindings: [
          {
            name: 'groupAcc',
            view: groupAcc,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: groupCapacity,
        body: `let base = index * ${GROUP_ACC_STRIDE}u;
  for (var word = 0u; word < 6u; word++) {
    groupAcc[groupAccOffset + base + word] = 0u;
  }
  groupAcc[groupAccOffset + base + 6u] = 0xffffffffu;
  groupAcc[groupAccOffset + base + 7u] = 0xffffffffu;
  groupAcc[groupAccOffset + base + 8u] = 0u;
  groupAcc[groupAccOffset + base + 9u] = 0u;`
      })
    );

    // Vertices: liveness, label validation, counts, position/value sums and bounds.
    const vertexBindings: WGSLKernelBinding[] = [
      {name: 'labels', view: props.labels, type: 'u32', access: 'read'}
    ];
    if (props.nodeMask) {
      vertexBindings.push({
        name: 'vertexMask',
        view: props.nodeMask,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.topology.nodePositions) {
      vertexBindings.push({
        name: 'positions',
        view: props.topology.nodePositions,
        type: 'f32',
        access: 'read'
      });
    }
    if (props.nodeValues) {
      vertexBindings.push({
        name: 'values',
        view: props.nodeValues,
        type: 'f32',
        access: 'read'
      });
    }
    vertexBindings.push(
      {
        name: 'effectiveLabels',
        view: effectiveLabels,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'groupCounts',
        view: props.output.groupNodeCount,
        type: 'atomic<u32>',
        access: 'read_write'
      },
      {
        name: 'groupAcc',
        view: groupAcc,
        type: 'atomic<u32>',
        access: 'read_write'
      },
      {name: 'stats', view: stats, type: 'atomic<u32>', access: 'read_write'}
    );
    // Summary counters fold per workgroup, one global atomic per word per workgroup (rule 14).
    const vertexCounters = getWorkgroupCounterKernelSource({
      counters: [
        {target: 'stats', word: WORD.liveNodeCount},
        {target: 'stats', word: WORD.overflowedNodeCount},
        {target: 'stats', word: WORD.groupCount}
      ],
      rowCount: 'NODE_COUNT',
      rowSource: `${props.nodeMask ? 'if (vertexMask[vertexMaskOffset + index] == 0u) {' : 'if (false) {'}
    effectiveLabels[effectiveLabelsOffset + index] = DEAD;
    return;
  }
  addCounter(0u, 1u);
  let label = labels[labelsOffset + index];
  if (label >= GROUP_CAPACITY) {
    effectiveLabels[effectiveLabelsOffset + index] = OVERFLOWED;
    addCounter(1u, 1u);
    return;
  }
  effectiveLabels[effectiveLabelsOffset + index] = label;
  let previous = atomicAdd(&groupCounts[groupCountsOffset + label], 1u);
  if (previous == 0u) {
    addCounter(2u, 1u);
  }
  let base = label * ${GROUP_ACC_STRIDE}u;
  ${
    hasPositions
      ? `let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let quantizedX = quantize(position.x);
  let quantizedY = quantize(position.y);
  ${getFixedPointAddSource('groupAcc', 'groupAccOffset + base', 'quantizedX')}
  ${getFixedPointAddSource('groupAcc', 'groupAccOffset + base + 2u', 'quantizedY')}
  atomicMin(&groupAcc[groupAccOffset + base + 6u], getOrderedKey(position.x));
  atomicMin(&groupAcc[groupAccOffset + base + 7u], getOrderedKey(position.y));
  atomicMax(&groupAcc[groupAccOffset + base + 8u], getOrderedKey(position.x));
  atomicMax(&groupAcc[groupAccOffset + base + 9u], getOrderedKey(position.y));`
      : ''
  }
  ${
    hasValues
      ? `let quantizedValue = quantize(values[valuesOffset + index]);
  ${getFixedPointAddSource('groupAcc', 'groupAccOffset + base + 4u', 'quantizedValue')}`
      : ''
  }`
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertices`,
        operation: OPERATION,
        variant: 'vertices',
        bindings: vertexBindings,
        invocationCount: nodeCount,
        workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const GROUP_CAPACITY: u32 = ${groupCapacity}u;
const DEAD: u32 = 0xffffffffu;
const OVERFLOWED: u32 = 0xfffffffeu;
${getFixedPointSource(scale)}
${vertexCounters.declarations}`,
        body: vertexCounters.body
      })
    );
    // Slots: classify each counted slot, count intra-group edges and emit inter-group keys.
    const slotBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
      {
        name: 'effectiveLabels',
        view: effectiveLabels,
        type: 'u32',
        access: 'read'
      }
    ];
    if (props.edgeMask) {
      slotBindings.push({
        name: 'edgeMask',
        view: props.edgeMask,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.topology.weights && hasIntraWeight) {
      slotBindings.push({
        name: 'weights',
        view: props.topology.weights,
        type: 'f32',
        access: 'read'
      });
    }
    slotBindings.push({
      name: 'sortKeys',
      view: sortKeys,
      type: 'u32',
      access: 'read_write'
    });
    if (intraAcc) {
      slotBindings.push({
        name: 'intraAcc',
        view: intraAcc,
        type: 'atomic<u32>',
        access: 'read_write'
      });
    }
    slotBindings.push({
      name: 'stats',
      view: stats,
      type: 'atomic<u32>',
      access: 'read_write'
    });
    const pairKey = directed
      ? 'sourceLabel * GROUP_CAPACITY + targetLabel'
      : 'min(sourceLabel, targetLabel) * GROUP_CAPACITY + max(sourceLabel, targetLabel)';
    const slotCounters = getWorkgroupCounterKernelSource({
      counters: [
        {target: 'stats', word: WORD.countedEdgeCount},
        {target: 'stats', word: WORD.droppedEdgeCount},
        {target: 'stats', word: WORD.intraEdgeCount},
        {target: 'stats', word: WORD.interEdgeCount}
      ],
      rowCount: 'NODE_COUNT',
      rowSource: `let sourceLabel = effectiveLabels[effectiveLabelsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var counted = 0u;
  var dropped = 0u;
  var intra = 0u;
  var inter = 0u;
  var intraWeightLow = 0u;
  var intraWeightHigh = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    var key = SENTINEL_KEY;
    let targetVertex = neighbors[neighborsOffset + slot];
    if (sourceLabel != DEAD && targetVertex < NODE_COUNT) {
      let targetLabel = effectiveLabels[effectiveLabelsOffset + targetVertex];
      var live = targetLabel != DEAD;
      ${props.edgeMask ? 'if (live) { live = edgeMask[edgeMaskOffset + slot] != 0u; }' : ''}
      ${directed ? '' : 'live = live && index <= targetVertex;'}
      if (live) {
        counted++;
        if (sourceLabel == OVERFLOWED || targetLabel == OVERFLOWED) {
          dropped++;
        } else if (sourceLabel == targetLabel) {
          intra++;
          ${
            hasIntraWeight
              ? `let quantizedWeight = quantize(${props.topology.weights ? 'weights[weightsOffset + slot]' : '1.0'});
          let previousLow = intraWeightLow;
          intraWeightLow += quantizedWeight.x;
          intraWeightHigh += quantizedWeight.y + select(0u, 1u, intraWeightLow < previousLow);`
              : ''
          }
        } else {
          inter++;
          key = ${pairKey};
        }
      }
    }
    sortKeys[sortKeysOffset + slot] = key;
  }
  ${
    intraAcc
      ? `// Every intra-group slot of a row lands in the row's own group, so the row folds its intra
  // count and 64-bit fixed-point weight locally and issues one atomic set instead of one per slot
  // (integer addition is associative, so the sums are unchanged).
  if (intra > 0u) {
    let intraBase = sourceLabel * 3u;
    atomicAdd(&intraAcc[intraAccOffset + intraBase], intra);
    ${
      hasIntraWeight
        ? `let rowIntraWeight = vec2<u32>(intraWeightLow, intraWeightHigh);
    ${getFixedPointAddSource('intraAcc', 'intraAccOffset + intraBase + 1u', 'rowIntraWeight')}`
        : ''
    }
  }`
      : ''
  }
  addCounter(0u, counted);
  addCounter(1u, dropped);
  addCounter(2u, intra);
  addCounter(3u, inter);`
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slots`,
        operation: OPERATION,
        variant: 'slots',
        bindings: slotBindings,
        invocationCount: nodeCount,
        workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const SLOT_COUNT: u32 = ${slotCount}u;
const GROUP_CAPACITY: u32 = ${groupCapacity}u;
const SENTINEL_KEY: u32 = ${sentinelKey}u;
const DEAD: u32 = 0xffffffffu;
const OVERFLOWED: u32 = 0xfffffffeu;
${getFixedPointSource(scale)}
${slotCounters.declarations}`,
        body: slotCounters.body
      })
    );
    // Superedges: stable sort of (key, slot), segment numbering, then per-segment accumulation.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-iota`,
        operation: OPERATION,
        variant: 'iota',
        bindings: [
          {
            name: 'sortIndices',
            view: sortIndices,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: slotCount,
        body: 'sortIndices[sortIndicesOffset + index] = index;'
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortIndices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices,
        keyBits: getSortKeyBits(sentinelKey)
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-flags`,
        operation: OPERATION,
        variant: 'flags',
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: slotCount,
        declarations: `const SENTINEL_KEY: u32 = ${sentinelKey}u;`,
        body: `let key = sortedKeys[sortedKeysOffset + index];
  var isStart = key != SENTINEL_KEY;
  if (isStart && index > 0u) {
    isStart = sortedKeys[sortedKeysOffset + index - 1u] != key;
  }
  flags[flagsOffset + index] = select(0u, 1u, isStart);`
      })
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: flags,
        output: segmentScan,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segment-total`,
        operation: OPERATION,
        variant: 'segment-total',
        bindings: [
          {
            name: 'segmentScan',
            view: segmentScan,
            type: 'u32',
            access: 'read'
          },
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'stats', view: stats, type: 'u32', access: 'read_write'},
          {
            name: 'totalOut',
            view: segmentTotal,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'overflowOut',
            view: overflowFlag,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `let total = segmentScan[segmentScanOffset + ${slotCount - 1}u] + flags[flagsOffset + ${slotCount - 1}u];
  totalOut[totalOutOffset] = total;
  stats[statsOffset + ${WORD.superedgeCount}u] = total;
  let labelOverflow = stats[statsOffset + ${WORD.overflowedNodeCount}u] > 0u ||
    stats[statsOffset + ${WORD.droppedEdgeCount}u] > 0u;
  overflowOut[overflowOutOffset] = select(0u, 1u, labelOverflow);`
      })
    );
    const segmentBindings: WGSLKernelBinding[] = [
      {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
      {name: 'segmentScan', view: segmentScan, type: 'u32', access: 'read'},
      {
        name: 'edgeIds',
        view: props.output.edges.ids,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'edgeTargets',
        view: props.output.edgeTargetGroups,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'edgeCounts',
        view: props.output.edgeCounts,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ];
    if (segmentWeightAcc) {
      segmentBindings.push(
        {
          name: 'sortedIndices',
          view: sortedIndices,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'segmentWeightAcc',
          view: segmentWeightAcc,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      );
      if (props.topology.weights) {
        segmentBindings.push({
          name: 'weights',
          view: props.topology.weights,
          type: 'f32',
          access: 'read'
        });
      }
    }
    // Each invocation owns SEGMENT_RUN_LENGTH consecutive sorted slots and folds every run of one
    // key locally (count and 64-bit fixed-point weight), flushing one atomic set per run. A heavy
    // superedge then costs one atomic set per run instead of one per slot on a single address;
    // integer addition is associative, so counts and weights are unchanged.
    const flushRun = `if (runCount > 0u) {
      atomicAdd(&edgeCounts[edgeCountsOffset + runSegment], runCount);
      ${
        segmentWeightAcc
          ? `let runWeight = vec2<u32>(runWeightLow, runWeightHigh);
      ${getFixedPointAddSource('segmentWeightAcc', 'segmentWeightAccOffset + 2u * runSegment', 'runWeight')}`
          : ''
      }
    }`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segments`,
        operation: OPERATION,
        variant: 'segments',
        bindings: segmentBindings,
        invocationCount: Math.ceil(slotCount / SEGMENT_RUN_LENGTH),
        declarations: `const SENTINEL_KEY: u32 = ${sentinelKey}u;
const GROUP_CAPACITY: u32 = ${groupCapacity}u;
const EDGE_CAPACITY: u32 = ${edgeCapacity}u;
const SLOT_COUNT: u32 = ${slotCount}u;
${getFixedPointSource(scale)}`,
        body: `let first = index * ${SEGMENT_RUN_LENGTH}u;
  let last = min(first + ${SEGMENT_RUN_LENGTH}u, SLOT_COUNT);
  var runSegment = 0xffffffffu;
  var runCount = 0u;
  var runWeightLow = 0u;
  var runWeightHigh = 0u;
  for (var position = first; position < last; position++) {
    let key = sortedKeys[sortedKeysOffset + position];
    if (key == SENTINEL_KEY) {
      break;
    }
    var isStart = true;
    if (position > 0u) {
      isStart = sortedKeys[sortedKeysOffset + position - 1u] != key;
    }
    let segment = segmentScan[segmentScanOffset + position] + select(0u, 1u, isStart) - 1u;
    if (segment != runSegment) {
      ${flushRun}
      runSegment = segment;
      runCount = 0u;
      runWeightLow = 0u;
      runWeightHigh = 0u;
    }
    if (segment >= EDGE_CAPACITY) {
      continue;
    }
    if (isStart) {
      edgeIds[edgeIdsOffset + segment] = key / GROUP_CAPACITY;
      edgeTargets[edgeTargetsOffset + segment] = key % GROUP_CAPACITY;
    }
    runCount++;
    ${
      segmentWeightAcc
        ? `let quantizedWeight = quantize(${props.topology.weights ? 'weights[weightsOffset + sortedIndices[sortedIndicesOffset + position]]' : '1.0'});
    let previousLow = runWeightLow;
    runWeightLow += quantizedWeight.x;
    runWeightHigh += quantizedWeight.y + select(0u, 1u, runWeightLow < previousLow);`
        : ''
    }
  }
  ${flushRun}`
      })
    );
    if (segmentWeightAcc && props.output.edgeWeights) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-edge-weights`,
          operation: OPERATION,
          variant: 'edge-weights',
          bindings: [
            {
              name: 'segmentWeightAcc',
              view: segmentWeightAcc,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'edgeWeights',
              view: props.output.edgeWeights,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: edgeCapacity,
          declarations: getFixedPointSource(scale),
          body: `edgeWeights[edgeWeightsOffset + index] = dequantize(
    segmentWeightAcc[segmentWeightAccOffset + 2u * index],
    segmentWeightAcc[segmentWeightAccOffset + 2u * index + 1u]);`
        })
      );
    }

    // Per-group finish: centroids, bounds, value sums and intra-group outputs.
    const finishBindings: WGSLKernelBinding[] = [
      {
        name: 'groupCounts',
        view: props.output.groupNodeCount,
        type: 'u32',
        access: 'read'
      },
      {name: 'groupAcc', view: groupAcc, type: 'u32', access: 'read'}
    ];
    const finishStatements: string[] = [];
    if (props.output.groupCentroid) {
      finishBindings.push({
        name: 'centroidOut',
        view: props.output.groupCentroid,
        type: 'f32',
        access: 'read_write'
      });
      finishStatements.push(`centroidOut[centroidOutOffset + 2u * index] = select(0.0, dequantize(groupAcc[groupAccOffset + base], groupAcc[groupAccOffset + base + 1u]) / f32(count), count > 0u);
  centroidOut[centroidOutOffset + 2u * index + 1u] = select(0.0, dequantize(groupAcc[groupAccOffset + base + 2u], groupAcc[groupAccOffset + base + 3u]) / f32(count), count > 0u);`);
    }
    if (props.output.groupBounds) {
      finishBindings.push({
        name: 'boundsOut',
        view: props.output.groupBounds,
        type: 'f32',
        access: 'read_write'
      });
      finishStatements.push(`boundsOut[boundsOutOffset + 4u * index] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 6u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 1u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 7u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 2u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 8u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 3u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 9u]), count > 0u);`);
    }
    if (props.output.groupValueSum) {
      finishBindings.push({
        name: 'valueOut',
        view: props.output.groupValueSum,
        type: 'f32',
        access: 'read_write'
      });
      finishStatements.push(
        'valueOut[valueOutOffset + index] = dequantize(groupAcc[groupAccOffset + base + 4u], groupAcc[groupAccOffset + base + 5u]);'
      );
    }
    if (intraAcc) {
      finishBindings.push({
        name: 'intraAcc',
        view: intraAcc,
        type: 'u32',
        access: 'read'
      });
      if (props.output.groupIntraEdgeCount) {
        finishBindings.push({
          name: 'intraCountOut',
          view: props.output.groupIntraEdgeCount,
          type: 'u32',
          access: 'read_write'
        });
        finishStatements.push(
          'intraCountOut[intraCountOutOffset + index] = intraAcc[intraAccOffset + 3u * index];'
        );
      }
      if (props.output.groupIntraWeight) {
        finishBindings.push({
          name: 'intraWeightOut',
          view: props.output.groupIntraWeight,
          type: 'f32',
          access: 'read_write'
        });
        finishStatements.push(
          'intraWeightOut[intraWeightOutOffset + index] = dequantize(intraAcc[intraAccOffset + 3u * index + 1u], intraAcc[intraAccOffset + 3u * index + 2u]);'
        );
      }
    }
    if (finishStatements.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-groups`,
          operation: OPERATION,
          variant: 'groups',
          bindings: finishBindings,
          invocationCount: groupCapacity,
          declarations: getFixedPointSource(scale),
          body: `let base = index * ${GROUP_ACC_STRIDE}u;
  let count = groupCounts[groupCountsOffset + index];
  ${finishStatements.join('\n  ')}`
        })
      );
    }

    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: segmentTotal,
        output: props.output.edges,
        overflowSources: [overflowFlag]
      })
    );
    return nodes;
  }
}
