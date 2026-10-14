// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {validateGPUStatusPort} from '../../analysis-core/contracts/index';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {GPUNetworkReachability} from '../network-reachability/index';
import {recommendLaneCount} from '../network-accessibility/network-accessibility-lanes';

const OPERATION = 'GPUNetworkCostTable';
const DEFAULT_LOCAL_ITERATIONS = 16;
/** Workgroup size of the per-lane count, scan and fill kernels. Fixed, so slot order is fixed. */
const WORKGROUP_SIZE = 256;
/** Start node of an unused lane: no expanded node. */
const NO_START = 0xffffffff;
/** Words of the GPU state buffer: entry cursor, row cursor, converged flag, spare. */
const STATE_WORDS = 4;

/** Caller-owned outputs of {@link GPUNetworkCostTable}. */
export type GPUNetworkCostTableOutput = {
  /**
   * `nodeCount + 1` exclusive row offsets into `targets` and `costs`, clamped to the capacity so
   * the CSR is always consistent (possibly truncated).
   */
  offsets: GraphDataView<'uint32'>;
  /**
   * Target node per entry, ascending within each row. Its length is the entry capacity. Row `u`
   * lists every node `v` whose network cost from `u` is finite and at most `costLimit`, including
   * `u` itself at cost 0.
   */
  targets: GraphDataView<'uint32'>;
  /** Network cost per entry, aligned with `targets`. */
  costs: GraphDataView<'float32'>;
  /** Optional GPU-written status words. */
  status?: {
    /** Optional one-row flag: 1 when the entries exceed the capacity, so the table is truncated. */
    overflow?: GraphDataView<'uint32'>;
    /**
     * Optional one-row exact (unclamped) number of entries, saturating at `0xffffffff`, for the
     * grow loop that sizes the capacity.
     */
    requiredCount?: GraphDataView<'uint32'>;
    /** Optional one-row flag: 1 when every batch reached a fixpoint within `maximumIterations`. */
    converged?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkCostTable}.
 *
 * Compile-time: node and edge counts, `laneCount`, `maximumIterations`, `localIterations`, the
 * entry capacity (`output.targets` length) and which status views exist. Per-frame: the contents of
 * the topology (live weights, closures) and `costLimit`.
 */
export type GPUNetworkCostTableProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-cost-table'`. */
  id?: string;
  /**
   * The network, with `weights` as the travel cost. Targets `>= nodeCount` are ignored and negative
   * or NaN weights are impassable. For {@link GPUMapMatching} pass weights equal to the edge
   * lengths it measures from node positions.
   */
  topology: GPUNetworkTopology;
  /** One-row per-frame cost bound `delta`: the table lists every cost up to and including it. */
  costLimit: GraphDataView<'float32'>;
  /**
   * Compile-time source nodes searched together in one lane-expanded reachability batch. Defaults
   * to `recommendLaneCount()` with one row per node: about 1M expanded nodes per batch, at least
   * 32 lanes, at most `nodeCount`. The table is identical for every lane count.
   */
  laneCount?: number;
  /** Rounds of every batch, as `GPUNetworkReachability.maximumIterations`. Defaults to 64. */
  maximumIterations?: number;
  /** Hops chained per workgroup per round, as `GPUNetworkReachability.localIterations`. Defaults to 16. */
  localIterations?: number;
  /** The GPU-written views. */
  output: GPUNetworkCostTableOutput;
};

/**
 * Precomputes a bounded all-pairs cost table of a static directed network, in the spirit of the
 * UBODT of fast map matching (Yang and Gidofalvi 2018): for every node `u`, every node `v` whose
 * shortest-path cost from `u` is at most `costLimit`, as a CSR with ascending targets so a consumer
 * binary-searches a row (`GPUMapMatching.routeTable`).
 *
 * Source nodes are searched `laneCount` at a time with one lane-expanded `GPUNetworkReachability`
 * per batch into one scratch block, then three kernels per batch compact the block: a per-lane
 * count (one workgroup per lane), a single-workgroup scan of the batch's lane counts on top of a
 * GPU running entry cursor that writes the batch's `offsets` rows, and a per-lane ordered fill.
 * Costs are the f32 fixpoint of reachability, so they equal the single-source costs for every lane
 * count. There are no float atomics, one search per row, and results are deterministic. Per-batch
 * values (first row, running entry count) live in a GPU state buffer, so the batch count does not
 * change any pipeline.
 *
 * Capacity: `output.targets.length` entries. Entries past it are dropped and the offsets are
 * clamped, so a reader always sees a consistent (possibly truncated) CSR;
 * `output.status.overflow` and `output.status.requiredCount` report it. A consumer must not use a
 * table that overflowed or did not converge.
 *
 * Scaling: each batch sweeps its `laneCount x nodeCount` scratch block (dense lane layout) a few
 * times, so work and memory traffic are `O(nodeCount^2 / laneCount)` words per batch and
 * `O(nodeCount^2)` words in all, plus the searches. That targets static networks up to roughly
 * 1e5 nodes, built once and reused across many encodings; the table itself holds only the entries
 * within `costLimit`. A network whose weights change per frame can rebuild it, at that price.
 */
export class GPUNetworkCostTable implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkCostTableProps;
  /** Number of network nodes, which is also the number of table rows. */
  readonly nodeCount: number;
  /** Resolved source nodes per batch. */
  readonly laneCount: number;
  /** Number of lane-expanded reachability batches. */
  readonly batchCount: number;
  /** Entry capacity, `output.targets.length`. */
  readonly capacity: number;

  /**
   * Validates the topology, `costLimit` and output layout.
   *
   * @throws If a view has the wrong format or length, `laneCount` is not an integer in
   * `[1, nodeCount]`, `laneCount * nodeCount` does not fit in uint32, or an output shares a buffer
   * with an input or with another output.
   */
  constructor(props: GPUNetworkCostTableProps) {
    const id = props.id ?? 'network-cost-table';
    this.id = id;
    this.props = props;
    const {output} = props;
    const {nodeCount, edgeCount} = validateGPUNetworkTopology(id, props.topology, {weights: true});
    this.nodeCount = nodeCount;
    validatePackedView(props.costLimit, ['float32'], `${id} costLimit`);
    if (props.costLimit.length !== 1) {
      throw new Error(`${id} costLimit must contain exactly one row`);
    }
    validatePackedView(output.offsets, ['uint32'], `${id} output.offsets`);
    if (output.offsets.length !== nodeCount + 1) {
      throw new Error(`${id} output.offsets must contain nodeCount + 1 rows`);
    }
    validatePackedView(output.targets, ['uint32'], `${id} output.targets`);
    validatePackedView(output.costs, ['float32'], `${id} output.costs`);
    this.capacity = output.targets.length;
    validateCapacity(id, 'output.targets length', this.capacity);
    if (output.costs.length !== this.capacity) {
      throw new Error(`${id} output.costs length must equal output.targets length`);
    }
    validateGPUStatusPort(id, {
      overflow: output.status?.overflow,
      requiredCount: output.status?.requiredCount
    });
    const converged = output.status?.converged;
    for (const [name, view] of [
      ['overflow', output.status?.overflow],
      ['requiredCount', output.status?.requiredCount],
      ['converged', converged]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.status.${name}`);
        if (view.length !== 1) {
          throw new Error(`${id} output.status.${name} must contain exactly one row`);
        }
      }
    }
    this.laneCount =
      props.laneCount ?? recommendLaneCount({rowCount: nodeCount, nodeCount, edgeCount});
    validateCapacity(id, 'laneCount', this.laneCount, 1, nodeCount);
    if (this.laneCount * nodeCount >= NO_START) {
      throw new Error(`${id} laneCount * nodeCount must fit in uint32`);
    }
    this.batchCount = Math.ceil(nodeCount / this.laneCount);
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns `${id}-init`, then per batch `${id}-batch-${batch}-seeds`, the nodes of one
   * `GPUNetworkReachability` (IDs `${id}-batch-${batch}-*`), and `-count`, `-scan` and `-fill`,
   * then `${id}-finalize`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, laneCount, batchCount, capacity} = this;
    const {output} = props;
    const {offsets, neighbors, weights} = props.topology;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const batchNodeCount = laneCount * nodeCount;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'frontier queues',
      2 * batchNodeCount * 4,
      'lower laneCount'
    );
    const converged = output.status?.converged;
    const state = createTransientView(graph, `${id}-state`, 'uint32', STATE_WORDS);
    const seeds = createTransientView(graph, `${id}-seeds`, 'uint32', laneCount);
    const scratch = createTransientView(graph, `${id}-scratch`, 'float32', batchNodeCount);
    const laneCounts = createTransientView(graph, `${id}-lane-counts`, 'uint32', laneCount);
    const laneBases = createTransientView(graph, `${id}-lane-bases`, 'uint32', laneCount);
    const batchConverged = converged
      ? createTransientView(graph, `${id}-batch-converged`, 'uint32', 1)
      : undefined;
    const declarations = `const NODE_COUNT: u32 = ${nodeCount}u;
const LANE_COUNT: u32 = ${laneCount}u;
const CAPACITY: u32 = ${capacity}u;
const NO_START: u32 = ${NO_START}u;
// state[0]: entries written so far (unclamped, saturating); state[1]: first row of the next
// batch; state[2]: all batches converged so far.
const ENTRY_CURSOR: u32 = 0u;
const ROW_CURSOR: u32 = 1u;
const CONVERGED_FLAG: u32 = 2u;
${IS_FINITE_BITS_WGSL}
fn addSaturated(left: u32, right: u32) -> u32 {
  return select(0xffffffffu, left + right, left <= 0xffffffffu - right);
}
fn isTableEntry(cost: f32) -> bool {
  // Unreached entries are +Infinity.
  return isFiniteBits(cost) && cost >= 0.0;
}`;
    const read = (name: string, view: GraphDataView, type: WGSLKernelBinding['type'] = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: WGSLKernelBinding['type'] = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-init`,
        operation: OPERATION,
        variant: 'init',
        bindings: [write('state', state)],
        invocationCount: 1,
        body: `state[stateOffset + 0u] = 0u;
  state[stateOffset + 1u] = 0u;
  state[stateOffset + 2u] = 1u;
  state[stateOffset + 3u] = 0u;`
      })
    ];
    for (let batch = 0; batch < batchCount; batch++) {
      nodes.push(
        // Lane `l` searches from node `firstRow + l`, which is expanded node `l * nodeCount + row`.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-${batch}-seeds`,
          operation: OPERATION,
          variant: 'seeds',
          bindings: [read('state', state), write('seeds', seeds)],
          invocationCount: laneCount,
          declarations,
          body: `let row = state[stateOffset + ROW_CURSOR] + index;
  seeds[seedsOffset + index] = select(NO_START, index * NODE_COUNT + row, row < NODE_COUNT);`
        }),
        ...new GPUNetworkReachability({
          id: `${id}-batch-${batch}`,
          topology: {offsets, neighbors, weights},
          laneCount,
          startNodes: seeds,
          costLimit: props.costLimit,
          maximumIterations: props.maximumIterations,
          localIterations: props.localIterations ?? DEFAULT_LOCAL_ITERATIONS,
          output: {costs: scratch, status: {converged: batchConverged}}
        }).getCommandNodes(graph),
        // One workgroup per lane counts the finite entries of the lane's costs.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-${batch}-count`,
          operation: OPERATION,
          variant: 'count',
          bindings: [read('scratch', scratch, 'f32'), write('laneCounts', laneCounts)],
          workgroupSize: WORKGROUP_SIZE,
          invocationCount: laneCount * WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${declarations}
var<workgroup> partials: array<u32, ${WORKGROUP_SIZE}>;`,
          // No early return: every invocation of a workgroup must reach the barriers.
          body: `let lane = index / ${WORKGROUP_SIZE}u;
  var count = 0u;
  for (var node = localInvocationIndex; node < NODE_COUNT; node += ${WORKGROUP_SIZE}u) {
    if (isTableEntry(scratch[scratchOffset + lane * NODE_COUNT + node])) {
      count += 1u;
    }
  }
  partials[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    laneCounts[laneCountsOffset + lane] = partials[0];
  }`
        }),
        // One workgroup scans the lane counts on top of the running entry cursor, writes the
        // batch's offsets rows and each lane's unclamped first entry, then advances the cursors.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-${batch}-scan`,
          operation: OPERATION,
          variant: 'scan',
          bindings: [
            write('state', state),
            read('laneCounts', laneCounts),
            write('laneBases', laneBases),
            write('offsets', output.offsets),
            ...(batchConverged ? [read('batchConverged', batchConverged)] : [])
          ],
          workgroupSize: WORKGROUP_SIZE,
          invocationCount: WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${declarations}
var<workgroup> prefix: array<u32, ${WORKGROUP_SIZE}>;`,
          // Every thread reads the cursors before the first barrier; only thread 0 writes them,
          // after the last one.
          body: `let firstRow = state[stateOffset + ROW_CURSOR];
  var base = state[stateOffset + ENTRY_CURSOR];
  for (var chunk = 0u; chunk < LANE_COUNT; chunk += ${WORKGROUP_SIZE}u) {
    let lane = chunk + localInvocationIndex;
    var count = 0u;
    if (lane < LANE_COUNT) {
      count = laneCounts[laneCountsOffset + lane];
    }
    prefix[localInvocationIndex] = count;
    workgroupBarrier();
    for (var stride = 1u; stride < ${WORKGROUP_SIZE}u; stride = stride * 2u) {
      var addend = 0u;
      if (localInvocationIndex >= stride) {
        addend = prefix[localInvocationIndex - stride];
      }
      workgroupBarrier();
      prefix[localInvocationIndex] += addend;
      workgroupBarrier();
    }
    if (lane < LANE_COUNT) {
      let start = addSaturated(base, prefix[localInvocationIndex] - count);
      laneBases[laneBasesOffset + lane] = start;
      if (firstRow + lane < NODE_COUNT) {
        offsets[offsetsOffset + firstRow + lane] = min(start, CAPACITY);
      }
    }
    base = addSaturated(base, prefix[${WORKGROUP_SIZE - 1}u]);
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    state[stateOffset + ENTRY_CURSOR] = base;
    state[stateOffset + ROW_CURSOR] = firstRow + LANE_COUNT;
    // The last batch's write wins, and it holds the total.
    offsets[offsetsOffset + NODE_COUNT] = min(base, CAPACITY);
    ${
      batchConverged
        ? 'state[stateOffset + CONVERGED_FLAG] = min(state[stateOffset + CONVERGED_FLAG], batchConverged[batchConvergedOffset]);'
        : ''
    }
  }`
        }),
        // One workgroup per lane writes the lane's entries in ascending node order: a chunk of 256
        // nodes at a time, an inclusive scan of the entry flags gives each its slot.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-${batch}-fill`,
          operation: OPERATION,
          variant: 'fill',
          bindings: [
            read('scratch', scratch, 'f32'),
            read('laneBases', laneBases),
            write('targets', output.targets),
            write('costs', output.costs, 'f32')
          ],
          workgroupSize: WORKGROUP_SIZE,
          invocationCount: laneCount * WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${declarations}
var<workgroup> prefix: array<u32, ${WORKGROUP_SIZE}>;`,
          body: `let lane = index / ${WORKGROUP_SIZE}u;
  var base = laneBases[laneBasesOffset + lane];
  for (var chunk = 0u; chunk < NODE_COUNT; chunk += ${WORKGROUP_SIZE}u) {
    let node = chunk + localInvocationIndex;
    var cost = -1.0;
    if (node < NODE_COUNT) {
      cost = scratch[scratchOffset + lane * NODE_COUNT + node];
    }
    let isIncluded = isTableEntry(cost);
    prefix[localInvocationIndex] = select(0u, 1u, isIncluded);
    workgroupBarrier();
    for (var stride = 1u; stride < ${WORKGROUP_SIZE}u; stride = stride * 2u) {
      var addend = 0u;
      if (localInvocationIndex >= stride) {
        addend = prefix[localInvocationIndex - stride];
      }
      workgroupBarrier();
      prefix[localInvocationIndex] += addend;
      workgroupBarrier();
    }
    let rank = prefix[localInvocationIndex] - 1u;
    // The base saturates, so test the slot without adding past the capacity.
    if (isIncluded && base < CAPACITY && rank < CAPACITY - base) {
      targets[targetsOffset + base + rank] = node;
      costs[costsOffset + base + rank] = cost;
    }
    base = addSaturated(base, prefix[${WORKGROUP_SIZE - 1}u]);
    workgroupBarrier();
  }`
        })
      );
    }
    const finalizeBindings: WGSLKernelBinding[] = [read('state', state)];
    if (output.status?.overflow) {
      finalizeBindings.push(write('overflow', output.status.overflow));
    }
    if (output.status?.requiredCount) {
      finalizeBindings.push(write('requiredCount', output.status.requiredCount));
    }
    if (converged) {
      finalizeBindings.push(write('converged', converged));
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: finalizeBindings,
        invocationCount: 1,
        declarations,
        body: `let total = state[stateOffset + ENTRY_CURSOR];
  ${output.status?.overflow ? 'overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);' : ''}
  ${output.status?.requiredCount ? 'requiredCount[requiredCountOffset] = total;' : ''}
  ${converged ? 'converged[convergedOffset] = state[stateOffset + CONVERGED_FLAG];' : ''}`
      })
    );
    return nodes;
  }
}

/** Returns every read-only view of a cost table. */
function getInputs(props: GPUNetworkCostTableProps): (GraphDataView | undefined)[] {
  return [...getGPUNetworkTopologyViews(props.topology), props.costLimit];
}

/** Returns every written view of a cost table. */
function getOutputs(props: GPUNetworkCostTableProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.offsets,
    output.targets,
    output.costs,
    output.status?.overflow,
    output.status?.requiredCount,
    output.status?.converged
  ];
}
