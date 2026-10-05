// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  getBoundedDispatchLayout,
  GPUGroupAggregation,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  GPUNetworkReachability
} from '../network-reachability/gpu-network-reachability';
import {createReachabilityGateNode} from '../network-reachability/network-reachability-passes';
import {
  createServiceAreasFinalizeNode,
  createServiceAreasLabelInitializeNode,
  createServiceAreasLabelRelaxNode,
  createServiceAreasLabelSeedNode
} from './network-service-areas-passes';

const DEFAULT_MAXIMUM_ITERATIONS = 64;

/**
 * Properties for {@link GPUNetworkServiceAreas}.
 *
 * Compile-time: node and edge counts, facility capacity, `maxIterations`, and which optional views
 * exist. Per-frame: the contents of the CSR, facilities, facility costs and count, `costLimit`,
 * and `activeIterations`.
 */
export type GPUNetworkServiceAreasProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-service-areas'`. */
  id?: string;
  /** Forward CSR row offsets with `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** CSR destination node per edge. Indices `>= nodeCount` are ignored. */
  neighbors: GraphDataView<'uint32'>;
  /** Non-negative travel cost per edge. Negative or NaN edges are impassable. */
  weights: GraphDataView<'float32'>;
  /**
   * Facility node indices; the facility ID is the row. Compile-time capacity, per-frame contents.
   * Out-of-range nodes are ignored.
   */
  facilities: GraphDataView<'uint32'>;
  /** Optional per-facility starting cost, for example handling time. Per-frame. Defaults to 0. */
  facilityCosts?: GraphDataView<'float32'>;
  /** Optional one-row active facility count; rows at or after it are ignored. Per-frame. */
  facilityCount?: GraphDataView<'uint32'>;
  /** Optional one-row cost cutoff; nodes beyond it stay unassigned. Per-frame. */
  costLimit?: GraphDataView<'float32'>;
  /**
   * Compile-time iteration bound for EACH of the two phases, 1 to
   * `GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS`. Defaults to 64.
   */
  maxIterations?: number;
  /** Optional one-row per-frame iteration limit for each phase, clamped to `maxIterations`. */
  activeIterations?: GraphDataView<'uint32'>;
  /**
   * Per-node assigned facility row, or `GPU_NETWORK_REACHABILITY_NONE` when no facility reaches
   * the node. Its length defines the node count.
   */
  assignments: GraphDataView<'uint32'>;
  /**
   * Optional per-node minimum cost to the nearest facility, `+Infinity` when unassigned. A graph
   * transient is used when omitted.
   */
  costs?: GraphDataView<'float32'>;
  /** Optional per-node shortest-path predecessor, forwarded to {@link GPUNetworkReachability}. */
  predecessors?: GraphDataView<'uint32'>;
  /** Optional node count per facility row. Its length must equal `facilities.length`. */
  facilityNodeCounts?: GraphDataView<'uint32'>;
  /** Optional sum of node costs per facility row. Its length must equal `facilities.length`. */
  facilityCostSums?: GraphDataView<'float32'>;
  /** Optional one-row flag: 1 only when both phases reached a fixpoint, otherwise 0. */
  converged?: GraphDataView<'uint32'>;
};

/**
 * Assigns every node of a directed CSR road network to its nearest facility ("Voronoi on the
 * network", the evaluation step of location-allocation), with per-facility node counts and total
 * cost served.
 *
 * Ties are broken deterministically: when several facilities achieve the minimum cost at a node,
 * the smallest facility row wins.
 *
 * Design: two phases rather than one packed (cost, facility) relaxation. Phase 1 composes
 * {@link GPUNetworkReachability} with the facilities as multi-source seeds and yields exact f32
 * costs plus the GPU convergence gate. Phase 2 propagates the minimum facility label over the
 * tight-edge subgraph: an edge `u -> v` is tight when `costs[u] + weight == costs[v]` in f32
 * (zero-weight edges included, so zero-cost connectors keep their facility). The label fixpoint
 * is the smallest facility row among all facilities that achieve the minimum cost, regardless of
 * atomic scheduling. A packed relaxation is not used because WGSL has no 64-bit atomics, so
 * packing cost and facility into one u32 would either quantize costs or cap the facility count.
 * Reachability predecessors are not used for labels because they pick the smallest node index
 * rather than facility row and are `NONE` across zero-weight edges.
 *
 * Every encoding recomputes from scratch and rewrites every output word; nothing is read back.
 *
 * @remarks Non-goals: capacity constraints, facility selection (p-median or coverage
 * optimization), turn penalties, and time-dependent costs. Facility costs and edge weights are
 * f32; assignments follow the f32 tight-edge rule.
 */
export class GPUNetworkServiceAreas implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'network-service-areas';
  /** Validated properties. */
  readonly props: GPUNetworkServiceAreasProps;
  /** Resolved compile-time iteration bound per phase. */
  readonly maxIterations: number;

  constructor(props: GPUNetworkServiceAreasProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    this.maxIterations = props.maxIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    const {id} = this;
    for (const [name, view] of [
      ['offsets', props.offsets],
      ['neighbors', props.neighbors],
      ['facilities', props.facilities],
      ['facilityCount', props.facilityCount],
      ['activeIterations', props.activeIterations],
      ['assignments', props.assignments],
      ['predecessors', props.predecessors],
      ['facilityNodeCounts', props.facilityNodeCounts],
      ['converged', props.converged]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['weights', props.weights],
      ['facilityCosts', props.facilityCosts],
      ['costLimit', props.costLimit],
      ['costs', props.costs],
      ['facilityCostSums', props.facilityCostSums]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    const nodeCount = props.assignments.length;
    if (nodeCount < 1) {
      throw new Error(`${id} assignments must contain at least one node`);
    }
    if (props.offsets.length !== nodeCount + 1) {
      throw new Error(`${id} offsets must contain one more row than assignments`);
    }
    if (props.weights.length !== props.neighbors.length) {
      throw new Error(`${id} weights length must equal neighbors length`);
    }
    if (props.facilityCosts && props.facilityCosts.length !== props.facilities.length) {
      throw new Error(`${id} facilityCosts length must equal facilities length`);
    }
    for (const [name, view] of [
      ['facilityNodeCounts', props.facilityNodeCounts],
      ['facilityCostSums', props.facilityCostSums]
    ] as const) {
      if (view && view.length !== props.facilities.length) {
        throw new Error(`${id} ${name} length must equal facilities length`);
      }
    }
    for (const [name, view] of [
      ['facilityCount', props.facilityCount],
      ['activeIterations', props.activeIterations],
      ['costLimit', props.costLimit],
      ['converged', props.converged]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    for (const [name, view] of [
      ['costs', props.costs],
      ['predecessors', props.predecessors]
    ] as const) {
      if (view && view.length !== nodeCount) {
        throw new Error(`${id} ${name} length must equal the node count`);
      }
    }
    if (
      !Number.isSafeInteger(this.maxIterations) ||
      this.maxIterations < 1 ||
      this.maxIterations > GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
    ) {
      throw new Error(
        `${id} maxIterations must be an integer in [1, ${GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS}]`
      );
    }
    const outputs = [
      props.assignments,
      props.costs,
      props.predecessors,
      props.facilityNodeCounts,
      props.facilityCostSums,
      props.converged
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const inputs = [
      props.offsets,
      props.neighbors,
      props.weights,
      props.facilities,
      props.facilityCosts,
      props.facilityCount,
      props.activeIterations,
      props.costLimit
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (
      new Set(outputs).size !== outputs.length ||
      outputs.some(buffer => inputs.includes(buffer))
    ) {
      throw new Error(`${id} outputs must use separate buffers from each other and from inputs`);
    }
  }

  /**
   * Returns the reachability nodes, then label initialize, seed, gated relax and gate pairs, and
   * optional finalize and per-facility aggregation nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maxIterations} = this;
    validateGraphViewsBelongToGraph(id, graph, getViews(props));
    const nodeCount = props.assignments.length;
    const costs = props.costs ?? createTransientView(graph, `${id}-costs`, 'float32', nodeCount);
    const reachabilityConverged = props.converged
      ? createTransientView(graph, `${id}-reachability-converged`, 'uint32', 1)
      : undefined;
    const labels = props.assignments;
    const frontierA = createTransientView(graph, `${id}-label-frontier-a`, 'uint32', nodeCount);
    const frontierB = createTransientView(graph, `${id}-label-frontier-b`, 'uint32', nodeCount);
    const status = createTransientView(graph, `${id}-label-status`, 'uint32', 4);
    // The indirect command lives in its own buffer: one buffer must not be both an indirect
    // argument and storage-written within one dispatch.
    const dispatch = createTransientView(
      graph,
      `${id}-label-dispatch`,
      'uint32',
      3,
      Buffer.STORAGE | Buffer.INDIRECT
    );
    const relaxLayout = getBoundedDispatchLayout(
      'GPUNetworkServiceAreas',
      nodeCount,
      256,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const csr = {offsets: props.offsets, neighbors: props.neighbors, weights: props.weights};
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUNetworkReachability({
        id: `${id}-reachability`,
        offsets: props.offsets,
        neighbors: props.neighbors,
        weights: props.weights,
        sources: props.facilities,
        sourceCosts: props.facilityCosts,
        sourceCount: props.facilityCount,
        costLimit: props.costLimit,
        maxIterations,
        activeIterations: props.activeIterations,
        costs,
        predecessors: props.predecessors,
        converged: reachabilityConverged
      }).getCommandNodes(graph),
      createServiceAreasLabelInitializeNode<Parameters>(graph, {
        id: `${id}-label-initialize`,
        nodeCount,
        maxIterations,
        relaxLayout,
        labels,
        frontierA,
        frontierB,
        status,
        dispatch,
        activeIterations: props.activeIterations
      })
    ];
    if (props.facilities.length > 0) {
      nodes.push(
        createServiceAreasLabelSeedNode<Parameters>(graph, {
          id: `${id}-label-seed`,
          nodeCount,
          facilities: props.facilities,
          facilityCosts: props.facilityCosts,
          facilityCount: props.facilityCount,
          costLimit: props.costLimit,
          costs,
          labels,
          frontier: frontierA
        })
      );
    }
    for (let iteration = 0; iteration < maxIterations; iteration++) {
      const even = iteration % 2 === 0;
      nodes.push(
        createServiceAreasLabelRelaxNode<Parameters>(graph, {
          id: `${id}-label-relax-${iteration}`,
          nodeCount,
          csr,
          costs,
          current: even ? frontierA : frontierB,
          next: even ? frontierB : frontierA,
          labels,
          status,
          dispatch
        }),
        createReachabilityGateNode<Parameters>(graph, {
          id: `${id}-label-gate-${iteration}`,
          iteration,
          maxIterations,
          status,
          dispatch,
          activeIterations: props.activeIterations
        })
      );
    }
    if (props.converged && reachabilityConverged) {
      nodes.push(
        createServiceAreasFinalizeNode<Parameters>(graph, {
          id: `${id}-finalize`,
          status,
          reachabilityConverged,
          converged: props.converged
        })
      );
    }
    // GPUGroupAggregation skips out-of-range keys (NONE) and non-finite values, so unassigned
    // rows with +Infinity costs never contribute and no mask is needed.
    if (props.facilityNodeCounts && props.facilities.length > 0) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-facility-node-counts`,
          keys: props.assignments,
          output: props.facilityNodeCounts
        }).getCommandNodes(graph)
      );
    }
    if (props.facilityCostSums && props.facilities.length > 0) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-facility-cost-sums`,
          keys: props.assignments,
          values: costs,
          output: props.facilityCostSums,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}

/** Returns every view referenced by the props. */
function getViews(props: GPUNetworkServiceAreasProps): (GraphDataView | undefined)[] {
  return [
    props.offsets,
    props.neighbors,
    props.weights,
    props.facilities,
    props.facilityCosts,
    props.facilityCount,
    props.costLimit,
    props.activeIterations,
    props.assignments,
    props.costs,
    props.predecessors,
    props.facilityNodeCounts,
    props.facilityCostSums,
    props.converged
  ];
}
