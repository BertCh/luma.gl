// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';
import {
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS,
  GPUNetworkReachability
} from '../network-reachability/gpu-network-reachability';
import {
  createFrontierState,
  getFrontierPhaseLayout,
  getFrontierPhaseMaximumStamp,
  type FrontierPhase
} from '../network-reachability/network-frontier';
import {
  createReachabilityPredecessorsNode,
  createReachabilityTieInitializeNode,
  createReachabilityTieLevelNode,
  createReachabilityTiePredecessorsNode,
  createReachabilityTieRootsNode,
  createReachabilityTieSeedNode
} from '../network-reachability/network-reachability-passes';
import {createFillNode} from '../../utils/wgsl-kernel-nodes';
import {
  createServiceAreasFinalizeNode,
  createServiceAreasLabelFrontierSeedNode,
  createServiceAreasLabelInitializeNode,
  createServiceAreasLabelRoundNode,
  createServiceAreasLabelSeedNode,
  createServiceAreasTightEdgesNode
} from './network-service-areas-passes';

const DEFAULT_MAXIMUM_ITERATIONS = 64;
const DEFAULT_MAXIMUM_LABEL_ITERATIONS = 32;
const DEFAULT_MAXIMUM_TIE_ITERATIONS = 4;

/**
 * Properties for {@link GPUNetworkServiceAreas}.
 *
 * Compile-time: node and edge counts, facility capacity, `maximumIterations`, a numeric
 * `facilityCount`, and which optional views exist. Per-frame: the contents of the topology,
 * facilities, facility costs, a `facilityCount` view, `costLimit`, and `activeIterations`.
 */
export type GPUNetworkServiceAreasProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-service-areas'`. */
  id?: string;
  /**
   * Directed network. `neighbors` indices `>= nodeCount` are ignored; negative or NaN `weights` are
   * impassable. `offsets` must have `output.assignments.length + 1` rows.
   */
  topology: GPUNetworkTopology;
  /**
   * Facility node indices; the facility ID is the row. Compile-time capacity, per-frame contents.
   * Out-of-range nodes are ignored.
   */
  facilities: GraphDataView<'uint32'>;
  /** Optional per-facility starting cost, for example handling time. Per-frame. Defaults to 0. */
  facilityCosts?: GraphDataView<'float32'>;
  /**
   * Optional active facility count; rows at or after it are ignored. A number is a compile-time
   * constant in `[0, facilities.length]`; a one-row view is read per frame. Defaults to every row.
   */
  facilityCount?: number | GraphDataView<'uint32'>;
  /** Optional one-row cost cutoff; nodes beyond it stay unassigned. Per-frame. */
  costLimit?: GraphDataView<'float32'>;
  /**
   * Compile-time round bound of the cost phase, 1 to `GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS`.
   * Defaults to 64.
   */
  maximumIterations?: number;
  /**
   * Compile-time round bound of the label phase, 1 to
   * `GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS`. Each round is one graph node. Defaults to
   * `min(maximumIterations, 32)`.
   */
  maximumLabelIterations?: number;
  /**
   * Hops one workgroup chains per round in both phases, 1 to 64. Defaults to 16. A network whose
   * longest shortest path has `h` hops needs about `ceil(h / localIterations) + 1` rounds.
   */
  localIterations?: number;
  /**
   * Optional one-row per-frame round limit for each phase, clamped to the phase's compile-time
   * bound.
   */
  activeIterations?: GraphDataView<'uint32'>;
  /**
   * Compile-time rounds of the equal-cost level phase of `output.predecessors`, 1 to 1024.
   * Defaults to 4. A zero-weight plateau deeper than `maximumTieIterations * localIterations` hops
   * leaves its deepest nodes without a predecessor and clears `output.status.converged`.
   */
  maximumTieIterations?: number;
  /** Views the contributor writes. */
  output: GPUNetworkServiceAreasOutput;
};

/** Views written by {@link GPUNetworkServiceAreas}. */
export type GPUNetworkServiceAreasOutput = {
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
  /**
   * Optional per-node shortest-path predecessor toward the assigned facility, or
   * `GPU_NETWORK_REACHABILITY_NONE` for facility nodes and unassigned nodes. It follows the
   * `GPUNetworkReachability` rule restricted to tight edges between nodes of the same assignment:
   * the smallest strictly cheaper same-label tight in-neighbor, or across equal-cost edges the
   * smallest same-label tight in-neighbor one level closer to such a node or to the facility.
   * Following predecessors from any node therefore always ends at the facility `assignments` names,
   * also on cost ties between facilities (review B-12).
   */
  predecessors?: GraphDataView<'uint32'>;
  /** Optional node count per facility row. Its length must equal `facilities.length`. */
  facilityNodeCounts?: GraphDataView<'uint32'>;
  /** Optional sum of node costs per facility row. Its length must equal `facilities.length`. */
  facilityCostSums?: GraphDataView<'float32'>;
  /** Optional GPU-written status words. */
  status?: {
    /** Optional one-row flag: 1 only when both phases reached a fixpoint, otherwise 0. */
    converged?: GraphDataView<'uint32'>;
  };
};

/**
 * Assigns every node of a directed CSR road network to its nearest facility ("Voronoi on the
 * network", the evaluation step of location-allocation), with per-facility node counts and total
 * cost served.
 *
 * Ties are broken deterministically: when several facilities achieve the minimum cost at a node,
 * the smallest facility row wins.
 *
 * Design: two frontier phases rather than one packed (cost, facility) relaxation. Phase 1 composes
 * {@link GPUNetworkReachability} with the facilities as multi-source seeds and yields exact f32
 * costs plus the GPU convergence gate. Phase 2 propagates the minimum facility label over the
 * tight-edge subgraph with the same compact frontier rounds (atomic minimum, workgroup-chained
 * hops, no gate nodes), seeded from the facility nodes: an edge `u -> v` is tight when `costs[u] + weight == costs[v]` in f32
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
export class GPUNetworkServiceAreas implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkServiceAreasProps;
  /** Resolved compile-time round bound of the cost phase. */
  readonly maximumIterations: number;
  /** Resolved compile-time round bound of the label phase. */
  readonly maximumLabelIterations: number;
  /** Resolved hops chained per round. */
  readonly localIterations: number;
  /** Resolved rounds of the predecessor level phase. */
  readonly maximumTieIterations: number;

  /**
   * @throws If a view has the wrong format or length, an option is out of range, or an output
   * shares a buffer with another output or an input.
   */
  constructor(props: GPUNetworkServiceAreasProps) {
    this.id = props.id ?? 'network-service-areas';
    this.props = props;
    this.maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    this.maximumLabelIterations =
      props.maximumLabelIterations ??
      Math.min(this.maximumIterations, DEFAULT_MAXIMUM_LABEL_ITERATIONS);
    this.localIterations = props.localIterations ?? 16;
    this.maximumTieIterations = props.maximumTieIterations ?? DEFAULT_MAXIMUM_TIE_ITERATIONS;
    const {id} = this;
    const {output} = props;
    const {nodeCount: networkNodeCount} = validateGPUNetworkTopology(id, props.topology, {
      weights: true
    });
    for (const [name, view] of [
      ['facilities', props.facilities],
      ['activeIterations', props.activeIterations],
      ['output.assignments', output.assignments],
      ['output.predecessors', output.predecessors],
      ['output.facilityNodeCounts', output.facilityNodeCounts],
      ['output.status.converged', output.status?.converged]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    if (typeof props.facilityCount === 'object') {
      validatePackedView(props.facilityCount, ['uint32'], `${id} facilityCount`);
    }
    for (const [name, view] of [
      ['facilityCosts', props.facilityCosts],
      ['costLimit', props.costLimit],
      ['output.costs', output.costs],
      ['output.facilityCostSums', output.facilityCostSums]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    const nodeCount = output.assignments.length;
    if (nodeCount < 1) {
      throw new Error(`${id} output.assignments must contain at least one node`);
    }
    if (networkNodeCount !== nodeCount) {
      throw new Error(`${id} topology must contain one node per output.assignments row`);
    }
    if (props.facilityCosts && props.facilityCosts.length !== props.facilities.length) {
      throw new Error(`${id} facilityCosts length must equal facilities length`);
    }
    if (typeof props.facilityCount === 'number') {
      validateCapacity(id, 'facilityCount', props.facilityCount, 0, props.facilities.length);
    }
    for (const [name, view] of [
      ['output.facilityNodeCounts', output.facilityNodeCounts],
      ['output.facilityCostSums', output.facilityCostSums]
    ] as const) {
      if (view && view.length !== props.facilities.length) {
        throw new Error(`${id} ${name} length must equal facilities length`);
      }
    }
    for (const [name, view] of [
      ['facilityCount', typeof props.facilityCount === 'object' ? props.facilityCount : undefined],
      ['activeIterations', props.activeIterations],
      ['costLimit', props.costLimit],
      ['output.status.converged', output.status?.converged]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    for (const [name, view] of [
      ['output.costs', output.costs],
      ['output.predecessors', output.predecessors]
    ] as const) {
      if (view && view.length !== nodeCount) {
        throw new Error(`${id} ${name} length must equal the node count`);
      }
    }
    validateCapacity(
      id,
      'maximumIterations',
      this.maximumIterations,
      1,
      GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
    );
    validateCapacity(
      id,
      'maximumLabelIterations',
      this.maximumLabelIterations,
      1,
      GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
    );
    validateCapacity(
      id,
      'localIterations',
      this.localIterations,
      1,
      GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
    );
    validateCapacity(
      id,
      'maximumTieIterations',
      this.maximumTieIterations,
      1,
      GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS
    );
    const outputs = [
      output.assignments,
      output.costs,
      output.predecessors,
      output.facilityNodeCounts,
      output.facilityCostSums,
      output.status?.converged
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const inputs = [
      ...getGPUNetworkTopologyViews(props.topology),
      props.facilities,
      props.facilityCosts,
      typeof props.facilityCount === 'object' ? props.facilityCount : undefined,
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
   * Returns the reachability nodes, then label initialize, seed, tight-edge, frontier seed and
   * `maximumLabelIterations` label round nodes, and optional finalize and per-facility aggregation nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumIterations, localIterations} = this;
    const {output} = props;
    const {offsets, neighbors, weights} = props.topology;
    const labelIterations = this.maximumLabelIterations;
    validateGraphViewsBelongToGraph(id, graph, getViews(props));
    const nodeCount = output.assignments.length;
    const costs = output.costs ?? createTransientView(graph, `${id}-costs`, 'float32', nodeCount);
    const reachabilityConverged = output.status?.converged
      ? createTransientView(graph, `${id}-reachability-converged`, 'uint32', 1)
      : undefined;
    const labels = output.assignments;
    const phase: FrontierPhase = {
      maxRounds: labelIterations,
      stampBase: 0,
      controlWordOffset: 0,
      dispatchSlotOffset: 0
    };
    const layout = getFrontierPhaseLayout(labelIterations);
    // With predecessors, a level phase over same-label equal-cost edges follows the label phase in
    // the same frontier state (stamps continue above the label phase).
    const tieLayout = getFrontierPhaseLayout(this.maximumTieIterations);
    const tiePhase: FrontierPhase | undefined = output.predecessors
      ? {
          maxRounds: this.maximumTieIterations,
          stampBase: getFrontierPhaseMaximumStamp(phase),
          controlWordOffset: layout.controlWordCount,
          dispatchSlotOffset: layout.dispatchSlotCount
        }
      : undefined;
    const state = createFrontierState(graph, `${id}-label`, {
      nodeCount,
      controlWordCount: layout.controlWordCount + (tiePhase ? tieLayout.controlWordCount : 0),
      dispatchSlotCount: layout.dispatchSlotCount + (tiePhase ? tieLayout.dispatchSlotCount : 0)
    });
    const tightBits = createTransientView(
      graph,
      `${id}-label-tight-bits`,
      'uint32',
      Math.max(1, Math.ceil(neighbors.length / 32))
    );
    const csr = {offsets, neighbors, weights};
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUNetworkReachability({
        id: `${id}-reachability`,
        topology: props.topology,
        startNodes: props.facilities,
        startCosts: props.facilityCosts,
        startCount: props.facilityCount,
        costLimit: props.costLimit,
        maximumIterations,
        // The resolved value (default 16), so the cost phase and the label phase chain the same
        // number of hops (review B-12).
        localIterations,
        activeIterations: props.activeIterations,
        output: {costs, status: {converged: reachabilityConverged}}
      }).getCommandNodes(graph),
      createServiceAreasLabelInitializeNode<Parameters>(graph, {
        id: `${id}-label-initialize`,
        nodeCount,
        maxIterations: labelIterations,
        state,
        phase,
        labels,
        tightBits,
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
          costs,
          labels
        })
      );
    }
    nodes.push(
      createServiceAreasTightEdgesNode<Parameters>(graph, {
        id: `${id}-label-tight-edges`,
        nodeCount,
        csr,
        costs,
        tightBits
      }),
      createServiceAreasLabelFrontierSeedNode<Parameters>(graph, {
        id: `${id}-label-frontier-seed`,
        nodeCount,
        state,
        phase,
        labels
      })
    );
    for (let round = 0; round < labelIterations; round++) {
      nodes.push(
        createServiceAreasLabelRoundNode<Parameters>(graph, {
          id: `${id}-label-round-${round}`,
          nodeCount,
          state,
          phase,
          round,
          localIterations,
          csr,
          tightBits,
          labels
        })
      );
    }
    if (output.predecessors && tiePhase) {
      nodes.push(
        ...this.getPredecessorNodes(graph, {
          state,
          tiePhase,
          costs,
          predecessors: output.predecessors
        })
      );
    }
    if (output.status?.converged && reachabilityConverged) {
      nodes.push(
        createServiceAreasFinalizeNode<Parameters>(graph, {
          id: `${id}-finalize`,
          state,
          phase,
          tiePhase,
          reachabilityConverged,
          converged: output.status.converged
        })
      );
    }
    // GPUGroupAggregation skips out-of-range keys (NONE) and non-finite values, so unassigned
    // rows with +Infinity costs never contribute and no mask is needed.
    if (output.facilityNodeCounts && props.facilities.length > 0) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-facility-node-counts`,
          keys: output.assignments,
          output: output.facilityNodeCounts
        }).getCommandNodes(graph)
      );
    }
    if (output.facilityCostSums && props.facilities.length > 0) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-facility-cost-sums`,
          keys: output.assignments,
          values: costs,
          output: output.facilityCostSums,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }

  /**
   * Predecessors inside each service area: the `GPUNetworkReachability` predecessor and tie-level
   * passes with every tight edge restricted to endpoints of the same label, and roots restricted
   * to facility nodes labelled with their own facility row.
   */
  private getPredecessorNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      state: ReturnType<typeof createFrontierState>;
      tiePhase: FrontierPhase;
      costs: GraphDataView<'float32'>;
      predecessors: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props, localIterations} = this;
    const {state, tiePhase, costs, predecessors} = context;
    const {offsets, neighbors, weights} = props.topology;
    const nodeCount = props.output.assignments.length;
    const edgeCount = neighbors.length;
    const labels = props.output.assignments;
    const csr = {offsets, neighbors, weights};
    const levels = createTransientView(graph, `${id}-tie-levels`, 'uint32', nodeCount);
    const tieBits = createTransientView(
      graph,
      `${id}-tie-bits`,
      'uint32',
      Math.max(1, Math.ceil(edgeCount / 32))
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-predecessors`,
        operation: 'GPUNetworkServiceAreas',
        view: predecessors,
        type: 'u32',
        value: '0xffffffffu'
      }),
      createReachabilityTieInitializeNode<Parameters>(graph, {
        id: `${id}-tie-initialize`,
        nodeCount,
        state,
        tiePhase,
        levels,
        tieBits
      }),
      createReachabilityPredecessorsNode<Parameters>(graph, {
        id: `${id}-predecessors`,
        nodeCount,
        csr,
        costs,
        predecessors,
        tieBits,
        labels
      })
    ];
    if (props.facilities.length > 0) {
      nodes.push(
        createReachabilityTieRootsNode<Parameters>(graph, {
          id: `${id}-tie-roots`,
          nodeCount,
          state,
          phase: tiePhase,
          sources: props.facilities,
          sourceCosts: props.facilityCosts,
          sourceCount: props.facilityCount,
          // A facility above the cost limit never equals its node's limited cost.
          hasCostLimit: false,
          costs,
          levels,
          labels
        })
      );
    }
    nodes.push(
      createReachabilityTieSeedNode<Parameters>(graph, {
        id: `${id}-tie-seed`,
        nodeCount,
        edgeCount,
        state,
        tiePhase,
        offsets: offsets,
        tieBits,
        predecessors,
        levels
      })
    );
    for (let round = 0; round < this.maximumTieIterations; round++) {
      nodes.push(
        createReachabilityTieLevelNode<Parameters>(graph, {
          id: `${id}-tie-level-${round}`,
          nodeCount,
          edgeCount,
          state,
          tiePhase,
          round,
          localIterations,
          offsets: offsets,
          neighbors: neighbors,
          tieBits,
          levels
        })
      );
    }
    nodes.push(
      createReachabilityTiePredecessorsNode<Parameters>(graph, {
        id: `${id}-tie-predecessors`,
        nodeCount,
        edgeCount,
        offsets: offsets,
        neighbors: neighbors,
        tieBits,
        levels,
        predecessors
      })
    );
    return nodes;
  }
}

/** Returns every view referenced by the props. */
function getViews(props: GPUNetworkServiceAreasProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    ...getGPUNetworkTopologyViews(props.topology),
    props.facilities,
    props.facilityCosts,
    typeof props.facilityCount === 'object' ? props.facilityCount : undefined,
    props.costLimit,
    props.activeIterations,
    output.assignments,
    output.costs,
    output.predecessors,
    output.facilityNodeCounts,
    output.facilityCostSums,
    output.status?.converged
  ];
}
