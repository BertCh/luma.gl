// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUHistogram,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';
import {
  createReachabilityBandsNode,
  createReachabilityFinalizeNode,
  createReachabilityInitializeNode,
  createReachabilityPredecessorsNode,
  createReachabilityReachedEdgesNode,
  createReachabilityRelaxNode,
  createReachabilitySeedNode,
  createReachabilityTieInitializeNode,
  createReachabilityTieLevelNode,
  createReachabilityTiePredecessorsNode,
  createReachabilityTieRootsNode,
  createReachabilityTieSeedNode,
  createReachabilityUnresolvedCountNode,
  NETWORK_REACHABILITY_COUNTER_WORD_COUNT
} from './network-reachability-passes';
import {
  createFrontierState,
  getFrontierPhaseLayout,
  getFrontierPhaseMaximumStamp,
  type FrontierPhase
} from './network-frontier';

/** Sentinel written for "no predecessor" and "no band" (unreached or beyond the last threshold). */
export const GPU_NETWORK_REACHABILITY_NONE = 0xffffffff;

/** Largest accepted compile-time `maximumIterations`, matching the `GPUGraphTraversal` depth cap. */
export const GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS = 1024;

/** Largest accepted compile-time `localIterations`. */
export const GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS = 64;

/** Largest accepted compile-time `maximumTieIterations`. */
export const GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS = 1024;

/**
 * Largest `output.costs` row count of {@link GPUNetworkReachability} on `device`, that is
 * `laneCount * nodeCount`: the frontier queues hold two 4-byte slots per row in one storage
 * binding, so 16777216 on a default device (128 MiB bindings).
 */
export function getGPUNetworkReachabilityMaximumNodeCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

const DEFAULT_MAXIMUM_ITERATIONS = 64;
const DEFAULT_MAXIMUM_TIE_ITERATIONS = 4;
const DEFAULT_LOCAL_ITERATIONS = 32;

/**
 * Properties for {@link GPUNetworkReachability}.
 *
 * Compile-time: node and edge counts, start capacity, cost break count, `maximumIterations`,
 * `localIterations`, a numeric `startCount`, and which optional views exist. Per-frame: the contents
 * of the topology (live traffic, closures), start nodes, start costs, a `startCount` view,
 * `activeIterations`, `costLimit`, and `costBreaks`.
 */
export type GPUNetworkReachabilityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-reachability'`. */
  id?: string;
  /**
   * Directed network. `offsets` has `nodeCount + 1` rows as in `GPUGraphTraversal.offsets`;
   * `neighbors` indices `>= nodeCount` are ignored; negative or NaN `weights` are impassable.
   */
  topology: GPUNetworkTopology;
  /** Start node indices. Capacity is the length; out-of-range indices are ignored. */
  startNodes: GraphDataView<'uint32'>;
  /** Optional initial cost per start row. Defaults to 0. */
  startCosts?: GraphDataView<'float32'>;
  /**
   * Optional active start count; rows at or after it are ignored. A number is a compile-time
   * constant in `[0, startNodes.length]`; a one-row view is read per frame. Defaults to every row.
   */
  startCount?: number | GraphDataView<'uint32'>;
  /**
   * Compile-time number of unrolled relaxation rounds (graph nodes), 1 to 1024. Defaults to 64.
   * Each round covers up to `localIterations` hops along a chain, in practice about
   * `0.85 * localIterations` (13.6 hops at 16, 28 at 32, 50 at 64 on a grid), so a network whose
   * longest shortest path has `h` hops needs about `ceil(h / (0.7 * localIterations)) + 4` rounds.
   *
   * Cost: every round is one graph node (one dispatch, roughly 20 to 30 us) whether or not its
   * queue is empty, on top of a fixed 1.5 to 2 ms submit and readback floor. Real work is small
   * below about 500k nodes, so oversizing `maximumIterations` costs time on low-diameter graphs while
   * undersizing leaves `converged` at 0. See `recommendReachabilityIterations` and
   * `adaptReachabilityIterations` for sizing from node count and from last frame's result.
   */
  maximumIterations?: number;
  /**
   * Compile-time number of hops one workgroup chains inside a single round dispatch, 1 to 64.
   * Defaults to 32. `1` is the pure round-per-hop queue algorithm. Costs are identical for every
   * value; only the number of rounds needed to converge, and the work per round, change. Longer
   * chains cut rounds (and so graph nodes) for narrow single-source wavefronts; wide frontiers,
   * such as lane-expanded batches, serialize the chain in one workgroup and prefer about 16.
   */
  localIterations?: number;
  /**
   * Compile-time number of unrolled rounds (graph nodes) of the tie-level phase that gives nodes
   * reached only across equal-cost edges a predecessor, 1 to 1024. Defaults to 4. Only used when
   * `output.predecessors` is given. Each round chains up to `localIterations` hops along a
   * zero-weight plateau, so a plateau `d` hops deep needs about `ceil(d / localIterations)` rounds.
   * When a plateau is deeper, its deepest nodes keep `GPU_NETWORK_REACHABILITY_NONE`,
   * `output.status.converged` is 0 and `output.unresolvedCount` counts them.
   */
  maximumTieIterations?: number;
  /** Optional one-row per-frame round limit, clamped to `maximumIterations`. */
  activeIterations?: GraphDataView<'uint32'>;
  /** Optional one-row per-frame cost cutoff. Candidates above it stay unreached. */
  costLimit?: GraphDataView<'float32'>;
  /**
   * Compile-time number of independent searches ("lanes") that share one CSR. With `laneCount > 1`,
   * `output.costs` holds `laneCount` consecutive blocks of `nodeCount` costs (lane `l`, node `u` is
   * `l * nodeCount + u`) and `topology.offsets` still describes the single network of `nodeCount`
   * nodes. `startNodes` and `startCosts` address expanded nodes. Every lane then reads the same
   * offsets, neighbors and weights, so no per-lane copy of the CSR is needed and lanes that visit
   * the same node share cache lines. Costs are bit-identical to searching a lane-expanded copy of
   * the CSR. Not supported together with `output.predecessors` or `output.unresolvedCount`.
   * Defaults to 1.
   */
  laneCount?: number;
  /** Optional ascending cost breaks. Required when `output.bands` or `output.bandCounts` is given. */
  costBreaks?: GraphDataView<'float32'>;
  /** Views the contributor writes. */
  output: GPUNetworkReachabilityOutput;
};

/** Views written by {@link GPUNetworkReachability}. */
export type GPUNetworkReachabilityOutput = {
  /**
   * Per-node minimum cost, `+Infinity` when unreached. Its length is `laneCount * nodeCount`; with
   * one lane it defines the node count.
   */
  costs: GraphDataView<'float32'>;
  /**
   * Optional per-node predecessor on a shortest path, or `GPU_NETWORK_REACHABILITY_NONE` for
   * starts and unreached nodes. The predecessor is the smallest strictly-cheaper tight in-neighbor
   * when one exists. A node reached only across equal-cost (zero-weight, or weight below half an
   * ulp) edges gets the smallest equal-cost tight in-neighbor one tie level closer to a node with a
   * strict predecessor or to a start, so following predecessors always ends at a start.
   */
  predecessors?: GraphDataView<'uint32'>;
  /** Optional per-node band ID, or `GPU_NETWORK_REACHABILITY_NONE` beyond the last cost break. */
  bands?: GraphDataView<'uint32'>;
  /** Optional node count per band (composes `GPUHistogram`). */
  bandCounts?: GraphDataView<'uint32'>;
  /**
   * Optional one-row count of reached nodes that have no predecessor although they are not starts
   * or strict-predecessor roots, because the tie-level phase stopped at `maximumTieIterations`
   * before reaching them. 0 means every reached node has its tie-rule predecessor. Requires
   * `predecessors`. Unlike `converged`, it names how many nodes are affected.
   */
  unresolvedCount?: GraphDataView<'uint32'>;
  /** Optional GPU-written status words. */
  status?: {
    /**
     * Optional one-row flag: 1 when relaxation reached a fixpoint and, when `predecessors` is given,
     * the tie-level phase finished; 0 when a round limit stopped either.
     */
    converged?: GraphDataView<'uint32'>;
    /**
     * Optional one-row count of relaxation rounds executed: rounds below the limit whose queue was
     * not empty, and at least one when the limit is above zero.
     */
    iterationCount?: GraphDataView<'uint32'>;
    /**
     * Optional two-row count of edge relaxations attempted by all relax rounds, as a 64-bit value
     * in (low, high) `uint32` words: every edge scanned by a visit (`lastEdge - firstEdge`),
     * summed over every visit of every round, including the workgroup-local chained hops, and over
     * every lane. Compare with `reachedEdgeCount` to see how much extra work the chaotic
     * Bellman-Ford does ({@link getGPUNetworkReachabilityInflation}). Adds one atomic per workgroup
     * to each relax round; leave it off in production.
     */
    relaxationCount?: GraphDataView<'uint32'>;
    /**
     * Optional two-row count, as (low, high) `uint32` words, of the edges a label-setting Dijkstra
     * scans exactly once: the sum of the out-degree (edges with an in-range target, any weight) of
     * every node with a finite final cost, over every lane.
     */
    reachedEdgeCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Combines the (low, high) words of a 64-bit counter into a number, `low + high * 2^32`.
 * @internal
 */
function readCounterWords(words: Uint32Array | readonly number[]): number {
  return words[0] + words[1] * 2 ** 32;
}

/**
 * Returns the work inflation of a {@link GPUNetworkReachability} run: `relaxationCount` divided by
 * `reachedEdgeCount`, each a (low, high) pair read from `output.status`. 1 is work-efficient (every
 * reachable edge scanned once, as in label-setting Dijkstra). The result is NaN when
 * `reachedEdgeCount` is 0.
 *
 * Decision rule for delta-stepping or near-far: the algorithms roadmap rejects them unless the
 * measured inflation exceeds 3 on the target networks; at or below 3 the chaotic relaxation is
 * already within a small factor of work-efficient and a bucketed scheme would add rounds and
 * bookkeeping for little saved work.
 */
export function getGPUNetworkReachabilityInflation(
  relaxationCount: Uint32Array | readonly number[],
  reachedEdgeCount: Uint32Array | readonly number[]
): number {
  const reached = readCounterWords(reachedEdgeCount);
  return reached === 0 ? Number.NaN : readCounterWords(relaxationCount) / reached;
}

/**
 * Computes single- or multi-source shortest-path costs over a directed CSR road network, with
 * isochrone bands, shortest-path predecessors, and a GPU convergence flag.
 *
 * The algorithm is frontier-based chaotic Bellman-Ford with atomic float minima. Each round
 * relaxes a compact queue of the nodes improved in the previous round (no per-node scan), and a node
 * is queued at most once per round through a round-stamped visited set that never needs clearing.
 * Rounds are unrolled to `maximumIterations` graph nodes with GPU-written indirect dispatches: every
 * push that opens a new 256-entry queue chunk grows the next round's dispatch, so an empty round, and
 * every round after it, dispatches zero workgroups without any gate node, and nothing is read back.
 * Inside a round, each workgroup processes 256 queue entries and then chains up to
 * `localIterations - 1` further hops on the nodes it improved through a workgroup-local queue, so
 * one round covers many hops of a chain. Final costs are the unique fixpoint of the f32 relaxations,
 * so they are bit-identical for every `localIterations` and `maximumIterations` large enough to converge.
 *
 * Predecessors use a second, level-based phase so zero-weight (equal-cost) plateaus stay acyclic:
 * a node with a strictly-cheaper tight in-neighbor takes the smallest such neighbor; a node reached
 * only across equal-cost tight edges takes the smallest equal-cost in-neighbor one level closer to
 * a node of the first kind or to a source, where levels are hop counts over equal-cost edges
 * computed by a second frontier phase of `maximumTieIterations` rounds. Predecessor links therefore
 * always lead back to a source.
 *
 * Edges are directed: undirected roads list both directions. "Cost to reach a destination" uses a
 * reverse CSR, which callers can build on the GPU with `GPUCOOToCSR`. Every encoding recomputes
 * from scratch.
 *
 * Band `i` holds nodes with cost in `(threshold[i - 1], threshold[i]]`; band 0 holds
 * `cost <= threshold[0]`.
 *
 * Why this is not `GPUGraphSingleSourceShortestPath`: that algorithm takes one source, a physical
 * `GPUGraphTopology` (weights from the graph, physical `GPUVector` outputs) and runs dense
 * synchronized Bellman-Ford rounds that scan every vertex. This contributor takes command-graph
 * views whose contents change every frame (topology, start nodes, start costs and count, cost
 * limit, round limit), seeds many start nodes with initial costs, and relaxes a compact frontier with
 * GPU-written indirect dispatches. Its predecessor rule (smallest strictly-cheaper tight
 * in-neighbor, plus a level phase for equal-cost plateaus) also differs from the lowest-parent
 * rule of the single-source algorithm. A virtual super-source would need a per-frame graph rebuild
 * and would change predecessors and the cost limit semantics, so the two stay separate.
 */
export class GPUNetworkReachability implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkReachabilityProps;
  /** Resolved compile-time round count. */
  readonly maximumIterations: number;
  /** Resolved compile-time hops per workgroup per round. */
  readonly localIterations: number;
  /** Resolved compile-time round count of the tie-level phase. */
  readonly maximumTieIterations: number;
  /** Resolved compile-time number of lanes sharing the CSR. */
  readonly laneCount: number;

  /**
   * @throws If a view has the wrong format or length, an option is out of range, or an output
   * shares a buffer with another output or an input.
   */
  constructor(props: GPUNetworkReachabilityProps) {
    this.id = props.id ?? 'network-reachability';
    this.props = props;
    this.maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    this.localIterations = props.localIterations ?? DEFAULT_LOCAL_ITERATIONS;
    this.maximumTieIterations = props.maximumTieIterations ?? DEFAULT_MAXIMUM_TIE_ITERATIONS;
    this.laneCount = props.laneCount ?? 1;
    const {id} = this;
    const {output} = props;
    const {nodeCount: networkNodeCount} = validateGPUNetworkTopology(id, props.topology, {
      weights: true
    });
    for (const [name, view] of [
      ['startNodes', props.startNodes],
      ['activeIterations', props.activeIterations],
      ['output.predecessors', output.predecessors],
      ['output.bands', output.bands],
      ['output.bandCounts', output.bandCounts],
      ['output.status.converged', output.status?.converged],
      ['output.unresolvedCount', output.unresolvedCount],
      ['output.status.iterationCount', output.status?.iterationCount],
      ['output.status.relaxationCount', output.status?.relaxationCount],
      ['output.status.reachedEdgeCount', output.status?.reachedEdgeCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    if (typeof props.startCount === 'object') {
      validatePackedView(props.startCount, ['uint32'], `${id} startCount`);
    }
    for (const [name, view] of [
      ['startCosts', props.startCosts],
      ['costLimit', props.costLimit],
      ['output.costs', output.costs],
      ['costBreaks', props.costBreaks]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    const nodeCount = output.costs.length;
    if (nodeCount < 1) {
      throw new Error(`${id} output.costs must contain at least one node`);
    }
    validateCapacity(id, 'laneCount', this.laneCount);
    if (this.laneCount > 1 && (output.predecessors || output.unresolvedCount)) {
      throw new Error(
        `${id} laneCount > 1 does not support output.predecessors or output.unresolvedCount`
      );
    }
    if (nodeCount % this.laneCount !== 0) {
      throw new Error(`${id} output.costs length must be a multiple of laneCount`);
    }
    if (networkNodeCount !== nodeCount / this.laneCount) {
      throw new Error(`${id} topology must contain one node per output.costs row per lane`);
    }
    if (props.startCosts && props.startCosts.length !== props.startNodes.length) {
      throw new Error(`${id} startCosts length must equal startNodes length`);
    }
    if (typeof props.startCount === 'number') {
      validateCapacity(id, 'startCount', props.startCount, 0, props.startNodes.length);
    }
    for (const [name, view] of [
      ['startCount', typeof props.startCount === 'object' ? props.startCount : undefined],
      ['activeIterations', props.activeIterations],
      ['costLimit', props.costLimit],
      ['output.status.converged', output.status?.converged],
      ['output.unresolvedCount', output.unresolvedCount],
      ['output.status.iterationCount', output.status?.iterationCount]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    for (const [name, view] of [
      ['output.status.relaxationCount', output.status?.relaxationCount],
      ['output.status.reachedEdgeCount', output.status?.reachedEdgeCount]
    ] as const) {
      if (view && view.length !== 2) {
        throw new Error(`${id} ${name} must contain exactly two rows`);
      }
    }
    for (const [name, view] of [
      ['output.predecessors', output.predecessors],
      ['output.bands', output.bands]
    ] as const) {
      if (view && view.length !== nodeCount) {
        throw new Error(`${id} ${name} length must equal the node count`);
      }
    }
    const needsCostBreaks = Boolean(output.bands || output.bandCounts);
    if (needsCostBreaks !== Boolean(props.costBreaks)) {
      throw new Error(
        `${id} costBreaks is required exactly when output.bands or output.bandCounts is given`
      );
    }
    if (props.costBreaks && props.costBreaks.length < 1) {
      throw new Error(`${id} costBreaks must contain at least one break`);
    }
    if (output.bandCounts && output.bandCounts.length !== props.costBreaks?.length) {
      throw new Error(`${id} output.bandCounts length must equal costBreaks length`);
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
    if (output.unresolvedCount && !output.predecessors) {
      throw new Error(`${id} output.unresolvedCount requires output.predecessors`);
    }
    const outputs = [
      output.costs,
      output.predecessors,
      output.bands,
      output.bandCounts,
      output.status?.converged,
      output.unresolvedCount,
      output.status?.iterationCount,
      output.status?.relaxationCount,
      output.status?.reachedEdgeCount
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const inputs = [
      ...getGPUNetworkTopologyViews(props.topology),
      props.startNodes,
      props.startCosts,
      typeof props.startCount === 'object' ? props.startCount : undefined,
      props.activeIterations,
      props.costLimit,
      props.costBreaks
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
   * Returns initialize, seed, one relax node per round, and optional output nodes. With
   * `predecessors`, the strict predecessor pass is followed by the tie-level phase: tie-bit and
   * level initialization, root marking, level seeding, `maximumTieIterations` level rounds, and a tie
   * predecessor pass (`maximumTieIterations + 4` extra nodes beyond the strict pass).
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumIterations, localIterations} = this;
    const {output} = props;
    const {offsets, neighbors, weights} = props.topology;
    validateGraphViewsBelongToGraph(id, graph, getViews(props));
    const nodeCount = output.costs.length;
    validateMaximumCount(
      id,
      'output.costs rows',
      nodeCount,
      getGPUNetworkReachabilityMaximumNodeCount(graph.device)
    );
    const csr = {
      offsets,
      neighbors,
      weights
    };
    const phase: FrontierPhase = {
      maxRounds: maximumIterations,
      stampBase: 0,
      controlWordOffset: 0,
      dispatchSlotOffset: 0
    };
    const layout = getFrontierPhaseLayout(maximumIterations);
    const tieLayout = getFrontierPhaseLayout(this.maximumTieIterations);
    const tiePhase: FrontierPhase = {
      maxRounds: this.maximumTieIterations,
      stampBase: getFrontierPhaseMaximumStamp(phase),
      controlWordOffset: layout.controlWordCount,
      dispatchSlotOffset: layout.dispatchSlotCount
    };
    const hasTiePhase = Boolean(output.predecessors);
    const countsRelaxations = Boolean(output.status?.relaxationCount);
    const countsReachedEdges = Boolean(output.status?.reachedEdgeCount);
    const hasCounters = countsRelaxations || countsReachedEdges;
    // The counter words follow every phase region of the control buffer.
    const phaseControlWordCount =
      layout.controlWordCount + (hasTiePhase ? tieLayout.controlWordCount : 0);
    const counterWordOffset = hasCounters ? phaseControlWordCount : undefined;
    const state = createFrontierState(graph, id, {
      nodeCount,
      controlWordCount:
        phaseControlWordCount + (hasCounters ? NETWORK_REACHABILITY_COUNTER_WORD_COUNT : 0),
      dispatchSlotCount: layout.dispatchSlotCount + (hasTiePhase ? tieLayout.dispatchSlotCount : 0)
    });
    const edgeCount = neighbors.length;
    const levels = hasTiePhase
      ? createTransientView(graph, `${id}-tie-levels`, 'uint32', nodeCount)
      : undefined;
    const tieBits = hasTiePhase
      ? createTransientView(
          graph,
          `${id}-tie-bits`,
          'uint32',
          Math.max(1, Math.ceil(edgeCount / 32))
        )
      : undefined;
    const hasCostLimit = Boolean(props.costLimit);
    const nodes: GPUCommandNode<Parameters>[] = [
      createReachabilityInitializeNode<Parameters>(graph, {
        id: `${id}-initialize`,
        nodeCount,
        state,
        phase,
        costs: output.costs,
        predecessors: output.predecessors,
        activeIterations: props.activeIterations,
        costLimit: props.costLimit,
        counterWordOffset
      })
    ];
    if (props.startNodes.length > 0) {
      nodes.push(
        createReachabilitySeedNode<Parameters>(graph, {
          id: `${id}-seed`,
          nodeCount,
          state,
          phase,
          sources: props.startNodes,
          sourceCosts: props.startCosts,
          sourceCount: props.startCount,
          hasCostLimit,
          costs: output.costs
        })
      );
    }
    for (let round = 0; round < maximumIterations; round++) {
      nodes.push(
        createReachabilityRelaxNode<Parameters>(graph, {
          id: `${id}-relax-${round}`,
          record: {
            loop: {
              id: `${id}-relax`,
              iteration: round,
              phase: 'gated',
              maximumIterations: maximumIterations
            },
            datum: {keySpace: 'node', relation: 'frontier-queue'},
            formula: 'gpu-network.reachability.relax',
            attrs: {chaotic: true}
          },
          nodeCount,
          state,
          phase,
          round,
          localIterations,
          csr,
          hasCostLimit,
          costs: output.costs,
          laneNodeCount: this.laneCount > 1 ? nodeCount / this.laneCount : undefined,
          counterWordOffset: countsRelaxations ? counterWordOffset : undefined
        })
      );
    }
    if (countsReachedEdges && counterWordOffset !== undefined) {
      nodes.push(
        createReachabilityReachedEdgesNode<Parameters>(graph, {
          id: `${id}-reached-edges`,
          nodeCount,
          state,
          csr,
          costs: output.costs,
          laneNodeCount: this.laneCount > 1 ? nodeCount / this.laneCount : undefined,
          counterWordOffset
        })
      );
    }
    if (output.predecessors && levels && tieBits) {
      nodes.push(
        createReachabilityTieInitializeNode<Parameters>(graph, {
          id: `${id}-tie-initialize`,
          nodeCount,
          state,
          tiePhase,
          levels,
          tieBits,
          unresolvedCount: output.unresolvedCount
        }),
        createReachabilityPredecessorsNode<Parameters>(graph, {
          id: `${id}-predecessors`,
          nodeCount,
          csr,
          costs: output.costs,
          predecessors: output.predecessors,
          tieBits
        })
      );
      if (props.startNodes.length > 0) {
        nodes.push(
          createReachabilityTieRootsNode<Parameters>(graph, {
            id: `${id}-tie-roots`,
            nodeCount,
            state,
            phase,
            sources: props.startNodes,
            sourceCosts: props.startCosts,
            sourceCount: props.startCount,
            hasCostLimit,
            costs: output.costs,
            levels
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
          predecessors: output.predecessors,
          levels
        })
      );
      for (let round = 0; round < this.maximumTieIterations; round++) {
        nodes.push(
          createReachabilityTieLevelNode<Parameters>(graph, {
            id: `${id}-tie-level-${round}`,
            record: {
              loop: {
                id: `${id}-tie-level`,
                iteration: round,
                phase: 'gated',
                maximumIterations: this.maximumTieIterations
              },
              datum: {keySpace: 'node', relation: 'frontier-queue'},
              attrs: {chaotic: true}
            },
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
          predecessors: output.predecessors
        })
      );
    }
    if (output.predecessors && levels && output.unresolvedCount) {
      nodes.push(
        createReachabilityUnresolvedCountNode<Parameters>(graph, {
          id: `${id}-unresolved-count`,
          nodeCount,
          costs: output.costs,
          levels,
          predecessors: output.predecessors,
          unresolvedCount: output.unresolvedCount
        })
      );
    }
    if (
      output.status?.converged ||
      output.status?.iterationCount ||
      output.status?.relaxationCount ||
      output.status?.reachedEdgeCount
    ) {
      nodes.push(
        createReachabilityFinalizeNode<Parameters>(graph, {
          id: `${id}-finalize`,
          state,
          phase,
          tiePhase: hasTiePhase ? tiePhase : undefined,
          converged: output.status?.converged,
          iterationCount: output.status?.iterationCount,
          counters:
            counterWordOffset === undefined
              ? undefined
              : {
                  wordOffset: counterWordOffset,
                  relaxationCount: output.status?.relaxationCount,
                  reachedEdgeCount: output.status?.reachedEdgeCount
                }
        })
      );
    }
    if (props.costBreaks && (output.bands || output.bandCounts)) {
      const bands =
        output.bands ?? createTransientView(graph, `${id}-bands-scratch`, 'uint32', nodeCount);
      nodes.push(
        createReachabilityBandsNode<Parameters>(graph, {
          id: `${id}-bands`,
          nodeCount,
          costs: output.costs,
          thresholds: props.costBreaks,
          bands
        })
      );
      if (output.bandCounts) {
        const bandCount = props.costBreaks.length;
        nodes.push(
          ...new GPUHistogram({
            id: `${id}-band-counts`,
            input: bands,
            output: output.bandCounts,
            domain: [0, bandCount]
          }).getCommandNodes(graph)
        );
      }
    }
    return nodes;
  }
}

/** Returns every view referenced by the props. */
function getViews(props: GPUNetworkReachabilityProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    ...getGPUNetworkTopologyViews(props.topology),
    props.startNodes,
    props.startCosts,
    typeof props.startCount === 'object' ? props.startCount : undefined,
    props.activeIterations,
    props.costLimit,
    props.costBreaks,
    output.costs,
    output.predecessors,
    output.bands,
    output.bandCounts,
    output.status?.converged,
    output.unresolvedCount,
    output.status?.iterationCount,
    output.status?.relaxationCount,
    output.status?.reachedEdgeCount
  ];
}
