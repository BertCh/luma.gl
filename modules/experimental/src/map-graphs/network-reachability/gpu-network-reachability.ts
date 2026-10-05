// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUHistogram,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  createReachabilityBandsNode,
  createReachabilityFinalizeNode,
  createReachabilityInitializeNode,
  createReachabilityPredecessorsNode,
  createReachabilityRelaxNode,
  createReachabilitySeedNode,
  createReachabilityTieInitializeNode,
  createReachabilityTieLevelNode,
  createReachabilityTiePredecessorsNode,
  createReachabilityTieRootsNode,
  createReachabilityTieSeedNode
} from './network-reachability-passes';
import {
  createFrontierState,
  getFrontierPhaseLayout,
  getFrontierPhaseMaximumStamp,
  type FrontierPhase
} from './network-frontier';

/** Sentinel written for "no predecessor" and "no band" (unreached or beyond the last threshold). */
export const GPU_NETWORK_REACHABILITY_NONE = 0xffffffff;

/** Largest accepted compile-time `maxIterations`, matching the `GPUGraphTraversal` depth cap. */
export const GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS = 1024;

/** Largest accepted compile-time `localIterations`. */
export const GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS = 64;

/** Largest accepted compile-time `maxTieIterations`. */
export const GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS = 1024;

const DEFAULT_MAXIMUM_ITERATIONS = 64;
const DEFAULT_MAXIMUM_TIE_ITERATIONS = 4;
const DEFAULT_LOCAL_ITERATIONS = 16;

/**
 * Properties for {@link GPUNetworkReachability}.
 *
 * Compile-time: node and edge counts, source capacity, threshold count, `maxIterations`,
 * `localIterations`, and which optional views exist. Per-frame: the contents of the CSR (live traffic, closures), sources,
 * source costs and count, `activeIterations`, `costLimit`, and `bandThresholds`.
 */
export type GPUNetworkReachabilityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-reachability'`. */
  id?: string;
  /** Forward CSR row offsets with `nodeCount + 1` rows, as `GPUGraphTraversal.offsets`. */
  offsets: GraphDataView<'uint32'>;
  /** CSR destination node per edge. Indices `>= nodeCount` are ignored. */
  neighbors: GraphDataView<'uint32'>;
  /** Non-negative travel cost per edge. Negative or NaN edges are impassable. */
  weights: GraphDataView<'float32'>;
  /** Source node indices. Capacity is the length; out-of-range indices are ignored. */
  sources: GraphDataView<'uint32'>;
  /** Optional initial cost per source row. Defaults to 0. */
  sourceCosts?: GraphDataView<'float32'>;
  /** Optional one-row active source count; rows at or after it are ignored. */
  sourceCount?: GraphDataView<'uint32'>;
  /**
   * Compile-time number of unrolled relaxation rounds (graph nodes), 1 to 1024. Defaults to 64.
   * Each round covers up to `localIterations` hops along a chain, so a network whose longest
   * shortest path has `h` hops needs about `ceil(h / localIterations) + 1` rounds.
   */
  maxIterations?: number;
  /**
   * Compile-time number of hops one workgroup chains inside a single round dispatch, 1 to 64.
   * Defaults to 16. `1` is the pure round-per-hop queue algorithm. Costs are identical for every
   * value; only the number of rounds needed to converge, and the work per round, change.
   */
  localIterations?: number;
  /**
   * Compile-time number of unrolled rounds (graph nodes) of the tie-level phase that gives nodes
   * reached only across equal-cost edges a predecessor, 1 to 1024. Defaults to 4. Only used when
   * `predecessors` is given. Each round chains up to `localIterations` hops along a zero-weight
   * plateau, so a plateau `d` hops deep needs about `ceil(d / localIterations)` rounds. When a
   * plateau is deeper, its deepest nodes keep `GPU_NETWORK_REACHABILITY_NONE` and `converged` is 0.
   */
  maxTieIterations?: number;
  /** Optional one-row per-frame round limit, clamped to `maxIterations`. */
  activeIterations?: GraphDataView<'uint32'>;
  /** Optional one-row per-frame cost cutoff. Candidates above it stay unreached. */
  costLimit?: GraphDataView<'float32'>;
  /** Per-node minimum cost, `+Infinity` when unreached. Its length defines the node count. */
  costs: GraphDataView<'float32'>;
  /**
   * Optional per-node predecessor on a shortest path, or `GPU_NETWORK_REACHABILITY_NONE` for
   * sources and unreached nodes. The predecessor is the smallest strictly-cheaper tight in-neighbor
   * when one exists. A node reached only across equal-cost (zero-weight, or weight below half an
   * ulp) edges gets the smallest equal-cost tight in-neighbor one tie level closer to a node with a
   * strict predecessor or to a source, so following predecessors always ends at a source.
   */
  predecessors?: GraphDataView<'uint32'>;
  /** Isochrone thresholds, ascending. Required when `bands` or `bandCounts` is given. */
  bandThresholds?: GraphDataView<'float32'>;
  /** Optional per-node band ID, or `GPU_NETWORK_REACHABILITY_NONE` beyond the last threshold. */
  bands?: GraphDataView<'uint32'>;
  /** Optional node count per band (composes `GPUHistogram`). */
  bandCounts?: GraphDataView<'uint32'>;
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
};

/**
 * Computes single- or multi-source shortest-path costs over a directed CSR road network, with
 * isochrone bands, shortest-path predecessors, and a GPU convergence flag.
 *
 * The algorithm is frontier-based chaotic Bellman-Ford with atomic float minima. Each round
 * relaxes a compact queue of the nodes improved in the previous round (no per-node scan), and a node
 * is queued at most once per round through a round-stamped visited set that never needs clearing.
 * Rounds are unrolled to `maxIterations` graph nodes with GPU-written indirect dispatches: every
 * push that opens a new 256-entry queue chunk grows the next round's dispatch, so an empty round, and
 * every round after it, dispatches zero workgroups without any gate node, and nothing is read back.
 * Inside a round, each workgroup processes 256 queue entries and then chains up to
 * `localIterations - 1` further hops on the nodes it improved through a workgroup-local queue, so
 * one round covers many hops of a chain. Final costs are the unique fixpoint of the f32 relaxations,
 * so they are bit-identical for every `localIterations` and `maxIterations` large enough to converge.
 *
 * Predecessors use a second, level-based phase so zero-weight (equal-cost) plateaus stay acyclic:
 * a node with a strictly-cheaper tight in-neighbor takes the smallest such neighbor; a node reached
 * only across equal-cost tight edges takes the smallest equal-cost in-neighbor one level closer to
 * a node of the first kind or to a source, where levels are hop counts over equal-cost edges
 * computed by a second frontier phase of `maxTieIterations` rounds. Predecessor links therefore
 * always lead back to a source.
 *
 * Edges are directed: undirected roads list both directions. "Cost to reach a destination" uses a
 * reverse CSR, which callers can build on the GPU with `GPUCOOToCSR`. Every encoding recomputes
 * from scratch.
 *
 * Band `i` holds nodes with cost in `(threshold[i - 1], threshold[i]]`; band 0 holds
 * `cost <= threshold[0]`.
 */
export class GPUNetworkReachability implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'network-reachability';
  /** Validated properties. */
  readonly props: GPUNetworkReachabilityProps;
  /** Resolved compile-time round count. */
  readonly maxIterations: number;
  /** Resolved compile-time hops per workgroup per round. */
  readonly localIterations: number;
  /** Resolved compile-time round count of the tie-level phase. */
  readonly maxTieIterations: number;

  constructor(props: GPUNetworkReachabilityProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    this.maxIterations = props.maxIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    this.localIterations = props.localIterations ?? DEFAULT_LOCAL_ITERATIONS;
    this.maxTieIterations = props.maxTieIterations ?? DEFAULT_MAXIMUM_TIE_ITERATIONS;
    const {id} = this;
    for (const [name, view] of [
      ['offsets', props.offsets],
      ['neighbors', props.neighbors],
      ['sources', props.sources],
      ['sourceCount', props.sourceCount],
      ['activeIterations', props.activeIterations],
      ['predecessors', props.predecessors],
      ['bands', props.bands],
      ['bandCounts', props.bandCounts],
      ['converged', props.converged],
      ['iterationCount', props.iterationCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['weights', props.weights],
      ['sourceCosts', props.sourceCosts],
      ['costLimit', props.costLimit],
      ['costs', props.costs],
      ['bandThresholds', props.bandThresholds]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    const nodeCount = props.costs.length;
    if (nodeCount < 1) {
      throw new Error(`${id} costs must contain at least one node`);
    }
    if (props.offsets.length !== nodeCount + 1) {
      throw new Error(`${id} offsets must contain one more row than costs`);
    }
    if (props.weights.length !== props.neighbors.length) {
      throw new Error(`${id} weights length must equal neighbors length`);
    }
    if (props.sourceCosts && props.sourceCosts.length !== props.sources.length) {
      throw new Error(`${id} sourceCosts length must equal sources length`);
    }
    for (const [name, view] of [
      ['sourceCount', props.sourceCount],
      ['activeIterations', props.activeIterations],
      ['costLimit', props.costLimit],
      ['converged', props.converged],
      ['iterationCount', props.iterationCount]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    for (const [name, view] of [
      ['predecessors', props.predecessors],
      ['bands', props.bands]
    ] as const) {
      if (view && view.length !== nodeCount) {
        throw new Error(`${id} ${name} length must equal the node count`);
      }
    }
    const needsThresholds = Boolean(props.bands || props.bandCounts);
    if (needsThresholds !== Boolean(props.bandThresholds)) {
      throw new Error(`${id} bandThresholds is required exactly when bands or bandCounts is given`);
    }
    if (props.bandThresholds && props.bandThresholds.length < 1) {
      throw new Error(`${id} bandThresholds must contain at least one threshold`);
    }
    if (props.bandCounts && props.bandCounts.length !== props.bandThresholds?.length) {
      throw new Error(`${id} bandCounts length must equal bandThresholds length`);
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
    if (
      !Number.isSafeInteger(this.localIterations) ||
      this.localIterations < 1 ||
      this.localIterations > GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
    ) {
      throw new Error(
        `${id} localIterations must be an integer in [1, ${GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS}]`
      );
    }
    if (
      !Number.isSafeInteger(this.maxTieIterations) ||
      this.maxTieIterations < 1 ||
      this.maxTieIterations > GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS
    ) {
      throw new Error(
        `${id} maxTieIterations must be an integer in [1, ${GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS}]`
      );
    }
    const outputs = [
      props.costs,
      props.predecessors,
      props.bands,
      props.bandCounts,
      props.converged,
      props.iterationCount
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const inputs = [
      props.offsets,
      props.neighbors,
      props.weights,
      props.sources,
      props.sourceCosts,
      props.sourceCount,
      props.activeIterations,
      props.costLimit,
      props.bandThresholds
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
   * level initialization, root marking, level seeding, `maxTieIterations` level rounds, and a tie
   * predecessor pass (`maxTieIterations + 4` extra nodes beyond the strict pass).
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maxIterations, localIterations} = this;
    validateGraphViewsBelongToGraph(id, graph, getViews(props));
    const nodeCount = props.costs.length;
    const csr = {
      offsets: props.offsets,
      neighbors: props.neighbors,
      weights: props.weights
    };
    const phase: FrontierPhase = {
      maxRounds: maxIterations,
      stampBase: 0,
      controlWordOffset: 0,
      dispatchSlotOffset: 0
    };
    const layout = getFrontierPhaseLayout(maxIterations);
    const tieLayout = getFrontierPhaseLayout(this.maxTieIterations);
    const tiePhase: FrontierPhase = {
      maxRounds: this.maxTieIterations,
      stampBase: getFrontierPhaseMaximumStamp(phase),
      controlWordOffset: layout.controlWordCount,
      dispatchSlotOffset: layout.dispatchSlotCount
    };
    const hasTiePhase = Boolean(props.predecessors);
    const state = createFrontierState(graph, id, {
      nodeCount,
      controlWordCount: layout.controlWordCount + (hasTiePhase ? tieLayout.controlWordCount : 0),
      dispatchSlotCount: layout.dispatchSlotCount + (hasTiePhase ? tieLayout.dispatchSlotCount : 0)
    });
    const edgeCount = props.neighbors.length;
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
        costs: props.costs,
        predecessors: props.predecessors,
        activeIterations: props.activeIterations,
        costLimit: props.costLimit
      })
    ];
    if (props.sources.length > 0) {
      nodes.push(
        createReachabilitySeedNode<Parameters>(graph, {
          id: `${id}-seed`,
          nodeCount,
          state,
          phase,
          sources: props.sources,
          sourceCosts: props.sourceCosts,
          sourceCount: props.sourceCount,
          hasCostLimit,
          costs: props.costs
        })
      );
    }
    for (let round = 0; round < maxIterations; round++) {
      nodes.push(
        createReachabilityRelaxNode<Parameters>(graph, {
          id: `${id}-relax-${round}`,
          nodeCount,
          state,
          phase,
          round,
          localIterations,
          csr,
          hasCostLimit,
          costs: props.costs
        })
      );
    }
    if (props.predecessors && levels && tieBits) {
      nodes.push(
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
          costs: props.costs,
          predecessors: props.predecessors,
          tieBits
        })
      );
      if (props.sources.length > 0) {
        nodes.push(
          createReachabilityTieRootsNode<Parameters>(graph, {
            id: `${id}-tie-roots`,
            nodeCount,
            state,
            phase,
            sources: props.sources,
            sourceCosts: props.sourceCosts,
            sourceCount: props.sourceCount,
            hasCostLimit,
            costs: props.costs,
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
          offsets: props.offsets,
          tieBits,
          predecessors: props.predecessors,
          levels
        })
      );
      for (let round = 0; round < this.maxTieIterations; round++) {
        nodes.push(
          createReachabilityTieLevelNode<Parameters>(graph, {
            id: `${id}-tie-level-${round}`,
            nodeCount,
            edgeCount,
            state,
            tiePhase,
            round,
            localIterations,
            offsets: props.offsets,
            neighbors: props.neighbors,
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
          offsets: props.offsets,
          neighbors: props.neighbors,
          tieBits,
          levels,
          predecessors: props.predecessors
        })
      );
    }
    if (props.converged || props.iterationCount) {
      nodes.push(
        createReachabilityFinalizeNode<Parameters>(graph, {
          id: `${id}-finalize`,
          state,
          phase,
          tiePhase: hasTiePhase ? tiePhase : undefined,
          converged: props.converged,
          iterationCount: props.iterationCount
        })
      );
    }
    if (props.bandThresholds && (props.bands || props.bandCounts)) {
      const bands =
        props.bands ?? createTransientView(graph, `${id}-bands-scratch`, 'uint32', nodeCount);
      nodes.push(
        createReachabilityBandsNode<Parameters>(graph, {
          id: `${id}-bands`,
          nodeCount,
          costs: props.costs,
          thresholds: props.bandThresholds,
          bands
        })
      );
      if (props.bandCounts) {
        const bandCount = props.bandThresholds.length;
        nodes.push(
          ...new GPUHistogram({
            id: `${id}-band-counts`,
            input: bands,
            output: props.bandCounts,
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
  return [
    props.offsets,
    props.neighbors,
    props.weights,
    props.sources,
    props.sourceCosts,
    props.sourceCount,
    props.activeIterations,
    props.costLimit,
    props.costs,
    props.predecessors,
    props.bandThresholds,
    props.bands,
    props.bandCounts,
    props.converged,
    props.iterationCount
  ];
}
