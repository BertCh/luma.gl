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
import {
  getCoreBindingElementLimit,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';
import {
  GPUNetworkReachability,
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
} from '../network-reachability/index';
import {recommendLaneCount} from '../network-accessibility/network-accessibility-lanes';
import {
  createFrontierState,
  getFrontierPhaseLayout,
  getFrontierPhaseMaximumStamp,
  type FrontierPhase
} from '../network-reachability/network-frontier';
import {
  CENTRALITY_ALIGNMENT_ELEMENTS,
  createCentralityAccumulateNode,
  createCentralityBatchInfoNode,
  createCentralityBatchStatusNode,
  createCentralityClosenessNode,
  createCentralityConvergedNode,
  createCentralityDeltaInitializeNode,
  createCentralityDeltaRoundNode,
  createCentralityExpandSeedsNode,
  createCentralityPackNode,
  createCentralityResetNode,
  createCentralitySigmaInitializeNode,
  createCentralitySigmaRoundNode,
  createCentralityZeroNode,
  getCentralityAdjacencyLayout
} from './network-centrality-passes';

const DEFAULT_MAXIMUM_ITERATIONS = 64;
const DEFAULT_LOCAL_ITERATIONS = 16;
/**
 * Hops per workgroup in the sigma and delta phases. A node becomes ready when the last of its
 * tight predecessors decrements its counter, but the other predecessors' sums may have been
 * written by other workgroups of the same dispatch, and WGSL gives no cross-workgroup ordering
 * inside one dispatch. With one hop per round a ready node is visited in the next dispatch, after
 * every predecessor's write.
 */
const DEPENDENCY_LOCAL_ITERATIONS = 1;

/** Views written by {@link GPUNetworkCentrality}. At least one of the four values is required. */
export type GPUNetworkCentralityOutput = {
  /**
   * Optional per-node betweenness: the sum over the pivots of the Brandes dependency of the pivot
   * on the node (the pivot itself excluded, endpoints not counted), times `betweennessScale`. Its
   * length is the node count. Requesting it adds the sigma and delta phases to every batch.
   */
  betweenness?: GraphDataView<'float32'>;
  /**
   * Optional per-pivot closeness `(r - 1) / S`, where `r` counts the nodes reached from the pivot
   * (the pivot included) and `S` is the sum of their costs; 0 when `S` is 0. Its length is the
   * pivot count; out-of-range pivots get 0.
   */
  closeness?: GraphDataView<'float32'>;
  /**
   * Optional per-pivot harmonic closeness: the sum of `1 / cost` over the reached nodes with a
   * cost above 0. Its length is the pivot count.
   */
  harmonicCloseness?: GraphDataView<'float32'>;
  /** Optional per-pivot reached node count (the pivot included). Its length is the pivot count. */
  reachedCounts?: GraphDataView<'uint32'>;
  /** Optional GPU-written status words. */
  status?: {
    /**
     * Optional one-row flag: 1 when every batch's reachability and, with `betweenness`, both
     * accumulation phases finished within `maximumIterations`; 0 when a round limit stopped one.
     */
    converged?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkCentrality}.
 *
 * Compile-time: node and edge counts, pivot capacity, `laneCount`, `maximumIterations`,
 * `localIterations`, `betweennessScale`, and which outputs exist. Per-frame: the contents of both
 * topologies, the pivot nodes and `costLimit`.
 */
export type GPUNetworkCentralityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-centrality'`. */
  id?: string;
  /**
   * Directed network with weights; distances are measured outward from the pivots along its edges.
   * An undirected network lists both directions of every edge. Targets `>= nodeCount` are ignored
   * and negative or NaN weights are impassable.
   */
  topology: GPUNetworkTopology;
  /**
   * The transpose of `topology`: the same edges reversed, with bitwise-identical weights (for
   * example from `GPUCOOToCSR` on the reversed edge list). An undirected network that lists both
   * directions of every edge with equal weights may pass the same topology. The sigma and delta
   * phases walk both directions to find each node's tight predecessors and successors, so a
   * reverse topology that is not the exact transpose gives wrong centrality values.
   */
  reverseTopology: GPUNetworkTopology;
  /**
   * Pivot (source) nodes, one centrality row each. Use every node for exact centrality, or a
   * sample for estimates (set `betweennessScale`). Out-of-range nodes are ignored and their
   * per-pivot output rows are 0.
   */
  pivotNodes: GraphDataView<'uint32'>;
  /**
   * Optional one-row per-frame radius: paths longer than it are not counted, and nodes whose cost
   * from a pivot exceeds it are not part of that pivot's shortest-path graph. Inclusive.
   */
  costLimit?: GraphDataView<'float32'>;
  /**
   * Compile-time number of pivots searched together in one lane-expanded batch, 1 to the pivot
   * count. Defaults to `recommendLaneCount()` (about 1M expanded nodes per batch), at most the pivot
   * count, and low enough that the three scratch arrays of `laneCount * nodeCount` words fit one
   * storage binding. Every batch runs at the full lane count (a short last batch masks its unused
   * lanes), so all batches compile to the same pipelines. Closeness outputs are bit-identical for
   * every lane count; betweenness changes only in the last bits, because the lane count sets the
   * grouping of the sums.
   */
  laneCount?: number;
  /**
   * Compile-time number of unrolled rounds of the reachability search and of each of the sigma and
   * delta phases of every batch, 1 to 1024. Defaults to 64. The search needs about
   * `ceil(h / (0.7 * localIterations)) + 4` rounds for `h` hops along the longest shortest path;
   * the sigma and delta phases run one hop per round, so each needs `h + 1` rounds for the longest
   * tight chain (`h` is small for radius-bounded centrality). A shortfall clears `status.converged`.
   */
  maximumIterations?: number;
  /**
   * Compile-time number of hops one workgroup chains inside a round, 1 to 64. Defaults to 16, which
   * suits lane-expanded batches. Applies to the reachability search only: the sigma and delta
   * phases always run one hop per round, so every hand-off between workgroups crosses a dispatch
   * boundary.
   */
  localIterations?: number;
  /**
   * Compile-time factor applied to the sum of each batch before it is added to `betweenness`.
   * Defaults to 1. For estimates from sampled pivots use `nodeCount / pivotCount`.
   */
  betweennessScale?: number;
  /** The GPU-written views. */
  output: GPUNetworkCentralityOutput;
};

/**
 * Computes radius-bounded betweenness and closeness centrality of a directed weighted CSR network
 * from a set of pivot nodes, exactly or from a sample.
 *
 * Pivots are processed `laneCount` at a time. Each batch runs one lane-expanded
 * `GPUNetworkReachability` (lane `l`, node `u` is expanded node `l * nodeCount + u`; all lanes read
 * the one CSR), whose costs give closeness directly: one workgroup per lane reduces the reached
 * count, the cost sum and the harmonic sum in a fixed order. For betweenness the batch then runs
 * Brandes' (2001) accumulation on the tight edges, those with `cost[u] < cost[w]` and
 * `cost[u] + weight == cost[w]` in f32 exactly as the relaxation computes it:
 *
 * 1. The sigma phase (shortest-path counts) is a push-ready, pull-sum frontier schedule. A node
 *    becomes ready when its last tight predecessor releases it (`atomicSub` on a per-node counter
 *    that starts at its tight in-degree); its visit then sums its predecessors' sigma in reverse-CSR
 *    order and releases its tight successors.
 * 2. The delta phase mirrors it from the nodes without tight out-edges: a node sums
 *    `sigma(v) / sigma(w) * (1 + delta(w))` over its tight successors in forward-CSR order, then
 *    releases its tight predecessors.
 * 3. The batch adds, per node, the lane-ordered sum of the lanes' delta (skipping each lane's own
 *    pivot) times `betweennessScale` to the output. Betweenness is zeroed once, and batches add in
 *    batch order.
 *
 * Every sum has a fixed order, so results are deterministic for a given `laneCount`. Pivots may repeat; each
 * counts once per occurrence.
 *
 * Conventions (networkx parity; the oracle is `test/oracles/network-centrality.py`):
 * - Betweenness equals `networkx.betweenness_centrality(G, weight='weight', normalized=False)` on a
 *   `DiGraph` when all nodes are pivots, and `betweenness_centrality_subset(G, sources=pivots,
 *   targets=list(G), ...)` for a sample (times `betweennessScale`). Endpoints are not counted. No
 *   1/2 factor is applied; halve the result for an undirected graph that lists both directions.
 * - Closeness is outward: the cost from the pivot to the nodes it reaches. networkx's
 *   `closeness_centrality` on a `DiGraph` uses incoming distances, so compare against
 *   `closeness_centrality(G.reverse(), u=pivot, wf_improved=False)` (and `harmonic_centrality(G.reverse(), ...)`),
 *   or pass the reverse topologies (swap `topology` and `reverseTopology`) to get networkx's incoming
 *   convention. The Wasserman-Faust factor `(r - 1) / (nodeCount - 1)` is not applied.
 * - Zero-weight edges (and weights below half an ulp of the cost) are never tight, so they are
 *   excluded from path counts: a node reached only across them is treated as a root with one path.
 *   Closeness counts such nodes as reached at the cost of their predecessor; harmonic closeness skips
 *   nodes of cost 0. Parallel edges count as separate paths.
 * - Path counts are f32: exact below 2^24 paths per pair, rounded beyond, with the same relative
 *   error as the delta sums.
 *
 * Accuracy: sigma sums are exact below 2^24. Delta rounds once per division (within 2.5 ULP in
 * WGSL), addition and multiplication of a term and once per added term, with only positive terms, so a
 * node's relative error is at most `h (7 + K) 2^-24` for `h` hops along the longest tight chain and
 * a largest tight out-degree `K`; adding `P` pivots adds `(P + 2) 2^-24`.
 *
 * Scratch memory is the frontier state plus `3 * laneCount * nodeCount` words and a packed copy of
 * both CSRs, copied once per encoding.
 */
export class GPUNetworkCentrality implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkCentralityProps;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of pivot rows. */
  readonly pivotCount: number;
  /** Resolved pivots per batch. */
  readonly laneCount: number;
  /** Number of lane-expanded batches. */
  readonly batchCount: number;
  /** Resolved compile-time round count of the reachability search and of each phase. */
  readonly maximumIterations: number;
  /** Resolved compile-time hops per workgroup per round. */
  readonly localIterations: number;
  /** Resolved compile-time betweenness factor. */
  readonly betweennessScale: number;

  /**
   * @throws If a view has the wrong format or length, no output is requested, an option is out of
   * range, or an output shares a buffer with another output or an input.
   */
  constructor(props: GPUNetworkCentralityProps) {
    this.id = props.id ?? 'network-centrality';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const forward = validateGPUNetworkTopology(id, props.topology, {weights: true});
    const reverse = validateGPUNetworkTopology(id, props.reverseTopology, {weights: true});
    if (forward.nodeCount !== reverse.nodeCount) {
      throw new Error(`${id} reverseTopology must have the node count of topology`);
    }
    this.nodeCount = forward.nodeCount;
    validatePackedView(props.pivotNodes, ['uint32'], `${id} pivotNodes`);
    this.pivotCount = props.pivotNodes.length;
    if (this.pivotCount < 1) {
      throw new Error(`${id} pivotNodes must contain at least one pivot`);
    }
    if (props.costLimit) {
      validatePackedView(props.costLimit, ['float32'], `${id} costLimit`);
      if (props.costLimit.length !== 1) {
        throw new Error(`${id} costLimit must contain exactly one row`);
      }
    }
    if (
      !output.betweenness &&
      !output.closeness &&
      !output.harmonicCloseness &&
      !output.reachedCounts
    ) {
      throw new Error(
        `${id} output must contain betweenness, closeness, harmonicCloseness or reachedCounts`
      );
    }
    for (const [name, view, length] of [
      ['output.betweenness', output.betweenness, this.nodeCount],
      ['output.closeness', output.closeness, this.pivotCount],
      ['output.harmonicCloseness', output.harmonicCloseness, this.pivotCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== length) {
          throw new Error(
            `${id} ${name} must contain one row per ${name === 'output.betweenness' ? 'node' : 'pivot'}`
          );
        }
      }
    }
    if (output.reachedCounts) {
      validatePackedView(output.reachedCounts, ['uint32'], `${id} output.reachedCounts`);
      if (output.reachedCounts.length !== this.pivotCount) {
        throw new Error(`${id} output.reachedCounts must contain one row per pivot`);
      }
    }
    const converged = output.status?.converged;
    if (converged) {
      validatePackedView(converged, ['uint32'], `${id} output.status.converged`);
      if (converged.length !== 1) {
        throw new Error(`${id} output.status.converged must contain exactly one row`);
      }
    }
    this.maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    this.localIterations = props.localIterations ?? DEFAULT_LOCAL_ITERATIONS;
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
    this.betweennessScale = props.betweennessScale ?? 1;
    if (!Number.isFinite(Math.fround(this.betweennessScale))) {
      throw new Error(`${id} betweennessScale must be a finite number`);
    }
    // Reachability bounds the expanded size; betweenness also keeps three scratch arrays
    // (sigma, counters, delta) of one word per expanded node in one storage binding.
    // The lane count is fixed here, before the device is known, so it sizes the lanes for the
    // default 128 MiB binding (the reachability frontier queues hold 8 bytes per expanded node).
    const maximumExpandedNodeCount = output.betweenness
      ? Math.min(getCoreBindingElementLimit(8), getCoreBindingElementLimit(12))
      : getCoreBindingElementLimit(8);
    const maximumLaneCount = Math.max(
      1,
      Math.min(this.pivotCount, Math.floor(maximumExpandedNodeCount / this.nodeCount))
    );
    this.laneCount =
      props.laneCount ??
      Math.min(
        maximumLaneCount,
        recommendLaneCount({
          rowCount: this.pivotCount,
          nodeCount: this.nodeCount,
          edgeCount: forward.edgeCount
        })
      );
    validateCapacity(id, 'laneCount', this.laneCount, 1, this.pivotCount);
    validateMaximumCount(
      id,
      'laneCount * nodeCount',
      this.laneCount * this.nodeCount,
      maximumExpandedNodeCount
    );
    this.batchCount = Math.ceil(this.pivotCount / this.laneCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.betweenness,
        output.closeness,
        output.harmonicCloseness,
        output.reachedCounts,
        converged
      ],
      getInputViews(props)
    );
    const outputBuffers = [
      output.betweenness,
      output.closeness,
      output.harmonicCloseness,
      output.reachedCounts,
      converged
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns, in order: the adjacency copy, the seed expansion, the betweenness zeroing and batch
   * info nodes, then per batch `b` the nodes of one `GPUNetworkReachability` (IDs
   * `${id}-batch-${b}-reachability-*`), the closeness reduction (`-closeness`), and with betweenness
   * the phase reset, `-sigma-initialize`, `-sigma-${round}`, `-delta-initialize`, `-delta-${round}`,
   * `-accumulate` and `-status` nodes; finally an optional `-converged` reduction.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, pivotCount, laneCount, batchCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getInputViews(props),
      output.betweenness,
      output.closeness,
      output.harmonicCloseness,
      output.reachedCounts,
      output.status?.converged
    ]);
    const hasBetweenness = Boolean(output.betweenness);
    const hasCloseness = Boolean(
      output.closeness || output.harmonicCloseness || output.reachedCounts
    );
    const expandedNodeCount = laneCount * nodeCount;
    const maximumRounds = this.maximumIterations;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Per-batch sub-views start at multiples of the 256-byte alignment (rule 12): the seeds of
    // batch b at b * seedStride, its info and converged slots at b * 64.
    const seedStride = alignElementCount(laneCount);
    const expandedSeeds = createTransientView(
      graph,
      `${id}-expanded-seeds`,
      'uint32',
      batchCount * seedStride
    );
    nodes.push(
      createCentralityExpandSeedsNode<Parameters>(graph, {
        id: `${id}-expand-seeds`,
        nodeCount,
        laneCount,
        batchCount,
        seedStride,
        pivotNodes: props.pivotNodes,
        expandedSeeds
      })
    );
    const costs = createTransientView(graph, `${id}-costs`, 'float32', expandedNodeCount);
    const batchConverged = output.status?.converged
      ? createTransientView(
          graph,
          `${id}-batch-converged`,
          'uint32',
          batchCount * CENTRALITY_ALIGNMENT_ELEMENTS
        )
      : undefined;
    const batchInfo = hasCloseness
      ? createTransientView(
          graph,
          `${id}-batch-info`,
          'uint32',
          batchCount * CENTRALITY_ALIGNMENT_ELEMENTS
        )
      : undefined;
    if (batchInfo) {
      nodes.push(
        createCentralityBatchInfoNode<Parameters>(graph, {
          id: `${id}-batch-info`,
          laneCount,
          batchCount,
          batchInfo
        })
      );
    }

    // Betweenness-only state, shared by every batch (batches run in order and reset it).
    let betweennessState:
      | {
          adjacency: GraphDataView<'uint32'>;
          scratch: GraphDataView<'uint32'>;
          layout: ReturnType<typeof getCentralityAdjacencyLayout>;
          state: ReturnType<typeof createFrontierState>;
          sigmaPhase: FrontierPhase;
          deltaPhase: FrontierPhase;
        }
      | undefined;
    if (hasBetweenness && output.betweenness) {
      const layout = getCentralityAdjacencyLayout(
        nodeCount,
        props.topology.neighbors.length,
        props.reverseTopology.neighbors.length
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'adjacency',
        4 * layout.wordCount,
        'reduce the edge count'
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'scratch',
        4 * 3 * expandedNodeCount,
        'lower laneCount'
      );
      const adjacency = createTransientView(graph, `${id}-adjacency`, 'uint32', layout.wordCount);
      const scratch = createTransientView(graph, `${id}-scratch`, 'uint32', 3 * expandedNodeCount);
      const phaseLayout = getFrontierPhaseLayout(maximumRounds);
      const sigmaPhase: FrontierPhase = {
        maxRounds: maximumRounds,
        stampBase: 0,
        controlWordOffset: 0,
        dispatchSlotOffset: 0
      };
      const deltaPhase: FrontierPhase = {
        maxRounds: maximumRounds,
        stampBase: getFrontierPhaseMaximumStamp(sigmaPhase),
        controlWordOffset: phaseLayout.controlWordCount,
        dispatchSlotOffset: phaseLayout.dispatchSlotCount
      };
      const state = createFrontierState(graph, `${id}-frontier`, {
        nodeCount: expandedNodeCount,
        controlWordCount: 2 * phaseLayout.controlWordCount,
        dispatchSlotCount: 2 * phaseLayout.dispatchSlotCount
      });
      betweennessState = {adjacency, scratch, layout, state, sigmaPhase, deltaPhase};
      nodes.push(
        createCentralityPackNode<Parameters>(graph, {
          id: `${id}-pack-adjacency`,
          layout,
          forward: props.topology,
          reverse: props.reverseTopology,
          adjacency
        }),
        createCentralityZeroNode<Parameters>(graph, {
          id: `${id}-zero-betweenness`,
          betweenness: output.betweenness
        })
      );
    }

    for (let batch = 0; batch < batchCount; batch++) {
      const batchId = `${id}-batch-${batch}`;
      const seeds = getSubView(graph, expandedSeeds, batch * seedStride, laneCount);
      const convergedSlot = batchConverged
        ? getSubView(graph, batchConverged, batch * CENTRALITY_ALIGNMENT_ELEMENTS, 1)
        : undefined;
      nodes.push(
        ...new GPUNetworkReachability({
          id: `${batchId}-reachability`,
          // Lanes share the one CSR; reachability maps expanded node `l * nodeCount + u` onto it.
          topology: {
            offsets: props.topology.offsets,
            neighbors: props.topology.neighbors,
            weights: props.topology.weights
          },
          laneCount,
          startNodes: seeds,
          costLimit: props.costLimit,
          maximumIterations: this.maximumIterations,
          localIterations: this.localIterations,
          output: {costs, status: {converged: convergedSlot}}
        }).getCommandNodes(graph)
      );
      if (batchInfo) {
        nodes.push(
          createCentralityClosenessNode<Parameters>(graph, {
            id: `${batchId}-closeness`,
            nodeCount,
            laneCount,
            rowCount: pivotCount,
            costs,
            batchInfo: getSubView(graph, batchInfo, batch * CENTRALITY_ALIGNMENT_ELEMENTS, 1),
            closeness: output.closeness,
            harmonicCloseness: output.harmonicCloseness,
            reachedCounts: output.reachedCounts
          })
        );
      }
      if (betweennessState && output.betweenness) {
        const {adjacency, scratch, layout, state, sigmaPhase, deltaPhase} = betweennessState;
        const phaseProps = {layout, expandedNodeCount, state, adjacency, costs, scratch};
        nodes.push(
          createCentralityResetNode<Parameters>(graph, {
            id: `${batchId}-reset`,
            state,
            sigmaPhase,
            deltaPhase
          }),
          createCentralitySigmaInitializeNode<Parameters>(graph, {
            ...phaseProps,
            id: `${batchId}-sigma-initialize`,
            phase: sigmaPhase
          })
        );
        for (let round = 0; round < maximumRounds; round++) {
          nodes.push(
            createCentralitySigmaRoundNode<Parameters>(graph, {
              ...phaseProps,
              id: `${batchId}-sigma-${round}`,
              phase: sigmaPhase,
              round,
              // One hop per round: every sigma and delta hand-off crosses a dispatch boundary.
              localIterations: DEPENDENCY_LOCAL_ITERATIONS
            })
          );
        }
        nodes.push(
          createCentralityDeltaInitializeNode<Parameters>(graph, {
            ...phaseProps,
            id: `${batchId}-delta-initialize`,
            phase: deltaPhase
          })
        );
        for (let round = 0; round < maximumRounds; round++) {
          nodes.push(
            createCentralityDeltaRoundNode<Parameters>(graph, {
              ...phaseProps,
              id: `${batchId}-delta-${round}`,
              phase: deltaPhase,
              round,
              // One hop per round: every sigma and delta hand-off crosses a dispatch boundary.
              localIterations: DEPENDENCY_LOCAL_ITERATIONS
            })
          );
        }
        nodes.push(
          createCentralityAccumulateNode<Parameters>(graph, {
            id: `${batchId}-accumulate`,
            nodeCount,
            laneCount,
            expandedNodeCount,
            betweennessScale: this.betweennessScale,
            scratch,
            seeds,
            betweenness: output.betweenness
          })
        );
        if (convergedSlot) {
          nodes.push(
            createCentralityBatchStatusNode<Parameters>(graph, {
              id: `${batchId}-status`,
              state,
              sigmaPhase,
              deltaPhase,
              batchConverged: convergedSlot
            })
          );
        }
      }
    }
    if (output.status?.converged && batchConverged) {
      nodes.push(
        createCentralityConvergedNode<Parameters>(graph, {
          id: `${id}-converged`,
          batchCount,
          batchConverged,
          converged: output.status.converged
        })
      );
    }
    return nodes;
  }
}

/** Returns a packed range of `length` rows starting at row `firstRow` of a packed view. */
function getSubView<Format extends 'uint32' | 'float32', Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<Format>,
  firstRow: number,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(view.buffer, {
    format: view.format as Format,
    length,
    byteOffset: view.byteOffset + firstRow * view.byteStride
  });
}

/** Rounds an element count up to a multiple of the 256-byte binding alignment. */
function alignElementCount(count: number): number {
  return Math.ceil(count / CENTRALITY_ALIGNMENT_ELEMENTS) * CENTRALITY_ALIGNMENT_ELEMENTS;
}

/** Returns every read-only view of a centrality contributor. */
function getInputViews(props: GPUNetworkCentralityProps): (GraphDataView | undefined)[] {
  return [
    ...getGPUNetworkTopologyViews(props.topology),
    ...getGPUNetworkTopologyViews(props.reverseTopology),
    props.pivotNodes,
    props.costLimit
  ];
}
