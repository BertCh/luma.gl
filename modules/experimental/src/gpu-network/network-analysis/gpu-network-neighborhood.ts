// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  GPUNetworkReachability,
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
} from '../network-reachability/index';
import {
  createNeighborhoodEdgeMaskNode,
  createNeighborhoodHopsNode
} from './network-neighborhood-passes';

const DEFAULT_MAXIMUM_HOPS = 8;
const OPERATION = 'GPUNetworkNeighborhood';

/**
 * Properties for {@link GPUNetworkNeighborhood}.
 *
 * Compile-time: node and edge counts, seed capacity, `maxHops`, output capacities, and which
 * optional views exist. Per-frame: the contents of the CSR, `seeds`, `seedCount`, `hops`,
 * `nodeIds`, and `edgeIds`.
 */
export type GPUNetworkNeighborhoodProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-neighborhood'`. */
  id?: string;
  /** CSR row offsets with `nodeCount + 1` rows. Offsets must be monotonic. Compile-time length. */
  offsets: GraphDataView<'uint32'>;
  /** CSR destination node per edge. Destinations `>= nodeCount` are ignored. Compile-time length. */
  neighbors: GraphDataView<'uint32'>;
  /** Seed node indices. Capacity is compile-time, contents are per-frame. Out-of-range ignored. */
  seeds: GraphDataView<'uint32'>;
  /** Optional one-row active seed count. Per-frame. */
  seedCount?: GraphDataView<'uint32'>;
  /** One-row hop radius k. Per-frame; clamped to `maxHops`. */
  hops: GraphDataView<'uint32'>;
  /**
   * Compile-time maximum hop radius, an integer in
   * `[1, GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS]`. Defaults to 8.
   */
  maxHops?: number;
  /** Optional stable node IDs with one row per node, emitted in `nodes.ids`. Per-frame contents. */
  nodeIds?: GraphDataView<'uint32'>;
  /** Optional stable edge IDs aligned with `neighbors`, emitted in `edges.ids`. Per-frame contents. */
  edgeIds?: GraphDataView<'uint32'>;
  /**
   * Per-node hop distance, or `GPU_NETWORK_REACHABILITY_NONE` outside the ego network. Its length
   * defines the node count.
   */
  hopDistances: GraphDataView<'uint32'>;
  /** Optional per-node 0/1 membership mask. */
  nodeMask?: GraphDataView<'uint32'>;
  /** Optional per-edge 0/1 mask of induced edges (both endpoints inside). Length equals `neighbors`. */
  edgeMask?: GraphDataView<'uint32'>;
  /** Optional bounded compact IDs of ego nodes in row order. Capacity is compile-time. */
  nodes?: GPUCompactOutput;
  /** Optional bounded compact IDs of induced edges in CSR order. Capacity is compile-time. */
  edges?: GPUCompactOutput;
};

/**
 * Computes the k-hop ego network around per-frame seed nodes: per-node hop distance, a node mask,
 * a mask of induced edges, and compact node and edge ID lists for highlighting.
 *
 * Design: composes {@link GPUNetworkReachability} over the same CSR with a graph-transient
 * unit-weight column, `maxIterations = maxHops`, and the caller's `hops` view as
 * `activeIterations`. After k gated relaxations every node within k hops holds its exact hop count
 * as cost (the Bellman-Ford invariant) and every farther node holds a cost above k or
 * `+Infinity`, so `hop = cost <= k ? cost : NONE` is exact. Converged frames stop early through
 * reachability's GPU gate. Every encoding recomputes every output word from scratch.
 *
 * `GPUGraphTraversal` is not used because it publishes only a reachability mask, not hop
 * distances. The gpu-graph `GPUGraphBreadthFirstSearch` is not used because it requires
 * caller-owned physical `GPUVector` distances and predecessors plus a `GPUGraphTopology`, not
 * graph views and transients.
 *
 * Direction: edges are followed outgoing over the given CSR. Pass a reverse CSR for incoming ego
 * networks. Undirected ego networks need a symmetric CSR, which road networks normally provide.
 *
 * Non-goals: following `'both'` directions over an asymmetric CSR, weighted radii (use
 * `GPUNetworkReachability` with a cost limit), and CSRs with non-monotonic offsets (induced-edge
 * rows of different nodes could overlap and race).
 */
export class GPUNetworkNeighborhood implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkNeighborhoodProps;
  /** Resolved compile-time maximum hop radius. */
  readonly maxHops: number;

  constructor(props: GPUNetworkNeighborhoodProps) {
    this.id = props.id ?? 'network-neighborhood';
    this.props = props;
    this.maxHops = props.maxHops ?? DEFAULT_MAXIMUM_HOPS;
    const {id} = this;
    for (const [name, view] of [
      ['offsets', props.offsets],
      ['neighbors', props.neighbors],
      ['seeds', props.seeds],
      ['seedCount', props.seedCount],
      ['hops', props.hops],
      ['nodeIds', props.nodeIds],
      ['edgeIds', props.edgeIds],
      ['hopDistances', props.hopDistances],
      ['nodeMask', props.nodeMask],
      ['edgeMask', props.edgeMask]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const output of [props.nodes, props.edges]) {
      if (output) {
        validateCompactOutput(id, output);
      }
    }
    const nodeCount = props.hopDistances.length;
    const edgeCount = props.neighbors.length;
    if (nodeCount < 1) {
      throw new Error(`${id} hopDistances must contain at least one node`);
    }
    if (props.offsets.length !== nodeCount + 1) {
      throw new Error(`${id} offsets must contain one more row than hopDistances`);
    }
    for (const [name, view] of [
      ['seedCount', props.seedCount],
      ['hops', props.hops]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    for (const [name, view, expected, label] of [
      ['nodeIds', props.nodeIds, nodeCount, 'hopDistances'],
      ['nodeMask', props.nodeMask, nodeCount, 'hopDistances'],
      ['edgeIds', props.edgeIds, edgeCount, 'neighbors'],
      ['edgeMask', props.edgeMask, edgeCount, 'neighbors']
    ] as const) {
      if (view && view.length !== expected) {
        throw new Error(`${id} ${name} length must equal ${label} length`);
      }
    }
    if (
      !Number.isSafeInteger(this.maxHops) ||
      this.maxHops < 1 ||
      this.maxHops > GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
    ) {
      throw new Error(
        `${id} maxHops must be an integer in [1, ${GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS}]`
      );
    }
    const outputs = getOutputViews(props);
    const inputs = [
      props.offsets,
      props.neighbors,
      props.seeds,
      props.seedCount,
      props.hops,
      props.nodeIds,
      props.edgeIds
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers from each other`);
    }
  }

  /** Returns weight fill, reachability, hop conversion, induced-edge, and compaction nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maxHops} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.offsets,
      props.neighbors,
      props.seeds,
      props.seedCount,
      props.hops,
      props.nodeIds,
      props.edgeIds,
      ...getOutputViews(props)
    ]);
    const nodeCount = props.hopDistances.length;
    const edgeCount = props.neighbors.length;
    const needsEdges = Boolean(props.edges || props.edgeMask);
    const needsNodeMask = Boolean(props.nodeMask || props.nodes || needsEdges);
    const nodes: GPUCommandNode<Parameters>[] = [];

    const unitWeights = createTransientView(graph, `${id}-unit-weights`, 'float32', edgeCount);
    if (edgeCount > 0) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-unit-weights`,
          operation: OPERATION,
          view: unitWeights,
          type: 'f32',
          value: '1.0'
        })
      );
    }
    const costs = createTransientView(graph, `${id}-costs`, 'float32', nodeCount);
    nodes.push(
      ...new GPUNetworkReachability({
        id: `${id}-reachability`,
        offsets: props.offsets,
        neighbors: props.neighbors,
        weights: unitWeights,
        sources: props.seeds,
        sourceCount: props.seedCount,
        maxIterations: maxHops,
        activeIterations: props.hops,
        costs
      }).getCommandNodes(graph)
    );

    const nodeMask =
      props.nodeMask ??
      (needsNodeMask
        ? createTransientView(graph, `${id}-node-mask`, 'uint32', nodeCount)
        : undefined);
    nodes.push(
      createNeighborhoodHopsNode<Parameters>(graph, {
        id: `${id}-hops`,
        nodeCount,
        maximumHops: maxHops,
        costs,
        hops: props.hops,
        hopDistances: props.hopDistances,
        nodeMask
      })
    );

    let edgeMask: GraphDataView<'uint32'> | undefined;
    if (needsEdges && edgeCount > 0 && nodeMask) {
      edgeMask =
        props.edgeMask ?? createTransientView(graph, `${id}-edge-mask`, 'uint32', edgeCount);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-edge-clear`,
          operation: OPERATION,
          view: edgeMask,
          type: 'u32',
          value: '0u'
        }),
        createNeighborhoodEdgeMaskNode<Parameters>(graph, {
          id: `${id}-edge-mask`,
          nodeCount,
          edgeCount,
          offsets: props.offsets,
          neighbors: props.neighbors,
          nodeMask,
          edgeMask
        })
      );
    }

    if (props.nodes && nodeMask) {
      nodes.push(
        ...getCompactNodes(graph, {
          id: `${id}-node`,
          mask: nodeMask,
          sourceIds: props.nodeIds,
          output: props.nodes
        })
      );
    }
    if (props.edges) {
      if (edgeMask) {
        nodes.push(
          ...getCompactNodes(graph, {
            id: `${id}-edge`,
            mask: edgeMask,
            sourceIds: props.edgeIds,
            output: props.edges
          })
        );
      } else {
        // No edges at all: publish an empty list without compaction.
        const total = createTransientView(graph, `${id}-edge-total`, 'uint32', 1);
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-edge-total`,
            operation: OPERATION,
            view: total,
            type: 'u32',
            value: '0u'
          }),
          createPublishNode<Parameters>(graph, {
            id: `${id}-edge-publish`,
            operation: OPERATION,
            requiredCount: total,
            output: props.edges
          })
        );
      }
    }
    return nodes;
  }
}

/** Compacts masked rows into a bounded output through visibility plus publish nodes. */
function getCompactNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    mask: GraphDataView<'uint32'>;
    sourceIds?: GraphDataView<'uint32'>;
    output: GPUCompactOutput;
  }
): GPUCommandNode<Parameters>[] {
  const {id, mask, output} = props;
  const rows = mask.length;
  // Compact straight into the caller's IDs when they can hold every row.
  const direct = output.ids.length >= rows;
  const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
  const compactIds = direct
    ? output.ids
    : createTransientView(graph, `${id}-compact-ids`, 'uint32', rows);
  return [
    ...new GPUVisibilityWorkflow({
      id: `${id}-visibility`,
      predicates: [{kind: 'selection', mask}],
      output: compactIds,
      count: total,
      // The mask is already canonical, so skip the redundant compose pass.
      outputMask: mask,
      sourceIds: props.sourceIds
    }).getCommandNodes(graph),
    createPublishNode<Parameters>(graph, {
      id: `${id}-publish`,
      operation: OPERATION,
      requiredCount: total,
      compactIds: direct ? undefined : compactIds,
      output
    })
  ];
}

/** Returns every writable view referenced by the props. */
function getOutputViews(
  props: GPUNetworkNeighborhoodProps
): (GraphDataView<'uint32'> | undefined)[] {
  return [
    props.hopDistances,
    props.nodeMask,
    props.edgeMask,
    props.nodes?.ids,
    props.nodes?.count,
    props.nodes?.overflow,
    props.nodes?.requiredCount,
    props.edges?.ids,
    props.edges?.count,
    props.edges?.overflow,
    props.edges?.requiredCount
  ];
}
