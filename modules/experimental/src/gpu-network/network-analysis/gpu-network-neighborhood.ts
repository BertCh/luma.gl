// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkAdjacency
} from '../../analysis-core/network-topology';
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
 * Compile-time: node and edge counts, start capacity, `maximumHops`, a numeric `startCount`,
 * output capacities, and which optional views exist. Per-frame: the contents of the topology,
 * `startNodes`, a `startCount` view, `hops`, `nodeIds`, and `edgeIds`.
 */
export type GPUNetworkNeighborhoodProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-neighborhood'`. */
  id?: string;
  /**
   * Network connectivity; edge weights are ignored (every edge is one hop). `offsets` has
   * `nodeCount + 1` monotonic rows and `neighbors` destinations `>= nodeCount` are ignored.
   * Compile-time lengths.
   */
  topology: GPUNetworkAdjacency;
  /** Start node indices. Capacity is compile-time, contents are per-frame. Out-of-range ignored. */
  startNodes: GraphDataView<'uint32'>;
  /**
   * Optional active start count. A number is a compile-time constant in `[0, startNodes.length]`;
   * a one-row view is read per frame. Defaults to every row.
   */
  startCount?: number | GraphDataView<'uint32'>;
  /** One-row hop radius k. Per-frame; clamped to `maximumHops`. */
  hops: GraphDataView<'uint32'>;
  /**
   * Compile-time maximum hop radius, an integer in
   * `[1, GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS]`. Defaults to 8.
   */
  maximumHops?: number;
  /** Optional stable node IDs with one row per node, emitted in `output.nodes.ids`. Per-frame contents. */
  nodeIds?: GraphDataView<'uint32'>;
  /** Optional stable edge IDs aligned with `topology.neighbors`, emitted in `output.edges.ids`. Per-frame contents. */
  edgeIds?: GraphDataView<'uint32'>;
  /** Views the contributor writes. */
  output: GPUNetworkNeighborhoodOutput;
};

/** Views written by {@link GPUNetworkNeighborhood}. */
export type GPUNetworkNeighborhoodOutput = {
  /**
   * Per-node hop distance, or `GPU_NETWORK_REACHABILITY_NONE` outside the ego network. Its length
   * defines the node count.
   */
  hopDistances: GraphDataView<'uint32'>;
  /** Optional per-node 0/1 membership mask. */
  nodeMask?: GraphDataView<'uint32'>;
  /** Optional per-edge 0/1 mask of induced edges (both endpoints inside). Length equals `topology.neighbors`. */
  edgeMask?: GraphDataView<'uint32'>;
  /** Optional bounded compact IDs of ego nodes in row order. Capacity is compile-time. */
  nodes?: GPUCompactOutput;
  /** Optional bounded compact IDs of induced edges in CSR order. Capacity is compile-time. */
  edges?: GPUCompactOutput;
};

/**
 * Computes the k-hop ego network around per-frame start nodes: per-node hop distance, a node mask,
 * a mask of induced edges, and compact node and edge ID lists for highlighting.
 *
 * Design: composes {@link GPUNetworkReachability} over the same CSR with a graph-transient
 * unit-weight column, `maximumIterations = maximumHops`, and the caller's `hops` view as
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
  readonly maximumHops: number;

  /**
   * @throws If a view has the wrong format or length, `maximumHops` or a numeric `startCount` is
   * out of range, or an output shares a buffer with an input or another output.
   */
  constructor(props: GPUNetworkNeighborhoodProps) {
    this.id = props.id ?? 'network-neighborhood';
    this.props = props;
    this.maximumHops = props.maximumHops ?? DEFAULT_MAXIMUM_HOPS;
    const {id} = this;
    const {output} = props;
    const {nodeCount: networkNodeCount, edgeCount} = validateGPUNetworkTopology(id, props.topology);
    for (const [name, view] of [
      ['startNodes', props.startNodes],
      ['startCount', typeof props.startCount === 'object' ? props.startCount : undefined],
      ['hops', props.hops],
      ['nodeIds', props.nodeIds],
      ['edgeIds', props.edgeIds],
      ['output.hopDistances', output.hopDistances],
      ['output.nodeMask', output.nodeMask],
      ['output.edgeMask', output.edgeMask]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    for (const compactOutput of [output.nodes, output.edges]) {
      if (compactOutput) {
        validateCompactOutput(id, compactOutput);
      }
    }
    const nodeCount = output.hopDistances.length;
    if (nodeCount < 1) {
      throw new Error(`${id} output.hopDistances must contain at least one node`);
    }
    if (networkNodeCount !== nodeCount) {
      throw new Error(`${id} topology must contain one node per output.hopDistances row`);
    }
    if (typeof props.startCount === 'number') {
      validateCapacity(id, 'startCount', props.startCount, 0, props.startNodes.length);
    }
    for (const [name, view] of [
      ['startCount', typeof props.startCount === 'object' ? props.startCount : undefined],
      ['hops', props.hops]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    for (const [name, view, expected, label] of [
      ['nodeIds', props.nodeIds, nodeCount, 'output.hopDistances'],
      ['output.nodeMask', output.nodeMask, nodeCount, 'output.hopDistances'],
      ['edgeIds', props.edgeIds, edgeCount, 'topology.neighbors'],
      ['output.edgeMask', output.edgeMask, edgeCount, 'topology.neighbors']
    ] as const) {
      if (view && view.length !== expected) {
        throw new Error(`${id} ${name} length must equal ${label} length`);
      }
    }
    validateCapacity(
      id,
      'maximumHops',
      this.maximumHops,
      1,
      GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
    );
    const outputs = getOutputViews(output);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputViews(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers from each other`);
    }
  }

  /** Returns weight fill, reachability, hop conversion, induced-edge, and compaction nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumHops} = this;
    const {output} = props;
    const {offsets, neighbors} = props.topology;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getInputViews(props),
      ...getOutputViews(output)
    ]);
    const nodeCount = output.hopDistances.length;
    const edgeCount = neighbors.length;
    const needsEdges = Boolean(output.edges || output.edgeMask);
    const needsNodeMask = Boolean(output.nodeMask || output.nodes || needsEdges);
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
        topology: {offsets, neighbors, weights: unitWeights},
        startNodes: props.startNodes,
        startCount: props.startCount,
        maximumIterations: maximumHops,
        activeIterations: props.hops,
        output: {costs}
      }).getCommandNodes(graph)
    );

    const nodeMask =
      output.nodeMask ??
      (needsNodeMask
        ? createTransientView(graph, `${id}-node-mask`, 'uint32', nodeCount)
        : undefined);
    nodes.push(
      createNeighborhoodHopsNode<Parameters>(graph, {
        id: `${id}-hops`,
        nodeCount,
        maximumHops,
        costs,
        hops: props.hops,
        hopDistances: output.hopDistances,
        nodeMask
      })
    );

    let edgeMask: GraphDataView<'uint32'> | undefined;
    if (needsEdges && edgeCount > 0 && nodeMask) {
      edgeMask =
        output.edgeMask ?? createTransientView(graph, `${id}-edge-mask`, 'uint32', edgeCount);
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
          offsets,
          neighbors,
          nodeMask,
          edgeMask
        })
      );
    }

    if (output.nodes && nodeMask) {
      nodes.push(
        ...getCompactNodes(graph, {
          id: `${id}-node`,
          mask: nodeMask,
          sourceIds: props.nodeIds,
          output: output.nodes
        })
      );
    }
    if (output.edges) {
      if (edgeMask) {
        nodes.push(
          ...getCompactNodes(graph, {
            id: `${id}-edge`,
            mask: edgeMask,
            sourceIds: props.edgeIds,
            output: output.edges
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
            output: output.edges
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

/** Returns every read-only view referenced by the props. */
function getInputViews(props: GPUNetworkNeighborhoodProps): (GraphDataView | undefined)[] {
  return [
    ...getGPUNetworkTopologyViews(props.topology),
    props.startNodes,
    typeof props.startCount === 'object' ? props.startCount : undefined,
    props.hops,
    props.nodeIds,
    props.edgeIds
  ];
}

/** Returns every writable view of the output bag. */
function getOutputViews(
  output: GPUNetworkNeighborhoodOutput
): (GraphDataView<'uint32'> | undefined)[] {
  return [
    output.hopDistances,
    output.nodeMask,
    output.edgeMask,
    output.nodes?.ids,
    output.nodes?.status.count,
    output.nodes?.status.overflow,
    output.nodes?.status.requiredCount,
    output.edges?.ids,
    output.edges?.status.count,
    output.edges?.status.overflow,
    output.edges?.status.requiredCount
  ];
}
