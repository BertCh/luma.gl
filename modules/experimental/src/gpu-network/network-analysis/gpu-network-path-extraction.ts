// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  createPathEdgeLengthsNode,
  createPathMeasureNode,
  createPathResolveEdgesNode,
  createPathTotalsNode,
  createPathWriteLinksNode,
  createPathWriteNodesNode
} from './network-path-extraction-passes';

/** Largest accepted compile-time `maxPathLength`. */
export const GPU_NETWORK_PATH_MAXIMUM_LENGTH = 65536;

/** Edge slot value when no CSR edge matches a predecessor link. */
export const GPU_NETWORK_PATH_NO_EDGE = 0xffffffff;

const DEFAULT_MAXIMUM_PATH_LENGTH = 1024;
const OPERATION = 'GPUNetworkPathExtraction';

/**
 * Optional edge extraction of {@link GPUNetworkPathExtractionProps}.
 *
 * Compile-time: the presence of this object, its view lengths, and `output` capacity. Per-frame:
 * the CSR contents and `edgeIds`.
 */
export type GPUNetworkPathExtractionEdges = {
  /** Forward CSR row offsets with `nodeCount + 1` rows, from the CSR that produced `costs`. */
  offsets: GraphDataView<'uint32'>;
  /** CSR destination node per edge. */
  neighbors: GraphDataView<'uint32'>;
  /** CSR edge cost, with the same length as `neighbors`. */
  weights: GraphDataView<'float32'>;
  /** Optional stable edge IDs aligned with `neighbors`; output is `edgeIds[edge]` instead of the CSR index. */
  edgeIds?: GraphDataView<'uint32'>;
  /**
   * Concatenated edge lists in target-row order. A path with `n` nodes contributes
   * `max(n - 1, 0)` edges ordered source to target. Slots with no matching CSR edge hold
   * {@link GPU_NETWORK_PATH_NO_EDGE}. Its overflow also reports truncated paths.
   */
  output: GPUCompactOutput;
  /** Optional `targets.length + 1` rows: unclamped start of each path in `output.ids`, then the total. */
  pathOffsets?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUNetworkPathExtraction}.
 *
 * Compile-time: node count, target capacity, `maxPathLength`, output capacities, and which
 * optional views exist. Per-frame: the contents of `predecessors`, `costs`, `targets`,
 * `targetCount`, `nodeIds`, and the CSR.
 */
export type GPUNetworkPathExtractionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-path-extraction'`. */
  id?: string;
  /** Per-node predecessor or `GPU_NETWORK_REACHABILITY_NONE`; its length defines the node count. Per-frame contents. */
  predecessors: GraphDataView<'uint32'>;
  /** Per-node shortest-path cost, `+Infinity` when unreached, with one row per node. Per-frame contents. */
  costs: GraphDataView<'float32'>;
  /** Target node indices. Capacity is compile-time, contents are per-frame. May have zero rows. */
  targets: GraphDataView<'uint32'>;
  /** Optional one-row active target count; rows at or after it are inactive. Per-frame. */
  targetCount?: GraphDataView<'uint32'>;
  /** Compile-time bound on nodes per path, 1 to 65536. Defaults to 1024. */
  maxPathLength?: number;
  /** Optional stable node IDs with one row per node; output IDs are `nodeIds[node]` instead of `node`. */
  nodeIds?: GraphDataView<'uint32'>;
  /**
   * Concatenated node paths in target-row order, each ordered source to target. `count` is
   * clamped to `ids.length`; `overflow` is set when the total exceeds capacity or any walk was
   * truncated by `maxPathLength`.
   */
  output: GPUCompactOutput;
  /** Optional `targets.length + 1` rows: unclamped start of each path in `output.ids`, then the total. */
  pathOffsets?: GraphDataView<'uint32'>;
  /** Optional per-target `costs[target]` when found, `+Infinity` otherwise. */
  pathCosts?: GraphDataView<'float32'>;
  /** Optional per-target flag: 1 when a complete path was found, otherwise 0. */
  pathFound?: GraphDataView<'uint32'>;
  /** Optional edge extraction. Requires the CSR that produced `costs`. */
  edges?: GPUNetworkPathExtractionEdges;
};

/**
 * Extracts ordered source-to-target node and edge lists from a shortest-path predecessor array
 * on the GPU, ready for a path layer or highlight.
 *
 * Each active target walks its predecessor chain, bounded by `maxPathLength`, so cycles and
 * garbage predecessors terminate: such a target reports not found, and a too-long walk also sets
 * the overflow flag. A found target that is itself a root yields a one-node path. Target rows that
 * are out of range, unreached (non-finite cost) or inactive yield empty paths. Path lengths are
 * prefix-summed with `GPUScan` so lists are packed in target-row order and written in parallel.
 * Edges link each predecessor to its successor using the smallest CSR edge whose weight explains
 * the cost difference exactly in f32, falling back to the smallest edge with the right neighbor.
 * Every encoding rewrites every output.
 *
 * Non-goals: k-shortest paths, path geometry, and computing the predecessors themselves, which
 * belong to `GPUNetworkReachability`.
 */
export class GPUNetworkPathExtraction implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkPathExtractionProps;
  /** Resolved compile-time walk bound. */
  readonly maxPathLength: number;

  constructor(props: GPUNetworkPathExtractionProps) {
    this.id = props.id ?? 'network-path-extraction';
    this.props = props;
    this.maxPathLength = props.maxPathLength ?? DEFAULT_MAXIMUM_PATH_LENGTH;
    const {id} = this;
    const {edges} = props;
    for (const [name, view] of [
      ['predecessors', props.predecessors],
      ['targets', props.targets],
      ['targetCount', props.targetCount],
      ['nodeIds', props.nodeIds],
      ['pathOffsets', props.pathOffsets],
      ['pathFound', props.pathFound],
      ['edges.offsets', edges?.offsets],
      ['edges.neighbors', edges?.neighbors],
      ['edges.edgeIds', edges?.edgeIds],
      ['edges.pathOffsets', edges?.pathOffsets]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['costs', props.costs],
      ['pathCosts', props.pathCosts],
      ['edges.weights', edges?.weights]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    const nodeCount = props.predecessors.length;
    const targetCapacity = props.targets.length;
    if (nodeCount < 1) {
      throw new Error(`${id} predecessors must contain at least one node`);
    }
    if (props.costs.length !== nodeCount) {
      throw new Error(`${id} costs length must equal predecessors length`);
    }
    if (props.nodeIds && props.nodeIds.length !== nodeCount) {
      throw new Error(`${id} nodeIds length must equal the node count`);
    }
    if (props.targetCount && props.targetCount.length !== 1) {
      throw new Error(`${id} targetCount must contain exactly one row`);
    }
    for (const [name, view] of [
      ['pathCosts', props.pathCosts],
      ['pathFound', props.pathFound]
    ] as const) {
      if (view && view.length !== targetCapacity) {
        throw new Error(`${id} ${name} length must equal targets length`);
      }
    }
    for (const [name, view] of [
      ['pathOffsets', props.pathOffsets],
      ['edges.pathOffsets', edges?.pathOffsets]
    ] as const) {
      if (view && view.length !== targetCapacity + 1) {
        throw new Error(`${id} ${name} must contain one more row than targets`);
      }
    }
    if (edges) {
      if (edges.offsets.length !== nodeCount + 1) {
        throw new Error(`${id} edges.offsets must contain one more row than predecessors`);
      }
      if (edges.weights.length !== edges.neighbors.length) {
        throw new Error(`${id} edges.weights length must equal neighbors length`);
      }
      if (edges.edgeIds && edges.edgeIds.length !== edges.neighbors.length) {
        throw new Error(`${id} edges.edgeIds length must equal neighbors length`);
      }
      validateCompactOutput(`${id} edges`, edges.output);
    }
    if (
      !Number.isSafeInteger(this.maxPathLength) ||
      this.maxPathLength < 1 ||
      this.maxPathLength > GPU_NETWORK_PATH_MAXIMUM_LENGTH
    ) {
      throw new Error(
        `${id} maxPathLength must be an integer in [1, ${GPU_NETWORK_PATH_MAXIMUM_LENGTH}]`
      );
    }
    validateCompactOutput(id, props.output);
    const outputs = [
      props.output.ids,
      props.output.count,
      props.output.overflow,
      props.output.requiredCount,
      props.pathOffsets,
      props.pathCosts,
      props.pathFound,
      edges?.output.ids,
      edges?.output.count,
      edges?.output.overflow,
      edges?.output.requiredCount,
      edges?.pathOffsets
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const inputs = [
      props.predecessors,
      props.costs,
      props.targets,
      props.targetCount,
      props.nodeIds,
      edges?.offsets,
      edges?.neighbors,
      edges?.weights,
      edges?.edgeIds
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

  /** Returns clear, measure, scan, total, write, and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maxPathLength} = this;
    const {edges} = props;
    validateGraphViewsBelongToGraph(id, graph, getViews(props));
    const nodeCount = props.predecessors.length;
    const targetCapacity = props.targets.length;
    const nodeTotal = createTransientView(graph, `${id}-node-total`, 'uint32', 1);
    const edgeTotal = edges
      ? createTransientView(graph, `${id}-edge-total`, 'uint32', 1)
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];

    if (targetCapacity === 0) {
      nodes.push(
        createPathTotalsNode<Parameters>(graph, {
          id: `${id}-totals`,
          targetCapacity,
          nodes: {total: nodeTotal, pathOffsets: props.pathOffsets},
          edges: edges && edgeTotal ? {total: edgeTotal, pathOffsets: edges.pathOffsets} : undefined
        }),
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          requiredCount: nodeTotal,
          output: props.output
        })
      );
      if (edges && edgeTotal) {
        nodes.push(
          createPublishNode<Parameters>(graph, {
            id: `${id}-edge-publish`,
            operation: OPERATION,
            requiredCount: edgeTotal,
            output: edges.output
          })
        );
      }
      return nodes;
    }

    const status = createTransientView(graph, `${id}-status`, 'uint32', 1);
    const nodeLengths = createTransientView(graph, `${id}-node-lengths`, 'uint32', targetCapacity);
    const nodeStarts = createTransientView(graph, `${id}-node-starts`, 'uint32', targetCapacity);
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        view: status,
        type: 'u32',
        value: '0u'
      }),
      createPathMeasureNode<Parameters>(graph, {
        id: `${id}-measure`,
        nodeCount,
        maxPathLength,
        predecessors: props.predecessors,
        costs: props.costs,
        targets: props.targets,
        targetCount: props.targetCount,
        nodeLengths,
        status,
        pathFound: props.pathFound,
        pathCosts: props.pathCosts
      }),
      ...new GPUScan({
        id: `${id}-node-scan`,
        input: nodeLengths,
        output: nodeStarts
      }).getCommandNodes(graph)
    );
    let edgeLengths: GraphDataView<'uint32'> | undefined;
    let edgeStarts: GraphDataView<'uint32'> | undefined;
    if (edges) {
      edgeLengths = createTransientView(graph, `${id}-edge-lengths`, 'uint32', targetCapacity);
      edgeStarts = createTransientView(graph, `${id}-edge-starts`, 'uint32', targetCapacity);
      nodes.push(
        createPathEdgeLengthsNode<Parameters>(graph, {
          id: `${id}-edge-lengths`,
          nodeLengths,
          edgeLengths
        }),
        ...new GPUScan({
          id: `${id}-edge-scan`,
          input: edgeLengths,
          output: edgeStarts
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      createPathTotalsNode<Parameters>(graph, {
        id: `${id}-totals`,
        targetCapacity,
        nodes: {
          lengths: nodeLengths,
          starts: nodeStarts,
          total: nodeTotal,
          pathOffsets: props.pathOffsets
        },
        edges:
          edges && edgeTotal
            ? {
                lengths: edgeLengths,
                starts: edgeStarts,
                total: edgeTotal,
                pathOffsets: edges.pathOffsets
              }
            : undefined
      })
    );
    if (props.output.ids.length > 0) {
      nodes.push(
        createPathWriteNodesNode<Parameters>(graph, {
          id: `${id}-write-nodes`,
          capacity: props.output.ids.length,
          predecessors: props.predecessors,
          targets: props.targets,
          nodeLengths,
          nodeStarts,
          nodeIds: props.nodeIds,
          ids: props.output.ids
        })
      );
    }
    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: nodeTotal,
        output: props.output,
        overflowSources: [status]
      })
    );
    if (edges && edgeTotal && edgeStarts) {
      const edgeCapacity = edges.output.ids.length;
      if (edgeCapacity > 0) {
        const linkTails = createTransientView(graph, `${id}-link-tails`, 'uint32', edgeCapacity);
        const linkHeads = createTransientView(graph, `${id}-link-heads`, 'uint32', edgeCapacity);
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-link-clear`,
            operation: OPERATION,
            view: linkTails,
            type: 'u32',
            value: '0xffffffffu'
          }),
          createPathWriteLinksNode<Parameters>(graph, {
            id: `${id}-write-links`,
            capacity: edgeCapacity,
            predecessors: props.predecessors,
            targets: props.targets,
            nodeLengths,
            edgeStarts,
            linkTails,
            linkHeads
          }),
          createPathResolveEdgesNode<Parameters>(graph, {
            id: `${id}-write-edges`,
            nodeCount,
            edgeCount: edges.neighbors.length,
            linkTails,
            linkHeads,
            costs: props.costs,
            offsets: edges.offsets,
            neighbors: edges.neighbors,
            weights: edges.weights,
            edgeIds: edges.edgeIds,
            ids: edges.output.ids
          })
        );
      }
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-edge-publish`,
          operation: OPERATION,
          requiredCount: edgeTotal,
          output: edges.output,
          overflowSources: [status]
        })
      );
    }
    return nodes;
  }
}

/** Returns every view referenced by the props. */
function getViews(props: GPUNetworkPathExtractionProps): (GraphDataView | undefined)[] {
  const {edges} = props;
  return [
    props.predecessors,
    props.costs,
    props.targets,
    props.targetCount,
    props.nodeIds,
    props.output.ids,
    props.output.count,
    props.output.overflow,
    props.output.requiredCount,
    props.pathOffsets,
    props.pathCosts,
    props.pathFound,
    edges?.offsets,
    edges?.neighbors,
    edges?.weights,
    edges?.edgeIds,
    edges?.output.ids,
    edges?.output.count,
    edges?.output.overflow,
    edges?.output.requiredCount,
    edges?.pathOffsets
  ];
}
