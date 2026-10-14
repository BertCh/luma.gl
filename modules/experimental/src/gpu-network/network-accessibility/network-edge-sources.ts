// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {ACCESSIBILITY_NONE} from './network-accessibility-passes';

/** Properties of {@link createNetworkEdgeSourceNodes}. */
export type NetworkEdgeSourceNodesProps = {
  /** Node ID prefix; the nodes are `${id}-clear` and `${id}`. */
  id: string;
  /** Workload operation name of the nodes. */
  operation: string;
  /** CSR row offsets with `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** Output: the source node of every CSR edge row; its length is the edge count. */
  edgeSources: GraphDataView<'uint32'>;
  /** Number of CSR nodes (`offsets.length - 1`). */
  nodeCount: number;
};

/**
 * Builds the nodes that expand CSR `offsets` into one source node per edge row.
 *
 * Every row of `edgeSources` is written each frame: the rows outside the live CSR range
 * `offsets[0]..offsets[nodeCount]` (a CSR shorter than its compile-time edge capacity) get
 * `0xffffffff`, which every consumer reads as an out-of-range node and skips. Without the clear
 * those rows would keep node IDs from an earlier frame or from another transient that aliases the
 * buffer, and become phantom edges.
 *
 * @internal
 */
export function createNetworkEdgeSourceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: NetworkEdgeSourceNodesProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, offsets, edgeSources, nodeCount} = props;
  return [
    createFillNode<Parameters>(graph, {
      id: `${id}-clear`,
      operation,
      view: edgeSources,
      type: 'u32',
      value: `${ACCESSIBILITY_NONE}u`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id,
      operation,
      variant: 'edge-sources',
      bindings: [
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'edgeSources', view: edgeSources, type: 'u32', access: 'read_write'}
      ],
      invocationCount: nodeCount,
      declarations: `const EDGE_COUNT: u32 = ${edgeSources.length}u;`,
      body: `let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    edgeSources[edgeSourcesOffset + edge] = index;
  }`
    })
  ];
}
