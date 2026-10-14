// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

/**
 * CPU reference: builds a node-to-position permutation by sorting nodes by `groups`, then
 * `tieKeys` (when given), then node index. Returns `order[node] = position`.
 */
export function computeAdjacencyMatrixOrder(
  groups: ArrayLike<number>,
  tieKeys?: ArrayLike<number>
): Uint32Array {
  const count = groups.length;
  const vertices = Array.from({length: count}, (_, vertex) => vertex);
  vertices.sort(
    (a, b) => groups[a] - groups[b] || (tieKeys ? tieKeys[a] - tieKeys[b] : 0) || a - b
  );
  const order = new Uint32Array(count);
  vertices.forEach((vertex, position) => {
    order[vertex] = position;
  });
  return order;
}

/** Properties for {@link GPUAdjacencyMatrixOrder}. */
export type GPUAdjacencyMatrixOrderProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'adjacency-matrix-order'`. */
  id?: string;
  /** Group label per node, for example a community ID. Primary key, ascending. */
  groups: GraphDataView<'uint32'>;
  /** Optional secondary key per node, for example a degree. Ascending; ties keep vertex order. */
  tieKeys?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUAdjacencyMatrixOrderOutput;
};

/** GPU-written output of {@link GPUAdjacencyMatrixOrder}. */
export type GPUAdjacencyMatrixOrderOutput = {
  /** `order[node] = position` permutation, `groups.length` rows. */
  order: GraphDataView<'uint32'>;
};

/**
 * GPU counterpart of {@link computeAdjacencyMatrixOrder}.
 *
 * Two stable `GPUSort` passes (tie key, then group) over graph-owned scratch, then one scatter
 * kernel inverting the sorted node list into `order`. Result is bit-identical to the CPU helper
 * because the sorts are stable and the initial list is the identity.
 */
export class GPUAdjacencyMatrixOrder implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAdjacencyMatrixOrderProps;

  constructor(props: GPUAdjacencyMatrixOrderProps) {
    this.id = props.id ?? 'adjacency-matrix-order';
    this.props = props;
    validatePackedView(props.groups, ['uint32'], `${this.id} groups`);
    validatePackedView(props.output.order, ['uint32'], `${this.id} output.order`);
    if (props.groups.length < 1) {
      throw new Error(`${this.id} groups must contain at least one row`);
    }
    for (const [name, view] of [
      ['output.order', props.output.order],
      ['tieKeys', props.tieKeys]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['uint32'], `${this.id} ${name}`);
      if (view.length !== props.groups.length) {
        throw new Error(`${this.id} ${name} must contain one row per node`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.output.order],
      [props.groups, props.tieKeys]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [props.groups, props.tieKeys, props.output.order]);
    const count = props.groups.length;
    const operation = 'GPUAdjacencyMatrixOrder';
    const transient = (name: string) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', count);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const identity = transient('identity');
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-identity`,
        operation,
        variant: 'identity',
        bindings: [{name: 'identityOut', view: identity, type: 'u32', access: 'read_write'}],
        invocationCount: count,
        body: 'identityOut[identityOutOffset + index] = index;'
      })
    );
    let sortedVertices = identity;
    if (props.tieKeys) {
      const sortedKeys = transient('tie-sorted-keys');
      const byTie = transient('tie-sorted-vertices');
      nodes.push(
        ...new GPUSort({
          id: `${id}-sort-ties`,
          keys: props.tieKeys,
          values: identity,
          outputKeys: sortedKeys,
          outputValues: byTie
        }).getCommandNodes(graph)
      );
      sortedVertices = byTie;
    }
    const groupKeys = transient('group-keys');
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather-groups`,
        operation,
        variant: 'gather-groups',
        bindings: [
          {
            name: 'vertices',
            view: sortedVertices,
            type: 'u32',
            access: 'read'
          },
          {name: 'groups', view: props.groups, type: 'u32', access: 'read'},
          {
            name: 'keysOut',
            view: groupKeys,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: count,
        body: 'keysOut[keysOutOffset + index] = groups[groupsOffset + vertices[verticesOffset + index]];'
      })
    );
    const sortedGroupKeys = transient('group-sorted-keys');
    const byGroup = transient('group-sorted-vertices');
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-groups`,
        keys: groupKeys,
        values: sortedVertices,
        outputKeys: sortedGroupKeys,
        outputValues: byGroup
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation,
        variant: 'scatter',
        bindings: [
          {name: 'vertices', view: byGroup, type: 'u32', access: 'read'},
          {
            name: 'orderOut',
            view: props.output.order,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: count,
        body: 'orderOut[orderOutOffset + vertices[verticesOffset + index]] = index;'
      })
    );
    return nodes;
  }
}
