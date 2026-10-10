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
import {createWGSLKernelNode} from './wgsl-kernel-nodes';

/** One uint32 component of a lexicographic key, listed from most to least significant. */
export type StableLexicographicKey = {
  /** Packed key rows indexed by the original row identifier. */
  view: GraphDataView<'uint32'>;
  /** Significant least-significant bits consumed by the radix sort. */
  keyBits?: number;
};

/** Result of {@link createStableLexicographicIndexSortNodes}. */
export type StableLexicographicIndexSortResult<Parameters> = {
  /** Gather and stable-sort nodes, in execution order. */
  nodes: readonly GPUCommandNode<Parameters>[];
  /** Original row identifiers in lexicographic key order. */
  sortedIndices: GraphDataView<'uint32'>;
  /** Most-significant key component in sorted order. */
  sortedPrimaryKeys: GraphDataView<'uint32'>;
};

/**
 * Stably sorts original row identifiers by a tuple of uint32 key columns.
 *
 * Stable least-to-most passes turn several narrow columns into one lexicographic order without
 * packing them into a topology-specific shader constant or running a serial sort inside each
 * variable-length geometry. Key columns always remain indexed by the original row identifier;
 * the gather before every pass follows the permutation produced by the preceding pass.
 *
 * @internal
 */
export function createStableLexicographicIndexSortNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    /** Identity (or other stable original-order) row identifiers. */
    indices: GraphDataView<'uint32'>;
    /** Tuple components from most to least significant. */
    keys: readonly StableLexicographicKey[];
  }
): StableLexicographicIndexSortResult<Parameters> {
  if (props.keys.length < 1) {
    throw new Error(`${props.id} requires at least one key`);
  }
  const length = props.indices.length;
  for (const [keyIndex, key] of props.keys.entries()) {
    if (key.view.length !== length) {
      throw new Error(`${props.id} key ${keyIndex} length must match indices length`);
    }
  }

  const nodes: GPUCommandNode<Parameters>[] = [];
  let currentIndices = props.indices;
  let sortedPrimaryKeys: GraphDataView<'uint32'> | undefined;
  const leastToMost = props.keys.slice().reverse();
  for (const [passIndex, key] of leastToMost.entries()) {
    const gatheredKeys = createTransientView<'uint32', Parameters>(
      graph,
      `${props.id}-pass-${passIndex}-gathered-keys`,
      'uint32',
      length
    );
    const sortedKeys = createTransientView<'uint32', Parameters>(
      graph,
      `${props.id}-pass-${passIndex}-sorted-keys`,
      'uint32',
      length
    );
    const nextIndices = createTransientView<'uint32', Parameters>(
      graph,
      `${props.id}-pass-${passIndex}-indices`,
      'uint32',
      length
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-pass-${passIndex}-gather`,
        operation: props.operation,
        variant: 'lexicographic-key-gather',
        bindings: [
          {name: 'originalKeys', view: key.view, type: 'u32', access: 'read'},
          {name: 'currentIndices', view: currentIndices, type: 'u32', access: 'read'},
          {name: 'gatheredKeys', view: gatheredKeys, type: 'u32', access: 'read_write'}
        ],
        invocationCount: length,
        body: `let originalIndex = currentIndices[currentIndicesOffset + index];
  gatheredKeys[gatheredKeysOffset + index] = originalKeys[originalKeysOffset + originalIndex];`
      }),
      ...new GPUSort({
        id: `${props.id}-pass-${passIndex}-sort`,
        keys: gatheredKeys,
        values: currentIndices,
        outputKeys: sortedKeys,
        outputValues: nextIndices,
        keyBits: key.keyBits
      }).getCommandNodes(graph)
    );
    currentIndices = nextIndices;
    sortedPrimaryKeys = sortedKeys;
  }

  return {
    nodes,
    sortedIndices: currentIndices,
    sortedPrimaryKeys: sortedPrimaryKeys as GraphDataView<'uint32'>
  };
}
