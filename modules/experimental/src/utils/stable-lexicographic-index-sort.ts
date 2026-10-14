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

/** Most key columns packed into one radix key: the gather also binds indices and output. */
const MAXIMUM_PACKED_KEY_COUNT = 6;

/**
 * Stably sorts original row identifiers by a tuple of uint32 key columns.
 *
 * Stable least-to-most passes turn several narrow columns into one lexicographic order without
 * packing them into a topology-specific shader constant or running a serial sort inside each
 * variable-length geometry. Key columns always remain indexed by the original row identifier;
 * the gather before every pass follows the permutation produced by the preceding pass.
 *
 * Adjacent keys whose `keyBits` sum to at most 32 are packed into one radix key by their gather,
 * so for example a `(slot, 1-bit phase)` pair costs one stable sort instead of two. Packed
 * components contribute only their `keyBits` least-significant bits, as they would to the sort.
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
  const columns = getPackedKeyColumns(props.keys);
  for (const [passIndex, column] of columns.entries()) {
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
    const packed = column.length > 1;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-pass-${passIndex}-gather`,
        operation: props.operation,
        variant: packed ? 'lexicographic-key-pack' : 'lexicographic-key-gather',
        bindings: [
          ...column.map((key, keyIndex) => ({
            name: `originalKeys${keyIndex}`,
            view: key.view,
            type: 'u32' as const,
            access: 'read' as const
          })),
          {name: 'currentIndices', view: currentIndices, type: 'u32', access: 'read'},
          {name: 'gatheredKeys', view: gatheredKeys, type: 'u32', access: 'read_write'}
        ],
        invocationCount: length,
        body: `let originalIndex = currentIndices[currentIndicesOffset + index];
  ${getPackedKeySource(column)}`
      }),
      ...new GPUSort({
        id: `${props.id}-pass-${passIndex}-sort`,
        keys: gatheredKeys,
        values: currentIndices,
        outputKeys: sortedKeys,
        outputValues: nextIndices,
        keyBits: packed ? getColumnKeyBits(column) : column[0].keyBits
      }).getCommandNodes(graph)
    );
    currentIndices = nextIndices;
    sortedPrimaryKeys = packed ? undefined : sortedKeys;
  }

  if (!sortedPrimaryKeys) {
    // The most significant pass sorted a packed key; gather the primary column on its own.
    sortedPrimaryKeys = createTransientView<'uint32', Parameters>(
      graph,
      `${props.id}-sorted-primary-keys`,
      'uint32',
      length
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-primary-gather`,
        operation: props.operation,
        variant: 'lexicographic-key-gather',
        bindings: [
          {name: 'originalKeys0', view: props.keys[0].view, type: 'u32', access: 'read'},
          {name: 'currentIndices', view: currentIndices, type: 'u32', access: 'read'},
          {name: 'gatheredKeys', view: sortedPrimaryKeys, type: 'u32', access: 'read_write'}
        ],
        invocationCount: length,
        body: `let originalIndex = currentIndices[currentIndicesOffset + index];
  ${getPackedKeySource([props.keys[0]])}`
      })
    );
  }

  return {
    nodes,
    sortedIndices: currentIndices,
    sortedPrimaryKeys
  };
}

/** Groups keys into sort columns from least to most significant; each column lists keys most-significant first. */
function getPackedKeyColumns(keys: readonly StableLexicographicKey[]): StableLexicographicKey[][] {
  const columns: StableLexicographicKey[][] = [];
  for (const key of keys.slice().reverse()) {
    const column = columns[columns.length - 1];
    if (
      column &&
      column.length < MAXIMUM_PACKED_KEY_COUNT &&
      isPackableKey(key) &&
      column.every(isPackableKey) &&
      getColumnKeyBits(column) + key.keyBits! <= 32
    ) {
      column.unshift(key);
    } else {
      columns.push([key]);
    }
  }
  return columns;
}

function isPackableKey(key: StableLexicographicKey): boolean {
  return Number.isInteger(key.keyBits) && key.keyBits! >= 1 && key.keyBits! < 32;
}

function getColumnKeyBits(column: readonly StableLexicographicKey[]): number {
  return column.reduce((bits, key) => bits + (key.keyBits ?? 32), 0);
}

/** WGSL that writes `gatheredKeys[index]` from one key, or from several packed keys. */
function getPackedKeySource(column: readonly StableLexicographicKey[]): string {
  if (column.length === 1) {
    return 'gatheredKeys[gatheredKeysOffset + index] = originalKeys0[originalKeys0Offset + originalIndex];';
  }
  const terms = column.map((key, keyIndex) => {
    const mask = 2 ** key.keyBits! - 1;
    return `packed = (packed << ${key.keyBits}u) | (originalKeys${keyIndex}[originalKeys${keyIndex}Offset + originalIndex] & ${mask}u);`;
  });
  return `var packed = 0u;
  ${terms.join('\n  ')}
  gatheredKeys[gatheredKeysOffset + index] = packed;`;
}
