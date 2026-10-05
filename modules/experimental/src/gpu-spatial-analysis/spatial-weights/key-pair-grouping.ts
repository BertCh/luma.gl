// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';

/** Key value that marks an invalid item. Both halves equal it, and invalid items sort last. @internal */
export const KEY_PAIR_INVALID = 0xffffffff;

/** Result of {@link getKeyPairSortNodes}. @internal */
export type KeyPairSort<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /** Item IDs ordered by ascending `(keyHigh, keyLow)`; ties keep ascending item order. */
  sortedItems: GraphDataView<'uint32'>;
};

/**
 * Sorts items `0..count-1` by the 64-bit key `(keyHigh, keyLow)` with two stable 32-bit radix sorts
 * (low half first, then high half). Items whose key is `(0xffffffff, 0xffffffff)` sort last.
 *
 * @internal
 */
export function getKeyPairSortNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  count: number,
  keyHigh: GraphDataView<'uint32'>,
  keyLow: GraphDataView<'uint32'>
): KeyPairSort<Parameters> {
  const itemIds = createTransientView(graph, `${id}-item-ids`, 'uint32', count);
  const sortedLow = createTransientView(graph, `${id}-sorted-low`, 'uint32', count);
  const lowOrder = createTransientView(graph, `${id}-low-order`, 'uint32', count);
  const gatheredHigh = createTransientView(graph, `${id}-gathered-high`, 'uint32', count);
  const sortedHigh = createTransientView(graph, `${id}-sorted-high`, 'uint32', count);
  const sortedItems = createTransientView(graph, `${id}-sorted-items`, 'uint32', count);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-iota`,
      operation,
      variant: 'iota',
      bindings: [{name: 'itemIds', view: itemIds, type: 'u32', access: 'read_write'}],
      invocationCount: count,
      body: 'itemIds[itemIdsOffset + index] = index;'
    }),
    ...new GPUSort({
      id: `${id}-sort-low`,
      keys: keyLow,
      values: itemIds,
      outputKeys: sortedLow,
      outputValues: lowOrder
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-gather-high`,
      operation,
      variant: 'gather-high',
      bindings: [
        {name: 'keyHigh', view: keyHigh, type: 'u32', access: 'read'},
        {name: 'lowOrder', view: lowOrder, type: 'u32', access: 'read'},
        {name: 'gatheredHigh', view: gatheredHigh, type: 'u32', access: 'read_write'}
      ],
      invocationCount: count,
      body: 'gatheredHigh[gatheredHighOffset + index] = keyHigh[keyHighOffset + lowOrder[lowOrderOffset + index]];'
    }),
    ...new GPUSort({
      id: `${id}-sort-high`,
      keys: gatheredHigh,
      values: lowOrder,
      outputKeys: sortedHigh,
      outputValues: sortedItems
    }).getCommandNodes(graph)
  ];
  return {nodes, sortedItems};
}

/** Result of {@link getKeyGroupNodes}. @internal */
export type KeyGroups<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /** Per sorted position, one plus the dense index of its run of equal keys. */
  groupIndex: GraphDataView<'uint32'>;
  /** Run `g` spans sorted positions `[groupStarts[g], groupStarts[g + 1])`. */
  groupStarts: GraphDataView<'uint32'>;
};

/**
 * Finds the runs of equal keys in a {@link getKeyPairSortNodes} result. Each invalid item forms its
 * own run, so no run mixes valid and invalid items.
 *
 * @internal
 */
export function getKeyGroupNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  count: number,
  keyHigh: GraphDataView<'uint32'>,
  keyLow: GraphDataView<'uint32'>,
  sortedItems: GraphDataView<'uint32'>
): KeyGroups<Parameters> {
  const flags = createTransientView(graph, `${id}-run-flags`, 'uint32', count);
  const groupIndex = createTransientView(graph, `${id}-group-index`, 'uint32', count);
  const groupStarts = createTransientView(graph, `${id}-group-starts`, 'uint32', count + 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-run-flags`,
      operation,
      variant: 'run-flags',
      bindings: [
        {name: 'sortedItems', view: sortedItems, type: 'u32', access: 'read'},
        {name: 'keyHigh', view: keyHigh, type: 'u32', access: 'read'},
        {name: 'keyLow', view: keyLow, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: count,
      body: `let item = sortedItems[sortedItemsOffset + index];
  let high = keyHigh[keyHighOffset + item];
  let low = keyLow[keyLowOffset + item];
  var start = index == 0u || (high == ${KEY_PAIR_INVALID}u && low == ${KEY_PAIR_INVALID}u);
  if (!start) {
    let previous = sortedItems[sortedItemsOffset + index - 1u];
    start = high != keyHigh[keyHighOffset + previous] || low != keyLow[keyLowOffset + previous];
  }
  flags[flagsOffset + index] = select(0u, 1u, start);`
    }),
    ...new GPUScan({
      id: `${id}-group-scan`,
      input: flags,
      output: groupIndex,
      mode: 'inclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-group-starts`,
      operation,
      variant: 'group-starts',
      bindings: [
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'groupIndex', view: groupIndex, type: 'u32', access: 'read'},
        {name: 'groupStarts', view: groupStarts, type: 'u32', access: 'read_write'}
      ],
      invocationCount: count,
      body: `if (flags[flagsOffset + index] != 0u) {
    groupStarts[groupStartsOffset + groupIndex[groupIndexOffset + index] - 1u] = index;
  }
  if (index + 1u == INVOCATION_COUNT) {
    groupStarts[groupStartsOffset + groupIndex[groupIndexOffset + index]] = INVOCATION_COUNT;
  }`
    })
  ];
  return {nodes, groupIndex, groupStarts};
}
