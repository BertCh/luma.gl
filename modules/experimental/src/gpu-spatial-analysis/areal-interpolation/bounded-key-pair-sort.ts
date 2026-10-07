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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import type {KeyPairSort} from '../spatial-weights/key-pair-grouping';

/**
 * {@link getKeyPairSortNodes} with the radix passes limited to the bits the keys actually use.
 *
 * Radix sort cost is proportional to the key width, and the generic pair sort always sorts two
 * full 32-bit halves (16 passes of 4 bits). Zone ids, dense point ids and similar keys need only
 * `ceil(log2(limit))` bits, so a 1000 x 1000 zone pair sorts in 3 + 3 passes instead of 8 + 8.
 *
 * Contract: every valid half is strictly below its limit (`lowKeyLimit`, `highKeyLimit`, both
 * below `2^32 - 1`), and invalid items hold `0xffffffff` in both halves. The invalid key is
 * truncated to all ones in the sorted bits, which is above every valid key, so invalid items still
 * sort last (stably, in item order). The result is identical to the full-width pair sort.
 *
 * @internal
 */
export function getBoundedKeyPairSortNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  count: number,
  keyHigh: GraphDataView<'uint32'>,
  keyLow: GraphDataView<'uint32'>,
  limits: {lowKeyLimit: number; highKeyLimit: number}
): KeyPairSort<Parameters> {
  const lowBits = getSortKeyBits(limits.lowKeyLimit);
  const highBits = getSortKeyBits(limits.highKeyLimit);
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
      outputValues: lowOrder,
      keyBits: lowBits
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
      outputValues: sortedItems,
      keyBits: highBits
    }).getCommandNodes(graph)
  ];
  return {nodes, sortedItems};
}
