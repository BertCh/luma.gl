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
import {createSortedSegmentOffsetsNode} from '../../utils/sorted-segment-offsets';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';

/** Result of {@link getSlotGroupingNodes}. @internal */
export type SlotGrouping<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /** Slot keys in ascending order (the unused key `columns` last). */
  sortedKeys: GraphDataView<'uint32'>;
  /** Slot IDs in the order of `sortedKeys`; slots with equal keys keep ascending slot order. */
  sortedSlots: GraphDataView<'uint32'>;
};

/**
 * Groups CSR slots by column (a transpose without the gather): a stable radix sort of the slot IDs
 * by key, then column `c` spans sorted positions `[columnOffsets[c], columnOffsets[c + 1])`.
 *
 * The offsets come from one binary search per column over the already sorted keys
 * ({@link createSortedSegmentOffsetsNode}), so there is no atomic counting pass, no clear and no scan: the
 * sort has already done the counting. Deterministic and independent of atomics ordering.
 *
 * @param props.slotKeys One key per slot: the column in `[0, columns)`, or `columns` for slots
 * that must not be grouped (they sort last).
 * @param props.slotIndices `slotIndices[s] = s`, written by the caller's key kernel.
 * @param props.columnOffsets Caller-owned `columns + 1` offsets; the last entry is the number of
 * grouped slots.
 *
 * @internal
 */
export function getSlotGroupingNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    slotKeys: GraphDataView<'uint32'>;
    slotIndices: GraphDataView<'uint32'>;
    columns: number;
    columnOffsets: GraphDataView<'uint32'>;
  }
): SlotGrouping<Parameters> {
  const {id, operation, slotKeys, slotIndices, columns, columnOffsets} = props;
  const capacity = slotKeys.length;
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', capacity);
  const sortedSlots = createTransientView(graph, `${id}-sorted-slots`, 'uint32', capacity);
  const nodes: GPUCommandNode<Parameters>[] = [
    ...new GPUSort({
      id: `${id}-sort`,
      keys: slotKeys,
      values: slotIndices,
      outputKeys: sortedKeys,
      outputValues: sortedSlots,
      keyBits: getSortKeyBits(columns)
    }).getCommandNodes(graph),
    createSortedSegmentOffsetsNode<Parameters>(graph, {
      id: `${id}-column-offsets`,
      operation,
      variant: 'column-offsets',
      segmentCount: columns,
      sortedKeys,
      segmentOffsets: columnOffsets
    })
  ];
  return {nodes, sortedKeys, sortedSlots};
}
