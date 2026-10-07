// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';

/**
 * Creates one node that writes CSR segment offsets from keys that are already sorted ascending.
 *
 * `segmentOffsets[s]` is the lower bound of `s` in `sortedKeys` (the first position whose key is at
 * least `s`), found by one binary search per segment, so `segmentOffsets[segmentCount]` is the
 * number of rows with a key below `segmentCount`. Rows that must be excluded carry the key
 * `segmentCount` and sort last. This replaces the usual atomic count / zero fill / exclusive scan
 * chain: it needs no atomics (a per-row `atomicAdd` into one counter per segment serializes when a
 * few segments own most rows) and no scan passes. Work is `O(segments * log rows)` reads, depth is
 * one dispatch, and the result is deterministic.
 *
 * @internal
 */
export function createSortedSegmentOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    /** Full node ID. */
    id: string;
    operation: string;
    /** Workload variant name; defaults to `segment-offsets`. */
    variant?: string;
    segmentCount: number;
    /** Segment key per row in ascending order. */
    sortedKeys: GraphDataView<'uint32'>;
    /** Receives `segmentCount + 1` offsets. */
    segmentOffsets: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {segmentCount, sortedKeys, segmentOffsets} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: props.variant ?? 'segment-offsets',
    bindings: [
      {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
      {name: 'segmentOffsets', view: segmentOffsets, type: 'u32', access: 'read_write'}
    ],
    invocationCount: segmentCount + 1,
    body: `var low = 0u;
  var high = ${sortedKeys.length}u;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (sortedKeys[sortedKeysOffset + middle] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  segmentOffsets[segmentOffsetsOffset + index] = low;`
  });
}

/**
 * Builds CSR segment offsets (and optionally per-segment counts) from sorted keys, see
 * {@link createSortedSegmentOffsetsNode}. Node IDs are `${id}-segment-offsets` and
 * `${id}-group-counts`.
 *
 * @internal
 */
export function getSortedSegmentOffsetNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    groupCount: number;
    /** Group key per row in ascending order, `groupCount` for excluded rows. */
    sortedGroupKeys: GraphDataView<'uint32'>;
    /** Receives `groupCount + 1` offsets. */
    segmentOffsets: GraphDataView<'uint32'>;
    /** Optional `groupCount` rows receiving `offsets[g + 1] - offsets[g]`. */
    counts?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, groupCount, sortedGroupKeys, segmentOffsets, counts} = props;
  const nodes: GPUCommandNode<Parameters>[] = [
    createSortedSegmentOffsetsNode<Parameters>(graph, {
      id: `${id}-segment-offsets`,
      operation,
      segmentCount: groupCount,
      sortedKeys: sortedGroupKeys,
      segmentOffsets
    })
  ];
  if (counts) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-group-counts`,
        operation,
        variant: 'group-counts',
        bindings: [
          {name: 'segmentOffsets', view: segmentOffsets, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: groupCount,
        body: `counts[countsOffset + index] =
    segmentOffsets[segmentOffsetsOffset + index + 1u] - segmentOffsets[segmentOffsetsOffset + index];`
      })
    );
  }
  return nodes;
}
