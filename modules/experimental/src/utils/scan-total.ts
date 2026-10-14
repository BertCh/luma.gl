// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';

/**
 * Value of a saturated scan total: the true sum of the counts reached `2^32`. It is above every
 * capacity (a uint32 capacity is at most `2^32 - 2` once a 128 MiB binding holds its slots), so a
 * `total > capacity` check reports overflow and a `requiredCount` copied from it saturates.
 *
 * @internal
 */
export const SATURATED_SCAN_TOTAL = 0xffffffff;

/** Properties for {@link createScanCarryNode} and {@link createSaturatedScanTotalNodes}. @internal */
export type ScanTotalProps = {
  /** Node ID prefix. */
  id: string;
  /** Operation name reported in the workload estimate. */
  operation: string;
  /** Per-row uint32 counts that the exclusive scan read. */
  counts: GraphDataView<'uint32'>;
  /** Exclusive prefix sums of `counts` written by `GPUScan` (wrapping modulo `2^32`). */
  offsets: GraphDataView<'uint32'>;
  /** Number of scanned rows, at most the length of both views. */
  rowCount: number;
  /** uint32 view holding the total word. */
  total: GraphDataView<'uint32'>;
  /** Row of `total` holding the total. Defaults to 0. */
  totalRow?: number;
};

/**
 * Saturates the total of an exclusive uint32 count scan whose true sum reached `2^32` (S-8).
 *
 * `GPUScan` wraps modulo `2^32`, so when the counts sum to `2^32` or more the usual total
 * `offsets[last] + counts[last]` is a small number, a `total > capacity` check passes and later
 * rows write over earlier slots with no overflow flag. A count is below `2^32`, so the true prefix
 * crosses a multiple of `2^32` inside row `i` exactly when `offsets[i] + counts[i] < offsets[i]` in
 * uint32 arithmetic, and the true sum reaches `2^32` exactly when some row carries. This node checks
 * every row and stores {@link SATURATED_SCAN_TOTAL} into `total[totalRow]` when one does.
 *
 * Run it after the node that writes the wrapped total and before every node that reads it.
 *
 * @internal
 */
export function createScanCarryNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ScanTotalProps
): GPUCommandNode<Parameters> {
  // Callers scan `rowCount` rows of both views and own the total row.
  assert(
    props.rowCount <= props.counts.length &&
      props.rowCount <= props.offsets.length &&
      (props.totalRow ?? 0) < props.total.length
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-carry`,
    operation: props.operation,
    variant: 'scan-carry',
    bindings: [
      {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'total', view: props.total, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.rowCount,
    declarations: 'override TOTAL_ROW: u32;',
    constants: {TOTAL_ROW: props.totalRow ?? 0},
    body: `let offset = offsets[offsetsOffset + index];
  if (offset + counts[countsOffset + index] < offset) {
    atomicStore(&total[totalOffset + TOTAL_ROW], ${SATURATED_SCAN_TOTAL}u);
  }`
  });
}

/**
 * Writes the total of an exclusive uint32 count scan into `total[totalRow]`, saturated at
 * {@link SATURATED_SCAN_TOTAL} when the true sum of the counts reaches `2^32` (S-8): one node
 * writes `offsets[last] + counts[last]` (0 for no rows), then {@link createScanCarryNode} saturates
 * it. Run them after the scan and before every node that reads the total.
 *
 * @internal
 */
export function createSaturatedScanTotalNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ScanTotalProps
): GPUCommandNode<Parameters>[] {
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-total`,
      operation: props.operation,
      variant: 'scan-total',
      bindings: [
        {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {name: 'total', view: props.total, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: 'override ROW_COUNT: u32;\noverride TOTAL_ROW: u32;',
      constants: {ROW_COUNT: props.rowCount, TOTAL_ROW: props.totalRow ?? 0},
      body: `var sum = 0u;
  if (ROW_COUNT > 0u) {
    sum = offsets[offsetsOffset + ROW_COUNT - 1u] + counts[countsOffset + ROW_COUNT - 1u];
  }
  total[totalOffset + TOTAL_ROW] = sum;`
    }),
    createScanCarryNode(graph, props)
  ];
}
