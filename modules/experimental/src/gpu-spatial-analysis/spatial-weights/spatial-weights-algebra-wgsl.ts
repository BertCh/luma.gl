// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelElementType
} from '../../utils/wgsl-kernel-nodes';

/** Slot marker for an operand that has no entry for the neighbor ID being merged. @internal */
export const NO_SLOT_WGSL = '0xffffffffu';

/**
 * One output array written by an emit pass of {@link createCSRStages}.
 *
 * @internal
 */
export type CSRStageAttribute = {
  /** Short name used in node IDs. */
  name: string;
  /** Destination array, aligned with the target neighbor slots. */
  output: GraphDataView;
  /** WGSL element type of `output`. */
  type: WGSLKernelElementType;
  /** WGSL expression of the value written for each emitted item, evaluated in the visit scope. */
  value: string;
  /** Bindings this attribute reads in addition to the stage's shared bindings. */
  bindings?: readonly WGSLKernelBinding[];
  /** Module-scope WGSL this attribute needs, for example helper functions over `bindings`. */
  declarations?: string;
};

/**
 * Properties for {@link createCSRStages}.
 *
 * @internal
 */
export type CSRStageProps = {
  /** Prefix for node IDs. */
  id: string;
  /** Operation name reported in workload estimates. */
  operation: string;
  /** Output row count. */
  rows: number;
  /** Target row offsets, `rows + 1` entries. Receives offsets clamped to `capacity`. */
  targetOffsets: GraphDataView<'uint32'>;
  /** Slot capacity of the target. */
  capacity: number;
  /** One-element flag: set to 1 when the result needs more than `capacity` slots. */
  overflow: GraphDataView<'uint32'>;
  /** True when this stage initializes `overflow` (writes 0 or 1); later stages only raise it. */
  initializesOverflow: boolean;
  /** Optional one-element view receiving the unclamped slot count this stage needs. */
  totalNeighbors?: GraphDataView<'uint32'>;
  /** Bindings every pass reads (names must avoid `counts`, `starts`, `outOffsets`, `outValues`). */
  bindings: readonly WGSLKernelBinding[];
  /** Module-scope WGSL (helper functions over `bindings`) shared by the count and emit passes. */
  declarations?: string;
  /**
   * Returns the WGSL statements of one row (`index`) that visit every output item in ascending
   * neighbor order, inserting `onItem` once per item. `onItem` is a statement block: it must not
   * be placed where `continue` or `break` would skip it, and may be inserted more than once.
   */
  visit: (onItem: string) => string;
  /** Arrays to write. */
  attributes: readonly CSRStageAttribute[];
  /** Sort each emitted row ascending by the first attribute (which must be the neighbor IDs). */
  sortRows?: boolean;
};

/**
 * Builds a two-pass CSR producer: per-row counts, an exclusive scan, clamped offsets with an
 * overflow flag, then one emit pass per attribute (each re-runs `visit`, so no pass binds more
 * than eight storage buffers). Rows are emitted in the order `visit` produces; truncation at
 * `capacity` keeps the first slots of the row in which the capacity runs out and drops later rows.
 *
 * @internal
 */
export function createCSRStages<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CSRStageProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, rows, capacity} = props;
  const counts = createTransientView(graph, `${id}-counts`, 'uint32', rows);
  const starts = createTransientView(graph, `${id}-starts`, 'uint32', rows);
  const constants = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;
const NO_SLOT: u32 = ${NO_SLOT_WGSL};`;
  // Shared declarations may read the shared bindings, which the offsets pass does not bind.
  const declarations = `${constants}\n${props.declarations ?? ''}`;
  const nodes: GPUCommandNode<Parameters>[] = [];
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-counts`,
      operation,
      variant: 'counts',
      bindings: [
        ...props.bindings,
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations,
      body: `var count = 0u;
  ${props.visit('count++;')}
  counts[countsOffset + index] = count;`
    })
  );
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan`,
      input: counts,
      output: starts,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation,
      variant: 'offsets',
      bindings: [
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'starts', view: starts, type: 'u32', access: 'read'},
        {name: 'outOffsets', view: props.targetOffsets, type: 'u32', access: 'read_write'},
        {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
        ...(props.totalNeighbors
          ? [
              {
                name: 'totalNeighbors',
                view: props.totalNeighbors,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: rows + 1,
      declarations: constants,
      body: `if (index < ROWS) {
    outOffsets[outOffsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = starts[startsOffset + ROWS - 1u] + counts[countsOffset + ROWS - 1u];
    outOffsets[outOffsetsOffset + ROWS] = min(total, CAPACITY);
    ${
      props.initializesOverflow
        ? 'overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);'
        : 'if (total > CAPACITY) {\n      overflow[overflowOffset] = 1u;\n    }'
    }
    ${props.totalNeighbors ? 'totalNeighbors[totalNeighborsOffset] = total;' : ''}
  }`
    })
  );
  for (const attribute of props.attributes) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit-${attribute.name}`,
        operation,
        variant: `emit-${attribute.name}`,
        bindings: [
          ...props.bindings,
          ...(attribute.bindings ?? []),
          {name: 'outOffsets', view: props.targetOffsets, type: 'u32', access: 'read'},
          {name: 'outValues', view: attribute.output, type: attribute.type, access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${declarations}\n${attribute.declarations ?? ''}`,
        body: `var next = outOffsets[outOffsetsOffset + index];
  let end = outOffsets[outOffsetsOffset + index + 1u];
  ${props.visit(`if (next < end) {
    outValues[outValuesOffset + next] = ${attribute.value};
    next++;
  }`)}`
      })
    );
  }
  if (props.sortRows) {
    const neighbors = props.attributes[0];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-rows`,
        operation,
        variant: 'sort-rows',
        bindings: [
          {name: 'outOffsets', view: props.targetOffsets, type: 'u32', access: 'read'},
          {name: 'outValues', view: neighbors.output, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: `let begin = outOffsets[outOffsetsOffset + index];
  let end = outOffsets[outOffsetsOffset + index + 1u];
  for (var slot = begin + 1u; slot < end; slot++) {
    let key = outValues[outValuesOffset + slot];
    var position = slot;
    while (position > begin && outValues[outValuesOffset + position - 1u] > key) {
      outValues[outValuesOffset + position] = outValues[outValuesOffset + position - 1u];
      position--;
    }
    outValues[outValuesOffset + position] = key;
  }`
      })
    );
  }
  return nodes;
}

/** WGSL binary search over `array[begin, end)` for `value`, returning whether it is present. */
export function getContainsFunctionWGSL(name: string, array: string): string {
  return `fn ${name}(begin: u32, end: u32, value: u32) -> bool {
  var low = begin;
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (${array}[${array}Offset + middle] < value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low < end && ${array}[${array}Offset + low] == value;
}`;
}

/** Set operation of a two-operand merge. @internal */
export type SpatialWeightsMergeOperation =
  | 'union'
  | 'intersection'
  | 'difference'
  | 'symmetricDifference';

/**
 * WGSL for the per-row two-pointer merge of `leftOffsets/leftNeighbors` and
 * `rightOffsets/rightNeighbors`. For every neighbor ID the operation keeps, inserts `onItem` with
 * `id` (the neighbor), `leftSlot` and `rightSlot` (slot in the operand, or `NO_SLOT`) in scope.
 *
 * @internal
 */
export function getMergeVisitWGSL(operation: SpatialWeightsMergeOperation, onItem: string): string {
  const keep = {
    union: 'true',
    intersection: 'leftSlot != NO_SLOT && rightSlot != NO_SLOT',
    difference: 'leftSlot != NO_SLOT && rightSlot == NO_SLOT',
    symmetricDifference: '(leftSlot == NO_SLOT) != (rightSlot == NO_SLOT)'
  }[operation];
  return `var leftNext = leftOffsets[leftOffsetsOffset + index];
  let leftEnd = leftOffsets[leftOffsetsOffset + index + 1u];
  var rightNext = rightOffsets[rightOffsetsOffset + index];
  let rightEnd = rightOffsets[rightOffsetsOffset + index + 1u];
  loop {
    let hasLeft = leftNext < leftEnd;
    let hasRight = rightNext < rightEnd;
    if (!hasLeft && !hasRight) {
      break;
    }
    let leftId = select(NO_SLOT, leftNeighbors[leftNeighborsOffset + leftNext], hasLeft);
    let rightId = select(NO_SLOT, rightNeighbors[rightNeighborsOffset + rightNext], hasRight);
    let id = min(leftId, rightId);
    var leftSlot = NO_SLOT;
    var rightSlot = NO_SLOT;
    if (leftId == id) {
      leftSlot = leftNext;
      leftNext++;
    }
    if (rightId == id) {
      rightSlot = rightNext;
      rightNext++;
    }
    if (${keep}) {
      ${onItem}
    }
  }`;
}
