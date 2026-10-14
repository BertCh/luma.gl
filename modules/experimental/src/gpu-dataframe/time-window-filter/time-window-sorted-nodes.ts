// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {FLOAT_TIME_AT_LEAST_WGSL} from './time-window-classify-node';
import {TIME_WORDS_WGSL} from './time-words';

/** Properties for {@link createSortedTimeWindowRangeNode}. @internal */
export type SortedTimeWindowRangeProps = {
  id: string;
  /** Packed row times: float32 or `uint32x2` Int64 words, non-decreasing in row order. */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** Float32 mode only. */
  timestampsLow?: GraphDataView<'float32'>;
  /** Float window (8) or word window (8). */
  window: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Two words: `[firstRow, rowCount]`. */
  range: GraphDataView<'uint32'>;
  /** One word that receives `rowCount`. */
  total: GraphDataView<'uint32'>;
  /** Optional one word that receives `firstRow`. */
  firstInstance?: GraphDataView<'uint32'>;
};

/**
 * One single-thread kernel with two binary searches over sorted row times: the first row at or
 * after the window start and the first row after the window end.
 *
 * Acceptance uses the helpers of the mask kernel (`isTimeAtLeast`, `isTimeWordsAtLeast`), so the
 * range holds exactly the rows the mask path accepts on sorted input. A reversed window writes an
 * empty range at row 0. The loops run at most `log2(rows) + 1` times each; the row count is a
 * pipeline-overridable size, so graphs that differ only in row count share one module, and not a
 * loop index (rule 12).
 *
 * @internal
 */
export function createSortedTimeWindowRangeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SortedTimeWindowRangeProps
): GPUCommandNode<Parameters> {
  const rows = props.timestamps.length;
  const isWordMode = props.timestamps.format === 'uint32x2';
  const bindings: WGSLKernelBinding[] = [
    {name: 'params', view: props.window, type: isWordMode ? 'u32' : 'f32', access: 'read'}
  ];
  if (rows > 0) {
    bindings.push({
      name: 'rowTimes',
      view: props.timestamps,
      type: isWordMode ? 'u32' : 'f32',
      access: 'read'
    });
    if (props.timestampsLow) {
      bindings.push({name: 'rowTimesLow', view: props.timestampsLow, type: 'f32', access: 'read'});
    }
  }
  bindings.push(
    {name: 'rangeOut', view: props.range, type: 'u32', access: 'read_write'},
    {name: 'totalOut', view: props.total, type: 'u32', access: 'read_write'}
  );
  if (props.firstInstance) {
    bindings.push({
      name: 'firstInstanceOut',
      view: props.firstInstance,
      type: 'u32',
      access: 'read_write'
    });
  }

  const floatPredicates = /* wgsl */ `
  let windowStartHigh = params[paramsOffset + 0u];
  let windowStartLow = params[paramsOffset + 1u];
  let windowEndHigh = params[paramsOffset + 2u];
  let windowEndLow = params[paramsOffset + 3u];
  isWindowValid = isTimeAtLeast(windowEndHigh, windowEndLow, windowStartHigh, windowStartLow);`;
  const wordPredicates = /* wgsl */ `
  let windowStart = vec2<u32>(params[paramsOffset + 0u], params[paramsOffset + 1u]);
  let windowEnd = vec2<u32>(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let windowStartFraction = bitcast<f32>(params[paramsOffset + 4u]);
  let windowEndFraction = bitcast<f32>(params[paramsOffset + 5u]);
  isWindowValid = isTimeWordsAtLeast(windowEnd, windowEndFraction, windowStart, windowStartFraction);`;

  const declarations = isWordMode
    ? /* wgsl */ `${TIME_WORDS_WGSL}
override ROW_COUNT: u32;

fn isRowBeforeStart(row: u32, windowStart: vec2<u32>, windowStartFraction: f32) -> bool {
  let time = vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
  return !isTimeWordsAtLeast(time, 0.0, windowStart, windowStartFraction);
}

fn isRowAfterEnd(row: u32, windowEnd: vec2<u32>, windowEndFraction: f32) -> bool {
  let time = vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
  return !isTimeWordsAtLeast(windowEnd, windowEndFraction, time, 0.0);
}`
    : /* wgsl */ `${FLOAT_TIME_AT_LEAST_WGSL}
override ROW_COUNT: u32;

fn getRowLow(row: u32) -> f32 {
  return ${props.timestampsLow ? 'rowTimesLow[rowTimesLowOffset + row]' : '0.0'};
}

fn isRowBeforeStart(row: u32, windowStartHigh: f32, windowStartLow: f32) -> bool {
  return !isTimeAtLeast(rowTimes[rowTimesOffset + row], getRowLow(row), windowStartHigh, windowStartLow);
}

fn isRowAfterEnd(row: u32, windowEndHigh: f32, windowEndLow: f32) -> bool {
  return !isTimeAtLeast(windowEndHigh, windowEndLow, rowTimes[rowTimesOffset + row], getRowLow(row));
}`;

  const beforeArguments = isWordMode
    ? 'windowStart, windowStartFraction'
    : 'windowStartHigh, windowStartLow';
  const afterArguments = isWordMode
    ? 'windowEnd, windowEndFraction'
    : 'windowEndHigh, windowEndLow';

  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTimeWindowFilter',
    variant: isWordMode ? 'sorted-range-words' : 'sorted-range',
    bindings,
    invocationCount: 1,
    workgroupSize: 32,
    // With no rows there is no time binding for the helpers to read.
    declarations: rows === 0 ? undefined : declarations,
    constants: rows === 0 ? undefined : {ROW_COUNT: rows},
    body:
      rows === 0
        ? `rangeOut[rangeOutOffset] = 0u;
  rangeOut[rangeOutOffset + 1u] = 0u;
  totalOut[totalOutOffset] = 0u;
  ${props.firstInstance ? 'firstInstanceOut[firstInstanceOutOffset] = 0u;' : ''}`
        : /* wgsl */ `
  var isWindowValid = false;
  ${isWordMode ? wordPredicates : floatPredicates}
  var firstRow = 0u;
  var endRow = 0u;
  if (isWindowValid) {
    // Lower bound: the first row that is not before the window start.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (isRowBeforeStart(middle, ${beforeArguments})) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    firstRow = low;
    // Upper bound: the first row after the window end, searched from the lower bound.
    high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (isRowAfterEnd(middle, ${afterArguments})) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    endRow = low;
  }
  let rowCount = endRow - firstRow;
  rangeOut[rangeOutOffset] = firstRow;
  rangeOut[rangeOutOffset + 1u] = rowCount;
  totalOut[totalOutOffset] = rowCount;
  ${props.firstInstance ? 'firstInstanceOut[firstInstanceOutOffset] = firstRow;' : ''}`
  });
}

/** Properties for {@link createSortedTimeWindowIdsNode}. @internal */
export type SortedTimeWindowIdsProps = {
  id: string;
  range: GraphDataView<'uint32'>;
  /** Caller IDs; the capacity is their length (at least 1). */
  ids: GraphDataView<'uint32'>;
  sourceIds?: GraphDataView<'uint32'>;
};

/**
 * Writes the first `min(rowCount, capacity)` IDs of the sorted range: `sourceIds[firstRow + i]`, or
 * the row index. One invocation per capacity slot, returning early past the count, so a caller who
 * only needs the range pays for the slots of a small `ids` view.
 *
 * @internal
 */
export function createSortedTimeWindowIdsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SortedTimeWindowIdsProps
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'range', view: props.range, type: 'u32', access: 'read'},
    {name: 'idsOut', view: props.ids, type: 'u32', access: 'read_write'}
  ];
  if (props.sourceIds) {
    bindings.push({name: 'sourceIds', view: props.sourceIds, type: 'u32', access: 'read'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTimeWindowFilter',
    variant: 'sorted-ids',
    bindings,
    invocationCount: props.ids.length,
    body: `let firstRow = range[rangeOffset];
  // The invocation count is the capacity.
  let count = min(range[rangeOffset + 1u], INVOCATION_COUNT);
  if (index >= count) {
    return;
  }
  idsOut[idsOutOffset + index] = ${props.sourceIds ? 'sourceIds[sourceIdsOffset + firstRow + index]' : 'firstRow + index'};`
  });
}

/** Properties for {@link createSortedTimeWindowOrderCheckNode}. @internal */
export type SortedTimeWindowOrderCheckProps = {
  id: string;
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  timestampsLow?: GraphDataView<'float32'>;
  /** One word. Must be zeroed earlier in the same encoding. */
  violations: GraphDataView<'uint32'>;
};

/**
 * Counts adjacent row pairs whose times decrease, or are not comparable (NaN), into `violations`
 * with one atomic add per offending pair, using the same comparison helpers as acceptance.
 *
 * @internal
 */
export function createSortedTimeWindowOrderCheckNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SortedTimeWindowOrderCheckProps
): GPUCommandNode<Parameters> {
  const isWordMode = props.timestamps.format === 'uint32x2';
  const bindings: WGSLKernelBinding[] = [
    {name: 'rowTimes', view: props.timestamps, type: isWordMode ? 'u32' : 'f32', access: 'read'}
  ];
  if (props.timestampsLow) {
    bindings.push({name: 'rowTimesLow', view: props.timestampsLow, type: 'f32', access: 'read'});
  }
  bindings.push({
    name: 'violations',
    view: props.violations,
    type: 'atomic<u32>',
    access: 'read_write'
  });
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTimeWindowFilter',
    variant: isWordMode ? 'sorted-order-check-words' : 'sorted-order-check',
    bindings,
    // One invocation per adjacent pair.
    invocationCount: Math.max(props.timestamps.length - 1, 1),
    declarations: isWordMode ? TIME_WORDS_WGSL : FLOAT_TIME_AT_LEAST_WGSL,
    body: isWordMode
      ? `if (index + 1u >= ${props.timestamps.length}u) {
    return;
  }
  let previous = vec2<u32>(rowTimes[rowTimesOffset + 2u * index], rowTimes[rowTimesOffset + 2u * index + 1u]);
  let next = vec2<u32>(rowTimes[rowTimesOffset + 2u * index + 2u], rowTimes[rowTimesOffset + 2u * index + 3u]);
  if (!isTimeWordsAtLeast(next, 0.0, previous, 0.0)) {
    atomicAdd(&violations[violationsOffset], 1u);
  }`
      : `if (index + 1u >= ${props.timestamps.length}u) {
    return;
  }
  ${
    props.timestampsLow
      ? `let previousLow = rowTimesLow[rowTimesLowOffset + index];
  let nextLow = rowTimesLow[rowTimesLowOffset + index + 1u];`
      : 'let previousLow = 0.0;\n  let nextLow = 0.0;'
  }
  if (!isTimeAtLeast(rowTimes[rowTimesOffset + index + 1u], nextLow, rowTimes[rowTimesOffset + index], previousLow)) {
    atomicAdd(&violations[violationsOffset], 1u);
  }`
  });
}
