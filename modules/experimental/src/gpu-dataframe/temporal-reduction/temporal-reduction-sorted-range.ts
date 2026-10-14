// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import {TIME_WORDS_WGSL} from '../time-window-filter/time-words';

/** Properties for {@link createTemporalReductionRangeNode}. @internal */
export type TemporalReductionRangeProps = {
  id: string;
  /** Sorted packed row times: float32 or `uint32x2` Int64 words. At least one row. */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** Bucketing parameters in the layout of the time format. */
  parameters: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Scratch whose first two words receive `[firstRow, rowCount]`. */
  rowSlots: GraphDataView<'uint32'>;
  bucketCount: number;
};

/**
 * One single-thread kernel with two binary searches over sorted row times: the rows whose bucket
 * assignment lands in `[0, bucketCount)` form one contiguous range, written to the first two words
 * of `rowSlots`.
 *
 * Float times: a row is below the range when `d = time - origin` is negative and above it when
 * `d >= bucketCount * width` (one f32 product, the same correctly rounded product the bucket
 * correction step uses). Word times: below when the exact signed comparison with the origin is
 * negative, above when the exact 64-bit by 32-bit quotient reaches `bucketCount`. Invalid
 * parameters (NaN, non-positive or infinite width, zero word width) give an empty range.
 *
 * @internal
 */
export function createTemporalReductionRangeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TemporalReductionRangeProps
): GPUCommandNode<Parameters> {
  const isWordMode = props.timestamps.format === 'uint32x2';
  const rows = props.timestamps.length;
  const declarations = isWordMode
    ? /* wgsl */ `${TIME_WORDS_WGSL}
override BUCKET_COUNT: u32;
override ROW_COUNT: u32;

fn getRowTime(row: u32) -> vec2<u32> {
  return vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
}

// Same exact division as the bucket kernel: high word first, then the low word bit by bit.
fn isQuotientAtLeastBucketCount(difference: vec2<u32>, width: u32) -> bool {
  let quotientHigh = difference.y / width;
  var remainder = difference.y % width;
  var quotientLow = 0u;
  for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
    let carry = remainder >> 31u;
    remainder = (remainder << 1u) | ((difference.x >> u32(bit)) & 1u);
    if (carry != 0u || remainder >= width) {
      remainder = remainder - width;
      quotientLow = quotientLow | (1u << u32(bit));
    }
  }
  return quotientHigh != 0u || quotientLow >= BUCKET_COUNT;
}

fn isRowBelowRange(row: u32, origin: vec2<u32>) -> bool {
  return timeWordsCompare(getRowTime(row), origin) < 0i;
}

// Only called for rows at or after the lower bound, which are not before the origin.
fn isRowAboveRange(row: u32, origin: vec2<u32>, width: u32) -> bool {
  return isQuotientAtLeastBucketCount(timeWordsSubtract(getRowTime(row), origin), width);
}`
    : /* wgsl */ `${IS_NAN_BITS_WGSL}
override BUCKET_COUNT: u32;
override ROW_COUNT: u32;

fn isRowBelowRange(row: u32, origin: f32) -> bool {
  return rowTimes[rowTimesOffset + row] - origin < 0.0;
}

fn isRowAboveRange(row: u32, origin: f32, limit: f32) -> bool {
  return !(rowTimes[rowTimesOffset + row] - origin < limit);
}`;
  const parameterBody = isWordMode
    ? /* wgsl */ `let origin = vec2<u32>(params[paramsOffset], params[paramsOffset + 1u]);
  let width = params[paramsOffset + 2u];
  let isValid = width != 0u;`
    : /* wgsl */ `let origin = params[paramsOffset];
  let width = params[paramsOffset + 1u];
  let isValid = !isNanBits(origin) && !isNanBits(width) && width > 0.0 &&
    (bitcast<u32>(width) & 0x7fffffffu) != 0x7f800000u;
  let limit = f32(BUCKET_COUNT) * width;`;
  const below = 'isRowBelowRange(middle, origin)';
  const above = isWordMode
    ? 'isRowAboveRange(middle, origin, width)'
    : 'isRowAboveRange(middle, origin, limit)';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTemporalReduction',
    variant: isWordMode ? 'range-search-words' : 'range-search',
    bindings: [
      {name: 'rowTimes', view: props.timestamps, type: isWordMode ? 'u32' : 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: isWordMode ? 'u32' : 'f32', access: 'read'},
      {name: 'rowSlots', view: props.rowSlots, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    workgroupSize: 32,
    declarations,
    constants: {BUCKET_COUNT: props.bucketCount, ROW_COUNT: rows},
    body: /* wgsl */ `
  ${parameterBody}
  var firstRow = 0u;
  var endRow = 0u;
  if (isValid) {
    // Lower bound: the first row that is not below the range.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (${below}) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    firstRow = low;
    // Upper bound: the first row above the range, searched from the lower bound.
    high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (${above}) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    endRow = low;
  }
  rowSlots[rowSlotsOffset] = firstRow;
  rowSlots[rowSlotsOffset + 1u] = endRow - firstRow;`
  });
}
