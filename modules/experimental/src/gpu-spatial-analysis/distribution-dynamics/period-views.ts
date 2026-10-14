// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';

/**
 * Elements per period in a padded period-major layout. 64 four-byte elements are 256 bytes, so
 * every period view starts at the same offset residue modulo 256 and all periods share one
 * pipeline (rule 12: the period index never reaches shader text).
 */
const PERIOD_STRIDE_ALIGNMENT = 64;

/** Returns `rows` rounded up to a multiple of 64, the stride of a padded period-major layout. */
export function getPaddedPeriodStride(rows: number): number {
  return Math.ceil(rows / PERIOD_STRIDE_ALIGNMENT) * PERIOD_STRIDE_ALIGNMENT;
}

/**
 * Returns the packed view of period `period` of a period-major column, sharing the column's
 * buffer. The contiguous layout has period `t` at `[t * rows, (t + 1) * rows)`; pass `stride`
 * (see {@link getPaddedPeriodStride}) for a padded layout with period `t` at `t * stride`.
 *
 * @internal
 */
export function getPeriodView<Parameters, Format extends GPUVectorFormat>(
  graph: GPUCommandGraph<Parameters>,
  column: GraphDataView<Format>,
  period: number,
  rows: number,
  stride: number = rows
): GraphDataView<Format> {
  return graph.createDataView(column.buffer, {
    format: column.format,
    length: rows,
    byteOffset: column.byteOffset + period * stride * column.rowByteLength
  });
}

/**
 * One node that copies a contiguous period-major column (`rows * periods` entries) into a padded
 * one (`stride` per period). The padding entries are left untouched and are never read.
 *
 * @internal
 */
export function createPeriodPackNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    source: GraphDataView<'float32'>;
    padded: GraphDataView<'float32'>;
    rows: number;
    periods: number;
  }
): GPUCommandNode<Parameters> {
  const {rows, periods} = props;
  const stride = getPaddedPeriodStride(rows);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'pack-periods',
    bindings: [
      {name: 'contiguous', view: props.source, type: 'f32', access: 'read'},
      {name: 'padded', view: props.padded, type: 'f32', access: 'read_write'}
    ],
    invocationCount: rows * periods,
    body: `let period = index / ${rows}u;
  padded[paddedOffset + period * ${stride}u + (index - period * ${rows}u)] = contiguous[contiguousOffset + index];`
  });
}

/**
 * One node that copies a padded period-major column (`stride` per period) into a contiguous one
 * (`rows * periods` entries); the inverse of {@link createPeriodPackNode}.
 *
 * @internal
 */
export function createPeriodUnpackNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    padded: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
    rows: number;
    periods: number;
  }
): GPUCommandNode<Parameters> {
  const {rows, periods} = props;
  const stride = getPaddedPeriodStride(rows);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'unpack-periods',
    bindings: [
      {name: 'padded', view: props.padded, type: 'f32', access: 'read'},
      {name: 'contiguous', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: rows * periods,
    body: `let period = index / ${rows}u;
  contiguous[contiguousOffset + index] = padded[paddedOffset + period * ${stride}u + (index - period * ${rows}u)];`
  });
}
