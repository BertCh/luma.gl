// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {getColumnQuantileNodes, type GPUColumnQuantilesOutput} from './column-quantiles-nodes';
import {
  getGPUColumnQuantilesParameterLength,
  GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT
} from './column-quantiles-parameters';

export type {GPUColumnQuantilesOutput} from './column-quantiles-nodes';

const OPERATION = 'GPUColumnQuantiles';
/** Largest column: row counts are held in f32 arithmetic, exact up to 2^24. */
const MAXIMUM_ROW_COUNT = 2 ** 24;

/**
 * Properties for {@link GPUColumnQuantiles}.
 *
 * Per-frame (no recompile): the contents of `parameters` (probabilities, interpolation, filter
 * range) and of `values` and `mask`. Topology (needs a new graph): `quantileCount`, view lengths,
 * whether `mask` is present, and which optional outputs are present.
 */
export type GPUColumnQuantilesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'column-quantiles'`. */
  id?: string;
  /** Packed float32 column. NaN and +-Infinity rows are missing and skipped. At most 2^24 rows. */
  values: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero skips the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Float32 parameters written with `getGPUColumnQuantilesParameterValues`:
   * `[interpolationCode, filterLower, filterUpper, 0, p_0 ... p_{quantileCount-1}]`.
   */
  parameters: GraphDataView<'float32'>;
  /** Number of probability slots, compile-time, 1 to 64. */
  quantileCount: number;
  /** Caller-owned outputs. */
  output: GPUColumnQuantilesOutput;
};

/**
 * Exact quantiles of a float32 column by GPU radix selection, plus a per-frame percentile-range
 * filter mask, with no readback and no sort.
 *
 * With `n` valid rows (unmasked and finite: NaN and +-Infinity are missing, as in every
 * interpolating kernel; `-0` equals `+0`) sorted ascending as `x[0..n-1]` and probability `p`,
 * `h = fround(fround(n - 1) * p)`, `lo = floor(h)`, `hi = min(lo + 1, n - 1)`:
 * `lower` is `x[lo]`, `higher` is `x[ceil(h)]`, `nearest` is `x[round-half-to-even(h)]` (numpy),
 * `linear` is `x[lo] + (x[hi] - x[lo]) * (h - lo)` (numpy `linear`, d3 `quantile`, R-7), and
 * `midpoint` is `(x[lo] + x[ceil(h)]) / 2`. When the two order statistics compare equal the first
 * is returned unchanged. A probability that is NaN or outside `[0, 1]`, or `n == 0`, gives NaN.
 * This is `numpy.quantile(x[numpy.isfinite(x)], p, method=...)`; numpy keeps infinities and
 * raises on an invalid `p`. `GPUClassBreaks` and the `GPUGroupStatistics` medians and percentiles
 * use the same policy and the same `linear` arithmetic. When `x[hi] - x[lo]` overflows f32, the
 * convex form `x[lo] * (1 - t) + x[hi] * t` keeps the result finite.
 *
 * The percentile filter follows deck.gl `lowerPercentile` / `upperPercentile` semantics with
 * fractions `fl`, `fu` in `[0, 1]` (NaN counts as 0 and 1): `lowerIndex = clamp(floor(n * fl), 0,
 * n - 1)` and `upperIndex = clamp(ceil(n * fu) - 1, 0, n - 1)`, with the products rounded to f32.
 * `filterBounds` is `[x[lowerIndex], x[upperIndex]]` and `outputMask[row]` is 1 when the row is
 * valid and `lowerBound <= value <= upperBound` in ordered-key order. With `n == 0` the bounds are
 * NaN and the mask is all 0.
 *
 * Algorithm: four 8-bit digit passes over order-preserving u32 keys, most significant first. Each
 * pass builds histograms with integer atomics (workgroup-local, flushed to global) per distinct
 * active prefix and one single-workgroup kernel narrows every target rank to a bucket. Targets are
 * the needed order statistics (two per probability plus two for the filter). The result key is
 * exact, so every output, including `linear` and `midpoint` (each f32 step is rounded separately,
 * never fused into a multiply-add), is bitwise equal to a sort-based `Math.fround` oracle and
 * independent of GPU thread order.
 *
 * Limits: at most 2^24 rows (row counts and ranks use f32 arithmetic) and 64 probabilities.
 * Inputs must be single packed views (chunked vectors are rejected).
 */
export class GPUColumnQuantiles implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUColumnQuantilesProps;

  constructor(props: GPUColumnQuantilesProps) {
    this.id = props.id ?? 'column-quantiles';
    this.props = props;
    const id = this.id;
    const {quantileCount, output} = props;
    validateCapacity(
      id,
      'quantileCount',
      quantileCount,
      1,
      GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT
    );
    const rows = props.values.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (rows > MAXIMUM_ROW_COUNT) {
      throw new Error(`${id} supports at most ${MAXIMUM_ROW_COUNT} rows`);
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.mask) {
      validatePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUColumnQuantilesParameterLength(quantileCount);
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    validatePackedView(output.quantiles, ['float32'], `${id} output.quantiles`);
    if (output.quantiles.length < quantileCount) {
      throw new Error(`${id} output.quantiles must hold quantileCount rows`);
    }
    validatePackedView(output.validCount, ['uint32'], `${id} output.validCount`);
    if (output.validCount.length < 1) {
      throw new Error(`${id} output.validCount must hold one row`);
    }
    if (output.outputMask) {
      validatePackedView(output.outputMask, ['uint32'], `${id} output.outputMask`);
      if (output.outputMask.length < rows) {
        throw new Error(`${id} output.outputMask must hold values.length rows`);
      }
    }
    if (output.filterBounds) {
      validatePackedView(output.filterBounds, ['float32'], `${id} output.filterBounds`);
      if (output.filterBounds.length < 2) {
        throw new Error(`${id} output.filterBounds must hold two rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.quantiles, output.validCount, output.outputMask, output.filterBounds],
      [props.values, props.mask, props.parameters]
    );
  }

  /**
   * Returns init, four histogram and select pairs, finish, and optionally filter-mask nodes.
   * Node IDs are `${id}-init`, `${id}-histogram-0..3`, `${id}-select-0..3`, `${id}-finish`, and
   * `${id}-filter-mask`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    return getColumnQuantileNodes(graph, {
      id,
      operation: OPERATION,
      values: props.values,
      mask: props.mask,
      parameters: props.parameters,
      quantileCount: props.quantileCount,
      output: props.output
    });
  }
}
