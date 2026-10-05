// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedView,
  type GraphDataView,
  type GPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';

/** Nodata description shared by raster-algebra contributors. @internal */
export type RasterAlgebraNoData = {
  /** Optional finite sentinel compared exactly against cell values. NaN is always nodata. */
  noDataValue?: number;
};

/**
 * WGSL helpers: `getNaN()`, `isFiniteValue(value)` and `isNoDataValue(value)`.
 *
 * NaN is always nodata (bit test, because WGSL may optimise `x != x` away); infinities are values;
 * an optional finite sentinel is compared exactly.
 *
 * @internal
 */
export function getRasterAlgebraValueWGSL(noDataValue: number | undefined): string {
  return /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn isNoDataValue(value: f32) -> bool {
  return isNaNValue(value)${noDataValue !== undefined ? ` || value == ${getWGSLFloatLiteral(noDataValue)}` : ''};
}`;
}

/**
 * Throws unless `view` is a single packed float32 or uint32 view with at least `rowCount` rows.
 *
 * @internal
 */
export function validateRasterAlgebraView(
  id: string,
  name: string,
  view: GraphDataView | GraphVectorView | undefined,
  format: 'float32' | 'uint32',
  rowCount: number
): void {
  if (!view) {
    return;
  }
  if (view instanceof GraphVectorView) {
    throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
  }
  validatePackedView(view, [format], `${id} ${name}`);
  if (view.length < rowCount) {
    throw new Error(`${id} ${name} must hold at least ${rowCount} rows`);
  }
}

/** Throws unless `value` is a positive safe integer not above `maximum`. @internal */
export function validateRasterAlgebraCount(
  id: string,
  name: string,
  value: number,
  minimum: number,
  maximum: number
): void {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${id} ${name} must be an integer in [${minimum}, ${maximum}]`);
  }
}

/** Throws unless the optional sentinel is finite. @internal */
export function validateRasterAlgebraNoData(id: string, noDataValue: number | undefined): void {
  if (noDataValue !== undefined && !Number.isFinite(noDataValue)) {
    throw new Error(`${id} noDataValue must be finite (NaN cells are always nodata)`);
  }
}

/**
 * Throws when outputs share buffers with each other or with inputs.
 *
 * @internal
 */
export function validateRasterAlgebraAliasing(
  id: string,
  outputs: readonly (GraphDataView | undefined)[],
  inputs: readonly (GraphDataView | undefined)[]
): void {
  const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
  if (new Set(outputBuffers).size !== outputBuffers.length) {
    throw new Error(`${id} outputs must not share buffers`);
  }
  const inputBuffers = new Set(inputs.filter(view => view !== undefined).map(view => view.buffer));
  if (outputBuffers.some(buffer => inputBuffers.has(buffer))) {
    throw new Error(`${id} outputs must not share buffers with inputs`);
  }
}

/** Validates that every defined view belongs to `graph`. @internal */
export function validateRasterAlgebraGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly (GraphDataView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, views);
}

/** Throws unless the parameter target holds `length` elements. @internal */
export function validateParameterTarget(name: string, target: ArrayLike<number>, length: number) {
  if (target.length < length) {
    throw new Error(`${name} parameter target must hold ${length} elements`);
  }
}

/**
 * WGSL `fn ${functionName}(first: u32, count: u32, value: f32, closedRight: bool) -> u32` that
 * counts ascending breaks `binding[first .. first + count)` that are `<= value` (or `< value` when
 * `closedRight`) by binary search. The breaks must be sorted ascending; NaN breaks give an
 * unspecified but deterministic class.
 *
 * @internal
 */
export function getBreakSearchWGSL(functionName: string, bindingName: string): string {
  return /* wgsl */ `
fn ${functionName}(first: u32, count: u32, value: f32, closedRight: bool) -> u32 {
  var low = 0u;
  var high = count;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    let boundary = ${bindingName}[${bindingName}Offset + first + middle];
    let isBelow = select(boundary <= value, boundary < value, closedRight);
    if (isBelow) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`;
}
