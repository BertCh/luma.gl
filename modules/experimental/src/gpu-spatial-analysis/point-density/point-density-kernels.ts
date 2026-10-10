// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {GPU_POINT_DENSITY_HEXAGON_WGSL} from './point-density-hexagon';

const OPERATION = 'GPUPointDensity';

/** Bounds resolved to either a literal or one GPU `float32x4` row. @internal */
export type PointDensityResolvedBounds =
  | readonly [number, number, number, number]
  | GraphDataView<'float32x4'>;

/** Writes `overflow[0] = 0` at the start of every encoding. @internal */
export function createPointDensityClearOverflowNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  overflow: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: 'clear-overflow',
    bindings: [{name: 'overflowFlags', view: overflow, type: 'u32', access: 'read_write'}],
    invocationCount: 1,
    body: 'overflowFlags[overflowFlagsOffset] = 0u;'
  });
}

/**
 * Writes one odd-r hexagon cell key per position in a chunk, or `0xffffffff` for rejected rows.
 *
 * @internal
 */
export function createPointDensityHexagonKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView;
    keys: GraphDataView<'uint32'>;
    keyStart: number;
    gridSize: readonly [number, number];
    bounds: PointDensityResolvedBounds;
    radius: number | GraphDataView<'float32'>;
    overflow?: GraphDataView<'uint32'>;
    /** Optional per-row mask over the whole key range; rows with `0` get the invalid key. */
    mask?: GraphDataView<'uint32'>;
    /** Row of `mask` matching the first position of this chunk. */
    maskStart?: number;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'keys', view: props.keys, type: 'u32', access: 'read_write'}
  ];
  if (props.mask) {
    bindings.push({name: 'rowMask', view: props.mask, type: 'u32', access: 'read'});
  }
  const boundsIsView = !Array.isArray(props.bounds);
  if (boundsIsView) {
    bindings.push({
      name: 'boundsValues',
      view: props.bounds as GraphDataView,
      type: 'f32',
      access: 'read'
    });
  }
  if (typeof props.radius !== 'number') {
    bindings.push({name: 'radiusValues', view: props.radius, type: 'f32', access: 'read'});
  }
  if (props.overflow) {
    bindings.push({
      name: 'overflowFlags',
      view: props.overflow,
      type: 'atomic<u32>',
      access: 'read_write'
    });
  }
  const literal = boundsIsView
    ? undefined
    : (props.bounds as readonly number[]).map(getWGSLFloatLiteral);
  const boundsSource = literal
    ? `let minimumX = ${literal[0]}; let minimumY = ${literal[1]};
  let maximumX = ${literal[2]}; let maximumY = ${literal[3]};`
    : `let minimumX = boundsValues[boundsValuesOffset];
  let minimumY = boundsValues[boundsValuesOffset + 1u];
  let maximumX = boundsValues[boundsValuesOffset + 2u];
  let maximumY = boundsValues[boundsValuesOffset + 3u];`;
  const radiusSource =
    typeof props.radius === 'number'
      ? `let radius = ${getWGSLFloatLiteral(props.radius)};`
      : 'let radius = radiusValues[radiusValuesOffset];';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'hexagon-keys',
    bindings,
    invocationCount: props.positions.length,
    declarations: `const KEY_START: u32 = ${props.keyStart}u;
const COLUMNS: u32 = ${props.gridSize[0]}u;
const ROWS: u32 = ${props.gridSize[1]}u;
const MASK_START: u32 = ${props.maskStart ?? 0}u;
const INVALID_KEY: u32 = 0xffffffffu;
const MAXIMUM_FLOAT: f32 = 3.402823466e+38;
${GPU_POINT_DENSITY_HEXAGON_WGSL}`,
    body: `${boundsSource}
  ${radiusSource}
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var key = INVALID_KEY;
  let included = ${props.mask ? 'rowMask[rowMaskOffset + MASK_START + index] != 0u' : 'true'};
  let finite = included && abs(x) <= MAXIMUM_FLOAT && abs(y) <= MAXIMUM_FLOAT;
  let finiteBounds = abs(minimumX) <= MAXIMUM_FLOAT && abs(minimumY) <= MAXIMUM_FLOAT &&
    abs(maximumX) <= MAXIMUM_FLOAT && abs(maximumY) <= MAXIMUM_FLOAT;
  let orderedBounds = minimumX <= maximumX && minimumY <= maximumY;
  let inside = x >= minimumX && x <= maximumX && y >= minimumY && y <= maximumY;
  let validRadius = radius > 0.0 && radius <= MAXIMUM_FLOAT;
  if (finite && finiteBounds && orderedBounds && inside && validRadius) {
    let cell = getPointDensityHexagonCell(x, y, minimumX, minimumY, radius);
    if (cell.x >= 0 && cell.y >= 0 && u32(cell.x) < COLUMNS && u32(cell.y) < ROWS) {
      key = u32(cell.y) * COLUMNS + u32(cell.x);
    } else {
      ${props.overflow ? 'atomicStore(&overflowFlags[overflowFlagsOffset], 1u);' : ''}
    }
  }
  keys[keysOffset + KEY_START + index] = key;`
  });
}

/**
 * Copies one chunk of positions into `maskedPositions` at `rowStart`, writing NaN for rows whose
 * mask is 0. `GPUGridBinning` and `GPUGridAggregation` ignore non-finite positions.
 *
 * @internal
 */
export function createPointDensityMaskedPositionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView;
    mask: GraphDataView<'uint32'>;
    maskedPositions: GraphDataView;
    rowStart: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'mask-positions',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'rowMask', view: props.mask, type: 'u32', access: 'read'},
      {name: 'maskedPositions', view: props.maskedPositions, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `const ROW_START: u32 = ${props.rowStart}u;`,
    body: `var nanBits = 0x7fc00000u;
  let nan = bitcast<f32>(nanBits);
  let included = rowMask[rowMaskOffset + ROW_START + index] != 0u;
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let slot = maskedPositionsOffset + (ROW_START + index) * 2u;
  maskedPositions[slot] = select(nan, x, included);
  maskedPositions[slot + 1u] = select(nan, y, included);`
  });
}

/**
 * Converts counts and optional sums into the float field and optional means.
 *
 * @internal
 */
export function createPointDensityFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    statistic: 'count' | 'sum' | 'mean';
    counts: GraphDataView<'uint32'>;
    sums?: GraphDataView<'float32'>;
    values: GraphDataView<'float32'>;
    means?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'}
  ];
  if (props.sums) {
    bindings.push({name: 'sums', view: props.sums, type: 'f32', access: 'read'});
  }
  bindings.push({name: 'values', view: props.values, type: 'f32', access: 'read_write'});
  if (props.means) {
    bindings.push({name: 'means', view: props.means, type: 'f32', access: 'read_write'});
  }
  const valueExpression =
    props.statistic === 'count' ? 'f32(count)' : props.statistic === 'sum' ? 'sum' : 'mean';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'finalize',
    bindings,
    invocationCount: props.values.length,
    body: `let count = counts[countsOffset + index];
  let sum = ${props.sums ? 'sums[sumsOffset + index]' : 'f32(count)'};
  let mean = select(0.0, sum / f32(max(count, 1u)), count > 0u);
  values[valuesOffset + index] = ${valueExpression};
  ${props.means ? 'means[meansOffset + index] = mean;' : ''}`
  });
}

/** Items combined by one workgroup of {@link createPointDensityWorkgroupSumNode}. @internal */
export const POINT_DENSITY_WORKGROUP_SUM_SIZE = 256;

/**
 * Writes one row-major grid cell key per position in `keys`, or `0xffffffff` for non-finite and
 * out-of-bounds rows. Uses the same cell arithmetic as `GPUGridBinning`.
 *
 * @internal
 */
export function createPointDensityGridKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView;
    keys: GraphDataView<'uint32'>;
    keyStart: number;
    gridSize: readonly [number, number];
    bounds: PointDensityResolvedBounds;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'keys', view: props.keys, type: 'u32', access: 'read_write'}
  ];
  const boundsIsView = !Array.isArray(props.bounds);
  if (boundsIsView) {
    bindings.push({
      name: 'boundsValues',
      view: props.bounds as GraphDataView,
      type: 'f32',
      access: 'read'
    });
  }
  const literal = boundsIsView
    ? undefined
    : (props.bounds as readonly number[]).map(getWGSLFloatLiteral);
  const boundsSource = literal
    ? `let minimumX = ${literal[0]}; let minimumY = ${literal[1]};
  let maximumX = ${literal[2]}; let maximumY = ${literal[3]};`
    : `let minimumX = boundsValues[boundsValuesOffset];
  let minimumY = boundsValues[boundsValuesOffset + 1u];
  let maximumX = boundsValues[boundsValuesOffset + 2u];
  let maximumY = boundsValues[boundsValuesOffset + 3u];`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'grid-keys',
    bindings,
    invocationCount: props.positions.length,
    declarations: `const KEY_START: u32 = ${props.keyStart}u;
const COLUMNS: u32 = ${props.gridSize[0]}u;
const ROWS: u32 = ${props.gridSize[1]}u;
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}`,
    body: `${boundsSource}
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var key = 0xffffffffu;
  let finite = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
  let finiteBounds = minimumX == minimumX && minimumY == minimumY &&
    maximumX == maximumX && maximumY == maximumY &&
    abs(minimumX) <= 3.402823466e+38 && abs(minimumY) <= 3.402823466e+38 &&
    abs(maximumX) <= 3.402823466e+38 && abs(maximumY) <= 3.402823466e+38;
  let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
  let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
  if (finiteBounds && maximumX >= minimumX && maximumY >= minimumY && finite && inX && inY) {
    key = getCoordinate(y, minimumY, maximumY, ROWS) * COLUMNS + getCoordinate(x, minimumX, maximumX, COLUMNS);
  }
  keys[keysOffset + KEY_START + index] = key;`
  });
}

/** Sets every float sum cell to zero. @internal */
export function createPointDensityClearSumsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  sums: GraphDataView<'float32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: 'clear-sums',
    bindings: [{name: 'sums', view: sums, type: 'atomic<u32>', access: 'read_write'}],
    invocationCount: sums.length,
    body: 'atomicStore(&sums[sumsOffset + index], 0u);'
  });
}

/**
 * Adds one chunk of weights into per-cell float sums with workgroup-local pre-aggregation.
 *
 * Each 256-thread workgroup stages its rows in shared memory; every row sums the rows that
 * share its key (in row order) and only the first row of each key issues one compare-exchange
 * float add. Rows that share a cell collapse by up to 256x before reaching global memory, so hotspot
 * cells no longer serialize one compare-exchange loop per point. Rows with key at or above
 * `cellCount` or a non-finite weight contribute nothing. `sums` must be cleared beforehand. The
 * add order across workgroups is nondeterministic.
 *
 * @internal
 */
export function createPointDensityWorkgroupSumNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    weights: GraphDataView<'float32'>;
    keys: GraphDataView<'uint32'>;
    /** Row of `keys` matching the first weight of this chunk. */
    keyStart: number;
    sums: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const size = POINT_DENSITY_WORKGROUP_SUM_SIZE;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'workgroup-sums',
    bindings: [
      {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
      {name: 'keys', view: props.keys, type: 'u32', access: 'read'},
      {name: 'sums', view: props.sums, type: 'atomic<u32>', access: 'read_write'}
    ],
    workgroupSize: size,
    invocationCount: props.weights.length,
    guardIndex: false,
    declarations: `const KEY_START: u32 = ${props.keyStart}u;
const CELL_COUNT: u32 = ${props.sums.length}u;
var<workgroup> sortKeys: array<u32, ${size}>;
var<workgroup> sortValues: array<f32, ${size}>;
fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var oldBits = atomicLoad(destination);
  loop {
    let newBits = bitcast<u32>(bitcast<f32>(oldBits) + value);
    let result = atomicCompareExchangeWeak(destination, oldBits, newBits);
    if (result.exchanged) { break; }
    oldBits = result.old_value;
  }
}`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let lane = localInvocationIndex;
  var key = 0xffffffffu;
  var value = 0.0;
  if (index < INVOCATION_COUNT) {
    let weight = weights[weightsOffset + index];
    let rowKey = keys[keysOffset + KEY_START + index];
    if (weight == weight && abs(weight) <= 3.402823466e+38 && rowKey < CELL_COUNT) {
      key = rowKey;
      value = weight;
    }
  }
  sortKeys[lane] = key;
  sortValues[lane] = value;
  workgroupBarrier();
  // Every row sums the rows of its workgroup that share its key, in row order. Only the first
  // row of each key issues the global add, so no sort or scan is needed.
  var runSum = 0.0;
  var isFirst = true;
  for (var other = 0u; other < ${size}u; other = other + 1u) {
    if (sortKeys[other] == key) {
      runSum = runSum + sortValues[other];
      isFirst = isFirst && other >= lane;
    }
  }
  if (isFirst && key < CELL_COUNT) {
    atomicAddFloat(&sums[sumsOffset + key], runSum);
  }`
  });
}
