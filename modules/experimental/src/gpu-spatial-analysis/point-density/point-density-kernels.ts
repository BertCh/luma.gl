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
  let inside = x >= minimumX && x <= maximumX && y >= minimumY && y <= maximumY;
  let validRadius = radius > 0.0 && radius <= MAXIMUM_FLOAT;
  if (finite && inside && validRadius) {
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
