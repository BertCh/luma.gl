// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';

/** Workgroup size of the fixed-order moment kernels. */
export const MOMENT_WORKGROUP_SIZE = 256;
/** Rows each invocation folds sequentially before the workgroup tree. */
export const MOMENT_ROWS_PER_INVOCATION = 8;
/** Rows reduced by one workgroup tile. */
export const MOMENT_TILE_SIZE = MOMENT_WORKGROUP_SIZE * MOMENT_ROWS_PER_INVOCATION;
/** Floats per moment record: `[count, mean, M2, minimum, maximum]`. */
export const MOMENT_STRIDE = 5;

/** Indices into the `state` transient written by the prepare kernel. @internal */
export const CLASS_BREAKS_STATE = {method: 0, classCount: 1} as const;
/** Indices into the `extremes` transient. @internal */
export const CLASS_BREAKS_EXTREMES = {
  minimumKey: 0,
  maximumKey: 1,
  validCount: 2,
  finiteCount: 3
} as const;
/** Indices into the `headTail` transient. @internal */
export const CLASS_BREAKS_HEAD_TAIL = {
  done: 0,
  breakCount: 1,
  thresholdKey: 2,
  previousCount: 3
} as const;

/**
 * WGSL moment record and the Chan merge used by every fixed-order moment kernel.
 *
 * The merge is the pairwise update of Chan, Golub and LeVeque; adding one row is a merge with
 * `(1, x, 0, x, x)`. Both kernels fold in a fixed order, so results are bitwise reproducible.
 */
const MOMENTS_WGSL = /* wgsl */ `
struct Moments {
  count: f32,
  mean: f32,
  m2: f32,
  minimum: f32,
  maximum: f32,
}

fn mergeMoments(left: Moments, right: Moments) -> Moments {
  if (right.count == 0.0) {
    return left;
  }
  if (left.count == 0.0) {
    return right;
  }
  let count = left.count + right.count;
  let delta = right.mean - left.mean;
  let mean = left.mean + delta * (right.count / count);
  let m2 = left.m2 + right.m2 + delta * delta * (left.count * right.count / count);
  return Moments(count, mean, m2, min(left.minimum, right.minimum), max(left.maximum, right.maximum));
}

fn emptyMoments() -> Moments {
  return Moments(0.0, 0.0, 0.0, 0.0, 0.0);
}

var<workgroup> sharedMoments: array<Moments, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> gateShared: u32;

fn reduceSharedMoments(local: u32) {
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedMoments[local] = mergeMoments(sharedMoments[local], sharedMoments[local + stride]);
    }
  }
}
`;

/** Gate for one fixed-order moment reduction. @internal */
export type ClassBreaksMomentGate =
  | {kind: 'method'; methodCode: number}
  | {kind: 'head-tail'; methodCode: number; headTail: GraphDataView<'uint32'>; round: number};

/**
 * Returns the tile and merge nodes of one fixed-order `(count, mean, M2, min, max)` reduction over
 * the unmasked finite values, optionally restricted to values whose ordered key is above the
 * head/tail threshold key. The merged record is written to `moments[slot]`.
 *
 * @internal
 */
export function getClassBreaksMomentNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    values: GraphDataView<'float32'>;
    mask?: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    partials: GraphDataView<'float32'>;
    moments: GraphDataView<'float32'>;
    slot: number;
    gate: ClassBreaksMomentGate;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, values, mask, state, partials, moments, slot, gate} = props;
  const rowCount = values.length;
  const tileCount = Math.ceil(rowCount / MOMENT_TILE_SIZE);
  const headTail = gate.kind === 'head-tail' ? gate.headTail : undefined;
  const gateExpression =
    gate.kind === 'head-tail'
      ? `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] == ${gate.methodCode}u && headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.done}u] == 0u`
      : `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] == ${gate.methodCode}u`;
  const tileBindings: WGSLKernelBinding[] = [
    {name: 'values', view: values, type: 'f32', access: 'read'},
    ...(mask ? [{name: 'rowMask', view: mask, type: 'u32', access: 'read'} as const] : []),
    {name: 'state', view: state, type: 'u32', access: 'read'},
    ...(headTail ? [{name: 'headTail', view: headTail, type: 'u32', access: 'read'} as const] : []),
    {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
  ];
  const tileNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-tile`,
    operation,
    variant: 'moments-tile',
    bindings: tileBindings,
    invocationCount: tileCount * MOMENT_WORKGROUP_SIZE,
    workgroupSize: MOMENT_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const TILE_SIZE: u32 = ${MOMENT_TILE_SIZE}u;
const ROWS_PER_INVOCATION: u32 = ${MOMENT_ROWS_PER_INVOCATION}u;
${COLUMN_ORDERED_KEY_WGSL}
${MOMENTS_WGSL}`,
    body: `let local = localInvocationIndex;
  if (local == 0u) {
    gateShared = select(0u, 1u, ${gateExpression});
  }
  if (workgroupUniformLoad(&gateShared) == 0u) {
    return;
  }
  let tile = workgroupIndex;
  ${headTail ? `let thresholdKey = headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.thresholdKey}u];` : ''}
  var accumulated = emptyMoments();
  for (var step = 0u; step < ROWS_PER_INVOCATION; step = step + 1u) {
    // Uniform trip count: rows past the end are skipped by a predicate, never by a break, so
    // the barriers after the loop stay in uniform control flow.
    let row = min(tile * TILE_SIZE + step * ${MOMENT_WORKGROUP_SIZE}u + local, ROW_COUNT - 1u);
    let inRange = tile * TILE_SIZE + step * ${MOMENT_WORKGROUP_SIZE}u + local < ROW_COUNT;
    let value = values[valuesOffset + row];
    var include = inRange && (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
    ${mask ? 'include = include && rowMask[rowMaskOffset + row] != 0u;' : ''}
    ${headTail ? 'include = include && getOrderedKey(value) > thresholdKey;' : ''}
    if (include) {
      accumulated = mergeMoments(accumulated, Moments(1.0, value, 0.0, value, value));
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = partialsOffset + tile * ${MOMENT_STRIDE}u;
    partials[base] = record.count;
    partials[base + 1u] = record.mean;
    partials[base + 2u] = record.m2;
    partials[base + 3u] = record.minimum;
    partials[base + 4u] = record.maximum;
  }`
  });

  const mergeBindings: WGSLKernelBinding[] = [
    {name: 'state', view: state, type: 'u32', access: 'read'},
    ...(headTail ? [{name: 'headTail', view: headTail, type: 'u32', access: 'read'} as const] : []),
    {name: 'partials', view: partials, type: 'f32', access: 'read'},
    {name: 'moments', view: moments, type: 'f32', access: 'read_write'}
  ];
  const mergeNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-merge`,
    operation,
    variant: 'moments-merge',
    bindings: mergeBindings,
    invocationCount: MOMENT_WORKGROUP_SIZE,
    workgroupSize: MOMENT_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const TILE_COUNT: u32 = ${tileCount}u;
const STEP_COUNT: u32 = ${Math.ceil(tileCount / MOMENT_WORKGROUP_SIZE)}u;
${MOMENTS_WGSL}`,
    body: `let local = localInvocationIndex;
  if (local == 0u) {
    gateShared = select(0u, 1u, ${gateExpression});
  }
  if (workgroupUniformLoad(&gateShared) == 0u) {
    return;
  }
  var accumulated = emptyMoments();
  for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
    let tile = step * ${MOMENT_WORKGROUP_SIZE}u + local;
    if (tile < TILE_COUNT) {
      let base = partialsOffset + tile * ${MOMENT_STRIDE}u;
      accumulated = mergeMoments(
        accumulated,
        Moments(partials[base], partials[base + 1u], partials[base + 2u], partials[base + 3u], partials[base + 4u])
      );
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = momentsOffset + ${slot * MOMENT_STRIDE}u;
    moments[base] = record.count;
    moments[base + 1u] = record.mean;
    moments[base + 2u] = record.m2;
    moments[base + 3u] = record.minimum;
    moments[base + 4u] = record.maximum;
  }`
  });
  return [tileNode, mergeNode];
}

/** Returns the number of moment tile partials for `rowCount` rows. @internal */
export function getClassBreaksMomentTileCount(rowCount: number): number {
  return Math.ceil(rowCount / MOMENT_TILE_SIZE);
}

/**
 * WGSL that assigns a finite value to one of `BIN_COUNT` equal-width bins of `[minimum, maximum]`
 * reproducibly: `width = fround(range * fround(1 / BIN_COUNT))`, then `floor(d / width)` is
 * corrected with one step each way against correctly rounded products, so the CPU oracle can
 * mirror the bin with `Math.fround`. Requires `INVERSE_BIN_COUNT` and `BIN_COUNT` constants.
 *
 * @internal
 */
export const CLASS_BREAKS_BIN_WGSL = /* wgsl */ `
fn getBinIndex(value: f32, minimum: f32, width: f32) -> u32 {
  let difference = value - minimum;
  var bin = floor(difference / width);
  bin = clamp(bin, 0.0, f32(BIN_COUNT - 1u));
  if (difference < bin * width) {
    bin = bin - 1.0;
  } else if (difference >= (bin + 1.0) * width) {
    bin = bin + 1.0;
  }
  return u32(clamp(bin, 0.0, f32(BIN_COUNT - 1u)));
}
`;
