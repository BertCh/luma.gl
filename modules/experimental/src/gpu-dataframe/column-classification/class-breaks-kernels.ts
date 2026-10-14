// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
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

/**
 * Indices into the `state` transient written by the prepare kernel. `layerCursor` is the
 * natural-breaks dynamic-program layer and `refinementCursor` the natural-breaks refinement
 * half-round, both advanced on the GPU between nodes (rule 12: a loop index never reaches WGSL
 * text). `refinementMoved` is set by any cut that moved during the current round and
 * `refinementDone` once a round settled (rule 13: later half-rounds then launch nothing). @internal
 */
export const CLASS_BREAKS_STATE = {
  method: 0,
  classCount: 1,
  layerCursor: 2,
  refinementCursor: 3,
  refinementMoved: 4,
  refinementDone: 5
} as const;
/** Number of words of the `state` transient. @internal */
export const CLASS_BREAKS_STATE_LENGTH = 6;
/** Indices into the `extremes` transient. @internal */
export const CLASS_BREAKS_EXTREMES = {
  minimumKey: 0,
  maximumKey: 1,
  validCount: 2,
  finiteCount: 3
} as const;
/**
 * Indices into the `headTail` transient. While a head/tail run is not `done`, `breakCount` equals
 * the index of the current round, so the split kernel reads its round from it. @internal
 */
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
  | {kind: 'head-tail'; methodCode: number; headTail: GraphDataView<'uint32'>};

/** Words of one indirect dispatch triple `(x, y, z)`. */
const DISPATCH_TRIPLE_LENGTH = 3;

/** A GPU indirect-dispatch gate of one node: its condition and the `'indirect'` resource use. @internal */
export type ClassBreaksIndirectGate<Parameters> = {
  condition: GPUCommandGraphNodeCondition<Parameters>;
  extraResources: GraphBufferUse[];
};

/**
 * Returns the indirect gate that dispatches a node from triple `slot` of `dispatch` (rule 13: an
 * unrolled loop with a data-dependent exit launches nothing once its state word says it is done).
 * The writer of the triple must never be gated by it. @internal
 */
export function getClassBreaksIndirectGate<Parameters>(
  id: string,
  dispatch: GraphDataView<'uint32'>,
  slot: number
): ClassBreaksIndirectGate<Parameters> {
  return {
    condition: {
      id: `${id}-gate`,
      source: 'gpu',
      mode: 'indirect',
      buffer: dispatch.buffer,
      byteOffset: dispatch.byteOffset + slot * DISPATCH_TRIPLE_LENGTH * 4
    },
    extraResources: [{buffer: dispatch, usage: 'indirect'}]
  };
}

/**
 * Returns WGSL statements that write triple `slot` of the `dispatch` binding: the bounded layout
 * of `invocationCount` invocations in workgroups of `workgroupSize` when `activeExpression` holds,
 * otherwise zero workgroups. The layout matches `createWGSLKernelNode`'s, so the gated kernel's
 * linear index mapping is unchanged. @internal
 */
export function getClassBreaksDispatchWGSL<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    operation: string;
    slot: number;
    invocationCount: number;
    workgroupSize: number;
    activeExpression: string;
  }
): string {
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.invocationCount, 1),
    props.workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const base = props.slot * DISPATCH_TRIPLE_LENGTH;
  return `dispatch[dispatchOffset + ${base}u] = select(0u, ${layout.x}u, ${props.activeExpression});
  dispatch[dispatchOffset + ${base + 1}u] = ${layout.y}u;
  dispatch[dispatchOffset + ${base + 2}u] = ${layout.z}u;`;
}

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
    /** Optional indirect dispatch gates of the tile and merge kernels (head/tail rounds). */
    indirectGates?: {
      tile: ClassBreaksIndirectGate<Parameters>;
      merge: ClassBreaksIndirectGate<Parameters>;
    };
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, values, mask, state, partials, moments, slot, gate, indirectGates} = props;
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
    ...indirectGates?.tile,
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
    ...indirectGates?.merge,
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

/**
 * Fixed-point steps per coarse bin of the position in which the coarse dynamic program sums
 * moments: a row at bin `b` and fraction `f` of it has position `b * 256 + floor(f * 256)`, so the
 * program sees the true spread inside bins, not a point mass at the centre. The largest position
 * is below 2^19 with 2048 bins. @internal
 */
export const NATURAL_BREAKS_SUBBIN_COUNT = 256;
/**
 * Cells of one natural-breaks refinement window. A power of two, so cell widths scale by exact
 * powers of two. @internal
 */
export const NATURAL_BREAKS_CELL_COUNT = 1024;
/**
 * Words per refinement element:
 * `[count, sumLow, sumHigh, squaresLow, squaresHigh, minimumKey, maximumKey]`. @internal
 */
export const NATURAL_BREAKS_ELEMENT_STRIDE = 7;
/**
 * Most refinement rounds. Each round updates every break once, odd breaks and then even breaks, so
 * a round is two GPU passes over the rows; the rounds stop earlier once settled. @internal
 */
export const NATURAL_BREAKS_MAXIMUM_ROUND_COUNT = 8;
/**
 * Zoom steps every break must have taken (without its window being reset) before a round that
 * moves no cut counts as settled. @internal
 */
export const NATURAL_BREAKS_SETTLED_ZOOM_LEVEL = 2;
/**
 * Bits of the signed fixed-point offset `q` (`|q| <= 2^bits`) in which refinement accumulates
 * moments. The sum of squares of 2^24 rows then stays below 2^60 and the recentred sum below 2^62,
 * inside the unsigned 64-bit accumulators. @internal
 */
export const NATURAL_BREAKS_QUANTIZATION_BITS = 18;
/**
 * Cells on each side of the chosen boundary that the next window of a break keeps, so a window
 * shrinks by `NATURAL_BREAKS_CELL_COUNT / (2 * NATURAL_BREAKS_ZOOM_CELLS)` per round. @internal
 */
export const NATURAL_BREAKS_ZOOM_CELLS = 4;

/**
 * WGSL 64-bit integer helpers (`vec2<u32>` as `(low, high)`, arithmetic modulo 2^64) and the
 * class cost `sum((x - mean)^2)` of integer-valued positions, shared by the coarse dynamic program
 * (positions are bin indices) and the refinement (positions are fixed-point offsets).
 *
 * `getClassCost(count, sum, squares)` takes the exact integer power sums `sum = S1 = sum(x)`
 * (two's complement) and `squares = S2 = sum(x^2)` and returns `S2 - S1^2 / count` without the
 * catastrophic cancellation of evaluating that difference in f32. It recentres on the integer
 * `m = round(S1 / count)`: the f32 mean is within a few 1e-2 of the true one (|mean| < 2^19), so
 * `m` is within 0.65 of it, and
 * - `S1 - m * count` is below 0.65 * count < 2^31 in magnitude, so its low 32 bits are exact;
 * - `S2 - 2 m S1 + m^2 count = sum((x - m)^2)` is computed exactly modulo 2^64 and is below 2^63;
 * - the cost is that value minus `(S1 - m count)^2 / count`, and since `(x - m)^2 >= |x - m|` for
 *   integers, `(S1 - m count)^2 / count <= 0.65 * |S1 - m count| <= 0.65 * (that value)`, so the
 *   difference is at least 0.35 of it: no cancellation, relative error at most a few 2^-24.
 * Only integer operations and correctly rounded f32 products and sums are used, nothing that
 * relies on error-free transforms or on the order of a float reduction (Metal may reassociate).
 *
 * @internal
 */
export const CLASS_BREAKS_WIDE_WGSL = /* wgsl */ `
fn addWide(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x + right.x;
  let carry = select(0u, 1u, low < left.x);
  return vec2<u32>(low, left.y + right.y + carry);
}

fn subtractWide(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x - right.x;
  let borrow = select(0u, 1u, left.x < right.x);
  return vec2<u32>(low, left.y - right.y - borrow);
}

// Full 32 x 32 -> 64 bit product from 16-bit limbs; no partial product exceeds 2^32.
fn multiplyWide(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowProduct = leftLow * rightLow;
  let crossA = leftLow * rightHigh;
  let crossB = leftHigh * rightLow;
  let highProduct = leftHigh * rightHigh;
  let cross = crossA + crossB;
  let crossCarry = select(0u, 1u, cross < crossA);
  let low = lowProduct + ((cross & 0xffffu) << 16u);
  let lowCarry = select(0u, 1u, low < lowProduct);
  return vec2<u32>(low, highProduct + (cross >> 16u) + (crossCarry << 16u) + lowCarry);
}

// Product modulo 2^64.
fn multiplyWideLow(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let product = multiplyWide(left.x, right.x);
  return vec2<u32>(product.x, product.y + left.x * right.y + left.y * right.x);
}

fn wideToUnsignedFloat(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}

// A small negative number has a huge low word in two's complement, so convert the magnitude.
fn wideToSignedFloat(value: vec2<u32>) -> f32 {
  if (bitcast<i32>(value.y) < 0) {
    return -wideToUnsignedFloat(subtractWide(vec2<u32>(0u, 0u), value));
  }
  return wideToUnsignedFloat(value);
}

fn getClassCost(count: u32, sum: vec2<u32>, squares: vec2<u32>) -> f32 {
  if (count == 0u) {
    return 0.0;
  }
  let countFloat = f32(count);
  let center = i32(round(wideToSignedFloat(sum) / countFloat));
  let centerBits = bitcast<u32>(center);
  let centerWide = vec2<u32>(centerBits, select(0u, 0xffffffffu, (center < 0)));
  let centeredSum = bitcast<i32>(sum.x - centerBits * count);
  let crossTerm = multiplyWideLow(centerWide, sum);
  let centerSquared = multiplyWideLow(centerWide, centerWide);
  let centeredSquares = addWide(
    subtractWide(squares, addWide(crossTerm, crossTerm)),
    multiplyWideLow(centerSquared, vec2<u32>(count, 0u))
  );
  let centeredSumFloat = f32(centeredSum);
  let correction = centeredSumFloat * centeredSumFloat / countFloat;
  return max(wideToUnsignedFloat(centeredSquares) - correction, 0.0);
}
`;
