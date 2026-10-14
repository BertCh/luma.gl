// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';

/**
 * Shared deterministic fixed-point accumulator (algorithms roadmap decision 4).
 *
 * A sum is held as one 96-bit two's-complement integer per cell (three `u32` words, low word
 * first) in units of `2^unitExponent`. Integer addition is associative, so the sum is the same
 * for every add order, workgroup split, chunking and atomic interleaving: results are bitwise
 * reproducible across runs and dispatch orders.
 *
 * The unit is data-derived but quantized to a power of two, in the style of ReproBLAS binned
 * summation: `unitExponent = E + 1 + h - 95`, where `E` is the largest binary exponent of any term
 * (an order-independent maximum, reduced on the GPU into one scale word) and `h` is the bit length
 * of the maximum number of terms per cell. Then every term is below `2^(95 - h)` units and no cell
 * can overflow. A term keeps all 24 significand bits while its magnitude is at least
 * `max * 2^(h - 71)`, a dynamic range of `2^(71 - h)` (about `1e15` at a million terms), which is
 * why one outlier no longer zeroes typical terms (library rule 7, review B-01, which pegged a
 * 24-bit unit to the maximum). Smaller terms round to the nearest unit (ties to even); the
 * absolute error of a sum of `n` terms is at most `n / 2` units.
 *
 * Quantizing and decoding use integer bit manipulation only (no f32 arithmetic before the final
 * `ldexp`), so the result does not depend on fused multiply-add or reassociation. Decoding rounds
 * the exact integer to the nearest f32 (ties to even); sums below `2^-126` may flush to zero and
 * sums above the f32 range decode to `±Infinity`.
 *
 * @internal
 */
export const FIXED_POINT_WORD_COUNT = 3;

/** Largest number of terms per cell a fixed-point sum supports (`h <= 32`). @internal */
export const FIXED_POINT_MAXIMUM_TERM_COUNT = 0xffffffff;

/**
 * Returns `h`, the headroom bits of a fixed-point sum of at most `maximumTermCount` terms per
 * cell: the bit length of `maximumTermCount` (at least 1).
 *
 * @internal
 */
export function getFixedPointHeadroomBits(maximumTermCount: number): number {
  if (
    !Number.isSafeInteger(maximumTermCount) ||
    maximumTermCount < 0 ||
    maximumTermCount > FIXED_POINT_MAXIMUM_TERM_COUNT
  ) {
    throw new Error('fixed-point maximumTermCount must be an integer in [0, 2^32 - 1]');
  }
  return Math.max(1, maximumTermCount.toString(2).length);
}

/**
 * WGSL helpers of the fixed-point accumulator. A value is a `vec3<u32>` (`x` low, `z` high word,
 * two's complement over 96 bits). Requires no bindings.
 *
 * - `getFixedPointExponentBits(value: f32) -> u32`: the biased exponent of a finite nonzero
 *   value (1 for subnormals), 0 for zero and non-finite values. Reduce it with `atomicMax` into
 *   the scale word.
 * - `getFixedPointUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32`.
 * - `quantizeFixedPoint(value: f32, unitExponent: i32) -> vec3<u32>`: round to the nearest unit,
 *   ties to even; non-finite values give 0.
 * - `addFixedPoint(a: vec3<u32>, b: vec3<u32>) -> vec3<u32>` and `negateFixedPoint`.
 * - `getFixedPointFloat(value: vec3<u32>, unitExponent: i32) -> f32`: nearest f32, ties to even.
 *
 * @internal
 */
export const FIXED_POINT_WGSL = /* wgsl */ `
fn getFixedPointExponentBits(value: f32) -> u32 {
  let magnitudeBits = bitcast<u32>(value) & 0x7fffffffu;
  if (magnitudeBits == 0u || magnitudeBits >= 0x7f800000u) { return 0u; }
  return max(magnitudeBits >> 23u, 1u);
}
fn getFixedPointUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32 {
  // A term with biased exponent E is below 2^(E - 126) = 2^(E - 127 + 1); 95 - h bits remain.
  return i32(max(maximumExponentBits, 1u)) - 127 + 1 + i32(headroomBits) - 95;
}
fn addFixedPoint(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
  let low = a.x + b.x;
  let lowCarry = select(0u, 1u, low < a.x);
  let middlePartial = a.y + b.y;
  let middleCarry = select(0u, 1u, middlePartial < a.y);
  let middle = middlePartial + lowCarry;
  let middleRoundCarry = select(0u, 1u, middle < middlePartial);
  return vec3<u32>(low, middle, a.z + b.z + middleCarry + middleRoundCarry);
}
fn negateFixedPoint(value: vec3<u32>) -> vec3<u32> {
  return addFixedPoint(~value, vec3<u32>(1u, 0u, 0u));
}
fn shiftFixedPointLeft(value: u32, shift: u32) -> vec3<u32> {
  var words = array<u32, 4>(0u, 0u, 0u, 0u);
  let word = shift / 32u;
  let bit = shift % 32u;
  words[word] = value << bit;
  // (value >> 1) >> (31 - bit) is value >> (32 - bit) without a shift by 32 when bit is 0.
  words[word + 1u] = (value >> 1u) >> (31u - bit);
  return vec3<u32>(words[0], words[1], words[2]);
}
fn shiftFixedPointRight(value: vec3<u32>, shift: u32) -> u32 {
  var words = array<u32, 4>(value.x, value.y, value.z, 0u);
  let word = shift / 32u;
  let bit = shift % 32u;
  return (words[word] >> bit) | ((words[word + 1u] << 1u) << (31u - bit));
}
fn getFixedPointBit(value: vec3<u32>, position: u32) -> u32 {
  var words = array<u32, 3>(value.x, value.y, value.z);
  return (words[position / 32u] >> (position % 32u)) & 1u;
}
fn hasFixedPointBitsBelow(value: vec3<u32>, count: u32) -> bool {
  var words = array<u32, 3>(value.x, value.y, value.z);
  let word = count / 32u;
  var found = (words[word] & ((1u << (count % 32u)) - 1u)) != 0u;
  if (word > 0u) { found = found || words[0] != 0u; }
  if (word > 1u) { found = found || words[1] != 0u; }
  return found;
}
fn quantizeFixedPoint(value: f32, unitExponent: i32) -> vec3<u32> {
  let bits = bitcast<u32>(value);
  let exponentBits = (bits >> 23u) & 0xffu;
  if (exponentBits == 0xffu) { return vec3<u32>(0u); }
  var significand = bits & 0x7fffffu;
  var exponent = -149;
  if (exponentBits != 0u) {
    significand = significand | 0x800000u;
    exponent = i32(exponentBits) - 150;
  }
  // value = significand * 2^exponent; the term in units is significand * 2^shift.
  let shift = exponent - unitExponent;
  var magnitude = vec3<u32>(0u);
  if (shift >= 0) {
    // The scale bounds shift by 71 - h; the clamp only keeps a misused scale memory-safe.
    magnitude = shiftFixedPointLeft(significand, u32(min(shift, 71)));
  } else if (shift >= -24) {
    let right = u32(-shift);
    var quotient = significand >> right;
    let remainder = significand & ((1u << right) - 1u);
    let half = 1u << (right - 1u);
    if (remainder > half || (remainder == half && (quotient & 1u) != 0u)) {
      quotient = quotient + 1u;
    }
    magnitude = vec3<u32>(quotient, 0u, 0u);
  }
  return select(magnitude, negateFixedPoint(magnitude), (bits >> 31u) != 0u);
}
fn getFixedPointFloat(value: vec3<u32>, unitExponent: i32) -> f32 {
  let negative = (value.z >> 31u) != 0u;
  let magnitude = select(value, negateFixedPoint(value), negative);
  var leading = 0u;
  if (magnitude.z != 0u) {
    leading = 64u + firstLeadingBit(magnitude.z);
  } else if (magnitude.y != 0u) {
    leading = 32u + firstLeadingBit(magnitude.y);
  } else if (magnitude.x != 0u) {
    leading = firstLeadingBit(magnitude.x);
  } else {
    return 0.0;
  }
  var significand = magnitude.x;
  var exponent = unitExponent;
  if (leading >= 24u) {
    let shift = leading - 23u;
    significand = shiftFixedPointRight(magnitude, shift);
    let roundBit = getFixedPointBit(magnitude, shift - 1u);
    if (roundBit != 0u && (hasFixedPointBitsBelow(magnitude, shift - 1u) || (significand & 1u) != 0u)) {
      significand = significand + 1u;
    }
    exponent = unitExponent + i32(shift);
  }
  // significand <= 2^24, so f32(significand) is exact and the value is significand * 2^exponent.
  var infinityBits = 0x7f800000u;
  var result = bitcast<f32>(infinityBits);
  if (exponent < 104 || (exponent == 104 && significand < 0x1000000u)) {
    // Two steps keep the ldexp exponent in range for normal results below 2^-103.
    let leadingExponent = i32(firstLeadingBit(significand));
    result = ldexp(ldexp(f32(significand), -leadingExponent), exponent + leadingExponent);
  }
  return select(result, -result, negative);
}`;

/**
 * Returns WGSL `fn addFixedPointTo<Name>(cell: u32, value: vec3<u32>)` that adds a fixed-point
 * value to cell `cell` of the `atomic<u32>` storage binding `bindingName` (three words per cell,
 * the binding's `${bindingName}Offset` applied). Each word takes one `atomicAdd`, skipped when
 * its addend is 0; the carry out of a word is detected from the returned old value. The final
 * words are the exact 96-bit sum whatever the interleaving (intermediate states are not).
 *
 * @internal
 */
export function getFixedPointAtomicAddWGSL(bindingName: string): string {
  const name = `addFixedPointTo${bindingName[0].toUpperCase()}${bindingName.slice(1)}`;
  return /* wgsl */ `
fn ${name}(cell: u32, value: vec3<u32>) {
  let base = ${bindingName}Offset + cell * ${FIXED_POINT_WORD_COUNT}u;
  var lowCarry = 0u;
  if (value.x != 0u) {
    let oldLow = atomicAdd(&${bindingName}[base], value.x);
    lowCarry = select(0u, 1u, oldLow + value.x < oldLow);
  }
  let middleAddend = value.y + lowCarry;
  var middleCarry = select(0u, 1u, middleAddend < value.y);
  if (middleAddend != 0u) {
    let oldMiddle = atomicAdd(&${bindingName}[base + 1u], middleAddend);
    middleCarry = middleCarry + select(0u, 1u, oldMiddle + middleAddend < oldMiddle);
  }
  let highAddend = value.z + middleCarry;
  if (highAddend != 0u) {
    atomicAdd(&${bindingName}[base + 2u], highAddend);
  }
}`;
}

/**
 * Builds the scale word of a fixed-point sum: one clear node, then per chunk of `values` one
 * workgroup max of {@link FIXED_POINT_WGSL}'s `getFixedPointExponentBits` and one `atomicMax`
 * per workgroup (library rule 14). Non-finite values do not raise the scale. Consumers whose
 * terms are computed in-kernel write the same word with their own `atomicMax` of the exponent
 * bits of their terms.
 *
 * @internal
 */
export function createFixedPointScaleNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    /** Operation name reported in workload estimates. */
    operation: string;
    /** Terms whose largest magnitude sets the unit. */
    values: GraphDataView<'float32'> | GraphVectorView<'float32'>;
    /** One `uint32` word: the largest biased exponent, 0 when every term is zero or non-finite. */
    scale: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-clear`,
      operation: props.operation,
      variant: 'fixed-point-scale-clear',
      bindings: [{name: 'fixedPointScale', view: props.scale, type: 'u32', access: 'read_write'}],
      invocationCount: 1,
      body: 'fixedPointScale[fixedPointScaleOffset] = 0u;'
    })
  ];
  const chunks = props.values instanceof GraphVectorView ? props.values.data : [props.values];
  for (const [chunkIndex, chunk] of chunks.entries()) {
    if (chunk.length === 0) {
      continue;
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-${chunkIndex}`,
        operation: props.operation,
        variant: 'fixed-point-scale',
        bindings: [
          {name: 'fixedPointValues', view: chunk, type: 'f32', access: 'read'},
          {
            name: 'fixedPointScale',
            view: props.scale,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: chunk.length,
        guardIndex: false,
        declarations: `${FIXED_POINT_WGSL}
var<workgroup> workgroupExponentBits: atomic<u32>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let exponentBits = getFixedPointExponentBits(fixedPointValues[fixedPointValuesOffset + index]);
    if (exponentBits != 0u) {
      atomicMax(&workgroupExponentBits, exponentBits);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&fixedPointScale[fixedPointScaleOffset], workgroupMaximum);
    }
  }`
      })
    );
  }
  return nodes;
}

/** Sets every word of a fixed-point sum array to zero. @internal */
export function createFixedPointClearNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; words: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'fixed-point-clear',
    bindings: [{name: 'fixedPointWords', view: props.words, type: 'u32', access: 'read_write'}],
    invocationCount: props.words.length,
    // A fixed size keeps one module for every cell count (rule 12).
    workgroupSize: 64,
    body: 'fixedPointWords[fixedPointWordsOffset + index] = 0u;'
  });
}

/**
 * Decodes `output.length` fixed-point cells (three words each) into the nearest f32 values.
 * `maximumTermCount` must be the one the accumulating kernels used for their unit.
 *
 * @internal
 */
export function createFixedPointDecodeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    words: GraphDataView<'uint32'>;
    scale: GraphDataView<'uint32'>;
    maximumTermCount: number;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  if (props.words.length < props.output.length * FIXED_POINT_WORD_COUNT) {
    throw new Error(`${props.operation}: fixed-point words must hold three words per output`);
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'fixed-point-decode',
    bindings: [
      {name: 'fixedPointWords', view: props.words, type: 'u32', access: 'read'},
      {name: 'fixedPointScale', view: props.scale, type: 'u32', access: 'read'},
      {name: 'fixedPointOutput', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.output.length,
    workgroupSize: 64,
    declarations: `override FIXED_POINT_HEADROOM_BITS: u32;
${FIXED_POINT_WGSL}`,
    constants: {FIXED_POINT_HEADROOM_BITS: getFixedPointHeadroomBits(props.maximumTermCount)},
    body: `let base = fixedPointWordsOffset + index * ${FIXED_POINT_WORD_COUNT}u;
  let unitExponent = getFixedPointUnitExponent(
    fixedPointScale[fixedPointScaleOffset],
    FIXED_POINT_HEADROOM_BITS
  );
  fixedPointOutput[fixedPointOutputOffset + index] = getFixedPointFloat(
    vec3<u32>(fixedPointWords[base], fixedPointWords[base + 1u], fixedPointWords[base + 2u]),
    unitExponent
  );`
  });
}
