// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact CPU model of `src/utils/fixed-point-accumulator.ts` with BigInt arithmetic: the same
 * data-derived unit, the same round-to-nearest-even quantization and decoding. GPU sums must
 * equal it bit for bit.
 */

const FLOAT_VIEW = new Float32Array(1);
const BITS_VIEW = new Uint32Array(FLOAT_VIEW.buffer);

function getFloatBits(value: number): number {
  FLOAT_VIEW[0] = value;
  return BITS_VIEW[0];
}

/** Bit length of `maximumTermCount` (at least 1), as `getFixedPointHeadroomBits`. */
export function getOracleHeadroomBits(maximumTermCount: number): number {
  return Math.max(1, maximumTermCount.toString(2).length);
}

/** Largest biased exponent of the finite nonzero values (1 for subnormals), 0 when none. */
export function getOracleMaximumExponentBits(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < values.length; index++) {
    const magnitudeBits = getFloatBits(values[index]) & 0x7fffffff;
    if (magnitudeBits !== 0 && magnitudeBits < 0x7f800000) {
      maximum = Math.max(maximum, Math.max(magnitudeBits >>> 23, 1));
    }
  }
  return maximum;
}

/** `getFixedPointUnitExponent`. */
export function getOracleUnitExponent(maximumExponentBits: number, headroomBits: number): number {
  return Math.max(maximumExponentBits, 1) - 127 + 1 + headroomBits - 95;
}

function roundShiftRight(value: bigint, shift: number): bigint {
  if (shift <= 0) {
    return value << BigInt(-shift);
  }
  const divisor = 1n << BigInt(shift);
  const quotient = value >> BigInt(shift);
  const remainder = value & (divisor - 1n);
  const half = divisor >> 1n;
  return remainder > half || (remainder === half && (quotient & 1n) === 1n)
    ? quotient + 1n
    : quotient;
}

/** `quantizeFixedPoint`: the value in units of `2^unitExponent`, ties to even; non-finite is 0. */
export function quantizeOracle(value: number, unitExponent: number): bigint {
  const bits = getFloatBits(value);
  const exponentBits = (bits >>> 23) & 0xff;
  if (exponentBits === 0xff) {
    return 0n;
  }
  let significand = BigInt(bits & 0x7fffff);
  let exponent = -149;
  if (exponentBits !== 0) {
    significand |= 0x800000n;
    exponent = exponentBits - 150;
  }
  const magnitude = roundShiftRight(significand, unitExponent - exponent);
  return bits >>> 31 ? -magnitude : magnitude;
}

/** `getFixedPointFloat`: the nearest f32 of `sum * 2^unitExponent`, ties to even. */
export function decodeOracle(sum: bigint, unitExponent: number): number {
  if (sum === 0n) {
    return 0;
  }
  const negative = sum < 0n;
  const magnitude = negative ? -sum : sum;
  const shift = Math.max(0, magnitude.toString(2).length - 24);
  const significand = roundShiftRight(magnitude, shift);
  const value = Math.fround(Number(significand) * 2 ** (unitExponent + shift));
  return negative ? -value : value;
}

/** Fixed-point sums of `values` per key (keys at or above `cellCount` are skipped). */
export function sumFixedPointOracle(
  values: ArrayLike<number>,
  keys: ArrayLike<number>,
  cellCount: number,
  maximumTermCount = values.length
): {sums: number[]; unitExponent: number} {
  const unitExponent = getOracleUnitExponent(
    getOracleMaximumExponentBits(values),
    getOracleHeadroomBits(maximumTermCount)
  );
  const totals = new Array<bigint>(cellCount).fill(0n);
  for (let index = 0; index < values.length; index++) {
    if (keys[index] < cellCount) {
      totals[keys[index]] += quantizeOracle(values[index], unitExponent);
    }
  }
  return {sums: totals.map(total => decodeOracle(total, unitExponent)), unitExponent};
}
