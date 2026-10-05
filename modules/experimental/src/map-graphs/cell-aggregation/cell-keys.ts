// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Discrete global grid families keyed by {@link GPUCellAggregation}. */
export type GPUCellFamily = 'quadbin' | 'h3';

/** Finest resolution supported per family. */
export const GPU_CELL_MAXIMUM_RESOLUTION: Readonly<Record<GPUCellFamily, number>> = {
  quadbin: 26,
  h3: 15
};

/**
 * Bit layout of the hierarchical part of a 64-bit cell key at one resolution.
 *
 * Both Quadbin and H3 store the cell path as a contiguous run of bits that ends at bit 52: the
 * Morton-interleaved tile coordinates for Quadbin, and the base cell followed by one 3-bit digit
 * per resolution for H3. Bits below the path are all ones (Quadbin footer, H3 unused digit 7).
 * The path, read as an integer, is the cell's "compact key": ascending compact keys are ascending
 * cell keys, and the parent's compact key is the child's shifted right by `bitsPerLevel` per level.
 *
 * @internal
 */
export type CellKeyLayout = {
  family: GPUCellFamily;
  resolution: number;
  /** Lowest bit of the path in the 64-bit key. */
  lowBit: number;
  /** Path bit count, `52 - lowBit`. Valid compact keys are below `2 ** width`. */
  width: number;
  /** Path bits added per resolution step: 2 for Quadbin, 3 for H3. */
  bitsPerLevel: number;
  /** Fixed high word bits shared by every valid key of the family (header and mode). */
  headerHigh: number;
};

/** Returns the key bit layout of `family` at `resolution`. @internal */
export function getCellKeyLayout(family: GPUCellFamily, resolution: number): CellKeyLayout {
  const maximum = GPU_CELL_MAXIMUM_RESOLUTION[family];
  if (!Number.isInteger(resolution) || resolution < 0 || resolution > maximum) {
    throw new Error(`${family} resolution must be an integer in [0, ${maximum}]`);
  }
  const lowBit = family === 'quadbin' ? 52 - 2 * resolution : 45 - 3 * resolution;
  return {
    family,
    resolution,
    lowBit,
    width: 52 - lowBit,
    bitsPerLevel: family === 'quadbin' ? 2 : 3,
    headerHigh: family === 'quadbin' ? 0x48000000 : 0x08000000
  };
}

/**
 * Two-word (`vec2u(high, low)`) helpers shared by the cell kernels: shifts, masks, compact key
 * extraction and key reconstruction. Storage rows use little-endian `(low, high)` words like Arrow
 * `Uint64` columns and the gpu-dggs projections.
 *
 * @internal
 */
export const CELL_KEY_WGSL = /* wgsl */ `
fn cellMaskLow32(bits: u32) -> u32 {
  return select(0xffffffffu, (1u << bits) - 1u, bits < 32u);
}

fn cellShiftRight(value: vec2u, shift: u32) -> vec2u {
  if (shift == 0u) {
    return value;
  }
  if (shift >= 64u) {
    return vec2u(0u);
  }
  if (shift >= 32u) {
    return vec2u(0u, value.x >> (shift - 32u));
  }
  return vec2u(value.x >> shift, (value.y >> shift) | (value.x << (32u - shift)));
}

fn cellShiftLeft(value: vec2u, shift: u32) -> vec2u {
  if (shift == 0u) {
    return value;
  }
  if (shift >= 64u) {
    return vec2u(0u);
  }
  if (shift >= 32u) {
    return vec2u(value.y << (shift - 32u), 0u);
  }
  return vec2u((value.x << shift) | (value.y >> (32u - shift)), value.y << shift);
}

fn cellMaskLow(bits: u32) -> vec2u {
  if (bits >= 32u) {
    return vec2u(cellMaskLow32(bits - 32u), 0xffffffffu);
  }
  return vec2u(0u, cellMaskLow32(bits));
}

/** Path bits [lowBit, lowBit + width) of a canonical key, as an integer. */
fn cellGetCompactKey(key: vec2u, lowBit: u32, width: u32) -> vec2u {
  return cellShiftRight(key, lowBit) & cellMaskLow(width);
}

/** Rebuilds a canonical key from its compact key, resolution and family header. */
fn cellGetKey(compact: vec2u, headerHigh: u32, resolution: u32, lowBit: u32) -> vec2u {
  let header = vec2u(headerHigh | (resolution << 20u), 0u);
  return header | cellShiftLeft(compact, lowBit) | cellMaskLow(lowBit);
}

/** Quadbin validity, mirroring quadbin-py \`is_valid_cell\`. */
fn cellIsValidQuadbin(key: vec2u) -> bool {
  let resolution = (key.x >> 20u) & 0x1fu;
  if ((key.x & 0x80000000u) != 0u || (key.x & 0x40000000u) == 0u ||
      ((key.x >> 27u) & 7u) != 1u || resolution > 26u) {
    return false;
  }
  let unused = cellMaskLow(52u - 2u * resolution);
  return all((key & unused) == unused);
}
`;

/** Rounds `value * 2^bits` to the nearest integer. */
function toFixedPoint(value: number, bits: number): number {
  return Math.round(value * 2 ** bits);
}

/** Inverse factorial `1 / n!` in Q0.32. */
function getInverseFactorial(n: number): number {
  let factorial = 1;
  for (let k = 2; k <= n; k++) {
    factorial *= k;
  }
  return toFixedPoint(1 / factorial, 32);
}

/**
 * Integer constants of the fixed-point Quadbin row computation, shared with the CPU reference.
 *
 * @internal
 */
export const QUADBIN_FIXED_POINT_CONSTANTS = {
  /** f32 bits of the Mercator latitude limit 85.051129 (quadbin-py `clip_latitude`). */
  maximumLatitudeBits: new Uint32Array(new Float32Array([85.051129]).buffer)[0],
  /** `90 * 2^25`: colatitudes are `90 - |latitude|` degrees in Q.25. */
  ninetyDegrees: 90 * 2 ** 25,
  /** `pi / 360` in Q.38, so `t = colatitude * scale >> 31` is the half colatitude in Q0.32. */
  halfColatitudeScale: toFixedPoint(Math.PI / 360, 38),
  /** `ln 2` in Q0.32. */
  ln2: toFixedPoint(Math.LN2, 32),
  /** `1 / (2 pi)` in Q.34. */
  inverseTwoPi: toFixedPoint(1 / (2 * Math.PI), 34),
  /** `sin(t) = t - t^3 b`, `b = 1/3! - t^2 (1/5! - t^2 (...))`, innermost first, Q0.32. */
  sinCoefficients: [11, 9, 7, 5, 3].map(getInverseFactorial),
  /** `cos(t) = 1 - t^2 b`, `b = 1/2! - t^2 (1/4! - ...)`, innermost first, Q0.32. */
  cosCoefficients: [12, 10, 8, 6, 4, 2].map(getInverseFactorial),
  /** `atanh(z) = z + z^3 b`, `b = 1/3 + z^2 (1/5 + ...)`, innermost first, Q0.32. */
  atanhCoefficients: [19, 17, 15, 13, 11, 9, 7, 5, 3].map(n => toFixedPoint(1 / n, 32))
} as const;

const Q = QUADBIN_FIXED_POINT_CONSTANTS;

function getConstantArraySource(name: string, values: readonly number[]): string {
  return `const ${name} = array<u32, ${values.length}>(${values.map(value => `${value}u`).join(', ')});`;
}

/**
 * Longitude/latitude to Quadbin tile column, row and Morton path with integer arithmetic only.
 *
 * WebGPU compilers may contract `a * b + c` into a fused multiply-add (Metal does), so f32
 * polynomials differ between devices in the last bit. Every step here is u32 integer arithmetic
 * (64-bit products from 16-bit halves, exact long division), so the result is identical on every
 * device and in the BigInt CPU reference.
 *
 * - Column: exact `floor((lng + 180) * 2^z / 360) mod 2^z` of the f32 longitude clipped to
 *   [-180, 180], from its mantissa and exponent.
 * - Row: `floor(2^z * (0.5 - psi / (2 pi)))` with `psi = ln(cos t) - ln(sin t)` (the Mercator
 *   ordinate, without the `1 - sin(phi)` cancellation), `t = (90 - |lat|) * pi / 360`, latitude
 *   clipped to ±85.051129 and truncated to 2^-25 degree. `sin` and `cos` are Taylor polynomials in
 *   Q0.32 and `ln x = -2 atanh((1 - x) / (1 + x)) - k ln 2` after normalizing by leading zeros.
 *   The sign comes from the input bits, so zero, subnormal, and tiny latitudes land on the
 *   correct side of the equator. The absolute error of `psi / (2 pi)` is a few 2^-32.
 *
 * @internal
 */
export const QUADBIN_TILE_WGSL = /* wgsl */ `
const QUADBIN_LONGITUDE_LIMIT_BITS: u32 = 0x43340000u;
const QUADBIN_LATITUDE_LIMIT_BITS: u32 = ${Q.maximumLatitudeBits}u;
const QUADBIN_NINETY_DEGREES: u32 = ${Q.ninetyDegrees}u;
const QUADBIN_HALF_COLATITUDE_SCALE: u32 = ${Q.halfColatitudeScale}u;
const QUADBIN_LN2: u32 = ${Q.ln2}u;
const QUADBIN_INVERSE_TWO_PI: u32 = ${Q.inverseTwoPi}u;
${getConstantArraySource('QUADBIN_SIN_COEFFICIENTS', Q.sinCoefficients)}
${getConstantArraySource('QUADBIN_COS_COEFFICIENTS', Q.cosCoefficients)}
${getConstantArraySource('QUADBIN_ATANH_COEFFICIENTS', Q.atanhCoefficients)}

/** Full 64-bit product of two u32 values as vec2u(high, low). */
fn quadbinMultiply(a: u32, b: u32) -> vec2u {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let middle = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  return vec2u(a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (middle >> 16u), (middle << 16u) | (p00 & 0xffffu));
}

/** floor(a * b / 2^32). */
fn quadbinMultiplyHigh(a: u32, b: u32) -> u32 {
  return quadbinMultiply(a, b).x;
}

/** floor(numerator / divisor) for a 64-bit numerator whose quotient fits in 32 bits. */
fn quadbinDivide(numerator: vec2u, divisor: u32) -> u32 {
  var remainder = numerator.x;
  var quotient = 0u;
  for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
    let carry = remainder >> 31u;
    remainder = (remainder << 1u) | ((numerator.y >> u32(bit)) & 1u);
    if (carry != 0u || remainder >= divisor) {
      remainder = remainder - divisor;
      quotient = quotient | (1u << u32(bit));
    }
  }
  return quotient;
}

/** floor(value / 360) for a 64-bit value below 2^41 whose quotient fits in 26 bits. */
fn quadbinDivide360(value: vec2u) -> u32 {
  // Divide by 8 exactly, then by 45 with 16-bit long division.
  let shifted = cellShiftRight(value, 3u);
  var remainder = shifted.x % 45u;
  var part = (remainder << 16u) | (shifted.y >> 16u);
  let quotientHigh = part / 45u;
  remainder = part % 45u;
  part = (remainder << 16u) | (shifted.y & 0xffffu);
  return (quotientHigh << 16u) | (part / 45u);
}

fn quadbinGetTileX(longitude: f32, resolution: u32) -> u32 {
  var bits = bitcast<u32>(longitude);
  if ((bits & 0x7fffffffu) > QUADBIN_LONGITUDE_LIMIT_BITS) {
    bits = (bits & 0x80000000u) | QUADBIN_LONGITUDE_LIMIT_BITS;
  }
  let isNegative = (bits >> 31u) != 0u;
  let exponentBits = (bits >> 23u) & 0xffu;
  var mantissa = bits & 0x7fffffu;
  var exponent = -149i;
  if (exponentBits != 0u) {
    mantissa = mantissa | 0x800000u;
    exponent = i32(exponentBits) - 150i;
  }
  // longitude * 2^z = mantissa * 2^shift; split into an integer part and a nonzero-fraction flag.
  let shift = exponent + i32(resolution);
  var integerPart = vec2u(0u);
  var hasFraction = false;
  if (shift >= 0i) {
    integerPart = cellShiftLeft(vec2u(0u, mantissa), u32(shift));
  } else if (shift > -32i) {
    integerPart = vec2u(0u, mantissa >> u32(-shift));
    hasFraction = (mantissa & ((1u << u32(-shift)) - 1u)) != 0u;
  } else {
    hasFraction = mantissa != 0u;
  }
  // numerator = 180 * 2^z + longitude * 2^z, rounded down to an integer (exact for the floor).
  let offset = cellShiftLeft(vec2u(0u, 180u), resolution);
  var numerator: vec2u;
  if (isNegative) {
    var subtrahend = integerPart;
    if (hasFraction) {
      subtrahend = vec2u(subtrahend.x + select(0u, 1u, subtrahend.y == 0xffffffffu), subtrahend.y + 1u);
    }
    numerator = vec2u(offset.x - subtrahend.x - select(0u, 1u, offset.y < subtrahend.y), offset.y - subtrahend.y);
  } else {
    let low = offset.y + integerPart.y;
    numerator = vec2u(offset.x + integerPart.x + select(0u, 1u, low < offset.y), low);
  }
  let tileCount = 1u << resolution;
  return quadbinDivide360(numerator) & (tileCount - 1u);
}

/** floor(|latitude| * 2^25) of an f32 magnitude clipped to the Mercator limit. */
fn quadbinGetLatitudeFixed(magnitudeBits: u32) -> u32 {
  let bits = min(magnitudeBits, QUADBIN_LATITUDE_LIMIT_BITS);
  let exponentBits = bits >> 23u;
  var mantissa = bits & 0x7fffffu;
  if (exponentBits != 0u) {
    mantissa = mantissa | 0x800000u;
  }
  let shift = i32(max(exponentBits, 1u)) - 125i;
  if (shift >= 0i) {
    return mantissa << u32(shift);
  }
  if (shift > -32i) {
    return mantissa >> u32(-shift);
  }
  return 0u;
}

/** sin(t) in Q0.32 for t in [0, pi/4] in Q0.32. */
fn quadbinSin(t: u32) -> u32 {
  let t2 = quadbinMultiplyHigh(t, t);
  var bracket = QUADBIN_SIN_COEFFICIENTS[0];
  for (var index = 1u; index < ${Q.sinCoefficients.length}u; index++) {
    bracket = QUADBIN_SIN_COEFFICIENTS[index] - quadbinMultiplyHigh(t2, bracket);
  }
  return t - quadbinMultiplyHigh(t, quadbinMultiplyHigh(t2, bracket));
}

/** cos(t) in Q0.32 for t in (0, pi/4] in Q0.32. */
fn quadbinCos(t: u32) -> u32 {
  let t2 = quadbinMultiplyHigh(t, t);
  var bracket = QUADBIN_COS_COEFFICIENTS[0];
  for (var index = 1u; index < ${Q.cosCoefficients.length}u; index++) {
    bracket = QUADBIN_COS_COEFFICIENTS[index] - quadbinMultiplyHigh(t2, bracket);
  }
  return 0u - quadbinMultiplyHigh(t2, bracket);
}

/** -ln(v) * 2^32 for v in (0, 1) in Q0.32, as a 64-bit value vec2u(high, low). */
fn quadbinNegativeLn(v: u32) -> vec2u {
  let leadingZeros = countLeadingZeros(v);
  let m = v << leadingZeros;
  // x = m / 2^32 in [0.5, 1); z = (1 - x) / (1 + x) in (0, 1/3] in Q0.32.
  let difference = 0u - m;
  let z = quadbinDivide(vec2u(difference >> 1u, difference << 31u), 0x80000000u + (m >> 1u));
  let z2 = quadbinMultiplyHigh(z, z);
  var bracket = QUADBIN_ATANH_COEFFICIENTS[0];
  for (var index = 1u; index < ${Q.atanhCoefficients.length}u; index++) {
    bracket = QUADBIN_ATANH_COEFFICIENTS[index] + quadbinMultiplyHigh(z2, bracket);
  }
  let atanhZ = z + quadbinMultiplyHigh(z, quadbinMultiplyHigh(z2, bracket));
  // -ln(v) = 2 atanh(z) + leadingZeros * ln 2.
  let twice = vec2u(atanhZ >> 31u, atanhZ << 1u);
  let logs = quadbinMultiply(leadingZeros, QUADBIN_LN2);
  let low = twice.y + logs.y;
  return vec2u(twice.x + logs.x + select(0u, 1u, low < twice.y), low);
}

fn quadbinGetTileY(latitude: f32, resolution: u32) -> u32 {
  if (resolution == 0u) {
    return 0u;
  }
  let bits = bitcast<u32>(latitude);
  let magnitudeBits = bits & 0x7fffffffu;
  let half = i32(1u << (resolution - 1u));
  var row = half;
  if (magnitudeBits != 0u) {
    let colatitude = QUADBIN_NINETY_DEGREES - quadbinGetLatitudeFixed(magnitudeBits);
    let t = cellShiftRight(quadbinMultiply(colatitude, QUADBIN_HALF_COLATITUDE_SCALE), 31u).y;
    // psi = -ln(sin t) - (-ln(cos t)) >= 0 in Q.32, clamped at zero near the equator.
    let sinPart = quadbinNegativeLn(quadbinSin(t));
    let cosPart = quadbinNegativeLn(quadbinCos(t));
    var psi = vec2u(0u);
    if (sinPart.x > cosPart.x || (sinPart.x == cosPart.x && sinPart.y > cosPart.y)) {
      psi = vec2u(sinPart.x - cosPart.x - select(0u, 1u, sinPart.y < cosPart.y), sinPart.y - cosPart.y);
    }
    // fraction = psi / (2 pi) in Q0.32; offset = fraction * 2^z.
    let fraction = quadbinMultiplyHigh(cellShiftRight(psi, 2u).y, QUADBIN_INVERSE_TWO_PI);
    let fractionBits = 32u - resolution;
    let whole = i32(fraction >> fractionBits);
    let hasFraction = (fraction & ((1u << fractionBits) - 1u)) != 0u;
    if ((bits >> 31u) != 0u) {
      row = half + whole;
    } else {
      row = half - max(whole + select(0i, 1i, hasFraction), 1i);
    }
  }
  return u32(clamp(row, 0i, i32((1u << resolution) - 1u)));
}

fn quadbinSpread16(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}

/** Morton path (compact key) of tile (x, y): x bits at even, y bits at odd positions. */
fn quadbinGetCompactKey(x: u32, y: u32) -> vec2u {
  let low = quadbinSpread16(x) | (quadbinSpread16(y) << 1u);
  let high = quadbinSpread16(x >> 16u) | (quadbinSpread16(y >> 16u) << 1u);
  return vec2u(high, low);
}
`;
