// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Back-to-front depth keys for Gaussian splat ordering.
 *
 * The GPU radix sort consumes these keys directly. {@link packSplatDepthKey} is the CPU reference
 * implementation of the same formulas (used for tests, tiled keys and custom CPU sorts); the CPU
 * `SplatRenderer` does not quantize at all and sorts exact float32 depths, so its ordering can
 * legitimately differ from a quantized GPU key where two depths share a bucket.
 *
 * The distribution of a quantized depth key matters more than its width. Normalized device depth
 * is hyperbolic: at the near/far ratios a geospatial camera uses, almost the whole key range is
 * spent within a few hundred metres of the eye, and everything beyond that collapses into ties
 * that resolve arbitrarily and pop as the camera rotates. Every distribution here is monotone in
 * view-space distance instead, so equal keys mean genuinely equal depth.
 *
 * @remarks
 * No published source measures the visual error of any particular key width or distribution for
 * splats. These modes exist so that question can be answered by measurement in-repo rather than
 * by argument; {@link SPLAT_DEPTH_KEY_BITS} is the default width for the quantized distributions.
 */

/** Distribution used to quantize view-space depth into an unsigned sort key. */
export type SplatDepthKeyMode =
  /** Legacy hyperbolic normalized device depth. Ties at distance under geospatial near/far. */
  | 'ndc'
  /** View-space distance normalized over an explicit range, quantized uniformly. */
  | 'linear'
  /** IEEE half-precision bit pattern of view-space distance: 31745 perceptually spaced buckets. */
  | 'float16'
  /** IEEE single-precision bit pattern of view-space distance: exact 32-bit ordering. */
  | 'float32';

/**
 * Default depth-key width in bits for {@link packSplatDepthKey} and the GPU radix sort.
 *
 * `'float16'` keys are always this wide and `'float32'` keys are always 32 bits; `'linear'` and
 * `'ndc'` keys quantize to this width unless a caller requests another (at most 24 bits).
 */
export const SPLAT_DEPTH_KEY_BITS = 16;

/** Widest supported depth key; the GPU radix sort accepts at most a 32-bit key. */
export const SPLAT_MAXIMUM_DEPTH_KEY_BITS = 32;

/**
 * Widest key the quantized `'linear'` and `'ndc'` distributions can produce exactly.
 *
 * The GPU quantizes in single precision, whose 24-bit significand cannot represent every integer
 * above `2^24`; a wider quantized key would round the far end past the maximum visible key and
 * wrap. The floating-point distributions are bit patterns and are not limited by this.
 */
const SPLAT_MAXIMUM_QUANTIZED_DEPTH_KEY_BITS = 24;

/** Sentinel sorted after every valid key of the same width, marking a culled Gaussian. */
export function getSplatInvalidDepthKey(keyBits: number = SPLAT_DEPTH_KEY_BITS): number {
  // `2 ** keyBits` rather than a shift: `1 << 31` is negative in JavaScript.
  return keyBits >= 32 ? 0xffffffff : 2 ** keyBits - 1;
}

/** Largest key a visible Gaussian may take, one below the culled sentinel. */
export function getSplatMaximumDepthKey(keyBits: number = SPLAT_DEPTH_KEY_BITS): number {
  return getSplatInvalidDepthKey(keyBits) - 1;
}

/**
 * Returns the key width a distribution actually uses for a requested width.
 *
 * The floating-point distributions are bit patterns, not quantizations, so their widths are fixed
 * (16 for `'float16'`, 32 for `'float32'`) whatever is requested. The quantized `'linear'` and
 * `'ndc'` distributions honor the request between 1 and 24 bits; wider requests are clamped to 24
 * because the GPU quantizes in single precision.
 */
export function getSplatDepthKeyBits(
  mode: SplatDepthKeyMode,
  requestedKeyBits: number = SPLAT_DEPTH_KEY_BITS
): number {
  switch (mode) {
    case 'float16':
      return 16;
    case 'float32':
      return 32;
    default:
      return Math.min(
        Math.max(Math.floor(requestedKeyBits), 1),
        SPLAT_MAXIMUM_QUANTIZED_DEPTH_KEY_BITS
      );
  }
}

/**
 * Returns the IEEE 754 half-precision bit pattern of a finite non-negative value.
 *
 * Half-precision bit patterns of non-negative values increase monotonically with the value, and
 * their spacing is proportional to magnitude - which is the property that makes them a good depth
 * key: near geometry is finely separated and distant geometry is coarsely separated, matching how
 * a perspective camera resolves depth in the first place.
 *
 * Values above the half-precision maximum saturate to positive infinity (`0x7c00`), which still
 * sorts after every finite value.
 */
export function packSplatFloat16Bits(value: number): number {
  if (!(value > 0)) {
    return 0;
  }
  if (!Number.isFinite(value)) {
    return 0x7c00;
  }
  const floatBits = getFloat32Bits(value);
  const exponent = (floatBits >>> 23) & 0xff;
  const mantissa = floatBits & 0x7fffff;

  // Overflow, including values that round up to the half-precision maximum exponent.
  if (exponent >= 143) {
    return 0x7c00;
  }
  // Subnormal halves: shift the implicit leading bit back in and round to nearest even. Rounding
  // up out of the subnormal range lands on `0x0400`, which is exactly the smallest normal half.
  if (exponent <= 112) {
    if (exponent < 102) {
      return 0;
    }
    return roundToNearestEven(mantissa | 0x800000, 126 - exponent);
  }

  const halfExponent = exponent - 112;
  const roundedMantissa = roundToNearestEven(mantissa, 13);
  // Rounding can carry into the exponent; the shifted layout absorbs that automatically.
  return ((halfExponent << 10) + roundedMantissa) & 0xffff;
}

/**
 * Quantizes one positive view-space depth into an ascending, back-to-front sortable key.
 *
 * Ascending key order renders back to front, which is what alpha compositing requires. Culled
 * Gaussians should be assigned {@link getSplatInvalidDepthKey} so they sort past every visible row.
 *
 * @param depth View-space distance along the view direction, in world units. For `'ndc'` this is
 * instead the already normalized device depth in `[0, 1]`.
 */
export function packSplatDepthKey(
  depth: number,
  options: {
    mode?: SplatDepthKeyMode;
    keyBits?: number;
    depthMin?: number;
    depthMax?: number;
    tileId?: number;
  } = {}
): number {
  const mode = options.mode ?? 'linear';
  const keyBits = getSplatDepthKeyBits(mode, options.keyBits);
  const maximumKey = getSplatMaximumDepthKey(keyBits);
  let quantizedDepth: number;

  switch (mode) {
    case 'float16':
      quantizedDepth = Math.min(packSplatFloat16Bits(depth), maximumKey);
      break;
    case 'float32':
      quantizedDepth = Math.min(depth > 0 ? getFloat32Bits(depth) : 0, maximumKey);
      break;
    default: {
      const depthMin = options.depthMin ?? 0;
      const depthMax = options.depthMax ?? 1;
      const depthRange = Math.max(depthMax - depthMin, Number.EPSILON);
      const normalizedDepth = Math.min(Math.max((depth - depthMin) / depthRange, 0), 1);
      quantizedDepth = Math.min(Math.round(normalizedDepth * maximumKey), maximumKey);
      break;
    }
  }

  const depthKey = (maximumKey - quantizedDepth) >>> 0;
  const tileId = options.tileId ?? 0;
  if (!tileId) {
    return depthKey;
  }
  if (keyBits + 8 > 32) {
    throw new RangeError('Tiled Gaussian depth keys require at most 24 depth bits');
  }
  return (((tileId & 0xff) << keyBits) | depthKey) >>> 0;
}

const float32Scratch = new Float32Array(1);
const uint32Scratch = new Uint32Array(float32Scratch.buffer);

/** Reinterprets a finite float as its IEEE 754 single-precision bit pattern. */
function getFloat32Bits(value: number): number {
  float32Scratch[0] = value;
  return uint32Scratch[0];
}

/** Rounds a mantissa right by `shift` bits, breaking exact halves toward an even result. */
function roundToNearestEven(mantissa: number, shift: number): number {
  const truncated = mantissa >>> shift;
  const remainder = mantissa & ((1 << shift) - 1);
  const half = 1 << (shift - 1);
  if (remainder > half || (remainder === half && (truncated & 1) === 1)) {
    return truncated + 1;
  }
  return truncated;
}

/**
 * Shared WGSL depth-key helpers matching {@link packSplatDepthKey}.
 *
 * `SPLAT_DEPTH_KEY_MODE_*` are compile-time constants rather than a uniform branch so a scene that
 * never changes mode compiles to one distribution.
 *
 * @internal
 */
export const SPLAT_DEPTH_KEY_WGSL = /* wgsl */ `\
const SPLAT_DEPTH_KEY_MODE_NDC: u32 = 0u;
const SPLAT_DEPTH_KEY_MODE_LINEAR: u32 = 1u;
const SPLAT_DEPTH_KEY_MODE_FLOAT16: u32 = 2u;
const SPLAT_DEPTH_KEY_MODE_FLOAT32: u32 = 3u;

// Largest finite half-precision value. \`pack2x16float\` of anything larger is indeterminate in
// WGSL, so depths are clamped to it before packing.
const SPLAT_MAXIMUM_FLOAT16_DEPTH: f32 = 65504.0;
// Smallest value that rounds to half-precision infinity. At and above it the CPU path saturates to
// \`0x7c00\`, and the GPU path reproduces that bit pattern explicitly so the two keys agree.
const SPLAT_FLOAT16_OVERFLOW_DEPTH: f32 = 65520.0;
const SPLAT_FLOAT16_INFINITY_BITS: u32 = 0x7c00u;

/** Quantizes a normalized depth uniformly, never exceeding \`maximumKey\`. */
fn quantizeSplatNormalizedDepth(normalizedDepth: f32, maximumKey: u32) -> u32 {
  return min(u32(round(clamp(normalizedDepth, 0.0, 1.0) * f32(maximumKey))), maximumKey);
}

/**
 * Packs one back-to-front sort key.
 *
 * \`viewDepth\` is the perspective divisor, which for a standard projection is the view-space
 * distance along the view direction; \`normalizedDeviceDepth\` is consulted in NDC mode and for
 * affine projections.
 *
 * An affine (orthographic) projection has a constant divisor, so \`viewDepth\` carries no ordering
 * at all. Its normalized device depth is instead linear in view distance, so every distribution
 * quantizes that uniformly across the full key width - the correct spacing when a projection does
 * not foreshorten. \`depthMin\` and \`depthMax\` are then unused; the near and far planes bound
 * the range instead.
 */
fn packSplatDepthKey(
  mode: u32,
  maximumKey: u32,
  viewDepth: f32,
  normalizedDeviceDepth: f32,
  depthMin: f32,
  depthMax: f32,
  affineProjection: bool
) -> u32 {
  var quantized: u32 = 0u;
  if (affineProjection || mode == SPLAT_DEPTH_KEY_MODE_NDC) {
    quantized = quantizeSplatNormalizedDepth(normalizedDeviceDepth, maximumKey);
  } else if (mode == SPLAT_DEPTH_KEY_MODE_FLOAT32) {
    quantized = min(bitcast<u32>(max(viewDepth, 0.0)), maximumKey);
  } else if (mode == SPLAT_DEPTH_KEY_MODE_FLOAT16) {
    // The low half of a packed pair is the half-precision bit pattern of the first component,
    // which is monotone in the value for every non-negative input up to the half maximum.
    var halfBits =
      pack2x16float(vec2<f32>(clamp(viewDepth, 0.0, SPLAT_MAXIMUM_FLOAT16_DEPTH), 0.0)) & 0xffffu;
    if (viewDepth >= SPLAT_FLOAT16_OVERFLOW_DEPTH) {
      halfBits = SPLAT_FLOAT16_INFINITY_BITS;
    }
    quantized = min(halfBits, maximumKey);
  } else {
    let range = max(depthMax - depthMin, 1e-6);
    quantized = quantizeSplatNormalizedDepth((viewDepth - depthMin) / range, maximumKey);
  }
  return maximumKey - quantized;
}

/**
 * Whether a column-major projection matrix is affine, i.e. its clip-space \`w\` is constant.
 *
 * Orthographic projections are affine; perspective projections are not.
 */
fn isSplatAffineProjection(matrix: mat4x4<f32>) -> bool {
  return matrix[0][3] == 0.0 && matrix[1][3] == 0.0 && matrix[2][3] == 0.0;
}
`;
