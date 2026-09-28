// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getSplatDepthKeyBits,
  getSplatInvalidDepthKey,
  getSplatMaximumDepthKey,
  packSplatDepthKey,
  packSplatFloat16Bits,
  SPLAT_DEPTH_KEY_BITS,
  SPLAT_DEPTH_KEY_WGSL
} from '../src/splat-depth-key';

it('depth keys default to one 16-bit width with a reserved culled sentinel', () => {
  expect(SPLAT_DEPTH_KEY_BITS, 'packSplatDepthKey and the GPU radix sort default to 16 bits').toBe(
    16
  );
  expect(getSplatInvalidDepthKey(16), 'culled rows take the final key of their width').toBe(0xffff);
  expect(getSplatMaximumDepthKey(16), 'visible rows stop one short of it').toBe(0xfffe);
  expect(getSplatInvalidDepthKey(32), 'a 32-bit width does not overflow the shift').toBe(
    0xffffffff
  );
  expect(getSplatMaximumDepthKey(32), 'nor does its maximum visible key').toBe(0xfffffffe);
  expect(getSplatInvalidDepthKey(31), 'a 31-bit width stays positive').toBe(0x7fffffff);
  expect(getSplatMaximumDepthKey(24), 'and a 24-bit width matches its mask').toBe(0xfffffe);
});

it('quantized distributions stop at 24 bits, the width single precision quantizes exactly', () => {
  expect(getSplatDepthKeyBits('linear', 32), 'a 32-bit linear request is clamped').toBe(24);
  expect(getSplatDepthKeyBits('ndc', 28), 'and so is a wide NDC request').toBe(24);
  expect(getSplatDepthKeyBits('linear', 0), 'with at least one bit').toBe(1);

  const maximumKey = getSplatMaximumDepthKey(24);
  expect(
    packSplatDepthKey(0, {mode: 'linear', keyBits: 32, depthMax: 1}),
    'the nearest depth takes the largest visible key without wrapping'
  ).toBe(maximumKey);
  expect(
    packSplatDepthKey(1, {mode: 'linear', keyBits: 32, depthMax: 1}),
    'and the farthest takes zero'
  ).toBe(0);
  expect(
    packSplatDepthKey(0.5, {mode: 'ndc', keyBits: 32}) < getSplatInvalidDepthKey(24),
    'no quantized key reaches the culled sentinel'
  ).toBe(true);
});

/**
 * JavaScript transcription of the WGSL `'float16'` branch of `packSplatDepthKey`: clamp to the
 * half-precision maximum, pack, then substitute infinity at the overflow threshold.
 * `packSplatFloat16Bits` stands in for `pack2x16float`, whose rounding it reproduces.
 */
function packGPUFloat16DepthKey(viewDepth: number): number {
  const maximumKey = getSplatMaximumDepthKey(16);
  const clamped = Math.min(Math.max(viewDepth, 0), 65504);
  let halfBits = packSplatFloat16Bits(clamped) & 0xffff;
  if (viewDepth >= 65520) {
    halfBits = 0x7c00;
  }
  return maximumKey - Math.min(halfBits, maximumKey);
}

it('the GPU half-precision key never packs an out-of-range depth and matches the CPU key', () => {
  expect(
    SPLAT_DEPTH_KEY_WGSL,
    'WGSL clamps before pack2x16float, whose result is indeterminate past 65504'
  ).toMatch(
    /pack2x16float\(vec2<f32>\(clamp\(viewDepth, 0\.0, SPLAT_MAXIMUM_FLOAT16_DEPTH\), 0\.0\)\)/
  );
  expect(SPLAT_DEPTH_KEY_WGSL, 'at the largest finite half').toMatch(
    /SPLAT_MAXIMUM_FLOAT16_DEPTH: f32 = 65504\.0;/
  );
  expect(SPLAT_DEPTH_KEY_WGSL, 'and substitutes infinity where the CPU path overflows').toMatch(
    /SPLAT_FLOAT16_OVERFLOW_DEPTH: f32 = 65520\.0;/
  );

  const saturated = packSplatDepthKey(1e6, {mode: 'float16'});
  expect(saturated, 'the CPU key saturates to half-precision infinity').toBe(0xfffe - 0x7c00);
  expect(
    saturated < packSplatDepthKey(65504, {mode: 'float16'}),
    'which still sorts before (behind) the farthest finite depth'
  ).toBe(true);

  for (const depth of [
    0, 1, 1000, 30_000, 65_000, 65_504, 65_510, 65_519.99, 65_520, 70_000, 1e6, 1e30
  ]) {
    expect(packGPUFloat16DepthKey(depth), `GPU and CPU formulas agree at ${depth}`).toBe(
      packSplatDepthKey(depth, {mode: 'float16'})
    );
  }
});

it('half-precision keys are coarse at geospatial distances', () => {
  // Bucket width at distance d is 2^(floor(log2 d) - 10): 16 m between 16 and 32 km and 32 m
  // beyond 32 km, and everything past 65.5 km ties.
  const bucketWidth = (depth: number): number => {
    const key = packSplatDepthKey(depth, {mode: 'float16'});
    let far = depth;
    while (packSplatDepthKey(far, {mode: 'float16'}) === key) {
      far += 0.25;
    }
    return far - depth;
  };
  expect(bucketWidth(20_000) <= 16, 'at 20 km one key spans up to 16 m').toBe(true);
  expect(bucketWidth(40_000) <= 32, 'at 40 km up to 32 m').toBe(true);
  expect(packSplatDepthKey(70_000, {mode: 'float16'}), 'and depths past 65.5 km all tie').toBe(
    packSplatDepthKey(100_000, {mode: 'float16'})
  );
});

it('the GPU key falls back to device depth under an affine projection', () => {
  expect(
    SPLAT_DEPTH_KEY_WGSL,
    'orthographic clip w is constant, so every mode quantizes normalized device depth'
  ).toMatch(/if \(affineProjection \|\| mode == SPLAT_DEPTH_KEY_MODE_NDC\)/);
  expect(SPLAT_DEPTH_KEY_WGSL, 'affine means the matrix w row is (0, 0, 0, w)').toMatch(
    /matrix\[0\]\[3\] == 0\.0 && matrix\[1\]\[3\] == 0\.0 && matrix\[2\]\[3\] == 0\.0/
  );
});

it('floating-point distributions fix their own width', () => {
  expect(
    getSplatDepthKeyBits('float16', 32),
    'a half bit pattern is 16 bits, whatever is asked'
  ).toBe(16);
  expect(getSplatDepthKeyBits('float32', 8), 'and a single bit pattern is 32').toBe(32);
  expect(getSplatDepthKeyBits('linear', 24), 'quantized distributions honor the request').toBe(24);
  expect(getSplatDepthKeyBits('ndc'), 'and fall back to the shared default').toBe(
    SPLAT_DEPTH_KEY_BITS
  );
});

it('half-precision bit patterns round-trip and stay monotone', () => {
  expect(packSplatFloat16Bits(1), 'one is the half-precision exponent bias').toBe(0x3c00);
  expect(packSplatFloat16Bits(65504), 'the largest finite half').toBe(0x7bff);
  expect(packSplatFloat16Bits(70000), 'anything larger saturates to infinity').toBe(0x7c00);
  expect(packSplatFloat16Bits(2 ** -24), 'the smallest subnormal half').toBe(1);
  expect(packSplatFloat16Bits(2 ** -25), 'exact halves round to even, which is zero here').toBe(0);
  expect(packSplatFloat16Bits(0), 'and non-positive inputs are zero').toBe(0);
  expect(packSplatFloat16Bits(-5), 'including negatives, which never reach a depth key').toBe(0);

  let previous = -1;
  for (let exponent = -14; exponent <= 15; exponent += 0.001) {
    const bits = packSplatFloat16Bits(2 ** exponent);
    expect(bits >= previous, `monotone at 2^${exponent}`).toBe(true);
    previous = bits;
  }
});

it('half-precision keys spend their range the way a perspective camera resolves depth', () => {
  // The bucket width grows in proportion to distance, so a metre at ten metres is separated far
  // more finely than a metre at ten kilometres - which is what a depth key should do and what
  // uniform quantization over a geospatial range cannot.
  const nearStep =
    packSplatDepthKey(10, {mode: 'float16'}) - packSplatDepthKey(11, {mode: 'float16'});
  const farStep =
    packSplatDepthKey(10000, {mode: 'float16'}) - packSplatDepthKey(10001, {mode: 'float16'});

  expect(nearStep > 0, 'a metre nearer is a strictly larger key').toBe(true);
  expect(farStep >= 0, 'and never inverts at distance').toBe(true);
  expect(nearStep > farStep * 100, 'with three orders of magnitude more resolution up close').toBe(
    true
  );
});

it('hyperbolic device depth spends its whole range within metres of the camera', () => {
  // A geospatial camera: a 1 m near plane and a 100 km far plane. Counting the distinct keys each
  // distribution produces over three slices of that range is the measurement the choice of key
  // actually rests on, and the one no published source provides.
  const near = 1;
  const far = 100_000;
  const getNormalizedDeviceDepth = (distance: number): number =>
    (far / (far - near)) * (1 - near / distance);

  const countDistinctKeys = (
    pack: (distance: number) => number,
    low: number,
    high: number
  ): number => {
    const keys = new Set<number>();
    for (let sample = 0; sample < 20_000; sample++) {
      keys.add(pack(low + ((high - low) * sample) / 19_999));
    }
    return keys.size;
  };

  const deviceKeys = (distance: number): number =>
    packSplatDepthKey(getNormalizedDeviceDepth(distance), {mode: 'ndc', keyBits: 16});
  const linearKeys = (distance: number): number =>
    packSplatDepthKey(distance, {mode: 'linear', keyBits: 16, depthMin: near, depthMax: far});
  const halfKeys = (distance: number): number => packSplatDepthKey(distance, {mode: 'float16'});

  expect(
    countDistinctKeys(deviceKeys, 1, 10) > 10_000,
    'device depth resolves the first ten metres into more than ten thousand keys'
  ).toBe(true);
  expect(
    countDistinctKeys(deviceKeys, 50_000, 100_000) <= 4,
    'and the last fifty kilometres into no more than a handful'
  ).toBe(true);
  expect(
    countDistinctKeys(linearKeys, 1, 10) < 20,
    'uniform quantization has the opposite failure: it cannot resolve near geometry'
  ).toBe(true);

  const halfNear = countDistinctKeys(halfKeys, 1, 10);
  const halfMiddle = countDistinctKeys(halfKeys, 1_000, 10_000);
  const halfFar = countDistinctKeys(halfKeys, 50_000, 100_000);
  expect(
    halfNear > 1_000 && halfMiddle > 1_000 && halfFar > 100,
    'half-precision bit patterns keep usable resolution across the whole range'
  ).toBe(true);
  expect(
    Math.max(halfNear, halfMiddle) / Math.min(halfNear, halfMiddle) < 2,
    'and are near-uniform in log distance, which is how a perspective camera sees depth'
  ).toBe(true);
});

it('every distribution orders back to front and reserves the culled sentinel', () => {
  for (const mode of ['ndc', 'linear', 'float16', 'float32'] as const) {
    const keyBits = getSplatDepthKeyBits(mode);
    const invalid = getSplatInvalidDepthKey(keyBits);
    const nearKey = packSplatDepthKey(mode === 'ndc' ? 0.1 : 10, {
      mode,
      depthMin: 0,
      depthMax: 1000
    });
    const farKey = packSplatDepthKey(mode === 'ndc' ? 0.9 : 900, {
      mode,
      depthMin: 0,
      depthMax: 1000
    });

    expect(farKey < nearKey, `${mode}: ascending keys render far Gaussians first`).toBe(true);
    expect(nearKey < invalid, `${mode}: no visible row reaches the culled sentinel`).toBe(true);
  }
});

it('a tiled key packs its tile above the depth bits without overflowing', () => {
  const tiled = packSplatDepthKey(500, {mode: 'linear', keyBits: 16, depthMax: 1000, tileId: 3});

  expect(tiled >>> 16, 'the tile occupies the high byte').toBe(3);
  expect(tiled & 0xffff, 'and the depth key is unchanged below it').toBe(
    packSplatDepthKey(500, {mode: 'linear', keyBits: 16, depthMax: 1000})
  );
  expect(
    () => packSplatDepthKey(1, {mode: 'float32', tileId: 1}),
    'a full-width key leaves no room for a tile'
  ).toThrow(/24 depth bits/);
});
