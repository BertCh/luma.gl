// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Private Philox 4x32-10 counter-based random number generator (Salmon et al. 2011, Random123).
 *
 * Not exported from the public barrel. The WGSL and CPU versions produce bit-identical words, so
 * every random decision is a pure function of `(key, counter)` and replays exactly. WGSL has no
 * 32x32->64 multiply, so the high word is assembled from 16-bit partial products; the CPU mirror
 * uses the same decomposition.
 */

/** Philox 4x32 round multipliers and Weyl key increments (Random123 constants). */
const PHILOX_M0 = 0xd2511f53;
const PHILOX_M1 = 0xcd9e8d57;
const PHILOX_W0 = 0x9e3779b9;
const PHILOX_W1 = 0xbb67ae85;

/**
 * WGSL source defining:
 * - `philox4x32(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32>`: ten-round Philox block.
 * - `philoxUnitFloat(word: u32) -> f32`: top 24 bits mapped exactly to `[0, 1)`.
 */
export const PHILOX_WGSL = /* wgsl */ `
fn philoxMulHiLo(a: u32, b: u32) -> vec2<u32> {
  let aLow = a & 0xffffu;
  let aHigh = a >> 16u;
  let bLow = b & 0xffffu;
  let bHigh = b >> 16u;
  let lowLow = aLow * bLow;
  let highLow = aHigh * bLow;
  let lowHigh = aLow * bHigh;
  let highHigh = aHigh * bHigh;
  let middle = (lowLow >> 16u) + (highLow & 0xffffu) + (lowHigh & 0xffffu);
  let high = highHigh + (highLow >> 16u) + (lowHigh >> 16u) + (middle >> 16u);
  return vec2<u32>(high, a * b);
}

fn philoxRound(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
  let product0 = philoxMulHiLo(${PHILOX_M0}u, counter.x);
  let product1 = philoxMulHiLo(${PHILOX_M1}u, counter.z);
  return vec4<u32>(
    product1.x ^ counter.y ^ key.x,
    product1.y,
    product0.x ^ counter.w ^ key.y,
    product0.y
  );
}

fn philox4x32(counterIn: vec4<u32>, keyIn: vec2<u32>) -> vec4<u32> {
  var counter = counterIn;
  var key = keyIn;
  for (var roundIndex = 0u; roundIndex < 9u; roundIndex = roundIndex + 1u) {
    counter = philoxRound(counter, key);
    key = key + vec2<u32>(${PHILOX_W0}u, ${PHILOX_W1}u);
  }
  return philoxRound(counter, key);
}

fn philoxUnitFloat(word: u32) -> f32 {
  return f32(word >> 8u) * (1.0 / 16777216.0);
}
`;

function multiplyHighLow(a: number, b: number): [number, number] {
  const aLow = a & 0xffff;
  const aHigh = a >>> 16;
  const bLow = b & 0xffff;
  const bHigh = b >>> 16;
  const lowLow = aLow * bLow;
  const highLow = aHigh * bLow;
  const lowHigh = aLow * bHigh;
  const highHigh = aHigh * bHigh;
  const middle = (lowLow >>> 16) + (highLow & 0xffff) + (lowHigh & 0xffff);
  const high = (highHigh + (highLow >>> 16) + (lowHigh >>> 16) + (middle >>> 16)) >>> 0;
  return [high, Math.imul(a, b) >>> 0];
}

/**
 * CPU mirror of WGSL `philox4x32`: returns four u32 words for a 4-word counter and 2-word key.
 */
export function getPhilox4x32(
  counter: readonly [number, number, number, number],
  key: readonly [number, number]
): [number, number, number, number] {
  let c0 = counter[0] >>> 0;
  let c1 = counter[1] >>> 0;
  let c2 = counter[2] >>> 0;
  let c3 = counter[3] >>> 0;
  let k0 = key[0] >>> 0;
  let k1 = key[1] >>> 0;
  for (let round = 0; round < 10; round++) {
    if (round > 0) {
      k0 = (k0 + PHILOX_W0) >>> 0;
      k1 = (k1 + PHILOX_W1) >>> 0;
    }
    const [high0, low0] = multiplyHighLow(PHILOX_M0, c0);
    const [high1, low1] = multiplyHighLow(PHILOX_M1, c2);
    const next0 = (high1 ^ c1 ^ k0) >>> 0;
    const next2 = (high0 ^ c3 ^ k1) >>> 0;
    c0 = next0;
    c1 = low1;
    c2 = next2;
    c3 = low0;
  }
  return [c0, c1, c2, c3];
}

/** CPU mirror of WGSL `philoxUnitFloat`: exact f32 value in `[0, 1)` from the top 24 bits. */
export function getPhiloxUnitFloat(word: number): number {
  return (word >>> 8) / 16777216;
}
