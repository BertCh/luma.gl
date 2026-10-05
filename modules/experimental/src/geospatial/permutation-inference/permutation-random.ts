// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Private counter-based random numbers for permutation inference: Philox 4x32-10 (Salmon et al.,
// "Parallel random numbers: as easy as 1, 2, 3", SC 2011), a bounded integer draw, and a keyed
// Feistel bijection on [0, n). The WGSL and TypeScript implementations produce bit-identical
// streams, so CPU oracles reproduce GPU permutations exactly. Proposed for promotion to the shared
// utils module once a second contributor needs it.

const PHILOX_M0 = 0xd2511f53;
const PHILOX_M1 = 0xcd9e8d57;
const PHILOX_W0 = 0x9e3779b9;
const PHILOX_W1 = 0xbb67ae85;
const PHILOX_ROUNDS = 10;

/** Counter word `c` that tags the Feistel round-key block, kept apart from row streams. @internal */
export const PERMUTATION_RANDOM_FEISTEL_STREAM = 0xfe157e10;

/** Feistel rounds of {@link getFeistelPermutationIndex}. @internal */
export const PERMUTATION_RANDOM_FEISTEL_ROUNDS = 4;

/** High 32 bits of the 64-bit product of two u32 values, using exact 16-bit partial products. */
function multiplyHigh(a: number, b: number): number {
  const aLow = a & 0xffff;
  const aHigh = a >>> 16;
  const bLow = b & 0xffff;
  const bHigh = b >>> 16;
  const lowLow = aLow * bLow;
  const lowHigh = aLow * bHigh;
  const highLow = aHigh * bLow;
  const highHigh = aHigh * bHigh;
  const middle = (lowLow >>> 16) + (lowHigh & 0xffff) + (highLow & 0xffff);
  return (highHigh + (lowHigh >>> 16) + (highLow >>> 16) + (middle >>> 16)) >>> 0;
}

/**
 * Philox 4x32-10 block function: encrypts a 128-bit `counter` under a 64-bit `key`.
 *
 * @param counter Four u32 counter words.
 * @param key Two u32 key words.
 * @returns Four u32 random words.
 * @internal
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
  for (let round = 0; round < PHILOX_ROUNDS; round++) {
    if (round > 0) {
      k0 = (k0 + PHILOX_W0) >>> 0;
      k1 = (k1 + PHILOX_W1) >>> 0;
    }
    const high0 = multiplyHigh(PHILOX_M0, c0);
    const low0 = Math.imul(PHILOX_M0, c0) >>> 0;
    const high1 = multiplyHigh(PHILOX_M1, c2);
    const low1 = Math.imul(PHILOX_M1, c2) >>> 0;
    c0 = (high1 ^ c1 ^ k0) >>> 0;
    c1 = low1;
    c2 = (high0 ^ c3 ^ k1) >>> 0;
    c3 = low0;
  }
  return [c0, c1, c2, c3];
}

/**
 * Sequential u32 stream for one `(key, a, b, c)` tuple: words are drawn from Philox blocks with
 * counter `(block, a, b, c)`, four words per block, block 0 first. Mirrors the WGSL
 * `PhiloxStream` exactly.
 *
 * @internal
 */
export class PhiloxStream {
  private readonly key: readonly [number, number];
  private readonly words: readonly [number, number, number];
  private block = 0;
  private lane = 4;
  private buffer: [number, number, number, number] = [0, 0, 0, 0];

  /**
   * @param key Two u32 key words, normally the seed.
   * @param a First caller counter word, for example a row index.
   * @param b Second caller counter word, for example a permutation index.
   * @param c Third caller counter word, a purpose tag.
   */
  constructor(key: readonly [number, number], a: number, b: number, c: number) {
    this.key = [key[0] >>> 0, key[1] >>> 0];
    this.words = [a >>> 0, b >>> 0, c >>> 0];
  }

  /** Returns the next u32 word of the stream. */
  nextUint32(): number {
    if (this.lane === 4) {
      this.buffer = getPhilox4x32([this.block, ...this.words], this.key);
      this.block = (this.block + 1) >>> 0;
      this.lane = 0;
    }
    return this.buffer[this.lane++];
  }

  /**
   * Returns an unbiased integer in `[0, bound)` with Lemire's multiply-shift and rejection.
   *
   * @param bound Exclusive upper bound, `1 <= bound < 2^32`.
   */
  nextBelow(bound: number): number {
    const range = bound >>> 0;
    let value = this.nextUint32();
    let low = Math.imul(value, range) >>> 0;
    if (low < range) {
      const threshold = (0x100000000 - range) % range;
      while (low < threshold) {
        value = this.nextUint32();
        low = Math.imul(value, range) >>> 0;
      }
    }
    return multiplyHigh(value, range);
  }
}

/**
 * Round keys of the keyed Feistel permutation number `permutation` under `key`.
 *
 * @internal
 */
export function getFeistelRoundKeys(
  key: readonly [number, number],
  permutation: number
): [number, number, number, number] {
  return getPhilox4x32([0, permutation >>> 0, 0, PERMUTATION_RANDOM_FEISTEL_STREAM], key);
}

/** lowbias32 integer hash (Chris Wellons), the Feistel round function. */
function hashUint32(value: number): number {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

/** Bits per Feistel half for a domain of `count` values. @internal */
export function getFeistelHalfBits(count: number): number {
  const bits = Math.max(2, Math.ceil(Math.log2(Math.max(count, 2))));
  return Math.ceil(bits / 2);
}

/**
 * Keyed pseudorandom bijection of `[0, count)`: a balanced four-round Feistel network on
 * `2 * halfBits` bits with cycle walking until the image falls below `count`. The domain is less
 * than `4 * count`, so the expected walk is short; it always terminates because the start index
 * lies on its own cycle.
 *
 * @param index Value in `[0, count)`.
 * @param count Domain size, `1 <= count < 2^31`.
 * @param roundKeys Keys from {@link getFeistelRoundKeys}.
 * @internal
 */
export function getFeistelPermutationIndex(
  index: number,
  count: number,
  roundKeys: readonly [number, number, number, number]
): number {
  if (count <= 1) {
    return 0;
  }
  const halfBits = getFeistelHalfBits(count);
  const mask = (2 ** halfBits - 1) >>> 0;
  let value = index >>> 0;
  do {
    let left = value >>> halfBits;
    let right = value & mask;
    for (let round = 0; round < PERMUTATION_RANDOM_FEISTEL_ROUNDS; round++) {
      const next = (left ^ (hashUint32(right ^ roundKeys[round]) & mask)) >>> 0;
      left = right;
      right = next;
    }
    value = (left * 2 ** halfBits + right) >>> 0;
  } while (value >= count);
  return value;
}

/**
 * WGSL twin of this module: `getPhilox4x32`, `PhiloxStream` with `createPhiloxStream`,
 * `nextPhiloxUint32` and `nextPhiloxBelow`, `getFeistelRoundKeys` and
 * `getFeistelPermutationIndex`. Bit-identical to the TypeScript functions.
 *
 * @internal
 */
export const PERMUTATION_RANDOM_WGSL = /* wgsl */ `
const PHILOX_M0: u32 = 0xd2511f53u;
const PHILOX_M1: u32 = 0xcd9e8d57u;
const PHILOX_W0: u32 = 0x9e3779b9u;
const PHILOX_W1: u32 = 0xbb67ae85u;
const PERMUTATION_RANDOM_FEISTEL_STREAM: u32 = ${PERMUTATION_RANDOM_FEISTEL_STREAM}u;

fn getPhiloxMultiplyHigh(a: u32, b: u32) -> u32 {
  let aLow = a & 0xffffu;
  let aHigh = a >> 16u;
  let bLow = b & 0xffffu;
  let bHigh = b >> 16u;
  let lowLow = aLow * bLow;
  let lowHigh = aLow * bHigh;
  let highLow = aHigh * bLow;
  let highHigh = aHigh * bHigh;
  let middle = (lowLow >> 16u) + (lowHigh & 0xffffu) + (highLow & 0xffffu);
  return highHigh + (lowHigh >> 16u) + (highLow >> 16u) + (middle >> 16u);
}

fn getPhilox4x32(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
  var c = counter;
  var k = key;
  for (var round = 0u; round < 10u; round++) {
    if (round > 0u) {
      k = k + vec2<u32>(PHILOX_W0, PHILOX_W1);
    }
    let high0 = getPhiloxMultiplyHigh(PHILOX_M0, c.x);
    let low0 = PHILOX_M0 * c.x;
    let high1 = getPhiloxMultiplyHigh(PHILOX_M1, c.z);
    let low1 = PHILOX_M1 * c.z;
    c = vec4<u32>(high1 ^ c.y ^ k.x, low1, high0 ^ c.w ^ k.y, low0);
  }
  return c;
}

struct PhiloxStream {
  key: vec2<u32>,
  words: vec3<u32>,
  block: u32,
  lane: u32,
  buffer: vec4<u32>
}

fn createPhiloxStream(key: vec2<u32>, a: u32, b: u32, c: u32) -> PhiloxStream {
  var stream: PhiloxStream;
  stream.key = key;
  stream.words = vec3<u32>(a, b, c);
  stream.block = 0u;
  stream.lane = 4u;
  stream.buffer = vec4<u32>(0u);
  return stream;
}

fn nextPhiloxUint32(stream: ptr<function, PhiloxStream>) -> u32 {
  if ((*stream).lane == 4u) {
    (*stream).buffer = getPhilox4x32(
      vec4<u32>((*stream).block, (*stream).words.x, (*stream).words.y, (*stream).words.z),
      (*stream).key
    );
    (*stream).block = (*stream).block + 1u;
    (*stream).lane = 0u;
  }
  let word = (*stream).buffer[(*stream).lane];
  (*stream).lane = (*stream).lane + 1u;
  return word;
}

fn nextPhiloxBelow(stream: ptr<function, PhiloxStream>, bound: u32) -> u32 {
  var value = nextPhiloxUint32(stream);
  var low = value * bound;
  if (low < bound) {
    let threshold = (0u - bound) % bound;
    while (low < threshold) {
      value = nextPhiloxUint32(stream);
      low = value * bound;
    }
  }
  return getPhiloxMultiplyHigh(value, bound);
}

fn getFeistelRoundKeys(key: vec2<u32>, permutation: u32) -> vec4<u32> {
  return getPhilox4x32(vec4<u32>(0u, permutation, 0u, PERMUTATION_RANDOM_FEISTEL_STREAM), key);
}

fn hashFeistelUint32(value: u32) -> u32 {
  var x = value;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

// Bits per Feistel half for a domain of count values; equals the TypeScript getFeistelHalfBits.
fn getFeistelHalfBits(count: u32) -> u32 {
  let bits = max(select(32u - countLeadingZeros(count - 1u), 1u, count <= 1u), 2u);
  return (bits + 1u) / 2u;
}

// halfBits is getFeistelHalfBits(count), passed in so callers can hoist it.
fn getFeistelPermutationIndex(index: u32, count: u32, halfBits: u32, roundKeys: vec4<u32>) -> u32 {
  if (count <= 1u) {
    return 0u;
  }
  let mask = (1u << halfBits) - 1u;
  var value = index;
  loop {
    var left = value >> halfBits;
    var right = value & mask;
    for (var round = 0u; round < 4u; round++) {
      let next = left ^ (hashFeistelUint32(right ^ roundKeys[round]) & mask);
      left = right;
      right = next;
    }
    value = (left << halfBits) | right;
    if (value < count) {
      break;
    }
  }
  return value;
}
`;
