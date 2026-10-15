import{t as e}from"./philox-random-CHwyk0dd.js";var t=`
const PHILOX_M0: u32 = 0xd2511f53u;
const PHILOX_M1: u32 = 0xcd9e8d57u;
const PHILOX_W0: u32 = 0x9e3779b9u;
const PHILOX_W1: u32 = 0xbb67ae85u;
const PERMUTATION_RANDOM_FEISTEL_STREAM: u32 = ${e.permutationFeistel.id}u;

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
`;export{t};