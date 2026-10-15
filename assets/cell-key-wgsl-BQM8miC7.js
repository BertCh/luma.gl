import{l as e}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as t,r as n,t as r}from"./wgsl-common-BNRVzgd8.js";var i=65536;function a(e,t){if(!Number.isFinite(t)||t<=0||Math.fround(t)!==t)throw Error(`${e} sumScale must be a positive finite float32 value`)}var o=`
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${n}

${t}

${r}
`,s=Math.fround(2**62-2**38);function c(t){return`
const SUM_SCALE: f32 = ${e(t)};
const MAXIMUM_SCALED_VALUE: f32 = ${e(s)};

// Not dggs_u64_add and friends: pulling in the dggs module costs compile time (see cell-keys.ts).
fn cellAddI64(left: vec2u, right: vec2u) -> vec2u {
  let low = left.y + right.y;
  return vec2u(left.x + right.x + select(0u, 1u, low < left.y), low);
}

fn cellNegateI64(value: vec2u) -> vec2u {
  let low = ~value.y + 1u;
  return vec2u(~value.x + select(0u, 1u, low == 0u), low);
}

/** roundHalfEven(value * SUM_SCALE), saturated, as a two's-complement 64-bit integer. */
fn cellScaleValue(value: f32) -> vec2u {
  let scaled = clamp(round(value * SUM_SCALE), -MAXIMUM_SCALED_VALUE, MAXIMUM_SCALED_VALUE);
  let magnitude = abs(scaled);
  let highFloat = floor(magnitude * 2.3283064365386963e-10);
  let lowFloat = magnitude - highFloat * 4294967296.0;
  let result = vec2u(u32(highFloat), u32(lowFloat));
  return select(result, cellNegateI64(result), scaled < 0.0);
}

fn cellI64ToF32(value: vec2u) -> f32 {
  let isNegative = (value.x & 0x80000000u) != 0u;
  let magnitude = select(value, cellNegateI64(value), isNegative);
  let result = f32(magnitude.x) * 4294967296.0 + f32(magnitude.y);
  return select(result, -result, isNegative);
}
`}export{a as i,o as n,c as r,i as t};