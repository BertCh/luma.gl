// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL helpers shared by the colour recipes: zero-normalised ordered keys, the d3 scale
 * transforms, and `rgba8` unpacking and blending. Requires `COLUMN_ORDERED_KEY_WGSL`.
 *
 * @internal
 */
export const COLOR_SCALE_WGSL = /* wgsl */ `
const NO_CLASS: u32 = 0xffffffffu;

// -0 and +0 compare equal, like d3 bisectRight. The zero test is on bits because a compiler may
// fold a float select on \`x == 0.0\` back to \`x\`.
fn getComparisonKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  if ((bits & 0x7fffffffu) == 0u) {
    return 0x80000000u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn signedPower(x: f32, exponent: f32) -> f32 {
  let magnitude = abs(x);
  if (magnitude == 0.0) {
    return 0.0;
  }
  let result = pow(magnitude, exponent);
  return select(result, -result, x < 0.0);
}

// Scale transform before normalisation. Codes: 0 linear, 1 sqrt, 2 pow, 3 log, 4 symlog.
fn transformValue(x: f32, scale: u32, exponent: f32, logFloor: f32) -> f32 {
  if (scale == 1u) {
    let magnitude = sqrt(abs(x));
    return select(magnitude, -magnitude, x < 0.0);
  }
  if (scale == 2u) {
    return signedPower(x, exponent);
  }
  if (scale == 3u) {
    var positive = x;
    if (!(positive > 0.0)) {
      positive = logFloor;
    }
    if (isNanBits(positive) || !(positive > 0.0)) {
      return getNaN();
    }
    return log(positive);
  }
  if (scale == 4u) {
    let magnitude = log(1.0 + abs(x));
    return select(magnitude, -magnitude, x < 0.0);
  }
  return x;
}

fn unpackChannel(color: u32, channel: u32) -> f32 {
  return f32((color >> (channel * 8u)) & 0xffu);
}

fn packChannel(value: f32) -> u32 {
  return u32(clamp(floor(value + 0.5), 0.0, 255.0));
}

fn blendColors(first: u32, second: u32, fraction: f32) -> u32 {
  var result = 0u;
  for (var channel = 0u; channel < 4u; channel++) {
    let a = unpackChannel(first, channel);
    let b = unpackChannel(second, channel);
    result = result | (packChannel(a + (b - a) * fraction) << (channel * 8u));
  }
  return result;
}
`;
